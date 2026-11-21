"""Single-model FastAPI server for Qwen3-TTS."""

import argparse
import io
import ipaddress
import logging
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Annotated, Callable, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPBearer
from pydantic import BaseModel, ConfigDict, StringConstraints
from starlette.datastructures import Headers
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from .inference.qwen3_tts_model import Qwen3TTSModel

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
MAX_NEW_TOKENS = 512
SPEECH_PATH = "/v1/tts/speech"
CLONE_PATH = "/v1/tts/voice-clone"
VOICES_PATH = "/v1/voices"
BODY_LIMITS = {SPEECH_PATH: 16 * 1024, CLONE_PATH: 8 * 1024 * 1024}
SYNTHESIS_RESPONSES = {
    200: {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}},
    409: {"description": "The active model does not support this endpoint"},
    413: {"description": "Request body exceeds the size limit"},
    503: {"description": "The model is busy or has run out of GPU memory"},
}
LOGGER = logging.getLogger(__name__)

SpeechText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
ReferenceText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    device: str = "auto"
    dtype: str = "auto"
    attn_implementation: str = "sdpa"
    cpu_threads: Optional[int] = None
    api_key: Optional[str] = field(default=None, repr=False)


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: SpeechText
    language: Name = "auto"
    speaker: Optional[Name] = None
    instruct: Optional[SpeechText] = None


class VoiceInfo(BaseModel):
    model: str
    mode: str
    languages: list[str]
    speakers: list[str]
    default_speaker: Optional[str]
    supports_instruct: bool


class WavResponse(Response):
    media_type = "audio/wav"


class SynthesisRequestMiddleware:
    """Check authentication and size before FastAPI parses a request body."""

    def __init__(self, app: ASGIApp, bearer: HTTPBearer, api_key: Optional[str]):
        self.app = app
        self.bearer = bearer
        self.api_key = api_key.encode() if api_key else None

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        root = scope.get("root_path", "").rstrip("/")
        if root and path.startswith(root + "/"):
            path = path[len(root):]
        path = path.rstrip("/")
        limit = BODY_LIMITS.get(path) if scope["method"] == "POST" else None
        request = Request(scope, receive)
        if self.api_key and (limit is not None or path == VOICES_PATH):
            credentials = await self.bearer(request)
            if not credentials or not secrets.compare_digest(credentials.credentials.encode(), self.api_key):
                unauthorized = JSONResponse(
                    {"detail": "A valid Bearer token is required."},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await unauthorized(scope, receive, send)
                return

        if limit is None:
            await self.app(scope, receive, send)
            return
        too_large = JSONResponse({"detail": "Request body exceeds the size limit."}, status_code=413)
        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None:
            try:
                size = int(content_length)
                if size < 0:
                    raise ValueError
            except ValueError:
                invalid = JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)
                await invalid(scope, receive, send)
                return
            if size > limit:
                await too_large(scope, receive, send)
                return

        body = bytearray()
        try:
            async for chunk in request.stream():
                if len(body) + len(chunk) > limit:
                    await too_large(scope, receive, send)
                    return
                body.extend(chunk)
        except ClientDisconnect:
            return

        payload: Optional[bytes] = bytes(body)
        del body

        async def replay():
            nonlocal payload
            if payload is None:
                return await receive()
            message = {"type": "http.request", "body": payload, "more_body": False}
            payload = None
            return message

        await self.app(scope, replay, send)


def load_model(settings: Settings) -> Qwen3TTSModel:
    device_name = settings.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use auto, cpu, cuda or cuda:N for the device.")
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is unavailable.")
        # The BF16 capability query uses the current CUDA device.
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    elif settings.cpu_threads is not None:
        torch.set_num_threads(settings.cpu_threads)

    dtype = settings.dtype
    if dtype == "auto":
        native_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
        dtype = "bfloat16" if native_bf16 else "float32"
    if settings.attn_implementation == "flash_attention_2" and (
        device.type != "cuda" or dtype == "float32"
    ):
        raise ValueError("FlashAttention 2 requires CUDA with bfloat16 or float16.")

    return Qwen3TTSModel.from_pretrained(
        settings.model,
        device_map=str(device),
        dtype=getattr(torch, dtype),
        attn_implementation=settings.attn_implementation,
    )


def decode_reference(data: bytes) -> tuple[np.ndarray, int]:
    try:
        with sf.SoundFile(io.BytesIO(data)) as audio:
            rate = audio.samplerate
            if audio.format not in {"WAV", "WAVEX", "FLAC", "OGG"}:
                raise HTTPException(422, "Upload a WAV, FLAC or OGG file.")
            if not 8000 <= rate <= 96000 or audio.channels not in (1, 2):
                raise HTTPException(422, "Use mono or stereo audio at 8–96 kHz.")
            if not rate <= audio.frames <= 30 * rate:
                raise HTTPException(422, "Reference audio must be 1–30 seconds long.")
            samples = audio.read(frames=audio.frames, dtype="float32", always_2d=True)
            if len(samples) != audio.frames:
                raise HTTPException(422, "Reference audio is incomplete.")
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(422, "Could not decode the reference audio.") from exc

    if not np.isfinite(samples).all():
        raise HTTPException(422, "Reference audio contains invalid samples.")
    mono = np.clip(samples, -1.0, 1.0).mean(axis=1)
    if not np.any(mono):
        raise HTTPException(422, "Reference audio is silent.")
    return mono, rate


class SpeechService:
    def __init__(self, model_name: str, model: Qwen3TTSModel):
        self.model = model
        self.slot = Lock()
        mode = model.model.tts_model_type
        if mode not in {"custom_voice", "voice_design", "base"}:
            raise ValueError(f"Unsupported Qwen3-TTS model type: {mode!r}")
        speakers = sorted(model.get_supported_speakers() or [])
        self.voices = VoiceInfo(
            model=model_name,
            mode=mode,
            languages=sorted(model.get_supported_languages() or []),
            speakers=speakers,
            default_speaker=next(
                (speaker for speaker in speakers if speaker.lower() == "sohee"),
                speakers[0] if speakers else None,
            ),
            supports_instruct=mode == "voice_design" or (
                mode == "custom_voice" and model.model.tts_model_size == "1b7"
            ),
        )

    def require_mode(self, *modes: str):
        if self.voices.mode not in modes:
            raise HTTPException(409, f"This endpoint requires a model of type {' or '.join(modes)}.")

    def validate_language(self, language: str):
        if self.voices.languages and not any(
            language.lower() == supported.lower() for supported in self.voices.languages
        ):
            raise HTTPException(422, "Unsupported language. See /v1/voices.")

    def render(self, synthesize: Callable[[], tuple[list[np.ndarray], int]]) -> WavResponse:
        # The worker owns this slot even if the awaiting HTTP request is cancelled.
        if not self.slot.acquire(blocking=False):
            raise HTTPException(503, "The model is busy. Retry shortly.", headers={"Retry-After": "1"})
        try:
            try:
                with torch.inference_mode():
                    waveforms, sample_rate = synthesize()
            except (ValueError, NotImplementedError) as exc:
                raise HTTPException(422, "The model rejected the synthesis input.") from exc
            except torch.cuda.OutOfMemoryError as exc:
                LOGGER.error("CUDA ran out of memory during speech synthesis")
                raise HTTPException(503, "The model ran out of GPU memory.") from exc

            if len(waveforms) != 1:
                raise HTTPException(500, "The model returned an unexpected number of waveforms.")
            waveform = np.asarray(waveforms[0])
            if not waveform.size or not np.isfinite(waveform).all():
                raise HTTPException(500, "The model generated invalid audio.")
            audio = io.BytesIO()
            sf.write(audio, np.clip(waveform, -1.0, 1.0), sample_rate, format="WAV", subtype="PCM_16")
            return WavResponse(
                audio.getvalue(),
                headers={"Content-Disposition": 'attachment; filename="speech.wav"', "Cache-Control": "no-store"},
            )
        finally:
            self.slot.release()

    def speech(self, data: SpeechRequest) -> WavResponse:
        self.require_mode("custom_voice", "voice_design")
        self.validate_language(data.language)
        kwargs = {"text": data.text, "language": data.language, "max_new_tokens": MAX_NEW_TOKENS}
        if self.voices.mode == "voice_design":
            if data.instruct is None or data.speaker is not None:
                raise HTTPException(422, "VoiceDesign requires instruct and accepts no speaker.")
            return self.render(lambda: self.model.generate_voice_design(instruct=data.instruct, **kwargs))

        speaker = data.speaker or self.voices.default_speaker
        if not speaker or not any(speaker.lower() == name.lower() for name in self.voices.speakers):
            raise HTTPException(422, "Select a supported speaker from /v1/voices.")
        if data.instruct is not None and not self.voices.supports_instruct:
            raise HTTPException(422, "This model does not support instruct.")
        return self.render(
            lambda: self.model.generate_custom_voice(speaker=speaker, instruct=data.instruct, **kwargs)
        )

    def clone(
        self, text: str, language: str, audio: bytes, ref_text: Optional[str], x_vector_only_mode: bool
    ) -> WavResponse:
        def synthesize():
            return self.model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=decode_reference(audio),
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
                non_streaming_mode=True,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        return self.render(synthesize)


def create_app(settings: Optional[Settings] = None, *, model_loader=load_model) -> FastAPI:
    settings = settings or Settings(api_key=os.environ.get("QWEN_TTS_API_KEY"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model = await run_in_threadpool(model_loader, settings)
        app.state.speech = SpeechService(settings.model, model)
        try:
            yield
        finally:
            del app.state.speech

    app = FastAPI(title="Qwen3-TTS", lifespan=lifespan)
    bearer = HTTPBearer(auto_error=False)
    app.add_middleware(SynthesisRequestMiddleware, bearer=bearer, api_key=settings.api_key)
    # Advertise the middleware's authentication requirement in Swagger UI.
    protected = [Depends(bearer)] if settings.api_key else []

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get(VOICES_PATH, response_model=VoiceInfo, dependencies=protected)
    async def voices(request: Request):
        return request.app.state.speech.voices

    @app.post(
        SPEECH_PATH, response_class=WavResponse, responses=SYNTHESIS_RESPONSES, dependencies=protected
    )
    async def speech(data: SpeechRequest, request: Request):
        service: SpeechService = request.app.state.speech
        return await run_in_threadpool(service.speech, data)

    @app.post(
        CLONE_PATH, response_class=WavResponse, responses=SYNTHESIS_RESPONSES, dependencies=protected
    )
    async def clone(
        request: Request,
        text: Annotated[SpeechText, Form()],
        ref_audio: Annotated[UploadFile, File()],
        ref_text: Annotated[Optional[ReferenceText], Form()] = None,
        language: Annotated[Name, Form()] = "auto",
        x_vector_only_mode: Annotated[bool, Form()] = False,
    ):
        service: SpeechService = request.app.state.speech
        service.require_mode("base")
        service.validate_language(language)
        if x_vector_only_mode == (ref_text is not None):
            raise HTTPException(422, "Provide ref_text or set x_vector_only_mode=true, but not both.")
        audio = await ref_audio.read()
        return await run_in_threadpool(service.clone, text, language, audio, ref_text, x_vector_only_mode)

    return app


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local model directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--cpu-threads", type=int, help="PyTorch intra-op thread count for CPU inference")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    api_key = os.environ.get("QWEN_TTS_API_KEY")
    if not api_key and not is_loopback(args.host):
        parser.error("Set QWEN_TTS_API_KEY before binding to a non-loopback address.")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive.")
    settings = Settings(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        cpu_threads=args.cpu_threads,
        api_key=api_key,
    )
    uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1, limit_concurrency=32)


if __name__ == "__main__":
    main()
