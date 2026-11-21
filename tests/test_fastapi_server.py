import asyncio
import io
import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
import soundfile as sf
import torch
import uvicorn
from fastapi.testclient import TestClient

from qwen_tts import server


def make_model(mode="custom_voice", size="0b6", speakers=None):
    model = Mock()
    model.model = SimpleNamespace(tts_model_type=mode, tts_model_size=size)
    model.get_supported_languages.return_value = ["auto", "korean", "english"]
    model.get_supported_speakers.return_value = speakers if speakers is not None else (
        ["ryan", "sohee"] if mode == "custom_voice" else None
    )
    output = ([np.array([-2.0, 0.0, 2.0], dtype="float32")], 24000)
    model.generate_custom_voice.return_value = output
    model.generate_voice_design.return_value = output
    model.generate_voice_clone.return_value = output
    return model


def make_app(backend, **settings):
    return server.create_app(server.Settings(**settings), model_loader=lambda _: backend)


def wav_bytes(frames=8000, stereo=False, value=0.25):
    samples = np.full(frames, value, dtype="float32")
    if stereo:
        samples = np.column_stack((samples, samples * 2))
    audio = io.BytesIO()
    sf.write(audio, samples, 8000, format="WAV", subtype="FLOAT")
    return audio.getvalue()


def test_custom_voice_metadata_and_wav():
    model = make_model()
    loader = Mock(return_value=model)
    app = server.create_app(model_loader=loader, settings=server.Settings())
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        responses = client.get("/openapi.json").json()["paths"][server.SPEECH_PATH]["post"]["responses"]
        assert responses["200"]["content"]["audio/wav"]["schema"]["format"] == "binary"
        voices = client.get("/v1/voices").json()
        assert voices["speakers"] == ["ryan", "sohee"]
        assert voices["default_speaker"] == "sohee"
        assert voices["supports_instruct"] is False
        response = client.post(server.SPEECH_PATH, json={"text": " 안녕하세요. ", "language": "Korean"})

    loader.assert_called_once_with(server.Settings())
    model.generate_custom_voice.assert_called_once_with(
        text="안녕하세요.", language="Korean", speaker="sohee", instruct=None,
        max_new_tokens=server.MAX_NEW_TOKENS,
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["cache-control"] == "no-store"
    with sf.SoundFile(io.BytesIO(response.content)) as audio:
        assert (audio.format, audio.subtype, audio.samplerate) == ("WAV", "PCM_16", 24000)
        assert audio.read(dtype="int16").tolist() == [-32768, 0, 32767]


@pytest.mark.parametrize("change", [
    {"text": "  "},
    {"text": "a" * 501},
    {"language": "unknown"},
    {"speaker": "unknown"},
    {"instruct": "Speak slowly"},
    {"max_new_tokens": 10000},
])
def test_invalid_speech_fields(change):
    model = make_model()
    with TestClient(make_app(model)) as client:
        response = client.post(server.SPEECH_PATH, json={"text": "Hello", **change})
    assert response.status_code == 422
    model.generate_custom_voice.assert_not_called()


def test_custom_voice_style_and_local_speaker():
    model = make_model(size="1b7", speakers=["local_speaker"])
    with TestClient(make_app(model, model="./local-model")) as client:
        assert client.get("/v1/voices").json()["supports_instruct"] is True
        response = client.post(
            server.SPEECH_PATH, json={"text": "Hello", "speaker": "Local_Speaker", "instruct": "Speak slowly"}
        )
    assert response.status_code == 200
    assert model.generate_custom_voice.call_args.kwargs["instruct"] == "Speak slowly"


def test_voice_design_requires_instructions():
    model = make_model(mode="voice_design", size="1b7")
    with TestClient(make_app(model)) as client:
        assert client.post(server.SPEECH_PATH, json={"text": "Hello"}).status_code == 422
        response = client.post(server.SPEECH_PATH, json={"text": "Hello", "instruct": "A low voice"})
    assert response.status_code == 200
    model.generate_voice_design.assert_called_once_with(
        text="Hello", language="auto", instruct="A low voice", max_new_tokens=server.MAX_NEW_TOKENS,
    )


@pytest.mark.parametrize("embedding_only", [False, True])
def test_clone_downmixes_stereo_and_uses_matching_mode(embedding_only):
    model = make_model(mode="base")
    fields = {"x_vector_only_mode": "true"} if embedding_only else {"ref_text": " Reference transcript "}
    with TestClient(make_app(model)) as client:
        assert client.post(server.SPEECH_PATH, json={"text": "Hello"}).status_code == 409
        response = client.post(
            server.CLONE_PATH, data={"text": "Hello", **fields},
            files={"ref_audio": ("reference.wav", wav_bytes(stereo=True))},
        )

    assert response.status_code == 200, response.text
    kwargs = model.generate_voice_clone.call_args.kwargs
    samples, rate = kwargs["ref_audio"]
    assert rate == 8000
    assert samples.shape == (8000,)
    assert samples.dtype == np.float32
    np.testing.assert_allclose(samples, 0.375)
    assert kwargs["x_vector_only_mode"] is embedding_only
    assert kwargs["ref_text"] == (None if embedding_only else "Reference transcript")


@pytest.mark.parametrize("data, fields", [
    (b"invalid", {"ref_text": "Transcript"}),
    (wav_bytes(7999), {"ref_text": "Transcript"}),
    (wav_bytes(240001), {"ref_text": "Transcript"}),
    (wav_bytes(value=0), {"ref_text": "Transcript"}),
    (wav_bytes(value=float("nan")), {"ref_text": "Transcript"}),
    (wav_bytes(), {"ref_text": "   "}),
    (wav_bytes(), {"ref_text": "a" * 1001}),
    (wav_bytes(), {}),
    (wav_bytes(), {"ref_text": "Transcript", "x_vector_only_mode": "true"}),
])
def test_clone_rejects_invalid_inputs(data, fields):
    model = make_model(mode="base")
    with TestClient(make_app(model)) as client:
        response = client.post(
            server.CLONE_PATH, data={"text": "Hello", **fields}, files={"ref_audio": ("reference.wav", data)},
        )
    assert response.status_code == 422
    model.generate_voice_clone.assert_not_called()


def test_clone_requires_base_model():
    with TestClient(make_app(make_model())) as client:
        response = client.post(
            server.CLONE_PATH, data={"text": "Hello", "ref_text": "Transcript"},
            files={"ref_audio": ("reference.wav", wav_bytes())},
        )
    assert response.status_code == 409


def test_body_limit_rejects_content_length_and_streamed_bytes(monkeypatch):
    monkeypatch.setitem(server.BODY_LIMITS, server.CLONE_PATH, 100)
    with TestClient(make_app(make_model(mode="base"))) as client:
        assert client.post(server.CLONE_PATH, content=b"a" * 101).status_code == 413
        response = client.post(server.CLONE_PATH, content=iter([b"a" * 60, b"b" * 50]))
    assert response.request.headers.get("content-length") is None
    assert response.status_code == 413


def test_optional_bearer_auth():
    with TestClient(make_app(make_model(), api_key="test-key")) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/docs").status_code == 200
        response = client.get("/v1/voices")
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert client.get("/v1/voices", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/v1/voices", headers={"Authorization": "Bearer test-key"}).status_code == 200
        assert client.post(server.SPEECH_PATH, json={"text": "Hello"}).status_code == 401
        assert client.post(
            server.SPEECH_PATH, json={"text": "Hello"}, headers={"Authorization": "Bearer test-key"}
        ).status_code == 200
        responses = client.get("/openapi.json").json()["paths"]
        assert responses[server.CLONE_PATH]["post"]["security"] == [{"HTTPBearer": []}]


def test_unauthenticated_upload_is_rejected_before_reading_the_body():
    async def upload():
        pytest.fail("The request body must not be read before authentication.")
        yield b""

    app = make_app(make_model(mode="base"), api_key="test-key")

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(server.CLONE_PATH, content=upload())
        assert response.status_code == 401

    asyncio.run(scenario())


def test_cancelled_request_keeps_model_busy_until_worker_finishes():
    entered, release = Event(), Event()
    model = make_model()
    output = model.generate_custom_voice.return_value

    def synthesize(**kwargs):
        entered.set()
        assert release.wait(5)
        return output

    model.generate_custom_voice.side_effect = synthesize
    app = make_app(model)

    async def scenario():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                pending = asyncio.create_task(client.post(server.SPEECH_PATH, json={"text": "Hello"}))
                try:
                    assert await asyncio.to_thread(entered.wait, 5)
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                    assert (await client.get("/healthz")).status_code == 200
                    busy = await client.post(server.SPEECH_PATH, json={"text": "Hello"})
                    assert busy.status_code == 503
                    assert busy.headers["retry-after"] == "1"
                finally:
                    release.set()

                for _ in range(100):
                    response = await client.post(server.SPEECH_PATH, json={"text": "Hello"})
                    if response.status_code == 200:
                        break
                    await asyncio.sleep(0.01)
                assert response.status_code == 200
                assert model.generate_custom_voice.call_count == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("error, status", [(NotImplementedError(), 422), (torch.cuda.OutOfMemoryError(), 503)])
def test_worker_error_releases_slot(error, status):
    model = make_model()
    model.generate_custom_voice.side_effect = [error, model.generate_custom_voice.return_value]
    with TestClient(make_app(model)) as client:
        assert client.post(server.SPEECH_PATH, json={"text": "Hello"}).status_code == status
        assert client.post(server.SPEECH_PATH, json={"text": "Hello"}).status_code == 200


@pytest.mark.parametrize(
    "cuda_available, bf16_supported, device, dtype",
    [(False, False, "cpu", torch.float32), (True, False, "cuda:0", torch.float32),
     (True, True, "cuda:0", torch.bfloat16)],
)
def test_auto_device_and_dtype(monkeypatch, cuda_available, bf16_supported, device, dtype):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: int(cuda_available))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **_: bf16_supported)
    monkeypatch.setattr(torch.cuda, "set_device", Mock())
    load = Mock()
    monkeypatch.setattr(server.Qwen3TTSModel, "from_pretrained", load)
    server.load_model(server.Settings())
    load.assert_called_once_with(
        server.DEFAULT_MODEL, device_map=device, dtype=dtype, attn_implementation="sdpa",
    )


def test_startup_failure_is_reported():
    loader = Mock(side_effect=RuntimeError("model unavailable"))
    with pytest.raises(RuntimeError, match="model unavailable"):
        with TestClient(server.create_app(model_loader=loader)):
            pytest.fail("The app should fail startup")


def test_non_loopback_requires_token_and_cli_uses_single_worker(monkeypatch):
    run = Mock()
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setattr(sys, "argv", ["qwen-tts-server", "--host", "0.0.0.0"])
    monkeypatch.delenv("QWEN_TTS_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        server.main()
    assert exc.value.code == 2
    run.assert_not_called()

    monkeypatch.setenv("QWEN_TTS_API_KEY", "test-key")
    server.main()
    assert run.call_args.kwargs["workers"] == 1
    assert run.call_args.kwargs["host"] == "0.0.0.0"
