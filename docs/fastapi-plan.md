# Qwen3-TTS FastAPI 서버 구현 계획

## 목표

기존 Qwen3-TTS SDK와 Gradio 데모에 FastAPI 서버를 옵션으로 추가한다. 기본 모델은 0.6B CustomVoice, 기본 화자는 한국어를 지원하는 Sohee다. Swagger UI에서 요청을 실행하고 WAV를 내려받을 수 있다.

## 리뷰 반영 후 구현 범위

이 계획은 Claude Opus 5.5의 `high` 설정으로 SDK와 대조해 검토했다. 검토 세션 ID는 `cmwpsxf8t5r4r272n44b2sxzw`다.

| 영역 | 구현 결정 |
| --- | --- |
| 모델 로딩 | FastAPI lifespan에서 프로세스당 한 모델을 로드한다. Uvicorn 워커는 하나로 고정한다. 기본 장치는 CUDA가 있으면 `cuda:0`, 그 외에는 CPU다. 자동 정밀도는 BF16을 지원하는 CUDA에서 BF16, 그 외에는 FP32다. FP16과 FlashAttention 2는 명시 옵션으로 제공한다. CPU 스레드 수도 설정할 수 있다. |
| 상태와 기능 | `GET /healthz`는 로드 이후 응답한다. 초기 로딩에 실패하면 서버 시작이 실패한다. `GET /v1/voices`는 실제 로드한 모델의 모드, 언어, 내장 화자와 스타일 지시 지원 여부를 반환한다. 로컬 모델은 Qwen3-TTS SDK가 읽을 수 있는 완전한 디렉터리를 지원한다. |
| 합성 | Qwen 전용 JSON API인 `POST /v1/tts/speech`를 제공한다. CustomVoice는 내장 화자를 사용하며 0.6B에 스타일 지시가 오면 422를 반환한다. VoiceDesign에서는 스타일 지시가 필수다. 기본 언어는 `auto`다. SDK의 샘플레이트로 PCM 16비트 WAV를 생성한다. |
| 음성 복제 | Base 모델에서 `POST /v1/tts/voice-clone`을 제공한다. multipart로 WAV/FLAC/OGG 파일과 전사문을 받는다. 전사문 없이 화자 임베딩을 쓰려면 모드를 명시한다. 길이와 샘플레이트를 디코딩 전에 검사한다. SDK에는 유한한 모노 float32 배열과 샘플레이트를 전달한다. 업로드는 요청 중 임시 스풀에 기록될 수 있으며 요청 종료 시 정리한다. |
| 자원 한도 | 청크를 포함한 본문 크기를 파싱 전에 제한하고 초과 시 413을 반환한다. 텍스트 길이와 생성 토큰 상한을 둔다. 추론 작업 스레드가 잠금을 소유하며, 클라이언트가 연결을 끊어도 생성은 EOS 또는 토큰 상한까지 계속된다. 슬롯이 차면 `Retry-After`와 함께 503을 반환한다. |
| 실행과 인증 | `pip install -e '.[server]'` 후 `qwen-tts-server`로 실행한다. loopback이 기본값이다. CLI에서 외부 바인딩을 선택하면 `QWEN_TTS_API_KEY`가 필요하다. 키가 설정되면 `/v1` 경로에 Bearer 인증을 적용한다. 상태 확인과 API 문서는 열어 둔다. |

## 검증 기준

- 가짜 모델로 세 모델 모드의 SDK 호출 인자, WAV 헤더, 스타일과 화자 검증, 오디오 형식과 길이, 청크 업로드 한도, 인증, 클라이언트 취소 후 슬롯 유지와 복구를 검증한다.
- CLI 설치와 시작을 확인한 뒤 공식 0.6B CustomVoice 모델로 한국어 WAV를 HTTP로 생성한다. 현재 작업환경에서는 CPU로 검증한다.

## 구현 검증 결과

- 서버 테스트 32개를 통과했다. 인증 키가 설정된 경우 업로드 본문을 읽기 전에 인증을 검사한다.
- 공식 CustomVoice 0.6B, Base 0.6B, VoiceDesign 1.7B 가중치를 각각 로드해 HTTP 요청으로 24 kHz PCM 16비트 WAV를 생성했다. 음성 복제에는 CustomVoice로 생성한 WAV를 참조로 사용했다.
- CPU 스레드 8개로 실행한 CustomVoice 한국어 요청은 15.45초 만에 5.12초 분량의 WAV를 반환했다. CUDA는 이 환경에서 검증하지 못했다.
