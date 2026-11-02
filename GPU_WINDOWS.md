# RTX 3070 Windows 비교 실행

대상: RTX 3070 전용 VRAM 8GB, 시스템 RAM 32GB. GPU 경로를 구현했지만 개발 환경에 사용자 GPU가 연결되지 않아 실제 CUDA 추론과 Windows 배치 실행은 미검증입니다. 기존 웹 앱 기본 분석은 CPU로 유지되며 이 안내는 별도 비교 실행기용입니다.

1. Python 3.12 64비트와 NVIDIA 드라이버를 준비합니다.
2. codex/sampling-comparison-draft 브랜치 ZIP을 받아 기존 설치와 다른 폴더에 압축을 풉니다. 기존 results/data를 덮어쓰지 않습니다.
3. setup_gpu_windows.cmd를 더블클릭합니다. 별도 .venv-gpu에 CUDA 12.6용 PyTorch 2.8.0, torchvision 0.23.0 및 의존성을 설치하고 CUDA 인식과 작은 FP16 행렬 연산을 검사합니다.
4. run_gpu_sample.cmd를 더블클릭합니다. 최초 모델 다운로드가 필요합니다. meva-01-04 한 영상으로 동일한 FP16 모델의 균등 3장과 6장을 순차 비교합니다.
5. 성공 시 review.html이 열립니다. 실패 시 콘솔 오류와 해당 experiments 폴더의 run.json을 전달하세요. 메모리 부족은 실패로 기록하고 CPU로 몰래 전환하거나 입력을 줄이지 않습니다.

입력은 최대 576×324로 축소합니다. 모델 프로세서의 추가 크기 조정이 있을 수 있습니다. batch size는 1이며 GPU peak allocated/reserved 메모리, 동기화한 실행 시간, 모델 원문과 PNG 입력을 저장합니다. 메모리 수치는 PyTorch 할당량으로, Windows 전체 GPU 사용량과 다릅니다. Windows의 CPU RSS 값은 null일 수 있습니다.

샘플 성공 후 전체 64개 비교(명령 프롬프트, 저장소 루트):
```bat
.venv-gpu\Scripts\python.exe scripts\compare_vlm.py --run-id gpu-full-001 --device cuda --dtype float16 --batch-size 1
.venv-gpu\Scripts\python.exe scripts\report_vlm.py --run-id gpu-full-001
```

같은 run ID는 덮어쓰지 않습니다. 재실행 시 새 ID를 쓰세요. 결과 폴더를 ZIP으로 전달하면 출력과 근거를 대조할 수 있습니다. 기존 CPU bfloat16 출력과 직접 비교하지 않고 이번 실행의 FP16 3장·6장을 비교합니다. JSON 형식 성공은 인식 정확도가 아닙니다.

설치 버전 출처: https://pytorch.org/get-started/previous-versions/ (PyTorch 2.8.0 CUDA 12.6 Windows).
검증: Python 구문 검사와 모의 CUDA 부재·잘못된 모드 조합 검사 2개 통과. 실제 CUDA/Windows 실행은 미검증.
