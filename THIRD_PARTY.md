# 출처와 외부 구성요소

## 영상·썸네일·공개 주석

- MEVA (Multiview Extended Video with Activities), Kitware Inc. / IARPA.
- 프로젝트: https://mevadata.org/
- 다운로드 안내: https://mevadata.org/resources/README-meva-kf1-data.html
- 라이선스: CC BY 4.0. 원문은 `licenses/MEVA-CC-BY-4.0.txt`.
- 공식 라이선스 파일: https://mevadata.org/resources/MEVA-data-license.txt
- 논문: Kellie Corona et al., *MEVA: A Large-Scale Multiview, Multimodal Video Dataset for Activity Detection*, WACV 2021.
- 변경: 원본 AVI에서 12초 또는 20초 발췌, 가로 960px, 15fps H.264 재인코딩, 오디오 제거, JPEG 썸네일 추출. 정답 오버레이 없음.
- 파일별 원본 URL·원본 SHA256·시각·공개 부분 주석은 `samples/manifest.json`.
- 제공 주석은 일부 행동 예시이며 완전한 정답집이 아닙니다. 원 제작자의 제품 보증이나 제휴를 의미하지 않습니다.

## 모델

모델 가중치를 이 배포물에 포함하지 않습니다. 최초 실행 시 별도 다운로드합니다.

- Qwen/Qwen3-VL-2B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct (Apache-2.0 표기). 기본 실측에 사용.
- HuggingFaceTB/SmolVLM-256M-Instruct: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct (Apache-2.0 표기). 선택 가능한 영문 캡션 기초 모델.
- Ollama와 OpenAI는 선택 연결 방식이며 각 소프트웨어·모델·서비스 조건을 따릅니다. OpenAI API 호출은 이번 검증에서 실행하지 않았습니다.

Python 의존성은 requirements 파일에서 설치하며 원 프로젝트의 라이선스를 따릅니다. 이 앱의 MIT 라이선스가 데이터·외부 모델·의존성의 라이선스를 대체하지 않습니다.
