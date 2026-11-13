# 학습된 모델 웹 실험실

주소: **https://hwanghj09.github.io/codingfly/**

자연어 요청을 입력하고 **코드 생성**을 누르면 실제 Fly-NL v3.2와 Fly-Code v2의 학습된 체크포인트로 IR과 Python을 생성한다. **Python 실행**은 별도 브라우저 실행 환경에서 코드를 실행하고 표준 출력·변수·함수 반환값을 표시한다. 함수 예시는 JSON 배열 인자를 바꿔 테스트할 수 있다. 기대 표준 출력 비교는 선택 사항이며 이 값은 생성 모델에 전달되지 않는다.

## 배포 구조

- GitHub Pages의 정적 사이트다. 체크포인트, 원본 그래프, 실제 생성 모듈은 같은 사이트에서 내려받는다. 기존 main/root Pages 배포본은 고정된 jsDelivr의 Pyodide 0.27.7/NumPy 런타임을 내려받고, Actions artifact는 런타임도 함께 포함한다. 외부 요청은 공개 런타임 파일 다운로드이며 사용자 입력은 브라우저 안에 머문다. 외부 LLM·추론 서버 API는 사용하지 않는다.
- 추론 worker는 저장된 시냅스를 불러오며 웹에서 재학습하지 않는다. 모델 zip의 SHA-256, 체크포인트·그래프·IR 계약과 평가 보고서의 가중치 해시를 확인한다.
- `web/checkpoints/`는 평가를 마친 실제 모델 릴리스다. 학습용 문장·평가 정답·teacher·legacy 생성기는 브라우저 모델 묶음에 포함하지 않는다. IR 사양은 검증용 해시 확인에만 사용한다.
- 실행 worker는 추론 worker와 분리하고 매번 새로 만든다. 구문 허용 목록, 제한된 builtin, 반복 범위·출력·단계 한도와 종료 타이머를 적용한다. 실행 중단 후에도 모델을 계속 시험할 수 있다.
- 브라우저 실행은 네이티브의 Linux OS namespace/자원 제한 채점과 다르며, 고정 메모리 한도를 보장하지 않는다. 사이트는 브라우저 관측 결과를 표시하고, 기대 출력을 명시한 경우에만 그 출력의 일치를 표시한다.

## GitHub Actions

`.github/workflows/pages.yml`은 PR에서 빌드·모델 무결성·네이티브 회귀·실제 Chromium 검증을 수행한다. 기존 Python 평가 108개(지원 96개, 미지원 12개)의 IR·Python을 브라우저 결과와 비교하고, 한국어 변수·음수·함수 실행, 기대 출력 비교, 실행 한도, 중단/재실행과 모바일 화면을 확인한다. 증거 JSON과 스크린샷을 artifact로 남긴다.

main push 또는 workflow_dispatch에서는 기존 Pages 소스를 확인한다. 이 저장소의 main/root 배포 설정은 그대로 유지하고, 검증된 `docs/site/` 정적 배포본을 Pages 빌드 API로 게시한 뒤 해당 main 커밋의 게시 완료까지 기다린다. 설정 전환에 필요한 관리 권한이나 새 비밀 키는 요청하지 않는다. 향후 소스가 workflow로 바뀌면 같은 작업이 OIDC와 `actions/deploy-pages`로 artifact를 배포한다. 재학습하지 않는다.

루트 index.html은 `docs/site/`의 실제 실험실로 연결한다. `python web/build.py --export-source`로 이 배포본을 갱신한다. CI의 `--check-source`는 모델·소스와 배포본이 다르면 실패하며, 실제 `/codingfly/` 경로와 CDN 런타임 경로도 브라우저에서 검사한다.

## 로컬 미리 보기

```bash
python -m pip install -r requirements.txt -r web/requirements-test.txt
python web/build.py --output _site
python -m http.server 8766 --directory _site
```

http://localhost:8766/ 에서 확인한다. 첫 실행에는 약 17MB의 Python/NumPy를 내려받는다. `file://`로 직접 열지 않는다.

```bash
python -m playwright install chromium
python web/test_browser.py --site _site --report artifacts/browser-results.json
```

테스트 서버는 프로젝트 Pages와 같은 `/codingfly/` 하위 경로에서 검증한다. `--url https://hwanghj09.github.io/codingfly/ --quick`은 실제 배포의 대표 사례와 실행 기능을 확인한다.

새 학습 모델을 배포하려면 평가를 마치고 보고서를 갱신한 뒤 `python web/export_models.py`와 `python web/build.py --export-source`를 실행한다. 체크포인트·릴리스 정보·보고서·정적 배포본을 함께 커밋한다. UI나 생성 모듈을 바꿀 때도 정적 배포본을 갱신한다. 빌드는 보고서와 가중치가 다르면 실패한다. 원본 FlyWire 데이터의 CC BY-NC 4.0 출처는 사이트의 LICENSES.txt와 저장소 README에서 확인할 수 있다.
