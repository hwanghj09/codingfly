# 단회 자연어 정확도 개선

P6의 재시도는 그대로 두고, Fly-NL의 감독 자료와 의도 선택용 고정 입력 뷰를 개선해 원본 FlyWire 간선 강도를 다시 학습한다. Fly-Code와 포인터·토큰 디코더는 유지한다. 외부 NLP 분류기·언어 모델·정답 코드 보정은 추가하지 않는다.

기존 v2의 알려진 조사·표현 변형은 v3에서 **공개 학습 자료**로 취급한다. 따라서 v2 문장의 개선 점수는 회귀 검사이며 새 일반화 결과가 아니다. 새 주 평가는 별도 문장 조합과 이름·상수로 잠근 test split에서 기존 체크포인트와 새 체크포인트를 같은 조건으로 비교한다. 두 모델 모두 피드백이나 재시도 없이 greedy로 한 번 생성한다.

같은 최종 평가에서 기존 실행 정답은 56/96, 개선 모델은 96/96이었다. 미지원 응답도 8/12에서 12/12로 개선됐다. 모델 선택에 쓰지 않은 결과이며 평가 후 가중치는 변경하지 않았다. [전체 결과](IMPLEMENTATION_RESULTS.md)와 [JSON 근거](reports/accuracy-v3.json)를 함께 확인한다.

## 자료와 학습

- 한국어 8종 제한된 과제와 미지원 요청을 유지한다. 식별자는 JSON 큰따옴표로, 수치·조사는 공백으로 구분한다.
- 최종 자료 버전은 v3.2다. train 552개(지원 504개), validation 72개(지원 64개), test 108개(지원 96개)로 나눈다. test는 지원 과제 8종의 새 표현 조합 3개와 미지원 표현 3개에 이름·수치 그룹 4개를 적용한 것이다. 96개 독립 의도나 알고리즘 계열을 평가하는 것은 아니다.
- 조사 변형과 위치 부사를 보강하고, 새 어순·단어 조합을 validation/test에 분리한다. 이름·숫자를 가린 입력 구조도 세 split 사이에 겹치지 않는지 검사한다.
- 원문 span을 복사하는 teacher와 문장 제작기는 offline 감독 전용이다. 생성 경로에서 새 데이터·teacher 모듈을 import하면 실패한다.
- 문맥 후보 점수에는 불투명 리터럴 종류와 원문 단어·바이트의 고정 해시 bag를 사용한다. 의도 선택에서 절대 숫자 위치와 단어 순서 자극을 제거해, 동일 단어를 순서만 바꾼 입력이 같은 의도 자극을 받도록 한다. 포인터 후보와 출력 토큰의 기존 위치 자극은 유지한다. 뜻풀이 사전·학습 가능한 외부 계수는 없고, 점수 계산과 최종 후보 선택은 여전히 그래프 신경 활동에 따른다.
- 새 고정 I/O 버전은 checkpoint에 기록한다. 복원은 알려진 두 인코더만 선택하고, 데이터·그래프·IR·I/O 버전을 검증한다. 기존 v2 checkpoint의 입력 계산은 유지한다.
- v2 A 체크포인트에서 시작해 기존 5,340개 간선만 Adam과 bounded L-BFGS-B로 갱신한다. B 체크포인트는 고정한다. 새 study 디렉터리·데이터 해시·parent 시냅스 해시를 기록한다.
- 체크포인트는 validation만으로 선택한다. 지원 IR 정확도 95%와 미지원 불확실 응답 90%를 넘기 전에는 test를 열지 않는다. test 시작 표식을 먼저 저장하고, 이후 재학습·최적화·재평가를 막는다.

## 재현

v2 문서의 `artifacts/nl-v2`와 `artifacts/code-v2`를 먼저 준비한다.

```bash
OPENBLAS_NUM_THREADS=1 python -m models.train_nl_v3 \
  --parent artifacts/nl-v2 --output artifacts/nl-v3 \
  --epochs 20 --polish-steps 160
python -m evaluation.accuracy_v3 --output artifacts/nl-v3
python -m evaluation.snapshot_accuracy --directory artifacts/nl-v3
python -m unittest -v
python benchmark.py
```

아직 test를 열지 않은 study의 추가 최적화는 `--polish-only`로 명시한다. 이미 평가한 v2/P6의 기록·가중치를 덮어쓰지 않는다. 전체 trace와 checkpoint는 `artifacts/nl-v3/`, 검토용 결과와 데이터는 `docs/reports/accuracy-v3.json`, `experiments/locked/nl-v3.json`에 저장한다.

초기 504개 자료의 validation에서는 지원 IR 64/64, 미지원 응답 4/8이었다. 부사 “지금”이 지원 요청에만 붙어 있던 상관관계를 제거하도록 미지원 학습 문장 24개에 위치 변형을 추가했다. 기존 위치 인코더에서는 자료 보강 후에도 같은 validation 결과였다. 의도 입력을 순서에 둔감한 뷰로 바꾼 528개 자료 실험에서는 지원 60/64, 미지원 8/8이었다. 남은 함수명/인자명 혼동에 대해 함수 생성 동사의 동등 표현을 추가한 것이 최종 552개 자료다. validation/test는 모든 개발 실험에서 동일하다. 예비 study의 test는 모두 열지 않았고 별도 디렉터리에 보존했다. `--legacy-context --profile initial --output artifacts/nl-v3-initial`, `--legacy-context --profile contrastive --output artifacts/nl-v3-position`, `--profile contrastive --output artifacts/nl-v3-invariant-probe`로 예비 실험도 재현할 수 있다.

실행 정답, IR 형식·구조·의미, 미지원 응답을 분리해 보고한다. 정답 IR→B는 별도 진단으로만 수행하며 생성물에 끼워 넣지 않는다. 함수는 여러 인자와 경계값으로 OS 격리 실행한다. 저장·복원 추론 일치, 평가 중 A/B 시냅스 불변, parent 대비 바뀐 간선 수도 확인한다.

범위는 같은 8종 과제의 문장 조합 일반화다. 순서가 의미를 바꾸는 임의 지시나 알고리즘 계열 전체를 홀드아웃한 평가, 자유로운 한국어·영어 이해를 입증하지 않는다. warm start 한 시드의 실험이며 다중 학습 시드의 우월성 주장을 하지 않는다.

순서 정규화 후 지원 test의 8/96개, 미지원 test의 12/12개는 학습과 동일한 의도 자극이다. 원문·포인터·전체 IR과 실행은 별도 평가하지만, 이 사례를 새로운 의도 표현의 완전한 독립 평가로 보지 않는다. `dataset_audit.context_view`에 실제 입력 해시의 겹침 수를 기록한다.
