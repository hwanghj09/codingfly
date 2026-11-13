# Fly-only 실험 절차

> 이 문서는 최초 v1 실험의 절차를 보존한다. 현재 v2 신경 시퀀스·Fly-NL·종단 검증의 재현 방법은 [DUAL_FLY_V2.md](DUAL_FLY_V2.md)를 사용한다. v1의 P3 실패 결과는 [초기 보고서](INITIAL_IMPLEMENTATION_RESULTS.md)에 있다.

## 경로와 연구 계약

`lab.py`/`fly.py`/기존 `benchmark.py`는 legacy다. 새 추론은 `dual_fly.py → models.fly_code → flybrain`으로 실행한다. `evaluation`은 학습 자료·오차·실행 채점 담당이며 모델이 답을 선택하는 데 사용하지 않는다. `ir`는 형식 검사만 한다. Python 생성기는 기존 모델/번역기/정규식/외부 LLM을 import할 수 없고, 추론 중 네트워크·프로세스 실행도 차단한다. 이 의존성 가드는 연구 경로의 실수 방지 장치이며 악의적인 Python을 격리하는 샌드박스와는 별개다.

가중치가 학습되는 곳은 CSV에 이미 존재하는 방향성 간선뿐이다. 별도 학습 출력층·임베딩·정답 검색 사전은 없다. 양의 `syn_count`를 목적지별로 정규화하여 초기화한다. 기본 부호는 모두 흥분성이라는 **인공 가정**이고, presynaptic 뉴런의 20%를 고정 난수로 억제성으로 정한 민감도 대조군을 함께 기록한다. 원본 신경전달물질 정보가 있다는 주장은 하지 않는다.

공통 뉴런 인덱스, 원본 ID/방향/시냅스 수, CSR/CSC 인덱스를 보존한다. 실행 커널은 원본 간선 배열과 `numpy.bincount`를 쓰는 O(뉴런+간선) 희소 합산이다. 모든 출력 뉴런은 입력 자극 대상에서 제외한다. 고정 indegree 순위로 출력 뉴런을 선택하며, 출력 없는 그래프/잘못된 CSV는 실패한다. 재연결은 directed double-edge swap으로 in/out degree와 간선 수를 보존한다. 무작위 대조군은 같은 뉴런 수·간선 수·시냅스 수 목록을 사용하되 정도 분포까지 일치시키지는 않는다.

## 고정된 역학과 학습 규칙

기본값은 `steps=4`, `leak=.65`, `gain=2`, `output_scale=6`, `max_weight=3`이다.

```
proposal[t] = tanh(stimulus + gain * sparse_weighted_activity[t-1])
activity[t] = (1-leak) * activity[t-1] + leak * proposal[t]
probabilities = softmax(output_scale * activity[last][readout])
```

감독 cross entropy의 BPTT 오차를 **기존 간선에만** 역전파한다. 업데이트는 부호 보존·크기 제한으로 투영한다. `eligibility`에는 이 감독 오차의 간선별 descent trace를 기록한다. 이것은 STDP나 생물학적 3요인 보상 학습을 재현했다는 뜻이 아니다. frozen에서는 가중치와 eligibility를 변경하지 않는다. 단위 테스트는 수치 미분으로 gradient를 교차 검증한다.

토큰별 순환 상태는 초기화하고 현재 원문·출력 접두사·출력 위치를 다시 고정 부호화한다. 따라서 신경 역학은 각 선택 안에서 4번 반복하지만 토큰을 넘나드는 장기 기억은 출력 접두사의 재입력에 의존한다. 이 제한을 장기 신경 기억으로 해석하지 않는다.

## 재현 명령

저장소 루트에서 실행한다. 과거 run의 manifest와 새 데이터/IR/그래프 계약이 다르면 덮어쓰지 않고 실패한다. 조건을 바꿀 때는 새 출력 디렉터리를 사용한다.

```bash
python -m pip install -r requirements.txt
python -m evaluation.baseline --output artifacts/baseline
python -m unittest -v
python benchmark.py
python -m evaluation.toy --output artifacts/toy-p2
python -m models.train_code --toy-report artifacts/toy-p2/report.json --output artifacts/fly-code
python dual_fly.py --checkpoint artifacts/fly-code/learned.json --dataset artifacts/fly-code/dataset.json --ir "LET novel_name 37" --trace
```

`report.json`은 Python/NumPy/플랫폼, 기준 커밋, dirty 여부, 소스 파일 해시, 그래프 파일/구조 해시, 데이터/IR 해시, 시드·예산·가중치 해시·변경 수·분포·학습 시간과 실패 출력을 포함한다. 체크포인트는 JSON checksum과 모델 역할/그래프/데이터/IR/encoder 계약을 검사하고 원자적으로 저장한다. A/B 역할이 다르면 읽지 못한다. 신경 상태·감독 trace·업데이트 횟수와 훈련 RNG 상태도 저장한다. 현재 CLI는 새 고정 예산 훈련을 수행하며, 자동 이어학습 인터페이스는 아직 없다. 엔진의 저장 후 추론 및 다음 시냅스 업데이트는 bit-exact 복원 테스트로 확인한다.

완전한 상태 로그 대신 생성 trace에 각 선택의 실제 출력 뉴런 ID/활동/확률, 전체 시간 활동의 해시, 그래프/시냅스 해시를 기록한다. 전체 시간별 뉴런 값은 `Brain.forward(..., record=True)`로 재현할 수 있다.

검토용 작은 JSON을 저장하려면 baseline/toy/code 하위 디렉터리를 가진 동일 run 폴더에서 `python -m evaluation.snapshot --run-dir artifacts/reproduction --test-log /tmp/codingfly-final-tests.log`를 실행한다. 이는 로컬 문서 파일을 작성하며 모델이나 평가 데이터를 변경하지 않는다.

## 데이터 잠금과 학습/평가 분리

P2는 첫 바이트 `A/B/C/D`의 4분류다. 두 번째 nuisance 바이트 ID를 기준으로 그룹을 나눠 train/validation/test를 **120/36/36**으로 분리한다. P3는 LET, PRINT, LET+PRINT의 세 형태이며 변수명·상수 묶음을 기준으로 **72/24/24**로 분리한다. 분할 시드는 1729이며, 그룹 해시 순서로 고정한다. [잠긴 데이터](../experiments/locked/)와 생성기의 일치를 테스트한다. 다른 이름·수치에 대한 **표면 일반화**만 측정하며 알고리즘 계열 홀드아웃은 아직 없다.

P2는 5개 시드(0~4), 매 시드 learned/frozen/degree-preserving rewired/random/inhibitory20을 같은 30 epoch, 학습률 .03으로 실행한다. 추가로 전달을 끈 평가에서 우회 신호가 없는지 확인한다. 모든 시드에서 learned≥.75, learned>frozen, learned 가중치 변경, frozen 불변, 정확 복원, 전달 제거 시 chance 이하가 P3 착수 조건이다. 출력 readout은 각 그래프의 고정 indegree 순위다. degree-preserving rewired에서는 원본과 같고, random에서는 달라질 수 있다. 이 차이를 구조적 우월성의 증거로 사용하지 않는다.

P3는 시드 0, 40 epoch, 학습률 .03으로 learned와 frozen을 비교한다. 모델 선택·재시도 없이 정해진 예산을 마친 뒤 validation과 test를 각각 한 번 평가한다. 훈련은 정답 접두사를 제공하는 teacher forcing, 평가는 자기 출력을 입력하는 자기회귀 생성이다. 최대 32동작, EOS/실패 상태를 기록한다. 출력 어휘는 EOS, 공백, 개행, `=`, `print`, `(`, `)`와 16개의 원문 span COPY 선택이다. 공백 분리는 바이트 경계만 정하며 명령 뜻·인자 역할을 추출하지 않는다. 정답 Python/teacher 동작은 **학습 자료 제작기 안에서만** 만든다.

문법 통과율은 24개 전체 대비, 실행 의미 정확도는 문법 통과 프로그램 대비, Pass@1은 EOS까지 완료하고 실제 동작이 맞은 수/24다. 단지 실행되는 미완성 접두사는 정답으로 세지 않는다. P3의 큰 목표(20~30개 유형, 제어 흐름, 문법≥90%, 실행 의미≥80%)를 충족하지 못하면 P4로 넘어가지 않는다. test 결과로 epoch나 encoder를 튜닝한 값을 같은 hidden 성과로 재보고하지 않는다.

5개 시드 평균/표본 표준편차와 정규 근사 95% 구간은 기술 통계다. 작은 표본과 겹치는 구간에서 유의한 배선 우월성을 주장하지 않는다.

## 격리 실행

Python은 Linux `bubblewrap --unshare-all`의 별도 user/mount/PID/network namespace에서 실행한다. 런타임은 읽기 전용이고 host workspace/home, 환경변수, 비밀키를 넘기지 않는다. 자원은 CPU 1초, 메모리 256MiB, 프로세스 1개, 출력 파일 64KiB로 제한하고 부모에서 wall timeout을 적용한다. AST 허용 목록으로 import/속성 접근/I/O 등도 거부한다. 이 AST 검사는 실행 코드를 만들어주지 않는다.

격리 기능이 없거나 namespace 생성이 실패하면 채점 실패로 기록한다. 호스트 `exec`/단순 subprocess로 대체하지 않는다. 테스트는 실제 네트워크 차단·읽기 전용 런타임·host 경로 비노출·환경변수 제거·자원 제한을 probe하고, 무한 루프·출력 폭주·외부 접근 거부도 확인한다. UI 공개/범용 Python 샌드박스 서비스는 이번 구현 범위에 없다.
