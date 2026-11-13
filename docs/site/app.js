const $ = (id) => document.getElementById(id);
const examples = [
  { label: "정수 출력", goal: "37 을 출력해", expected: "37\n" },
  { label: "대입", goal: '변수 "점수" 에 7 를 저장해', expected: "" },
  {
    label: "변수 계산",
    goal: '변수 "합계" 에 -13 를 저장한 뒤 4 를 더해서 출력해',
    expected: "-9\n",
  },
  {
    label: "함수",
    goal: '"왼쪽" 와 "오른쪽" 의 합을 반환하는 함수 "더하기" 를 만들어',
    args: "[7, 8]",
    expected: "",
  },
  {
    label: "반복문",
    goal: '"i" 로 0 이상 4 미만을 순서대로 출력해',
    expected: "0\n1\n2\n3\n",
  },
  {
    label: "문자열",
    goal: '"안녕, CodingFly!" 을 출력해',
    expected: "안녕, CodingFly!\n",
  },
];
let worker,
  executionWorker,
  ready = false,
  generating = false,
  executing = false;
let job = 0,
  generationTimer,
  startupTimer,
  executionTimer,
  result = null,
  executionResult = null;
let presetArgs = "[]";
const setMessage = (id, text, kind = "") => {
  $(id).textContent = text;
  $(id).className = `message ${kind}`;
};
function controls() {
  $("generate").disabled = !ready || generating || executing;
  $("generate-label").textContent = generating
    ? "두 모델이 코드를 만들고 있어요…"
    : ready
      ? "코드 생성"
      : "모델 불러오는 중…";
  $("goal").disabled = generating || executing;
  document.querySelectorAll(".example").forEach((b) => {
    b.disabled = generating || executing;
  });
  $("cancel-generation").hidden = !generating;
  $("run").disabled =
    generating || executing || result?.status !== "syntax_only";
  $("run").textContent = executing ? "실행 중…" : "▶ Python 실행";
  $("stop-run").hidden = !executing;
  $("copy-code").disabled = !result?.code;
}
function clearResult() {
  result = null;
  executionResult = null;
  $("python-code").textContent = "생성한 코드가 여기에 표시됩니다.";
  $("python-code").classList.add("empty");
  $("python-code").dataset.status = "";
  $("ir-code").textContent = "아직 생성된 의사코드가 없습니다.";
  $("syntax-badge").hidden = true;
  $("function-fields").hidden = true;
  $("output").textContent = "코드를 만든 뒤 실행해 보세요.";
  $("variables").hidden = $("return-value").hidden = true;
  $("execution-state").textContent = "실행 대기";
  setMessage("generation-message", "");
  setMessage("comparison", "");
  controls();
}
function initModel() {
  worker?.terminate();
  clearTimeout(startupTimer);
  ready = false;
  $("model-dot").className = "status-dot";
  $("model-state").textContent = "실행 환경 준비 중";
  $("loading-note").hidden = false;
  $("reload-model").hidden = true;
  controls();
  worker = new Worker(new URL("./inference-worker.js", import.meta.url), {
    name: "codingfly-inference",
  });
  startupTimer = setTimeout(
    () =>
      modelFailure(
        "모델 준비가 오래 걸리고 있습니다. 연결을 확인하고 다시 불러와 주세요.",
      ),
    120000,
  );
  worker.onerror = () =>
    modelFailure(
      "모델을 불러오지 못했습니다. 새로고침하거나 다시 불러와 주세요.",
    );
  worker.onmessage = ({ data }) => {
    if (data.type === "progress")
      $("loading-message").textContent = data.message;
    if (data.type === "ready") {
      clearTimeout(startupTimer);
      ready = true;
      $("model-dot").className = "status-dot ready";
      $("model-state").textContent = "준비 완료";
      $("loading-note").hidden = true;
      controls();
    }
    if (data.type === "error" && data.phase === "initialize")
      modelFailure(data.message);
    if (data.id !== job) return;
    if (data.type === "result") {
      clearTimeout(generationTimer);
      generating = false;
      result = data.result;
      $("python-code").dataset.status = result.status;
      $("python-code").classList.toggle("empty", !result.code);
      $("python-code").textContent =
        result.code || "완성된 Python 코드가 없습니다.";
      $("ir-code").textContent = result.ir || "의사코드가 생성되지 않았습니다.";
      $("syntax-badge").hidden = result.status !== "syntax_only";
      $("trace-label").textContent =
        `추론 ${result.trace_id.slice(0, 8)} · Fly-NL → Fly-Code`;
      $("function-name").replaceChildren();
      for (const fn of result.functions || []) {
        const option = document.createElement("option");
        option.value = fn.name;
        option.textContent = `${fn.name}(${fn.parameters.join(", ")})`;
        $("function-name").append(option);
      }
      $("function-fields").hidden = !result.functions?.length;
      $("function-args").value = presetArgs;
      if (result.status === "syntax_only")
        setMessage(
          "generation-message",
          "코드를 생성하고 구문을 확인했습니다. 실행해서 동작을 확인해 보세요.",
          "success",
        );
      else if (result.status === "uncertain")
        setMessage(
          "generation-message",
          "모델이 아직 지원하지 않는 요청으로 판단했습니다. 지원 범위와 예시를 확인해 주세요.",
        );
      else
        setMessage(
          "generation-message",
          `모델이 코드를 완성하지 못했습니다. 표현을 간단히 바꿔 보세요. (${result.status})`,
          "error",
        );
      controls();
    }
    if (data.type === "error" && generating) {
      clearTimeout(generationTimer);
      generating = false;
      $("python-code").dataset.status = "error";
      setMessage("generation-message", data.message, "error");
      controls();
    }
  };
}
function modelFailure(message) {
  worker?.terminate();
  clearTimeout(startupTimer);
  clearTimeout(generationTimer);
  ready = generating = false;
  $("model-dot").className = "status-dot error";
  $("model-state").textContent = "모델 준비 실패";
  $("loading-note").hidden = true;
  $("reload-model").hidden = false;
  $("python-code").dataset.status = "error";
  setMessage("generation-message", message, "error");
  controls();
}
$("goal-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!ready || generating || executing) return;
  const goal = $("goal").value.trim();
  if (!goal || new TextEncoder().encode(goal).length > 2048) {
    setMessage(
      "generation-message",
      "요청을 2,048바이트 이내로 입력해 주세요.",
      "error",
    );
    return;
  }
  clearResult();
  generating = true;
  $("python-code").dataset.status = "pending";
  $("python-code").textContent =
    "Fly-NL이 의사코드를 만들고, Fly-Code가 Python을 생성하고 있습니다…";
  controls();
  job += 1;
  generationTimer = setTimeout(
    () =>
      modelFailure(
        "생성 시간이 초과됐습니다. 모델을 다시 불러오고 더 짧은 요청으로 시도해 주세요.",
      ),
    45000,
  );
  worker.postMessage({ type: "generate", goal, id: job });
});
$("goal").addEventListener("input", () => {
  clearResult();
  $("compare-enabled").checked = false;
  presetArgs = "[]";
  document
    .querySelectorAll(".example")
    .forEach((b) => b.classList.remove("active"));
});
$("goal").addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter")
    $("goal-form").requestSubmit();
});
for (const example of examples) {
  const button = document.createElement("button");
  button.className = `example${example.label === "변수 계산" ? " active" : ""}`;
  button.type = "button";
  button.textContent = example.label;
  button.addEventListener("click", () => {
    clearResult();
    $("goal").value = example.goal;
    presetArgs = example.args || "[]";
    $("expected-output").value = example.expected;
    $("compare-enabled").checked = true;
    document
      .querySelectorAll(".example")
      .forEach((b) => b.classList.toggle("active", b === button));
    $("goal").focus();
  });
  $("examples").append(button);
}
$("expected-output").value = "-9\n";
$("compare-enabled").checked = true;
function compareOutput() {
  if (!executionResult || !$("compare-enabled").checked)
    return setMessage("comparison", "");
  const expected = $("expected-output").value.replace(/\r\n/g, "\n");
  const matched = executionResult.stdout === expected;
  setMessage(
    "comparison",
    matched
      ? "✓ 기대한 표준 출력과 일치합니다."
      : "기대한 표준 출력과 다릅니다. 코드와 요청을 확인해 주세요.",
    matched ? "success" : "error",
  );
}
$("expected-output").addEventListener("input", compareOutput);
$("compare-enabled").addEventListener("change", compareOutput);
function stopExecution(message) {
  clearTimeout(executionTimer);
  executionWorker?.terminate();
  executionWorker = null;
  executing = false;
  $("execution-state").textContent = "실행 종료";
  $("output").textContent = message;
  controls();
}
$("run").addEventListener("click", () => {
  if (result?.status !== "syntax_only" || executing) return;
  let args = [];
  try {
    if (!$("function-fields").hidden)
      args = JSON.parse($("function-args").value);
    if (
      !Array.isArray(args) ||
      args.length > 8 ||
      JSON.stringify(args).length > 2048
    )
      throw new Error();
  } catch {
    setMessage(
      "comparison",
      "함수 입력은 [7, 8]처럼 JSON 배열로 입력해 주세요.",
      "error",
    );
    return;
  }
  executionResult = null;
  executing = true;
  $("variables").hidden = $("return-value").hidden = true;
  $("execution-state").textContent = "실행 환경 준비";
  $("output").textContent = "별도의 Python 실행 환경을 준비하고 있습니다…";
  setMessage("comparison", "");
  controls();
  executionWorker = new Worker(
    new URL("./execution-worker.js", import.meta.url),
    { name: "codingfly-execution" },
  );
  executionTimer = setTimeout(
    () =>
      stopExecution("실행 환경 준비 시간이 초과됐습니다. 다시 실행해 주세요."),
    30000,
  );
  executionWorker.onerror = () =>
    stopExecution("실행 환경에 문제가 생겼습니다. 다시 실행해 주세요.");
  executionWorker.onmessage = ({ data }) => {
    if (data.type === "executing") {
      clearTimeout(executionTimer);
      executionTimer = setTimeout(
        () =>
          stopExecution("실행 시간이 초과됐습니다. 반복 범위를 줄여 주세요."),
        5000,
      );
      $("execution-state").textContent = "Python 실행 중";
      return;
    }
    if (data.type === "error") return stopExecution(data.message);
    if (data.type !== "result") return;
    executionResult = data.result;
    stopExecution(executionResult.stdout || "표준 출력이 없습니다.");
    $("execution-state").textContent = "실행 완료";
    const entries = Object.entries(executionResult.variables);
    if (entries.length) {
      $("variables").textContent = entries
        .map(([key, value]) => `${key} = ${JSON.stringify(value)}`)
        .join("\n");
      $("variables").hidden = false;
    }
    if (executionResult.called) {
      $("return-value").textContent =
        `반환값: ${JSON.stringify(executionResult.return_value)}`;
      $("return-value").hidden = false;
    }
    compareOutput();
  };
  executionWorker.postMessage({
    code: result.code,
    function_name: $("function-fields").hidden
      ? null
      : $("function-name").value,
    args,
  });
});
$("stop-run").addEventListener("click", () =>
  stopExecution("실행을 중단했습니다."),
);
$("cancel-generation").addEventListener("click", () => {
  generating = false;
  clearTimeout(generationTimer);
  clearResult();
  initModel();
  setMessage(
    "generation-message",
    "생성을 중단했습니다. 모델을 다시 준비합니다.",
  );
});
$("reload-model").addEventListener("click", () => {
  clearResult();
  initModel();
});
$("copy-code").addEventListener("click", async () => {
  if (!result?.code) return;
  try {
    await navigator.clipboard.writeText(result.code);
    $("copy-code").textContent = "복사 완료";
    setTimeout(() => {
      $("copy-code").textContent = "복사";
    }, 1500);
  } catch {
    setMessage("generation-message", "코드를 선택해 직접 복사해 주세요.");
  }
});
initModel();
