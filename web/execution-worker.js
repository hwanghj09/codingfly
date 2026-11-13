/* A fresh interpreter for each execution; it never loads or accesses model weights. */
onmessage = async ({ data }) => {
  try {
    importScripts("./runtime/pyodide.js");
    const python = await loadPyodide({
      indexURL: new URL("./runtime/", self.location.href).href,
    });
    const response = await fetch("./runner.py");
    if (!response.ok) throw new Error("실행 환경을 불러오지 못했습니다.");
    await python.runPythonAsync(await response.text());
    python.globals.set("browser_request", JSON.stringify(data));
    postMessage({ type: "executing" });
    const result = JSON.parse(
      python.runPython("execute(json.loads(browser_request))"),
    );
    postMessage({ type: "result", result });
  } catch (error) {
    postMessage({
      type: "error",
      message: String(error.message || error)
        .trim()
        .split("\n")
        .at(-1),
    });
  }
};
