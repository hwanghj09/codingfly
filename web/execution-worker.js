/* A fresh interpreter for each execution; it never loads or accesses model weights. */
onmessage = async ({ data }) => {
  try {
    const manifestResponse = await fetch("./site-manifest.json", {
      cache: "no-cache",
    });
    if (!manifestResponse.ok)
      throw new Error("실행 정보를 불러오지 못했습니다.");
    const manifest = await manifestResponse.json();
    const runtime = new URL(
      manifest.runtime_base || "./runtime/",
      self.location.href,
    );
    if (
      runtime.origin !== self.location.origin &&
      runtime.href !== "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/"
    )
      throw new Error("알 수 없는 Python 실행 환경입니다.");
    importScripts(new URL("pyodide.js", runtime).href);
    const python = await loadPyodide({
      indexURL: runtime.href,
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
