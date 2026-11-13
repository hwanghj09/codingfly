/* Runs the released Python implementation and actual synaptic weights in WebAssembly. */
let python;
const notify = (message) => postMessage({ type: "progress", message });
const ready = (async () => {
  notify("Python 실행 환경을 준비하고 있어요");
  const response = await fetch("./site-manifest.json", { cache: "no-cache" });
  if (!response.ok) throw new Error("모델 정보를 불러오지 못했습니다.");
  const manifest = await response.json();
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
  python = await loadPyodide({
    indexURL: runtime.href,
  });
  await python.loadPackage("numpy");
  notify("학습된 두 모델을 확인하고 있어요");
  const download = await fetch(new URL(manifest.bundle, self.location.href));
  if (!download.ok) throw new Error("모델 파일을 불러오지 못했습니다.");
  const bundle = await download.arrayBuffer();
  const digest = [
    ...new Uint8Array(await crypto.subtle.digest("SHA-256", bundle)),
  ]
    .map((x) => x.toString(16).padStart(2, "0"))
    .join("");
  if (digest !== manifest.bundle_sha256)
    throw new Error("모델 파일 검증에 실패했습니다. 새로고침해 주세요.");
  python.unpackArchive(bundle, "zip", { extractDir: "/codingfly" });
  await python.runPythonAsync(`
import os, sys
os.chdir('/codingfly')
sys.path.insert(0, '/codingfly')
import browser_bridge
`);
  postMessage({ type: "ready", manifest });
})();
ready.catch((error) => {
  console.error("CodingFly model initialization", error);
  postMessage({
    type: "error",
    phase: "initialize",
    message:
      "모델을 준비하지 못했습니다. 다시 불러오거나 잠시 후 새로고침해 주세요.",
  });
});
let queue = Promise.resolve();
onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try {
      await ready;
      if (data.type !== "generate" || typeof data.goal !== "string")
        throw new Error("잘못된 생성 요청입니다.");
      python.globals.set("browser_goal", data.goal);
      const result = JSON.parse(
        python.runPython("browser_bridge.generate(browser_goal)"),
      );
      python.globals.delete("browser_goal");
      postMessage({ type: "result", id: data.id, result });
    } catch (error) {
      postMessage({
        type: "error",
        id: data.id,
        message: String(error.message || error)
          .trim()
          .split("\n")
          .at(-1),
      });
    }
  });
};
