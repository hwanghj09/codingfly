"""Real browser validation of released neural inference and disposable execution."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class ProjectHandler(SimpleHTTPRequestHandler):
    def copyfile(self, source, outputfile):
        try:
            super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionResetError):
            # Stopping a disposable worker deliberately cancels its downloads.
            pass

    def do_GET(self):
        if not self.path.startswith("/codingfly/"):
            self.send_error(404)
            return
        self.path = self.path[len("/codingfly") :]
        super().do_GET()

    def log_message(self, *args):
        pass


def run(site, url, report_path, *, quick=False, executable=None):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    server = None
    if not url:
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            partial(ProjectHandler, directory=str(Path(site).resolve())),
        )
        Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}/codingfly/"
    expected = json.loads((ROOT / "docs/reports/accuracy-v3.json").read_text())[
        "hidden"
    ]["results"]
    if quick:
        expected = list({r["shape"]: r for r in expected}.values())
    result = {
        "url": url,
        "parity_cases": [],
        "console_errors": [],
        "external_requests": [],
    }
    origin = "{}://{}".format(*urlsplit(url)[:2])
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=executable)
        context = browser.new_context(viewport={"width": 1365, "height": 1150})
        context.on(
            "request",
            lambda request: (
                result["external_requests"].append(request.url)
                if not request.url.startswith(origin + "/")
                and not request.url.startswith(("data:", "blob:"))
                else None
            ),
        )
        page = context.new_page()
        page.on("pageerror", lambda error: result["console_errors"].append(str(error)))
        page.on(
            "console",
            lambda message: (
                result["console_errors"].append(message.text)
                if message.type == "error"
                else None
            ),
        )

        def generate(goal):
            page.locator("#goal").fill(goal)
            page.locator("#generate").click()
            page.wait_for_function(
                "!['', 'pending'].includes(document.querySelector('#python-code').dataset.status)",
                timeout=45000,
            )
            return {
                "status": page.locator("#python-code").get_attribute("data-status"),
                "ir": page.locator("#ir-code").text_content(),
                "code": page.locator("#python-code").text_content(),
            }

        def execute():
            page.locator("#run").click()
            page.wait_for_function(
                "['실행 완료','실행 종료'].includes(document.querySelector('#execution-state').textContent)",
                timeout=45000,
            )
            return page.locator("#output").text_content()

        try:
            page.goto(url, wait_until="networkidle")
            page.wait_for_function(
                "['준비 완료','모델 준비 실패'].includes(document.querySelector('#model-state').textContent)",
                timeout=120000,
            )
            assert page.locator("#model-state").inner_text() == "준비 완료", (
                page.locator("#generation-message").inner_text()
            )
            manifest = page.evaluate(
                "async () => (await fetch('./site-manifest.json', {cache:'no-cache'})).json()"
            )
            release = json.loads((ROOT / "web/checkpoints/release.json").read_text())
            assert manifest["models"] == release["models"], (
                "Live model release mismatch"
            )
            for number, row in enumerate(expected, 1):
                actual = generate(row["source"])
                status = "syntax_only" if row["supported"] else "uncertain"
                assert actual["status"] == status, (row["id"], actual)
                assert actual["ir"] == row["ir"], (row["id"], actual["ir"], row["ir"])
                if row["supported"]:
                    assert actual["code"] == row["code"], (
                        row["id"],
                        actual["code"],
                        row["code"],
                    )
                else:
                    assert not page.locator("#run").is_enabled()
                result["parity_cases"].append(row["id"])
                if number % 18 == 0:
                    print(f"Browser neural parity {number}/{len(expected)}", flush=True)
            page.get_by_role("button", name="변수 계산", exact=True).click()
            page.locator("#generate").click()
            page.wait_for_function(
                "document.querySelector('#python-code').dataset.status === 'syntax_only'",
                timeout=45000,
            )
            assert execute() == "-9\n"
            assert "합계 = -9" in page.locator("#variables").inner_text()
            assert "일치" in page.locator("#comparison").inner_text()
            page.locator(".compare-details > summary").click()
            page.locator("#expected-output").fill("incorrect")
            assert "다릅니다" in page.locator("#comparison").inner_text()
            page.locator("#expected-output").fill("-9\n")
            page.locator(".compare-details > summary").click()
            result["unicode_negative_execution"] = True
            page.screenshot(
                path=str(report_path.parent / "web-desktop.png"), full_page=True
            )
            page.get_by_role("button", name="함수", exact=True).click()
            page.locator("#generate").click()
            page.wait_for_function(
                "document.querySelector('#python-code').dataset.status === 'syntax_only'",
                timeout=45000,
            )
            execute()
            assert "반환값: 15" in page.locator("#return-value").inner_text()
            result["function_execution"] = True
            generate('"i" 로 0 이상 10001 미만을 순서대로 출력해')
            assert "10,000" in execute()
            assert page.locator("#execution-state").inner_text() == "실행 종료"
            result["execution_limit"] = True
            generate("37 을 출력해")
            page.locator("#run").click()
            page.locator("#stop-run").click()
            assert "중단" in page.locator("#output").inner_text()
            assert execute() == "37\n"
            result["cancel_and_rerun"] = True
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            ), "Mobile horizontal overflow"
            page.screenshot(
                path=str(report_path.parent / "web-mobile.png"), full_page=True
            )
            result["mobile"] = True
            runtime_base = manifest.get("runtime_base", "./runtime/")
            allowed_runtime = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/"
            unexpected = [
                request
                for request in result["external_requests"]
                if runtime_base != allowed_runtime
                or not request.startswith(allowed_runtime)
            ]
            assert not unexpected, unexpected
            assert not result["console_errors"], result["console_errors"]
            result.update(
                passed=True, browser=browser.version, models=manifest["models"]
            )
        except Exception as error:
            result.update(passed=False, error=str(error))
            page.screenshot(
                path=str(report_path.parent / "web-failure.png"), full_page=True
            )
            raise
        finally:
            report_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n"
            )
            browser.close()
            if server:
                server.shutdown()
    print(
        {"passed": True, "browser_parity": len(result["parity_cases"]), "url": url},
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=ROOT / "_site")
    parser.add_argument("--url")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "artifacts/browser-results.json"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--browser", default=os.environ.get("BROWSER_EXECUTABLE"))
    args = parser.parse_args()
    run(args.site, args.url, args.report, quick=args.quick, executable=args.browser)
