"""Request and verify a Pages build using its existing main/root source."""

import json
import os
from pathlib import Path
import subprocess
import time


def api(endpoint, *, method="GET"):
    return json.loads(
        subprocess.check_output(["gh", "api", "--method", method, endpoint], text=True)
    )


def deploy():
    repository = os.environ["GITHUB_REPOSITORY"]
    commit = os.environ["GITHUB_SHA"]
    site = api(f"repos/{repository}/pages")
    if site.get("build_type") != "legacy" or site.get("source") != {
        "branch": "main",
        "path": "/",
    }:
        raise ValueError("Expected the existing main/root Pages source")
    latest = api(f"repos/{repository}/pages/builds/latest")
    # A normal main push already schedules a build; reuse it when possible.
    requested = (
        latest
        if latest.get("commit") == commit
        and latest.get("status") in {"queued", "building", "built"}
        else api(f"repos/{repository}/pages/builds", method="POST")
    )
    prefix = f"https://api.github.com/repos/{repository}/pages/builds/"
    url = requested.get("url", prefix + "latest")
    if not url.startswith(prefix):
        raise ValueError("Unexpected Pages build endpoint")
    endpoint = f"repos/{repository}/pages/builds/" + url[len(prefix) :]
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        build = requested
        print(f"Pages build: {build['status']}", flush=True)
        if build["status"] == "errored":
            raise RuntimeError(
                build.get("error", {}).get("message") or "Pages build failed"
            )
        if build["status"] == "built" and build.get("commit") == commit:
            with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
                output.write(f"page_url={site['html_url']}\n")
            print(site["html_url"], flush=True)
            return
        time.sleep(5)
        requested = api(endpoint)
    raise TimeoutError("Pages did not publish this commit within ten minutes")


if __name__ == "__main__":
    deploy()
