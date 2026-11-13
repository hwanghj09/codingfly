"""Fail closed when Linux bubblewrap OS isolation is unavailable; never falls back to exec."""

import json
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile


WORKER = Path(__file__).with_name("_sandbox_worker.py")


def run_python(code, *, variables=(), calls=(), timeout=3., _probe=False):
    if not isinstance(code, str) or len(code.encode("utf-8")) > 12000 or not 0 < timeout <= 10:
        raise ValueError("Invalid sandbox input or timeout")
    bubblewrap = shutil.which("bwrap")
    if not bubblewrap or not Path("/usr/bin/python3").exists():
        return {"status": "sandbox_unavailable", "error": "Linux bubblewrap and /usr/bin/python3 required"}
    request = json.dumps({"code": code, "variables": list(variables), "calls": list(calls),
                          "probe": _probe}).encode()
    if len(request) > 65536:
        raise ValueError("Oversized sandbox request")
    # Fresh user/mount/PID/network/IPC/UTS/cgroup namespaces; no home, workspace,
    # credentials or host /etc. Only runtime mounts (read only) and this worker.
    command = [bubblewrap, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
               "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
               "--proc", "/proc", "--dev", "/dev", "--dir", "/tmp",
               "--ro-bind", str(WORKER), "/runner.py", "--clearenv", "--chdir", "/tmp"]
    if Path("/lib64").exists():
        command += ["--ro-bind", "/lib64", "/lib64"]
    command += ["--remount-ro", "/", "--", "/usr/bin/python3", "-I", "-S", "/runner.py"]
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr)
            try:
                process.communicate(request, timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                return {"status": "timeout", "error": "Wall-clock budget exceeded"}
        except OSError as error:
            return {"status": "sandbox_unavailable", "error": str(error)}
        stdout.seek(0)
        stderr.seek(0)
        output, errors = stdout.read(65537), stderr.read(4096)
    if process.returncode:
        if process.returncode in (-signal.SIGKILL, -signal.SIGXCPU,
                                  128 + signal.SIGKILL, 128 + signal.SIGXCPU):
            return {"status": "timeout", "error": "CPU/memory budget exceeded"}
        if process.returncode in (-signal.SIGXFSZ, 128 + signal.SIGXFSZ) or len(output) >= 65536:
            return {"status": "output_limit"}
        detail = errors.decode("utf-8", errors="replace")
        return {"status": "sandbox_unavailable" if "bwrap:" in detail else "sandbox_error", "error": detail}
    if len(output) > 65536:
        return {"status": "output_limit"}
    try:
        return json.loads(output)
    except (ValueError, UnicodeDecodeError):
        return {"status": "sandbox_error", "error": "Invalid worker response"}


def isolation_status():
    """Trusted fixed probe; does not run user-supplied probe code."""
    return run_python("", _probe=True)
