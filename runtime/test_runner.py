import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Tuple

from tools.repro_command_registry import ReproCommandRegistry, ReproRecipe
from tools.test_command_registry import TestCommandRegistry, format_command


# Some old Truffle projects mutate files and share a local Ganache instance.  A
# test therefore runs in an APFS clone of the already prepared project.  The
# semaphore limits those clones to the requested number of simultaneous tests.
_SERIAL_TEST_RUN_LOCK = threading.Lock()
_TEST_GATES: dict[int, threading.BoundedSemaphore] = {}
_TEST_GATES_LOCK = threading.Lock()
_LOCK_WAIT_LOCAL = threading.local()
_CHAIN_FAILURE_MARKERS = (
    "insufficient funds",
    "Could not connect to your Ethereum client",
    "timeout while attempting to connect to the network",
    "header not found",
    "Cannot convert string to buffer",
    "Migration._deploy",
    "ProviderError",
    "sender doesn't have enough funds",
    "Deployment Failed",
)
_CHAIN_START_RE = re.compile(r"\b(?:ganache-cli|ganache-core|testrpc(?:-[\w-]+)?)\b", re.I)


def reset_test_run_lock_wait_time() -> None:
    """Reset per-thread time spent waiting for a test slot."""
    _LOCK_WAIT_LOCAL.seconds = 0.0


def get_test_run_lock_wait_time() -> float:
    """Return per-thread time spent waiting for a test slot."""
    return float(getattr(_LOCK_WAIT_LOCAL, "seconds", 0.0))


def _record_lock_wait(seconds: float) -> None:
    _LOCK_WAIT_LOCAL.seconds = get_test_run_lock_wait_time() + seconds


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _log_indicates_chain_failure(log: str) -> bool:
    if not log:
        return False
    lowered = log.lower()
    return any(marker.lower() in lowered for marker in _CHAIN_FAILURE_MARKERS)


def _should_run_pre_test(recipe: ReproRecipe, test_file: str) -> bool:
    if os.environ.get("SCT_SKIP_PRE_TEST", "").strip() == "1":
        return False
    if os.environ.get("SCT_FORCE_CHAIN_RESET", "").strip() == "1":
        return True
    return ReproCommandRegistry.is_non_original_test_file(recipe, test_file)


def _test_gate(workers: int) -> threading.BoundedSemaphore:
    with _TEST_GATES_LOCK:
        gate = _TEST_GATES.get(workers)
        if gate is None:
            gate = threading.BoundedSemaphore(workers)
            _TEST_GATES[workers] = gate
        return gate


class TestRunner:
    def __init__(self, project_path: str, record_index: int | None = None):
        self.project_path = project_path
        self.record_index = record_index
        self.repro_registry = ReproCommandRegistry()
        self.command_registry = TestCommandRegistry()

    @staticmethod
    def reset_lock_wait_time() -> None:
        reset_test_run_lock_wait_time()

    @staticmethod
    def get_lock_wait_time() -> float:
        return get_test_run_lock_wait_time()

    @staticmethod
    def _parallel_tests_enabled() -> bool:
        if os.environ.get("SCT_PARALLEL_TESTS"):
            return _env_flag("SCT_PARALLEL_TESTS")
        return _env_flag("SCT_PARALLEL_MODELS")

    @staticmethod
    def _test_workers() -> int:
        raw_value = os.environ.get("SCT_TEST_WORKERS", "3")
        try:
            workers = int(raw_value)
        except ValueError:
            workers = 3
        return max(1, min(workers, 6))

    def _run_shell_script(
        self,
        script: str,
        cwd: str,
        timeout: int,
        label: str,
        env: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        print(f"⚙️ {label}")
        try:
            result = subprocess.run(
                script,
                cwd=cwd,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            log = (result.stdout or "") + (result.stderr or "")
            return result.returncode == 0, log
        except Exception as e:
            return False, str(e)

    def _refresh_chain_state(self, recipe: ReproRecipe, cwd: str) -> tuple[bool, str]:
        script = self.repro_registry.render_pre_test_script(recipe)
        if not script.strip():
            return True, ""
        return self._run_shell_script(
            script,
            cwd=cwd,
            timeout=int(os.environ.get("SCT_PRE_TEST_TIMEOUT", "900")),
            label="Refreshing blockchain state before test (pre_test_commands)",
        )

    def _run_repro_test_command(
        self,
        recipe: ReproRecipe,
        cwd: str,
        test_file: str,
        env: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        cmd = self.repro_registry.format_test_command(recipe, test_file)
        print(f"⚙️ Running repro recipe command: {cmd}")
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                shell=True,
                executable="/bin/bash",
                capture_output=True,
                text=True,
                timeout=int(os.environ.get("SCT_TEST_TIMEOUT", "7200")),
                env=env,
            )
            log = (result.stdout or "") + (result.stderr or "")
            return result.returncode == 0, log
        except Exception as e:
            return False, str(e)

    @staticmethod
    def _clone_project(source: Path) -> tuple[Path, Path]:
        cache_root = Path(__file__).resolve().parents[1] / ".cache" / "test_sandboxes"
        cache_root.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix=f"{source.name}-", dir=cache_root))
        sandbox = work_dir / "project"
        try:
            # On APFS this is copy-on-write, so node_modules is not copied byte-for-byte.
            subprocess.run(
                ["/bin/cp", "-cR", str(source), str(sandbox)],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            shutil.copytree(source, sandbox, symlinks=True)
        return work_dir, sandbox

    @staticmethod
    def _find_chain_start_command(recipe: ReproRecipe) -> tuple[str, int] | None:
        """Return the Ganache/testrpc launch command and the port it binds."""
        for command in reversed(recipe.setup_commands):
            if not _CHAIN_START_RE.search(command):
                continue
            port_match = re.search(r"(?:--port\s+|-p\s+)(\d{2,5})\b", command)
            if not port_match:
                port_match = re.search(r"\.listen\(\s*(\d{2,5})\b", command)
            if port_match:
                return command, int(port_match.group(1))
        return None

    @staticmethod
    def _reserve_local_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _replace_port(text: str, old_port: int, new_port: int) -> str:
        old = re.escape(str(old_port))
        new = str(new_port)
        text = re.sub(rf"(?:(--port\s+)|(-p\s+)){old}\b", lambda m: f"{m.group(1) or m.group(2)}{new}", text)
        text = re.sub(rf"(\bport\s*:\s*){old}\b", rf"\g<1>{new}", text)
        text = re.sub(rf"((?:127\.0\.0\.1|localhost)\s*:\s*){old}\b", rf"\g<1>{new}", text)
        return re.sub(rf"(\.listen\(\s*){old}\b", rf"\g<1>{new}", text)

    @classmethod
    def _rewrite_sandbox_ports(cls, sandbox: Path, old_port: int, new_port: int) -> None:
        for filename in ("truffle.js", "truffle-config.js", "hardhat.config.js", "hardhat.config.ts", "buidler.config.js"):
            config_path = sandbox / filename
            if not config_path.is_file():
                continue
            original = config_path.read_text(encoding="utf-8")
            rewritten = cls._replace_port(original, old_port, new_port)
            if rewritten != original:
                config_path.write_text(rewritten, encoding="utf-8")

    @staticmethod
    def _wait_for_port(port: int, process: subprocess.Popen, timeout: float = 45.0) -> tuple[bool, str]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                try:
                    sock.connect(("127.0.0.1", port))
                    return True, ""
                except OSError:
                    pass
            if process.poll() is not None:
                return False, f"Local chain exited before listening on port {port}."
            time.sleep(0.25)
        return False, f"Timed out waiting for local chain on port {port}."

    @staticmethod
    def _stop_process_group(process: subprocess.Popen | None) -> None:
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            try:
                process.terminate()
            except OSError:
                return
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass

    def _sandbox_cwd(self, sandbox: Path, source_cwd: str) -> Path:
        source_root = Path(self.project_path).resolve()
        target_cwd = Path(source_cwd).resolve()
        try:
            return sandbox / target_cwd.relative_to(source_root)
        except ValueError:
            # project_path is normally the recipe cwd.  Keeping the clone root
            # is safer than following an absolute path back into the shared repo.
            return sandbox

    def _run_isolated(
        self,
        test_file: str,
        framework: str,
        repro_recipe: ReproRecipe | None,
        command_entry: dict | None,
    ) -> Tuple[bool, str]:
        source = Path(self.project_path).resolve()
        relative_test = Path(test_file)
        if relative_test.is_absolute() or ".." in relative_test.parts:
            return False, f"Refusing to run a test outside its isolated project: {test_file}"
        if not source.is_dir():
            return False, f"Project directory does not exist: {source}"

        work_dir: Path | None = None
        chain_process: subprocess.Popen | None = None
        try:
            work_dir, sandbox = self._clone_project(source)
            if not (sandbox / relative_test).exists():
                return False, f"Generated test is missing from isolated project: {relative_test}"

            env = os.environ.copy()
            env["SCT_TEST_SANDBOX"] = str(sandbox)
            cwd = sandbox
            recipe = repro_recipe
            if recipe:
                cwd = self._sandbox_cwd(sandbox, recipe.cwd or self.project_path)
                if not cwd.is_dir():
                    return False, f"Isolated recipe working directory is missing: {cwd}"
                recipe = replace(recipe, cwd=str(cwd))

                chain = self._find_chain_start_command(recipe)
                if chain:
                    chain_command, original_port = chain
                    port = self._reserve_local_port()
                    self._rewrite_sandbox_ports(sandbox, original_port, port)
                    chain_command = self._replace_port(chain_command, original_port, port)
                    chain_command = re.sub(r"\s*&\s*$", "", chain_command.strip())
                    chain_log = work_dir / "local-chain.log"
                    print(f"⚙️ Isolated local chain: {original_port} → {port}")
                    with chain_log.open("w", encoding="utf-8") as log_file:
                        chain_process = subprocess.Popen(
                            chain_command,
                            cwd=cwd,
                            shell=True,
                            executable="/bin/bash",
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            text=True,
                            env=env,
                            start_new_session=True,
                        )
                    ready, message = self._wait_for_port(port, chain_process)
                    if not ready:
                        chain_output = chain_log.read_text(encoding="utf-8", errors="replace")
                        return False, f"{message}\n{chain_output}"
                    env["SCT_RPC_PORT"] = str(port)

                print(f"⚙️ Isolated test sandbox: {sandbox}")
                return self._run_repro_test_command(recipe, str(cwd), test_file, env=env)

            if command_entry:
                entry_cwd = command_entry.get("cwd") or self.project_path
                cwd = self._sandbox_cwd(sandbox, entry_cwd)
                if not cwd.is_dir():
                    return False, f"Isolated command working directory is missing: {cwd}"
                cmd = format_command(command_entry["command"], test_file)
                print(f"⚙️ Isolated test sandbox: {sandbox}")
                return self._run_shell_script(
                    cmd,
                    str(cwd),
                    int(os.environ.get("SCT_TEST_TIMEOUT", "7200")),
                    f"Running: {cmd}",
                    env=env,
                )

            fw = framework.strip().lower()
            if fw == "hardhat":
                cmd = "npx hardhat test " + subprocess.list2cmdline([test_file])
            elif fw == "foundry":
                cmd = "forge test --match-path " + subprocess.list2cmdline([test_file])
            elif fw == "truffle":
                local_truffle = sandbox / "node_modules" / ".bin" / "truffle"
                truffle_cmd = str(local_truffle) if local_truffle.exists() else "truffle"
                cmd = subprocess.list2cmdline([truffle_cmd, "test", test_file])
            else:
                return False, f"Unknown framework: {framework!r} (use hardhat, foundry, truffle)"
            print(f"⚙️ Isolated test sandbox: {sandbox}")
            return self._run_shell_script(
                cmd,
                str(sandbox),
                int(os.environ.get("SCT_TEST_TIMEOUT", "7200")),
                f"Running: {cmd}",
                env=env,
            )
        except Exception as e:
            return False, str(e)
        finally:
            self._stop_process_group(chain_process)
            if work_dir is not None:
                shutil.rmtree(work_dir, ignore_errors=True)

    def run(self, test_file: str, framework: str = "hardhat") -> Tuple[bool, str]:
        repro_recipe = self.repro_registry.find(
            record_index=self.record_index,
            project_path=self.project_path,
        )
        command_entry = None if repro_recipe else self.command_registry.find(
            record_index=self.record_index,
            test_file=test_file,
            project_path=self.project_path,
        )

        # Recipes with pre_test_commands deliberately reset the shared chain.
        # They remain serial; 311+ has no such recipe and therefore uses the
        # isolated path below.
        if self._parallel_tests_enabled() and not (repro_recipe and repro_recipe.pre_test_commands):
            wait_start = time.perf_counter()
            gate = _test_gate(self._test_workers())
            with gate:
                _record_lock_wait(time.perf_counter() - wait_start)
                return self._run_isolated(test_file, framework, repro_recipe, command_entry)

        wait_start = time.perf_counter()
        with _SERIAL_TEST_RUN_LOCK:
            _record_lock_wait(time.perf_counter() - wait_start)
            return self._run_unlocked(test_file, framework=framework, repro_recipe=repro_recipe, command_entry=command_entry)

    def _run_unlocked(
        self,
        test_file: str,
        framework: str = "hardhat",
        repro_recipe: ReproRecipe | None = None,
        command_entry: dict | None = None,
    ) -> Tuple[bool, str]:
        if repro_recipe:
            cwd = repro_recipe.cwd or self.project_path
            if repro_recipe.pre_test_commands and _should_run_pre_test(repro_recipe, test_file):
                ok, pre_log = self._refresh_chain_state(repro_recipe, cwd)
                if not ok:
                    return False, pre_log

            success, log = self._run_repro_test_command(repro_recipe, cwd, test_file)
            if (
                not success
                and repro_recipe.pre_test_commands
                and _log_indicates_chain_failure(log)
                and os.environ.get("SCT_SKIP_PRE_TEST", "").strip() != "1"
            ):
                print("⚙️ Chain/migration failure detected; resetting chain and retrying once")
                ok, pre_log = self._refresh_chain_state(repro_recipe, cwd)
                if not ok:
                    return False, pre_log + log
                retry_success, retry_log = self._run_repro_test_command(repro_recipe, cwd, test_file)
                return retry_success, log + "\n\n--- RETRY AFTER CHAIN RESET ---\n\n" + retry_log
            return success, log

        if command_entry:
            cwd = command_entry.get("cwd") or self.project_path
            cmd = format_command(command_entry["command"], test_file)
            return self._run_shell_script(
                cmd,
                cwd,
                int(os.environ.get("SCT_TEST_TIMEOUT", "7200")),
                f"Running: {cmd}",
            )

        fw = framework.strip().lower()
        if fw == "hardhat":
            cmd = ["npx", "hardhat", "test", test_file]
        elif fw == "foundry":
            cmd = ["forge", "test", "--match-path", test_file]
        elif fw == "truffle":
            local_truffle = Path(self.project_path) / "node_modules" / ".bin" / "truffle"
            truffle_cmd = str(local_truffle) if local_truffle.exists() else "truffle"
            cmd = [truffle_cmd, "test", test_file]
        else:
            return False, f"Unknown framework: {framework!r} (use hardhat, foundry, truffle)"

        return self._run_shell_script(
            " ".join(cmd),
            self.project_path,
            int(os.environ.get("SCT_TEST_TIMEOUT", "7200")),
            f"Running: {' '.join(cmd)}",
        )
