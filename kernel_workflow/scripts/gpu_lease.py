#!/usr/bin/env python3
"""Acquire a fixed group of GPU locks and run one command under the lease."""

# Python 3.8 is supported, so retain typing.Optional/Tuple instead of PEP 604/585.
# ruff: noqa: UP006, UP045, PYI034

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple

TIMEOUT_EXIT_CODE = 124


class GpuRequest:
    def __init__(self, visible_ids: Tuple[int, ...]):
        self.visible_ids = visible_ids
        self.lock_ids = tuple(sorted(visible_ids))

    @classmethod
    def from_fixed_ids(cls, value: str, *, count: int) -> GpuRequest:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("at least one GPU is required")

        ids = []
        for part in parts:
            if not part.isdigit():
                raise ValueError(f"invalid GPU id: {part!r}")
            ids.append(int(part))

        if len(ids) != len(set(ids)):
            raise ValueError("duplicate GPU ids are not allowed")
        if count != len(ids):
            raise ValueError(
                f"count={count} does not match fixed GPU group size {len(ids)}"
            )
        return cls(tuple(ids))


class LeaseTimeout(TimeoutError):
    """Raised when a complete GPU group cannot be acquired before the deadline."""


class CommandStartError(RuntimeError):
    def __init__(self, message: str, *, exit_code: int):
        super().__init__(message)
        self.exit_code = exit_code


class SysfsIdleChecker:
    def __init__(
        self,
        *,
        drm_root: Path = Path("/sys/class/drm"),
        max_busy_pct: int = 5,
        max_vram_mb: int = 1024,
        device_map: Optional[Mapping[int, Path]] = None,
        fail_open: bool = True,
    ):
        self.drm_root = Path(drm_root)
        self.max_busy_pct = int(max_busy_pct)
        self.max_vram_bytes = (
            None if int(max_vram_mb) < 0 else int(max_vram_mb) * 1024 * 1024
        )
        self.device_map = dict(device_map or {})
        self.fail_open = bool(fail_open)

    def __call__(self, gpu_id: int) -> bool:
        device = self._device_path(gpu_id)
        if device is None:
            return self.fail_open
        busy = _read_int(device / "gpu_busy_percent")
        if busy is None:
            return self.fail_open
        vram = _read_int(device / "mem_info_vram_used")
        if self.max_vram_bytes is not None and vram is None:
            return self.fail_open
        vram_ok = self.max_vram_bytes is None or vram <= self.max_vram_bytes
        return busy <= self.max_busy_pct and vram_ok

    def _device_path(self, gpu_id: int) -> Optional[Path]:
        mapped = self.device_map.get(gpu_id)
        if mapped is not None:
            return Path(mapped)
        card_device = self.drm_root / f"card{gpu_id}" / "device"
        if card_device.exists():
            return card_device
        render_devices = sorted(
            self.drm_root.glob("renderD*/device"),
            key=lambda path: int(path.parent.name[len("renderD") :]),
        )
        if 0 <= gpu_id < len(render_devices):
            return render_devices[gpu_id]
        return None


class GpuLease:
    def __init__(
        self,
        request: GpuRequest,
        *,
        lock_dir: Path,
        wait_timeout_s: float,
        poll_interval_s: float = 0.2,
        idle_checker: Optional[Callable[[int], bool]] = None,
    ):
        self.request = request
        self.lock_dir = Path(lock_dir)
        self.wait_timeout_s = max(0.0, float(wait_timeout_s))
        self.poll_interval_s = max(0.001, float(poll_interval_s))
        self.idle_checker = idle_checker
        self._gpu_fds = []
        self.lease_id = f"{os.getpid()}-{time.time_ns()}"
        self.metadata_path = self.lock_dir / f"lease_{self.lease_id}.json"
        self._metadata = {
            "lease_id": self.lease_id,
            "manager_pid": os.getpid(),
            "pid_namespace": _pid_namespace(),
            "gpu_ids": list(request.visible_ids),
            "lock_ids": list(request.lock_ids),
            "started_at_ns": time.time_ns(),
        }

    def __enter__(self) -> GpuLease:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def acquire(self) -> None:
        if self._gpu_fds:
            raise RuntimeError("GPU lease is already acquired")

        self.lock_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.wait_timeout_s
        attempted = False
        try:
            while True:
                if attempted and time.monotonic() >= deadline:
                    self._raise_timeout()
                attempted = True
                acquired = self._try_acquire_group()
                if acquired:
                    if self._has_live_stale_overlap() or self.idle_checker is not None and not all(
                        self.idle_checker(gpu_id)
                        for gpu_id in self.request.visible_ids
                    ):
                        self.release()
                    else:
                        self._write_metadata()
                        return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._raise_timeout()
                time.sleep(min(self.poll_interval_s, remaining))
        except BaseException:
            self.release()
            raise

    def _raise_timeout(self) -> None:
        ids = ",".join(str(gpu_id) for gpu_id in self.request.visible_ids)
        raise LeaseTimeout(
            f"failed to acquire GPU group [{ids}] after {self.wait_timeout_s:g}s"
        )

    def release(self) -> None:
        try:
            self.metadata_path.unlink()
        except FileNotFoundError:
            pass
        for fd in reversed(self._gpu_fds):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        self._gpu_fds.clear()

    def update_metadata(self, **values) -> None:
        self._metadata.update(values)
        self._write_metadata()

    def _write_metadata(self) -> None:
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.metadata_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._metadata, sort_keys=True))
        os.replace(temporary, self.metadata_path)

    def _has_live_stale_overlap(self) -> bool:
        requested = set(self.request.visible_ids)
        for path in self.lock_dir.glob("lease_*.json"):
            if path == self.metadata_path:
                continue
            try:
                metadata = json.loads(path.read_text())
                stale_ids = {int(gpu_id) for gpu_id in metadata.get("gpu_ids", [])}
                child_pgid = int(metadata.get("child_pgid", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                return True
            if not requested.intersection(stale_ids):
                continue
            stale_namespace = metadata.get("pid_namespace")
            if stale_namespace and stale_namespace != _pid_namespace():
                return True
            if child_pgid > 0 and _process_group_exists(child_pgid):
                return True
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        return False

    def _try_acquire_group(self) -> bool:
        allocator_fd = os.open(
            self.lock_dir / "allocator.lock",
            os.O_CREAT | os.O_RDWR,
            0o666,
        )
        allocator_locked = False
        try:
            try:
                fcntl.flock(allocator_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                allocator_locked = True
            except BlockingIOError:
                return False
            for gpu_id in self.request.lock_ids:
                fd = os.open(
                    self.lock_dir / f"gpu_{gpu_id}.lock",
                    os.O_CREAT | os.O_RDWR,
                    0o666,
                )
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    self.release()
                    return False
                self._gpu_fds.append(fd)
            return True
        finally:
            if allocator_locked:
                fcntl.flock(allocator_fd, fcntl.LOCK_UN)
            os.close(allocator_fd)


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_namespace() -> str:
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return "unknown"


def parse_amd_smi_device_map(
    payload: str, *, pci_root: Path = Path("/sys/bus/pci/devices")
) -> Mapping[int, Path]:
    records = json.loads(payload)
    mapping = {}
    for record in records:
        mapping[int(record["gpu"])] = Path(pci_root) / str(record["bdf"])
    return mapping


def discover_amd_smi_device_map() -> Mapping[int, Path]:
    try:
        result = subprocess.run(
            ["amd-smi", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return parse_amd_smi_device_map(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return {}


def run_command(
    request: GpuRequest,
    command: Sequence[str],
    *,
    lock_dir: Path,
    wait_timeout_s: float,
    run_timeout_s: float,
    term_grace_s: float,
    env: Optional[Mapping[str, str]] = None,
    idle_checker: Optional[Callable[[int], bool]] = None,
) -> int:
    if not command:
        raise ValueError("command must not be empty")

    child_env = os.environ.copy()
    if env is not None:
        child_env.update(env)
    visible = ",".join(str(gpu_id) for gpu_id in request.visible_ids)
    child_env.update(
        {
            "HIP_VISIBLE_DEVICES": visible,
            "CUDA_VISIBLE_DEVICES": visible,
            "GEAK_GPU_GROUP": visible,
            "GEAK_GPU_LEASE_ACTIVE": "1",
            "GEAK_GPU_LEASE_IDS": visible,
        }
    )

    process = None
    previous_handlers = {}

    def forward_signal(signum, _frame):
        if process is not None:
            _terminate_process_group(process, term_grace_s=term_grace_s)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, forward_signal)

    try:
        with GpuLease(
            request,
            lock_dir=lock_dir,
            wait_timeout_s=wait_timeout_s,
            idle_checker=idle_checker,
        ) as lease:
            try:
                managed_signals = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
                previous_mask = signal.pthread_sigmask(
                    signal.SIG_BLOCK, managed_signals
                )

                def restore_child_signal_mask():
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

                try:
                    try:
                        process = subprocess.Popen(
                            list(command),
                            start_new_session=True,
                            env=child_env,
                            close_fds=True,
                            # gpu_lease.py is a dedicated, single-threaded CLI process.
                            # The child must not inherit the short signal mask used to
                            # close the Popen-before-assignment race in the parent.
                            preexec_fn=restore_child_signal_mask,  # noqa: PLW1509
                        )
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                except FileNotFoundError as error:
                    raise CommandStartError(
                        f"command not found: {command[0]}", exit_code=127
                    ) from error
                except PermissionError as error:
                    raise CommandStartError(
                        f"command is not executable: {command[0]}", exit_code=126
                    ) from error
                lease.update_metadata(child_pgid=process.pid)
                return_code = process.wait(timeout=max(0.0, float(run_timeout_s)))
                _terminate_process_group(process, term_grace_s=term_grace_s)
                return _shell_exit_code(return_code)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process, term_grace_s=term_grace_s)
                return TIMEOUT_EXIT_CODE
            except BaseException:
                if process is not None:
                    _terminate_process_group(process, term_grace_s=term_grace_s)
                raise
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def _terminate_process_group(
    process: subprocess.Popen, *, term_grace_s: float
) -> None:
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return

    deadline = time.monotonic() + max(0.0, float(term_grace_s))
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(pgid):
            break
        time.sleep(0.01)

    if _process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _shell_exit_code(return_code: int) -> int:
    return 128 + (-return_code) if return_code < 0 else return_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run", help="run one command under a fixed GPU group lease")
    run.add_argument("--fixed-ids", required=True)
    run.add_argument("--count", type=int)
    run.add_argument(
        "--lock-dir",
        default=os.environ.get("GEAK_GPU_LOCK_DIR", "/tmp/team_gpu_locks"),
    )
    run.add_argument("--wait-timeout", type=float, default=1200.0)
    run.add_argument("--run-timeout", type=float, default=900.0)
    run.add_argument("--term-grace", type=float, default=5.0)
    run.add_argument("--require-idle", action="store_true")
    run.add_argument(
        "--sysfs-root",
        default=os.environ.get("GEAK_GPU_SYSFS_ROOT", "/sys/class/drm"),
    )
    run.add_argument(
        "--max-busy-pct",
        type=int,
        default=int(os.environ.get("GEAK_GPU_MAX_BUSY_PCT", "5")),
    )
    run.add_argument(
        "--max-vram-mb",
        type=int,
        default=int(os.environ.get("GEAK_GPU_MAX_VRAM_MB", "-1")),
    )
    run.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if os.environ.get("GEAK_GPU_LEASE_ACTIVE") == "1":
        parser.error("nested GPU lease requests are not supported")

    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    try:
        raw_ids = [part.strip() for part in args.fixed_ids.split(",") if part.strip()]
        count = args.count if args.count is not None else len(raw_ids)
        request = GpuRequest.from_fixed_ids(args.fixed_ids, count=count)
        idle_checker = None
        if args.require_idle:
            device_map = {}
            if Path(args.sysfs_root) == Path("/sys/class/drm"):
                device_map = discover_amd_smi_device_map()
                missing = set(request.visible_ids) - set(device_map)
                if missing:
                    missing_text = ",".join(str(gpu_id) for gpu_id in sorted(missing))
                    raise ValueError(
                        "cannot resolve sysfs devices for required GPU idle check: "
                        f"{missing_text}; set GEAK_GPU_REQUIRE_IDLE=0 to bypass"
                    )
            idle_checker = SysfsIdleChecker(
                drm_root=Path(args.sysfs_root),
                max_busy_pct=args.max_busy_pct,
                max_vram_mb=args.max_vram_mb,
                device_map=device_map,
                fail_open=False,
            )
        return run_command(
            request,
            command,
            lock_dir=Path(args.lock_dir),
            wait_timeout_s=args.wait_timeout,
            run_timeout_s=args.run_timeout,
            term_grace_s=args.term_grace,
            idle_checker=idle_checker,
        )
    except CommandStartError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return error.exit_code
    except (ValueError, LeaseTimeout) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
