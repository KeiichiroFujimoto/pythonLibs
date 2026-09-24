import platform
import socket
import subprocess
from typing import Dict, List, Optional

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - psutil not available in all envs
    psutil = None  # type: ignore


def _psutil_port_users(port: int, proto: str) -> List[Dict[str, Optional[str]]]:
    users: List[Dict[str, Optional[str]]] = []
    if psutil is None:
        return users

    proto = proto.lower()
    sock_type = socket.SOCK_STREAM if proto.startswith("t") else socket.SOCK_DGRAM
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.type == sock_type:
                pid = conn.pid
                name = None
                if pid:
                    try:
                        name = psutil.Process(pid).name()
                    except Exception:  # pragma: no cover - best-effort lookup
                        name = None
                users.append(
                    {
                        "pid": str(pid) if pid is not None else None,
                        "name": name,
                        "local_address": f"{conn.laddr.ip}:{conn.laddr.port}",
                    }
                )
    except Exception:  # pragma: no cover - psutil may fail on some platforms
        users = []
    return users


def _netstat_port_users_windows(port: int) -> List[Dict[str, Optional[str]]]:
    users: List[Dict[str, Optional[str]]] = []
    try:
        output = subprocess.check_output(["netstat", "-ano"], text=True, stderr=subprocess.DEVNULL)
    except Exception:  # pragma: no cover - netstat may not be available
        return users

    for line in output.splitlines():
        if "LISTEN" not in line.upper():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        local_addr = parts[1]
        pid = parts[-1]
        if local_addr.endswith(f":{port}"):
            users.append({"pid": pid, "name": None, "local_address": local_addr})
    return users


def _lsof_port_users_unix(port: int, proto: str) -> List[Dict[str, Optional[str]]]:
    users: List[Dict[str, Optional[str]]] = []
    proto_flag = "TCP" if proto.lower().startswith("t") else "UDP"
    try:
        output = subprocess.check_output(
            ["lsof", "-nP", f"-i{proto_flag}:{port}", "-s", "TCP:LISTEN"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # pragma: no cover - lsof may be unavailable
        return users

    lines = output.splitlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 9:
            continue
        name, pid, local_addr = parts[0], parts[1], parts[8]
        users.append({"pid": pid, "name": name, "local_address": local_addr})
    return users


def get_port_users(port: int, proto: str = "tcp") -> List[Dict[str, Optional[str]]]:
    """
    Return information about processes listening on the given port.

    Each entry contains ``pid``, ``name`` (if resolvable), and ``local_address``.
    """
    if port <= 0 or port > 65535:
        raise ValueError("port must be between 1 and 65535")

    users = _psutil_port_users(port, proto)
    if users:
        return users

    system = platform.system().lower()
    if "windows" in system:
        return _netstat_port_users_windows(port)

    return _lsof_port_users_unix(port, proto)


def is_port_in_use(port: int, proto: str = "tcp") -> bool:
    """Return True if at least one process is listening on the given port."""
    return bool(get_port_users(port, proto=proto))
