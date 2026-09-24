import platform
from typing import List


def get_terminate_command(pid: int, force: bool = True) -> List[str]:
    """
    Return a command list that can terminate the given process ID.

    Parameters
    ----------
    pid : int
        Process ID to terminate.
    force : bool, optional
        If True, request an immediate/forceful termination. Defaults to True.
    """
    if pid <= 0:
        raise ValueError("pid must be a positive integer")

    system = platform.system().lower()
    if "windows" in system:
        command = ["taskkill", "/PID", str(pid)]
        if force:
            command.append("/F")
        return command

    # Default to POSIX-compliant behaviour for Linux/macOS.
    signal_flag = "-9" if force else "-15"
    return ["kill", signal_flag, str(pid)]


def get_terminate_command_string(pid: int, force: bool = True) -> str:
    """
    Return the terminate command as a single string (for display purposes).
    """
    return " ".join(get_terminate_command(pid, force=force))
