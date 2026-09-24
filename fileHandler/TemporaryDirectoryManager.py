import logging
import shutil
from pathlib import Path
from typing import Optional


class TemporaryDirectoryManager:
    """Utility helpers for pruning temporary working directories."""

    @staticmethod
    def prune(base_path: str, max_count: int, prefix: Optional[str] = None) -> None:
        """Ensure only the newest `max_count` directories remain under `base_path`.

        Args:
            base_path: Directory whose immediate subdirectories will be pruned.
            max_count: Maximum number of directories to keep. Non-positive values skip pruning.
            prefix: Optional directory-name prefix filter (e.g. 'tmpDir_').
        """
        if max_count <= 0:
            return

        root = Path(base_path)
        if not root.exists() or not root.is_dir():
            return

        try:
            candidates = [
                entry for entry in root.iterdir()
                if entry.is_dir() and (prefix is None or entry.name.startswith(prefix))
            ]
        except OSError as exc:
            logging.warning("TemporaryDirectoryManager: failed listing %s: %s", base_path, exc)
            return

        if len(candidates) <= max_count:
            return

        # Oldest first (based on modification time)
        candidates.sort(key=lambda entry: entry.stat().st_mtime)
        to_remove = candidates[:-max_count]

        for directory in to_remove:
            try:
                shutil.rmtree(directory, ignore_errors=False)
            except Exception as exc:  # noqa: BLE001
                logging.warning("TemporaryDirectoryManager: failed removing %s: %s", directory, exc)
