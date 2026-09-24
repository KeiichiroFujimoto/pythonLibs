import os
import platform
import shutil
from pathlib import Path
from typing import List
import socket
import subprocess
from typing import Optional, Sequence, Callable

class osCommands:
    
    @staticmethod
    def getHomeDirectoryPath():
        return os.path.expanduser("~")
    
    @staticmethod
    def executeSystemCommand(cmd=None):
        if cmd is not None:
            os.system(cmd)
    
    @staticmethod
    def executeSystemCommandList(cmdList=None):
        for cmd in cmdList:
            osCommands.executeSystemCommand(cmd=cmd)

    @staticmethod
    def copy_file(source, destination):
        if source is None or destination is None:
            return
        src_path = Path(source)
        dest_path = Path(destination)
        dest_str = str(destination)

        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        if dest_path.exists() and dest_path.is_dir():
            dest_path = dest_path / src_path.name
        elif not dest_path.exists() and dest_str.endswith(('/', '\\')):
            dest_path = dest_path / src_path.name

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)

    @staticmethod
    def move_file(source, destination, exists_ok=False):

        # If the destination is a directory, append the source filename to the destination path
        if source is None or destination is None:
            return

        src_path = Path(source)
        dest_path = Path(destination)
        dest_str = str(destination)

        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        if dest_path.exists() and dest_path.is_dir():
            dest_path = dest_path / src_path.name
        elif not dest_path.exists() and dest_str.endswith(('/', '\\')):
            dest_path = dest_path / src_path.name

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists() and exists_ok is False:
            raise FileExistsError(f"Destination already exists: {dest_path}")
        if dest_path.exists() and exists_ok:
            if dest_path.is_file():
                dest_path.unlink()
            elif dest_path.is_dir():
                shutil.rmtree(dest_path)
        shutil.move(str(src_path), str(dest_path))
    
    @staticmethod
    def change_dir(destination=None):
        if destination is None:
            destination = osCommands.getHomeDirectoryPath()
        okayList = ['.','./','..','../']
        if os.path.isdir(destination) or destination in okayList:
            os.chdir(destination)
    
    @staticmethod
    def make_dir(target, overwrite=True):
        os.makedirs(target, exist_ok=overwrite)
    
    @staticmethod
    def getCommandList():
        system = platform.system()
        if system == 'Linux' or system == 'Darwin':  # Darwin is macOS
            import subprocess
            result   = subprocess.run(['bash', '-c', 'compgen -c'], capture_output=True, text=True)
            commandList = result.stdout.split()
            return commandList
        elif system == 'Windows':
            return None
        else:
            raise ValueError(f"Unsupported operating system: {system}")
    
    @staticmethod
    def getIpAddress(returnLocalIpAddress=False):
        if returnLocalIpAddress:
            return '127.0.0.1'
        else:
            try:
                # For Windows
                if platform.system() == "Windows":
                    hostname = socket.gethostname()
                    ip_address = socket.gethostbyname(hostname)
                else:
                    # For macOS and Linux
                    if platform.system() == "Darwin":  # macOS
                        ip_address = subprocess.check_output(['ifconfig', 'en0']).decode().strip()
                        ip_address = [line.split(' ')[1] for line in ip_address.split('\n') if 'inet ' in line][0]
                    else:  # Linux
                        ip_address = subprocess.check_output(['hostname', '-I']).decode().strip()                    
                return ip_address
            except Exception as e:
                return str(e)
    
    @staticmethod
    def makeSymbolicLink(source=None, destination=None):
        try:
            os.symlink(src=source, dst=destination)
        except Exception as e:
            return str(e)

    @staticmethod
    def deleteFile(filePath=None):
        try:
            os.remove(filePath)
            print(f"File {filePath} deleted successfully.")
        except Exception as e:
            print(f"Error deleting file {filePath}: {e}")
    
    @staticmethod
    def getCurrentDirectoryPath():
        return os.getcwd()
    
    @staticmethod
    def getMacAddress():
        import uuid
        import re
        mac = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
        return mac.replace(':', '')
    
    @staticmethod
    def copyFilesWithStructure(filePaths: List[str], src_root: str, dest_root: str):
        """
        Copy image files to a new destination while preserving folder structure.
        Args:
            file_paths (List[str]): List of image file paths relative to src_root.
            src_root (str): Source root directory.
            dest_root (str): Destination root directory.
        """
        for rel_path in filePaths:
            src_path  = os.path.join(src_root, rel_path)
            dest_path = os.path.join(dest_root, rel_path)
            # Create destination subdirectories if they don't exist
            osCommands.make_dir(target=os.path.dirname(dest_path), overwrite=True)
            # Copy file
            osCommands.copy_file(source=src_path, destination=dest_path)

    @staticmethod
    def copy_dir(source: str, destination: str, *, overwrite: bool = False,
                include_hidden: bool = True,
                ignore_patterns: Optional[Sequence[str]] = None,
                on_error: Optional[Callable[[Path, Exception], None]] = None) -> None:
        """
        Recursively copy a directory tree from `source` to `destination`.

        Parameters
        ----------
        source : str
            Source directory path.
        destination : str
            Destination directory path. If `overwrite=True`, an existing destination will be reused.
        overwrite : bool
            If True, allow copying into an existing directory (uses shutil.copytree(dirs_exist_ok=True)).
        include_hidden : bool
            If False, dotfiles/directories (e.g., .git) are skipped.
        ignore_patterns : Optional[Sequence[str]]
            Glob-style patterns to ignore (e.g., ["*.tmp", "__pycache__"]).
        on_error : Optional[Callable[[Path, Exception], None]]
            Optional error callback invoked per-file when a copy fails.
        """
        src = Path(source)
        dst = Path(destination)
        if not src.exists() or not src.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source}")
        # Prepare ignore callable
        ignore: Optional[Callable[[str, List[str]], List[str]]] = None
        patterns = list(ignore_patterns or [])
        if not include_hidden:
            patterns.append('.*')  # ignore dotfiles/dirs at each level
        if patterns:
            def _ignore(dirpath: str, names: List[str]) -> List[str]:
                from fnmatch import fnmatch
                ignore_list: List[str] = []
                for name in names:
                    for pat in patterns:
                        if fnmatch(name, pat):
                            ignore_list.append(name)
                            break
                return ignore_list
            ignore = _ignore
        # Execute copytree
        try:
            shutil.copytree(src, dst, dirs_exist_ok=overwrite, ignore=ignore)
        except Exception as exc:
            print('Error exc:'+str(exc))
            if on_error:
                on_error(dst, exc)
            else:
                raise
    
    @staticmethod
    def delete_dir(target: str, *, force: bool = True, only_if_empty: bool = False,
                    on_error: Optional[Callable[[Path, Exception], None]] = None) -> None:
        """
        Delete a directory.

        Parameters
        ----------
        target : str
            Target directory path to delete.
        force : bool
            If True (default), delete recursively with shutil.rmtree().
            If False, do not delete non-empty directories (use Path.rmdir()).
        only_if_empty : bool
            If True, delete only when the directory is empty; raises if not empty.
        on_error : Optional[Callable[[Path, Exception], None]]
            Optional error callback invoked when deletion fails.

        Notes
        -----
        - Windows, macOS, Linux で動作。
        - `force=True` は中身があっても削除（要注意）。安全運用では `only_if_empty=True`
          と組み合わせて使用してください。
        """
        tgt = Path(target)
        if not tgt.exists():
            # Nothing to do
            return
        if not tgt.is_dir():
            raise NotADirectoryError(f"Target is not a directory: {target}")

        try:
            if only_if_empty:
                # Remove only if empty; raises OSError if not empty
                tgt.rmdir()
            else:
                if force:
                    # Recursive delete
                    shutil.rmtree(tgt)
                else:
                    # Non-recursive: allow delete only if empty (same as rmdir)
                    tgt.rmdir()
        except Exception as exc:
            if on_error:
                on_error(tgt, exc)
            else:
                raise
    
    # ============================
    # Copy a directory (recursively) — folder itself
    # ============================
    @staticmethod
    def copy_dir(source: str, destination: str, *, overwrite: bool = False,
                include_hidden: bool = True,
                ignore_patterns: Optional[Sequence[str]] = None,
                on_error: Optional[Callable[[Path, Exception], None]] = None) -> None:
        """
        Recursively copy a directory tree from `source` to `destination`.

        Behavior:
        - If `destination` points to an existing directory or ends with a path separator,
          the *source folder itself* is created under `destination` (i.e., dest/SourceName/... ).
        - Otherwise, the tree is copied *into* `destination` path as the target folder name.

        Parameters
        ----------
        source : str
            Source directory path.
        destination : str
            Destination directory path or parent directory (see Behavior).
        overwrite : bool
            If True, allow copying into an existing directory. When copying into a concrete target
            (e.g., dest/SourceName), an existing target will be removed before copy.
        include_hidden : bool
            If False, dotfiles/directories (e.g., .git) are skipped.
        ignore_patterns : Optional[Sequence[str]]
            Glob-style patterns to ignore (e.g., ["*.tmp", "__pycache__"]).
        on_error : Optional[Callable[[Path, Exception], None]]
            Optional error callback invoked per-file when a copy fails.
        """
        src = Path(source).expanduser().resolve()
        dst_in = Path(destination).expanduser().resolve()
        if not src.exists() or not src.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source}")

        # Decide concrete destination (copy folder itself when dest is a directory or endswith separator)
        dest_str = str(destination)
        if dst_in.exists() and dst_in.is_dir():
            dst = dst_in / src.name
        elif not dst_in.exists() and dest_str.endswith(('/', '\\')):
            dst = dst_in / src.name
        else:
            dst = dst_in  # destination is the exact target folder path

        # Prepare ignore callable
        ignore: Optional[Callable[[str, List[str]], List[str]]] = None
        patterns = list(ignore_patterns or [])
        if not include_hidden:
            patterns.append('.*')  # ignore dotfiles/dirs at each level
        if patterns:
            def _ignore(dirpath: str, names: List[str]) -> List[str]:
                from fnmatch import fnmatch
                ignore_list: List[str] = []
                for name in names:
                    for pat in patterns:
                        if fnmatch(name, pat):
                            ignore_list.append(name)
                            break
                return ignore_list
            ignore = _ignore
        try:
            # If target exists and overwrite=True, remove before copy to ensure full folder copy
            if dst.exists() and overwrite:
                if dst.is_file():
                    dst.unlink()
                else:
                    shutil.rmtree(dst)
            shutil.copytree(src, dst, dirs_exist_ok=False, ignore=ignore)
        except Exception as exc:
            if on_error:
                on_error(dst, exc)
            else:
                raise

