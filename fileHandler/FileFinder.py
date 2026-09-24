import os
import fnmatch

class FileFinder:

    def __init__(self) -> None:
        pass

    def __del__(self) -> None:
        pass

    @staticmethod
    def findDirectory(root_dir, target_dir, max_depth):
        pathList = []
        for root, dirs, files in os.walk(root_dir):
            # Calculate the current depth
            current_depth = root[len(root_dir):].count(os.sep)
        
            if current_depth > max_depth:
                # Skip directories that are deeper than the max_depth
                dirs[:] = []
                continue
        
            for dir_name in dirs:
                if fnmatch.fnmatch(dir_name, target_dir):
                    pathList.append(os.path.join(root, dir_name))
        return pathList

    @staticmethod
    def findItem(root_dir, target_name, max_depth):
        pathList = []
        for root, dirs, files in os.walk(root_dir):
            # Calculate the current depth
            current_depth = root[len(root_dir):].count(os.sep)
        
            if current_depth > max_depth:
                # Skip directories that are deeper than the max_depth
                dirs[:] = []
                continue
        
            # Check for matching directories
            for dir_name in dirs:
                if fnmatch.fnmatch(dir_name, target_name):
                    pathList.append(os.path.join(root, dir_name))
        
            # Check for matching files
            for file_name in files:
                if fnmatch.fnmatch(file_name, target_name):
                    pathList.append(os.path.join(root, file_name))

        return pathList
