import glob
import os

from pythonLibs.fileHandler import FilePathHandler

class FileHandler:

    def __init__(self) -> None:
        pass

    def __del__(self) -> None:
        pass

    @staticmethod
    def removeFiles(filePathPattern=None):
        filePathList = glob.glob(filePathPattern, recursive=True)
        for filePath in filePathList:
            os.remove(filePath)
    
    @staticmethod
    def exists(filePath=None):
        if filePath is None:
            return False
        else:
            return os.path.isfile(filePath) or os.path.isdir(filePath)

    @staticmethod
    def makeDirectory(targetPath='.', directoryName=None):
        try:
            if os.path.exists(targetPath+'/'+directoryName)==False:
                os.mkdir(targetPath+'/'+directoryName)
        except Exception as e:
            print(e)
    
    @staticmethod
    def cloneFiles(filePathList=None,dirPathFrom=None,dirPathTo=None):
      import shutil
      for filePath in filePathList:
        destination_path = filePath.replace(FilePathHandler.getCurrentDirectoryFullPath(dirPathRelative=dirPathFrom), dirPathTo)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy(filePath, destination_path)

    @staticmethod
    def makeWorkDirectory(dirPath=None,entityType=None,name=None):
      entityId = FilePathHandler.generateEntityId(entityType=entityType,name=name)
      from pythonLibs.os import osCommands
      dirPathWork = os.path.join(dirPath, entityId)
      osCommands.make_dir(target=dirPathWork)
      return dirPathWork

