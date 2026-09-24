import random
from natsort import natsorted
import glob
import os
from pathlib import Path

import uuid
import re
import hashlib

class FilePathHandler:

    def __init__(self) -> None:
        pass

    def __del__(self) -> None:
        pass
    
    @staticmethod
    def isValidFilepath(s: str) -> bool:
        path = Path(s)
        return path.exists() and path.is_file()
    
    @staticmethod
    def isValidDirectory(s: str) -> bool:
        path = Path(s)
        return path.exists() and path.is_dir()

    @staticmethod
    def doGlobWithSortNatural(filePath=None):
        filePathList = glob.glob(filePath)
        filePathList = FilePathHandler.doFilePathSortNatural(filePathList=filePathList)
        return filePathList

    @staticmethod
    def doFilePathSortNatural(filePathList):
        random.shuffle(filePathList)
        filePathSorted = []
        for path in natsorted(filePathList):
            filePathSorted.append(path)
        return filePathSorted

    @staticmethod
    def getExtension(filePath=None):
        if filePath is None:
            return None
        normalized_path = os.path.normpath(filePath)
        _, ext = os.path.splitext(normalized_path)
        return ext[1:].lower()  # remove the dot and convert to lowercase

    # @staticmethod
    # def getExtension(filePath=None):
    #     path, ext = os.path.splitext(filePath)
    #     return ext[1:]
    
    @staticmethod
    def replaceExtension(filePath=None,extension=None):
        extCurrent = FilePathHandler.getExtension(filePath=filePath)
        return filePath.replace('.'+extCurrent,'.'+extension)

    @staticmethod
    def getFileName(filePath=None):
        return os.path.basename(filePath)

    @staticmethod
    def getFileNameBase(filePath=None):
        return os.path.basename(filePath).split('.', 1)[0]

    @staticmethod
    def getDirectoryPath(filePath=None):
        from os.path import dirname
        return dirname(filePath)

    @staticmethod
    def getContainingDirectoryName(filePath=None,level=None):
        if level is None or level==1:
            return os.path.split(os.path.dirname(filePath))[-1]
        else:
            tmp1 = os.path.split(os.path.dirname(filePath))[-2]
            tmp2 = tmp1.split('/')
            return tmp2[-(level-1)]

    @staticmethod
    def getFilePathWithExtentionChange(filePath=None,extention=None):
        fileNameBase   = FilePathHandler.getFileNameBase(filePath=filePath)
        dirPath        = FilePathHandler.getDirectoryPath(filePath=filePath)
        if len(dirPath)==0:
            filePath       = fileNameBase + "." + extention
        else:
            filePath       = dirPath + "/" + fileNameBase + "." + extention
        return filePath
    
    @staticmethod
    def getCurrentDirectoryFullPath(fileBase=None,dirPathRelative=None,withConvertPath=True):
        if fileBase is not None:
            import pathlib
            path = pathlib.Path(fileBase).parent.resolve()
        if dirPathRelative is not None:
            path = os.path.abspath(dirPathRelative)
        
        if withConvertPath:
            return FilePathHandler.convertPath(path=path)
        else:
            return path


    @staticmethod
    def convertPath(path):
        from pathlib import Path
        import platform
        system = platform.system()
        if system == 'Linux' or system == 'Darwin':  # Darwin is macOS
            return path
        elif system == 'Windows':
            return Path(path).as_posix().replace("/", "\\")            
        else:
            raise ValueError(f"Unsupported operating system: {system}")

    @staticmethod
    def isURL(path=None):
        import validators
        return validators.url(path)
    
    @staticmethod
    def getRelativePath(path=None,basePath=None):
        if basePath is None:
            basePath = os.getcwd()
        relativePath = Path(os.path.relpath(path, basePath)).as_posix()
        return f"./{relativePath}"

    @staticmethod
    def getMacAddress():
        mac = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
        return mac.replace(':', '')

    @staticmethod
    def generateEntityId(entityType="",name=None):
        import datetime
        currentTime     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        macAddress      = FilePathHandler.getMacAddress()
        nameHash        = hashlib.md5(name.encode()).hexdigest()[:8]
        randomComponent = uuid.uuid4().hex[:8]
        return f"{entityType}_{currentTime}_{macAddress}_{nameHash}_{randomComponent}"

