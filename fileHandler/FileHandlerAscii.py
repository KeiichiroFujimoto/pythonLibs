from pathlib import Path
import os

class FileHandlerAscii():
    def __init__(self):
        print("This is constructor of ConvertTable")

    @staticmethod
    def addLineOnTop(filePath=None,line=None):
        file = Path(filePath)
        s = file.read_text()
        s = line + s
        file.write_text(s)

    @staticmethod
    def deleteLineOnTop(filePath=None,line=None):
        file = Path(filePath)
        s = file.read_text()
        sList   = s.split('\n')
        topLine = sList[0]
        s = s.replace(topLine+"\n","")
        file.write_text(s)

    @staticmethod
    def getExtension(filePath=None):
        path, ext = os.path.splitext(filePath)
        return ext[1:]

    @staticmethod
    def getFileNameBase(filePath=None):
        return os.path.basename(filePath).split('.', 1)[0]

    @staticmethod
    def checkKeywordInFile(filePath, keyword):
        try:
            with open(filePath, 'r', encoding='ascii') as file:
                for line in file:
                    if keyword in line:
                        return True
            return False
        except FileNotFoundError:
            print(f"The file {filePath} does not exist.")
            return False
        except UnicodeDecodeError:
            print(f"The file {filePath} is not an ASCII file.")
            return False
