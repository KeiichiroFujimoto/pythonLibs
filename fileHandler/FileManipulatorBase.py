import sys,os
if not os.environ.get('NEXUS_PATH_CONFIGURED') and os.environ.get('PYTHON_PATH_PYTHONLIBS') and os.environ['PYTHON_PATH_PYTHONLIBS'] not in sys.path: sys.path.append(os.environ['PYTHON_PATH_PYTHONLIBS'])

import glob
import re
from datetime import datetime

class FileManipulatorBase:
  
  @staticmethod
  def getFilePathList(filePattern:str=None):
    filePathList = glob.glob(filePattern)
    return filePathList

  @staticmethod
  def checkDirectoryNameType(name: str) -> bool:
    # YYYY_MM_DD（e.g. 2025_08_14）
    pattern1 = r'^(\d{4})_(\d{2})_(\d{2})(?:_.+)?$'
    
    # YYYY_MM_DD_HH_MM_SS（e.g. 2025_08_14_11_13_44）
    pattern2 = r'^(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})(?:_.+)?$'
    
    if bool(re.match(pattern2, name)):
      return "YMDHMS"
    elif bool(re.match(pattern1, name)):
      return "YMD"
    else:
      return "None"

  @staticmethod
  def parse_datetime_from_dirname(directoryName: str) -> datetime | None:
    formats = [
        "%Y_%m_%d",
        "%Y_%m_%d_%H_%M_%S"
    ]
    for fmt in formats:
        try:
            # ラベルが付いている場合は除去
            base = "_".join(directoryName.split("_")[:fmt.count('%')])
            return datetime.strptime(base, fmt)
        except ValueError:
            continue
    return None

  @staticmethod
  def sort_by_datetime(directoryMap: dict) -> dict:
    return dict(sorted(directoryMap.items(), key=lambda item: item[1]['DateTime']))

  @staticmethod
  def filter_by_date_range(directoryMap: dict, start: datetime, end: datetime) -> dict:
    return {
        name: info for name, info in directoryMap.items()
        if start <= info['DateTime'] <= end
    }

  @staticmethod
  def build_entry_from_name(name: str) -> dict | None:
    dt = FileManipulatorBase.parse_datetime_from_dirname(name)
    if dt:
        return {'DateTime': dt}
    return None
  
  @staticmethod
  def generateDirecotryMap(filePattern: str = None):
    """
    Generate a mapping from directory names to metadata including parsed datetime.

    This method scans files matching the given pattern, extracts their base names
    (assumed to represent directory names), and builds a dictionary where each key
    is a directory name and the value is a dictionary containing a 'DateTime' entry.

    The datetime is parsed from the directory name using a predefined format
    (e.g., 'YYYY_MM_DD' or 'YYYY_MM_DD_HH_MM_SS').

    Parameters:
        filePattern (str, optional): A glob-style pattern to match files. If None,
                                     all files in the default scope will be scanned.

    Returns:
        dict: A dictionary mapping directory names to metadata dictionaries.
              Example:
              {
                  '2025_08_13_MarkdownHandler': {'DateTime': datetime.datetime(2025, 8, 13, 0, 0)},
                  '2025_08_14_MarkdownHandler': {'DateTime': datetime.datetime(2025, 8, 14, 0, 0)}
              }
    """
    from pythonLibs.fileHandler import FilePathHandler
    filePathList = FileManipulatorBase.getFilePathList(filePattern=filePattern)
    
    directoryMap = {}
    for filePath in filePathList:
        directoryName = FilePathHandler.getFileNameBase(filePath=filePath)
        directoryMap[directoryName] = FileManipulatorBase.build_entry_from_name(name=directoryName)
    return directoryMap

if __name__ == "__main__":
  directoryMap = FileManipulatorBase.generateDirecotryMap(filePattern='./fujimoto/*')
  directoryMap = FileManipulatorBase.sort_by_datetime(directoryMap=directoryMap)
  print(directoryMap)
