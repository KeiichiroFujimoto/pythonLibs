from pythonLibs.tool.toolInterfaceBase import *
from pythonLibs.tool.toolInterfaceConfig import *

class toolInterfaceSystemCall(toolInterfaceBase):
  
  def __init__(self):
    super().__init__()
  
  @staticmethod
  def execute(toolMap=None,toolName=None,paramDict=None,workDirPath=None):
    cmd = toolInterfaceConfig.getCommand(toolMap=toolMap, toolName=toolName, paramDict=paramDict)
    os.chdir(workDirPath)
    os.system(cmd)