from pythonLibs.dictHandler import DictHandler

from enum import Enum

class ToolInterfaceDirectionType(Enum):
  UNDEFINED       = 0
  SERIAL_FORWARD  = 1
  SERIAL_BACKWARD = 2

class toolInterfaceBase:

  name          = None
  toolBaseList  = None
  typeDirection = None
  paramMap      = None

  def __init__(self):
    self.initialization()
  
  def initialization(self):
    self.typeDirection = ToolInterfaceDirectionType.SERIAL_FORWARD
    self.toolBaseList  = []
    self.paramMap      = {}
  
  def setName(self,name=None):
    self.name = name

  def getName(self):
    return self.name

  def addTool(self,toolBase=None):
    if isinstance(toolBase,list):
      for toolBaseEach in toolBase:
        self.toolBaseList.append(toolBaseEach)
    else:
      self.toolBaseList.append(toolBase)
  
  def getTypeDirection(self):
    return self.typeDirection

  def getToolBaseList(self):
    return self.toolBaseList
  
  def getParameterMap(self):
    return self.paramMap

  def setParameterMap(self, paramMap=None):
    self.paramMap = paramMap
  
  def getToolBack(self):
    if len(self.getToolBaseList())==2:
      if self.getTypeDirection() is ToolInterfaceDirectionType.SERIAL_FORWARD:
        return self.getToolBaseList()[0]
      elif self.getTypeDirection() is ToolInterfaceDirectionType.SERIAL_BACKWARD:
        return self.getToolBaseList()[1]

  def getToolFore(self):
    if len(self.getToolBaseList())==2:
      if self.getTypeDirection() is ToolInterfaceDirectionType.SERIAL_FORWARD:
        return self.getToolBaseList()[1]
      elif self.getTypeDirection() is ToolInterfaceDirectionType.SERIAL_BACKWARD:
        return self.getToolBaseList()[0]

  def mapParameters(self):
    toolBaseBack = self.getToolBack()
    toolBaseFore = self.getToolFore()
    DictHandler.mapDict(d1=toolBaseFore.paramDict, d2=toolBaseBack.paramDict, mapPath=self.getParameterMap())