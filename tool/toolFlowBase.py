from pythonLibs.tool import toolBase

class toolFlowBase(toolBase):
  
  toolList           = None
  interfaceList      = None
  interfaceIndexList = None

  def __init__(self):
    super().__init__()
    # Use instance state to avoid cross-run bleed.
    self.toolList = []
    self.interfaceList = []
    self.interfaceIndexList = []
  
  def setToolList(self,toolList=None,withIndexGeneration=False):
    if withIndexGeneration:
      for ii in range(len(toolList)):
        self.addInterfaceIndex(index=ii)

    self.toolList = toolList

  def getToolList(self):
    return self.toolList
  
  def addTool(self,tool=None):
    self.getToolList().append(tool)

  def getInterfaceList(self):
    return self.interfaceList

  def getInterfaceIndexList(self):
    return self.interfaceIndexList

  def addInterface(self, interface=None, addIndex=False):
      self.interfaceList.append(interface)
      if addIndex:
        self.addInterfaceIndex(index=len(self.interfaceList)-1)

  def addInterfaceIndex(self, index):
      self.interfaceIndexList.append(index)
  
  def getInterface(self, index):
      if 0 <= index < len(self.interfaceList):
          return self.interfaceList[index]
      else:
          raise IndexError("Interface index out of range")

  def removeInterface(self, interface):
      self.interfaceList.remove(interface)

  def executeFlow(self):
    for index in self.getInterfaceIndexList():
        interface = self.getInterfaceList()[index]
        #[01] Execute tool
        toolBack  = interface.getToolBack()
        toolBack.execute()
        #[02] Map Parameter
        interface.mapParameters()
        #[03] Execute tool at end of work flow
        if index == self.getInterfaceIndexList()[-1]:
            toolFore  = interface.getToolFore()
            toolFore.execute()
  
  def getToolListFromInterface(self):
    toolList = []
    for index in self.getInterfaceIndexList():
        interface = self.getInterfaceList()[index]
        toolList.append(interface.getToolBack())
        if index == self.getInterfaceIndexList()[-1]:
            toolList.append(interface.getToolFore())
    return toolList

  def getParamDict(self,toolList=None,prefixTool='tool_'):
    if toolList is None:
      toolList = self.getToolListFromInterface()
    paramDict = {}
    for iop in range(len(toolList)):
        paramDict[prefixTool+toolList[iop].getName()] = toolList[iop].paramDict.toDict()
    return paramDict
