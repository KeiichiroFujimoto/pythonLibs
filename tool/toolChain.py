from pythonLibs.MethodHandler import MethodInspector
from pythonLibs.jsonHandler import JsonHandler

class toolChain:
  
  classList = []
  paramDict = None

  def __init__(self,filePathJson=None):
    self.classList = []
    self.paramDict = None
    if filePathJson is not None:
      self.paramDict = JsonHandler.read(filePathJson=filePathJson)
      self.initializer()
  
  def initializer(self):
    if self.paramDict is not None:
      self.setToolInstance()
      self.setParameters()

  def setToolInstance(self):
    
    for key in self.paramDict.keys():
        
        toolNameList     = self.paramDict[key]["toolNameList"]
        
        insCls           = None
        toolInstanceList = []
        for toolName in toolNameList:
            nameMethod = toolName.split('.')[-1]
            nameClass  = toolName.split('.')[-2]
            moduleName = toolName.replace(nameMethod,'').replace(nameClass,'')[0:-2]
            if insCls is None:
                insCls = MethodInspector.getClassInstanceByName(nameClass=nameClass,moduleName=moduleName)
                self.paramDict[key]["classInstance"] = insCls
                self.classList.append(insCls)
            insFnc = MethodInspector.getMethodByName(insCls=insCls, nameMethod=nameMethod)
            toolInstanceList.append(insFnc)
        
        self.paramDict[key]["toolInstanceList"] = toolInstanceList

  def setParameters(self):
    
    for key in self.paramDict.keys():
        params  = self.paramDict[key]["params"]
        insCls  = self.paramDict[key]["classInstance"]
        insCls.setName(key)

        for toolInstance in self.paramDict[key]["toolInstanceList"]:
            insCls.addFunction(toolInstance)

        for key_param in params.keys():
            insCls.paramDict[key_param] = params[key_param]
