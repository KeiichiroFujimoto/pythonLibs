from pythonLibs.os import osCommands
from pythonLibs.fileHandler import FilePathHandler
from pythonLibs.tool import toolBase

from pythonLibs.MethodHandler.MethodInspector import MethodInspector
from pythonLibs.MethodHandler.MethodParameterHandler import MethodParameterHandler

class MethodExecutor:
  
  @staticmethod
  def runForParameterDictBase(method=None,paramDict=None,toolName=None,moduleNameList=None):
    if method is None and toolName is not None and moduleNameList is not None:
      method = MethodInspector.getToolInstance(toolName=toolName,moduleNameList=moduleNameList)
    
    paramNameList  = MethodInspector.getParameterNameList(method=method)
    paramDictInput = MethodInspector.getParameterDefaultValuesAsDict(method=method)
    paramDictInput.update(paramDict)
    params         = MethodParameterHandler.getParameterValuesFromDict(d=paramDictInput, parameterNameList=paramNameList)
    return method(**params)
  
  @staticmethod
  def getParameterValue(method=None,p=None,nameParameter=None):
    if nameParameter in p.keys():
      parameterValue = p[nameParameter]
    else:
      parameterValue = MethodInspector.getParameterDefaultValue(method=method,nameParameter=nameParameter)
    return parameterValue

  @staticmethod
  def runForParameterDict(method=None,paramDict=None):
    if 'dirPathFileWrite' in paramDict.keys():
      osCommands.change_dir(FilePathHandler.convertPath(paramDict['dirPathFileWrite']+'/'))
    p   = toolBase.getVariablesFromInputDict(variablesDict=paramDict['input']['values'])
    res = MethodExecutor.runForParameterDictBase(method=method, paramDict=p)
    return p, res
