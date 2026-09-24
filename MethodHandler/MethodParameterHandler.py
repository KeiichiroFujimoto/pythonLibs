class MethodParameterHandler:

  @staticmethod
  def getParameterValuesFromDict(d=None,parameterNameList=None):
    paramValueList  = {name: d[name] for name in parameterNameList if name in d}  
    return paramValueList
  
  @staticmethod
  def getParameterValuesFromClassVariable(method=None,parameterNameList=None):
    parameterValueList = {name: getattr(method, name) for name in parameterNameList if hasattr(method, name)}
    return parameterValueList
