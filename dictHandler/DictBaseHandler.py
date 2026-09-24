from pythonLibs.dictHandler import DictItemEncoderToml, DictItemEncoderJson, DictHandler, DictBase

class DictBaseHandler:
  @staticmethod
  def getSummation(elemList=None, keyActive='@active', keyValue=None, unit=None, displayInfo=False):
    valueSum = 0.0
    for elem in elemList:
        if elem[keyActive]:
            elem.getItem(keyValue).getDataRaw().changeUnit(unitNameNew=unit,withDataListConversion=True)
            for data in elem.getItem(keyValue).getDataRaw().dataList:
                valueSum += data
    if displayInfo:
        print('Summation ['+unit+']:'+str(valueSum))
    return valueSum

  @staticmethod
  def writeAsToml(dictBase=None,filePathToml=None):
    import toml
    with open(filePathToml, 'w') as f:
      toml.dump(DictItemEncoderToml.encode(dictBase.toDict()), f)

  @staticmethod
  def getDictBaseFromDict(d=None):
      dictBase = DictBase()
      for key, value in d.items():
          if isinstance(value, dict):
              dictBase[key] = DictBaseHandler.getDictBaseFromDict(d=value)
          else:
              dictBase[key] = value
      return dictBase

  @staticmethod
  def readFromToml(filePathToml=None, displayInfo=False):
    paramDict   = DictHandler.readFromToml(filePathToml=filePathToml)
    paramDict   = DictBaseHandler.replace_none_with_None(d=paramDict)
    paramDict   = DictHandler.cast_strings_to_numbers(paramDict)
    dictBase    = DictBaseHandler.getDictBaseFromDict(d=paramDict)
    if displayInfo:
      print('====================================================== [After by Toml]')
      print(str(dictBase))
      print('===============================================================')
    return dictBase

  @staticmethod
  def writeAsJson(dictBase=None, filePathJson=None):
    import json
    with open(filePathJson, 'w') as f:
      json.dump(dictBase.toDict(), f, indent=2, cls=DictItemEncoderJson) 

  @staticmethod
  def replace_none_with_None(d):
    if isinstance(d, dict):
        return {k: DictBaseHandler.replace_none_with_None(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [DictBaseHandler.replace_none_with_None(i) for i in d]
    elif d == "None":
        return None
    else:
        return d  

  @staticmethod
  def readFromJson(filePathJson=None, displayInfo=False):
    paramDict   = DictHandler.readFromJson(filePathJson=filePathJson)
    paramDict   = DictBaseHandler.replace_none_with_None(d=paramDict)
    paramDict   = DictHandler.cast_strings_to_numbers(paramDict)
    dictBase    = DictBaseHandler.getDictBaseFromDict(d=paramDict)
    if displayInfo:
      print('====================================================== [After by JSON]')
      print(str(dictBase))
      print('===============================================================')
    return dictBase

  @staticmethod
  def writeAsYaml(dictBase=None, filePathYaml=None):
    d = DictBaseHandler.getDictRaw(dictBase=dictBase)
    DictHandler.writeAsYaml(d=d, filePathYaml=filePathYaml)

  @staticmethod
  def readFromYaml(filePathYaml=None, displayInfo=False):
    paramDict   = DictHandler.readFromYaml(filePathYaml=filePathYaml)
    paramDict   = DictBaseHandler.replace_none_with_None(d=paramDict)
    paramDict   = DictHandler.cast_strings_to_numbers(paramDict)
    dictBase    = DictBaseHandler.getDictBaseFromDict(d=paramDict)
    if displayInfo:
      print('====================================================== [After by YAML]')
      print(str(dictBase))
      print('===============================================================')
    return dictBase
  
  @staticmethod
  def getDictRaw(dictBase=None):
    if dictBase is None:
      return {}
    if hasattr(dictBase, "toDict"):
      return dictBase.toDict()
    import json
    json_str = json.dumps(dictBase)
    dictRaw  = json.loads(json_str)
    return dictRaw

  @staticmethod
  def writeAsXml(dictBase=None,filePathXml=None, rootElemName='root'):
    print('dictBase:'+str(dictBase))
    d = DictBaseHandler.getDictRaw(dictBase=dictBase)
    print(str(d))
    d = {rootElemName:d}
    DictHandler.writeAsXml(d=d, filePathXml=filePathXml)
  
  @staticmethod
  def readFromXml(filePathXml=None, displayInfo=False):
    paramDict_loc = DictHandler.readFromXml(filePathXml=filePathXml)
    paramDict_loc = paramDict_loc[list(paramDict_loc.keys())[0]]
    paramDict_loc = DictBaseHandler.replace_none_with_None(d=paramDict_loc)
    paramDict_loc = DictHandler.cast_strings_to_numbers(paramDict_loc)
    dictBase      = DictBaseHandler.getDictBaseFromDict(d=paramDict_loc)
    if displayInfo:
      print('====================================================== [After by XML]')
      print(str(dictBase))
      print('===============================================================')
    return dictBase

  @staticmethod
  def readFromConfig(filePathConfig=None, displayInfo=False):
    from pythonLibs.fileHandler import FilePathHandler
    ext = FilePathHandler.getExtension(filePath=filePathConfig)
    if ext == 'json':
      return DictBaseHandler.readFromJson(filePathJson=filePathConfig, displayInfo=displayInfo)
    elif ext == 'xml':
      return DictBaseHandler.readFromXml(filePathXml=filePathConfig, displayInfo=displayInfo)
    elif ext == 'toml':
      return DictBaseHandler.readFromToml(filePathToml=filePathConfig, displayInfo=displayInfo)
    elif ext == 'yaml' or ext == 'yml':
      return DictBaseHandler.readFromYaml(filePathYaml=filePathConfig, displayInfo=displayInfo)
    raise ValueError(f"Unsupported config extension: {ext}. Supported: json, toml, xml, yaml, yml")

  @staticmethod
  def writeAsConfig(dictBase=None, filePathConfig=None, rootElemName='root'):
    from pythonLibs.fileHandler import FilePathHandler
    ext = FilePathHandler.getExtension(filePath=filePathConfig)
    if ext == 'json':
      DictBaseHandler.writeAsJson(dictBase=dictBase, filePathJson=filePathConfig)
    elif ext == 'xml':
      DictBaseHandler.writeAsXml(dictBase=dictBase, filePathXml=filePathConfig, rootElemName=rootElemName)
    elif ext == 'toml':
      DictBaseHandler.writeAsToml(dictBase=dictBase, filePathToml=filePathConfig)
    elif ext == 'yaml' or ext == 'yml':
      DictBaseHandler.writeAsYaml(dictBase=dictBase, filePathYaml=filePathConfig)
    else:
      raise ValueError(f"Unsupported config extension: {ext}. Supported: json, toml, xml, yaml, yml")

  @staticmethod
  def readFromPasswordProtectedZip(filePathZip=None,filePath=None,directoryPath="./",password=None):
    from pythonLibs.archiveHandler import ArchiveHandlerZip
    from pythonLibs.os import osCommands
    
    ArchiveHandlerZip.decompressZip(filePathZip=filePathZip,directoryPath=directoryPath,password=password)
    paramBase = DictBaseHandler.readFromConfig(filePathConfig=filePath)
    
    osCommands.deleteFile(filePath=filePath)
    return paramBase
