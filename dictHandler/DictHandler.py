from pythonLibs.dataHandler import DataItemParser

from pythonLibs.tomlHandler import TomlHandler
from pythonLibs.jsonHandler import JsonHandler
from pythonLibs.yamlHandler import YamlHandler
from pythonLibs.xmlHandler import LxmlHandler
from pythonLibs.fileHandler import FilePathHandler, FileIOHandler

class DictHandler():
  
  @staticmethod
  def getAsXmlString(string=None):
    formatData = DictHandler.getFormatOfString(string=string)
    xmlString  = None
    if formatData == "stringXml":
        xmlString = string
    elif formatData == "stringJson":
        dictJson   = JsonHandler.getDictFromString(jsonStr=string)
        xmlString = DictHandler.getXMLFromDict(d=dictJson)
    return xmlString
  
  @staticmethod
  def getFormatOfString(string=None):
    import json
    import xml.etree.ElementTree as ET
    try:
        # Try parsing as JSON
        parsed_json = json.loads(string)
        if isinstance(parsed_json, dict):
            return "stringJson"
    except json.JSONDecodeError:
        pass
    
    try:
        # Try parsing as XML
        ET.fromstring(string)
        return "stringXml"
    except ET.ParseError:
        pass
    
    return "Unknown format"

  @staticmethod
  def getDictFromXML(xmlString=None):
    import xmltodict
    return xmltodict.parse(xmlString)

  @staticmethod
  def getXMLFromDict(d=None, rootName="root"):
    from dicttoxml import dicttoxml
    return dicttoxml(d, custom_root=rootName).decode()
      
  @staticmethod
  def get(d=None,glob=None):
    import dpath.util
    try:
      elemList = dpath.util.get(d,glob)
      return elemList
      # if type(elemList) is list:
      #   return elemList
      # elif type(elemList) is dict:
      #   return [elemList]
      # else:
      #   return elemList
    except Exception as e:
      print(f"Error: {e}")    
      return None
  
  @staticmethod
  def generateItem(d=None,glob=None,value=None):
    import dpath.util
    dpath.util.new(obj=d,path=glob,value=value)

  @staticmethod
  def search(d,glob):
    import dpath.util
    return dpath.util.search(d,glob)

  @staticmethod
  def getDictFromString(dictAsString=None):
    try:
      return eval(dictAsString)
    except Exception as e:
      print(f"An error occurred: {e}")
  
  @staticmethod
  def getItemAsList(item=None):
    itemAsList = None
    if isinstance(item,dict):
        itemAsList = [item]
    else:
        itemAsList = item
    return itemAsList

  @staticmethod
  def getArgsFromDict(argsDict=None,separator='&',withParentheses=True):
    args = ""
    for ii in range(len(argsDict.keys())):
      key  = list(argsDict.keys())[ii]
      if ii>0:
        args += separator
      if withParentheses:
        args += key + "=" + "\"" + str(argsDict[key]) + "\""
      else:
        args += key + "=" + str(argsDict[key])
    return args

  @staticmethod
  def flatten_dict(d, parent_key='', result={}):
    for key, value in d.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
          DictHandler.flatten_dict(value, full_key, result)
        else:
          result[full_key] = value
    return result

  @staticmethod
  def unflatten_dict(d):
    result = {}
    for key, value in d.items():
        keys = key.split('.')
        temp = result
        for k in keys[:-1]:
          temp = temp.setdefault(k, {})
        temp[keys[-1]] = value
    return result
  
  @staticmethod
  def readFromToml(filePathToml=None):
    return TomlHandler.read(filePathToml=filePathToml)

  @staticmethod
  def writeAsToml(d, filePathToml=None):
    TomlHandler.write(obj=d, filePathToml=filePathToml)

  @staticmethod
  def readFromJson(filePathJson=None):
    return JsonHandler.read(filePathJson=filePathJson)

  @staticmethod
  def writeAsJson(d, filePathJson=None):
    JsonHandler.write(dict_out=d,filePathJson=filePathJson)

  @staticmethod
  def readFromXml(filePathXml=None):
    tree      = LxmlHandler.readXML(filePathXml=filePathXml)
    d         = LxmlHandler.getXmlAsDictionary(tree=tree)
    del tree
    return d

  @staticmethod
  def writeAsXml(d, filePathXml=None):
    tree = LxmlHandler.getXmlFromDictionary(xmlDict=d)
    LxmlHandler.writeToXML(tree=tree,filePathXml=filePathXml)
    del tree

  @staticmethod
  def readFromYaml(filePathYaml=None):
    return YamlHandler.read(filePathYaml=filePathYaml)

  @staticmethod
  def writeAsYaml(d, filePathYaml=None):
    YamlHandler.write(dict_out=d, filePathYaml=filePathYaml)

  @staticmethod
  def readFromConfig(filePathConfig=None):
    ext = FilePathHandler.getExtension(filePath=filePathConfig)
    if ext == 'json':
      return DictHandler.readFromJson(filePathJson=filePathConfig)
    elif ext == 'toml':
      return DictHandler.readFromToml(filePathToml=filePathConfig)
    elif ext == 'xml':
      d = DictHandler.readFromXml(filePathXml=filePathConfig)
      if isinstance(d, dict) and len(d) == 1 and 'root' in d:
        return d['root']
      return d
    elif ext == 'yaml' or ext == 'yml':
      return DictHandler.readFromYaml(filePathYaml=filePathConfig)
    raise ValueError(f"Unsupported config extension: {ext}. Supported: json, toml, xml, yaml, yml")

  @staticmethod
  def writeAsConfig(d=None, filePathConfig=None):
    ext = FilePathHandler.getExtension(filePath=filePathConfig)
    if ext == 'json':
      DictHandler.writeAsJson(d=d, filePathJson=filePathConfig)
    elif ext == 'toml':
      DictHandler.writeAsToml(d=d, filePathToml=filePathConfig)
    elif ext == 'xml':
      if isinstance(d, dict) and len(d) == 1 and 'root' in d:
        xml_dict = d
      else:
        xml_dict = {'root': d}
      DictHandler.writeAsXml(d=xml_dict, filePathXml=filePathConfig)
    elif ext == 'yaml' or ext == 'yml':
      DictHandler.writeAsYaml(d=d, filePathYaml=filePathConfig)
    else:
      raise ValueError(f"Unsupported config extension: {ext}. Supported: json, toml, xml, yaml, yml")
  
  @staticmethod
  def cast_to_number(value):
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        return value

  @staticmethod
  def cast_strings_to_numbers(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = DictHandler.cast_strings_to_numbers(value)
        elif isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, str):
                    new_list.append(DictHandler.cast_to_number(item))
                elif isinstance(item, dict):
                    new_list.append(DictHandler.cast_strings_to_numbers(item))
                else:
                    new_list.append(item)
            d[key] = new_list            
        elif isinstance(value, str):
            if DataItemParser.isNumericalWithUnit(value):
              valueList, unitName = DataItemParser.getValueListAndUnit(value)
              #from pythonLibs.dictHandler import DictItem
              #d[key] = DictItem(data=valueList, unit=unitName)
              from pythonLibs.dataHandler import DataItem
              d[key] = DataItem(dataList=valueList,unitName=unitName)
            else:
              d[key] = DictHandler.cast_to_number(value)
    
    return d

  @staticmethod
  def mapDict(d1=None, d2=None, mapPath=None):
    for path1 in mapPath.keys():
        path2 = mapPath[path1]
        value = DictHandler.get(d=d2, glob=path2)
        # if isinstance(value, DictItem.DictItem):
        #   print('DictItem.DictItem was found type:'+str(type(value.getData())))
        #   if isinstance(value.getData(), str):
        #     value = value.getData()
        DictHandler.generateItem(d=d1, glob=path1, value=value)

  @staticmethod
  def mapDictByToml(d=None, filePathToml=None, mapPath=None):
    for path1 in mapPath.keys():
        path2 = mapPath[path1]
        DictHandler.generateItem(d=d, glob=path1, value=TomlHandler.getValue(filePathToml=filePathToml,keyPath=path2) )

  @staticmethod
  def updateTable(paramDict=None,pathTable='tab',filePathTable='inputTable.csv'):
    import pandas as pd
    tabDict = DictHandler.get(d=paramDict,glob=pathTable+'/content')
    if isinstance(tabDict,dict):
      df  = pd.DataFrame(tabDict)
      ext = FilePathHandler.getExtension(filePath=filePathTable)
      if ext == 'csv':
        df.to_csv(filePathTable,index=None)
      elif ext == 'xls' or 'xlsx':
        df.to_excel(filePathTable,index=None)
      
      DictHandler.generateItem(d=paramDict,glob=pathTable+'/content',value=filePathTable)
      DictHandler.generateItem(d=paramDict,glob=pathTable+'/type',   value='table-'+ext)
  
  @staticmethod
  def convertDictTableAsPandasDataframe(dictTab=None):
    import pandas as pd
    df = pd.DataFrame(dictTab)
    df.index = df.index.str.replace('item', '')
    return df
  
  @staticmethod
  def generateFileItem(d=None,glob=None,fileType=None,filePath=None,content=None):
    if filePath is not None:
      itemFile = { '@type':fileType, '@filePath':filePath }
    elif content is not None:
      itemFile = { '@type':fileType, '@content' :content }
    DictHandler.generateItem(d=d,glob=glob+'/file',value=itemFile)

  @staticmethod
  def getFile(d=None,glob=None):
    item     = DictHandler.get(d=d,glob=glob)

    if '@content' in item.keys():
      content  = item['@content']
    else:
      content  = None
  
    if '@type' in item.keys():
      fileType = item['@type']
    else:
      fileType = None

    if '@path' in item.keys():
      filePath = item['@path']
    else:
      filePath = None
  
    return fileType, filePath, content 

  @staticmethod
  def getFileFromDictFileContent(d=None,glob=None,returnType=None):
    fileType, filePath, content = DictHandler.getFile(d=d, glob=glob)
    file_ = FileIOHandler.readFileFromBase64EncodedContent(content=content, fileType=fileType, returnType=returnType)
    return file_
