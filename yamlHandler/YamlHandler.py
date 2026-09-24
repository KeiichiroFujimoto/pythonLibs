import sys, os
if not os.environ.get('NEXUS_PATH_CONFIGURED') and os.environ.get('PYTHON_PATH_PYTHONLIBS') and os.environ['PYTHON_PATH_PYTHONLIBS'] not in sys.path: sys.path.append(os.environ['PYTHON_PATH_PYTHONLIBS'])

from pythonLibs.jsonHandler import JsonHandler

import json
import yaml

class YamlHandler:
    
  @staticmethod
  def convertJsonToYamlData(dictJson=None):
    dataYaml = yaml.safe_dump(dictJson, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return dataYaml
  
  @staticmethod
  def convertYamlDataToJson(dataYaml=None):
    dictYaml = yaml.safe_load(dataYaml)
    dictJson = json.loads(json.dumps(dictYaml))
    return dictJson

  @staticmethod
  def read(filePathYaml=None, encoding='utf-8'):
    with open(filePathYaml, 'r', encoding=encoding) as yf:
      dictYaml = yaml.safe_load(yf)
    return dictYaml

  @staticmethod
  def write(dict_out=None, filePathYaml=None, encoding='utf-8'):
    with open(filePathYaml, 'w', encoding=encoding) as yf:
      yaml.safe_dump(dict_out, yf, default_flow_style=False, sort_keys=False, allow_unicode=True)
  
  @staticmethod
  def readYamlData(filePathYaml=None, filePathJson=None):
    if filePathYaml is None and filePathJson is not None:
      dictJson = JsonHandler.read(filePathJson=filePathJson)
      dataYaml = YamlHandler.convertJsonToYamlData(dictJson=dictJson)
    else:
      dictYaml = YamlHandler.read(filePathYaml=filePathYaml)
      dataYaml = YamlHandler.convertJsonToYamlData(dictJson=dictYaml)
      return dataYaml
    return dataYaml
  
  @staticmethod
  def writeYaml(dataYaml=None, dictJson=None, filePathYaml=None):
    if dataYaml is None and dictJson is not None:
      YamlHandler.write(dict_out=dictJson, filePathYaml=filePathYaml)
      return
    with open(filePathYaml, 'w', encoding='utf-8') as yf:
      yf.write(dataYaml)
