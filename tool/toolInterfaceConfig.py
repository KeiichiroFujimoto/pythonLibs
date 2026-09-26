import sys, os
if not os.environ.get('NEXUS_PATH_CONFIGURED') and os.environ.get('PYTHON_PATH_PYTHONLIBS') and os.environ['PYTHON_PATH_PYTHONLIBS'] not in sys.path: sys.path.append(os.environ['PYTHON_PATH_PYTHONLIBS'])

from lxml import etree
from pythonLibs.xmlHandler import LxmlHandler
from collections import defaultdict

class toolInterfaceConfig:

  tree = None

  def __init__(self, filePathConfigXML=None):
    print('############ filePath:'+str(filePathConfigXML))
    self.tree = LxmlHandler.readXML(filePathXml=filePathConfigXML)
    self.root = self.tree

  def getDataItems(self):
      data_items = []
      for item in self.root.findall('DataItem'):
          data_item = {
              'programOptionsName': item.get('programOptionsName'),
              'description': item.get('description'),
              'defaultValue': self.convertDataType(item.get('defaultValue'), item.find('varDataType').text),
              'varDataType': self.convertVarDataType(item.find('varDataType').text),
              'varName': item.find('varName').text,
              'xPath': item.find('xPath').text
          }
          data_items.append(data_item)
      return data_items

  def getItemByVarName(self, varName):
      for item in self.root.findall('DataItem'):
          if item.find('varName').text == varName:
              return {
                  'programOptionsName': item.get('programOptionsName'),
                  'description': item.get('description'),
                  'defaultValue': self.convertDataType(item.get('defaultValue'), item.find('varDataType').text),
                  'varDataType': self.convertVarDataType(item.find('varDataType').text),
                  'varName': item.find('varName').text,
                  'xPath': item.find('xPath').text
              }
      return None

  def convertDataType(self, value, dataType):
      if dataType == 'VALUE_TYPE_STRING':
          return str(value)
      elif dataType == 'VALUE_TYPE_DOUBLE':
          return float(value)
      elif dataType == 'VALUE_TYPE_INTEGER':
          return int(value)
      elif dataType == 'VALUE_TYPE_BOOL':
          return bool(int(value))  # Assuming the value is '0' or '1'
      else:
          return value

  def convertVarDataType(self, varDataType):
      if varDataType == 'VALUE_TYPE_STRING':
          return 'string'
      elif varDataType == 'VALUE_TYPE_DOUBLE':
          return 'float'
      elif varDataType == 'VALUE_TYPE_INTEGER':
          return 'int'
      elif varDataType == 'VALUE_TYPE_BOOL':
          return 'bool'
      else:
          return varDataType

  def generateDataValues(self):
      data_values = {}
      for item in self.getDataItems():
          data_values[item['varName']] = item['defaultValue']
      return data_values

  def generateXmlTemplate(self, dataValues):
      root = etree.Element("LSLUCA")
      elements = {"LSLUCA": root}

      for item in self.getDataItems():
          path_parts = item['xPath'].split('.')
          current_element = root

          for part in path_parts[:-1]:
              if part not in elements:
                  elements[part] = etree.SubElement(current_element, part)
              current_element = elements[part]

          final_part = path_parts[-1]
          element = etree.SubElement(current_element, final_part)
          element.text = str(dataValues.get(item['varName'], item['defaultValue']))

      return etree.tostring(root, pretty_print=True, encoding='unicode')

  def generateCommandString(self, dataValues, commandPrefix=''):
      command_parts = defaultdict(list)

      for item in self.getDataItems():
          program_option = item['programOptionsName'].split(',')[0]
          value = dataValues.get(item['varName'], item['defaultValue'])
          command_parts[program_option].append(str(value))

      commandString = " ".join(f"{'--'+key} {' '.join(values)}" for key, values in command_parts.items())
      return commandPrefix + commandString
  
  
  @staticmethod
  def getDataValuesDefault(toolMap=None, toolName=None):
    toolConf      = toolInterfaceConfig(filePathConfigXML=toolMap[toolName]['filePathConfig'])
    dataValues    = toolConf.generateDataValues()
    return dataValues
  
  @staticmethod
  def getCommand(toolMap=None, toolName=None, paramDict=None):
    toolConf      = toolInterfaceConfig(filePathConfigXML=toolMap[toolName]['filePathConfig'])
    dataValues    = toolConf.generateDataValues()

    for key, value in paramDict.items():
        dataValues[key] = value
    
    
    commandPrefix = toolMap[toolName]['modulePath'] + ' opt '
    cmd           = toolConf.generateCommandString(dataValues=dataValues,commandPrefix=commandPrefix)

    del toolConf
    return cmd
