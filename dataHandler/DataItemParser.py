import re

class DataItemParser:
  
  def __init__(self):
    pass

  @staticmethod
  def getNumberType(value):
    try:
        # Try to convert the string to an integer
        int_value = int(value)
        return int
    except ValueError:
        try:
            # Try to convert the string to a float
            float_value = float(value)
            return float
        except ValueError:
            return None
  
  @staticmethod
  def isBool(valueAsString=None):
    if valueAsString == 'True' or valueAsString == 'False' or valueAsString == 'true' or valueAsString == 'false':
      return True
    else:
      return False
  
  @staticmethod
  def getValueAsBool(valueAsString=None):
    if DataItemParser.isBool(valueAsString=valueAsString):
      if valueAsString == 'True' or valueAsString == 'true':
        return True
      elif valueAsString == 'False' or valueAsString == 'false':
        return False
    else:
      return None

  @staticmethod
  def isNumericalWithUnit(valueAsString=None):
    # Define a regular expression pattern for numerical values with unit
    pattern = r'^\d+(\.\d+)?([ ,]\d+(\.\d+)?)*\[\w+\]$'
    return bool(re.match(pattern, valueAsString))

  @staticmethod
  def getValueListAndUnit(valueAsString=None):
    # Define a regular expression pattern for numerical values with unit
    pattern = r'^(\d+(\.\d+)?([ ,]\d+(\.\d+)?)*?)\[(\w+)\]$'
    match = re.match(pattern, valueAsString)
    if match:
        values_str = match.group(1)
        unit       = match.group(5)
        valueList  = []
        for v in re.split(r'[ ,]', values_str):
          if DataItemParser.getNumberType(v) == int:
            valueList.append(int(v))
          elif DataItemParser.getNumberType(v) == float:
            valueList.append(float(v))
        return valueList, unit
    else:
        raise ValueError("String does not match the expected pattern")
