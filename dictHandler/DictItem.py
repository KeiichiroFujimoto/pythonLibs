from pythonLibs.dataHandler import DataItem, DataItemParser

class DictItem:
  
  type_ = None
  data_ = None

  def __init__(self,data=None,unit=None):
    self.setData(data=data)
    self.setUnit(unit=unit)
  
  def setType(self,type=None):
    self.type_ = type

  def getType(self):
    return self.type_

  def setData_(self,data=None):
    self.data_ = data
  
  def getData(self):
    if self.getType() is DataItem and len(self.data_.dataList)==1:
      return self.data_.dataList[0]
    elif self.getType() is DataItem and len(self.data_.dataList)>1:
      return self.data_.dataList
    else:
      return self.data_
  
  def getDataRaw(self):
      return self.data_
  
  def setUnit(self,unit=None):
    if self.getType() is DataItem and self.data_.getUnitName() is None:
      return self.data_.setUnit(unitName=unit)

  def getUnit(self):
    if self.getType() is DataItem:
      return self.data_.getUnit()
    else:
      return None
  
  def setData(self,data=None):
    if type(data) is int or type(data) is float:
      dataItem = DataItem()
      dataItem.setDataList(dataList=[data])
      self.setData_(data=dataItem)
      self.setType(type=DataItem)
    elif type(data) is str:
      if DataItemParser.isBool(data):
        data = DataItemParser.getValueAsBool(data)
        self.setData_(data=data)
        self.setType(type=bool)
      elif DataItemParser.isNumericalWithUnit(data):
        valueList, unitName = DataItemParser.getValueListAndUnit(data)
        dataItem = DataItem(dataList=valueList,unitName=unitName)
        self.setData_(data=dataItem)
        self.setType(type=DataItem)
      else:
        self.setData_(data=data)
        self.setType(type=str)
    elif type(data) is dict:
      self.setData_(data=data)
      self.setType(type=dict)
    else:
      self.setData_(data=data)

  def displayInfo(self):
    unit = self.getUnit()
    if unit is not None:
      print(str(self.getData())+ ' ' + '[' + str(self.getUnit()) +']')
    else:
      print(str(self.getData())+ ' ' + '[-]')
    
  def __repr__(self):
    return f"DictItem(data={self.data_}, unit={self.getUnit()})"

  def __str__(self):
    return f"{self.data_} {self.getUnit() if self.getUnit()else ''}".strip()
