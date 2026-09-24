from pythonLibs.dataHandler.EngUnitConversion import *
from pythonLibs.dataHandler.UnitHandler import *
import numpy as np

class DataItem():

    isPhysicalValue_   = None
    isDateTime_        = None
    hasConvertFormula_ = None
    unitName           = None
    unit               = None
    name               = None
    formula            = None
    type               = float
    dataList           = None
    indexList          = None

    unitNameDateTime   = "DateTime"

    def __init__(self, dataList=None, unitName=None):
        self.initialization()
        if dataList is not None:
            self.setDataList(dataList=dataList)
        if unitName is not None:
            self.setUnit(unitName=unitName)

    def __del__(self):
        self.dataList = None

    def initialization(self):
        self.setPhysicalValue(isPhysicalValue=True)
        self.setConvertFormula(hasConvertFormula=False)

    def isPhysicalValue(self):
        return self.isPhysicalValue_

    def setPhysicalValue(self, isPhysicalValue):
        self.isPhysicalValue_ = isPhysicalValue

    def isDateTime(self):
        return self.isDateTime_

    def setDateTime(self, isDateTime):
        self.isDateTime_ = isDateTime

    def hasConvertFormula(self):
        return self.hasConvertFormula_

    def setConvertFormula(self, hasConvertFormula):
        self.hasConvertFormula_ = hasConvertFormula

    def setName(self,name):
        self.name = name

    def getNameWithUnit(self):
        return self.name+'[' + self.unitName + ']'

    def getName(self):
        return self.name

    def getUnit(self):
        return self.unit

    def getUnitName(self):
        return self.unitName

    def setUnit(self,unitName):
        self.unitName = unitName
        self.unit = UnitHandler.getUnit(unitName=self.unitName)
        if self.unit==DataItem.unitNameDateTime:
            self.setDateTime(isDateTime=True)
        else:
            self.setDateTime(isDateTime=False)

    def changeUnit(self,unitNameNew,withDataListConversion=True):
        if withDataListConversion and self.isPhysicalValue():
            for i in range(len(self.dataList)):
                self.dataList[i] = UnitHandler.convertUnit(value=self.getValue(i),unitNameIn=self.getUnitName(),unitNameOut=unitNameNew)
        self.setUnit(unitName=unitNameNew)

    def getDataListWithUnitConversion(self,unitName=None):
        if self.isPhysicalValue():
            dataList = [0.0 for i in range(len(self.dataList))]
            for i in range(len(self.dataList)):
                dataList[i] = UnitHandler.convertUnit(value=self.getValue(i),unitNameIn=self.getUnitName(),unitNameOut=unitName)
            return dataList
        else:
            return None

    def getType(self):
        return self.type

    def setType(self,type):
        self.type = type

    def allocateDataList(self,numData):
        if self.getType() == float:
            self.dataList = np.array( [0.0] * numData )
        elif self.getType() == int:
            self.dataList = np.array( [0] * numData )
        elif self.getType() == str:
            self.dataList = np.array( [""] * numData )

    def setDataList(self,dataList):
        self.dataList = dataList

    def setIndexList(self,indexList):
        self.indexList = indexList
    
    def getDataByIndex(self,indexKey=None):
        for ii in range(len(self.indexList)):
            if self.indexList[ii] == indexKey:
                return self.dataList[ii]
        return None

    def getDataList(self):
        if self.isPhysicalValue():
            return self.dataList
        else:
            dataList = []
            for i in range(len(self.dataList)):
                dataList.append(self.getValueWithConvertByFormula(i))
            return dataList

    def setFormula(self,formula):
        if str(formula)!="nan":
            self.formula = str(formula)
            self.setConvertFormula(hasConvertFormula=True)
            self.setPhysicalValue(isPhysicalValue=False)
        else:
            self.formula = None

    def getValue(self, i):
        if self.isPhysicalValue():
            return self.getValue_Raw(i)
        else:
            return self.getValueWithConvertByFormula(i)

    def getValue_Raw(self, i):
        return self.dataList[i]

    def getValueWithConvertByFormula(self, i):
        return self.getValueWithConvertByFormulaForSpecifiedValue(V=self.dataList[i])

    def getValueWithConvertByFormulaForSpecifiedValue(self, V):
            return eval(str(self.formula))

    def displayData(self):
        print("Data Name:"+str(self.name))
        print("Unit     :"+str(self.unit))
        print("Formula  :"+str(self.formula))
        print("isDateTime:"+str(self.isDateTime()))
        if len(self.dataList)!=0:
            for i in range(len(self.dataList)):
                print(str(self.getValue(i)))

    def asDict(self):
        d = {'name':self.name, 'unit':self.unit, 'formula':self.formula, 'isDateTime':self.isDateTime(), 'dataList':self.dataList}
        return d