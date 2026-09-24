import pandas as pd

from pythonLibs.dataHandler.DataItemList import *
from pythonLibs.tableHandler import *

class DataHandler:
    fnameExcelConfig = None
    fnameExcelData   = None
    itemList         = None
    dfList           = None

    def __init__(self) -> None:
        pass
    
    def __del__(self) -> None:
        pass
    
    def constructDataItemList(self):
        if self.itemList==None:        
            self.itemList = DataItemList()
        else:
            del self.itemList
            self.itemList = None
            self.itemList = DataItemList()

    def readDataItemListConfig(self,fnameExcelConfig):
        self.constructDataItemList()
        self.itemList.readConfig(fnameExcelConfig=fnameExcelConfig)
        self.fnameExcelConfig = fnameExcelConfig

    def setupConfigByDataFileHeader(self,fnameExcelData=None):
        self.constructDataItemList()
        colNameList    = TableHandler.getHeaderAsColumnNameList(fnameExcelData)
        varNameList    = TableHandler.getNameListFromColumnNameList(columnNameList=colNameList)
        varUnitList    = TableHandler.getUnitListFromColumnNameList(columnNameList=colNameList)
        varFormulaList = ["" for nv in range(len(varNameList))]
        self.itemList.setConfig(varNameList=varNameList,varUnitList=varUnitList,varFormulaList=varFormulaList)

    def readDataContents(self,fnameExcelData):
        df = pd.read_excel(fnameExcelData)

        for nv in range(len(df.columns)):
            colName = df.columns[nv]
            if "[" in colName or "]" in colName:
                colNameNew = TableHandler.getNameFromColumnName(columnName=colName)
                df.rename(columns={colName: colNameNew}, inplace=True)

        for nv in range(self.itemList.getTotalItemNumbers()):
            item     = self.itemList.getItem(nv)
            itemName = item.name

            self.itemList.getItem(nv).setDataList(dataList=df[itemName])

    def generateDataFromPandasDataFrame(self,df,formulaList=None):
        self.itemList.generateDataFromPandasDataFrame(df=df,formulaList=formulaList)
    
    def write(self,fnameExcelConfig,fnameExcelData):
        self.itemList.writeConfig(fnameExcelConfig=fnameExcelConfig)
        self.itemList.writeData(fnameExcelData=fnameExcelData)

    def getItemList(self):
        return self.itemList

