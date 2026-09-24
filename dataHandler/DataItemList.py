import os

from pythonLibs.dataHandler.DataItem import *
import pandas as pd

class DataItemList():

    itemList  = None

    def __init__(self):
        self.initialization()

    def initialization(self):

        if self.itemList is not None:
            del self.itemList

        self.itemList = []
    
    def appendDataItem(self,dataItem):
        self.itemList.append(dataItem)
    
    def appendDataItemBaseByList(self,name,unitName,formula,dataList=None):
        numData = len(dataList)

        self.appendDataItemBase(name=name,unitName=unitName,formula=formula,numData=numData)

        for i in range(numData):
            self.itemList[-1].dataList[i] = dataList[i]

    def appendDataItemBase(self,name,unitName,formula,numData=None):
        item = DataItem()
        item.setUnit(unitName=unitName)
        item.setName(name=name)

        if unitName==DataItem.unitNameDateTime:
            item.setDateTime(isDateTime=True)

        if formula=="nan":
            item.setPhysicalValue(isPhysicalValue=True)
            item.setConvertFormula(hasConvertFormula=False)
        else:
            item.setPhysicalValue(isPhysicalValue=False)
            item.setConvertFormula(hasConvertFormula=True)
            item.setFormula(formula=formula)

        self.appendDataItem(item)

        if numData is not None:
            item.allocateDataList(numData=numData)

    @staticmethod
    def readExcelDataAsPandasDataFrame(fnameExcelData=None,readAllSheets=False,indexSheet=None):
        input_file       = pd.ExcelFile(fnameExcelData)

        input_sheet_name = input_file.sheet_names
        num_sheet        = len(input_sheet_name)

        if readAllSheets==False and indexSheet is None:
            indexSheet = 0

        if readAllSheets:
            df_list=[]
            for sheet in input_sheet_name:
                df_list.append(input_file.parse(sheet))

            df=pd.DataFrame()
            for i in range(num_sheet) :
                df=df.append(df_list[i])[df_list[0].columns.tolist()]

            df=df.dropna()
        else:
            df = input_file.parse(input_sheet_name[indexSheet])

        #print("====Display Start=====================================")
        #print(df)
        #print("====Display End  =====================================")
        return df

    def readData(self,fnameExcelData=None,readAllSheets=False,indexSheet=None):
        df = DataItemList.readExcelDataAsPandasDataFrame(fnameExcelData=fnameExcelData,readAllSheets=readAllSheets,indexSheet=indexSheet)
        for nv in range(self.getTotalItemNumbers()):
            item     = self.getItem(nv)
            itemName = item.name
            self.getItem(nv).setDataList(dataList=np.array(df[itemName]))
        del df
        self.buildSurogateModel()

    def readConfig(self,fnameExcelConfig=None):
        if os.path.splitext(fnameExcelConfig)[1]=='.xlsx':
            df      = pd.read_excel(fnameExcelConfig,header=None)
            numVars = df.shape[1]
            for nv in range(numVars):
                name     = str(df.iloc[0,nv])
                unitName = str(df.iloc[1,nv])
                if df.shape[0]==3:
                    formula  = str(df.iloc[2,nv])
                else:
                    formula = "nan"

                # Added by Keiichiro Fujimoto 2022/11/09                 
                self.appendDataItemBase(name=name,unitName=unitName,formula=formula)

            #self.displayData()

    def setConfig(self,varNameList=None,varUnitList=None,varFormulaList=None):
            numVars = len(varNameList)
            for nv in range(numVars):
                name     = varNameList[nv]
                unitName = varUnitList[nv]
                formula  = varFormulaList[nv]

                if formula == "":
                    formula  = "nan"
                
                # Added by Keiichiro Fujimoto 2022/11/09                 
                self.appendDataItemBase(name=name,unitName=unitName,formula=formula)

    def displayData(self):
        numVars = len(self.itemList)
        for nv in range(numVars):
            item = self.itemList[nv]
            print("------------------------------")
            item.displayData()

    def getTotalItemNumbers(self):
        return len(self.itemList)

    def getItem(self,index=None):
        return self.itemList[index]

    def getItemNameWithUnit(self,index=None):
        return self.itemList[index].getNameWithUnit()

    def getItemByName(self,name=None):
        index = self.getIndexByName(name=name)
        return self.itemList[index]

    def getItemAsListByName(self,name=None):
        return self.getItemByName(name=name).dataList

    def setValue(self,icol,irow,value):
        self.itemList[icol].dataList[irow] = value

    def getValue(self,icol,irow):
        return self.itemList[icol].dataList[irow]

    def getIndexByName(self,name=None):
        numVars = len(self.itemList)
        for nv in range(numVars):
            item = self.itemList[nv]
            if item.name==name:
                return nv
        return None

    def getDateTimeColumnIndexList(self):
        nvList = []
        for nv in range(len(self.itemList)):
            item = self.itemList[nv]
            if item.isDateTime():
                nvList.append(nv)
        return nvList

    def getDataAsPandasDataFrame(self,withUnitName=True):

        numVars = len(self.itemList)
        for nv in range(numVars):
            item     = self.itemList[nv]
            dataList = item.getDataList()
            dataList = dataList.copy()

            if nv==0:
                df  = pd.DataFrame(dataList,columns=[item.name])
            else:
                df  = pd.concat([df,pd.DataFrame(dataList,columns=[item.name])], axis=1)

            if withUnitName:
                varName = item.name + "[" + item.unitName + "]"
            else:
                varName = item.name

            #item.displayData()

            df = df.rename(columns={item.name: varName})

        return df

    def readDataFromExcelSheet(self,filePathExcel=None,formulaList=None):
        df = pd.read_excel(filePathExcel)
        self.generateDataFromPandasDataFrame(df=df,formulaList=formulaList)

    def generateDataFromPandasDataFrame(self,df,formulaList=None):
        self.initialization()
        numVars = df.shape[1]
        numData = df.shape[0]
        for nv in range(numVars):
            varName   = df.columns[nv]
            item_name = varName.split('[')[0]
            unit_name = varName.split('[')[1].split(']')[0]

            if formulaList==None:
                formula = 'nan'
            else:
                formula = formulaList[nv]

            self.appendDataItemBase(name=item_name,unitName=unit_name,formula=formula,numData=numData)

            for i in range(numData):
                self.itemList[nv].dataList[i] = df.iat[i,nv]

    def writeConfig(self,fnameExcelConfig=None):
        numVars = len(self.itemList)

        df = pd.DataFrame([["" for nv in range(numVars)] for i in range(3)])

        for nv in range(numVars):
            item = self.itemList[nv]
            df.iat[0,nv] = item.name
            df.iat[1,nv] = item.unit
            if item.formula!=None:
                df.iat[2,nv] = item.formula
            else:
                df.iat[2,nv] = ""

        df.to_excel(fnameExcelConfig,header=None,index=None)

    def writeData(self,fnameExcelData=None):
        numVars = len(self.itemList)
        nv      = 0
        item    = self.itemList[nv]
        numData = len(item.dataList)

        varNameList = [self.itemList[nv].name for nv in range(numVars)]

        df = pd.DataFrame([["" for nv in range(numVars)] for i in range(numData)],columns=varNameList)

        print("numVars:"+str(numVars))
        print("varNameList:"+str(varNameList))

        for nv in range(numVars):
            item = self.itemList[nv]
            for ii in range(len(item.dataList)):
                #print("ii:"+str(ii)+"-nv:"+str(nv))
                df.iat[ii,nv] = item.dataList[ii]

        df.to_excel(fnameExcelData,index=None)

    def writeDataAndConfigFromPandasDataFrame(self,df=None,fnameExcelConfig=None,fnameExcelData=None):
        self.generateDataFromPandasDataFrame(df=df)
        self.writeConfig(fnameExcelConfig=fnameExcelConfig)
        self.writeData  (fnameExcelData=fnameExcelData)
    
    def getDataSet(self,varNameList=None):
        dataSet = []
        for ivr in range(len(varNameList)):
            varName  = varNameList[ivr]
            itemName = varName.split('[')[0]
            unitName = varName.split('[')[1].split(']')[0]
            dataList = self.getItemByName(itemName).getDataListWithUnitConversion(unitName=unitName)
            dataSet.append(dataList)
        return dataSet

    def getDataSetAsPandasDataFrame(self,varNameList=None):
        dataSet = self.getDataSet(varNameList=varNameList)
        tab     = np.array(dataSet)
        df      = pd.DataFrame(np.transpose(tab))
        df.columns = varNameList
        return df

    def getDataItemNameList(self):
        dataItemNameList = []
        for n in range(self.getTotalItemNumbers()):
            dataItem = self.getItem(index=n)
            dataItemNameList.append(dataItem.getName())
        return dataItemNameList

    def getDataItemUnitList(self):
        dataItemUnitList = []
        for n in range(self.getTotalItemNumbers()):
            dataItem = self.getItem(index=n)
            dataItemUnitList.append(dataItem.getUnitName())
        return dataItemUnitList

    def displayData(self):
        for item in self.itemList:
            print('================================= '+str(item.getName())+'['+str(item.getUnit())+']')
            for ii in range(len(item.dataList)):
                print(item.dataList[ii])

if __name__ == '__main__':

    dataItemList = DataItemList()

    dataItemList.appendDataItemBase(name="V1",unitName="MPa",formula="nan",numData=100)
    dataItemList.appendDataItemBase(name="V2",unitName="K",  formula="nan",numData=100)
    dataItemList.appendDataItemBase(name="V3",unitName="km", formula="nan",numData=100)

    dataItemList.setValue(icol=0,irow=0,value=1.0)
    dataItemList.setValue(icol=0,irow=1,value=2.0)
    dataItemList.setValue(icol=0,irow=2,value=3.0)

    dataItemList.setValue(icol=1,irow=0,value=10.0)
    dataItemList.setValue(icol=1,irow=1,value=20.0)
    dataItemList.setValue(icol=1,irow=2,value=30.0)

    dataItemList.setValue(icol=2,irow=0,value=100.0)
    dataItemList.setValue(icol=2,irow=1,value=200.0)
    dataItemList.setValue(icol=2,irow=2,value=300.0)

    df = dataItemList.getDataAsPandasDataFrame(withUnitName=True)
    print(df)
    
    dataItemList.generateDataFromPandasDataFrame(df=df)
    df = dataItemList.getDataAsPandasDataFrame(withUnitName=True)
    print(df)
    
    dataItemList.writeConfig(fnameExcelConfig="config.xlsx")
    dataItemList.writeData(fnameExcelData="data.xlsx")

    del dataItemList