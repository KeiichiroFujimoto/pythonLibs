import numpy as np
from pythonLibs.dataHandler.DataItemList import *

class DataItemListTimeSequence(DataItemList):

    smList = None

    def buildSurogateModel(self):
        from pythonLibs.regressionHandler import RegressionHandler1D_SmoothSpline
        timeList    = self.getItem(index=0).dataList
        self.smList = []
        mdl = RegressionHandler1D_SmoothSpline()
        for i in range(self.getTotalItemNumbers()-1):
            yList   = self.getItem(index=i+1).dataList
            sm      = mdl.buildModel(x=timeList,y=yList,params=[0])
            print('>>>>>>>>> type(sm):'+str(type(sm)))
            self.smList.append(sm)
        del mdl

    def predictItemValues(self,timeList=None, allowExtrapolation=False):
        from pythonLibs.regressionHandler import RegressionHandler1D_SmoothSpline

        if allowExtrapolation or (self.getItem(index=0).dataList[0] <= timeList[0] and timeList[-1] <= self.getItem(index=0).dataList[-1]):
            mdl = RegressionHandler1D_SmoothSpline()
            for i in range(self.getTotalItemNumbers()-1):
                mdl.regModel = self.smList[i]
                ypred   = mdl.predict(np.array(timeList))
                if i==0:
                    tab = ypred
                else:
                    tab = np.vstack((tab, ypred))

            return tab.transpose()
            
            del mdl
        
        else:
            return None
