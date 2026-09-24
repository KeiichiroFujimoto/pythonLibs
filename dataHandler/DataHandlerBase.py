from   datetime import datetime as dt
import datetime
class DataHandlerBase:

    def __init__(self):
        print("Constructor")
    
    def __del__(self):
        print("Destructor")

    @staticmethod
    def getTimeStart(df,labelTime,withConvertToDateTime=False):
        value         = df[labelTime].iloc[0]
        if withConvertToDateTime:
            dateTimeStart = DataHandlerBase.getDateTime(str(value))
        else:
            dateTimeStart = value
        return dateTimeStart

    @staticmethod
    def getTimeEnd  (df,labelTime,withConvertToDateTime=False):
        nmax          = len(df[labelTime])
        value         = df[labelTime].iloc[nmax-1]
        if withConvertToDateTime:
            dateTimeEnd = DataHandlerBase.getDateTime(str(value))
        else:
            dateTimeEnd = value
        return dateTimeEnd

    @staticmethod
    def getTimeDuration(df,labelTime):
        dateTimeStart = DataHandlerBase.getTimeStart(df,labelTime)
        dateTimeEnd   = DataHandlerBase.getTimeEnd  (df,labelTime)
        duration      = dateTimeEnd - dateTimeStart
        return duration

    @staticmethod
    def getTimeInSeconds(df,labelTime):
        duration     = DataHandlerBase.getTimeDuration(df=df,labelTime=labelTime)
        totalSeconds = duration.seconds
        import numpy as np
        timeList = np.linspace(0,totalSeconds,len(df[labelTime]))
        return timeList

    @staticmethod
    def getDateTime(value,replaceListFrom=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],replaceListTo=["01","02","03","04","05","06","07","08","09","10","11","12"],format='%Y-%m-%d %H:%M:%S',withTailCut=False):
        tstr = value
        imax = len(replaceListFrom)
        for i in range(imax):
            tstr       = tstr.replace(replaceListFrom[i],replaceListTo[i])
        if withTailCut:
            tstr       = tstr[0:-4]
        return dt.strptime(tstr, format)
