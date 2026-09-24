from pythonLibs.dateTimeHandler import DateTimeHandler
from pythonLibs.os import osCommands

class toolExecutor:

    config      = None
    dirpathBase = None
    
    withWorkDirectoryManage = True

    def __init__(self,filePathConfigToolExecutor=None):
        
        self.getCurrentDirectoryPath()

        if filePathConfigToolExecutor is not None:
            from pythonLibs.jsonHandler import JsonHandler
            self.config = JsonHandler.read(filePathJson=filePathConfigToolExecutor)
    
    def runPreprocess(self,accountName:str=None,CaseName:str=None,modeDateTimeDirectory:str="Date"):
        if self.withWorkDirectoryManage:
            workdirRelPath = self.setupWorkDirectory(withDirectoryChange=True,accountName=accountName,CaseName=CaseName,modeDateTimeDirectory=modeDateTimeDirectory)
            return workdirRelPath
        return None

    def runPostprocess(self):
        if self.withWorkDirectoryManage:
            self.changeDirectoryBase()
    
    def getIdToolRun(self):
        if self.config:
            return self.config.get('IdToolRun')

    def getCurrentDirectoryPath(self):
        self.dirpathBase = osCommands.getCurrentDirectoryPath()

    def getCaseName(self,IdToolRun=None):
        self.config['CaseName'] = 'Case_' + self.config['nameItem'] + '_' + str(IdToolRun)
        return self.config['CaseName']

    @staticmethod
    def getWorkDirectoryName(dirPathBase='./', accountName=None, CaseName=None, withDateTimeDirectory=True, modeDateTimeDirectory="Date", withMakeDirectory=False):

        dateTime    = DateTimeHandler.getDateTimeCurrent()
        dt          = DateTimeHandler.parse(dateTimeString=dateTime['JST'])

        if modeDateTimeDirectory == "Date":
            workdirName_datetime = dt.strftime("%Y_%m_%d") if withDateTimeDirectory else ''
        else:
            workdirName_datetime = dt.strftime("%Y_%m_%d_%H_%M_%S") if withDateTimeDirectory else ''
        
        workdirName_account  = accountName + '_' if accountName is not None else ''
        
        if withDateTimeDirectory == False:
            workdirName_account = workdirName_account.replace('_','')

        workdirName_case     = '_' + CaseName if CaseName is not None else ''

        if withDateTimeDirectory == False and accountName == None:
            workdirName_case = workdirName_case[1:]
        
        workdirName    = workdirName_account + workdirName_datetime  + workdirName_case
        workdirRelPath = dirPathBase + workdirName

        if withMakeDirectory:
            from pythonLibs.os import osCommands
            osCommands.make_dir(target=workdirRelPath, overwrite=True)

        return workdirRelPath
    
    @staticmethod
    def getPathWorkDirectory(filePathConfigToolExecutor='./configToolExecutor.json', accountName:str=None, CaseName:str=None):
        toolExe         = toolExecutor(filePathConfigToolExecutor=filePathConfigToolExecutor)
        dirPathWork     = toolExe.setupWorkDirectory(withDirectoryChange=False, modeDateTimeDirectory="Date", accountName=accountName, CaseName=CaseName)
        del toolExe
        return dirPathWork
    
    def setupWorkDirectory(self, withDirectoryChange:bool=False, modeDateTimeDirectory:str="Date", accountName:str=None, CaseName:str=None):
        
        # Get Unique Analysis Id
        self.IdToolRun  = self.getIdToolRun()

        if CaseName is None:
            self.CaseName   = self.getCaseName(IdToolRun=self.IdToolRun)
        else:
            self.CaseName   = CaseName

        self.config['CaseName'] = self.CaseName

        if accountName is None:
            accountName = self.config['account_name']
        
        # Work Directory Preparation
        workdirRelPath  = toolExecutor.getWorkDirectoryName(dirPathBase=self.config['directoryPathBase'],
                                                        accountName=accountName, 
                                                        CaseName=None,
                                                        withDateTimeDirectory=False,
                                                        modeDateTimeDirectory=modeDateTimeDirectory,
                                                        withMakeDirectory=True)
        
        workdirRelPath = toolExecutor.getWorkDirectoryName(dirPathBase=workdirRelPath+'/',
                                                        accountName=None,
                                                        CaseName=self.config['CaseName'],
                                                        withDateTimeDirectory=self.config['withDateTimeDirectory'],
                                                        modeDateTimeDirectory=modeDateTimeDirectory,
                                                        withMakeDirectory=True)
        
        if withDirectoryChange:
            osCommands.change_dir(destination=workdirRelPath)

        return workdirRelPath

    def changeDirectoryBase(self):
        osCommands.change_dir(destination=self.dirpathBase)
