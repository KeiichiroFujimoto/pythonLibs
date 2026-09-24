import sys
import toml
import numpy as np

class TomlHandler:

    def __init__(self) -> None:
        pass

    def __del__(self) -> None:
        pass

    @staticmethod
    def displayContents(filePathToml):
        with open(filePathToml) as f:
            obj = toml.load(f)
            print(obj)
            data = toml.dumps(obj) 
            print(data)

    @staticmethod
    def read(filePathToml=None):
        return TomlHandler.getTomlObject(filePathToml=filePathToml) 

    @staticmethod
    def getTomlObject(filePathToml=None):
        if not filePathToml:
            print("Error: No file path provided.")
            return None
        try:
            with open(filePathToml, 'r') as f:
                obj = toml.load(f)
                return obj
        except Exception as e:
            print(f"Error reading TOML file: {e}")
            return None
    
    @staticmethod
    def getNewTomlForApp(appName=None,version=None):
        tomlDict = { 'app': appName, 'version':version }
        return tomlDict

    @staticmethod
    def addNewItem(obj=None,key=None,value=None):
        obj[key] = value
    
    @staticmethod
    def write(obj=None,filePathToml=None):
        TomlHandler.save(obj=obj,filePathToml=filePathToml)

    @staticmethod
    def save(obj=None,filePathToml=None):
        with open(filePathToml, 'w') as f:
            f.write(toml.dumps(obj))

    @staticmethod
    def storeAsValue(obj=None,keyTarget=None):
        for key in list(obj[keyTarget].keys()):
            exec(key + "=obj['" + keyTarget + "']['" + key + "']")

    @staticmethod
    def evaluateAndGetValue(formula=None):
        value = eval(str(formula))
        return value

    @staticmethod
    def generateInputFilesFromInputTable(filePathToml=None,df=None):

        objToml  = TomlHandler.getTomlObject(filePathToml=filePathToml)

        colNames = df.columns

        for irow in range(df.shape[0]):
            filePathToml = df.iloc[irow,0]
            for icol in range(1,df.shape[1]):
                colName = colNames[icol]
                groupName, varName = colName.split("/")
                dict               = objToml[groupName]

                value = df.iloc[irow,icol]

                if type(value)==np.int64 or type(value)==np.int32:
                    dict[varName] = int(value)
                elif type(value)==np.float64 or type(value)==np.float32:
                    dict[varName] = float(value)
                else:
                    dict[varName] = value

            TomlHandler.save(objToml,filePathToml=filePathToml)
    
    @staticmethod
    def decodeUnicodeEscape(string):
        string = bytes(string, 'utf-8')
        return string.decode('unicode-escape')

    @staticmethod
    def setMatrix(tomlObj=None,matrix=None,name=None,nameRow='row'):
        numRow = matrix.shape[0]
        data   = []
        for irow in range(numRow):
            row = matrix[irow,:]
            data.append({nameRow:row})
        tomlObj[name] = data

    @staticmethod
    def getMatrixAsNumpyArray(filePathToml=None,name=None,nameRow='row',dtype=np.int32):
        with open(filePathToml, 'r') as f:
            data = toml.load(f)

        matrix    = [entry[nameRow] for entry in data[name]]
        numRow    = len(matrix)
        numCol    = len(matrix[0])
        matrix_np = np.zeros((numRow,numCol),dtype=dtype)

        for icol in range(numCol):
            for irow in range(numRow):
                matrix_np[irow,icol]=int(matrix[irow][icol])

        return matrix_np

    @staticmethod
    def getListAsNumpyArray(filePathToml=None,name=None,dtype=np.int32):
        objToml  = TomlHandler.getTomlObject(filePathToml=filePathToml)
        numItem  = len(objToml[name])
        itemList = np.zeros((numItem),dtype=dtype)
        if dtype is np.int32:
            for ii in range(numItem):
                itemList[ii] = int(objToml[name][ii])
        return itemList

    @staticmethod
    def getValue(objToml=None,filePathToml=None,keyPath=None):
        if objToml is None:
            objToml    = TomlHandler.getTomlObject(filePathToml=filePathToml)

        keyList = keyPath.split('/')
        objCur = objToml
        for key in keyList:
            if key in objCur.keys():
                objCur = objCur[key]
            else:
                return None
        
        if filePathToml is not None:
            del objToml

        return objCur

    @staticmethod
    def getDictionary(objDict,key='key'):
        dictRes = {}
        for itemDict in objDict:
            key_loc = itemDict[key]
            del itemDict[key]
            dictRes[key_loc] = itemDict
        return dictRes

if __name__ == '__main__':
    import numpy as np

    import toml.encoder
    with open('write_sample.toml', 'w') as f:
        tomlDict = TomlHandler.getNewTomlForApp(appName="AutoScience",version="1.0")
        tomlDict['table'] = { 'array': np.zeros((3,3)).tolist() }
        print("===============================")
        print(tomlDict['table']['array'])

    TomlHandler.save(tomlDict,filePathToml="write_sample.toml")

    filePathToml = sys.argv[1]

    #[01] Display All Contents of TOML
    TomlHandler.displayContents(filePathToml=filePathToml)

    #[02] Get Object of TOML
    obj = TomlHandler.getTomlObject(filePathToml=filePathToml)

    #[03] Add New Item to TOML Object
    TomlHandler.addNewItem(obj=obj,key='new',value='HOGE')

    #[04] Do something by using TOML
    if obj['app']=='AutoScience':
        print("This Toms data is for AutoScience")

    if obj['configuration']['type']=='log':
        print("This Toms data is for Log")
    
    print("===============================")
    print(obj['table']['array'])

    import numpy as np
    tab = np.array(obj['table']['array'])
    print(tab)

    #[05] Save File
    TomlHandler.save(obj,filePathToml="output.toml")
