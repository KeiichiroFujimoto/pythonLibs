import base64

class FileIOHandler():
  
  @staticmethod
  def readFileBase64(filePath=None):
    with open(filePath, "rb") as f:
      content = base64.b64encode(f.read()).decode('utf-8')
    return content

  @staticmethod
  def writeFileBase64(content=None,filePath=None):
    with open(filePath, "wb") as f:
      f.write(base64.b64decode(content.encode('utf-8')))
    f.close()
  
  @staticmethod
  def isTable(fileType=None):
    return fileType == 'text/csv' or fileType == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or fileType=='application/vnd.ms-excel'

  @staticmethod
  def readFileFromBase64EncodedContent(content=None,fileType=None,returnType=None):
    from pythonLibs.tableHandler import TableHandler
    fileRead = None
    if FileIOHandler.isTable(fileType=fileType):
      import pandas as pd
      fileRead = TableHandler.readTableFromBase64EncodedContent(content=content, fileType=fileType, returnType=returnType)
    return fileRead