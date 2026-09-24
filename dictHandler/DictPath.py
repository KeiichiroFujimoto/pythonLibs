class DictPath():

  @staticmethod
  def getKeysFromPath(path=None):
    keyList = path.split('/')
    if '' in keyList:
      keyList.remove('')
    return keyList

  @staticmethod
  def getDictItem(dict=None, path=None):
    keyList = DictPath.getKeysFromPath(path=path)
    itemCur = None
    for key in keyList:
        if itemCur is None:
            itemCur = dict[key]
        else:
            itemCur = itemCur[key]
    return itemCur

  @staticmethod
  def getDictionary(objDict,key='key'):
    dictRes = {}
    for itemDict in objDict:
      key_loc = itemDict[key]
      del itemDict[key]
      dictRes[key_loc] = itemDict
    return dictRes
