class toolMap:
  
  toolMap = None

  def __init__(self):
    self.toolMap = {}

  def addItem(self, toolName=None, modulePath=None, filePathConfig=None):
    self.toolMap[toolName] = {'modulePath':modulePath, 'filePathConfig':filePathConfig}