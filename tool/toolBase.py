from pythonLibs.dictHandler import DictBase, DictBaseHandler, DictHandler

from pythonLibs.docstringHandler import DocstringHandler
from pythonLibs.fileHandler import FilePathHandler
# pythonLibs.fileWatcher は executionStart/executionEnd の作業ディレクトリ
# スナップショット用にのみ使用するため lazy import（SDK 閉包から fileWatcher
# と推移先 processHandler を外すため）。

import json
import ast
import os

from enum import Enum
class toolStatus(Enum):
    UNDEFINED        = 0
    READYFORRUN      = 1
    COMPLETED        = 2
    ERROR_DIVERGED   = 3
    ERROR_INVALID    = 4
    ERROR_TERMINATED = 5
    AWAITING_HUMAN   = 6
    PENDING_RESULT   = 7

class toolBase:
  
  nameIdentity = None
  name         = None
  status       = None
  paramDict    = None
  functionMap  = {}
  parameterMap = {}
  functionList = None

  fileStatusPrev = None
  fileStatusCur  = None

  toolExecutor   = None
  caseName       = None

  def __init__(self,instance_subcls=None,variablesDict=None,filePathConfigToolExecutor=None,caseName=None) -> None:
    self.initialization()
    if instance_subcls is not None:
      self.setFunctionAndParameterMap(instance=instance_subcls)
    
    if variablesDict is not None:
      variables = self.getVariablesFromInputDict(variablesDict=variablesDict)
      self.setParameters(variables=variables)
    
    self.constructToolExecutor(filePathConfigToolExecutor=filePathConfigToolExecutor)
    self.caseName = caseName
  
  def __del__(self):
    if self.functionList is not None:
      self.functionList.clear()

  def initialization(self):
    self.paramDict = DictBase(initial_dict={"documentResponse":{}, "executionContext": {}})
    self.status = toolStatus.UNDEFINED
    self.functionList = []
    # Avoid shared state across instances.
    self.functionMap = {}
    self.parameterMap = dict(getattr(self.__class__, 'parameterMap', {}))
  
  def constructToolExecutor(self, filePathConfigToolExecutor=None):
      self.toolExecutor = toolBase.constructToolExecutorBase(filePathConfigToolExecutor=filePathConfigToolExecutor)

  @staticmethod
  def constructToolExecutorBase(filePathConfigToolExecutor=None):
    if filePathConfigToolExecutor is not None:
      from pythonLibs.tool import toolExecutor
      toolExecutor = toolExecutor(filePathConfigToolExecutor=filePathConfigToolExecutor)
      return toolExecutor
    else:
      return None

  def runPreprocess(self, accountName:str=None, CaseName:str=None):
    if self.toolExecutor is None:
      return None
    workdir = self.toolExecutor.runPreprocess(accountName=accountName,CaseName=CaseName)
    if workdir is not None:
      self.setExecutionContext(working_directory=workdir)
    return workdir
  
  def runPostprocess(self):
    if self.toolExecutor is None:
      return
    self.toolExecutor.runPostprocess()

  def parseExecutionResultAsMd(self):
    return toolBase.parseExecutionResultAsMdBase(paramDict=self.paramDict)
  
  @staticmethod
  def parseExecutionResultAsMdBase(paramDict):
    import os
    import pandas as pd
    markdown = "# Execution Results\n\n"
    
    if 'executionResult' not in paramDict:
        return markdown + "_No execution results._\n"

    for func_name, details in paramDict['executionResult'].items():
        if details is None:
            details = {}
        markdown += f"## {func_name}\n\n"
        markdown += "| Key         | Value                                 |\n"
        markdown += "|-------------|---------------------------------------|\n"
        markdown += f"| Method      | `{func_name}`                         |\n"
        markdown += f"| Parameters  | `{details.get('args')}`               |\n"
        markdown += f"| Kwargs      | `{details.get('kwargs')}`             |\n"
        markdown += f"| Result      | `{details.get('result')}`             |\n"

        # saved_files = details.get('filePathListSaved')
        # if saved_files:
        #     links = ", ".join([f"[{file}" for file in saved_files])
        #     markdown += f"| Saved Files | {links} |\n"
        # else:
        #     markdown += "| Saved Files | `None`                                |\n"

        # markdown += "\n"

        # Embed images and render tables
        # if saved_files:
        #     for file in saved_files:
        #         if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".bmp"):
        #             print("############################### file:"+str(file))
        #             markdown += "![title](" + file + ")" + "\n\n"
        #         elif file.endswith(".csv") and os.path.exists(file):
        #             try:
        #                 df = pd.read_csv(file)
        #                 markdown += df.to_markdown(index=False) + "\n\n"
        #             except Exception as e:
        #                 markdown += f"**Error reading table**: {e}\n\n"
        #         elif (file.endswith(".xlsx") or file.endswith(".xls")) and os.path.exists(file):
        #             try:
        #                 df = pd.read_excel(file)
        #                 markdown += df.to_markdown(index=False) + "\n\n"
        #             except Exception as e:
        #                 markdown += f"**Error reading table**: {e}\n\n"

    return markdown

  @staticmethod
  def normalizeExecutionContextBase(context=None):
    if not isinstance(context, dict):
      return {}

    normalized = {}

    workdir = context.get("working_directory") or context.get("workdir")
    if workdir:
      normalized["working_directory"] = os.fspath(workdir)

    for key, aliases in {
      "input_files": ("input_files", "inputFiles"),
      "output_files": ("output_files", "outputFiles"),
      "declared_outputs": ("declared_outputs", "declaredOutputs"),
    }.items():
      values = []
      for alias in aliases:
        value = context.get(alias)
        if value is None:
          continue
        if isinstance(value, (list, tuple, set)):
          values.extend([os.fspath(item) for item in value if item is not None])
        else:
          values.append(os.fspath(value))
      if values:
        values_deduped = []
        seen = set()
        for value in values:
          if value in seen:
            continue
          seen.add(value)
          values_deduped.append(value)
        normalized[key] = values_deduped

    metadata = context.get("metadata")
    if isinstance(metadata, dict) and metadata:
      normalized["metadata"] = dict(metadata)

    return normalized

  @staticmethod
  def mergeExecutionContextBase(base_context=None, update_context=None):
    base = toolBase.normalizeExecutionContextBase(base_context)
    update = toolBase.normalizeExecutionContextBase(update_context)
    if not base:
      return dict(update)
    if not update:
      return dict(base)

    merged = dict(base)
    if update.get("working_directory"):
      merged["working_directory"] = update["working_directory"]

    for key in ("input_files", "output_files", "declared_outputs"):
      values = list(base.get(key, []))
      seen = set(values)
      for value in update.get(key, []):
        if value in seen:
          continue
        seen.add(value)
        values.append(value)
      if values:
        merged[key] = values

    metadata = dict(base.get("metadata", {}))
    metadata.update(update.get("metadata", {}))
    if metadata:
      merged["metadata"] = metadata

    return merged

  def getExecutionContext(self):
    return toolBase.normalizeExecutionContextBase(self.paramDict.get("executionContext", {}))

  def setExecutionContext(self, context=None, replace=False, **kwargs):
    update_context = {}
    if isinstance(context, dict):
      update_context.update(context)
    update_context.update(kwargs)

    normalized = toolBase.normalizeExecutionContextBase(update_context)
    if replace:
      merged = normalized
    else:
      merged = toolBase.mergeExecutionContextBase(self.getExecutionContext(), normalized)
    self.paramDict["executionContext"] = merged
    return merged

  def clearExecutionContext(self, preserve_working_directory=False):
    if preserve_working_directory:
      context = self.getExecutionContext()
      workdir = context.get("working_directory")
      self.paramDict["executionContext"] = {"working_directory": workdir} if workdir else {}
    else:
      self.paramDict["executionContext"] = {}

  def addExecutionInputFiles(self, file_paths=None):
    return self.setExecutionContext(input_files=file_paths)

  def addExecutionOutputFiles(self, file_paths=None):
    return self.setExecutionContext(output_files=file_paths)

  # paramDict = { 
  #     "documentResponse":{
  #         "titleDocument":"Analysis Report",
  #         "contents":[{"type":"title",    "params":["Introduction","1"]},
  #                     {"type":"paragraph","params":["This is paragraph1."]},
  #                     {"type":"paragraph","params":["This is paragraph2."]},
  #                     {"type":"title",    "params":["Result and Discussions","1"]},
  #                     {"type":"image",    "params":["./files/fujimoto/sts.jpg","sts"]},
  #                     {"type":"image",    "params":["./files/fujimoto/f9.jpg" ,"f9"]},
  #                     {"type":"table",    "params":["./files/fujimoto/table.xlsx","Table"]}]
  #     }
  # }

  def addDocumentResponseItem(self,key=None,value=None):
    self.paramDict["documentResponse"][key] = value
  
  def addDocumentResponseTitle(self,titleDocument=None):
    self.paramDict["documentResponse"]["titleDocument"] = titleDocument
  
  def addDocumentResponseContent(self,contentType=None,contentParams=None):
    if not("contents" in self.paramDict["documentResponse"].keys()):
      self.paramDict["documentResponse"]["contents"] = []
    self.paramDict["documentResponse"]["contents"].append({"type":contentType, "params":contentParams})
  
  def execute(self):
    print('==================================================================================')
    print('tool['+str(self.getName())+']'+' execusion [STARTED]')

    for func in self.functionList:
      func()
    
    print('tool['+str(self.getName())+']'+' execusion [FINISHED]')
    print('')
  
  def executeWithFileStatusCheck(self,function=None,args=None,withFileStatusCheck=False,displayInfo=False,dirPathWork=None,dirPathClone=None):
      if withFileStatusCheck:
          self.executionStart(dirPathWork=dirPathWork)
          if dirPathWork is not None:
              self.setExecutionContext(working_directory=os.path.abspath(dirPathWork))
      if function is None:
          raise ValueError("function is required")
      res = function(*list(args or []))
      if withFileStatusCheck:
          self.executionEnd(dirPathWork=dirPathWork,displayInfo=displayInfo)
          
          from pythonLibs.fileHandler import FileHandler
          changed = self.fileStatus.get('changed', []) if self.fileStatus else []
          new_files = self.fileStatus.get('new', []) if self.fileStatus else []
          generated = list(dict.fromkeys(list(changed) + list(new_files)))
          if generated:
              self.addExecutionOutputFiles(file_paths=generated)
          FileHandler.cloneFiles(filePathList=changed,
                                dirPathFrom=FilePathHandler.getCurrentDirectoryFullPath(dirPathRelative=dirPathWork),
                                dirPathTo=dirPathClone)
      return res

  def executionStart(self,dirPathWork=None):
      from pythonLibs.fileWatcher import FileWatcher  # lazy
      self.fileStatusPrev = FileWatcher.getLastModifiedDict_Base(path=dirPathWork)

  def executionEnd(self,dirPathWork=None,displayInfo=False):
      from pythonLibs.fileWatcher import FileWatcher  # lazy
      self.fileStatusCur  = FileWatcher.getLastModifiedDict_Base(path=dirPathWork)
      self.fileStatus     = FileWatcher.getUniqueData_Base(old_dict=self.fileStatusPrev, new_dict=self.fileStatusCur)
      
      if displayInfo:
          print(self.fileStatus)

  def addFunction(self,func=None):
    self.functionList.append(func)

  def setStatus(self, status=None):
    self.status = status

  def getStatus(self):
    return self.status

  def setName(self, name=None, withNameIdentity=True):
    if withNameIdentity and self.getNameIdentity() is not None:
      self.name = self.getNameIdentity() + '_' + name
    else:
      self.name = name

  def getName(self):
    return self.name

  def setNameIdentity(self, nameIdentity=None):
    self.nameIdentity = nameIdentity

  def getNameIdentity(self):
    return self.nameIdentity

  def setFunctionAndParameterMap(self,instance=None):
    super_methods = set(dir(instance.__class__.__bases__[0]))
    # Loop through all methods
    for method_name in dir(instance):
      method = getattr(instance, method_name)
      if callable(method) and method_name not in super_methods:
        docstring = DocstringHandler.parse(method)
        self.functionMap[method_name] = docstring.get('summary', method_name)
        for param in docstring.get('parameters', []):
          self.parameterMap[param['name']] = param['description']

  def setParameterByFileRead(self):
    for key in self.paramDict:
      if type(self.paramDict[key]) is str:
        filePath  = self.paramDict[key]
        extension = FilePathHandler.getExtension(filePath=filePath) 
        if extension == 'json':
          d=DictHandler.readFromJson(filePathJson=filePath)
          d=DictHandler.cast_strings_to_numbers(d=d)
          self.paramDict[key]  = toolBase.getDictBaseFromDict(d=d)
        elif extension == 'xml':
          d=DictHandler.readFromXml (filePathXml =filePath)
          d=DictHandler.cast_strings_to_numbers(d=d)
          self.paramDict[key]  = toolBase.getDictBaseFromDict(d=d)

  def setParameterDict(self,paramDict=None):
    if self.paramDict is not None:
      del self.paramDict
      self.paramDict = None
    self.paramDict = paramDict
  
  def setParameterDictByEval(self,namespace=None):
    toolBase.setParameterDictByEval_Base(paramDict=self.paramDict, namespace=namespace)

  @staticmethod
  def setParameterDictByEval_Base(paramDict=None, parent_key='', namespace=None):
    if namespace is None:
      namespace = globals()  # Use global namespace if none is provided

    for key, value in paramDict.items():
        if isinstance(value, dict):
            new_parent_key = f'{parent_key}.{key}' if parent_key else key
            toolBase.setParameterDictByEval_Base(value, new_parent_key, namespace=namespace)
        else:
          if type(value) is str and value.startswith('eval:'):
            cmd    = value.replace('eval:','')
            paramDict[key] = eval(cmd,namespace)

  def getDictRaw(self):
    json_str = json.dumps(self.paramDict, default=str).replace("'", '"')
    dictRaw  = json.loads(json_str)
    return dictRaw
    #return json.loads(str(self.paramDict).replace("'", '"'))

  @staticmethod
  def getDictBaseFromDict(d=None):
      dictBase = DictBase()
      for key, value in d.items():
          if isinstance(value, dict):
              dictBase[key] = toolBase.getDictBaseFromDict(d=value)
          else:
              dictBase[key] = value
      return dictBase

  def getParameterValue(self,glob=None):
    d = self.getDictRaw()
    return DictHandler.get(d,glob=glob)
  
  def searchParameter(self,glob=None):
    d = self.getDictRaw()
    return DictHandler.search(d,glob=glob)

  def readParameterFromToml(self, filePathToml=None, displayInfo=False):
    paramDict = DictBaseHandler.readFromToml(filePathToml=filePathToml, displayInfo=displayInfo)
    self.setParameterDict(paramDict=paramDict)
    # paramDict   = DictHandler.readFromToml(filePathToml=filePathToml)
    # paramDict   = DictHandler.cast_strings_to_numbers(paramDict)
    # self.setParameterDict(paramDict=toolBase.getDictBaseFromDict(d=paramDict))
    # if displayInfo:
    #   print('====================================================== [After by TOML]')
    #   print(str(self.paramDict))
    #   print('===============================================================')

  def writeParameterAsToml(self, filePathToml=None):
    DictBaseHandler.writeAsToml(dictBase=self.paramDict,filePathToml=filePathToml)
    # with open(filePathToml, 'w') as f:
    #   toml.dump(DictItemEncoderToml.encode(self.paramDict.toDict()), f)

  def readParameterFromJson(self, filePathJson=None, displayInfo=False):
    paramDict = DictBaseHandler.readFromJson(filePathJson=filePathJson, displayInfo=displayInfo)
    self.setParameterDict(paramDict=paramDict)
    # paramDict   = DictHandler.readFromJson(filePathJson=filePathJson)
    # paramDict   = DictHandler.cast_strings_to_numbers(paramDict)
    # self.setParameterDict(paramDict=toolBase.getDictBaseFromDict(d=paramDict))
    # if displayInfo:
    #   print('====================================================== [After by JSON]')
    #   print(str(self.paramDict))
    #   print('===============================================================')
  
  def writeParameterAsJson(self, filePathJson=None):
    DictBaseHandler.writeAsJson(dictBase=self.paramDict, filePathJson=filePathJson)
    # with open(filePathJson, 'w') as f:
    #   json.dump(self.paramDict.toDict(), f, indent=2, cls=DictItemEncoderJson) 
  
  def readParameterFromXml(self, filePathXml=None, displayInfo=False):
    paramDict = DictBaseHandler.readFromXml(filePathXml=filePathXml, displayInfo=displayInfo)
    self.setParameterDict(paramDict=paramDict)
    # paramDict_loc = DictHandler.readFromXml(filePathXml=filePathXml)
    # paramDict_loc = paramDict_loc[list(paramDict_loc.keys())[0]]
    # paramDict_loc = DictHandler.cast_strings_to_numbers(paramDict_loc)
    # self.setParameterDict(paramDict=toolBase.getDictBaseFromDict(d=paramDict_loc))
    # if displayInfo:
    #   print('====================================================== [After by XML]')
    #   print(str(self.paramDict))
    #   print('===============================================================')
  
  def writeParameterAsXml(self, filePathXml=None, rootElemName='root'):
    DictBaseHandler.writeAsXml(dictBase=self.paramDict, filePathXml=filePathXml, rootElemName=rootElemName)
    # d = self.getDictRaw()
    # print(str(d))
    # d = {rootElemName:d}
    # DictHandler.writeAsXml(d=d, filePathXml=filePathXml)

  @staticmethod
  def isList(value):
    if isinstance(value, list):
        return True
    try:
        # Try to evaluate the string to a Python object
        evaluated_value = ast.literal_eval(value)
        # Check if the evaluated value is a list
        return isinstance(evaluated_value, list)
    except (ValueError, SyntaxError, TypeError):
        # If evaluation fails, it's not a valid list
        return False
  
  @staticmethod
  def getVariablesFromInputDict(variablesDict=None):
    from pythonLibs.dataHandler import DataItemParser
    import numpy as np
    # Initialize an empty dictionary to store the generated variables
    variables = {}

    # Iterate over the keys in the values dictionary
    for key, value in variablesDict.items():
        if toolBase.isList(value=value):
            variables[key] = ast.literal_eval(value)
        elif isinstance(value, dict) and 'min' in value and 'max' in value and 'idiv' in value:
            # Generate a list using np.linspace for the given key
            variables[key] = np.linspace(float(value['min']), float(value['max']), int(value['idiv']))
        elif DataItemParser.isBool(valueAsString=value):
            variables[key] = DataItemParser.getValueAsBool(valueAsString=value)
        elif DataItemParser.getNumberType(value=value)==float:
            variables[key] = float(value)
        elif DataItemParser.getNumberType(value=value)==int:
            variables[key] = int(value)
        else:
            variables[key] = value
    
    return variables

  def executeParallel(self, method_name, params_list, max_workers=None):
    """同一メソッドを異なるパラメータで並列実行する。

    Args:
        method_name: 実行するメソッド名
        params_list: パラメータ辞書のリスト [{...}, {...}, ...]
        max_workers: 最大スレッド数 (None=自動設定)

    Returns:
        入力順に整列した結果リスト。失敗タスクは {"__error": str, "__traceback": str} を含む。
    """
    from pythonLibs.tool.ToolRunner import ThreadRunner, ToolTask
    import traceback as tb

    method = getattr(self, method_name, None)
    if method is None or not callable(method):
      raise AttributeError(f"method '{method_name}' not found on {self.__class__.__name__}")

    def _safe_call(idx, fn, kw):
      try:
        return {"__index": idx, "result": fn(**kw)}
      except Exception as e:
        return {"__index": idx, "__error": str(e), "__traceback": tb.format_exc()}

    tasks = [
      ToolTask(
        name=f"{method_name}_{i}",
        func=_safe_call,
        args=[i, method, params],
      )
      for i, params in enumerate(params_list)
    ]

    collector = getattr(self, "_tool_trace", None)
    runner = ThreadRunner(max_workers=max_workers)
    raw = runner.run(tasks, collector=collector)

    ordered = [None] * len(params_list)
    for item in raw:
      payload = item["result"]
      idx = payload["__index"]
      if "__error" in payload:
        ordered[idx] = {"__error": payload["__error"], "__traceback": payload["__traceback"]}
      else:
        ordered[idx] = payload["result"]

    return ordered

  def setParameters(self, variables=None):
    import inspect
    if variables is not None:
      for name, _ in inspect.getmembers(self, lambda a: not(inspect.isroutine(a))):
        if not name.startswith('__'):
          if name in variables:
            setattr(self, name, variables[name])
