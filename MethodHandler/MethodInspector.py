import sys, os
if not os.environ.get('NEXUS_PATH_CONFIGURED') and os.environ.get('PYTHON_PATH_PYTHONLIBS') and os.environ['PYTHON_PATH_PYTHONLIBS'] not in sys.path: sys.path.append(os.environ['PYTHON_PATH_PYTHONLIBS'])

import inspect
from typing import Any

import importlib
import importlib.util 

class MethodInspector:
    
    @staticmethod
    def getClassNameByInstance(instance=None):
        return instance.__class__.__name__
    
    @staticmethod
    def getParameterNameList(method=None):
        paramNameList = inspect.signature(method).parameters.keys()
        return paramNameList

    @staticmethod
    def getParameterFilePath(paramNameList=None,paramNamePrefix='filePath'):
        paramNameListFilePath = []
        for paramName in paramNameList:
            if paramName.startswith(paramNamePrefix):
                paramNameListFilePath.append(paramName)
        return paramNameListFilePath

    @staticmethod
    def getParameterDefaultValue(method=None,nameParameter=None):
        # Get the function signature
        sig = inspect.signature(method)
        # Access the parameters and their default values
        for name, param in sig.parameters.items():
            if name == nameParameter:
                if param.default is not inspect.Parameter.empty:
                    return param.default
                else:
                    return None

    @staticmethod
    def getParameterDefaultValuesAsDict(method=None):
        # Get the function signature
        sig = inspect.signature(method)
        # Access the parameters and their default values
        defaultValues = {}
        for name, param in sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                defaultValues[name] = param.default
            else:
                defaultValues[name] = None
        return defaultValues

    @staticmethod
    def extractMethods(toolInstance: Any, excludeSuperclassMethods=True):
        methods = {}
        # Get instance methods
        for name, method in toolInstance.__dict__.items():
            #print('ExtractMethods-------- name1:'+str(name))
            if callable(method) and not name.startswith('_'):
                params = {}
                sig = inspect.signature(method)
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue  # Skip 'self' parameter
                    param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
                    default = param.default if param.default != inspect.Parameter.empty else ...
                    params[param_name] = (param_type, default)
                methods[name] = (params, False)

        # Get class methods
        for name, method in inspect.getmembers(toolInstance.__class__, predicate=inspect.isfunction):
            # print('ExtractMethods-------- name2:'+str(name))
            if name.startswith("__") or name.startswith('_'):
                continue  # Skip special methods
            if name in methods:
                continue  # Skip if already added as instance method

            if excludeSuperclassMethods==False or ( excludeSuperclassMethods==True and method.__qualname__.split('.')[0] == toolInstance.__class__.__name__):
                
                params = {}
                sig = inspect.signature(method)
                for param_name, param in sig.parameters.items():
                    param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
                    default = param.default if param.default != inspect.Parameter.empty else ...
                    params[param_name] = (param_type, default)
                methods[name] = (params, True)
        
        # print(f"Extracted methods: {methods}")  # Debugging line
        return methods

    @staticmethod
    def generateMapFunctions(toolInstanceList=None,withClassName=True,excludeSuperclassMethods=True):
        mapFunctions = {}
        for toolInstance in toolInstanceList:
            className = toolInstance.__class__.__name__
            methods   = MethodInspector.extractMethods(toolInstance=toolInstance,excludeSuperclassMethods=excludeSuperclassMethods)
            for methodName, params in methods.items():
                method = getattr(toolInstance, methodName)
                if withClassName:
                    mapFunctions[f"{className}.{methodName}"] = method
                else:
                    mapFunctions[f"{methodName}"] = method

        # print(f"Map functions: {mapFunctions}")  # Debugging line
        return mapFunctions    

    @staticmethod
    def getClassInstanceByNameList(nameClass=None, moduleNameList=None, *args, **kwargs):
        if not moduleNameList:
             return None

        insCls = None
        for moduleName in moduleNameList:
            try:
                insCls = MethodInspector.getClassInstanceByName(nameClass=nameClass, moduleName=moduleName, *args, **kwargs)
                if insCls is not None:
                    return insCls
            except Exception as e:
                pass
        
        return None
    
    @staticmethod
    def getClassInstanceByName(nameClass=None, moduleName=None, *args, **kwargs):
        if moduleName:
            module = importlib.import_module(moduleName)
            cls    = getattr(module, nameClass, None)
            if cls:
                return cls(*args, **kwargs)
            else:
                return None
        else:
            cls = globals().get(nameClass)
            if not cls:
                raise ValueError(f"Class '{nameClass}' not found")
            return cls(*args, **kwargs)

    @staticmethod
    def getMethodByName(insCls, nameMethod=None):
        if insCls is None:
             raise AttributeError(f"Instance is None, cannot get method '{nameMethod}'")

        method = getattr(insCls, nameMethod, None)
        if callable(method):
            return method
        else:
            raise AttributeError(f"Method '{nameMethod}' not found")
    
    @staticmethod
    def getToolInstance(toolName=None,moduleNameList=None):
        nameMethod = toolName.split('.')[-1]
        nameClass  = toolName.split('.')[-2]
        
        insCls     = MethodInspector.getClassInstanceByNameList(nameClass=nameClass,moduleNameList=moduleNameList)
        
        insTool    = MethodInspector.getMethodByName(insCls=insCls, nameMethod=nameMethod)
        
        return insTool

    @staticmethod
    def getClassNameListOfPackageSet(dirPathPackageSetRoot=None):
        dirPathPackageSetRoot = dirPathPackageSetRoot.replace('//','/')
        from pythonLibs.fileHandler import FilePathHandler
        dirPathList = FilePathHandler.doGlobWithSortNatural(os.path.join(dirPathPackageSetRoot, '*'))
        classNameList = []
        rootPackage = os.path.basename(dirPathPackageSetRoot)
        for dirPath in dirPathList:
            subPackage = os.path.basename(dirPath)
            for cls in MethodInspector.getClassListInDirectory(dirPath, package_root=dirPathPackageSetRoot):
                #classNameList.append(f"{rootPackage}.{subPackage}.{cls.__module__}")
                classNameList.append(f"{cls.__module__}")
        
        return classNameList

    @staticmethod
    def getClassMapOfPackageSet(dirPathPackageSetRoot=None):
        from pythonLibs.fileHandler import FilePathHandler
        dirPathList   = FilePathHandler.doGlobWithSortNatural(os.path.join(dirPathPackageSetRoot, '*'))
        classMap      = {}
        rootPackage   = os.path.basename(dirPathPackageSetRoot)
        for dirPath in dirPathList:
            subPackage = os.path.basename(dirPath)
            for cls in MethodInspector.getClassListInDirectory(dirPath, package_root=dirPathPackageSetRoot):
                className = cls.__module__
                classMap[className] = cls
        return classMap

    @staticmethod
    def getMethodNameListOfPackageSet(dirPathPackageSetRoot=None):
        from pythonLibs.fileHandler import FilePathHandler
        dirPathList = FilePathHandler.doGlobWithSortNatural(os.path.join(dirPathPackageSetRoot, '*'))
        methodNameList = []
        rootPackage = os.path.basename(dirPathPackageSetRoot)
        for dirPath in dirPathList:
            subPackage = os.path.basename(dirPath)
            for cls in MethodInspector.getClassListInDirectory(dirPath):
                for mod in MethodInspector.getMethodInstanceList(cls=cls):
                    methodNameList.append(f"{rootPackage}.{subPackage}.{cls.__module__}.{mod}")
        return methodNameList

    # @staticmethod
    # def getClassNameListOfPackageSet(dirPathPackageSetRoot=None):
    #     from pythonLibs.fileHandler import FilePathHandler
    #     dirPathList    = FilePathHandler.doGlobWithSortNatural(dirPathPackageSetRoot+'/*')
    #     classNameList  = []
    #     for dirPath in dirPathList:
    #         for cls in MethodInspector.getClassListInDirectory(dirPath):
    #             classNameList.append(dirPathPackageSetRoot.split('/')[-1]+'.'+dirPath.split('/')[-1]+'.'+cls.__module__)
    #     return classNameList

    # @staticmethod
    # def getMethodNameListOfPackageSet(dirPathPackageSetRoot=None):
    #     from pythonLibs.fileHandler import FilePathHandler
    #     dirPathList    = FilePathHandler.doGlobWithSortNatural(dirPathPackageSetRoot+'/*')
    #     methodNameList = []
    #     for dirPath in dirPathList:
    #         for cls in MethodInspector.getClassListInDirectory(dirPath):
    #             for mod in MethodInspector.getMethodInstanceList(cls=cls):
    #                 methodNameList.append(dirPathPackageSetRoot.split('/')[-1]+'.'+dirPath.split('/')[-1]+'.'+cls.__module__+'.'+str(mod))
    #     return methodNameList
    
    # @staticmethod
    # def getClassListInDirectory(directory_path): 
    #     class_list = [] 
    #     # Walk through the directory 
    #     for root, dirs, files in os.walk(directory_path): 
    #         for file in files: # Check if the file is a Python file 
    #             if file.endswith('.py'): 
    #                 file_path = os.path.join(root, file) 
    #                 module_name = os.path.splitext(file)[0] 
    #                 try: # Import the module dynamically 
    #                     spec   = importlib.util.spec_from_file_location(module_name, file_path) 
    #                     module = importlib.util.module_from_spec(spec) 

    #                     spec.loader.exec_module(module) # Get all classes in the module 

    #                     for name, obj in inspect.getmembers(module): 
    #                         if inspect.isclass(obj) and obj.__module__ == module.__name__: 
    #                             class_list.append(obj) 
    #                 except Exception as e: 
    #                     if file != '__init__.py':
    #                         print('========================================================================[Begin]')
    #                         print(f"An error occurred: {e}")
    #                         print('>>> file:'+str(file)+' directory_path:'+str(directory_path))
    #                         print('========================================================================[End]')

    #     return class_list 

    @staticmethod
    def getClassListInDirectory(directory_path, package_root=None):
        if package_root == None:
            package_root = os.path.join(os.environ['PYTHON_PATH_PYTHONLIBS'], "pythonLibs")
        
        class_list = []
        
        if package_root:
            package_root_parent = os.path.dirname(package_root)
            if package_root_parent not in sys.path:
                sys.path.insert(0, package_root_parent)
        else:
            package_root_parent = os.path.dirname(directory_path)
            if package_root_parent not in sys.path:
                sys.path.insert(0, package_root_parent)

        for root, dirs, files in os.walk(directory_path):
            # Exclude specific directories
            dirs[:] = [d for d in dirs if d not in {'node_modules', '__pycache__', '.git', '.venv', 'venv', 'env', 'scripts_original', 'examples', 'scripts', 'configs', 'studies', 'tests', 'docs', 'Playground'}]
            
            for file in files:
                if (
                    file.endswith('.py')
                    and file not in {'__init__.py', '__main__.py', 'setup.py'}
                    and not file.startswith('test_')
                ):
                    file_path = os.path.join(root, file)

                    rel_path = os.path.relpath(file_path, package_root_parent)
                    module_name = rel_path.replace(os.sep, ".").replace(".py", "")

                    try:
                        spec = importlib.util.spec_from_file_location(module_name, file_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.name = module_name
                            spec.loader.exec_module(module)

                            for name, obj in inspect.getmembers(module, inspect.isclass):
                                if obj.__module__ == module.__name__:
                                    class_list.append(obj)

                    except Exception as e:
                        # Log the error but continue gracefully
                        # print(f"Warning: Could not inspect module '{module_name}' in '{file}': {e}")
                        pass
        
        # print('directory_path:'+str(directory_path))
        # print('package_root:'+str(package_root))
        # print('class_list:'+str(class_list))

        return class_list

    @staticmethod
    def getMethodInstanceList(cls=None):
        # Get a list of all attributes of the object
        attributes = dir(cls)
        # Filter out the methods
        methodInstanceList = [attr for attr in attributes if callable(getattr(cls, attr)) and not attr.startswith('__')]
        return methodInstanceList
