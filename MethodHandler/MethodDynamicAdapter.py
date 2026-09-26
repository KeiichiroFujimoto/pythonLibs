import inspect
import json
from typing import Callable
from fastapi.responses import JSONResponse
from fastapi import HTTPException

import os, sys
if not os.environ.get('NEXUS_PATH_CONFIGURED') and os.environ.get('PYTHON_PATH_PYTHONLIBS') and os.environ['PYTHON_PATH_PYTHONLIBS'] not in sys.path: sys.path.append(os.environ['PYTHON_PATH_PYTHONLIBS'])

from pythonLibs.fileHandler import FilePathHandler
# pythonLibs.webHandler は HTML レスポンス生成 (316行目) でのみ使用するため、
# 当該経路に入った時だけ lazy import する（SDK 閉包から webHandler を外すため）。

import re
from typing import Dict, Iterable, Tuple, Optional

def parse_kv(text: str, keys: Optional[Iterable[str]] = None,
             separators: str = ',;') -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Parse a loose key=value parameter string into a dict.

    Features:
      - Accepts quotes: single '...' or double "..."
      - Handles escaped quotes inside values
      - Tolerates missing closing quote by reading to end or next separator
      - Ignores leading stray quotes or separators before a key
      - Accepts unquoted values up to the next separator
      - Optionally filter to a set of `keys`

    Returns:
      (result, errors) where:
        result: dict of parsed key -> value
        errors: dict of key -> error message (if any minor issues occurred)
    """
    result: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    keys_set = set(keys) if keys is not None else None

    s = text
    n = len(s)
    i = 0

    def skip_ws_and_pre_key_noise(j: int) -> int:
        # Skip whitespace, separators, and leading stray quotes before a key
        while j < n and (s[j].isspace() or s[j] in separators + '\'"'):
            j += 1
        return j

    while i < n:
        i = skip_ws_and_pre_key_noise(i)
        if i >= n:
            break

        # Parse key: [A-Za-z_]\w*
        if not (s[i].isalpha() or s[i] == '_'):
            # Not a valid key start; skip this char and continue
            i += 1
            continue

        k_start = i
        i += 1
        while i < n and (s[i].isalnum() or s[i] == '_'):
            i += 1
        key = s[k_start:i]

        # Expect '='
        while i < n and s[i].isspace():
            i += 1
        if i >= n or s[i] != '=':
            # Not a key=value; skip to next separator
            while i < n and s[i] not in separators:
                i += 1
            i += 1
            continue
        i += 1  # skip '='

        # Skip whitespace before value
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            # key with empty value
            if keys_set is None or key in keys_set:
                result[key] = ''
                errors[key] = 'missing value'
            break

        # Parse value
        val = []
        had_quote = False
        quote_char = ''
        if s[i] in ('"', "'"):
            had_quote = True
            quote_char = s[i]
            i += 1
            escaped = False
            closed = False
            while i < n:
                c = s[i]
                i += 1
                if escaped:
                    val.append(c)
                    escaped = False
                elif c == '\\':
                    escaped = True
                elif c == quote_char:
                    closed = True
                    break
                else:
                    val.append(c)
            if not closed:
                # Missing closing quote: we already captured until end-of-string.
                if keys_set is None or key in keys_set:
                    errors[key] = 'missing closing quote'
        else:
            # Unquoted: read until separator
            while i < n and s[i] not in separators:
                val.append(s[i])
                i += 1

        value = ''.join(val).strip()

        # Store if requested
        if keys_set is None or key in keys_set:
            result[key] = value

        # Move past separator if present
        if i < n and s[i] in separators:
            i += 1

    return result, errors

def extract_fields(text: str, *fields: str) -> Dict[str, str]:
    """
    Convenience wrapper to extract specific fields.
    Missing fields will not be present in the dict.
    """
    parsed, _ = parse_kv(text, keys=fields)
    return {k: v for k, v in parsed.items() if k in fields}

import ast
def parse_kwargs_string(s: str) -> dict:
    try:
        tree = ast.parse(f"func({s})", mode='eval')
        if isinstance(tree.body, ast.Call):
            return {kw.arg: ast.literal_eval(kw.value) for kw in tree.body.keywords}
    except Exception as e:
        print(f"Failed to parse kwargs-style string: {e}")
    return {}

def convert_keys_to_quoted(input_str: str) -> str:
    """
    Converts unquoted keys in a key=value formatted string to quoted keys.
    Already quoted keys are left unchanged.
    
    Example:
        Input:  'filePathMarkdown="hoge.md", "level"=1, text="HOGE"'
        Output: '"filePathMarkdown"="hoge.md", "level"=1, "text"="HOGE"'
    """
    parts = input_str.split(', ')
    converted_parts = []

    for part in parts:
        key, value = part.split('=')
        key = key.strip()
        value = value.strip()
        
        # すでにキーが引用されているかチェック
        if not (key.startswith('"') and key.endswith('"')):
            key = f'"{key}"'
        
        converted_parts.append(f'{key}={value}')

    return ', '.join(converted_parts)

class MethodDynamicAdapter:
    
    @staticmethod
    def generateWrapper(targetClass=None, 
                        methodName: str = None, 
                        filePathConfigToolExecutor: str = None, 
                        modeDateTimeDirectory:str="Date",
                        CaseName:str=None) -> Callable:
        
        print('_/_/_/_/_/_/_/_/ generateWrapper')
        print('targetClass:'+str(targetClass))
        print('methodName:' +str(methodName))

        method = getattr(targetClass, methodName)
        if not callable(method):
            raise ValueError(f"{methodName} is not a callable method of {targetClass.__name__}")

        # If the method requires 'self' (instance method / decorated wrapper),
        # create an instance and bind it so callers don't need to pass 'self'.
        # Note: inspect.signature follows __wrapped__ (from functools.wraps),
        # so we check the actual code object to detect 'self'.
        _needs_self = (
            hasattr(method, '__code__')
            and method.__code__.co_varnames
            and method.__code__.co_varnames[0] == 'self'
        )
        if _needs_self:
            _instance = targetClass()
            method = getattr(_instance, methodName)

        def wrapper(*args, **kwargs):
            
            if kwargs and "accountName" in kwargs.keys():
                accountName = kwargs["accountName"]
                del kwargs["accountName"]
            else:
                accountName = "anonymous"

            nonlocal CaseName
            try:
                print('WRAPPER ###################### args:', args)
                print('WRAPPER ###################### kwargs:', kwargs)
                print('WRAPPER ###################### accountName:'+str(accountName))
                
                # toolExecutorの前処理
                toolExe = None
                if filePathConfigToolExecutor is not None:
                    print('PREPROCESS ######################')
                    toolExe  = targetClass.constructToolExecutorBase(filePathConfigToolExecutor=filePathConfigToolExecutor)
                    CaseName = targetClass.__name__ if CaseName is None else CaseName
                    toolExe.runPreprocess(accountName=accountName, CaseName=CaseName, modeDateTimeDirectory=modeDateTimeDirectory)

                print('===================================================================1')
                
                filePathList = []
                
                from pythonLibs.docstringHandler import DocstringHandler
                paramNameList  = DocstringHandler.getParameterNameList(function=method)
                paramName_dict = {paramName: None for paramName in paramNameList}

                print('===================================================================2')
                
                if type(args) is tuple and len(args)>0:
                    if len(list(args)) != len(paramNameList):
                        vals, errs = parse_kv(args[0], keys=paramName_dict)
                        print(vals)
                        print(errs)
                        args = list(vals.values())
                
                if type(args) == tuple:
                    args = list(args)

                print('===================================================================3')

                if args and type(args[0])==str:
                    args[0] = args[0].replace('\n','')

                    try:
                        parsed = json.loads(args[0])
                        if isinstance(parsed, dict):
                            args = [parsed] + list(args[1:])
                    except json.JSONDecodeError:
                        pass

                print('===================================================================4')

                if args and type(args[0])==str:
                    parsed = None
                    try:
                        parsed = json.loads(args[0])
                        if isinstance(parsed, dict):
                            args = [parsed] + list(args[1:])
                    except json.JSONDecodeError:
                        pass


                    if parsed is None and isinstance(args[0], str) and args[0].strip().startswith("{"):
                        parsed_kwargs = parse_kwargs_string(args[0])
                        if parsed_kwargs:
                            kwargs.update(parsed_kwargs)
                            args = args[1:]

                    # dict風
                    if parsed is None:
                        try:
                            parsed_dict = ast.literal_eval(args[0])
                            if isinstance(parsed_dict, dict):
                                args = [parsed_dict] + list(args[1:])
                        except Exception:
                            pass

                print('===================================================================5')

                # dict → methodの引数にマッピング
                if args and isinstance(args[0], dict):
                    input_dict = args[0]
                    sig = inspect.signature(method)
                    bound_args = sig.bind_partial(**input_dict)
                    bound_args.apply_defaults()
                    kwargs.update(bound_args.arguments)
                    args = args[1:]

                print('###################### final args:', args)
                print('###################### final kwargs:', kwargs)

                result = method(*args, **kwargs)
                
                print('Result:'+str(result))

                # HTMLレスポンス
                if isinstance(result, str) and FilePathHandler.isValidFilepath(s=result):
                    if FilePathHandler.getExtension(filePath=result) == 'html':
                        if toolExe is not None:
                            toolExe.runPostprocess()
                        from pythonLibs.webHandler import FastAPIHandler  # lazy
                        return FastAPIHandler.getHTMLResponse(filePathHTML=result)
                
                # executionResultを含む辞書 → JSONレスポンス
                if isinstance(result, dict) and "executionResult" in result:
                    if toolExe is not None:
                        toolExe.runPostprocess()
                    return JSONResponse(content={"status": "success", "result": str(result["executionResult"])})

                # その他 → 通常レスポンス
                if toolExe is not None:
                    toolExe.runPostprocess()
                return result
            
            except Exception as e:
                if toolExe is not None:
                    toolExe.runPostprocess()
                raise HTTPException(status_code=500, detail=str(e))

        original_signature    = inspect.signature(method)
        parameters            = list(original_signature.parameters.values())
        wrapper.__signature__ = original_signature.replace(parameters=parameters)
        wrapper.__doc__       = method.__doc__

        # Expose the toolBase instance (if any) so the FastAPI endpoint
        # can access paramDict["executionResult"] after invocation.
        wrapper._tool_instance = _instance if _needs_self else None

        return wrapper

        
        


                


# Test class
class UTIL:

    @staticmethod
    def printText(text1: str = None, text2: str = None, valueInt1: int = None):
        print("text1:", text1)
        print("text2:", text2)
        print("valueInt1:", str(valueInt1))
        return {"executionResult": f"{text1} + {text2}"}

# Execution test
if __name__ == "__main__":
    func = MethodDynamicAdapter.generateWrapper(targetClass=UTIL, methodName='printText')
    
    
    # Call with JSON string
    result = func('{"text1": "hoge1", "text2": "hoge2", "valueInt1": 10}')
    print("Returned:", result)

    result = func(text1="hoge1",text2="hoge2",valueInt1=10)
    print("Returned:", result)
