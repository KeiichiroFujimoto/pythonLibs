import ast
from typing import Any, Dict, List
import json

class DictValueConverter:
    """
    A general-purpose utility class for converting values in a dictionary.
    Provides methods to convert string representations of lists, dictionaries, numbers, etc.
    into proper Python literal types.
    """
    
    @staticmethod
    def convert(data: Dict[str, Any], withPreprocessDefault:bool=True, withConvertLiteralValues:bool=True, resturnAsDict=True ):

        typeInitial = type(data)

        if withPreprocessDefault:
            dataString = str(data)
            data = DictValueConverter.preprocess_default(data=str(data))
        
        if withConvertLiteralValues:
            data = DictValueConverter.convert_literal_values(data=json.loads(data),keys=None)
            print('data(After ConvertLiteralValues):'+str(data))
        
        if resturnAsDict:
            return data
        else:
            return str(data)
    
    @staticmethod
    def convert_literal_values(data: Dict[str, Any], keys: List[str]=None) -> Dict[str, Any]:
        """
        Converts string values associated with specified keys into Python literals.
        Example: '[1, 2, 3]' → [1, 2, 3]
        """
        if keys is None:
            keys = data.keys()
        
        for key in keys:
            if key in data and isinstance(data[key], str):
                try:
                    data[key] = ast.literal_eval(data[key])
                except Exception as e:
                    print(f"Conversion failed for key '{key}': {data[key]} | Error: {e}")
        return data

    @staticmethod
    def convert_numeric_values(data: Dict[str, Any], keys: List[str]=None) -> Dict[str, Any]:
        """
        Converts string values associated with specified keys into numeric types (int or float).
        Example: '1.0' → 1.0, '42' → 42
        """
        if keys is None:
            keys = data.keys()
        
        for key in keys:
            if key in data and isinstance(data[key], str):
                try:
                    if '.' in data[key]:
                        data[key] = float(data[key])
                    else:
                        data[key] = int(data[key])
                except Exception as e:
                    print(f"Numeric conversion failed for key '{key}': {data[key]} | Error: {e}")
        return data

    @staticmethod
    def preprocess_default(data: str) -> str:
        """
        Preprocesses a tool call string by:
        - Replacing single quotes with double quotes
        - Converting Python-style booleans to JSON-style booleans
        """
        json_string = (
            data.replace("'", '"')
                .replace("True", "true")
                .replace("False", "false")
        )
        return json_string

    @staticmethod
    def getAsDict(jsonString:str):
        jsonString = jsonString.replace("'",'"')
        return json.loads(jsonString)
