import os
import json
from jsonpath_ng import parse

class JsonHandler():

    @staticmethod
    def write(dict_out=None,filePathJson=None,indent=2,encoding='utf-8'):
        print('WRITE:'+str(dict_out)+' filePathJson:'+str(filePathJson))

        with open(filePathJson, 'w', encoding=encoding) as f:
            json.dump(dict_out, f, indent=indent)

    @staticmethod
    def read(filePathJson=None, encoding='utf-8'):
        with open(filePathJson, encoding=encoding) as f:
            dict_read = json.load(f)
            return dict_read
    
    @staticmethod
    def update(dict_json,key=None,value=None):

        jsonpath_expr = parse(key)
        for match in jsonpath_expr.find(dict_json):
            field = match.path.fields[0]
            parent = match.context.value
            parent[field] = value
        
        print('dict_json:'+str(dict_json))
        return dict_json

    @staticmethod
    def dumps(obj=None):
        return json.dumps(obj=obj)
    
    @staticmethod
    def getDictFromString(jsonStr=None):
        d = json.loads(jsonStr)
        return d

    @staticmethod
    def getJsonStringFromDict(dict_json=None):
        return JsonHandler.dumps(obj=dict_json)
    
    @staticmethod
    def isJsonString(s):
        try:
            json.loads(s)
            return True
        except ValueError:
            return False
