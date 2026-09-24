from pythonLibs.dictHandler import DictItem
import json

class DictItemEncoderJson(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, DictItem):
        return obj.getData()
    return super().default(obj)

class DictItemEncoderToml:
    @staticmethod
    def encode(obj):
        if isinstance(obj, DictItem):
            return obj.getData()
        elif isinstance(obj, dict):
            return {k: DictItemEncoderToml.encode(v) for k, v in obj.items()}
        else:
            return obj
