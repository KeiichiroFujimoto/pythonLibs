from pythonLibs.dictHandler import DictItem, DictHandler
from pythonLibs.dataHandler import DataItem

class DictBase(dict):

    def __init__(self, initial_dict=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if initial_dict:
            self.map_from_dict(initial_dict)

    def __setitem__(self, key, value):
        import numpy as np
        if isinstance(value,np.ndarray):
            value = value.tolist()
        if isinstance(value, tuple) and len(value) == 2:
            value, unit = value
            item = DictItem(data=value, unit=unit)
            super().__setitem__(key, item)
        elif isinstance(value, DictBase):
            super().__setitem__(key, value)
        elif isinstance(value, dict):
            # Recursively convert nested dictionaries to DictBase
            item = DictBase(value)
            super().__setitem__(key, item)
        elif isinstance(value, list):
            # Recursively convert lists of dictionaries to lists of DictBase
            item = [DictBase(v) if isinstance(v, dict) else v for v in value]
            super().__setitem__(key, item)
        elif isinstance(value, float) or isinstance(value, int) or isinstance(value, str) or isinstance(value,bytes) or isinstance(value,type(None)):
            super().__setitem__(key, value)
        else:
            item = DictItem(data=value)
            super().__setitem__(key, item)

    def __getitem__(self, key):
        item = super().__getitem__(key)
        return item.getData() if isinstance(item, DictItem) else item
    
    def getItem(self, key):
        item = super().__getitem__(key)
        return item

    def __repr__(self):
        items = ", ".join(f'"{key}": {repr(value)}' for key, value in self.items())
        return f"{self.__class__.__name__}({{{items}}})"

    def __str__(self):
        result = None
        items  = []
        for key, value in self.items():
            if isinstance(value, DictItem):
                # data = f'"{value.getData()}"' if isinstance(value.getData(), str) else value.getData()
                # item_str = f'"{key}": {data} {value.getUnit() if value.getUnit() else ""}'.strip()
                #===========================================================
                # Handling for Data
                #===========================================================
                if isinstance(value.getData(), str):
                    data = f'{value.getData()}' 
                elif isinstance(value.getData(), list):
                    buff = ""
                    for dataItem in value.getData():
                        if buff == "":
                            buff += str(dataItem)
                        else:
                            buff += ","+str(dataItem)
                    data = f'{buff}' 
                elif isinstance(value.getData(), DictItem):
                    dictItem = value.getData()
                    buff = ""
                    for dataItem in dictItem.getData():
                        if buff == "":
                            buff += str(dataItem)
                        else:
                            buff += ","+str(dataItem)
                    data = f'{buff}' 
                elif isinstance(value.getData(), DataItem):
                    buff = ""
                    for dataItem in value.getData().dataList:
                        if buff == "":
                            buff += str(dataItem)
                        else:
                            buff += ","+str(dataItem)
                    data = f'{buff}' 
                else:
                    data = value.getData()
                
                #===========================================================
                # Handling for Unit
                #===========================================================
                if value.getUnit():
                    unit = f'[{value.getUnit()}]'
                elif isinstance(value.getData(), DictItem):
                    dictItem = value.getData()
                    unit = ""
                elif isinstance(value.getData(), DataItem):
                    dataItem = value.getData().dataList
                    unit = f'[{value.getData().getUnit()}]'
                else:
                    unit = ""

                # data = f'{value.getData()}' if isinstance(value.getData(), str) else value.getData()
                # unit = f'[{value.getUnit()}]' if value.getUnit() else ""

                item_str = f'"{key}": "{data}{unit}"'
            elif isinstance(value, DictBase):
                item_str = f'"{key}": {value}'
            elif isinstance(value, list):
                # Handle lists of DictBase objects
                list_items = []
                for item in value:
                    if isinstance(item, DictBase):
                        list_items.append(str(item))
                    else:
                        list_items.append(repr(item))
                item_str = f'"{key}": [{", ".join(list_items)}]'
            elif isinstance(value, str):
                item_str = f'"{key}": "{value}"'
            else:
                item_str = f'"{key}": {value}'

            items.append(item_str)

            result = f"{{ {', '.join(items)} }}"

            result = result.replace('"None"','None')

        if result is None:
            return '{}'
        else:
            return result

    def items(self):
        return super().items()
    
    def toDict(self):
        """
        Proper recursive conversion of DictBase/DictItem structure to a plain dict.
        Bypasses the fragile string-based conversion.
        """
        def _convert(obj):
            if isinstance(obj, DictBase):
                res = {}
                for k in super(DictBase, obj).keys():
                    val = super(DictBase, obj).__getitem__(k)
                    res[k] = _convert(val)
                return res
            elif isinstance(obj, DictItem):
                data = obj.getData()
                unit = obj.getUnit()
                if unit:
                    # Maintain the "data[unit]" string format if unit exists
                    try:
                        return f"{_convert(data)}[{unit}]"
                    except:
                        return f"{data}[{unit}]"
                return _convert(data)
            elif isinstance(obj, list):
                return [_convert(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj
            
        return _convert(self)
    
    def map_from_dict(self, input_dict):
        for key, value in input_dict.items():
            self[key] = value