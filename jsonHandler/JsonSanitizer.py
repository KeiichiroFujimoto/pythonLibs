class JsonSanitizer:
    TEMP_TOKEN = '__TEMP_EMPTY__'

    @staticmethod
    def sanitize(json_string: str, displayInfo:bool=False) -> str:
        # 一時トークンが含まれていない場合のみ置換
        if JsonSanitizer.TEMP_TOKEN not in json_string:
            json_string = json_string.replace('""}', f'{JsonSanitizer.TEMP_TOKEN}}}')
            json_string = json_string.replace('"",', f'{JsonSanitizer.TEMP_TOKEN},')
            json_string = json_string.replace('""', '"')
            json_string = json_string.replace(f'{JsonSanitizer.TEMP_TOKEN}}}', '""}')
            json_string = json_string.replace(f'{JsonSanitizer.TEMP_TOKEN},', '"",')

        json_string = json_string.replace('None', '""').replace('null', '""')
        json_string = json_string.replace("'", '"').replace('True', 'true').replace('False', 'false')
        
        if displayInfo:
          print("#### jsonString:"+str(json_string))

        return json_string