from typing import Optional, Dict
import json

class JsonExtractor:
    @classmethod
    def extract_json_dict(cls, text: str) -> Optional[Dict]:
        """
        Extracts the first JSON-like dictionary from a given string.

        Args:
            text (str): A string containing a JSON dictionary followed by other text.

        Returns:
            Optional[Dict]: The extracted dictionary if found and valid, otherwise None.
        """
        brace_stack = []
        start_idx = None

        for i, char in enumerate(text):
            if char == '{':
                if not brace_stack:
                    start_idx = i
                brace_stack.append(char)
            elif char == '}':
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and start_idx is not None:
                        json_str = text[start_idx:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            return None
        return None

