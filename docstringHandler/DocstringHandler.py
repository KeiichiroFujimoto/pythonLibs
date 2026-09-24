import re
import inspect
from typing import Dict

class DocstringHandler:
  
  @staticmethod
  def parse(function):
    
    print('_/_/_/_/_/_/_/ parse function:'+str(function))
    print('_/_/_/_/_/_/_/ type(function):'+str(type(function)))
    
    docstring = inspect.getdoc(function)

    return DocstringHandler.parseBase(docstring=docstring,functionName=function.__name__)
  
  # @staticmethod
  # def parseBase(docstring=None,functionName=None):
  #   doc_dict  = {}
    
  #   if docstring:
  #       doc_dict['name'] = functionName
  #       # Parse summary
  #       summary = docstring.split('\n\n')[0].strip()
  #       doc_dict['summary'] = summary

  #       # Parse parameters and returns using regex
  #       param_pattern = re.compile(r'(\w+)\s\((\w+)\):\s(.+)')
  #       params_match = param_pattern.findall(docstring)
  #       returns_match = re.search(r'Returns:\n\s*(\w+):\s(.+)', docstring)

  #       doc_dict['parameters'] = [{'name': p[0], 'type': p[1], 'description': p[2].strip()} for p in params_match]
  #       if returns_match:
  #           doc_dict['returns'] = {'type': returns_match.group(1), 'description': returns_match.group(2).strip()}

  #   return doc_dict

  @staticmethod
  def parseBase(docstring=None, functionName=None):
    doc_dict = {}

    if docstring:
        doc_dict['name'] = functionName

        # Parse summary
        summary = docstring.strip().split('\n\n')[0].strip()
        doc_dict['summary'] = summary

        # Extract Args and Returns sections
        args_section = ''
        returns_section = ''
        args_match = re.search(r'Args:\s*(.*?)(?=\n\S|Returns:|$)', docstring, re.DOTALL)
        if args_match:
            args_section = args_match.group(1)

        returns_match = re.search(r'Returns:\s*(.*)', docstring, re.DOTALL)
        if returns_match:
            returns_section = returns_match.group(1)

        # Parse parameters from Args section only
        doc_dict['parameters'] = []
        param_pattern = re.compile(r'^\s*(\w+)\s*\(([^)]+)\):\s*(.+)$', re.MULTILINE)
        params_match = param_pattern.findall(args_section)
        doc_dict['parameters'] = [
            {'name': p[0], 'type': p[1], 'description': p[2].strip()}
            for p in params_match
        ]

        # Parse return from Returns section only
        return_pattern = re.compile(r'^\s*(\w+)\s*\(([^)]+)\):\s*(.+)$', re.MULTILINE)
        return_match = return_pattern.search(returns_section)
        if return_match:
            doc_dict['returns'] = {
                'name': return_match.group(1),
                'type': return_match.group(2),
                'description': return_match.group(3).strip()
            }

    return doc_dict

  @staticmethod
  def getMethodName(function=None):
    doc_dict    = DocstringHandler.parse(function=function)
    return doc_dict['name']

  @staticmethod
  def getParameters(function=None):
    doc_dict    = DocstringHandler.parse(function=function)
    params      = doc_dict['parameters']
    return params
  
  @staticmethod
  def getParameterNameList(function=None):
    paramNameList = []
    for param in DocstringHandler.getParameters(function=function):
      paramNameList.append(param['name'])
    return paramNameList

  @staticmethod
  def getDescriptionParameters(function=None):
    doc_dict    = DocstringHandler.parse(function=function)
    print("################### doc_dict:"+str(doc_dict))
    params      = doc_dict['parameters']
    description = "Parameters:\n"
    for param in params:
      param_name = param['name']
      param_type = param['type']
      param_desc = param['description']
      description += "- " + param_name + "(" + param_type + "): " + param_desc + "\n"
    return description

  @staticmethod
  def getSummary(function=None):
    doc_dict    = DocstringHandler.parse(function=function)
    return doc_dict['summary']

  @staticmethod
  def generateDocumentAsHTML(doc_dict):
    html_content = f"""
    <html>
    <head>
        <title>{doc_dict.get('name', 'Method Documentation')} Documentation</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; }}
            p {{ color: #7f8c8d; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ margin-bottom: 10px; }}
            pre {{ background-color: #ecf0f1; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>{doc_dict.get('name', 'Method Documentation')} Documentation</h1>
        <p>{doc_dict.get('summary', '')}</p>
        
        <h2>Parameters</h2>
        <ul>
    """
    for param in doc_dict.get('parameters', []):
        html_content += f"""
            <li>
                <strong>{param['name']}</strong> ({param['type']}): {param['description']}
            </li>
        """
    
    html_content += """
        </ul>
    </body>
    </html>
    """
    
    return html_content
  
  @staticmethod
  def extractDocTags(docstring: str) -> Dict[str, str]:
    """
    Extract key-value pairs from a docstring in the format '@key:value'.

    Parameters:
        docstring (str): The docstring to parse.

    Returns:
        Dict[str, str]: A dictionary of extracted key-value pairs.
    """
    pattern = r"@(\w+):\s*(.+)"
    return {match[0]: match[1] for match in re.findall(pattern, docstring)}