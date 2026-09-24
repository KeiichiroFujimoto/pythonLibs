# pythonLibs

Public-domain extraction of the `toolBase` / `toolBaseSecured` automation layer.

This repository is intended as a small foundation for building executable tools
that can expose the same operation through:

- `@secure_expose` command catalogs
- `invoke(alias, **kwargs)` pipeline calls
- local CLI / REPL backends
- optional REST wrappers
- optional LangChain structured tools

## Core Packages

The initial cut contains the toolBase runtime and its small utility dependency
closure:

```text
tool
MethodHandler
fileHandler
dictHandler
jsonHandler
xmlHandler
yamlHandler
tomlHandler
docstringHandler
dateTimeHandler
os
dataHandler
tableHandler
SchemaVersions.py
```

Domain packages such as aerospace, world models, CAD, agents, and application
services are intentionally not included.

Login, password, JWT, and user-database implementations are also intentionally
not included. Applications can inject their own verifier when they need access
control.

The repository root is mapped as the `pythonLibs` package so GitHub paths stay
flat while Python imports remain compatible with existing code.

## Example

```python
from pythonLibs.tool.toolBaseSecured import toolBaseSecured
from pythonLibs.tool.decorators import secure_expose


class NoteService(toolBaseSecured):
    def __init__(self):
        super().__init__(exposure_mode="expose_only", secure_enabled=False)
        self._notes = {}

    @secure_expose(alias="addNote", category="notes")
    def add_note(self, note_id: str, text: str) -> dict:
        self._notes[note_id] = text
        return {"noteId": note_id}


svc = NoteService()
print(svc.buildCatalog())
print(svc.invoke("addNote", noteId="n1", text="hello"))
```

## Optional Dependencies

Install only the extras you need:

```bash
pip install -e ".[config,numeric]"
pip install -e ".[rest]"
pip install -e ".[llm]"
```

## License

This project is released into the public domain under the Unlicense.
