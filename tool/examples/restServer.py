"""Serve any toolBaseSecured service over HTTP with FastAPI.

The two endpoints below are the protocol ``ServiceREPL.RemoteBackend`` speaks, so the
same service can be driven from a browser, curl, another program, or a remote REPL:

    GET  /api/commands          -> {"commands": [...]}   (buildCatalog())
    POST /api/commands/invoke   {"command": "tipDeflection", "params": {"load": 1000, "length": 2}}
                                -> {"success": true, "result": {...}}

Run (needs the ``rest`` extra: pip install -e ".[rest]"):

    python -m pythonLibs.tool.examples.restServer            # serves on http://127.0.0.1:8322
    python -m pythonLibs.tool http://127.0.0.1:8322          # remote REPL on the same commands
    curl http://127.0.0.1:8322/api/commands
"""
from typing import Any, Dict

from pythonLibs.tool.decorators import secure_expose
from pythonLibs.tool.toolBaseSecured import toolBaseSecured


class BeamService(toolBaseSecured):
    """Example service: one exposed method becomes one command on every interface."""

    def __init__(self):
        super().__init__(instance_subcls=self, exposure_mode="expose_only", secure_enabled=False)

    @secure_expose(alias="tipDeflection", category="structures")
    def tip_deflection(self, load: float, length: float, e_modulus: float = 200e9, inertia: float = 8e-6) -> dict:
        """Tip deflection of a cantilever beam under an end load.

        Args:
            load (float): End load [N].
            length (float): Beam length [m].
            e_modulus (float): Young's modulus [Pa].
            inertia (float): Second moment of area [m^4].
        """
        return {"deflectionM": load * length ** 3 / (3.0 * e_modulus * inertia)}


def createApp(service: toolBaseSecured):
    """Build a FastAPI app exposing ``service`` through the /api/commands protocol."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    class InvokeRequest(BaseModel):
        command: str
        params: Dict[str, Any] = {}

    app = FastAPI(title=type(service).__name__)

    @app.get("/api/commands")
    def listCommands():
        return {"commands": service.buildCatalog()}

    @app.post("/api/commands/invoke")
    def invokeCommand(request: InvokeRequest):
        try:
            return {"success": True, "result": service.invoke(request.command, **request.params)}
        except Exception as exc:  # report failures to the client instead of a bare HTTP 500
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(createApp(BeamService()), host="127.0.0.1", port=8322)
