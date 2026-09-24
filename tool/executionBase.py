from pythonLibs.tool.toolBase import toolBase


class executionBase(toolBase):
    """General execution-oriented base class.

    `toolBase` remains for compatibility, but `executionBase` is the preferred
    upper concept for new code because the base class is used not only for
    tools, but also for handlers, model runners, simulators, and human-facing
    execution surfaces.
    """

    pass
