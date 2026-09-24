from pythonLibs.dataHandler.EngUnitConversion import (
    Temperature,
    Length,
    Mass,
    Pressure,
    Force,
    Current,
    Power,
    Density,
    EngTime,
)


class DataService:
    _UNIT_TYPE_MAP = {
        "temperature": Temperature,
        "length": Length,
        "mass": Mass,
        "pressure": Pressure,
        "force": Force,
        "current": Current,
        "power": Power,
        "density": Density,
        "engtime": EngTime,
        "time": EngTime,
    }

    @staticmethod
    def getUnitNameListByType(unitType: str) -> list[str]:
        if not unitType:
            return []
        unit_class = DataService._UNIT_TYPE_MAP.get(unitType.strip().lower())
        if unit_class is None:
            return []
        return DataService._extractUnitNames(unit_class)

    @staticmethod
    def _extractUnitNames(unit_class) -> list[str]:
        unit_cls = getattr(unit_class, "Unit", None)
        if unit_cls is not None:
            values = [
                value
                for name, value in vars(unit_cls).items()
                if not name.startswith("_") and isinstance(value, str)
            ]
            if values:
                return sorted(set(values))
        conversions = getattr(unit_class, "conversions", None)
        if isinstance(conversions, dict):
            return sorted(conversions.keys())
        return []
