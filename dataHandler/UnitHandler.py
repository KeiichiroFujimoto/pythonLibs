import sys
import os

if not os.environ.get('NEXUS_PATH_CONFIGURED') and os.environ.get('PYTHON_PATH_PYTHONLIBS') and os.environ['PYTHON_PATH_PYTHONLIBS'] not in sys.path:
    sys.path.append(os.environ['PYTHON_PATH_PYTHONLIBS'])

from pint import UnitRegistry

from pythonLibs.dataHandler.EngUnitConversion import (
    Temperature, Length, Mass, Pressure, Force, Current, Power, Density, EngTime,
)

try:
    import pandas as pd
except Exception:
    class _DummySeries:
        pass
    class _DummyPandas:
        core = type('core', (), {'series': type('series', (), {'Series': _DummySeries})})
    pd = _DummyPandas()

try:
    import numpy as np
except Exception:
    class _DummyNumpy:
        float64 = float
    np = _DummyNumpy()

_ureg = UnitRegistry()

# 既存の unitName を pint が理解する形式に変換するエイリアス
_PINT_ALIASES = {
    # Temperature
    "C": "degC", "F": "degF", "R": "degR",
    # Length
    "nautMi": "nautical_mile", "lightYear": "light_year",
    # Mass
    "metricTon": "metric_ton", "shortTon": "short_ton", "longTon": "long_ton",
    # Pressure
    "kgcm2": "kgf/cm**2",
    # Force
    "gf": "gram_force", "kgf": "kilogram_force", "tonf": "metric_ton * g_0",
    "shortTonF": "short_ton * g_0", "longTonF": "long_ton * g_0",
    "kipf": "kip", "pdl": "poundal", "ozf": "ounce_force",
    # Power
    "hp_mech": "horsepower", "hp_ele": "electrical_horsepower",
    "hp_metric": "metric_horsepower",
    "BTU/hr": "BTU/hour", "BTU/min": "BTU/minute", "BTU/sec": "BTU/second",
    "cal/sec": "calorie/second", "cal/min": "calorie/minute", "cal/hr": "calorie/hour",
    "kCal/sec": "kilocalorie/second", "kCal/min": "kilocalorie/minute", "kCal/hr": "kilocalorie/hour",
    "erg/sec": "erg/second", "erg/min": "erg/minute", "erg/hr": "erg/hour",
    "ftlb/sec": "foot*pound_force/second",
    "J/sec": "J/second", "J/min": "J/minute", "J/hr": "J/hour",
    "kgf-m/sec": "kilogram_force*meter/second",
    # Density
    "g/cm3": "g/cm**3", "kg/m3": "kg/m**3",
    # Time
    "minute": "min", "hr": "hour",
    # AngularRate
    "deg/s": "degree/second", "rad/s": "radian/second",
    "rpm": "revolution/minute",
}

# getUnitType 用のマッピング (互換性維持)
_UNIT_TYPE_MAP = {
    "DateTime": "DateTime",
    "K": "Temperature", "C": "Temperature", "F": "Temperature", "R": "Temperature",
    "fm": "Length", "pm": "Length", "nm": "Length", "um": "Length",
    "mm": "Length", "cm": "Length", "m": "Length", "dam": "Length",
    "hm": "Length", "km": "Length", "Mm": "Length", "Gm": "Length",
    "Tm": "Length", "Pm": "Length", "inch": "Length", "ft": "Length",
    "yd": "Length", "mi": "Length", "nautMi": "Length", "lightYear": "Length",
    "kg": "Mass", "g": "Mass", "mg": "Mass", "metricTon": "Mass",
    "lb": "Mass", "oz": "Mass", "grain": "Mass", "shortTon": "Mass",
    "longTon": "Mass", "slug": "Mass",
    "bar": "Pressure", "mbar": "Pressure", "ubar": "Pressure",
    "Pa": "Pressure", "hPa": "Pressure", "kPa": "Pressure", "MPa": "Pressure",
    "kgcm2": "Pressure", "atm": "Pressure", "mmHg": "Pressure",
    "mmH2O": "Pressure", "mH2O": "Pressure", "psi": "Pressure",
    "ftH2O": "Pressure", "inH2O": "Pressure", "inHg": "Pressure",
    "N": "Force", "kN": "Force", "MN": "Force", "GN": "Force",
    "gf": "Force", "kgf": "Force", "dyn": "Force",
    "shortTonF": "Force", "longTonF": "Force", "tonf": "Force",
    "kipf": "Force", "lbf": "Force", "ozf": "Force", "pdl": "Force",
    "A": "Current", "mA": "Current", "kA": "Current",
    "W": "Power", "kW": "Power", "MW": "Power", "GW": "Power", "mW": "Power",
    "VA": "Power", "hp_mech": "Power", "hp_ele": "Power", "hp_metric": "Power",
    "g/cm3": "Density", "kg/m3": "Density",
    "ms": "EngTime", "s": "EngTime", "minute": "EngTime", "hr": "EngTime", "day": "EngTime",
    "deg/s": "AngularRate", "rad/s": "AngularRate", "rpm": "AngularRate",
}

# getUnit 用のマッピング (互換性維持 — DataItem が使用)
_UNIT_OBJ_MAP = {
    "K": Temperature.Unit.K, "C": Temperature.Unit.C, "F": Temperature.Unit.F,
    "km": Length.Unit.km, "m": Length.Unit.m, "cm": Length.Unit.cm, "mm": Length.Unit.mm,
    "kg": Mass.Unit.kg, "g": Mass.Unit.g,
    "Pa": Pressure.Unit.Pa, "MPa": Pressure.Unit.MPa, "atm": Pressure.Unit.atm, "psi": Pressure.Unit.psi,
    "N": Force.Unit.N, "kN": Force.Unit.kN, "kgf": Force.Unit.kgf, "tonf": Force.Unit.tonf,
    "A": Current.Unit.A, "mA": Current.Unit.mA,
    "W": Power.Unit.W, "kW": Power.Unit.kW,
    "g/cm3": Density.Unit.g_cm3, "kg/m3": Density.Unit.kg_m3,
    "ms": EngTime.Unit.ms, "s": EngTime.Unit.s,
    "minute": EngTime.Unit.minute, "hr": EngTime.Unit.hr, "day": EngTime.Unit.day,
}


class UnitHandler:

    def __init__(self):
        pass

    @staticmethod
    def getUnit(unitName):
        """互換性維持: unitName から EngUnit オブジェクトを返す。"""
        return _UNIT_OBJ_MAP.get(unitName, None)

    @staticmethod
    def getUnitType(unitName):
        """互換性維持: unitName から単位カテゴリ名を返す。"""
        return _UNIT_TYPE_MAP.get(unitName, None)

    @staticmethod
    def parseStringToList(s=None):
        def convert_value(value):
            try:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value
        return [convert_value(x) for x in s.split(',')]

    @staticmethod
    def convertUnit(value=None, unitNameIn: str = None, unitNameOut: str = None):
        """pint をバックエンドにした単位変換。

        Args:
            value: 変換対象の値 (scalar, list, pandas Series, str)
            unitNameIn: 変換元の単位名
            unitNameOut: 変換先の単位名

        Returns:
            変換後の値
        """
        if unitNameIn == unitNameOut:
            return value

        if isinstance(value, str):
            value = UnitHandler.parseStringToList(s=value)

        pintIn = _PINT_ALIASES.get(unitNameIn, unitNameIn)
        pintOut = _PINT_ALIASES.get(unitNameOut, unitNameOut)

        if isinstance(value, (pd.core.series.Series, list)):
            values = list(value)
            result = [UnitHandler._convertScalar(v, pintIn, pintOut) for v in values]
            if len(result) == 1:
                return result[0]
            return result

        if isinstance(value, (float, int, np.float64)):
            return UnitHandler._convertScalar(value, pintIn, pintOut)

        return value

    @staticmethod
    def convertUnitForList(valueList=None, unitNameIn=None, unitNameOut=None):
        """リストの全要素を単位変換する。"""
        pintIn = _PINT_ALIASES.get(unitNameIn, unitNameIn)
        pintOut = _PINT_ALIASES.get(unitNameOut, unitNameOut)
        return [UnitHandler._convertScalar(v, pintIn, pintOut) for v in valueList]

    @staticmethod
    def _convertScalar(value, pintUnitIn, pintUnitOut):
        """スカラー値を pint で変換する。"""
        q = _ureg.Quantity(value, pintUnitIn)
        result = q.to(pintUnitOut).magnitude
        if isinstance(result, float) and result.is_integer() and isinstance(value, int):
            return int(result)
        return float(result)
