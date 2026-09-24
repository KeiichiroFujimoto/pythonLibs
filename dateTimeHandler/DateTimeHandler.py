from datetime import datetime
from datetime import timezone, timedelta

class DateTimeHandler():
    
    @staticmethod
    def parse(dateTimeString=None):
        from dateutil import parser
        dt = parser.parse(dateTimeString)
        return dt
    
    @staticmethod
    def getDateTimeCurrent():
        """
        Returns the current time in both UTC and JST (Japan Standard Time).

        This function retrieves the current time in UTC and converts it to JST (UTC+9).
        The returned times are formatted as strings including date, time, and timezone.

        Returns:
            dict: A dictionary containing:
                - "UTC" (str): Current time in UTC.
                - "JST" (str): Current time in Japan Standard Time (UTC+9).
        """
        # UTC
        now_utc = datetime.now(timezone.utc)

        # JST (UTC+9)
        jst = timezone(timedelta(hours=9))
        now_jst = now_utc.astimezone(jst)

        return {
            "UTC": now_utc.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "JST": now_jst.strftime("%Y-%m-%d %H:%M:%S %Z")
        }

    @staticmethod
    def to_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def to_jst(dt: datetime) -> datetime:
        jst = timezone(timedelta(hours=9))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(jst)

    @staticmethod
    def to_unix_seconds(dt: datetime) -> float:
        dt_utc = DateTimeHandler.to_utc(dt)
        return dt_utc.timestamp()

    @staticmethod
    def from_unix_seconds(ts: float, tz: timezone = timezone.utc) -> datetime:
        return datetime.fromtimestamp(ts, tz=tz)

    @staticmethod
    def to_iso(dt: datetime) -> str:
        dt_utc = DateTimeHandler.to_utc(dt)
        return dt_utc.isoformat()

    @staticmethod
    def from_iso(iso_str: str) -> datetime:
        return datetime.fromisoformat(iso_str)

    @staticmethod
    def to_julian_date(dt: datetime) -> float:
        # Algorithm valid for Gregorian calendar; assumes UTC if naive.
        dt_utc = DateTimeHandler.to_utc(dt)
        year = dt_utc.year
        month = dt_utc.month
        day = dt_utc.day
        hour = dt_utc.hour
        minute = dt_utc.minute
        second = dt_utc.second + dt_utc.microsecond / 1_000_000.0

        if month <= 2:
            year -= 1
            month += 12
        a = year // 100
        b = 2 - a + (a // 4)
        jd_day = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
        jd_frac = (hour + minute / 60.0 + second / 3600.0) / 24.0
        return jd_day + jd_frac

    @staticmethod
    def from_julian_date(jd: float, tz: timezone = timezone.utc) -> datetime:
        # Inverse of to_julian_date; returns timezone-aware datetime.
        jd += 0.5
        z = int(jd)
        f = jd - z
        if z < 2299161:
            a = z
        else:
            alpha = int((z - 1867216.25) / 36524.25)
            a = z + 1 + alpha - int(alpha / 4)
        b = a + 1524
        c = int((b - 122.1) / 365.25)
        d = int(365.25 * c)
        e = int((b - d) / 30.6001)
        day = b - d - int(30.6001 * e) + f
        month = e - 1 if e < 14 else e - 13
        year = c - 4716 if month > 2 else c - 4715

        day_int = int(day)
        frac = day - day_int
        hours = frac * 24.0
        hour = int(hours)
        minutes = (hours - hour) * 60.0
        minute = int(minutes)
        seconds = (minutes - minute) * 60.0
        second = int(seconds)
        microsecond = int((seconds - second) * 1_000_000.0)

        return datetime(year, month, day_int, hour, minute, second, microsecond, tzinfo=tz)

class DateTimeConditionChecker():

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def getOverlap(timeStart1=None,timeEnd1=None,timeStart2=None,timeEnd2=None,returnType="seconds"):
        from collections import namedtuple
        Range = namedtuple('Range', ['start', 'end'])
        r1 = Range(start=timeStart1, end=timeEnd1)
        r2 = Range(start=timeStart2, end=timeEnd2)
        latest_start = max(r1.start, r2.start)
        earliest_end = min(r1.end, r2.end)

        if returnType == "seconds":
            delta = (earliest_end - latest_start).seconds + 1
        else:
            delta = None
        
        overlap = max(0, delta)
        return overlap


if __name__ == "__main__":
    # Quick usage examples
    now = datetime.now(timezone.utc)
    print("UTC:", DateTimeHandler.to_utc(now))
    print("JST:", DateTimeHandler.to_jst(now))
    print("Unix:", DateTimeHandler.to_unix_seconds(now))
    print("ISO:", DateTimeHandler.to_iso(now))
    jd = DateTimeHandler.to_julian_date(now)
    print("JD:", jd)
    print("From JD:", DateTimeHandler.from_julian_date(jd))
