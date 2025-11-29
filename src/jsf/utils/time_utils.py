"""Time utilities for JSF-Core."""

from datetime import datetime, date, timedelta
from typing import Union, Optional
import pandas as pd


DateLike = Union[str, datetime, date, pd.Timestamp]


def parse_date(date_input: DateLike) -> pd.Timestamp:
    """
    Parse various date formats into a pandas Timestamp.

    Args:
        date_input: Date in various formats (string, datetime, date, Timestamp)

    Returns:
        Parsed date as pandas Timestamp

    Examples:
        >>> parse_date("2020-01-01")
        Timestamp('2020-01-01 00:00:00')
        >>> parse_date(datetime(2020, 1, 1))
        Timestamp('2020-01-01 00:00:00')
    """
    if isinstance(date_input, pd.Timestamp):
        return date_input
    return pd.Timestamp(date_input)


def date_range(
    start: DateLike,
    end: DateLike,
    freq: str = "D",
    inclusive: str = "both",
) -> pd.DatetimeIndex:
    """
    Generate a date range.

    Args:
        start: Start date
        end: End date
        freq: Frequency string ('D' for daily, 'B' for business days, etc.)
        inclusive: Include boundaries ('both', 'left', 'right', 'neither')

    Returns:
        DatetimeIndex of dates
    """
    start_ts = parse_date(start)
    end_ts = parse_date(end)
    return pd.date_range(start=start_ts, end=end_ts, freq=freq, inclusive=inclusive)


def business_days_between(start: DateLike, end: DateLike) -> int:
    """
    Count business days between two dates.

    Args:
        start: Start date
        end: End date

    Returns:
        Number of business days
    """
    start_ts = parse_date(start)
    end_ts = parse_date(end)
    return len(pd.bdate_range(start=start_ts, end=end_ts))


def offset_date(
    base_date: DateLike,
    days: Optional[int] = None,
    weeks: Optional[int] = None,
    months: Optional[int] = None,
    years: Optional[int] = None,
) -> pd.Timestamp:
    """
    Offset a date by specified time periods.

    Args:
        base_date: Base date
        days: Number of days to offset
        weeks: Number of weeks to offset
        months: Number of months to offset
        years: Number of years to offset

    Returns:
        Offset date
    """
    base_ts = parse_date(base_date)
    
    if days:
        base_ts += pd.Timedelta(days=days)
    if weeks:
        base_ts += pd.Timedelta(weeks=weeks)
    if months:
        base_ts += pd.DateOffset(months=months)
    if years:
        base_ts += pd.DateOffset(years=years)
    
    return base_ts


def to_utc(dt: DateLike) -> pd.Timestamp:
    """
    Convert date to UTC timezone.

    Args:
        dt: Date to convert

    Returns:
        Date in UTC timezone
    """
    ts = parse_date(dt)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def is_business_day(dt: DateLike) -> bool:
    """
    Check if a date is a business day (Monday-Friday).

    Args:
        dt: Date to check

    Returns:
        True if business day, False otherwise
    """
    ts = parse_date(dt)
    return ts.weekday() < 5
