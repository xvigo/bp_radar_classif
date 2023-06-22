""" 
Author: Vilem Gottwald

Module for converting datetime to string and vice versa.
"""

from datetime import datetime


def dt2str(datetime_timestamp: datetime) -> str:
    """Convert datetime to string with %Y%m%dT%H%M%S%f formats and return it.

    :param datetime_timestamp: datetime object
    :return: string with format %Y%m%dT%H%M%S%f
    """
    return datetime_timestamp.strftime("%Y%m%dT%H%M%S%f")


def str2dt(str_timestamp: str) -> datetime:
    """Convert string with format %Y%m%dT%H%M%S%f to datetime and return it.

    :param str_timestamp: string with format %Y%m%dT%H%M%S%f
    :return: datetime object
    """
    return datetime.strptime(str_timestamp, "%Y%m%dT%H%M%S%f")
