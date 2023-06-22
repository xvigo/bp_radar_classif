"""
Author: Vilem Gottwald

Module for named tuples representing radar data types.
"""

from typing import NamedTuple


class FrameHeader(NamedTuple):
    """
    Radar frame header named tuple.
    """

    magicWord: int
    version: int
    timestamp: int
    totalFrameLen: int
    frameNumber: int
    subFrameNumber: int
    reserved: int
    numTLVs: int
    checksum: int


class TlvHeader(NamedTuple):
    """
    TLV header named tuple containing the type and length.
    """

    type: int
    length: int


class Target(NamedTuple):
    """
    Radar target named tuple.
    """

    tid: int
    posX: float
    posY: float
    posZ: float
    velX: float
    velY: float
    velZ: float
    accX: float
    accY: float
    accZ: float
    dimX: float
    dimY: float
    dimZ: float
    pointCount: int
    tickCount: int
    state: int
    frame: int


class Point(NamedTuple):
    """
    Radar point named tuple.
    """

    range: float
    azimuth: float
    elevation: float
    doppler: float
    targetID: int
    snr: int
    noise: int
    frame: int


class Stats(NamedTuple):
    """Radar platform stats named tuple"""

    temperature: float
    voltageAvg: float
    powerAvg: float
    powerPeak: float
    pointsCount: int
    tracksCount: int
