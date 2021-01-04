# -*- coding: utf-8 -*-
# Copyright 2021 Jonathan Wheeler
"""
Driver module for Hewlett Packard oscilloscopes. Currently supports

* 54520 and 54540 series oscilloscopes
"""
import visa
from pyvisa.constants import InterfaceType
import numpy as np
from pint import UndefinedUnitError
from . import Scope
from .. import VisaMixin, SCPI_Facet, Facet
from ..util import visa_context
from ... import u, Q_
from enum import Enum

MODEL_CHANNELS = {
    'DSO1024': 4,
}

class PointsMode(Enum):
    normal = 'NORMal'
    maximum = 'MAXimum'
    raw = 'RAW'

# This code is not used right now, it was a preliminary effort to do things the
# "clean" way...
class DataFormat(Enum):
    word = 'WORD'
    byte = 'BYTE'
    ascii = 'ASCii'

def infer_termination(msg_str):
    if msg_str.endswith('\r\n'):
        return '\r\n'
    elif msg_str.endswith('\r'):
        return '\r'
    elif msg_str.endswith('\n'):
        return '\n'
    return None

class HPScope(Scope, VisaMixin):
    """
    A base class for Hewlett-Packard Scopes
    """
    
    def _initialize(self):
        msg = self.query('*IDN?')
        self._rsrc.read_termination = infer_termination(msg)
        
    points_mode = SCPI_Facet(':WAVeform:POINts:MODE', convert=PointsMode,
                                 doc="sets the type of data returned by the :WAV:DATA? query")
    points = SCPI_Facet(':WAVeform:POINts', convert=int,
                        doc="Number of points to return on :WAV:DATA?")
    
    format = SCPI_Facet(':WAVeform:FORMat',
                        doc="Format of the returned data")

    def run(self):
        """Performs continuous acquisitions"""
        self._rsrc.write(':RUN')

    def stop(self):
        """Stops continuous acquisitions"""
        self._rsrc.write(':STOP')

    def single(self):
        """Runs a single acquisition"""
        self._rsrc.write(':SINGLE')

    def capture(self, channel):
        """ Returns unitful arrays for (time, voltage)"""
        self.write(':WAVeform:SOURce CHAN%i' % channel)
        preamble = self.query(':WAVeform:PREAMBLE?')
        (fmt, typ, num_points, _, 
            x_increment, x_origin, x_reference,
            y_increment, y_origin, y_reference,
        ) = np.fromstring(preamble, sep=',')
        
        t = x_origin + (np.arange(num_points) * x_increment) * u.second
        
        if fmt == 0:
            # WORD
            dtype = 'h'
        if fmt == 1:
            # BYTE
            dtype = 'B'
        if fmt >= 0 and fmt <= 1:
            data = self._rsrc.query_binary_values(':WAVeform:DATA? CHAN%i' % channel, datatype=dtype, is_big_endian=True)
            data = (y_reference - data) * y_increment - y_origin
            return t, data * u.volt
        if fmt == 2:
            # ASCII
            data = self.query(':WAVeform:DATA? CHAN%i' % channel)
            data = np.fromstring(data[13:], sep=',') # Take off some header information
            return t, data * u.volt

class Preamble():
    def __init__(self, string):
        preamble_strings = string.split(',')

        self.string = string
        self.format = DataFormat(int(preamble_strings[0]))
        self.type = DataType(int(preamble_strings[1]))
        self.points = int(preamble_strings[2])
        self.count = int(preamble_strings[3])
        self.xincrement = float(preamble_strings[4])
        self.xorigin = float(preamble_strings[5])
        self.xreference = float(preamble_strings[6])
        self.yincrement = float(preamble_strings[7])
        self.yorigin = float(preamble_strings[8])
        self.yreference = float(preamble_strings[9])

    def __str__(self):
        return self.string
        
    def get_xdata(self):
        return (np.arange(self.points) - self.xreference) * self.xincrement + self.xorigin
    
    def get_ydata(self, data_values):
        # TODO: Only works for COMP style data
        return_values = (data_values - self.yreference) * self.yincrement + self.yorigin
        return_values[data_values == 255] = np.nan
        return return_values

class DataFormat(Enum):
    ASCII = 0
    BYTE = 1
    WORD = 2
    COMP = 4

class DataType(Enum):
    INVALID = 0
    REPETITIVE_NORMAL = 1
    REPETITIVE_AVERAGE = 2
    REPETITIVE_ENVELOPE = 3
    RAW_DATA = 4
    PDETECT = 5

class HP5420A(HPScope):

    _INST_PARAMS_ = ['visa_address']
    _INST_VISA_INFO_ = ('HEWLETT-PACKARD', ['54520A'])

    def preamble(self):
        return Preamble(self.query(':WAVeform:PREAMBLE?'))

    def quick_capture(self, data_format=None):
        if data_format is None:
            data_format = self._rsrc.query(':WAVEFORM:FORMAT?')
            data_format = DataFormat[data_format]

        if data_format is DataFormat.COMP:
            self._rsrc.write('WAVEFORM:DATA?')
            data = self._rsrc.read_raw()
            header_length = int(chr(data[1]))
            payload_length = int(data[2:2+header_length])
            assert len(data) == (
                1 # Initial #
                + 1 # Integer representing header length
                + header_length # String representing payload length
                + payload_length # The data
                + 1 # Termination character
            )

            return np.frombuffer(data[2 + header_length:-1], dtype=np.uint8)

        else:
            raise NotImplementedError('Quick capture is not implemented for %s' % data_format)
            
