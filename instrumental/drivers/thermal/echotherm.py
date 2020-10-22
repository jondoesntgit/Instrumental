# -*- coding: utf-8 -*-
"""
Drive for controlling Echotherm IN30

A manual can be found here:
http://www.levanchimica.it/assets/Uploads/catalog/article/attachments/1173/MAN-IN30-IN40.pdf

"""

from . import Thermal
from .. import ParamSet
from ...log import get_logger
from .. import VisaMixin
from ..util import check_units
from ... import u
import visa

log = get_logger(__name__)
__all__ = ['IN30']

class IN30(Thermal):
    """Class for controlling Echotherm IN30.

    """
    _INST_PARAMS_ = ['serial']

    def _initialize(self):
        self.serial = self._paramset['serial']
        self._rsrc = visa.ResourceManager().open_resource(
                self.serial,
                read_termination = '\r',
                write_termination='\r')
        self._rsrc.timeout = 1000
        self._rsrc.baud_rate = 2400


    @property
    def temperature(self):
        self._rsrc.write('a')
        raw_temperature = self._rsrc.read()
        if raw_temperature == 'Command Failed':
            # Try again
            self._rsrc.write('a')
            raw_temperature = self._rsrc.read()
        return float(raw_temperature)

    @temperature.setter
    def temperature(self, value):
        self._rsrc.write('A %04.1f' % value)
        response = self._rsrc.read('\r')
        assert response == 'Command OK'
        

