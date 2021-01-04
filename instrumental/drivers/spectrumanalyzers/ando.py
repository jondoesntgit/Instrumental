# -*- coding: utf-8  -*-
"""
Driver for ILX Lightwave Lasers
"""

from . import SpectrumAnalyzer
from .. import VisaMixin, SCPI_Facet
import numpy as np
from enum import Enum
from ... import Q_
import pint

_INST_PARAMS = ['visa_address']

class XUnit(Enum):
    wavelength = 0
    frequency = 1
    
class Trace(Enum):
    A = 0
    B = 1
    C = 2


class AQ6713B(SpectrumAnalyzer, VisaMixin):

    def _initialize(self):
        self._rsrc.read_termination = '\r\n'

    #center = SCPI_Facet('FREQ:CENT', units='Hz', convert=float)
    #span = SCPI_Facet('FREQ:SPAN', units='Hz', convert=float)
    #start = SCPI_Facet('FREQ:STAR', units='Hz', convert=float)
    #stop = SCPI_Facet('FREQ:STOP', units='Hz', convert=float)
    #reference = SCPI_Facet('DISPLAY:TRACE:Y:RLEVEL', convert=float)
    #sweep_time = SCPI_Facet('SWEEP:TIME', units='s', convert=float)
    #vbw = SCPI_Facet('BAND:VID', units='Hz', convert=float)
    #rbw = SCPI_Facet('BAND', units='Hz', convert=float)
    averages = SCPI_Facet('AVG', convert=int)
    samples = SCPI_Facet('SMPL', convert=int)

    #attenuation = SCPI_Facet('INP1:ATT')

    def get_trace(self, channel=1):
        """Get the trace for a given channel.

        Returns a tuple (frequencies, power)

        """
        data_string = self.query('TRAC? TRACE%i' % channel)
        power = np.array(data_string.split(',')).astype(float)
        frequency = np.linspace(self.start.m, self.stop.m, len(power))
        return frequency, power

    @property
    def x_unit(self):
        x_unit = float(self._rsrc.query('XUNT?'))
        return XUnit(x_unit)
    
    @x_unit.setter
    def x_unit(self, x_unit):

        if x_unit in [0, 1]:
            index = x_unit
        elif x_unit in [e.name for e in XUnit]:
            index = XUnit[x_unit].value
        else:
            raise ValueError("New value must be either 'frequency' or 'wavelength'")
        self._rsrc.write('XUNT%i' % index)
        
    @property
    def span(self):
        if self.x_unit == XUnit['frequency']:
            return Q_(self._rsrc.query('SPANF?'), 'THz')
        elif self.x_unit == XUnit['wavelength']:
            return Q_(self._rsrc.query('SPAN?'), 'nm')
        else:
            raise NotImplemented('Device is neither in frequency nor wavelength mode')

    @span.setter
    def span(self, span):
        span = Q_(span)
        if span.check('[frequency]'):
            self._rsrc.write('SPANF %07.3f' % span.to('THz').m)
        elif span.check('[length]'):
            self._rsrc.write('SPAN %06.1f' % span.to('nm').m)
        else:
            if self.x_unit == XUnit['frequency']:
                target_unit = u.THz
            else:
                target_unit = u.nm
            raise pint.DimensionalityError(span.units, target_unit)
            
    @property
    def active_trace(self):
        return Trace(int(self._rsrc.query('ACTV?'))).name
    
    @active_trace.setter
    def active_trace(self, val):
        if val in 'ABC':
            self.write('ACTV' + val)
        else:
            raise ValueError("'Trace must be either 'A', 'B', or 'C'")
            
    def get_wavelengths(self, trace=None):
        """
        Returns the measured spectrum from a single reading of the instrument.
        Aliases to acquire
        Returns:
            tuple of arrays:
                The first array contains the wavelengths in nanometers.
                The second array contains the optical power in dBm.
        """
        cached_timeout = self._rsrc.timeout
        self._rsrc.timeout = 3000
        if trace is None:
            trace = self.active_trace
        if trace not in 'ABC':
            raise ValueError("Trace must be either 'A', 'B' or 'C'")

        wavelength_string = self._rsrc.query('WDAT%s' % trace)
        wavelength = np.array(wavelength_string[:-2].split(','))
        wavelength = wavelength.astype(np.float)[2:]
        
        if self.x_unit == XUnit['frequency']:
            wavelength = 299792458 / wavelength * 1e-3
        
        self._rsrc.timeout = cached_timeout
        
        return wavelength

    def get_amplitudes(self, trace=None):
        """
        Returns the measured spectrum from a single reading of the instrument.
        Aliases to acquire
        Returns:
            tuple of arrays:
                The first array contains the wavelengths in nanometers.
                The second array contains the optical power in dBm.
        """
        cached_timeout = self._rsrc.timeout
        self._rsrc.timeout = 3000
        if trace is None:
            trace = self.active_trace
        if trace not in 'ABC':
            raise ValueError("Trace must be either 'A', 'B' or 'C'")
        power_string = self._rsrc.query('LDAT%s' % trace)
        power = np.array(power_string[:-2].split(','))
        power = power.astype(np.float)[2:]
        
        self._rsrc.timeout = cached_timeout
        
        return power
        # pint doesn't have units for dBm

    def get_spectrum(self, trace=None):
        """
        Returns the measured spectrum from a single reading of the instrument.
        Aliases to acquire
        Returns:
            tuple of arrays:
                The first array contains the wavelengths in nanometers.
                The second array contains the optical power in dBm.
        """
        cached_timeout = self._rsrc.timeout
        self._rsrc.timeout = 3000
        if trace is None:
            trace = self.active_trace
        if trace not in 'ABC':
            raise ValueError("Trace must be either 'A', 'B' or 'C'")
        power_string = self._rsrc.query('LDAT%s' % trace)
        power = np.array(power_string.split(','))
        power = power.astype(np.float)[2:]

        wavelength_string = self._rsrc.query('WDAT%s' % trace)
        wavelength = np.array(wavelength_string.split(','))
        wavelength = wavelength.astype(np.float)[2:]
        
        if self.x_unit == XUnit['frequency']:
            wavelength = 299792458 / wavelength * 1e-3
        
        self._rsrc.timeout = cached_timeout
        
        return wavelength, power
        # pint doesn't have units for dBm

    def is_scanning(self):
        scan_code = int(self._rsrc.query('SWEEP?'))
        # 0 : stopped
        # 1 : single
        # 2 : repeat
        # 3 : auto
        # 4 : segment measure
        # 11: WL Cal
        # 12: Optical alignment
        return scan_code != 0

    def single(self):
        self._rsrc.write('SGL')

    def repeat(self):
        self._rsrc.write('RPT')
    
    def stop(self):
        self._rsrc.write('STP')


    # .01 to 2.0
    @property
    def resolution(self):
        resolution_nm = self._rsrc.query('RESLN?')
        return Q_(resolution_nm, 'nm')

    @resolution.setter
    def resolution(self, val):
        if isinstance(val, str):
            val = Q_(val)
        
        if val.check('nm'):
            val = val.to('nm')
            if val.m in [.01, .02, .05, .1, .2, .5, 1.0, 2.0]:
                self._rsrc.write('RESLN%.2f' % val.m)
            else: 
                raise ValueError('Resolution must be in [.01, .02, .05, .1, .2, .5, 1.0, 2.0] nm')
        elif val.check('GHz'):
            val = val.to('GHz')
            if val.m in [2, 4, 10, 20, 40, 100, 200, 400]:
                self._rsrc.write('RESLNF%i' % val.m)
            else:
                raise ValueError('Resolution must be in [2, 4, 10, 20, 40, 100, 200, 400] GHz')

        else:
            raise ValueError('Must pass resolution that is either GHz or nm')