# -*- coding: utf-8 -*-
# Copyright 2020 Jonathan Wheeler
"""
Driver module for Marconi signal generators.
"""
import re
from . import FunctionGenerator
from .. import VisaMixin
from ... import Q_, u

unit_table = {
    'MZ': u.MHz,
    'KZ': u.kHz,
    'HZ': u.Hz,
    #'DB': u.dBm,
    'VL': u.V,
    'MV': u.mV,
    'UV': u.uV
}    


class Marconi2022(FunctionGenerator, VisaMixin):

    _INST_PARAMS_ = ['visa_address']    
    
    def _initialize(self):
        self._rsrc.read_termination = '\r\n'
        self._rsrc.timeout = 3000 # It's a slow device
        self._current_front_panel = None
        
    @property
    def current_front_panel(self):
        """Returns the front panel display type
        
        This driver is for a slow machine. The machine emulates button presses
        with each GPIB command. Given the slowness, it is faster to cache
        the front panel display than to set it redundantly with every request
        """
        if self._current_front_panel:
            return self._current_front_panel
        # Write to carrier frequency
        self.write('CF')
        self._current_front_panel = 'CF'
        return 'CF'
    
    def set_front_panel(self, command):
        if command != self._current_front_panel:
            self._rsrc.write(command)
            
    def query(self, command):
        self.set_front_panel(command)
        return self._rsrc.query('QU').strip()
    
    @property
    def frequency(self):
        """Query the current frequency"""
        raw_string = self.query('CF')
        recipe = '([A-Z]+)\s+([0-9.]+)([A-Z]{2})([A-Z]{2})'
        p = re.match(recipe, raw_string)
        command, magnitude, unit, reference_type = p.groups()
        return Q_(float(magnitude), unit_table[unit])
        
        # Reference types (for future reference)
        #IS : Internal Frequency Standard
        #XS : External Frequency Standard

    @frequency.setter
    def frequency(self, value):
        
        value = Q_(value)
        value.check(['frequency'])
        
        if value > Q_(1000, 'MHz'):
            raise ValueError('Requested value exeeds maximum frequency')
        
        units_to_cmd_dict = {
            u.Hz: 'HZ',
            u.kHz: 'KZ',
            u.MHz: 'MZ'
        }
        
        try:
            command = 'CF %7g %s' % (value.m, units_to_cmd_dict[value.u]) 
        except KeyError:
            raise ValueError('Please pass a value in Hz, kHz, or MHz') 
        self._current_front_panel = 'CF'
        self._rsrc.write(command)

        
    @property
    def level(self):
        """Query the current level"""
        
        raw_string = self.query('LV')
        recipe = '([A-Z]+)\s*(-?[0-9.]+)([A-Z]{2})C([0-1])'
        p = re.match(recipe, raw_string)
        command, magnitude, unit, is_carrier_on = p.groups()
        
        magnitude = float(magnitude)
                
        #C0 : Carrier off
        #C1 : Carrier on
        is_carrier_on = is_carrier_on == '1'
        
        if unit == 'DB':
            power = 1e-3 * 10**(magnitude / 10) # watts
            resistance = 50 # ohm
            # PR = V^2, Scale by sqrt(2) to convert rms to peak
            return Q_((power * resistance * 2)**.5, 'V')
        else:
            return Q_(magnitude, unit_table[unit])
        
    @level.setter
    def level(self, value):
        
        if 'dBm' in value:
            magnitude = float(value.strip().split('dBm')[0])
            power = 1e-3 * 10**(magnitude / 10) # watts
            resistance = 50 # ohm
            voltage = Q_((power * resistance * 2)**.5)
            if voltage > 1:
                command = 'LV %.4g VL' % voltage
            elif voltage > 1e-3:
                command = 'LV %.4g MV' % (voltage * 1e3)               
            else:
                command = 'LV %.4g UV' % (voltage * 1e6)               
            self._current_front_panel = 'LV'
            print(command)
            self._rsrc.write(command)
            return
            
        value = Q_(value)
        value.check(['voltage'])
        units_to_cmd_dict = {
            u.uV: 'UV',
            u.mV: 'MV',
            u.V: 'VL'
        }
        command = 'LV %.4g %s' % (value.m, units_to_cmd_dict[value.u])
        self._rsrc.write(command)
        
    @property
    def output(self):
        """Returns true if the carrier is on, false otherwise."""
        return self.query('LV').strip()[-1] == '1'
    
    @output.setter
    def output(self, value):
        if not isinstance(value, bool):
            raise ValueError('Output must either be True or False')
        self._current_front_panel = 'LV'
        self._rsrc.write('LV C%i' % (int(value)))