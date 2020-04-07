import time
import subprocess
import re

from .. import ParamSet
from . import DAQ

import numpy as np
from nicelib import (NiceLib, load_lib, RetHandler,
        Sig, NiceObject, sig_pattern) # req: nicelib >= 0.5

def list_instruments():
    list_devices_program = '/etc/natinst/nidaqmxbase/bin/daqmxbase_listdevices'
    raw_string = subprocess.check_output(list_devices_program).decode('utf-8')
    paramsets = []
    for line in raw_string.split('\n'):
        print('newline', line)
        pattern = '([^:]*):"([^"]*)".*\((.*)\).*'
        p = re.match(pattern, line)
        if p is None: continue
        model, dev_name, port = p.groups()
        print(model)
        paramset = ParamSet(NIDAQBase,
                name=dev_name,
                model=model)
        paramsets.append(paramset)
    return paramsets

class NiceNIBase():#NiceLib):
    #_info_ = load_lib('ni', __package__)
    _prefix_ = ('DAQmxBase_', 'DAQmxBase', 'DAQmx_', 'DAQmx')
    _buflen_ = 1024
    _use_numpy = True
    #_ret_ = ret_errcheck

    class Task(NiceObject):
        """A Nice-wrapped NI Task"""
        pass
        """
        _init_ = 'CreateTask'

        StartTask = Sig('in')
        StopTask = Sig('in')
        ClearTask = Sig('in')

        IsTaskDone = Sig('in', 'out')
        CreateAIVoltageChan = Sig('in', 'in', 'in', 'in', 'in', 'in', 'in', 'in')
        CreateAOVoltageChan = Sig('in', 'in', 'in', 'in', 'in', 'in', 'in')
        CreateDIChan = Sig('in', 'in', 'in', 'in')
        CreateDOChan = Sig('in', 'in', 'in', 'in')
        ReadAnalogF64 = Sig('in', 'in', 'in', 'in', 'arr', 'len=in', 'out', 'ignore')
        ReadDigitalScalarU32 = Sig('in', 'in', 'out', 'ignore')
        ReadDigitalU32 = Sig('in', 'in', 'in', 'in', 'arr', 'len=in', 'out', 'ignore')
        WriteAnalogF64 = Sig('in', 'in', 'in', 'in', 'in', 'in', 'out', 'ignore')
        WriteDigitalU32 = Sig('in', 'in', 'in', 'in', 'in', 'in', 'out', 'ignore')
        WriteDigitalScalarU32 = Sig('in', 'in', 'in', 'in', 'ignore')
        CfgSampClkTiming = Sig('in', 'in', 'in', 'in', 'in', 'in')
        CfgImplicitTiming = Sig('in', 'in', 'in')
        CfgAnlgEdgeStartTrig = Sig('in', 'in', 'in', 'in')
        CfgDigEdgeStartTrig = Sig('in', 'in', 'in')
        CfgDigEdgeRefTrig = Sig('in', 'in', 'in', 'in')
        """

class NIDAQBase(DAQ):
    _INST_PARAMS_ = ['name', 'model']

    #mx = NiceNIBase
#    Task = Task

    def _initialize(self):
        self.name = self._paramset['name']
        self._dev = self._paramset['model']
        #self.mx.Device(self.name)
        # TODO: Load analog channels
        # Todo Load internal channels
        #s TODO load digital ports

