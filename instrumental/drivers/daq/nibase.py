import PyDAQmx
import weakref
from enum import Enum, EnumMeta
import sys
from ..util import check_units, check_enums, as_enum
import subprocess
import re
import time
from past.builtins import unicode, basestring

import numpy as np
from ctypes import byref

from ...errors import Error, TimeoutError
from ... import Q_, u
from .. import ParamSet
from . import DAQ

def list_instruments():
    list_devices_program = '/etc/natinst/nidaqmxbase/bin/daqmxbase_listdevices'
    raw_string = subprocess.check_output(list_devices_program).decode('utf-8')
    paramsets = []
    for line in raw_string.split('\n'):
        pattern = '([^:]*):"([^"]*)".*\((.*)\).*'
        p = re.match(pattern, line)
        if p is None: continue
        model, dev_name, port = p.groups()
        paramset = ParamSet(NIDAQBase,
                device=dev_name,
                model=model)
        paramsets.append(paramset)
    return paramsets

class DAQError(Error):
    def __init__(self, code):
        msg = "({}) {}".format(code, NicePyDAQmx.GetExtendedErrorInfo())
        self.code = code
        super(DAQError, self).__init__(msg)

def to_bytes(value, codec='utf-8'):
        """Encode a unicode string as bytes or pass through an existing bytes object"""
        if isinstance(value, bytes):
            return value
        elif isinstance(value, unicode):
            value.encode(codec)
        else:
            return bytes(value)

# TODO: Move to shared library
class ValEnumMeta(EnumMeta):
    """Enum metaclass that looks up values and removes undefined members"""
    @classmethod
    def __prepare__(metacls, cls, bases, **kwds):
        return {}

    def __init__(cls, *args, **kwds):
        super(ValEnumMeta, cls).__init__(*args)

    def __new__(metacls, cls, bases, clsdict, **kwds):
        # Look up values, exclude nonexistent ones
        for name, value in list(clsdict.items()):
            try:
                clsdict[name] = getattr(Val, value)
            except AttributeError:
                del clsdict[name]

        enum_dict = super(ValEnumMeta, metacls).__prepare__(cls, bases, **kwds)
        # Must add members this way because _EnumDict.update() doesn't do everything needed
        for name, value in clsdict.items():
            enum_dict[name] = value
        return super(ValEnumMeta, metacls).__new__(metacls, cls, bases, enum_dict, **kwds)

# TODO: Move to shared librar
if 'sphinx' in sys.modules:
    # Use mock class to allow sphinx to import this module
    class Values(object):
        def __getattr__(self, name):
            return name
else:
    class Values(object):
        pass

class NicePyDAQmx:

    # Simplify and tdo not rely on PyDAQmx Task
    #class Task():
    #    def __init__(self, task_name):
    class Task(PyDAQmx.Task):
        
        def ReadAnalog64(self, samples, timeout_s, group_by, buf_size):
            data = np.zeros((buf_size,), dtype=np.float64)
            read = PyDAQmx.int32()
            self.ReadAnalogF64(samples, timeout_s, group_by, data, buf_size, byref(read), None)
            return data, read
        #data, n_samples_read = self._mx_task.ReadAnalog64(samples, timeout_s, Val.GroupByChannel, buf_size)

Val = Values()
for name, attr in PyDAQmx.__dict__.items():
    if name.startswith('Val_'):
        setattr(Val, name[4:], attr)

        

ValEnum = ValEnumMeta('ValEnum', (Enum,), {})

# TODO Move to a shared file
class SampleMode(ValEnum):
    finite = 'FiniteSamps'
    continuous = 'ContSamps'
    hwtimed = 'HWTimedSinglePoint'

# TODO: Move to shared library
class SampleTiming(ValEnum):
    sample_clk = 'SampClk'
    burst_handshake = 'BurstHandshake'
    handshake = 'Handshake'
    on_demand = 'OnDemand'
    change_detection = 'ChangeDetection'
    pipelined_sample_clk = 'PipelinedSampClk'


# TODO: Move to shared library
class EdgeSlope(ValEnum):
    rising = 'RisingSlope'
    falling = 'FallingSlope'


# TODO: Move to shared library
class TerminalConfig(ValEnum):
    default = 'Cfg_Default'
    RSE = 'RSE'
    NRSE = 'NRSE'
    diff = 'Diff'
    pseudo_diff = 'PseudoDiff'


# TODO: Move to shared library
class RelativeTo(ValEnum):
    FirstSample = 'FirstSample'
    CurrReadPos = 'CurrReadPos'
    RefTrig = 'RefTrig'
    FirstPretrigSamp = 'FirstPretrigSamp'
    MostRecentSamp = 'MostRecentSamp'


# TODO: Move to shared library
class ProductCategory(ValEnum):
    MSeriesDAQ = 'MSeriesDAQ'
    XSeriesDAQ = 'XSeriesDAQ'
    ESeriesDAQ = 'ESeriesDAQ'
    SSeriesDAQ = 'SSeriesDAQ'
    BSeriesDAQ = 'BSeriesDAQ'
    SCSeriesDAQ = 'SCSeriesDAQ'
    USBDAQ = 'USBDAQ'
    AOSeries = 'AOSeries'
    DigitalIO = 'DigitalIO'
    TIOSeries = 'TIOSeries'
    DynamicSignalAcquisition = 'DynamicSignalAcquisition'
    Switches = 'Switches'
    CompactDAQChassis = 'CompactDAQChassis'
    CSeriesModule = 'CSeriesModule'
    SCXIModule = 'SCXIModule'
    SCCConnectorBlock = 'SCCConnectorBlock'
    SCCModule = 'SCCModule'
    NIELVIS = 'NIELVIS'
    NetworkDAQ = 'NetworkDAQ'
    SCExpress = 'SCExpress'
    Unknown = 'Unknown'


# TODO: Move to shared library
def mk_property(name, conv_in, conv_out, doc=None):
    getter_name = 'Get' + name
    setter_name = 'Set' + name

    def fget(mtask):
        getter = getattr(mtask._mx_task, getter_name)
        return conv_out(getter())

    def fset(mtask, value):
        setter = getattr(mtask._mx_task, setter_name)
        setter(conv_in(value))

    return property(fget, fset, doc=doc)

# TODO: Move to shared library
def enum_property(name, enum_type, doc=None):
        return mk_property(name, conv_in=lambda x: as_enum(enum_type, x).value,
            conv_out=enum_type, doc=doc)


        def int_property(name, doc=None):
                return mk_property(name, conv_in=int, conv_out=int, doc=doc)

# TODO: Move to a shared library
def enum_property(name, enum_type, doc=None):
    return mk_property(name, conv_in=lambda x: as_enum(enum_type, x).value,
            conv_out=enum_type, doc=doc)

# TODO Move to a shared library
def int_property(name, doc=None):
    return mk_property(name, conv_in=int, conv_out=int, doc=doc)

# TODO move to a shared library
@check_units(duration='?s', fsamp='?Hz')
def handle_timing_params(duration, fsamp, n_samples):
    if duration is not None:
        if fsamp is not None:
            n_samples = int((duration * fsamp).to('')) # Exclude endpoint
        elif n_samples is not None:
            if n_samples <= 0:
                raise ValueError("`n_samples` must be greater than zero")
            fsamp = (n_samples - 1) / duration
    return fsamp, n_samples

def num_not_none(*args):
    return sum(int(arg is not None) for arg in args)


class Task(object):
    """A high-level task that can synchronize use of multiple channel types.

    Note that true DAQmx tasks can only include one type of channel (e.g., AI).
    To run multiple synchronized read/writes, we need to make one Minitask for
    each type, then use the same sample clock for each.
    """

    # TODO: Move to shared file
    def __init__(self, *channels):
        """Create a task that uses the given channels.

        Each arg can either be a Channel or a tuple of (Channel, name_str)
        """
        self._trig_set_up = False
        self.fsamp = None
        self.n_samples = 1
        self.is_scalar = True

        self.channels = OrderedDict()
        self._mtasks = {}
        self.AOs, self.AIs, self.DOs, self.DIs, self.COs, self.CIs = [], [], [], [], [], []
        TYPED_CHANNELS = {'AO': self.AOs, 'AI': self.AIs, 'DO': self.DOs,
                'DI':self.DIs, 'CO': self.COs, 'CI': self.CIs}

        for arg in channels:
            if isinstance(arg, Channel):
                channel = arg
                name = channel.name
            else:
                channel, name = arg

            if name is self.channels:
                raise Exception("Duplicate channel name {}".format(name))

            if channel.type not in self._mtasks:
                self._mtasks[channel.type] = MiniTask(channel.daq, channel.type)

            self.channels[name] = channel
            channel._add_to_minitask(self._mtasks[channel.type])

            TYPED_CHANNELS[channel.type].append(channel)
        self._setup_master_channel()

    # TODO: Move to shared file
    def _setup_master_channel(self):
        master_clock = ''
        self.master_trig = ''
        self.master_type = None
        for ch_type  in ['AI', 'AO', 'DI', 'DO']:
            if ch_type in self._mtasks:
                devname = ''
                for ch in self.channels.values():
                    if ch.type == ch_type:
                        devname =ch.daq.name
                        break
                master_clock = '/{}/{}/SampleClock'.format(devname, ch_type.lower())
                self.master_trig = '/{}/{}/StartTrigger'.format(devname, ch_type. lower())
                self.master_type = ch_type
                break

    @check_enums(mode=SampleMode, edge=EdgeSlope)
    @check_units(duration='?s', fsamp='?Hz')
    def set_timing(self, duration=None, fsamp=None, n_samples=None, mode='finite', edge='rising',
               clock=''):
        self.edge = edge
        num_args_specified = num_not_none(duration, fsamp, n_samples)
        if num_args_specified == 0:
            self.n_samples = 1
        elif num_args_specified == 2:
            self.fsamp, self.n_samples = handle_timing_params(duration, fsamp, n_samples)
            for ch_type, mtask in self._mtasks.items():
                mtask.config_timing(self.fsamp, self.n_samples, mode, self.edge, '')
        else:
            raise DAQError("Must specify 0 or 2 of duration, fsamp, and n_samples")

    def _setup_triggers(self):
        for ch_type, mtask in self._mtasks.items():
            if ch_type != self.master_type:
                mtask._mx_task.CfgDigEdgeStartTrig(self.master_trig, self.edge.value)
        self._trig_set_up = True

    def run(self, write_data=None):
        """Run a task from start to finish
        Writes output data, starts the task, reads input data, stops the task, then returns the
        input data. Will wait indefinitely for the data to be received. If you need more control,
        you may instead prefer to use `write()`, `read()`, `start()`, `stop()`, etc. directly.
        """
        if not self._trig_set_up:
            self._setup_triggers()

        self.write(write_data, autostart=False)
        self.start()
        read_data = self.read()
        self.stop()

        return read_data

    @check_units(timeout='?s')
    def read(self, timeout=None):
        timeout_s = float(-1. if timeout is None else timeout.m_as('s'))
        read_data = self._read_AI_channels(timeout_s)
        return read_data

    def write(self, write_data, autostart=True):
        """Write data to the output channels.
        Useful when you need finer-grained control than `run()` provides.
        """
        # Need to make sure we get data array for each output channel (AO, DO, CO...)
        for ch_name, ch in self.channels.items():
            if ch.type in ('AO', 'DO', 'CO'):
                if write_data is None:
                    raise ValueError("Must provide write_data if using output channels")
                elif ch_name not in write_data:
                    raise ValueError('write_data missing an array for output channel {}'
                         .format(ch_name))

        # Then set up writes for each channel, don't auto-start
        self._write_AO_channels(write_data, autostart=autostart)
        # self.write_DO_channels()
        # self.write_CO_channels()

    def verify(self):
        """Verify the Task.
        This transitions all subtasks to the `verified` state. See the NI documentation for details
        on the Task State model.
        """
        for mtask in self._mtasks.values():
            mtask.verify()

    def reserve(self):
        """Reserve the Task.
        This transitions all subtasks to the `reserved` state. See the NI documentation for details
        on the Task State model.
        """
        for mtask in self._mtasks.values():
            mtask.reserve()

    def unreserve(self):
        """Unreserve the Task.
        This transitions all subtasks to the `verified` state. See the NI documentation for details
        on the Task State model.
        """
        for mtask in self._mtasks.values():
            mtask.unreserve()

    def abort(self):
        """Abort the Task.
        This transitions all subtasks to the `verified` state. See the NI documentation for details
        on the Task State model.
        """
        for mtask in self._mtasks.values():
            mtask.abort()

    def commit(self):
        """Commit the Task.
        This transitions all subtasks to the `committed` state. See the NI documentation for details
        on the Task State model.
        """
        for mtask in self._mtasks.values():
            mtask.commit()

    def start(self):
        """Start the Task.
        This transitions all subtasks to the `running` state. See the NI documentation for details
        on the Task State model.
        """
        for ch_type, mtask in self._mtasks.items():
            if ch_type != self.master_type:
                mtask.start()
        self._mtasks[self.master_type].start()  # Start the master last

    def stop(self):
        """Stop the Task and return it to the state it was in before it started.
        This transitions all subtasks to the state they in were before they were started, either due
        to an explicit `start()` or a call to `write()` with `autostart` set to True. See the NI
        documentation for details on the Task State model.
        """
        self._mtasks[self.master_type].stop()  # Stop the master first
        for ch_type, mtask in self._mtasks.items():
            if ch_type != self.master_type:
                mtask.stop()

    def clear(self):
        """Clear the task and release its resources.
        This clears all subtasks and releases their resources, aborting them first if necessary.
        """
        for mtask in self._mtasks.values():
            mtask.clear()

    def _read_AI_channels(self, timeout_s):
        """ Returns a dict containing the AI buffers. """
        is_scalar = self.fsamp is None
        mx_task = self._mtasks['AI']._mx_task
        buf_size = self.n_samples * len(self.AIs)
        data, n_samps_read = mx_task.ReadAnalogF64(-1, timeout_s, Val.GroupByChannel, buf_size)

        res = {}
        for i, ch in enumerate(self.AIs):
            start = i * n_samps_read
            stop = (i + 1) * n_samps_read
            ch_data = data[start:stop] if not is_scalar else data[start]
            res[ch.path] = Q_(ch_data, 'V')

        if is_scalar:
            res['t'] = Q_(0., 's')
        else:
            end_t = (n_samps_read-1)/self.fsamp.m_as('Hz') if self.fsamp is not None else 0
            res['t'] = Q_(np.linspace(0., end_t, n_samps_read), 's')
        return res

    def _write_AO_channels(self, data, autostart=True):
        if 'AO' not in self._mtasks:
            return
        mx_task = self._mtasks['AO']._mx_task
        ao_names = [name for (name, ch) in self.channels.items() if ch.type == 'AO']
        arr = np.concatenate([Q_(data[ao]).to('V').magnitude for ao in ao_names])
        arr = arr.astype(np.float64)
        n_samps_per_chan = list(data.values())[0].magnitude.size
        mx_task.WriteAnalogF64(n_samps_per_chan, autostart, -1., Val.GroupByChannel, arr)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        try:
            self.stop()
        except:
            if value is None:
                raise  # Only raise new error from StopTask if we started with one
        finally:
            self.clear()  # Always clean up our memory

    def __del__(self):
        self.clear()


class MiniTask(object):
    def __init__(self, daq, io_type):
        self.daq = daq
        self._mx_task = NicePyDAQmx.Task('')
        self.io_type = io_type
        self.chans = []
        self.fsamp = None
        
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        try:
            self.stop()
        except:
            if value is None:
                raise # Only raise new error from StopTask if we started one
        finally:
            self.clear() # Always clean up the memory

    sample_timing_type = enum_property('SampTimingType', SampleTiming)
    sample_mode = enum_property('SampQuantSampMode', SampleMode)
    samples_per_channel = int_property('SampQuantSampPerChan')
    input_buf_size = int_property('BufInputBufSize')
    output_buf_size = int_property('BufOutputBufSize')
    output_onboard_buf_size = int_property('BufOutputOnbrdBufSize')

    @check_enums(mode=SampleMode, edge=EdgeSlope)
    @check_units(fsamp='Hz')
    def config_timing(self, fsamp, n_samples, mode='finite', edge='rising', clock=''):
        clock = to_bytes(clock)
        self._mx_task.CfgSampClkTiming(clock, fsamp.m_as('Hz'), edge.value, mode.value, n_samples)

        # Save for later
        self.n_samples = n_samples
        self.fsamp = fsamp

    def reserve_with_timeout(self, timeout):
        """Try, multiple times if necessary, to reserve the hardware resources needed for the task.
        If `timeout` is None, only tries once. Otherwise, tries repeatedly until successful,
        raising a TimeoutError if the given timeout elapses. To retry without limit, use a negative
        `timeout`.
        """
        # TODO: Emulate this in software
        pass

    def start(self):
        self._mx_task.StartTask()

    def stop(self):
        self._mx_task.StopTask()

    def clear(self):
        self._mx_task.ClearTask()

    def _assert_io_type(self, io_type):
        if io_type != self.io_type:
            raise TypeError("MiniTask must have io_type '{}'} for this operation, but is of "
                    "type '{}'".format(io_type, self.io_type))

    @check_enums(term_cfg=TerminalConfig)
    @check_units(vmin='?V', vmax='?V')
    def add_AI_channel(self, ai, term_cfg='default', vmin=None, vmax=None):
        self._assert_io_type('AI')
        ai_path = ai if isinstance(ai, basestring) else ai.path
        self.chans.append(ai_path)
        default_min, default_max = self.daq._max_AI_range()
        vmin = default_min if vmin is None else vmin
        vmax = default_max if vmax is None else vmax
        self._mx_task.CreateAIVoltageChan(ai_path, '', term_cfg.value, vmin.m_as('V'),
                vmax.m_as('V'), Val.Volts, '')

    def add_AO_channel(self, ao):
        # TODO
        raise NotImplemented

    def add_DI_channel(self, di, split_lines=False):
        # TODO
        raise NotImplemented

    def add_DO_channel(self, do):
        # TODO
        raise NotImplemented

    def add_AO_only_onboard_mem(self, channel, onboard_only):
        # TODO
        raise NotImplemented

    @check_units(timeout='?s')
    def read_AI_scalar(self, timeout=None):
        self._assert_io_type('AI')
        samples = 2
        timeout_s = float(1)
        buf_size = 2

        self.config_timing('1000.0 Hz', samples)
        self.start()
        data, n_samples_read = self._mx_task.ReadAnalog64(samples, timeout_s, Val.GroupByChannel, buf_size)
        self.stop()
        value = np.mean(data)
        return Q_(value, 'V')

    @check_units(timeout='?s')
    def read_AI_channels(self, samples=-1, timeout=None):
        """Perform an AI read and get a dict containing the AI buffers"""
        self._assert_io_type('AI')
        samples = int(samples)
        timeout_s = float(-1. if timeout is None else timeout.m_as('s'))

        if samples == -1:
            # TODO
            buf_size = 1000 # TODO
        else:
            buf_size = samples * len(self.chans)

        data, n_samples_read = self._mx_task.ReadAnalogF64(samples, timeout_s, Val.GroupByChannel, buf_size)

        res = {}
        for i in ch_name in enumerate(self.chans):
            start = i * n_samples_read
            stop = (i+1) * n_samples_read
            res[ch_name] = Q_(data[start:stop], 'V')
        res['T'] = Q_(np.linspace(0, n_samples_read/self.fsamp.m_as('Hz'),
            n_samples_read, endpoint=False), 's')
        return res

    @check_units(values='V', timeout='?s')
    def write_AO_scalar(self, value, timeout=None):
        raise NotImplemented

    @check_units(timeout='?s')
    def read_DI_scalar(self, timeout=None):
        raise NotImplemented

    @check_units(timeout='?s')
    def read_DI_channels(self, samples=-1, timeout=None):
        """Perform a DI read and get a dict containing the DI buffers"""
        raise NotImplemented

    def _reorder_digital_int(self, data):
        """Reorders the bits of a digital int returned by DAQmx based on the line order."""
        raise NotImplemented

    @check_units(timeout='?s')
    def write_DO_scalar(self, value, timeout=None):
        raise NotImplemented

    @check_units(timeout='?s')
    def wait_until_done(self, timeout=None):
        raise NotImplemented

    def overwrite(self, overwrite):
        """Set whether to overwrite samples in the buffer that have not been read yet."""
        raise NotImplemented

    @check_enums(relative_to=RelativeTo)
    def relative_to(self, relative_to):
        raise NotImplemented

    def offset(self, offset):
        raise NotImplemented

    def write_AO_channels(self, data, timeout=-1.0, autostart=True):
        raise NotImplemented

class Channel(object):
    def __init__(self, daq):
        # Hold onto the DAQ object as a weakref to avoid cycles in the reference graph.
        self._daqref = weakref.ref(daq)

    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other):
        if isinstance(other, Channel):
            return self.path == other.path
        else:
            return self.path == other

    @property
    def daq(self):
        daq = self._daqref()
        if daq is None:
            raise RuntimeError("This channel's DAQ object no longer exists")
        return daq


class AnalogIn(Channel):
    type = 'AI'
    
    def __init__(self, daq, chan_name):
        Channel.__init__(self, daq)
        self.name = chan_name
        self.path = '{}/{}'.format(daq.name, chan_name)
        self._mtask = None


    # Same as other class
    @check_enums(term_cfg=TerminalConfig)
    def _add_to_minitask(self, minitask, term_cfg='default'):
        min, max = self.daq._max_AI_range()
        mx_task = minitask._mx_task
        mx_task.CreateAIVoltageChan(self.path, '', term_cfg.value, min.m_as('V'), max.m_as('V'),
                                    Val.Volts, '')

    @check_units(duration='?s', fsamp='?Hz')
    def read(self, duration=None, fsamp=None, n_samples=None, vmin=None, vmax=None, reserve_timeout=None):
        """Read one or more analog input samples.
        By default, reads and returns a single sample. If two of `duration`, `fsamp`,
        and `n_samples` are given, an array of samples is read and returned.
        Parameters
        ----------
        duration : Quantity
            How long to read from the analog input, specified as a Quantity.
            Use with `fsamp` or `n_samples`.
        fsamp : Quantity
            The sample frequency, specified as a Quantity. Use with `duration`
            or `n_samples`.
        n_samples : int
            The number of samples to read. Use with `duration` or `fsamp`.
        Returns
        -------
        data : scalar or array Quantity
            The data that was read from analog input.
        """

        with self.daq._create_mini_task('AI') as mtask:
            mtask.add_AI_channel(self, vmin=vmin, vmax=vmax)
            num_args_specified = num_not_none(duration, fsamp, n_samples)
            if num_args_specified == 0:
                mtask.reserve_with_timeout(reserve_timeout)
                try:
                    data = mtask.read_AI_scalar()
                except DAQError as e:
                    print((e.error))
            elif num_args_specified == 2:
                fsamp, n_samples = handle_timing_params(duration, fsamp, n_samples)
                mtask.config_timing(fsamp, n_samples)
                mtask.reserve_with_timeout(reserve_timeout)
                data = mtask.read_AI_channels()
            else:
                raise DAQError("Must specify 0 or 2 of duration, fsamp, and n_samples")
        return data

    def start_reading(self, fsamp=None, vmin=None, vmax=None, overwrite=False,
            relative_to=RelativeTo.CurrReadPos, offset=0, buf_size=10):
        self._matsk = mtask = self.daq._create_mini_task('AI')
        mtask.add_AI_channel(self, vmin=vmin, vmax=vmax)
        mtask.config_timing(fsamp, buf_size, mode=SampleMode.continuous)
        mask.overwrite(overwrite)
        mtask.relative_to(relative_to)
        mtask.offset(offset)
        self._mtask.start()

    def read_sample(self, timeout=None):
        try:
            return self._mtask.read_AI_scalar(timeout)
        except DAQError as e:
            if e.code == NicePyDAQmx.ErrorSamplesNotYetAvailable:
                return None
            raise
    
    def stop_reading(self):
        self._mtask.stop()
        self._mtask = None

# TODO
_ANALOG_CHANNELS_DICT = {
    '9215': ['ai0', 'ai1', 'ai2', 'ai3']
    }

_MAX_AI_RANGES_DICT = {
    '9215': [-10, 10]
    }

class NIDAQBase(DAQ):
    _INST_PARAMS_ = ['model', 'device']

    mx = NicePyDAQmx
    Task = Task


    def _initialize(self):
        self.model = self._paramset['model']
        self._dev = self._paramset['device']
        self.name = self._paramset['device']
        self._load_analog_channels()
        self._load_internal_channels()
        self._load_digital_ports()

    def _create_mini_task(self, io_type):
        return MiniTask(self, io_type)

    def _load_analog_channels(self):
        channels = None
        for key, val, in _ANALOG_CHANNELS_DICT.items():
            if key in self.model:
                channels = val
        if channels is None:
            raise NotImplemented('This model (%s) is not is not supported. Please file an issue.' % self.model)
        for ai_name in channels:
            setattr(self, ai_name, AnalogIn(self, ai_name))

    def _load_internal_channels(self):
        # TODO
        pass

    def _load_digital_ports(self):
        # TODO
        pass

    def _max_AI_range(self):
        for key, v_range, in _MAX_AI_RANGES_DICT.items():
            if key in self.model:
                return map(lambda val: val * u.V, v_range)
        return _MAX_AI_RANGES_DICT['9215']
