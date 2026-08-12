import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from time import sleep, time
import warnings
from tqdm.auto import tqdm

# Directory containing GenericDevice must be in path
# Add parent directory of current file to path
parent_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, parent_directory)
# Add also directory two levels up
sys.path.insert(0, os.path.dirname(parent_directory))
from GenericDevice import _GenericDevice

plt.ioff()

def set_time_unit(times):
    """Determine most appropriate SI prefix for time. 
    Return unit and rescaled time axis.

    Args:
        times (array-like): time values

    Returns:
        array-like, str: rescaled axis, new SI unit
    """
    times = times.copy() # the input list or array is mutable
    if (times[-1] < 1e-3):
        times *= 1e6
        tUnit = "µs"
    elif (times[-1] < 1):
        times *= 1e3
        tUnit = "ms"
    else:
        tUnit = "s"
    return times, tUnit

def set_freq_unit(freqs):
    """Determine most appropriate SI prefix for frequency. 
    Return unit and rescaled frequency axis.

    Args:
        freqs (array-like): frequency values

    Returns:
        array-like, str: rescaled axis, new SI unit
    """
    freqs = freqs.copy() # the input list or array is mutable
    if (freqs[-1] >= 1e9):
        freqs *= 1e-9
        fUnit = "GHz"
    elif (freqs[-1] >= 1e6):
        freqs *= 1e-6
        fUnit = "MHz"
    elif (freqs[-1] >= 1e3):
        freqs *= 1e-3
        fUnit = "kHz"
    else:
        fUnit = "Hz"
    return freqs, fUnit

class _Preamble:
    def __init__(self, s):
        elems = s.split(',')
        self.elems = elems
        self.points = int(elems[2])
        self.count = int(elems[3])
        self.x_inc = float(elems[4])
        self.x_orig = float(elems[5])
        self.x_ref = float(elems[6])
        self.y_inc = float(elems[7])
        self.y_orig = int(elems[8])
        self.y_ref = int(elems[9])

    def normalize(self, raw_y):
        yvals = raw_y.astype(np.float64)
        yvals -= (self.y_orig + self.y_ref)
        yvals *= self.y_inc
        return yvals

    def x_values(self):
        xvals = np.linspace(0, self.points-1, self.points)
        xvals *= self.x_inc
        xvals += self.x_ref
        # xvals += self.x_orig
        return xvals

class Scope(_GenericDevice):

    def get_waveform_raw(self, channels: list = [1], memdepth: str | int = None,
                          single = False, plot: bool = False, ndivs: int = None,
                          start: int = 1, stop: int = None,
                          barrier = None) -> np.ndarray:
        """
        Gets the entire waveform data in the internal memory for a selection of channels
        (!) To retrieve long timescale waveforms, enable single trigger mode
        :param list channels: List of channels
        :param float memdepth: Memory depth (number of points), specify as integer
            (see device manual for possible values) or "AUTO", 
            defaults to None (does not modify)
        :param boolean single: Use single trigger mode
        :param bool plot: Will plot the traces
        :param int ndivs: DEPRECATED, now set by query to oscilloscope.
        :param int start: Set start position of waveform data reading, defaults to 1.
        :param int stop: Set stop position of waveform data reading, 
            included in interval, defaults to maximum memory depth.
        :param <multiprocessing.Barrier> barrier (optional): Useful for
                synchronizing multiple processes (measurements)
        :returns: Data, Time np.ndarrays containing the traces of shape
            (channels, nbr of points) if len(channels)>1
        """
        no_channels = len(channels)
        if len(channels) > 4:
            print("ERROR : Invalid channel list provided" +
                  " (List too long)")
            sys.exit()
        # Print message indicating from which channels waveforms are being retrieved
        if no_channels == 1: message = f'{self.short_name} | Getting raw waveform from'
        else: message = f'{self.short_name} | Getting raw waveforms from'
        for chan in channels:
            if chan > 4:
                print("ERROR : Invalid channel list provided" +
                    " (Channels are 1,2,3,4)")
                sys.exit()
            if no_channels==2 and chan == channels[-1]:
                message+= f' and Channel {chan}'
            elif no_channels>2:
                if chan != channels[-1]:
                    message+= f' Channel {chan},'
                if chan == channels[-1]:
                    message+= f' and Channel {chan}'
            else:
                message+= f' Channel {chan}'
        print(message)

        Data = []
        Time = []
        trig_status = self.resource.query(':TRIGger:STATus?')

        # Set memory depth if specified
        if memdepth is not None:
            prev_memdepth = self.get_mdepth()

            if prev_memdepth != memdepth:
                if trig_status == "STOP\n":
                    # Memory depth cannot be set while oscilloscope is in STOP state
                    self.resource.query(':RUN;*OPC?')
                if memdepth == 'AUTO':
                    self.resource.query(f":ACQuire:MDEPth {memdepth};*OPC?")
                else:
                    self.resource.query(f":ACQuire:MDEPth {int(memdepth)};*OPC?")

        # Get memory depth
        memory_depth = self.get_mdepth()

        # Measure waveform, afterwards scope must be in STOP state to read from internal memory
        if single:
            if barrier is not None:
                barrier.wait()
                self.resource.write(":SINGle")
                print(f'{self.short_name} | Waiting for trigger as of {time()} s')
            else:
                self.resource.write(":SINGle")
            if trig_status.replace('\n','') == "STOP":
                # Wait for trig status to change from "STOP" before while loop
                sleep(1)
            while self.resource.query(':TRIGger:STATus?').replace('\n','') != 'STOP':
                sleep(0.1)
            print(f"{self.short_name} | Waveform complete as of {time()} s")
        else:
            self.resource.write(":STOP")
            
        # Transfer data from scope
        for chan in channels:
            self.resource.write(f":WAV:SOUR CHAN{chan}")
            # Y origin for wav data
            YORigin = self.resource.query_ascii_values(":WAV:YOR?")[0]
            # Y REF for wav data
            YREFerence = self.resource.query_ascii_values(":WAV:YREF?")[0]
            # Y INC for wav data
            YINCrement = self.resource.query_ascii_values(":WAV:YINC?")[0]
            # X REF for wav data
            XREFerence = self.resource.query_ascii_values(":WAV:XREF?")[0]
            # X INC for wav data
            XINCrement = self.resource.query_ascii_values(":WAV:XINC?")[0]
            # Set the waveform reading mode to RAW.
            self.resource.write(":WAV:MODE RAW")
            # Set return format to Byte.
            self.resource.write(":WAV:FORM BYTE")
            # Set start position of waveform data reading
            self.resource.write(f":WAV:STAR {int(start)}")

            if stop is None:
                stop = memory_depth
            n_points = stop - start + 1 #+1 because both endpoints are included

            if (n_points > 250000):
                # Read 250000 points at a time, provide completion updates             
                loopmax = int(np.ceil(n_points/250000))
                rawdata = []
                for loopcount in range(0, loopmax):
                    iteration_start = (loopcount*250000) + start
                    # iteration_stop is included in interval, hence -1
                    iteration_stop = min(iteration_start + 250000 - 1, stop)
                    self.resource.write(f":WAV:STAR {iteration_start}")
                    self.resource.write(f":WAV:STOP {iteration_stop}")
                    sys.stdout.write(f"\rReading {iteration_stop - start + 1}/{n_points}")
                    # Extent the rawdata variables with the new values.
                    rawdata.extend(self.resource.query_binary_values(":WAV:DATA?",
                                   datatype='B'))
            else:
                # Read whole waveform in one go
                self.resource.write(f":WAV:STOP {int(stop)}")
                # Read data from the resource, excluding the first 9 bytes
                # (TMC header).
                sys.stdout.write(f"\rReading {n_points}/{n_points}")
                rawdata = self.resource.query_binary_values(":WAV:DATA?",
                                                        datatype='B')

            data = (np.asarray(rawdata) - YORigin - YREFerence) * YINCrement
            Data.append(data)
            # Create time axis
            times = np.linspace(XREFerence, XINCrement*len(data), len(data))
            Time.append(times)
        if plot: 
            # Assumes waveforms all have same time axis
            fig, ax = plt.subplots()
            times_rescaled, tUnit = set_time_unit(Time[0])
            [ax.plot(times_rescaled, Data[n], label = f"Channel {chan}") for n, chan in enumerate(channels)]
            ax.set_ylabel("Voltage (V)")
            ax.set_xlabel("Time (" + tUnit + ")")
            ax.set_xlim(times_rescaled[0], times_rescaled[-1])
            ax.legend()
            plt.show()
        self.resource.write(":RUN")
        Data = np.asarray(Data)
        Time = np.asarray(Time)
        if len(channels) == 1:
            Data = Data[0, :]
            Time = Time[0, :]
        return Time, Data

    def get_wf_fromtrig(self, timespan,
                        channels: list = [1], memdepth: str | int = None,
                        single = False, plot: bool = False, ndivs: int = None,) -> np.ndarray:
        """Get the raw waveform from oscilloscope for a specified time duration
          starting from the trigger.

        Args:
            timespan (float): duration of retrieved waveform.
            See get_waveform_raw for info on other parameters.

        Returns:
            np.ndarray, np.ndarray : Data, Time arrays of shape
            (channels, nbr of points) if len(channels)>1
        """        
        
        # Determine index of trigger for raw waveform
        XORigin = self.resource.query_ascii_values(":WAV:XOR?")[0]
        XINCrement = self.resource.query_ascii_values(":WAV:XINC?")[0]
        trig_index = int(abs(XORigin)/XINCrement)

        # Determine index of final point to retrieve
        n_points = int(timespan / XINCrement)
        endpoint = trig_index + n_points - 1 #-1 because endpoint is retrieved

        # Call get_waveform_raw
        return self.get_waveform_raw(channels = channels,
                                     memdepth = memdepth,
                                     single = single,
                                     plot = plot,
                                     ndivs = ndivs,
                                     start = trig_index,
                                     stop = endpoint)

    def get_waveform(self, channels: list = [1], memdepth: str | int = None,
                     single = False, plot: bool = False,
                     ndivs: int = None, barrier = None) -> np.ndarray:
        """Retrieves the displayed waveform.
        Gets the waveform data in the internal memory for the time interval displayed on screen.
        From the displayed time scale and the sampling rate, will compute how many
        points of the memory correspond to the displayed signal.
        It will then retrieve the displayed signal (the part delimited by the 
        shaded area on top of the screen).
        See the :WAVeform Commands documentation for futher details.
        (!) To retrieve long timescale waveforms, enable single trigger mode, and
        beware of VISA timeout if transferring large number of data points.
        (timeout issue can be resolved by reducing memory depth).

        Args:
            channels (list, optional): List of channels. Defaults to [1].
            memdepth (float, optional): Memory depth (number of points), specify 
                as integer (see device manual for possible values) or "AUTO", 
                defaults to None (does not modify)
            single (boolean, optional): Use single trigger mode
            plot (bool, optional): Whether to plot the result. Defaults to False.
            ndivs (int, optional): DEPRECATED, now set by query to oscilloscope.
            barrier (<multiprocessing.Barrier>, optional): Useful for
                synchronizing multiple processes (measurements)
        Returns:
            np.ndarray: Data, Time
        """
        no_channels = len(channels)
        if len(channels) > 4:
            print("ERROR : Invalid channel list provided" +
                  " (List too long)")
            sys.exit()
        # Print message indicating from which channels waveforms are being retrieved
        if no_channels == 1: message = f'{self.short_name} | Getting waveform from'
        else: message = f'{self.short_name} | Getting waveforms from'
        for chan in channels:
            if chan > 4:
                print("ERROR : Invalid channel list provided" +
                    " (Channels are 1,2,3,4)")
                sys.exit()
            if no_channels==2 and chan == channels[-1]:
                message+= f' and Channel {chan}'
            elif no_channels>2:
                if chan != channels[-1]:
                    message+= f' Channel {chan},'
                if chan == channels[-1]:
                    message+= f' and Channel {chan}'
            else:
                message+= f' Channel {chan}'
        print(message)

        Data = []
        Time = []
        trig_status = self.resource.query(':TRIGger:STATus?')

        # Set memory depth if specified
        if memdepth is not None:
            prev_memdepth = self.get_mdepth()

            if prev_memdepth != memdepth:
                if trig_status == "STOP\n":
                    # Memory depth cannot be set while oscilloscope is in STOP state
                    self.resource.query(':RUN;*OPC?')
                if memdepth == 'AUTO':
                    self.resource.query(f":ACQuire:MDEPth {memdepth};*OPC?")
                else:
                    self.resource.query(f":ACQuire:MDEPth {int(memdepth)};*OPC?")
        
        # ndivs is number of horizontal grids on the screen
        ndivs = self.get_ndivs()

        # get horizontal timebase
        time_scale = self.get_xscale()
        # get sample rate
        sample_rate = self.get_srate()
        # Get memory depth
        memory_depth = self.get_mdepth()

        # Exit before measurement if data transfer from scope will be unsuccessful
        x_inc = 1/sample_rate
        screen_points = np.floor(time_scale/x_inc)*ndivs
        if screen_points > 250000:
            sys.exit("ERROR: The number of waveform data points exceeds the" +
                  " maximum number which can be read from the oscilloscope at" + 
                  " a single time (see manual).\nEither reduce memory depth" +
                  " or use <Scope>.get_waveform_raw().")

        # Measure waveform, afterwards scope must be in STOP state to read from internal memory
        if single:
            if barrier is not None:
                barrier.wait()
                self.resource.write(":SINGle")
                print(f'{self.short_name} | Waiting for trigger as of {time()} s')
            else:
                self.resource.write(":SINGle")
            if trig_status == "STOP\n":
                # Wait for trig status to change from "STOP" before while loop
                sleep(1)
            while self.resource.query(':TRIGger:STATus?').replace('\n','') != 'STOP':
                sleep(0.1)
            print(f"{self.short_name} | Waveform complete as of {time()} s")
        else:
            self.resource.write(":STOP")
         
        # Transfer data from scope
        for chan in channels:
            self.resource.write(f':WAV:SOUR CHAN{chan}')
            self.resource.write(':WAV:MODE RAW')
            self.resource.write(':WAV:FORM BYTE')
            self.resource.query('*OPC?')
            preamble = _Preamble(self.resource.query(':WAV:PRE?'))
            # we look for the middle of the memory and take what's displayed
            # on the screen
            self.resource.write(
                f"WAV:STAR {memory_depth//2 - screen_points//2+1}")
            self.resource.write(
                f"WAV:STOP {memory_depth//2 + screen_points//2}")
            print(f'{self.short_name} | Transferring {int(screen_points)} data points from Channel {chan}')
            data = self.resource.query_binary_values(':WAV:DATA?', datatype='B',
                                                     container=np.array,
                                                     delay=0.5,
                                                     data_points=screen_points)
            data = preamble.normalize(data)
            times = np.arange(0, np.round(len(data)*preamble.x_inc, 9), preamble.x_inc)
            Data.append(data)
            Time.append(times)
        if plot: 
            # Assumes waveforms all have same time axis
            fig, ax = plt.subplots()
            times_rescaled, tUnit = set_time_unit(Time[0])
            [ax.plot(times_rescaled, Data[n], label = f"Channel {chan}") for n, chan in enumerate(channels)]
            ax.set_ylabel("Voltage (V)")
            ax.set_xlabel("Time (" + tUnit + ")")
            ax.set_xlim(times_rescaled[0], times_rescaled[-1])
            ax.legend()
            plt.show()
        self.resource.write(":RUN")
        return np.asarray(Time), np.asarray(Data)
    
    def get_waveform_screen(self, channels: list = [1], plot: bool = False) -> np.ndarray:
        """Gets the waveform data currently displayed on the screen.
        Unlike reading waveform data from the internal memory, the oscilloscope does not need to be put into STOP state.

        Args:
            channels (list, optional): List of channels. Defaults to [1].
            plot (bool, optional): Whether to plot the result. Defaults to False.

        Returns:
            np.ndarray: Data, Time
        """
        Data = []
        Time = []
        for chan in channels:
            # Set the channel source of waveform data
            self.resource.write(f':WAVeform:SOURce CHANnel{chan}')
            # Set the waveform data reading mode to NORMal
            self.resource.write(':WAVeform:MODE NORMal')
            # Set the return format of waveform data to BYTE
            self.resource.write(':WAVeform:FORMat BYTE')
            # Query and return ten different waveform parameters, see manual
            # Required to convert retrieved waveform data into time and volts below
            preamble = _Preamble(self.resource.query(':WAVeform:PREamble?'))
            # Obtain data from the buffer
            data = self.resource.query_binary_values(':WAVeform:DATA?', datatype='B',
                                            container=np.array,
                                            delay=0.5)
            data = preamble.normalize(data)
            times = np.arange(0, np.round(len(data)*preamble.x_inc, 9), preamble.x_inc)
            Data.append(data)
            Time.append(times)
        if plot:
            # Assumes waveforms all have same time axis
            fig, ax = plt.subplots()
            times_rescaled, tUnit = set_time_unit(Time[0])
            [ax.plot(times_rescaled, Data[n], label = f"Channel {chan}") for n, chan in enumerate(channels)]
            ax.set_ylabel("Voltage (V)")
            ax.set_xlabel("Time (" + tUnit + ")")
            ax.set_xlim(times_rescaled[0], times_rescaled[-1])
            ax.legend()
            plt.show()
        return np.asarray(Time), np.asarray(Data)
    
    def get_mdepth(self):
        mdepth = self.resource.query(":ACQuire:MDEPth?").strip()
        try:
            mdepth = int(float(mdepth))
        except Exception: # query can return "AUTO"
             if mdepth == 'AUTO':
                self.resource.write(":WAVeform:MODE RAW")
                mdepth = int(float(self.resource.query(":WAVeform:STOP?").strip()))
        return mdepth

    def get_srate(self):
        """Query the current sample rate. The default unit is Sa/s.

        Returns:
            float: sampling rate
        """        
        srate = float(self.resource.query(":ACQuire:SRATe?"))
        return srate
    
    def get_xinc(self):
        """Queries the time interval between two neighboring points of the currently selected
        channel source in the X direction.

        Returns:
           float: time [s]
        """        
        return self.resource.query_ascii_values(":WAV:XINC?")[0]
    
    def get_xscale(self):
        """Queries the scale of the main time base.

        Returns:
           float: time/div [s/div]
        """        
        return float(self.resource.query_ascii_values(":TIMebase:SCALe?")[0])

    def set_yscale(self, scale:float, channels : list = [1]):
        for chan in channels:
             self.resource.write(f":CHAN{chan}:SCAL {scale}")

    def get_yscale(self, channels: list = [1]):
        for chan in channels:
            return self.resource.query_ascii_values(f":CHAN{chan}:SCAL?")[0]
        
    def get_trig_position(self):
        """Queries the offset of the main time base (ie. the trigger position).
        When the waveform trigger point is at the left (right) side of the screen center,
        the horizontal position is a positive (negative) value.

        Returns:
           float: time [s]
        """        
        return self.resource.query_ascii_values("TIMebase:OFFSet?")[0]
    
    def get_ndivs(self):
        # get number of horizontal grids on the screen,
        # ndivs is fixed for each instrument
        return int(self.resource.query_ascii_values(":SYSTem:GAM?")[0])

    def get_dur_scr(self):
        """Return duration of waveform displayed on the screen

        Returns:
            float: waveform length [s]
        """        
        # Get duration of displayed waveform
        # obtained by multiplying the horizontal 
        # time base by the number of grids in the horizontal direction.
        time_scale = self.get_xscale()
        ndivs = self.get_ndivs()
        return time_scale * ndivs
    
    def get_dur_raw(self, onscreen = False):
        """Return duration of waveform in internal memory

        Returns:
            float: waveform length [s]
        """        
        return self.get_mdepth() / self.get_srate()

    def get_screenshot(self, filename: str = None, format: str = 'png'):
        """
        Recovers a screenshot of the screen and returns the image
        :param filename: Location where the image will be saved
        :param format: Image format in ['jpg', 'png', 'tiff','bmp8', 'bmp24']
        """
        assert format in ('jpeg', 'png', 'bmp8', 'bmp24', 'tiff')
        self.resource.timeout = 60000
        self.resource.write(':disp:data? on,off,%s' % format)
        raw_img = self.resource.read()
        self.resource.timeout = 25000
        img = np.asarray(raw_img).reshape((600, 1024))
        if filename:
            try:
                os.remove(filename)
            except OSError:
                pass
            with open(filename, 'wb') as fs:
                fs.write(raw_img)
        return img
    
    def close(self):
        self.resource.write(":RUN")
        self.resource.close()
        self.rm.close()


class SpectrumAnalyzer(_GenericDevice):

    def zero_span(self, center: float = 1e6, rbw: int = 100,
                  vbw: int = 30, swt: float = 'auto', 
                  trig: bool = None, single = False,
                  plot: bool = False):
        """Zero span measurement.

        (!) For long sweep times, use single sweep mode

        :param float center: Center frequency in Hz, converted to int
        :param float rbw: Resolution bandwidth
        :param float vbw: Video bandwidth
        :param float swt: Total measurement time. Except if set to 'auto'
        :param bool trig: External trigger
        :param bool single: Set True for single sweep mode,
            defaults to False for continuous sweep mode
        :param bool plot: option to plot
        :return: data, time for data and time
        :rtype: np.ndarray

        """
        self.resource.write(':FREQuency:SPAN 0')
        self.resource.write(f':FREQuency:CENTer {center}')
        self.resource.write(f':BANDwidth:RESolution {int(rbw)}')
        self.resource.write(f':BANDwidth:VIDeo {int(vbw)}')
        if swt != 'auto':
            self.resource.write(f':SENSe:SWEep:TIME {swt}')  # in s.
        else:
            self.resource.write(':SENSe:SWEep:TIME:AUTO ON')
        self.resource.write(
            ':DISPlay:WINdow:TRACe:Y:SCALe:SPACing LOGarithmic')
        
        if trig is not None:
            trigstate = self.resource.query(':TRIGger:SEQuence:SOURce?').replace('\n','')
            istrigged = trigstate != 'IMM' # whether SA is initially triggered
            # If trigger true and initial trigger type IMMediate, then set to EXTernal
            if trig and not (istrigged): 
                self.resource.write(':TRIGger:SEQuence:SOURce EXTernal')
                self.resource.write(
                    ':TRIGger:SEQuence:EXTernal:SLOPe POSitive')
            # If trigger false and initial trigger type not IMMediate, set to IMMediate
            elif not (trig) and istrigged:
                self.resource.write(':TRIGger:SEQuence:SOURce IMMediate')
        set_trigstate = self.resource.query(':TRIGger:SEQuence:SOURce?').replace('\n','')

        # Query current instrument sweep state
        sweep_state = int(self.resource.query(':INITiate:CONTinuous?'))
        if single == False:
            if sweep_state == 1: 
                # Already in continuous
                pass
            elif sweep_state == 0:
                # Put into continous
                self.resource.write(':INITiate:CONTinuous 1')
            triginfo_msg = ' with trigger' if set_trigstate == 'EXT' else ''
            print(f'{self.short_name} | Getting zero span scan (continuous sweep mode' +
                  triginfo_msg + ')')
        elif single == True:
            # In following conditional blocks, *OPC command (operation complete)
            # is sent subsequent to command intializing scan so that operation 
            # complete status can be polled (must wait for scan completion).
            self.resource.write('*CLS') # Reset status registers, clear error queue
            if sweep_state == 1:
                # Put into single
                self.resource.write(':INITiate:CONTinuous 0')
                self.resource.write(':INITiate:IMMediate; *OPC')
            elif sweep_state == 0:
                if trig is None or trig == False:
                    self.resource.write(':INITiate:IMMediate; *OPC')
                elif trig == True:
                    # Must reset trigger just before initiating scan, otherwise
                    # it seems trigger success condition is stored, because scan
                    # starts immediately instead of waiting for next trigger
                    self.resource.write(':TRIGger:SEQuence:SOURce EXTernal')
                    self.resource.write(':INITiate:IMMediate; *OPC')
            triginfo_msg = ' on trigger' if set_trigstate == 'EXT' else ''
            print(f'{self.short_name} | Getting zero span scan (single sweep' +
                  triginfo_msg + ')')
            # Poll the operation complete status in the event status 
            # register (ESR). (Once all commands before *OPC have been executed, 
            # the operation complete bit in the ESR is set to 1)
            while int(self.resource.query('*ESR?')) != 1:
                sleep(0.1)
        self.resource.write(':FORMat:TRACe:DATA ASCii')
        data = self.query_data()
         # If SA was trigged before, put it back in the same state
        if trig is not None:
            if not (trig) and istrigged:
                self.resource.write(f":TRIGger:SEQuence:SOURce {trigstate}")
        # Put SA back into the state it started in
        if sweep_state != int(self.resource.query(':INITiate:CONTinuous?')):
            self.resource.write(f':INITiate:CONTinuous {int(sweep_state)}')
        sweeptime = float(self.resource.query(':SWEep:TIME?'))
        times = np.linspace(0, sweeptime, len(data))
        if plot:
            fig, ax = plt.subplots()
            times_rescaled, tUnit = set_time_unit(times)
            ax.plot(times_rescaled, data)
            ax.set_xlabel(f'Time ({tUnit})')
            ax.set_ylabel('Noise Power (dBm)')
            plt.show()
        return data, times

    def span(self, center: float = 22.5e6, span: float = 45e6, rbw: int = 100,
             vbw: int = 30, swt: float = 'auto', trig: bool = None, single = False,
             plot: bool = False):
        """Configure and execute measurement of noise power spectrum
        (!) For long sweep times, use single sweep mode

        :param float center: Center frequency in Hz
        :param float span: span
        :param float rbw: Resolution bandwidth
        :param float vbw: Video bandwidth
        :param float swt: Total measurement time
        :param bool trig: External trigger
        :param bool single: Set True for single sweep mode,
            defaults to False for continuous sweep mode
        :param bool plot: option to plot
        :return: data, freqs for data and frequencies
        :rtype: np.ndarray

        """
        self.resource.write(f':FREQuency:SPAN {span}')
        self.resource.write(f':FREQuency:CENTer {center}')
        self.resource.write(f':BANDwidth:RESolution {int(rbw)}')
        self.resource.write(f':BANDwidth:VIDeo {int(vbw)}')
        if swt != 'auto':
            self.resource.write(f':SENSe:SWEep:TIME {swt}')  # in s.
        else:
            self.resource.write(':SENSe:SWEep:TIME:AUTO ON')
        self.resource.write(
            ':DISPlay:WINdow:TRACe:Y:SCALe:SPACing LOGarithmic')
        self.resource.query('*OPC?')

        if trig is not None:
            trigstate = self.resource.query(':TRIGger:SEQuence:SOURce?').replace('\n','')
            istrigged = trigstate != 'IMM' # whether SA is initially triggered
            # If trigger true and initial trigger type IMMediate, then set to EXTernal
            if trig and not (istrigged): 
                self.resource.write(':TRIGger:SEQuence:SOURce EXTernal')
                self.resource.write(
                    ':TRIGger:SEQuence:EXTernal:SLOPe POSitive')
            # If trigger false and initial trigger type not IMMediate, set to IMMediate
            elif not (trig) and istrigged:
                self.resource.write(':TRIGger:SEQuence:SOURce IMMediate')
        self.resource.query('*OPC?')
        set_trigstate = self.resource.query(':TRIGger:SEQuence:SOURce?').replace('\n','')
    
        # Query current instrument sweep state
        sweep_state = int(self.resource.query(':INITiate:CONTinuous?'))
        if single == False:
            if sweep_state == 1: 
                # Already in continuous
                pass
            elif sweep_state == 0:
                # Put into continous
                self.resource.write(':INITiate:CONTinuous 1')
            triginfo_msg = ' with trigger' if set_trigstate == 'EXT' else ''
            print(f'{self.short_name} | Getting power spectrum (continuous sweep mode' +
                  triginfo_msg + ')')
        elif single == True:
            # In following conditional blocks, *OPC command (operation complete)
            # is sent subsequent to command intializing scan so that operation 
            # complete status can be polled (must wait for scan completion).
            self.resource.write('*CLS') # Reset status registers, clear error queue
            if sweep_state == 1:
                # Put into single
                self.resource.write(':INITiate:CONTinuous 0')
                self.resource.write(':INITiate:IMMediate; *OPC')
            elif sweep_state == 0:
                if trig is None or trig == False:
                    self.resource.write(':INITiate:IMMediate; *OPC')
                elif trig == True:
                    # Must reset trigger just before initiating scan, otherwise
                    # it seems trigger success condition is stored, because scan
                    # starts immediately instead of waiting for next trigger
                    self.resource.write(':TRIGger:SEQuence:SOURce EXTernal')
                    self.resource.write(':INITiate:IMMediate; *OPC')
            triginfo_msg = ' on trigger' if set_trigstate == 'EXT' else ''
            print(f'{self.short_name} | Getting power spectrum (single sweep' +
                  triginfo_msg + ')')
            # Poll the operation complete status in the event status 
            # register (ESR). (Once all commands before *OPC have been executed, 
            # the operation complete bit in the ESR is set to 1)
            while int(self.resource.query('*ESR?')) != 1:
                sleep(0.1)
        self.resource.write(':FORMat:TRACe:DATA ASCii')
        data = self.query_data()
        # If SA was trigged before, put it back in the same state
        if trig is not None:
            if not (trig) and istrigged:
                self.resource.write(f":TRIGger:SEQuence:SOURce {trigstate}")
        # Put SA back into the state it started in
        if sweep_state != int(self.resource.query(':INITiate:CONTinuous?')):
            self.resource.write(f':INITiate:CONTinuous {int(sweep_state)}')
        freqs = np.linspace(center-span//2, center+span//2, len(data))
        if plot:
            fig, ax = plt.subplots()
            freq_rescaled, fUnit = set_freq_unit(freqs)
            ax.plot(freq_rescaled, data)
            ax.set_xlabel(f'Frequency ({fUnit})')
            ax.set_ylabel('Noise Power (dBm)')
            plt.show()
        return data, freqs

    def query_data(self):
        """Lower level function to grab the data from the SpecAnalyzer

        :return: data
        :rtype: list

        """
        rawdata = self.resource.query(':TRACe? TRACE1')
        data = rawdata.split(', ')[1:]
        data = [float(i) for i in data]
        return np.asarray(data)
    
     ### BANDwidth Subsection

    @property
    def rbw(self):
        """Resolution bandwidth
        """            
        return float(self.resource.query(":SENSe:BANDwidth:RESolution?").strip())
    
    @rbw.setter
    def rbw(self, value):      
        self.resource.write(f":SENSe:BANDwidth:RESolution {value}")

    @property
    def vbw(self):
        """Video bandwidth
        """            
        return float(self.resource.query(":SENSe:BANDwidth:VIDeo?").strip())
    
    @vbw.setter
    def vbw(self, value):      
        self.resource.write(f":SENSe:BANDwidth:VIDeo {value}")
    
    ### FREQuency Subsection
    
    @property
    def center(self):
        """Center frequency
        """            
        return float(self.resource.query(":SENSe:FREQuency:CENTer?").strip())
    
    @center.setter
    def center(self, value):      
        self.resource.write(f":SENSe:FREQuency:CENTer {value}")

    @property
    def span(self):
        """Frequency span
        """            
        return float(self.resource.query(":SENSe:FREQuency:SPAN?").strip())
    
    @span.setter
    def span(self, value):      
        self.resource.write(f":SENSe:FREQuency:SPAN {value}")


class ArbitraryFG(_GenericDevice):

    def get_waveform(self, output: int = 1, amp_unit: int = False) -> list:
        """
        Gets the waveform type as well as its specs
        :param int output: Description of parameter `output`.
        :param int amp_unit: If true, amp is returned as tuple with its unit, eg. Vrms
            Otherwise, BY DEFAULT amplitude returned as Vpp
        :return: List containing all the parameters
        :rtype: list

        """
        if output not in [1, 2]:
            print("ERROR : Invalid output specified")
            return None
        ison = self.resource.query(f"OUTPut{output}?")[:-1] == "ON"
        if amp_unit == False:
            # Before querying waveform parameters, need to change amplitude unit
            # to Vpp. Otherwise, value is returned for whatever unit user specified
            curr_amp_unit =  self.resource.query(f"SOURce{output}:VOLT:UNIT?").replace('\n','')
            self.resource.write(f"SOURce{output}:VOLT:UNIT VPP")
        ret = self.resource.query(f"SOURce{output}:APPLy?")
        ret = ret[1:-2].split(",")
        for i in range(len(ret)):
            try:
                ret[i] = float(ret[i])
            except ValueError: # When generating noise, phase and freq are undefined
                pass
        type = ret[0]
        freq = ret[1]
        amp = ret[2]
        offset = ret[3]
        phase = ret[4]
        if amp_unit == True:
            unit = self.resource.query(f"SOURce{output}:VOLT:UNIT?").replace('\n','')
            amp = (amp, unit)
        else:
            # Change amplitude unit back to whatever user had set
            self.resource.write(f"SOURce{output}:VOLT:UNIT {curr_amp_unit}")
        return [ison, type, freq, amp, offset, phase]

    def turn_on(self, output: int = 1):
        """
        Turns on an output channel on the last preset
        :param int output: Output channel
        :return: None
        """
        self.resource.write(f"OUTPut{output} ON")

    def turn_off(self, output: int = 1):
        """
        Turns off an output channel on the last preset
        :param int output: Output channel
        :return: None
        """
        self.resource.write(f"OUTPut{output} OFF")

    def set_impedance(self, output: int = 1, load: str = 'INF'):
        """
        Sets the output impedance to specified value. It doesn't actually
        change the physical impendance of the instrument, but changes the
        displayed voltage to match the actual voltage on the device under test.
        :param int output: Output channel
        :param str load: specified impedance value. {<ohms>|INFinity|MINimum|MAXimum}
        :return: None
        """
        if output not in [1, 2]:
            print("ERROR : Invalid output specified")
            return None
        self.resource.write(f':OUTP{output}:IMP ' + load) 
        print(f'Impedance OUTP{output} set to :', self.resource.query(f':OUTP{output}:IMP?'))

    def get_impedance(self, output: int = 1):  
        """Queries the output impedance of the specified channel(s).
        If no channel is specified, returns output impedance for Channel 1

        Args:
            output (int or list of int, optional): Channel. Defaults to 1.
            
        Returns:
            float or list of float: impedances
        """
        if type(output) is list:
            output_impedance=[]
            for ch in output:
                ch_impedance = float(self.resource.query(f':OUTP{ch}:IMP?'))
                output_impedance.append(ch_impedance)
        elif type(output) is int:
            output_impedance = float(self.resource.query(f':OUTP{output}:IMP?'))
        return output_impedance
    
    def set_voltage_unit(self, output: int = 1, unit: str = 'VPP'):
        """
        Sets the amplitude unit of the specified channel to 
        Vpp (VPP), Vrms (VRMS), or dBm (DBM)
        :param int output: Output channel
        :param str unit: amplitude unit
        :return: None
        """
        self.resource.write(f":SOURce{output}:VOLTage:UNIT {unit}")
        
    def get_voltage_unit(self, output: int = 1):
        """
        Returns the amplitude unit of the specified channel
        :param int output: Output channel
        :return: amplitude unit
        """
        unit = self.resource.query(f":SOURce{output}:VOLTage:UNIT?")
        return unit.replace('\n','')

    def dc_offset(self, output: int = 1, offset: float = 2.0):
        """
        Applies a constant voltage on the specified output
        :param int output: Output channel
        :param float offset: Voltage applied in Volts
        :return: None
        """
        if output not in [1, 2]:
            print("ERROR : Invalid output specified")
            return None
        self.resource.write(f":SOURce{output}:FUNCtion DC")
        self.resource.write(f":SOURce{output}:APPLy:USER 1, 1, {offset}, 0")
        self.turn_on(output)

    def sine(self, output: int = 1, freq: float = 100.0, ampl: float = 2.0,
             offset: float = 0.0, phase: float = 0.0):
        """
        Sets a sine wave on specified output
        :param int output: Output channel
        :param float freq: Frequency of the signa in Hz
        :param float ampl: Amplitude of the wave in Volts
        :param float offset: Voltage offset in Volts
        :param float phase: Signal phase in degree
        :return: None
        """
        if output not in [1, 2]:
            print("ERROR : Invalid output specified")
            return None
        self.resource.write(f":SOURce{output}:APPLy:SINusoid {freq}, {ampl}, " +
                            f"{offset}, {phase}")
        self.turn_on(output)

    def square(self, output: int = 1, freq: float = 100.0, ampl: float = 2.0,
               offset: float = 0.0, phase: float = 0.0, duty: float = 50.0):
        """
        Sets a square wave on specified output
        :param int output: Output channel
        :param float freq: Frequency of the signa in Hz
        :param float ampl: Amplitude of the wave in Volts
        :param float offset: Voltage offset in Volts
        :param float phase: Signal phase in degree
        :param float duty: Duty cycle in percent
        :return: None
        """
        if output not in [1, 2]:
            print("ERROR : Invalid output specified")
            return None
        self.resource.write(f":SOURce{output}:APPLy:SQUare {freq}, {ampl}, " +
                            f"{offset}, {phase}")
        self.resource.write(f":SOURce{output}:FUNCtion:SQUare:DCYCle {duty}")
        self.turn_on(output)

    def ramp(self, output: int = 1, freq: float = 100.0, ampl: float = 2.0,
             offset: float = 0.0, phase: float = 0.0, symm: float = 50.0):
        """
        Sets a triangular wave on specified output
        :param int output: Output channel
        :param float freq: Frequency of the signa in Hz
        :param float ampl: Amplitude of the wave in Volts
        :param float offset: Voltage offset in Volts
        :param float phase: Signal phase in degree
        :param float symm: Symmetry factor in percent (equivalent to duty)
        :return: None
        """
        if output not in [1, 2]:
            print("ERROR : Invalid output specified")
            return None
        self.resource.write(f":SOURce{output}:APPLy:RAMP {freq}, {ampl}, " +
                            f"{offset}, {phase}")
        self.resource.write(f":SOURce{output}:FUNCtion:RAMP:SYMMetry {symm}")
        self.turn_on(output)

    def pulse(self, output: int = 1, freq: float = 100.0, ampl: float = 2.0,
              offset: float = 0.0, phase: float = 0.0, duty: float = 50.0,
              rise: float = 10e-9, fall: float = 10e-9):
        """
        Sets a triangular wave on specified output
        :param int output: Output channel
        :param float freq: Frequency of the signa in Hz
        :param float ampl: Amplitude of the wave in Volts
        :param float offset: Voltage offset in Volts
        :param float phase: Signal phase in degree
        :param float duty: Duty cycle in percent
        :param float rise: Rise time in seconds
        :param float fall: Fall time in seconds
        :return: None
        """
        if output not in [1, 2]:
            print("ERROR : Invalid output specified")
            return None
        self.resource.write(f":SOURce{output}:APPLy:PULSe {freq}, {ampl}, " +
                            f"{offset}, {phase}")
        self.resource.write(f":SOURce{output}:FUNCtion:PULSe:DCYCLe {duty}")
        self.resource.write(
            f":SOURce{output}:FUNCtion:TRANsition:LEADing {rise}")
        self.resource.write(
            f":SOURce{output}:FUNCtion:TRANsition:TRAiling {fall}")
        self.turn_on(output)

    def noise(self, output: int = 1, ampl: float = 5.0, offset: float = 0.0):
        """
        Sends noise on specified output
        :param int output: Output channel
        :param float ampl: Amplitude in Volts
        :param float offset: Voltage offset in Volts
        :return: None
        """
        self.resource.write(f":SOURce{output}:APPLy:NOISe {ampl}, {offset}")
        self.turn_on(output)

    def arbitrary(self, output: int = 1, freq: float = 100, ampl: float = 5.0,
                  offset: float = 0.0, phase: float = 0.0,
                  function: str = 'SINC'):
        """
        Arbitrary function signal
        :param int output: Output channel
        :param float freq: Frequency of the signa in Hz
        :param float ampl: Amplitude of the wave in Volts
        :param float offset: Voltage offset
        :param float phase: Signal phase in degree
        :param str function: Function type
        :return: Description of returned object.
        :rtype: type

        """
        # List of all possible functions
        funcnames = ["KAISER", "ROUNDPM", "SINC", "NEGRAMP", "ATTALT",
                     "AMPALT", "STAIRDN", "STAIRUP", "STAIRUD", "CPULSE",
                     "NPULSE", "TRAPEZIA", "ROUNDHALF", "ABSSINE",
                     "ABSSINEHALF", "SINETRA", "SINEVER", "EXPRISE", "EXPFALL",
                     "TAN", "COT", "SQRT", "X2DATA", "GAUSS", "HAVERSINE",
                     "LORENTZ", "DIRICHLET", "GAUSSPULSE", "AIRY", "CARDIAC",
                     "QUAKE", "GAMMA", "VOICE", "TV", "COMBIN", "BANDLIMITED",
                     "STEPRESP", "BUTTERWORTH", "CHEBYSHEV1", "CHEBYSHEV2",
                     "BOXCAR", "BARLETT", "TRIANG", "BLACKMAN", "HAMMING",
                     "HANNING", "DUALTONE", "ACOS", "ACOSH", "ACOTCON",
                     "ACOTPRO", "ACOTHCON", "ACOTHPRO", "ACSCCON", "ACSCPRO",
                     "ACSCHCON", "ACSCHPRO", "ASECCON", "ASECPRO", "ASECH",
                     "ASIN", "ASINH", "ATAN", "ATANH", "BESSELJ", "BESSELY",
                     "CAUCHY", "COSH", "COSINT", "COTHCON", "COTHPRO",
                     "CSCCON", "CSCPRO", "CSCHCON", "CSCHPRO", "CUBIC,", "ERF",
                     "ERFC", "ERFCINV", "ERFINV", "LAGUERRE", "LAPLACE",
                     "LEGEND", "LOG", "LOGNORMAL", "MAXWELL", "RAYLEIGH",
                     "RECIPCON", "RECIPPRO", "SECCON", "SECPRO", "SECH",
                     "SINH", "SININT", "TANH", "VERSIERA", "WEIBULL",
                     "BARTHANN", "BLACKMANH", "BOHMANWIN", "CHEBWIN",
                     "FLATTOPWIN", "NUTTALLWIN", "PARZENWIN", "TAYLORWIN",
                     "TUKEYWIN", "CWPUSLE", "LFPULSE", "LFMPULSE", "EOG",
                     "EEG", "EMG", "PULSILOGRAM", "TENS1", "TENS2", "TENS3",
                     "SURGE", "DAMPEDOSC", "SWINGOSC", "RADAR", "THREEAM",
                     "THREEFM", "THREEPM", "THREEPWM", "THREEPFM", "RESSPEED",
                     "MCNOSIE", "PAHCUR", "RIPPLE", "ISO76372TP1",
                     "ISO76372TP2A", "ISO76372TP2B", "ISO76372TP3A",
                     "ISO76372TP3B", "ISO76372TP4", "ISO76372TP5A",
                     "ISO76372TP5B", "ISO167502SP", "ISO167502VR", "SCR",
                     "IGNITION", "NIMHDISCHARGE", "GATEVIBR", "PPULSE"]
        if function not in funcnames:
            print("ERROR : Unknwown function specified")
            pass
        self.resource.write(f":SOURce{output}:FUNCtion {function}")
        self.resource.write(f":SOURce{output}:APPLy:USER {freq}, {ampl}, " +
                            f"{offset}, {phase}")
        self.turn_on(output)

    def align_phase(self, output: int = 1):
        """Reconfigures output of specified channel to align phase with other output channel.
        The phases specified for the channels may still differ - this function aligns their phase references.

        Args:
            output (int, optional): Output channel. Defaults to 1.
        """
        self.resource.write(f":SOURce{output}:PHAS:SYNC")
        
""" 
The following class replaces the ArbitraryFG class. In the new implementation,
device and waveform parameters are defined as class properties,
so that 'get' and 'set' functionalities are implicitly defined.

e.g. the frequency of Channel 1 can be set to 100 Hz by entering:
    ch1.freq = 100
    the frequency of Channel 1 can be accessed by entering:
    ch1.freq
"""

class ArbitraryFG2(_GenericDevice):
    """A class for controlling Rigol Arbitrary Function generators.
    Each channel is represented by an instance of the inner class Channel.
    Device settings and waveform parameters are accessed and set as attributes 
    of the Channel class.

    """    

    def __init__(self, address):
        _GenericDevice.__init__(self, address)

        # Inner class does not automatically have access to the outer class instance
        # must pass the outer class instance as a parameter
        self.ch1 = self.Channel(self, 1)
        self.ch2 = self.Channel(self, 2)

    @property
    def clock(self):
        """Query reference clock source.

        Returns:
            str: 'INT' or 'EXT'
        """        
        return self.resource.query("SYSTem:ROSCillator:SOURce?").strip()
    
    @clock.setter
    def clock(self, value):
        # Set the reference clock source to INTernal or EXTernal.
        sources = ['INT', 'INTernal',
                   'EXT', 'EXTernal']
        if value.casefold() not in (source.casefold() for source in sources):
            raise Exception("Invalid clock source specified.") 
        self.resource.write(f"SYSTem:ROSCillator:SOURce {value}")

        # If system does not detect valid external clock source,
        # it falls back to internal source
        if value.casefold() == 'ext' or value.casefold() == 'external':
            if self.clock != 'EXT':
                raise Exception("Reference clock source could NOT be set to EXTERNAL.")

    class Channel():
        def __init__(self, afg, number):
            self.afg = afg
            self.number = number
            self.am = self.AM(self)
            self.pm = self.PM(self)

        @property
        def type(self):
            ret = self.afg.resource.query(f"SOURce{self.number}:APPLy?")
            return ret.strip().replace("\"","").split(",")[0]
                   
        @type.setter
        def type(self, value):
            waveform_types = ['SIN', 'SINusoid',
                              'SQU', 'SQUare',
                              'RAMP',
                              'PULS', 'PULSe',
                              'NOIS', 'NOISe',
                              'DC',
                              'USER']
            if value.casefold() not in (wf_type.casefold() for wf_type in waveform_types):
                raise Exception("Invalid waveform type specified.") 
            self.afg.resource.write(f":SOURce{self.number}:APPLy:{value}")

        @property
        def freq(self):
            return float(self.afg.resource.query(f":SOURce{self.number}:FREQ?").strip())
        
        @freq.setter
        def freq(self, value):
            if not isinstance(value, (int, float)):
                raise TypeError("Frequency must be int or float")
            self.afg.resource.write(f":SOURce{self.number}:FREQ {value}")

        @property
        def amp(self):
            return float(self.afg.resource.query(f":SOURce{self.number}:VOLT?").strip())
        
        @amp.setter
        def amp(self, value):
            if not isinstance(value, (int, float)):
                raise TypeError("Amplitude must be int or float")
            self.afg.resource.write(f":SOURce{self.number}:VOLT {value}")

        @property
        def offset(self):
            return float(self.afg.resource.query(f":SOURce{self.number}:VOLT:OFFS?").strip())
        
        @offset.setter
        def offset(self, value):
            if not isinstance(value, (int, float)):
                raise TypeError("Offset must be int or float")
            self.afg.resource.write(f":SOURce{self.number}:VOLT:OFFS {value}")

        @property
        def phase(self):
            return float(self.afg.resource.query(f":SOURce{self.number}:PHAS?").strip())

        @phase.setter
        def phase(self, value):
            if not isinstance(value, (int, float)):
                 raise TypeError("Phase must be int or float")
            self.afg.resource.write(f":SOURce{self.number}:PHAS {value}")

        @property
        def imped(self):
            return float(self.afg.resource.query(f":OUTPut{self.number}:IMPedance?").strip())

        @imped.setter
        def imped(self, value):
            if not isinstance(value, (int, float)) and value.casefold() != "INF".casefold():
                 raise TypeError("Output impedance must be int, float, or 'INF'")
            self.afg.resource.write(f":OUTPut{self.number}:IMPedance {value}")

        @property
        def symm(self):
            """Ramp symmetry defined as the percentage that the rising period of the ramp takes up in the whole period."""
            return float(self.afg.resource.query(f":SOURce{self.number}:FUNCtion:RAMP:SYMMetry?").strip())
        
        @symm.setter
        def symm(self, value):
            try:
                if not 0 < value < 100:
                    raise Exception("Ramp symmetry must be between 0 and 100.")
            except TypeError:
                raise TypeError("Ramp symmetry must be int or float.")
            self.afg.resource.write(f":SOURce{self.number}:FUNCtion:RAMP:SYMMetry {value}")

        @property
        def on(self):
            ret = self.afg.resource.query(f":OUTPut{self.number}?").strip()
            if ret == "ON":
                return True
            elif ret == "OFF":
                return False

        @on.setter
        def on(self, value):
            if (isinstance(value, bool) or value == 0 or value == 1):
                self.afg.resource.write(f":OUTPut{self.number} {int(value)}")
            elif value.casefold() == "ON".casefold():
                self.afg.resource.write(f":OUTPut{self.number} ON")
            elif value.casefold() == "OFF".casefold():
                self.afg.resource.write(f":OUTPut{self.number} OFF")
            else:
                raise TypeError("Output state must be boolean, 'ON', or 'OFF'")
    
        @property
        def sync_on(self):
            ret = self.afg.resource.query(f":OUTPut{self.number}:SYNC?").strip()
            if ret == "ON":
                return True
            elif ret == "OFF":
                return False

        @sync_on.setter
        def sync_on(self, value):
            if (isinstance(value, bool) or value == 0 or value == 1):
                self.afg.resource.write(f":OUTPut{self.number}:SYNC {int(value)}")
            elif value.casefold() == "ON".casefold():
                self.afg.resource.write(f":OUTPut{self.number}:SYNC ON")
            elif value.casefold() == "OFF".casefold():
                self.afg.resource.write(f":OUTPut{self.number}:SYNC OFF")
            else:
                raise TypeError("Sync output state must be boolean, 'ON', or 'OFF'")
            
        @property
        def sync_pol(self):
            """Output polarity of the sync signal on the rear-panel [Sync/Ext Mod/Trig/FSK] connector of the specified channel"""
            return self.afg.resource.query(f":OUTPut{self.number}:SYNC:POLarity?").strip()

        @sync_pol.setter
        def sync_pol(self, value):
            if self.sync_on is False:
                raise Exception("Sync output state must be ON to set the polarity.")
            sync_pol_options = ['POS', 'POSitive', 'NEG', 'NEGative']
            if value.casefold() not in (option.casefold() for option in sync_pol_options):
                raise Exception("Sync polarity must be specifed as \"positive\" or \"negative\".") 
            self.afg.resource.write(f":OUTPut{self.number}:SYNC:POLarity {value}")

        @property
        def sync_mode(self):
            if 'DG2102' in self.afg.identity:
                return "N/A" # DG2102 has no Sync Mode feature
            else:
                return self.afg.resource.query(f":OUTPut{self.number}:SYNC:MODE?").strip()
        
        @sync_mode.setter
        def sync_mode(self, value):
            if 'DG2102' in self.afg.identity:
                raise Exception("DG2102 does not have Sync Mode feature.")
            else:
                sync_modes = ["carr", "carrier", "norm", "normal"]
                if value.casefold() not in [mode.casefold() for mode in sync_modes]:
                    raise TypeError("Invalid sync mode specified.")
                self.afg.resource.write(f":OUTPut{self.number}:SYNC:MODE {value}")                

        def align(self):
            """Reconfigures output of specified channel to align phase with other output channel.
            The phases specified for the channels may still differ - this function aligns their phase references.

            Args:
                output (int, optional): Output channel. Defaults to 1.
            """
            self.afg.resource.write(f":SOURce{self.number}:PHAS:SYNC")
        
        def waveform(self, type = None, freq = None, amp = None, 
                     offset = None, phase = None, imped = None,
                     symm = None):
            # A function which allows multiple waveform parameters to be set at once
            if type is not None:
                self.type = type
            if freq is not None:
                self.freq = freq
            # Must set impedance before amplitude
            # otherwise amplitude is changed by generator when impedance is changed
            if imped is not None:
                self.imped = imped
            if amp is not None:
                self.amp = amp
            if offset is not None:
                self.offset = offset
            if phase is not None:
                self.phase = phase
            if symm is not None:
                self.symm = symm

        @property
        def arb_srate(self):
            # sampling rate for arbitrary waveform output
            if 'DG4202' in self.afg.identity:
                # Sampling rate is fixed
                return 500e6 # 1/[s]
            else:
                return float(self.afg.resource.query(f":SOURce{self.number}:FUNCtion:SEQuence:SRATe?").strip())
            
        @arb_srate.setter
        def arb_srate(self, value):

            if 'DG2102' in self.afg.identity:

                extrema = ['minimum', 'maximum', 'min', 'max']

                if isinstance(value, (int, float)):
                    if not 2e3 <= value <= 60e6:  # [Sa]/[s]
                        raise Exception(("Sample rate"
                        " for arbitrary waveform must be between 2 kSa/s and 60 MSa/s.")) 
                elif value.casefold() in [extremum.casefold() for extremum in extrema]:
                    pass
                else:
                    raise Exception("Invalid sample rate specified.") 
                
                self.afg.resource.write(f":SOURce{self.number}:FUNCtion:SEQuence:SRATe {value}")

            elif 'DG4202' in self.afg.identity:
                raise Exception("Sample rate cannot be specifed for DG4202.") 

        def arbitrary(self, waveform, stepbystep = True):
            """Transmit user-defined arbitrary waveform 
            to digital function generator.

            Args:
                waveform (numpy.ndarray): 1-D array containing points
                    defining the waveform. Points correspond to 
                    vertical levels of instrument and are cast to integer
                    values. Points must be within range defined by
                    instrument. The allowed waveform length depends on 
                    the instrument.
                stepbystep (bool, optional): If True, enable step-by-step 
                    output of arbitrary waveform for DG4202. In this mode,
                    every point of waveform is output at sample rate.
                    The DG2102 always outputs arbitrary waveforms in 
                    step-be-step mode.

            Raises:
                Exception: For the DG2102, waveform must contain 
                    between 8 and 16777216 points.
                Exception: For the DG2102, waveform values must be 
                    in range [-32768, 32767].
                Exception: For the DG4202, the waveform must contain 
                    16384 points.
                Exception: For the DG4202, waveform values must be 
                    in range [0, 16383].
                    
            Returns:
                int: length of waveform
            """    
            # Check input and raise exceptions

            if not all(level.is_integer() for level in waveform):
                warnings.warn(("Waveform defined with noninteger values,"
                " casting to int16 (unsigned short). Consider rounding!"))
                # values will be cast as np.ndarray.astype('h')
            
            # Below, astype(int) is used instead of astype('h'),
            # as the latter permits "round tripping",
            # e.g., np.float64(2**15).astype('h') returns np.int16(-32768)
            max_level = np.max(waveform).astype(int)
            min_level = np.min(waveform).astype(int)

            if 'DG2102' in self.afg.identity:
                if not 8 <= len(waveform) <= 2**24:
                    raise Exception("Waveform must contain between 8 and 16777216 points.")
                if max_level > 2**15 - 1 or min_level < -2**15:
                    raise Exception("Waveform values must be in range [-32768, 32767].")
                if max_level != 2**15 - 1 or min_level != -2**15:
                    warnings.warn("Not exploiting full dynamic range of 2^16 levels for waveform.")

            elif 'DG4202' in self.afg.identity:
                if len(waveform) != 2**14:
                    raise Exception("Waveform must contain 16384 points.")
                if max_level > 2**14 - 1 or min_level < 0:
                    raise Exception("Waveform values must be in range [0, 16383].")
                if max_level != 2**14 - 1 or min_level != 0:
                    warnings.warn("Not exploiting full dynamic range of 2^14 levels for waveform.")

            # Transfer data in packets of 2**14 points
            # plus final packet containing remaining points 

            # Split data into packets
            n_full = len(waveform) // 2**14
            extra_pts = len(waveform) % 2**14

            if extra_pts == 0:
                datapacks = np.split(waveform, n_full)
            else:
                datapacks = np.split(waveform[:-extra_pts], n_full)
                datapacks.append(waveform[-extra_pts:])

            # Transmit data to instrument
            for pack in tqdm(datapacks[:-1]):
                _ = self.afg.resource.write_binary_values(
                    (f":SOURce{self.number}:TRACe:DATA:DAC16 VOLATILE,CON,"),
                    pack,
                    datatype='h')
                sleep(0.04)

            _ = self.afg.resource.write_binary_values(
                (f":SOURce{self.number}:TRACe:DATA:DAC16 VOLATILE,END,"),
                datapacks[-1],
                datatype='h')
            
            """When 'END' is sent, the instrument should automatically
            switch to arbitrary waveform output. There is a bug with 
            the DG2102 - if multiple datapackets are sent,
            the instrument does not automatically switch to arbitrary 
            waveform output. The following command ensures that the 
            instrument switches to output the arbitrary waveform. """

            if 'DG2102' in self.afg.identity:
                self.afg.resource.write(f":SOURce{self.number}:APPLy:SEQ")
            
            if stepbystep is True and 'DG4202' in self.afg.identity:
                self.afg.resource.write(f":SOURce{self.number}:FUNCtion:ARB:STEP")

            print(f"Loaded waveform of {len(waveform)} points.")

            print(f"Arbitrary waveform sample rate: {self.arb_srate/1e6} MHz")

            # If output is step-by-step, print waveform repetition rate
            if 'DG2102' in self.afg.identity or ('DG4202' in self.afg.identity
                                                  and self.freq == 30517.58):
                print(f"Arbitrary waveform repetition rate: {self.arb_srate/len(waveform)} Hz")

            return len(waveform)
        
        @property
        def seq_filt(self):
            # Sequence filter type
            if 'DG4202' in self.afg.identity:
                return "N/A"
            else:
                return self.afg.resource.query(f":SOURce{self.number}:FUNCtion:SEQuence:FILTer?").strip()
            
        @seq_filt.setter
        def seq_filt(self, value):

            if 'DG4202' in self.afg.identity:
                raise Exception("Sequence filter is not a valid setting for DG4202.")
            else:
                filters = ['SMOOth', 'smoo',
                           'STEP',
                           'INSErt', 'inse']
                if value.casefold() not in (filt.casefold() for filt in filters):
                    raise Exception("Invalid Sequence filter type specified.")
                
                self.afg.resource.write(f":SOURce{self.number}:FUNCtion:SEQuence:FILTer {value}")

        @property
        def mod_on(self):
            """ON/OFF status of the modulation function
            """            
            ret = self.afg.resource.query(f":SOURce{self.number}:MOD:STATe?").strip()
            if ret == "ON":
                return True
            elif ret == "OFF":
                return False
        
        @mod_on.setter
        def mod_on(self, value):
            # While DG2102 accepts ON|1|OFF|0 as parameters, DG4202 only accepts ON|OFF.
            if (isinstance(value, bool) or value == 0 or value == 1):
                if bool(value):
                    state = 'ON'
                else:
                    state = "OFF"
            elif value.casefold() == "ON".casefold():
                state = 'ON'
            elif value.casefold() == "OFF".casefold():
                state = 'OFF'
            else:
                raise TypeError("Modulation state must be boolean, 'ON', or 'OFF'")
            
            self.afg.resource.write(f":SOURce{self.number}:MOD:STATe {state}")

        @property
        def mod_type(self):
            """Modulation type
            """            
            return self.afg.resource.query(f":SOURce{self.number}:MOD:TYPe?").strip()

        @mod_type.setter
        def mod_type(self, value):
            mod_types = ['AM', 'FM', 'PM', 'ASK', 'FSK', 'PSK', 'PWM']
            if value.casefold() not in (t.casefold() for t in mod_types):
                    raise Exception("Invalid modulation type specified.")
            self.afg.resource.write(f":SOURce{self.number}:MOD:TYPe {value}")

        class PM():
            def __init__(self, channel): 
                # channel is outer class instance
                self.channel = channel

            @property
            def on(self):
                """ON/OFF status of phase modulation
                """            
                if 'DG4202' in self.channel.afg.identity:
                    raise Exception("DG4202 does not have commands to set/query state of specific modulation types. " \
                    "Use mod_on property of Channel class instead.")
                else:
                    ret = self.channel.afg.resource.query(f":SOURce{self.channel.number}:MOD:PM:STATe?").strip()
                    if ret == "ON":
                        return True
                    elif ret == "OFF":
                        return False
            
            @on.setter
            def on(self, value):
                if 'DG4202' in self.channel.afg.identity:
                    raise Exception("DG4202 does not have commands to set/query state of specific modulation types. " \
                    "Use mod_on property of Channel class instead.")
                else:
                    if (isinstance(value, bool) or value == 0 or value == 1):
                        self.channel.afg.resource.write(f":SOURce{self.channel.number}:MOD:PM:STATe {int(value)}")
                    elif value.casefold() == "ON".casefold():
                        self.channel.afg.resource.write(f":SOURce{self.channel.number}:MOD:PM:STATe ON")
                    elif value.casefold() == "OFF".casefold():
                        self.channel.afg.resource.write(f":SOURce{self.channel.number}:MOD:PM:STATe OFF")
                    else:
                        raise TypeError("PM modulation state must be boolean, 'ON', or 'OFF'")

            @property
            def deviation(self):
                """PM phase deviation [degrees]
                """            
                return float(self.channel.afg.resource.query(f":SOURce{self.channel.number}:MOD:PM:DEViation?").strip())
            
            @deviation.setter
            def deviation(self, value):
                
                extrema = ['minimum', 'maximum', 'min', 'max']

                if isinstance(value, (int, float)):
                    if not 0 <= value <= 360:
                        raise Exception("Deviation must be between 0° and 360°")
                elif value.casefold() not in [extremum.casefold() for extremum in extrema]:
                    raise Exception("Invalid deviation specified.") 
                self.channel.afg.resource.write(f":SOURce{self.channel.number}:MOD:PM:DEViation {value}")
                  
            @property
            def freq(self):
                """PM modulation frequency
                """            
                return float(self.channel.afg.resource.query(f":SOURce{self.channel.number}:MOD:PM:INTernal:FREQuency?").strip())
            
            @freq.setter
            def freq(self, value):
                if not isinstance(value, (int, float)):
                    raise TypeError("Frequency must be int or float")
                self.channel.afg.resource.write(f":SOURce{self.channel.number}:MOD:PM:INTernal:FREQuency {value}")

            @property
            def function(self):
                """PM modulation waveform
                """            
                return self.channel.afg.resource.query(f":SOURce{self.channel.number}:MOD:PM:INTernal:FUNCtion?").strip()
            
            @function.setter
            def function(self, value):
                
                functions = ['SIN', 'SINusoid',
                             'SQU', 'SQUare',
                             'TRI', 'TRIangle',
                             'RAMP',
                             'NRAMp',
                             'NOIS', 'NOISe',
                             'USER']

                if value.casefold() not in (f.casefold() for f in functions):
                    raise Exception("Invalid modulation function specified.")  
                
                self.channel.afg.resource.write(f":SOURce{self.channel.number}:MOD:PM:INTernal:FUNCtion {value}")
            
            @property
            def source(self):
                """PM modulation source (can be internal or external)
                """            
                return self.channel.afg.resource.query(f":SOURce{self.channel.number}:MOD:PM:SOURce?").strip()
            
            @source.setter
            def source(self, value):
                
                sources = ['INT', 'INTernal',
                           'EXT', 'EXTernal',]

                if value.casefold() not in (s.casefold() for s in sources):
                    raise Exception("Source must be specified as 'Internal' or 'External'.")  
                
                self.channel.afg.resource.write(f":SOURce{self.channel.number}:MOD:PM:SOURce {value}")

        class AM():
            def __init__(self, channel): 
                # channel is outer class instance
                self.channel = channel

            @property
            def on(self):
                """ON/OFF status of amplitude modulation
                """            
                if 'DG4202' in self.channel.afg.identity:
                    raise Exception("DG4202 does not have commands to set/query state of specific modulation types. " \
                    "Use mod_on property of Channel class instead.")
                else:
                    ret = self.channel.afg.resource.query(f":SOURce{self.channel.number}:MOD:AM:STATe?").strip()
                    if ret == "ON":
                        return True
                    elif ret == "OFF":
                        return False
            
            @on.setter
            def on(self, value):
                if 'DG4202' in self.channel.afg.identity:
                    raise Exception("DG4202 does not have commands to set/query state of specific modulation types. " \
                    "Use mod_on property of Channel class instead.")
                else:
                    if (isinstance(value, bool) or value == 0 or value == 1):
                        self.channel.afg.resource.write(f":SOURce{self.channel.number}:MOD:AM:STATe {int(value)}")
                    elif value.casefold() == "ON".casefold():
                        self.channel.afg.resource.write(f":SOURce{self.channel.number}:MOD:AM:STATe ON")
                    elif value.casefold() == "OFF".casefold():
                        self.channel.afg.resource.write(f":SOURce{self.channel.number}:MOD:AM:STATe OFF")
                    else:
                        raise TypeError("AM modulation state must be boolean, 'ON', or 'OFF'")

            @property
            def depth(self):
                """AM modulation depth [%]
                """            
                return float(self.channel.afg.resource.query(f":SOURce{self.channel.number}:MOD:AM:DEPTh?").strip())
            
            @depth.setter
            def depth(self, value):
                
                extrema = ['minimum', 'maximum', 'min', 'max']

                if isinstance(value, (int, float)):
                    if not 0 <= value <= 120:
                        raise Exception(r"AM modulation depth must be between 0% and 120%")
                elif value.casefold() not in [extremum.casefold() for extremum in extrema]:
                    raise Exception("Invalid AM modulation depth specified.") 
                self.channel.afg.resource.write(f":SOURce{self.channel.number}:MOD:AM:DEPTh {value}")
                  
            @property
            def freq(self):
                """AM modulation frequency
                """            
                return float(self.channel.afg.resource.query(f":SOURce{self.channel.number}:MOD:AM:INTernal:FREQuency?").strip())
            
            @freq.setter
            def freq(self, value):
                if not isinstance(value, (int, float)):
                    raise TypeError("Frequency must be int or float")
                self.channel.afg.resource.write(f":SOURce{self.channel.number}:MOD:AM:INTernal:FREQuency {value}")

            @property
            def function(self):
                """AM modulation waveform
                """            
                return self.channel.afg.resource.query(f":SOURce{self.channel.number}:MOD:AM:INTernal:FUNCtion?").strip()
            
            @function.setter
            def function(self, value):
                
                functions = ['SIN', 'SINusoid',
                             'SQU', 'SQUare',
                             'TRI', 'TRIangle',
                             'RAMP',
                             'NRAMp',
                             'NOIS', 'NOISe',
                             'USER']

                if value.casefold() not in (f.casefold() for f in functions):
                    raise Exception("Invalid modulation function specified.")  
                
                self.channel.afg.resource.write(f":SOURce{self.channel.number}:MOD:AM:INTernal:FUNCtion {value}")
            
            @property
            def source(self):
                """AM modulation source (can be internal or external)
                """            
                return self.channel.afg.resource.query(f":SOURce{self.channel.number}:MOD:AM:SOURce?").strip()
            
            @source.setter
            def source(self, value):
                
                sources = ['INT', 'INTernal',
                           'EXT', 'EXTernal',]

                if value.casefold() not in (s.casefold() for s in sources):
                    raise Exception("Source must be specified as 'Internal' or 'External'.")  
                
                self.channel.afg.resource.write(f":SOURce{self.channel.number}:MOD:AM:SOURce {value}")