import os
import sys
from time import time

# Directory containing GenericDevice must be in path
# Add parent directory of current file to path
parent_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, parent_directory)
# Add also directory two levels up
sys.path.insert(0, os.path.dirname(parent_directory))
from GenericDevice import _GenericDevice

import numpy as np
import matplotlib.pyplot as plt
from time import sleep

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

class SpectrumAnalyzer(_GenericDevice):
    def get_max_point(self) -> float:
        self.resource.write(':CALCulate:MARKer1:MAXimum')
        return float(self.resource.query(':CALCulate:MARKer1:X?'))

    def get_trace(self) -> np.ndarray:
        """Retrieves the trace as displayed on the screen

        Returns:
            np.ndarray: 2 numpy arrays data and frequencies
        """
        span = float(self.resource.query(':FREQuency:SPAN?'))
        center = float(self.resource.query(':FREQuency:CENTer?'))
        self.resource.write(':FORMat:TRACe:DATA ASCii')
        data = self.query_data()
        freqs = np.linspace(center-span/2, center+span/2, len(data))
        return data, freqs

    def zero_span(self, center: float = 1e6, rbw: int = 100,
                  vbw: int = 30, swt: float = 'auto', 
                  trig: bool = None, single = False,
                  plot: bool = False, barrier = None, arm_only = False):
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
        :param barrier: <multiprocessing.Barrier> instance,
           useful for synchronizing multiple processes (measurements)
        :param bool arm_only: if True, function returns after arming trigger
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
                if barrier is not None:
                        barrier.wait()
                        self.resource.write(':INITiate:IMMediate; *OPC')
                        print(f'{self.short_name} | Waiting for trigger as of {time()} s')
                else:
                    self.resource.write(':INITiate:IMMediate; *OPC')
            elif sweep_state == 0:
                if trig is None or trig == False:
                    self.resource.write(':INITiate:IMMediate; *OPC')
                elif trig == True:
                    # Must reset trigger just before initiating scan, otherwise
                    # it seems trigger success condition is stored, because scan
                    # starts immediately instead of waiting for next trigger
                    self.resource.write(':TRIGger:SEQuence:SOURce EXTernal')
                    if barrier is not None:
                        barrier.wait()
                        self.resource.write(':INITiate:IMMediate; *OPC')
                        print(f'{self.short_name} | Waiting for trigger as of {time()} s')
                    else:
                        self.resource.write(':INITiate:IMMediate; *OPC')
            triginfo_msg = ' on trigger' if set_trigstate == 'EXT' else ''
            if arm_only:
                print(f"{self.short_name} | Armed for zero span scan (perform single sweep on trigger)")
                return
            print(f'{self.short_name} | Getting zero span scan (single sweep' +
                  triginfo_msg + ')')
            # Poll the operation complete status in the event status 
            # register (ESR). (Once all commands before *OPC have been executed, 
            # the operation complete bit in the ESR is set to 1)
            while int(self.resource.query('*ESR?')) != 1:
                sleep(0.1)
        print(f"{self.short_name} | Zero span scan complete as of {time()} s")
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

    def spectrum(self, center: float = 22.5e6, span: float = 45e6,
                                     rbw: int = 100,
                                     vbw: int = 30, swt: float = 'auto',
                                     trig: bool = None, single = False,
                                     plot: bool = False) -> np.ndarray:
        """Configure and execute measurement of noise power spectrum

        THIS FUNCTION WAS PREVIOUSLY NAMED "SPAN"
        THIS FUNCTION REPLACES NOW DEPRECATED FUNCTION <set_trace_parameters_and_get>

        This function should work identically to the <span> function defined in
        SpectrumAnalyzer class of RigolInterface.py

        (!) For long sweep times, use single sweep mode
        
        :param float center: Center frequency in Hz
        :param float span: span in Hz
        :param float rbw: Resolution bandwidth in Hz
        :param float vbw: Video bandwidth in Hz
        :param float swt: Total measurement time in s
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

    def query_data(self) -> np.ndarray:
        """Lower level function to grab the data from the SpecAnalyzer

        :return: data
        :rtype: list

        """
        rawdata = self.resource.query(':TRACe:DATA? TRACE1')
        data = rawdata.split(',')[:]
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

    ### Channel Power Measurement

    def chpower_read(self, density = False):
        """Returns a single value that corresponds to the Channel Power 
        or Power Spectral Density.
        Does not preset the measurement to the factory default settings.
        Initiates the measurement and puts valid data into the output buffer.

        Uses integration bandwidth (IBW) method - important to set the 
        resolution bandwidth correctly before making this measurement
        using the following formula: RBW = k(span)/n Where k is 
        a value between 1.2 and 4.0 and n is the number of trace points.
        VBW should be ≥ 10 times the RBW. See reference manual.

        Args:
            density (bool, optional): If true, returns Power Spectral Density. 
            Defaults to False.

        Returns:
            float: _channel power (dBm)
        """
        if density is False:
            return float(self.resource.query(":READ:CHPower:CHPower?").strip())
        else: 
            return float(self.resource.query(":READ:CHPower:DENSity?").strip())
            
    @property
    def chpower_avg_on(self):
        """"ON/OFF status of averaging
        """
        return bool(int(self.resource.query(":SENSe:CHPower:AVERage:STATe?").strip()))

    @chpower_avg_on.setter
    def chpower_avg_on(self, value):
        if (isinstance(value, bool) or value == 0 or value == 1):
            self.resource.write(f":SENSe:CHPower:AVERage:STATe {int(value)}")
        elif value.casefold() == "ON".casefold():
            self.resource.write(":SENSe:CHPower:AVERage:STATe ON")
        elif value.casefold() == "OFF".casefold():
            self.resource.write(":SENSe:CHPower:AVERage:STATe OFF")
        else:
            raise ValueError("Averaging must be set as boolean, 'ON', or 'OFF'")

    @property
    def chpower_avg_num(self):
        """"Number of measurements for averaging
        """
        return int(self.resource.query(":SENSe:CHPower:AVERage:COUNt?").strip())

    @chpower_avg_num.setter
    def chpower_avg_num(self, value):
        if not isinstance(value, int):
            raise TypeError("Number of measurements for averaging must be specifed with integer")
        self.resource.write(f":SENSe:CHPower:AVERage:COUNt {value}")

    @property
    def chpower_avg_mode(self):
        """"Determines the averaging action after the specified number of measurements 
        (average count) is reached:

        Exponential Averaging mode: each successive data acquisition after the average count is 
        reached is exponentially weighted and combined with the existing average. 
        Exponential averaging weights new data more than old data, which facilitates tracking of 
        slow-changing signals. The average will be displayed at the end of each sweep.

        Repeat mode: after reaching the average count, all previous result data is cleared and the 
        average count is set back to 1
        """
        return self.resource.query(":SENSe:CHPower:AVERage:TCONtrol?").strip()

    @chpower_avg_mode.setter
    def chpower_avg_mode(self, value):
        modes = ['EXPonential', 'EXP', 'REPeat', 'REP']
        if value.casefold() not in (mode.casefold() for mode in modes):
            raise Exception("Invalid averaging mode specified.")  
        self.resource.write(f":SENSe:CHPower:AVERage:TCONtrol {value}")

    @property
    def chpower_integbw(self):
        """Range of integration used in calculating the power in the channel
        """            
        return float(self.resource.query(":SENSe:CHPower:BANDwidth:INTegration?").strip())
    
    @chpower_integbw.setter
    def chpower_integbw(self, value):      
        self.resource.write(f":SENSe:CHPower:BANDwidth:INTegration {value}")

    @property
    def chpower_span(self):
        """Analyzer span for the channel power measurement
        """            
        return float(self.resource.query(":SENSe:CHPower:FREQuency:SPAN?").strip())
    
    @chpower_span.setter
    def chpower_span(self, value):      
        self.resource.write(f":SENSe:CHPower:FREQuency:SPAN {value}")

    def power_autorange(self):
        """Sets the input attenuator and reference level to optimize the 
        robustness of the measurement, which is its freedom from errors 
        due to input compression and log amp range limitations.
        """        
        self.resource.write("POW:RANG:AUTO ONCE")