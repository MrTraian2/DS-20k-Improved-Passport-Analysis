#!/usr/bin/env python3

import sys
import os, glob
from manager import Manager
import time, re
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib.patches as mpatches
import os



class AnaNA:
    def __init__(self):
        self.m = Manager(midas=True, elec=False, tpc=True, toy=False, mc_gain=False, pdm_reco=True, is_mc=False, is_physics=False)
        self.m.info('vPDU MIDAS Data Reconstruction')

        # getting info from .ini files
        self.sampling         = self.m.config('daq', 'sampling', 'float')
        self.nevents          = self.m.config('base', 'nevents', 'float')
        self.baseline_to      = self.m.config('pdm_reco', 'baseline_to', 'float')
        self.downsample_by    = self.m.config('pdm_reco', 'downsample_by', 'int')
        self.reconstructed_as = self.m.config('pdm_reco', 'reconstructed_as', 'str')
        self.polarity         = self.m.config('pdm_reco', 'polarity', 'float')
        self.running_gate     = self.m.config('pdm_reco', 'running_gate', 'float')
        self.threshold_in_rms = self.m.config('pdm_reco', 'threshold_in_rms', 'int')
        self.min_time_between_peaks     = self.m.config('pdm_reco', 'min_time_between_peaks', 'float')
        self.extended_time_for_integral = self.m.config('pdm_reco', 'extended_time_for_integral', 'float')
        self.add_left_time_for_integral = self.m.config('pdm_reco', 'add_left_time_for_integral', 'float')
        self.trigger_pos                = self.m.config('pdm_reco', 'trigger_pos', 'float')
        self.roi_left                   = self.m.config('pdm_reco', 'roi_left', 'float')
        self.roi_right                  = self.m.config('pdm_reco', 'roi_right', 'float')
        self.channel_to_plot            = int(self.m.config('pdm_reco', 'channel_to_plot', 'int'))
        
        if self.downsample_by > 1 : self.sampling /= self.downsample_by

        # conversion from seconds to samples
        self.baseline_to                    = round(self.baseline_to * self.sampling)
        self.running_gate                   = round(self.running_gate * self.sampling)
        self.min_samples_between_peaks      = round(self.min_time_between_peaks * self.sampling)
        self.extended_samples_for_integral  = round(self.extended_time_for_integral * self.sampling)
        self.add_left_samples_for_integral  = round(self.add_left_time_for_integral * self.sampling)
        self.trigger_pos_samples            = round(self.trigger_pos * self.sampling)
        self.roi_left_samples               = round(self.roi_left * self.sampling)
        self.roi_right_samples              = round(self.roi_right * self.sampling)
        self.roi_tot_samples                = self.roi_right_samples - self.roi_left_samples

        self.timebin = 1e9 / self.sampling  # ns
        self.n_trig_events = self.m.config('roi', 'n_trigs', 'int')

        # getting the data type from the input file extension (case insensitive)
        input_path = self.m.config.input[0]
        file_extension = os.path.splitext(self.m.config.input[0])[1].lower()
        self.file_type = "unknown"
        
        #Added this section to allow the option of using directories in the files -i input for multiple files reconstruction
        if os.path.isdir(input_path):
          file_extension = os.path.splitext(glob.glob(f'{self.m.config.input}/*mid*')[0])[1].lower()
        else:
          file_extension = os.path.splitext(input_path)[1].lower()

        file_types = {
        ".root": "rootfile",
        ".lz4": "midas",
        ".mid": "midas",
        ".wfm": "wfm"
        }
        
        self.file_type = file_types.get(file_extension, "unknown")  # Default to "unknown" if not found
        print("Using", self.file_type, "data.")

    # returns the sample index of the max amplitude of first given number of events in the root file
    def get_trigger_position(self):
        is_first = True
        n = 0
        wf_list = []
        for i, wf in enumerate(self.events):
            if wf is None:
                continue
            if wf.nchannels == 0:
                continue
            if n > self.n_trig_events:
                break
            if is_first:
                wf_all = wf.adc_data
                print('First event:', wf.midas_event)
                is_first = False
            else:
                wf_list.append(wf.adc_data)
            n += 1

        if len(wf_list) == 0:
            raise ValueError("No waveforms found")

        wf_all = np.concatenate(wf_list, axis=0)
        if len(wf_all.shape) != 2:
            raise ValueError("Invalid dimensions for wf_all array")

        wf_all = wf_all
        trig_pos = int(np.argmax(np.mean(wf_all, axis=0),axis=0))
        self.m.info(f'trigger position: {trig_pos} samples')
        return trig_pos
    

    def plot_one_channel(self, wf, pulses, trigger_position , event_number):
        nsamp = wf.size
        t = np.arange(0., nsamp * self.timebin, self.timebin)
        line_1, = plt.plot(t,wf, color='darkorange', label='Baseline Subtracted Waveform')
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        plt.plot([xmin, xmax], [0, 0], ls='--', lw=2, color='darkgray', zorder=-10)
        ax.add_patch(mpatches.Rectangle(((trigger_position - self.roi_left_samples) * self.timebin, ymin),
                                         (self.roi_left_samples + self.roi_tot_samples) * self.timebin, ymax - ymin,
                                         fc='darkgray', alpha=0.3, color='darkgray', lw=2))
        plt.title("Waveform number " + str(event_number) + ", channel # " + str(self.channel_to_plot))
        plt.ylabel('Amplitude')
        plt.xlabel('Time [ns]')
        roi_patch = mpatches.Patch(alpha=0.3, color='darkgray', label='Region of interest')
        if pulses.size > 0:
            mark = plt.scatter(pulses[:, 0], pulses[:, 1], s=80, facecolors='none', edgecolors='r', label='hit')
            plt.legend(handles=[line_1, roi_patch, mark])
        else:
            plt.legend(handles=[line_1, roi_patch])
        fig = plt.gcf()
        fig.set_size_inches(11, 8)
        fig.tight_layout()
        # Save the figure to a file
        import os; os.makedirs("waveforms", exist_ok=True)
        plt.savefig(f"waveforms/wf_plot_{event_number}-ch{self.channel_to_plot}.png")
        # Close the figure
        plt.close()
        # Additionally, save the waveform data to a text file
        #self.save_waveform_to_txt(wf, event_number)

    def reco(self):
        # evaluate execution time
        t0 = time.time()
        t1 = t0
        self.m.info(f'{sys._getframe().f_code.co_name} started at {time.strftime("%H:%M on %d %B %Y", time.localtime(t0))}')
        # get the MIDAS object
        self.events = self.m.midas
        empty_event = 0
        tot_bytes = 0

        # calculate trigger position, aka "Auto-ROI"
        # NOTE: get_trigger_position() consumes the first "n_trig_events" events
        if self.trigger_pos>=0:
            trigger_position = self.trigger_pos_samples
            print(f"Trigger position set at {trigger_position} samples")
        if self.trigger_pos<0:
            print(f"Trigger position was not defined. Using the get_trigger_position() function.")
            trigger_position = self.get_trigger_position()
            
        start_roi = trigger_position - self.roi_left_samples

        # measure total run time
        run_duration = 0.0

        first_event = True
        # event loop - main loop
        peaks = []
        nonsense = 0
        for nev, event in enumerate(self.events):

            if event is None:
                empty_event += 1
                continue

            if event.nchannels == 0:
                empty_event += 1
                continue

            # End reconstruction if nevents have been reconstructed
            if self.nevents>0 and nev > self.nevents:
                break
            
            if nev % 1000 == 0 and nev > 0:  # progress print
                print(f'{nev:6d} events {time.time() - t1:1.3f}s / 1000 ev')
                t1 = time.time()

            # verbose information about midas event
            # event.info()

            nchan, nsamp = event.adc_data.shape
            tot_bytes += (nchan * nsamp * 2)
            if first_event:
                print("First event:", event.midas_event, " -->  Number of channels is", nchan, "- Number of samples is", nsamp)
                first_event = False

            # at this moment the saturation is just = 0 for all channels
            channel = np.arange(nchan, dtype=int)
            saturation = np.zeros(nchan, dtype=int)

            # calculate baseline
            bal = self.m.algos.get_baseline(event.adc_data, gate=self.baseline_to)
            rms = self.m.algos.get_rms(event.adc_data, gate=self.baseline_to)
            # subtract baseline, signals have negative polarity
            wfs = (event.adc_data - bal)

            # ROI integration using gate from config file
            roi = self.m.algos.get_roi(wfs, gate=self.roi_tot_samples, start=start_roi)
            roi = np.concatenate(roi)

            # WF analysis using the ARMA filter
            # [[time, amplitude, charge, channel],...]
            #peaks = self.m.tpc.get_hits(channel, wfs)

            # WF analysis using the running mean (optimal for ASIC readout)
            if self.reconstructed_as == 'asic' :
              run      = self.m.algos.running_mean(wfs, gate=self.running_gate)
              step     = np.where(run > self.threshold_in_rms*self.m.algos.get_rms(run, gate=self.baseline_to), 1, 0)
              segments = self.m.algos.get_segments(step, min_samples_to_merge=self.min_samples_between_peaks)
             
              # peaks is an array with 4 columns:
              # (0) time of the peak, (1) max height of the peak, (2) integral of the peak, (3) channel number of the peak
              peaks = np.zeros(shape=(len(segments),4))
              if segments.any() == False: # if 0 segments found fill peaks with -1s
                peaks[0,:] = [-1, -1, -1, 0]
              else:
                if self.file_type == 'rootfile' or self.file_type == 'wfm': segments = segments[:,1::]
                for i,s in enumerate(segments): # fill peaks with values for every segment found for given event
                  charge_left = 0
                  if self.file_type == 'rootfile' or self.file_type == 'wfm':
                    if (s[0] - self.add_left_samples_for_integral > 0): charge_left = s[0] - self.add_left_samples_for_integral
                    peaks[i,:] = [s[0], np.max(run[s[0]:s[1]]), np.sum(run[charge_left:s[1]+self.extended_samples_for_integral]), 0]
                  if self.file_type == 'midas':
                    if s is None: 
                        print(f'Nonsense segment in event {nev}, skipping...')
                        nonsense += 1
                        continue
                    else:               
                        charge_left = max(0, s[1] - self.add_left_samples_for_integral)
                        charge_right = min(len(run[s[0]]), s[2]+self.extended_samples_for_integral)
                        peaks[i,:] = [s[1], #Start position of peak
                                    np.max(run[s[0]][s[1]:s[2]]), #Peak max amplitude
                                    np.sum(run[s[0]][charge_left:charge_right]), #peak charge
                                    s[0]] #Channel index

            # trigger time stamp
            if event.trigger_time > run_duration:
                run_duration = event.trigger_time

            if self.file_type == 'midas':
                bl  = bal.flatten()
                rms = rms.flatten()

            # filling the output root file
            if (nev==0) :
                self.m.tree.fill_header( nchannels = len(channel) )
            self.m.tree.fill_event_header(nev)
            self.m.tree.fill_channels_pdu( channel, bl, rms, roi/self.sampling, saturation )
            if (len(peaks)>0 and peaks[0,0] > -1 ) :
                peaks[:,2] /= self.sampling # normalizes the charge integral with the time bin width
                self.m.tree.fill_peaks(peaks)
                #Eleason comment out for now
                #self.m.tree.fill_peaks_by_channel( peaks, channel)
            self.m.event.add_event()

            if self.channel_to_plot != -1:
                # plot one channel
                wf = wfs[self.channel_to_plot]
                pulses = []
                for p in peaks:
                    print(f'Event {nev}: Peak found in CH {int(p[3])} at {p[0] * self.timebin:1.0f}, amplitude: {p[1]:1.0f}')
                    if p[3] == self.channel_to_plot:
                        pulses.append([p[0] * self.timebin, p[1]])
                pulses = np.array(pulses)
                self.plot_one_channel(wf, pulses, trigger_position=trigger_position, event_number=nev)

        self.m.event.save()
        self.m.info(f'Run duration: {run_duration * 1e-9:1.2f}s')
        self.m.info(f'Events processed: {nev}')
        self.m.info(f'Processing {tot_bytes * 1e-6:1.3f}MB took {time.time() - t0:1.3f}s')
        self.m.info(f'Number of nonsense segments: {nonsense}\n')


if __name__ == '__main__':
    r = AnaNA()
    r.reco()
