#Uses the passport_auto.sh file to run the passport analysis
import concurrent.futures #import used to parallelise analysis
import argparse
import ast
import glob
import csv
import numpy as np
import subprocess
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from datetime import date
from scipy import stats
import sys
sys.path.append("../")

import csv
from scipy import interpolate
import os
import configparser

import AnalysisTools.CharacterisationTools as analysis
import AnalysisTools.Algos as algo
import AnalysisTools.IVAnalysis as IVAnalysis
import AnalysisTools.BedofNailsAnalysis as BoN
from AnalysisTools.CharacterisationTools import gaussian, sum_of_gaussians, fit_gaussian_peaks

###########
#Functions
###########
def calculate_position_2x2(i):
    # Direct mapping for filling in a 2x2 grid
    # Order: top-left, top-right, bottom-left, bottom-right
    mapping = [(0, 0), (0, 1), (1, 0), (1, 1)]
    return mapping[i % 4]

def calculate_position_4x4(i):
    # Determine the quadrant (0 to 3)
    quadrant  = i // 4
    # Position within the quadrant (0 to 3), following the specified order
    position  = i % 4
    # Mapping for the specified filling order within a quadrant
    mapping   = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # Calculate base row and column for each quadrant
    base_row  = (quadrant // 2) * 2
    base_col  = (quadrant % 2) * 2
    # Offset within the quadrant
    offset    = mapping[position]
    row       = base_row + offset[0]
    col       = base_col + offset[1]
    return row, col

# Get the git version of the repository to track analysis with different passport versions
try:
    version = subprocess.check_output(['git', 'describe'])
    GitVersion = str(version, 'utf-8').strip()
except:
    print("Error reading Git tag")
    GitVersion = "Unknown"
print('Git tag:', GitVersion)
    
today = date.today()
datestr = today.strftime("%d_%m_%Y")

def process_channel_data(x,pkp,pkt,charge,chrms,channel,channelrms,row=None,col=None):
    config = configparser.ConfigParser()
    if args.ini_file:  # Check if custom .ini file is provided
        config.read(args.ini_file)
        print("Using config file: ", args.ini_file)
    else:
        config.read('AnalysisConfig.ini')  # Default .ini file
        #print("Using default config file")
    # Extract values from the 'PE Fitting' section
    Amp1Bins                     = int(config  ['PE Fitting']['1PeBins'])
    Amp1histrangelower           = float(config['PE Fitting']['1PEhistrangelow'])
    Amp1histrangeupper           = float(config['PE Fitting']['1PEhistrangeup'])
    Amp1PedestalLowerThreshold   = int(config  ['PE Fitting']['1PEPedestalLowerThreshold'])
    Amp1PedestalUpperThreshold   = int(config  ['PE Fitting']['1PEPedestalUpperThreshold'])

    Amp2Bins                     = int(config  ['PE Fitting']['2PEBins'])
    Amp2PedestalLowerThreshold   = int(config  ['PE Fitting']['2PEPedestalLowerThreshold'])
    Amp2PedestalUpperThreshold   = int(config  ['PE Fitting']['2PEPedestalUpperThreshold'])
    Amp2histrangelower           = float(config['PE Fitting']['2PEhistrangelow'])
    Amp2histrangeupper           = float(config['PE Fitting']['2PEhistrangeup'])

    ChargeBins                   = int(config  ['PE Fitting'][ '1PEChargeBins'])
    ChargePedestalLowerThreshold = int(config  ['PE Fitting']['1PEChargePedestalLowerThreshold'])
    ChargePedestalUpperThreshold = int(config  ['PE Fitting']['1PEChargePedestalUpperThreshold'])
    Chargehistrangelower         = float(config['PE Fitting']['1PEChargehistrangelow'])
    Chargehistrangeupper         = float(config['PE Fitting']['1PEChargehistrangeup'])

    if Amp1PedestalLowerThreshold       == -1:
        Amp1PedestalLowerThreshold      = None
    if Amp1PedestalUpperThreshold       == -1:
        Amp1PedestalUpperThreshold      = None
    if Amp2PedestalLowerThreshold       == -1:
        Amp2PedestalLowerThreshold      = None
    if Amp2PedestalUpperThreshold       == -1:
        Amp2PedestalUpperThreshold      = None
    if ChargePedestalLowerThreshold     == -1:
        ChargePedestalLowerThreshold    = None
    if ChargePedestalUpperThreshold     == -1:
        ChargePedestalUpperThreshold    = None


    TriggerPos, std, trig_err = pa.GetTriggerPosition(
    pkt[(pkt > 0) & (channel == x)].flatten(),
    nBins=500,
    method="max"
    )
    Trig_left               = TriggerPos - pa.PromptRoiLeft*pa.SamplingRate
    Trig_right              = TriggerPos + pa.PromptRoiRight*pa.SamplingRate
    DN_start                = DN_start_from_trigger + TriggerPos/SamplingRate
    Trigger_ROI_left        = Trig_left   #sample number
    Trigger_ROI_right       = Trig_right #sample number
    pkp_trig                = pkp[(pkt>Trigger_ROI_left) & (pkt<Trigger_ROI_right) & (channel == x)]
    pkt_trig                = pkt[(pkt>Trigger_ROI_left) & (pkt<Trigger_ROI_right) & (channel == x)]
    charge_trig             = charge[(pkt>Trigger_ROI_left) & (pkt<Trigger_ROI_right) & (channel == x)]
    print("Trigger Position      :",TriggerPos,"+/-",trig_err)
    print("Trigger ROI           :",Trig_left,",",Trig_right)
    #### Filling the plots
    if plot == "True":
        axsC[row,col].hist(charge[(charge>-9999) & (channel == x)].flatten(),bins=500,histtype="step",color = "m",range = [0,0.005])
        axsC[row,col].set_title("Channel {} Charge of Pulse".format(x+1))
        axsC[row,col].set_xlabel("Charge [arb.]")
        axsC[row,col].set_ylabel("Counts")
        axsC[row,col].text(0.95, 0.75, "Git Tag: {}".format(GitVersion),
            horizontalalignment='right', verticalalignment='top', transform=axsC[row,col].transAxes,fontsize=10)

        axsRMS[row,col].hist(ADCmVConv/1000*chrms[(chrms>-9999) & (channelrms == x)].flatten(),bins=200,histtype="step",color = "m", label= "RMS Mean: {}mV".format(ADCmVConv/1000*np.mean(chrms[channelrms == x].flatten())),range = [0,0.02])
        axsRMS[row,col].set_title("Channel {} RMS Distribution".format(x+1))
        axsRMS[row,col].set_xlabel("RMS [V]")
        axsRMS[row,col].set_ylabel("Counts")
        axsRMS[row,col].set_yscale('log')
        axsRMS[row,col].text(0.95, 0.75, "Git Tag: {}".format(GitVersion),
            horizontalalignment='right', verticalalignment='top', transform=axsRMS[row,col].transAxes,fontsize=10)
        axsRMS[row,col].get_legend()

        axsT[row,col].hist(pkt[(pkt>-9999) & (channel == x)].flatten()/pa.SamplingRate*1e6,bins=200,histtype="step",color = "m")
        axsT[row,col].set_title("Channel {} Time of Hits".format(x+1))
        axsT[row,col].set_xlabel("Time [\u03BCs]")
        axsT[row,col].text(0.95, 0.75, "Git Tag: {}".format(GitVersion),
            horizontalalignment='right', verticalalignment='top', transform=axsT[row,col].transAxes,fontsize=10)
        axsT[row,col].set_ylabel("Counts")
        axsT[row,col].set_yscale("log")
        axsT[row,col].axvline(Trig_right/pa.SamplingRate*1e6, color="k")
        axsT[row,col].axvline(Trig_left/pa.SamplingRate*1e6, color="k")

        axsAMP[row,col].hist(ADCmVConv/1000*pkp[(pkp>-9999) & (channel == x)].flatten(),bins=500,histtype="step",color = "m",range=[0,0.2])
        axsAMP[row,col].set_title("Channel {} Amplitude of Hits".format(x+1))
        axsAMP[row,col].set_xlabel("Amplitude [V]")
        axsAMP[row,col].text(0.95, 0.75, "Git Tag: {}".format(GitVersion),
            horizontalalignment='right', verticalalignment='top', transform=axsAMP[row,col].transAxes,fontsize=10)
        axsAMP[row,col].set_ylabel("Counts")

        axsAMPROI[row,col].hist(ADCmVConv/1000*pkp_trig.flatten(),bins=500,histtype="step",color = "m",range=[0,0.075])
        axsAMPROI[row,col].set_title("Channel {} Amplitude of Hits in trigger ROI".format(x+1))
        axsAMPROI[row,col].set_xlabel("Amplitude [V]")
        axsAMPROI[row,col].text(0.95, 0.75, "Git Tag: {}".format(GitVersion),
            horizontalalignment='right', verticalalignment='top', transform=axsAMPROI[row,col].transAxes,fontsize=10)
        axsAMPROI[row,col].set_ylabel("Counts")

        ##Below lines are for Amplitude_GS plot
        #axsGS[row,col].hist(ADCmVConv*pkp[(pkp>-9999) & (channel == x)].flatten(),bins=500,histtype="step",color = "m",range=[0,200])
        axsGS[row,col].set_title("Channel {} Amplitude of Hits".format(x+1))
        #axsGS[row,col].set_xlabel("Amplitude")
        axsGS[row,col].text(0.90, 0.835, "Git Tag: {}".format(GitVersion),
             horizontalalignment='right', verticalalignment='top', transform=axsGS[row,col].transAxes,fontsize=10)
        #axsGS[row,col].set_ylabel("count[#]")

    elif plot == "False":
        pass

    ############################################
    #1PE and 2PE amplitude and charge extraction
    ############################################
    
    if plot == "True":
         
        try:
            Amp1peMean, Amp1peStd, Amp1peError, Amp1peStdError, chi2, ndof, countdens = pa.Get1pePeak(
                pkp_trig.flatten(),
                bins=500,
                PedestalThreshold=0,
                histRange=[0, 4000],
                verbose=False,
                Plot=ast.literal_eval(plot),
                upper_threshold=750,
                axis=axs1PE[row, col]
            )
            #pe_threshold_upperLimit = Amp1peMean + 2*Amp1peStd
            #pe_thresh_lowerLimit    = Amp1peMean - 2*Amp1peStd
        except:
            print("Encountered error in 1PE fitting, skipping...")
            Amp1peMean      = np.nan
            Amp1peStd       = np.nan
            Amp1peError     = np.nan
            Amp1peStdError  = np.nan
            chi2            = np.nan
            ndof            = np.nan

            axs1PE[row, col].text(0.5, 0.5, 'Error in 1PE Fitting\nSkipping...', ha='center', va='center', transform=axs1PE[row, col].transAxes)
            axs1PE[row, col].set_xticks([])
            axs1PE[row, col].set_yticks([])

        # ---- SAFE FALLBACK THRESHOLDS FROM 1PE FIT ----
        if not np.isnan(Amp1peMean) and not np.isnan(Amp1peStd):
            position_largest = Amp1peMean
            position_largest_std = Amp1peStd
            position_largest_error = Amp1peError

            pe_th_lower = Amp1peMean - 2 * Amp1peStd
            pe_th_upper = Amp1peMean + 2 * Amp1peStd
        else:
            position_largest = np.nan
            position_largest_std = np.nan
            position_largest_error = np.nan

            pe_th_lower = np.nan
            pe_th_upper = np.nan

        try:
            Amp2peMean, Amp2peStd, Amp2peError, Amp2peStdError, chi2_2, ndof_2, countdens = pa.Get1pePeak(
                pkp_trig.flatten(),
                bins=500,
                PedestalThreshold=Amp1peMean + 2 * Amp1peStd if not np.isnan(Amp1peMean) and not np.isnan(Amp1peStd) else 0,
                histRange=[0, 4000],
                verbose=False,
                Plot=ast.literal_eval(plot),
                upper_threshold=None,
                axis=axs2PE[row, col]
            )
            TwoPE_threshold_upperLimit = Amp2peMean + 2*Amp1peStd
            TwoPE_thresh_lowerLimit    = Amp2peMean - 2*Amp1peStd
        except:
            print("Encountered error in 2PE fitting, skipping...")
            Amp2peMean      = np.nan
            Amp2peStd       = np.nan
            Amp2peError     = np.nan
            Amp2peStdError  = np.nan
            chi2_2          = np.nan
            ndof_2          = np.nan

            axs2PE[row, col].text(0.5, 0.5, 'Error in 2PE Fitting\nSkipping...', ha='center', va='center', transform    =axs2PE[row, col].transAxes)
            axs2PE[row, col].set_xticks([])
            axs2PE[row, col].set_yticks([])
        floatarray          = np.array(pkp_trig.flatten(), dtype=float)
        all_peak_data       = []
        snr_per_quadrant    = {}
    
        try:
            rms = np.mean(chrms[(channelrms==x) & (chrms>-9999)].flatten())
        except:
            print("Encountered error in RMS calculation, skipping...")
            rms = np.nan
         
        try:

            # Call the updated function
            peak_parameters, peak_errors = fit_gaussian_peaks(floatarray, sigma=4, num_peaks=3, axis=axsGS[row, col])
            peak_data                    = []
            all_peak_data                = []
            for i, ((amplitude, position, stddev), (amp_error, pos_error, stddev_error)) in enumerate(zip(peak_parameters, peak_errors), start=1):
                if position >= 0:  # Exclude peaks with positions < 0
                    peak_data.append([amplitude, position, stddev, amp_error, pos_error, stddev_error])
                    all_peak_data.append([row, col, i, amplitude, position, stddev, amp_error, pos_error, stddev_error])

            # Calculate SNR for the current quadrant
            if len(peak_data) >= 2:
                # Sort peaks by amplitude in descending order
                sorted_peaks_amp = sorted(peak_data, key=lambda x: x[0], reverse=True)  # Sort by amplitude

                # Extract positions of the largest and second-largest peaks
                if sorted_peaks_amp[0][1] > 0:
                    position_largest       = sorted_peaks_amp[0][1]
                    position_largest_error = sorted_peaks_amp[0][4]  # Corresponding position error
                    position_largest_std = sorted_peaks_amp[0][2]
                    if sorted_peaks_amp[1][1] > 0:
                        position_second_largest       = sorted_peaks_amp[1][1]
                        position_second_largest_error = sorted_peaks_amp[1][4]  # Corresponding position error
                    position_other       = sorted_peaks_amp[2][1]
                    position_other_error = sorted_peaks_amp[2][4]
            
                else:
                    position_largest              = sorted_peaks_amp[1][1]
                    position_largest_error        = sorted_peaks_amp[1][4]  # Corresponding position error
                    position_second_largest       = sorted_peaks_amp[2][1]
                    position_second_largest_error = sorted_peaks_amp[2][4]  # Corresponding position error

                if (position_second_largest - position_largest)<0: 
                    position_second_largest = position_other
                    position_second_largest_error = position_other_error
                if position_second_largest > 3*position_largest:
                    position_largest = position_second_largest
                    position_second_largest = position_other
                    position_largest_error = position_second_largest_error
                    position_second_largest_error = position_other_error

                ###############################
                #  SNR from gaussian fitting  #
                ###############################

                try:
                    rms = np.mean(chrms[(channelrms==x) & (chrms>-9999)].flatten())
                except:
                    print("Encountered error in RMS calculation, skipping...")
                    rms = np.nan
                snr       = (position_second_largest - position_largest) / rms
                '''
                if snr < 0:
                    # print("I yearn for the forbidden peak")
                    # snr = (position_other - position_largest) / rms
                    position_second_largest = position_other
                    position_second_largest_error = position_other_error
                if position_second_largest > 3*position_largest:
                    # print("The pedestal would have gotten away with it too, if it weren't for you meddling physicists!")
                    # snr = (position_other - position_second_largest) / rms
                    position_largest = position_second_largest
                    position_second_largest = position_other
                    position_largest_error = position_second_largest_error
                    position_second_largest_error = position_other_error
                
                # Alternative SNR calculation (if simpler)
                snr       = (position_second_largest - position_largest) / rms
                ''' 
                snr_error = algo.ErrorDivide((position_second_largest_error - position_largest_error),rms,algo.ErrorAddition(position_largest_error,position_second_largest_error),np.std(chrms[(channelrms==x) & (chrms>-9999)].flatten())/len(chrms[(channelrms==x) & (chrms>-9999)].flatten()))
                snr_per_quadrant[(row, col)] = (snr, snr_error)
                ampData.append((position_largest * ADCmVConv)/1000)
                ampData_error.append((position_largest_error * ADCmVConv)/1000)
                #print parameters, SNR, 1pe amplitude, 2pe amplitude, RMS

                print("1PE Amplitude (V)     : {:.5f} +/- {:.5f}".format((position_largest*ADCmVConv)/1000,(position_largest_error*ADCmVConv)/1000))
                print("2PE Amplitude (V)     : {:.5f} +/- {:.5f}".format((position_second_largest*ADCmVConv)/1000,(position_second_largest_error*ADCmVConv)/1000))
                print("RMS (V)               : {:.5f} +/- {:.1e}".format((rms * ADCmVConv)/1000,(ADCmVConv * np.std(chrms[(channelrms == x) & (chrms > -9999)].flatten()) / len(chrms[(channelrms == x) & (chrms > -9999)].flatten())) / 1000))
                print("Signal-to-Noise Ratio : {:.5f} +/- {:.5f}".format(snr,snr_error))

                print("position largest      : {:.5f}".format(position_largest))
                print("position 2nd largest  : {:.5f}".format(position_second_largest))
                # Overwrite fallback ONLY if values are valid
                if not np.isnan(position_largest) and not np.isnan(position_largest_std):
                    pe_th_upper = position_largest + 2 * position_largest_std
                    pe_th_lower = position_largest - 2 * position_largest_std
                print("DCR th",pe_th_upper ) 
                print("DCR th",pe_th_lower )


            else:
                print(f"Not enough peaks in quadrant ({row}, {col}) to calculate SNR.")
                snr_per_quadrant[(row, col)] = None
                snr=np.nan
                snr_error=np.nan
                ampData.append(None)            
                ampData_error.append(None)

        except Exception as e:
            print(f"Error processing quadrant ({row}, {col}): {e}")

        except:
            print("Encountered error in sum of Gaussians fit, skipping...")

        try:
            Charge1peMean, Charge1peStd, Charge1peError,Charge1peStdError,chi2,ndof,countdens = pa.Get1pePeak(charge_trig.flatten(),
                                                                                bins = ChargeBins,
                                                                                PedestalThreshold = ChargePedestalLowerThreshold,
                                                                                histRange = [Chargehistrangelower,Chargehistrangeupper],
                                                                                verbose=False,
                                                                                Plot=ast.literal_eval(plot),
                                                                                upper_threshold=ChargePedestalUpperThreshold,
                                                                                axis=axs1PEc[row,col])
            pe_chargethreshold_upperLimit = Charge1peMean + 2*Charge1peStd
            pe_chargethresh_lowerLimit    = Charge1peMean - 2*Charge1peStd
        except:
            print("Encountered error in 1PE charge fitting, skipping...")
            Charge1peMean       = np.nan
            Charge1peStd        = np.nan
            Charge1peError      = np.nan
            Charge1peStdError   = np.nan
            chi2                = np.nan
            ndof                = np.nan
            axs1PEc[row, col].text(0.5, 0.5, 'Error in 1PE Charge Fitting\nSkipping...', ha='center', va='center', transform=axs1PEc[row, col].transAxes)
            axs1PEc[row, col].set_xticks([])
            axs1PEc[row, col].set_yticks([])
   
    elif plot == "False":
        try:
            if np.isnan(pe_th_lower) or np.isnan(pe_th_upper):
                try:
                    # Use full waveform amplitude spectrum to find 1PE position
                    pkp_full_channel = pkp[(pkp > 0) & (channel == x)].flatten()
                    
                    Amp1peMean_full, Amp1peStd_full, _, _, _, _, countdens_full = pa.Get1pePeak(
                        pkp_full_channel,
                        bins=500,
                        PedestalThreshold=0,
                        histRange=[0, 4000],
                        verbose=False,
                        Plot=False,
                        upper_threshold=750,
                        axis=None
                    )
                    
                    if not np.isnan(Amp1peMean_full) and not np.isnan(Amp1peStd_full):
                        pe_th_lower = Amp1peMean_full - 2 * Amp1peStd_full
                        pe_th_upper = Amp1peMean_full + 2 * Amp1peStd_full
                        position_largest = Amp1peMean_full
                        position_largest_std = Amp1peStd_full
                        countdens = countdens_full
                        print(f"Full-window fallback: 1PE at {Amp1peMean_full:.1f} ADC, "
                            f"thresholds {pe_th_lower:.1f} → {pe_th_upper:.1f}")
                except Exception as e:
                    print(f"Full-window fallback failed: {e}")
        except:
            print("Encountered error in 1PE fitting, skipping...")
            Amp1peMean      = np.nan
            Amp1peStd       = np.nan
            Amp1peError     = np.nan
            Amp1peStdError  = np.nan
            chi2            = np.nan
            ndof            = np.nan
        try:
            Amp2peMean, Amp2peStd, Amp2peError,Amp2peStdError,chi2_2,ndof_2,countdens = pa.Get1pePeak(pkp_trig.flatten(),
                                                                                bins = 500,
                                                                                PedestalThreshold = pe_threshold_upperLimit,
                                                                                histRange =[0,4000],
                                                                                verbose=False,
                                                                                Plot=False,
                                                                                upper_threshold=None,
                                                                                axis= None)
            TwoPE_threshold_upperLimit = Amp2peMean + 2*Amp1peStd
            TwoPE_thresh_lowerLimit    = Amp2peMean - 2*Amp1peStd
        except:
            print("Encountered error in 2PE fitting, skipping...")
            Amp2peMean      = np.nan
            Amp2peStd       = np.nan
            Amp2peError     = np.nan
            Amp2peStdError  = np.nan
            chi2_2          = np.nan
            ndof_2          = np.nan
      
        try:
            Charge1peMean, Charge1peStd, Charge1peError,Charge1peStdError,chi2,ndof,countdens = pa.Get1pePeak(charge_trig.flatten(),
                                                                                bins = 500,
                                                                                PedestalThreshold = 0 ,
                                                                                histRange = [0,0.0035],
                                                                                verbose=False,
                                                                                Plot=False,
                                                                                upper_threshold=None,
                                                                                axis=None)
            pe_chargethreshold_upperLimit = Charge1peMean + 2*Charge1peStd
            pe_chargethresh_lowerLimit    = Charge1peMean - 2*Charge1peStd
        except:
            print("Encountered error in 1PE charge fitting, skipping...")
            Charge1peMean       = np.nan
            Charge1peStd        = np.nan
            Charge1peError      = np.nan
            Charge1peStdError   = np.nan
            Charge1peStdError   = np.nan
            chi2                = np.nan
            ndof                = np.nan

    #####
    #DCR
    #####
    print("Dark Noise start      : {:.2e}".format(DN_start))

    # Skip DCR and downstream if thresholds are invalid
    # Skip DCR and downstream if thresholds are invalid
    try:
        _th_bad = (np.isnan(pe_th_lower) or np.isnan(pe_th_upper) or
                   pe_th_upper <= 0 or pe_th_lower <= 0)
    except NameError:
        _th_bad = True

    if _th_bad:
        print("Using fallback threshold (low statistics mode)")
        print("Using full-window fallback (trigger ROI too sparse).")

        try:
            # Step 1: extract all valid amplitudes for this channel
            pkp_flat_all = pkp.flatten()
            ch_flat_all  = channel.flatten()
            pkp_full_channel = pkp_flat_all[
                (ch_flat_all == x) &
                (pkp_flat_all > -9000) &
                (pkp_flat_all > 0)
            ]

            print(f"Full-channel valid entries: {len(pkp_full_channel)}")
            above_ped = pkp_full_channel[
                (pkp_full_channel > 35) & (pkp_full_channel < 500)
            ]
            print(f"Entries above pedestal threshold: {len(above_ped)}")

            if len(above_ped) < 5:
                raise ValueError(
                    f"Only {len(above_ped)} counts above pedestal — "
                    f"insufficient for threshold estimate."
                )

            # Step 2: coarse histogram peak finding — no curve_fit needed
            counts_fb, edges_fb = np.histogram(
                above_ped, bins=15, range=(35, 500)
            )
            centers_fb   = 0.5 * (edges_fb[:-1] + edges_fb[1:])
            peak_bin     = np.argmax(counts_fb)
            Amp1peMean_full = float(centers_fb[peak_bin])

            # Estimate sigma from FWHM
            half_max  = counts_fb[peak_bin] / 2.0
            above_half = centers_fb[counts_fb >= half_max]
            if len(above_half) >= 2:
                Amp1peStd_full = (above_half[-1] - above_half[0]) / 2.355
            else:
                Amp1peStd_full = 15.0

            binwidth_fb    = float(edges_fb[1] - edges_fb[0])
            countdens_full = float(counts_fb[peak_bin]) / binwidth_fb

            pe_th_lower          = Amp1peMean_full - 2.5 * Amp1peStd_full
            pe_th_upper          = Amp1peMean_full + 2.5 * Amp1peStd_full
            position_largest     = Amp1peMean_full
            position_largest_std = Amp1peStd_full
            countdens            = countdens_full

            print(f"Full-window fallback succeeded: 1PE at "
                  f"{Amp1peMean_full:.1f} ADC "
                  f"({Amp1peMean_full*0.244:.1f} mV), "
                  f"thresholds {pe_th_lower:.1f} → {pe_th_upper:.1f}")

        except Exception as e:
            print(f"Full-window fallback failed: {e}")
            return (
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                (np.nan, np.nan), np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan
            )

    try:
        pa.update_analysis_params(DarkNoiseRoiStart=DN_start)
        DN_rate_in_roi, DN_std, DN_rate_err= pa.GetDCR(pkt,pkp,pe_th_lower,
                                            pe_th_upper,Plot=ast.literal_eval(plot),nbins=200,pk_ch=channel,channel_number=x,axis=axsDCR[row,col])

        DCR_val = (DN_rate_in_roi * us_to_Hz_normalisation) / dcr_area
        DCR_err = (DN_rate_err * us_to_Hz_normalisation) / dcr_area

        print("DN Rate (counts/ROI)    :", DN_rate_in_roi, "+/-", DN_rate_err)
        print("DCR (Hz/mm2)            :", DCR_val, "+/-", DCR_err)
    except:
        print("Encountered error in DCR calculation, skipping...")
        DN_rate_in_roi = np.nan
        DN_std         = np.nan
        DN_rate_err    = np.nan
        axsDCR[row, col].text(0.5, 0.5, 'Error in DCR Calculation\nSkipping...', ha='center', va='center', transform=axsDCR[row, col].transAxes)
        axsDCR[row, col].set_xticks([])
        axsDCR[row, col].set_yticks([])

    

    ####
    #DCR from gaussian subtraction of 0pe
    ####
    try:
        pkpflat=pa.getpkproi(pkt,pkp,pk_ch=channel,channel_number=x)
        Amp0peMean, Amp0peStd, Amp0peError,Amp0peStdError,chi2_0,ndof_0,countamp = pa.Get1pePeak(pkpflat,
                                                                            bins = 500,
                                                                            PedestalThreshold = 0,
                                                                            histRange =[0,pe_th_upper],
                                                                            verbose=True,
                                                                            Plot=False,
                                                                            upper_threshold=pe_th_upper,
                                                                            axis=axsGSDCR[row,col])
        GSDN_rate_in_roi, pe0GaussianDCR, GSDN_rate_err, pe0GaussianDCR_err=pa.GetGSDCR(pkp,pkpflat,pe_th_lower,pe_th_upper,Amp0peMean,Amp0peStd,countamp,Plot=True,nbins=500,pk_ch=channel,channel_number=x,axis=axsGSDCR[row,col])
        GSDCR_val = (GSDN_rate_in_roi * us_to_Hz_normalisation) / dcr_area
        GSDCR_err = (GSDN_rate_err * us_to_Hz_normalisation) / dcr_area

        print("GSD count after subtraction :", pe0GaussianDCR, "+/-", pe0GaussianDCR_err)
        print("GSD rate per waveform       :", GSDN_rate_in_roi, "+/-", GSDN_rate_err)
        print("GSDCR (Hz/mm2)              :", GSDCR_val, "+/-", GSDCR_err)
        
        n_waveforms = pkp.shape[0]

        print("--------------------------------------------------")
        print("GSDCR breakdown:")
        print("Counts in 1PE region (after subtraction):", pe0GaussianDCR)
        print("Number of waveforms:", n_waveforms)
        print("Rate per waveform (ROI):", GSDN_rate_in_roi)
        print("ROI time (s):", DarkNoiseRoiTime)
        print("Normalisation factor (1/ROI time):", us_to_Hz_normalisation)
        print("Tile area (mm^2):", dcr_area)
        print("Final GSDCR (Hz/mm^2):", (GSDN_rate_in_roi * us_to_Hz_normalisation) / dcr_area)
        print("--------------------------------------------------")
        print("GSDCR counts in 1PE region (after pedestal subtraction):", pe0GaussianDCR)
    except Exception as e:
        print ("0pe Gaussian fit failed", e)
        GSDN_rate_in_roi = np.nan
        axsGSDCR[row, col].text(0.5, 0.5, 'Error in GS DCR Calculation\nSkipping...', ha='center', va='center', transform=axsGSDCR[row, col].transAxes)
        axsGSDCR[row, col].set_xticks([])
        axsGSDCR[row, col].set_yticks([])

    #######
    #CDA
    ######
    try:
        CDA_Mean, CDA_std, CDA_mean_Err = pa.GetCDA(pkt,pkp,pe_th_upper,pe_th_lower,200, Plot=ast.literal_eval(plot),pk_ch=channel,channel_number=x,axis=axsCDA[row,col])
        DarknoiseSubtractedCDA          = CDA_Mean - DN_rate_in_roi
    except:
        print("Encountered error in CDA calculation, skipping...")
        CDA_Mean     = np.nan
        CDA_std      = np.nan
        CDA_mean_Err = np.nan
        DarknoiseSubtractedCDA = np.nan
        axsCDA[row, col].text(0.5, 0.5, 'Error in CDA Calculation\nSkipping...', ha='center', va='center', transform=axsCDA[row, col].transAxes)
        axsCDA[row, col].set_xticks([])
        axsCDA[row, col].set_yticks([])
    print("Mean CDA in ROI       : {:.5f}".format(CDA_Mean))
    print("DN subtraced CDA      : {:.5f}".format(DarknoiseSubtractedCDA))

    #######
    #AP & DiCT
    #######
    try:
        N_APA, pkp_normStdError_0_5pe = pa.GetMeanAPA(pkp,pkt,position_largest,position_largest_std,pk_ch=channel,channel_number=x,axis=axsNAPA[row,col])
    except:
        print("Encountered error in APA calculation, skipping...")
        N_APA = np.nan
        pkp_normStdError_0_5pe = np.nan
        axsNAPA[row, col].text(0.5, 0.5, 'Error in APA Calculation\nSkipping...', ha='center', va='center', transform=axsNAPA[row, col].transAxes)
        axsNAPA[row, col].set_xticks([])
        axsNAPA[row, col].set_yticks([])
    try:
        DiCTProb = pa.GetDiCT(pkp,pkt,position_largest,plot=ast.literal_eval(plot),nbins=250,pk_ch=channel,channel_number=x,axis1=axsnormspe[row,col],axis2 = axsapaspec[row,col])
    except:
        print("Encountered error in DiCT calculation, skipping...")
        DiCTProb = (np.nan, np.nan)
        axsnormspe[row, col].text(0.5, 0.5, 'Error in DiCT Calculation\nSkipping...', ha='center', va='center', transform=axsnormspe[row, col].transAxes)
        axsnormspe[row, col].set_xticks([])
        axsnormspe[row, col].set_yticks([])
    print("Probability of DiCT   : {:.5f} +/- {:.5f}".format(DiCTProb[0],float(DiCTProb[1])))
    print("NAPA                  : {:.5f} +/- {:.5f}".format(N_APA, pkp_normStdError_0_5pe))

    #print("Mean number of additional Prompt Avalanches :", N_APA)

    try:
        snr
    except NameError:
        snr = np.nan
        snr_error = np.nan
    
    return snr,Amp1peMean,DN_rate_in_roi,GSDN_rate_in_roi,CDA_Mean,rms,DiCTProb,N_APA,snr_error,Amp1peError, pkp_normStdError_0_5pe,DN_rate_err,CDA_std

##################
#Main Body
##################
if __name__ == '__main__':
    ################
    #Initialisation
    ################
    parser = argparse.ArgumentParser()
    parser.add_argument("-I","--InputReco",   help     ="Directory/to/data/for/spe")
    parser.add_argument("-P","--plot",        help     ="Plot = True or False")
    parser.add_argument("-B","--bias",        help     ="Bias voltage (V)")
    parser.add_argument("-T","--temp",        help     ="Warm(w) or Cold Test(c)")
    parser.add_argument("-S", "--ini_file",   help     ="Path to .ini file")
    parser.add_argument("-R", "--reading",    help     ="Which channels to read in")
    parser.add_argument("-N", "--Name",       help     ="Output csv names")
    parser.add_argument("-V", "--vPDUs_connected",type =int, required=False, help ="Number of connected vPDUs")
    parser.add_argument("--run_number",       type     =int, required=False, help ="Run number")
    parser.add_argument('--vPDU_index',       type     =int, required=False, help="Index of the current vPDU")
    parser.add_argument("--IV_path",          type     =str, required=False, help="Path to IV data directory")
    parser.add_argument("-Q", "--qr_code",    required =False,   help="QR code for the vPDU")
    parser.add_argument("-F", "--folder",     required =False,   help="Folder to save the output")
    parser.add_argument("-C", '--IV',         type     =str,  choices=['True', 'False'], default='False', help="Enable IV analysis (True/False)")
    parser.add_argument('--mode', choices=['Quadrant', 'Tile'], required=True, help="Choose between 'quadrants' or 'tiles'")

    args       = parser.parse_args()
    mode       = args.mode  # Quadrant or 'Tile', you can change this based on input or command line args
    run_number = args.run_number
    config     = configparser.ConfigParser()

    if args.ini_file:  # Check if custom .ini file is provided
        config.read(args.ini_file)
        print("Using config file: ", args.ini_file)
    else:
        config.read('AnalysisConfig.ini')  # Default .ini file
        print("Using default config file")

    print('Mode:', mode)
    
    #Assigning input arguments to variable
    plot        = args.plot
    Bias        = args.bias
    temp        = args.temp
    #chan = args.channel
    spefilepath = args.InputReco

    if args.IV  == 'True' and args.IV_path is None:
    	parser.error("--IV_path is required when --IV is set to True")

    user_input  = args.Name
    #reading parameters from config file
    config = configparser.ConfigParser()
    config.read('AnalysisConfig.ini')
    #Analysis Details overwritten by config file
    PromptRoiLeft           = float(config['parameters']['PromptRoiLeft'])        #seconds
    PromptRoiRight          = float(config['parameters']['PromptRoiRight'])       #seconds
    CDAroiTime              = float(config['parameters']['CDAroiTime'])           #seconds
    DarkNoiseRoiTime        = float(config['parameters']['DarkNoiseRoiTime'])     #seconds
    DN_start_from_trigger   = float(config['parameters']['DN_start_from_trigger'])#seconds
    Tile_Area               = float(config['parameters']['Tile_Area']) #mm^2
    SamplingRate            = float(config['daq']       ['samplingRate']) #S/s
    ADCmVConv               = float(config['daq']       ['Dig_V_pp'])/(2**(int(config['daq']['Dig_Bit_Value'])))
    DN_start                = 0
    us_to_Hz_normalisation  = 1/DarkNoiseRoiTime #factor used to scale darknoise to Hz
    print("ADCmVConv : {:.5f}".format(ADCmVConv))
    #creating variabls for analysis quantaties
    pkt             = []
    pkp             = []
    charge          = []
    chrms           = []
    channel         = []
    BiasData        = []
    SNRData         = []
    ampData         = []
    DNData          = []
    GSDNData        = []
    CDAData         = []
    rmsData         = []
    DiCTData        = []
    APAData         = []
    SNR_Error       = []
    ampData_error   = []
    DN_Error        = []
    CDA_Error       = []
    rms_Error       = []
    DiCT_Error      = []
    APA_Error       = []

    if args.InputReco:
        pa = analysis.CharacterisationTools(PromptRoiLeft=PromptRoiLeft,PromptRoiRight=PromptRoiRight,PromptRoiTime=None,CDAroiTime = CDAroiTime,DarkNoiseRoiTime=DarkNoiseRoiTime, DarkNoiseRoiStart=DN_start_from_trigger,SamplingRate=SamplingRate)
        pa.LoadFile(spefilepath)
        pkt         = pa.GetBranch("dstree","pk_t")   #Time of hits
        pkp         = pa.GetBranch("dstree","pk_p")   #Amplitude of each hit
        charge      = pa.GetBranch("dstree","pk_k")   #Integral of each hit
        chrms       = pa.GetBranch("dstree","ch_rms") #rms
        channel     = pa.GetBranch("dstree","pk_ch")  #Channel
        channelrms  = pa.GetBranch("dstree","ch_id")  #Channel
    else:
        pass
    ###########################
    #Channel Loop over SPE data
    ###########################
    if args.InputReco:
        figAMPROI, axsAMPROI    = plt.subplots(2, 2, figsize=(10, 10))
        figGS, axsGS            = plt.subplots(2, 2, figsize=(10, 10))
        figAMP, axsAMP          = plt.subplots(2, 2, figsize=(10, 10))
        figC, axsC              = plt.subplots(2, 2, figsize=(10, 10))
        figRMS, axsRMS          = plt.subplots(2, 2, figsize=(10, 10))
        figT, axsT              = plt.subplots(2, 2, figsize=(10, 10))
        fig1PE, axs1PE          = plt.subplots(2, 2, figsize=(10, 10))
        fig2PE, axs2PE          = plt.subplots(2, 2, figsize=(10, 10))
        fig1PEc, axs1PEc        = plt.subplots(2, 2, figsize=(10, 10))
        figGSDCR, axsGSDCR      = plt.subplots(2, 2, figsize=(10, 10))
        figDCR, axsDCR          = plt.subplots(2, 2, figsize=(10, 10))
        figCDA, axsCDA          = plt.subplots(2, 2, figsize=(10, 10))
        fignomspe, axsnormspe   = plt.subplots(2, 2, figsize=(10, 10))
        figapaspec, axsapaspec  = plt.subplots(2, 2, figsize=(10, 10))
        figNAPA, axsNAPA        = plt.subplots(2, 2, figsize=(10, 10))

        # Define a function that creates folders based on the desired number of folder ranges.
        def create_folders_by_count(vPDUs_connected, step=4):
            folders = {}
            for i in range(vPDUs_connected):
                start                   = i * step
                end                     = (i + 1) * step
                folder_name             = f"{start}_{end}"
                folder_path             = os.path.join(folder_name)
                folders[(start, end)]   = folder_path
            return folders

        # Create folder ranges based on the provided vPDUs_connected argument
        folders        = create_folders_by_count(args.vPDUs_connected, step=4)
        channel_ranges = [(low, high, path) for (low, high), path in folders.items()]

        for lower, upper, folder in channel_ranges:
            #print(f"Processing channels {lower} to {upper-1}")
            folder = args.folder
        BiasData, SNRData, ampData, DNData,GSDNdata, CDAData, rmsData, DiCTData,APAData         = [],[], [], [], [], [], [], [],[]
        SNR_Error, ampData_error, DN_Error, CDA_Error, rms_Error, DiCT_Error,APA_Error = [], [], [], [], [], [], []

        # Flag to choose between processing specific channels or all channels in the range
        use_desired_channels = False  # Set to False to use the original (range-based) approach
        if use_desired_channels:
            print(" ----------------------------------")
            print(f" |   Desired channels is enabled.  |")
            print(" ----------------------------------")
        # List of desired channels if using the specific list
        desired_channels = [4,5]

        if args.qr_code:
            args.qr_code, lower, upper, datestr = folder.split('_')
        else:
            args.qr_code = "qr"
            lower        = int(1)
            upper        = int(4)
            datestr      = 'not_required'

        lower, upper = int(lower), int(upper) + 1

        # Determine which channels to process: specific ones or all in range
        if use_desired_channels:
            channels_to_process = desired_channels
        else:
            channels_to_process = range(lower, upper)

        def process_channel(x):
            # Check if channel is in data (or valid data) based on the unique channels available
            if x not in np.unique(channel):
                print(f"Channel {x} not found in data, skipping.")
                return  # Skip processing if channel not in data
            print("\n ----------------------------------")
            print(f" |      Processing channel {x}      |")
            print(" ----------------------------------")
            # Check if data for the channel is valid (not empty or with low stats)
            if len(pkt[(pkt > -9999) & (channel == x)].flatten()) < 1000:
                print(f"Channel {x} data is empty or has low stats, skipping...")

                for data_list in [BiasData, SNRData, ampData, DNData,GSDNData, CDAData, rmsData, DiCTData, APAData,
                                SNR_Error, ampData_error, DN_Error, CDA_Error, rms_Error, DiCT_Error, APA_Error]:
                    data_list.append(np.nan)
                return
        if mode == "Tile":
            dcr_area = Tile_Area
        elif mode == "Quadrant":
            dcr_area = Tile_Area * 4
        
        for x in channels_to_process:
            if use_desired_channels:
                if x not in desired_channels:
                    continue
            else:
                if not (lower <= x < upper):
                    continue

            if x not in np.unique(channel):
                print(f"Channel {x} not found in data, skipping.")
                continue

            if len(pkt[(pkt > -9999) & (channel == x)].flatten()) < 1000:
                print(f"Channel {x} data is empty or has low stats, skipping...")
                continue

            row, col = calculate_position_2x2(x)
            try:
                (snr,Amp1peMean, DN_rate_in_roi,GSDN_rate_in_roi, CDA_Mean, rms, DiCTProb,N_APA,
                snr_error,Amp1peError,pkp_normStdError_0_5pe,DN_rate_err,CDA_std) = process_channel_data(x, pkp, pkt, charge, chrms, channel, channelrms, row, col)

                BiasData.append(Bias)
                SNRData.append(snr)
                DNData.append((DN_rate_in_roi*us_to_Hz_normalisation)/(dcr_area))
                GSDNData.append((GSDN_rate_in_roi*us_to_Hz_normalisation)/(dcr_area))
                CDAData.append(CDA_Mean)
                rmsData.append(rms*ADCmVConv/1000)
                DiCTData.append(DiCTProb[0])
                APAData.append(N_APA)
                SNR_Error.append(snr_error)
                DN_Error.append((DN_rate_err*us_to_Hz_normalisation)/(dcr_area))  # Quadrant area = Tile area x 4
                CDA_Error.append(CDA_std)
                rms_Error.append(ADCmVConv * np.std(chrms[(channelrms == x) & (chrms > -9999)].flatten()) / len(chrms[(channelrms == x) & (chrms > -9999)].flatten()) / 1000)
                DiCT_Error.append(DiCTProb[1])
                APA_Error.append(pkp_normStdError_0_5pe)
                print("DCR (Hz/mm2): ",(DN_rate_in_roi*us_to_Hz_normalisation)/(dcr_area))
                print("GSDCR (Hz/mm2): ",(GSDN_rate_in_roi*us_to_Hz_normalisation)/(dcr_area))
            except Exception as e:            
                print(f"Error in processing channel {x}: {e}")            
                for lst in [BiasData, SNRData, DNData,GSDNData, CDAData, rmsData, DiCTData, APAData,            
                    SNR_Error, DN_Error, CDA_Error, rms_Error, DiCT_Error, APA_Error]:            
                    lst.append(np.nan)
        #########The figures are produced here################
        figures = {
            "AmplitudeROI"    : figAMPROI,
            "AmplitudeROI_GS" : figGS,
            "RMS"             : figRMS,
            "Time"            : figT,
            "1PEcharge"       : fig1PEc,
            "DCR"             : figDCR,
            "DCR-pedestal"    : figGSDCR,
            "CDA"             : figCDA,
            "NormalisedSPE"   : fignomspe,
            "apa_spectrum"    : figapaspec,
        }
        #############This prints the SNR value on top of the Amplitude ROI plot##################
        for i, ax in enumerate(axsGS.flat):  # Flatten in case of multi-panel plots
            if i < len(SNRData) and i < len(SNR_Error):  # Ensure we don't go out of index range
                snr_value = SNRData[i]
                snr_error = SNR_Error[i]
                if not np.isnan(snr_value) and not np.isnan(snr_error):  # Avoid NaN values
                    ax.text(0.77, 0.77, f"SNR: {snr_value:.2f}±{snr_error:.2f}",fontsize=12, color='green', fontweight='bold', ha='center', va='top', transform=ax.transAxes)
        #############This produces the plot according to the names specified above##############
        for name, fig in figures.items():
            plt.figure(fig.number)
            plt.suptitle(f"vPDU-{args.qr_code}", fontsize=13, fontweight='bold')
            plt.savefig(f"{folder}/{args.run_number}_{name}.png",dpi=500)
            plt.clf()

        ######IV analysis happens here#######
        if args.IV == 'True':
            print(" ----------------------------------")
            print(f" |      IV Analysis is enabled.   |")
            print(" ----------------------------------")
        else:
            print(" ----------------------------------")
            print(f" |      IV Analysis is disabled.  |")
            print(" ----------------------------------")

        vPDU_index = args.vPDU_index

        # Set No_text_files based on the mode
        if mode == 'Quadrant':
            No_text_files = len(desired_channels) if use_desired_channels else (upper - lower)
            print(f"Processing in 'Quadrants' mode.")
        elif mode == 'Tile':
            No_text_files = 16
            print(f"Processing in 'Tiles' mode.")
        elif mode == '-':
            No_text_files = 0
            print(f"Processing")
        else:
            print("Invalid mode selected.")

        start_quadrant = vPDU_index * No_text_files

        # Determine IV files to read based on use_desired_channels
        if use_desired_channels:
            file = [f"{args.IV_path}/IV_q{channel}.txt" for channel in desired_channels] #This is only for quadrant, and cannot be accessed by Tiles
            print(f"Processing Quadrants  : {desired_channels}")
            #print(f"Files                 : {file}")
        else:
            start_quadrant = vPDU_index * No_text_files
            file = [f"{args.IV_path}/IV_q{i}.txt" for i in range(start_quadrant, start_quadrant + No_text_files)]
            print(f"Processing Quadrants  : {list(range(start_quadrant, start_quadrant + No_text_files))}")
            #print(f"Files                 : {file}")

        #print(f"IV files being read: {file}")
        Vbd_Cold     = ["-"] * No_text_files
        Vbd_cold_err = ["-"] * No_text_files
        IVDataCsv    = [None] * No_text_files
        IVword       = ["-"] * No_text_files

        # Process each of the 4 quadrants for this vPDU
        if args.IV == 'True':
            IV = IVAnalysis.IVAnalysis()
            for i in range(No_text_files):
                try:
                  Vbd_Cold[i], Vbd_cold_err[i] = IV.calc_Vbd(file[i], Vstart=42, Vstop=65, plot_opt=None)
                except Exception as e:
                  print(f"Error processing file {file[i]}: {e}")
                  Vbd_Cold[i], Vbd_cold_err[i] = np.nan, np.nan

                if use_desired_channels:
                    print(f"Q{desired_channels[i]} breakdown voltage  : {Vbd_Cold[i]:.5f}")
                else:
                    print(f"Q{start_quadrant + i} breakdown voltage  : {Vbd_Cold[i]:.5f}")
                IVword[i] = IV.Combine_IV(file[i])
                if len(IVword) < No_text_files:
                    IVword.extend(["-"] * (No_text_files - len(IVword)))
                IVDataCsv[i] = np.loadtxt(file[i], delimiter=",")
            if str(args.plot).lower() == 'true':
                fig, ax         = plt.subplots(figsize=(8, 6))
                #git_tag         = subprocess.check_output(['git', 'describe', '--tags']).strip().decode()
                legend_handles  = []
                legend_labels   = []

                for i in range(No_text_files):
                    IV.plot_tileIV(file[i], label=f"{mode} {start_quadrant+1 + i}")
                    line = ax.get_lines()[-1]
                    line_color = line.get_color()
                    ax.axvline(Vbd_Cold[i], linestyle='dashed', color=line_color)
                    legend_handles.append(line)
                    legend_labels.append(f"{mode} {start_quadrant+1 + i}: {Vbd_Cold[i]:.3f} V")
                ax.set_title(f"IV Curves {args.qr_code}")
                ax.set_xlabel("Voltage (V)",fontsize=12)
                ax.set_ylabel(r"Current ($\mu$A)",fontsize=12)
                ax.legend(legend_handles, legend_labels, loc="best", fontsize=8)
                ax.text(0.50, 0.55, f"Git Tag: {GitVersion}", horizontalalignment='right', verticalalignment='bottom',
                        transform=ax.transAxes, fontsize=10, color='black')
                plt.tight_layout()
                plt.savefig(f"{args.folder}/{args.qr_code}_combined_IV_curves.png",dpi=500)
                plt.close(fig)
            else:
                print("IV analysis disabled. Skipping IV data processing.")
        Git_tag = [GitVersion, GitVersion, GitVersion, GitVersion]

        ##########creation of the csv file happens here############
        passportFile = f'{args.Name}'
        with open(f"{passportFile}", 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                    'Quadrant','Bias_V', 'SNR','1peAmp_V','DN_Hz_per_mm2','GSDN_Hz_per_mm2','CDA','rms_V','DiCT_prob','N_APA','SNRError','1peAmpError_V','DNError','CDAError','rmsStd_V','DiCTError','N_APAError','breakdown','breakdownErr','IV_data','Git_tag'])
            
            vars_to_check = [
                'BiasData', 'SNRData', 'ampData', 'DNData','GSDNData', 'CDAData', 'rmsData',
                'DiCTData', 'APAData', 'SNR_Error', 'ampData_error', 'DN_Error',
                'CDA_Error', 'rms_Error', 'DiCT_Error', 'APA_Error',
                'Vbd_Cold', 'Vbd_cold_err', 'IVword', 'Git_tag'
            ]

            for var_name in vars_to_check:
                var = globals().get(var_name)
                if not isinstance(var, list):
                    print(f"Warning: {var_name} is not a list — skipping length check.")
                    continue
                while len(var) < 4:
                    var.append(np.nan)
                globals()[var_name] = var  # update the global variable

            csv_writer.writerows(zip([1,2,3,4],BiasData,SNRData,ampData,DNData,GSDNData,CDAData,rmsData,DiCTData,APAData,SNR_Error,ampData_error,DN_Error,CDA_Error,rms_Error,DiCT_Error,APA_Error,Vbd_Cold,Vbd_cold_err,IVword,Git_tag))

        print("Output csv file name  :",user_input)
        import AnalysisTools.IVAnalysis as IVAnalysis

