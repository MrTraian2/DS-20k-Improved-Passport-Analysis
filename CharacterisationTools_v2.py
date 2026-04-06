'''
Characterisation Tools used for the characterisation of vPDUs

These tools have been developed with the analysis of ROOT files in mind. However, most functions should
be able to handle multi-channel data get1pepeak being the exception

The functions contained  in the class allow for the calculation of:
1pe mean
Dark Noise Rate
Gaussian Subtracted Dark Noise Rate
Correlated Delayed Avalanche Rate
Mean Additional Prompt Avalanches
Direct Cross Talk Probability


Seraphim Koulosousas - seraphim.koulosousas@live.rhul.ac.uk
'''

import subprocess
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
import AnalysisTools.Algos as algos
from scipy import stats
from scipy.optimize import curve_fit
import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import configparser

try:
    version = subprocess.check_output(['git', 'describe', '--tags', '--abbrev=0'])
    GitVersion = str(version, 'utf-8').strip()
except:
    print("Error reading Git tag")
    GitVersion = "Unknown"

def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Define the sum of Gaussians function
def sum_of_gaussians(x, *params):
    gaussians = [
        gaussian(x, params[i], params[i + 1], params[i + 2])
        for i in range(0, len(params), 3)
    ]
    return np.sum(gaussians, axis=0)

def fit_gaussian_peaks(float_array, sigma=4, num_peaks=3, axis=None, fallback_peaks=2):
    """
    Fits a sum of Gaussians to histogram data from an array of floats and plots it on a provided axis.

    Args:
        float_array (array-like): Input data to generate the histogram.
        sigma (float): Standard deviation for Gaussian smoothing.
        num_peaks (int): Number of peaks to fit.
        axis (matplotlib.axes.Axes): The subplot axis to plot the results.

    Returns:
        list of tuples: Fitted parameters for each Gaussian in the format
                        [(amplitude1, mean1, stddev1, mean_error1), ..., (amplitudeN, meanN, stddevN, mean_errorN)].
    """
    config = configparser.ConfigParser()
    config.read('AnalysisConfig.ini')
    ADCmVConv = float(config['daq']['Dig_V_pp'])/(2**(int(config['daq']['Dig_Bit_Value'])))

    if axis is None:
        raise ValueError("An axis must be provided for the plot.")

    # Generate histogram data
    counts, bin_edges = np.histogram(float_array, bins=500, range=(0, 2500))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #print(f"Histogram Counts: Max = {np.max(counts)}, Min = {np.min(counts)}")

    # Apply Gaussian smoothing
    smoothed_counts = gaussian_filter1d(counts, sigma=sigma)

    # Identify peaks
    peak_indices, _ = find_peaks(smoothed_counts, height=1,distance=30)
    peaks = sorted(zip(bin_centers[peak_indices], smoothed_counts[peak_indices]), key=lambda x: -x[1])
    first_peaks = peaks[:num_peaks] if len(peaks) >= num_peaks else peaks
    detected_peaks = len(peaks)

    if detected_peaks < num_peaks:
        if detected_peaks >= fallback_peaks:
            num_peaks = detected_peaks
        else:
            raise ValueError(f"Insufficient peaks found: {detected_peaks}. Minimum required: {fallback_peaks}.")

    # Use the first `num_peaks` identified peaks
    first_peaks = peaks[:num_peaks]

    max_stddev = 100
    min_stddev = 10

    # Initial guesses for fitting
    initial_guess = []
    for peak in first_peaks:
        amplitude = peak[1]
        mean = peak[0]
        stddev = (max_stddev + min_stddev) / 2
        initial_guess.extend([amplitude, mean, stddev])

    lower_bounds = []
    upper_bounds = []
    for peak in first_peaks:
        lower_bounds.extend([0, peak[0] - 50, min_stddev])
        upper_bounds.extend([np.inf, peak[0] + 50, max_stddev])

    # Fit the sum of Gaussians to the data
    popt, pcov = curve_fit(
        sum_of_gaussians,
        bin_centers,
        counts,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        maxfev=10000
        )

    # Compute errors from the diagonal of the covariance matrix
    errors = np.sqrt(np.diag(pcov))

    # Generate fitted curve
    fitted_curve = sum_of_gaussians(bin_centers, *popt)

    # Extract mean positions and corresponding indices
    peak_positions = [(popt[i * 3 + 1], i) for i in range(num_peaks)]

    # Sort peaks by x-axis position (ascending order)
    sorted_peaks_2 = sorted(peak_positions, key=lambda x: x[0])

    # Get the first and last peaks by position
    first_peak_index = sorted_peaks_2[0][1]
    last_peak_index = sorted_peaks_2[-1][1]

    # Extract their positions from popt
    first_peak_position = popt[first_peak_index * 3 + 1]
    last_peak_position = popt[last_peak_index * 3 + 1]

    print("first peak position   : {:.5f}".format(first_peak_position))
    print("last peak position    : {:.5f}".format(last_peak_position))


    # Find the corresponding indices in bin_centers
    first_index = np.searchsorted(bin_centers, first_peak_position)
    last_index = np.searchsorted(bin_centers, last_peak_position)

    # Compute chi-squared in the restricted range
    chi_squared = np.sum(((counts[first_index:last_index] - fitted_curve[first_index:last_index]) ** 2) /
                        (fitted_curve[first_index:last_index] + 1))
    reduced_chi_squared = chi_squared / (len(counts[first_index:last_index]) - len(popt))
    axis.plot(bin_centers * ADCmVConv, counts, label='Original Data', color='gray', alpha=0.7)
    axis.plot(bin_centers * ADCmVConv, smoothed_counts, label=f'Smoothed Data (σ={sigma})', color='red')
    axis.plot(bin_centers * ADCmVConv, fitted_curve, label='Sum of Gaussians Fit', color='blue', linestyle='--')


    axis.scatter(
        [p[0] * ADCmVConv for p in first_peaks],
        [p[1] for p in first_peaks],
        color='black', label='Identified Peaks', zorder=5
    )


    offset = 3  # Adjust this value as needed to shift the text to the right

    # for i, (x, y) in enumerate(first_peaks, 1):
    #     fitted_mean = popt[3 * (i - 1) + 1]  # Extract the mean value from the fit parameters
    #     axis.text(x + offset, y+offset, f"Peak {i}: {fitted_mean*ADCmVConv:.1f}", fontsize=7, color="black", ha="left", va="bottom")

    for i, (x, y) in enumerate(first_peaks, 1):
        fitted_mean = popt[3 * (i - 1) + 1]
        axis.text(
            x * ADCmVConv + offset, y + offset,
            f"Peak Amplitude: {fitted_mean * ADCmVConv:.2f}",
            fontsize=8, color="black", ha="left", va="bottom"
        )
    axis.text(
        0.65, 0.95,
        rf"$\chi^2$: {chi_squared:.1f}" + '\n' + rf"Reduced $\chi^2$: {reduced_chi_squared:.1f}",
        transform=axis.transAxes,
        fontsize=10,
        verticalalignment='top'
    )

    axis.set_xlabel("Amplitude (mV)")
    axis.set_ylabel("Counts")
    axis.grid(axis='y', linestyle='--', alpha=0.3)

    return [(popt[3 * i], popt[3 * i + 1], popt[3 * i + 2]) for i in range(num_peaks)], \
        [(errors[3 * i], errors[3 * i + 1], errors[3 * i + 2]) for i in range(num_peaks)]

class CharacterisationTools:
    def __init__(self,PromptRoiLeft=None,PromptRoiRight=None,PromptRoiTime=None,CDAroiTime = None,DarkNoiseRoiTime=None, DarkNoiseRoiStart=None,SamplingRate=None):
        print("Super cool and super duper efficient pulse analysis class Initialised! May your finger plots be beautiful!")
        self.PromptRoiLeft     = PromptRoiLeft
        self.PromptRoiRight    = PromptRoiRight
        self.PromptRoiTime     = PromptRoiTime       #the time in seconds around the trigger that defines the prompt region
        self.CDAroiTime        = CDAroiTime          #The time in seconds that will define the CDA region
        self.DarkNoiseRoiTime  = DarkNoiseRoiTime    #The time in seconds that will define the Region of Dark Noise
        self.DarkNoiseRoiStart = DarkNoiseRoiStart   #The time in seconds into the waveform that the DCR roi should begin
        self.SamplingRate      = SamplingRate


    def update_analysis_params(self,PromptRoiLeft=None,PromptRoiRight=None,PromptRoiTime=None,CDAroiTime = None,DarkNoiseRoiTime=None, DarkNoiseRoiStart=None):
        '''
        Function to update analysis parameters, so that you dont have to reinitialise the class
        '''
        if PromptRoiLeft     != None:
            self.PromptRoiLeft     = PromptRoiLeft
        if PromptRoiRight    != None:
            self.PromptRoiRight    = PromptRoiRight
        if PromptRoiTime     != None:
            self.PromptRoiTime     = PromptRoiTime       #the time in seconds around the trigger that defines the prompt region
        if CDAroiTime        != None:
            self.CDAroiTime        = CDAroiTime          #The time in seconds that will define the CDA region
        if DarkNoiseRoiTime  != None:
            self.DarkNoiseRoiTime  = DarkNoiseRoiTime    #The time in seconds that will define the Region of Dark Noise
        if DarkNoiseRoiStart !=None:
            self.DarkNoiseRoiStart = DarkNoiseRoiStart   #The time in seconds into the waveform that the DCR roi should begin


    def LoadFile(self,RootFile):
        self.RootFile = uproot.open(RootFile)

    def GetSampling(self,LogFile):
        '''
        Get the sampling rate from the log file that accompanies the root file
        '''
        log = []
        with open(LogFile) as fileopen:
            log = fileopen.readlines()
        for line in log:
            if "sampling" in line: sample_freq = float(line.split(" ")[2])
        self.SamplingRate     = sample_freq   #Sampling rate that the dat was taken
        return sample_freq

    def SetSampling(self,samplingrate):
        self.SamplingRate     = samplingrate

    def GetTriggerPosition(self,pkt_flattened,nBins, method = "max",verbose=False):
        '''
        find the position of the trigger via two methods

        max method - simply plots a histogram and looks for the bin with the highest count

        Gauss method - fits a gaussian to the max position **note check that it actually looks gaussian**
                     - a 405nm laser is used so it should look pretty gaussian in theory
        '''

        if method == "max":
            pk_t_hist   = np.histogram(pkt_flattened,bins=nBins)
            max_val     = np.where(pk_t_hist[0] == max(pk_t_hist[0]))        #This gives the index position of the maximum value in pk_t histogram i.e teh time
            max_val_bin = pk_t_hist[1][max_val]
            return max_val_bin[0]  , np.nan , np.nan

        if method == "Gauss":
            '''
            placeholder
            '''
            print("Gauss")
            plt.figure(figsize=[17,9])
            pk_t_hist = plt.hist(pkt_flattened,bins=nBins,histtype="step")
            x1, x2, max_y, max_x = algos.box(pk_t_hist,sig_level=3, mean_shift=True,verbose = verbose)
            xs = pk_t_hist[1][:-1][(pk_t_hist[1][:-1]<x2)& (pk_t_hist[1][:-1]>x1)]
            ys =  pk_t_hist[0][(pk_t_hist[1][:-1]<x2)& (pk_t_hist[1][:-1]>x1)]
            popt, pcov = curve_fit(algos.gauss,xs,ys,p0=[pk_t_hist[1][max_x],pk_t_hist[1][max_x],max_y])
            mu_err = np.sqrt(pcov[0][0])
            # plt.hist(pkt_flattened,bins=nBins)
            plt.plot(xs,algos.gauss(xs,*popt),label="Gaussian fit")
            plt.ylabel("Counts")
            plt.xlabel("Hist arb. units")
            plt.legend()
            return popt[0], np.abs(popt[1]), mu_err

    def GetBranch(self,TreeName, BranchName):
        '''
        Get Branch from root file. Since the root file is in general jagged we use Awkward fill_none to make the
        array regular. Reason for this is it makes life a lot easer making cuts using numpy logic statments which
        are way quicker than python loops

        The array is being normalised to a length of 20, our waveforms are normally 20us long so it is unlikely
        to have 20 hits in a waveform if the waveform is over a larger time range this will need to be changed
        '''
        array_max_length = 20
        self.fill_value = -9999 #make sure this is negative if you want to make it positive check for where this is used

        tree = self.RootFile[TreeName]
        branches = tree.arrays() #namedecode='utf-8'
        return np.asarray(ak.fill_none(ak.pad_none(branches[BranchName], array_max_length, clip=True), self.fill_value))

    def Get1pePeak(self,Branch_flattened,bins = 500,PedestalThreshold = 50, histRange =[0,100000],verbose=False,Plot=False,upper_threshold=None,axis=None):
        if upper_threshold==None:
            Branch_flattened=Branch_flattened[Branch_flattened>PedestalThreshold]
        else:
            Branch_flattened=Branch_flattened[(Branch_flattened>PedestalThreshold)& (Branch_flattened<upper_threshold)]

        # plt.figure(figsize=[17,7])
        (counts,bins) = np.histogram(Branch_flattened,bins,range=histRange)

        # Hist_Hist= plt.hist(Branch_flattened,bins,range=histRange)
        x1, x2, max_y, max_x = algos.box((counts,bins),sig_level=3, mean_shift=True,verbose = False)
        if (verbose ==True):
            print(x1,x2,max_y, bins[max_x])
        xs = bins[:-1][(bins[:-1]<x2)& (bins[:-1]>x1)] + ( bins[:-1][1]-bins[:-1][0]) /2 # shift to centre of bin
        ys =  counts[(bins[:-1]<x2)& (bins[:-1]>x1)]
        ndof = len(xs) - 3 #number of bins in the fitting range - degrees of freedom for gaussian, 3 = N, mu, std
        chi2_5percent_sig = stats.chi2.ppf(1-0.05,ndof)
        popt, pcov = curve_fit(algos.gauss,xs,ys,p0=[bins[max_x],bins[max_x],max_y])
        if (verbose==True):
            print("pcov: ",pcov)
        mu_err = np.sqrt(pcov[0][0])
        std_err = np.sqrt(pcov[1][1])


        chi2 = 0

        for i,j in zip(xs,ys):
            # print(i,j)
            value = (j - algos.gauss(i,*popt))**2 #divide by the sqaure troot of the count assume poisson error
            if np.sqrt(j) ==0:
                chi2+=value/1
            else:
                chi2+=value/(j**2)

        if verbose == True: print("chi2 (not reduced):",  chi2)

        if (Plot == True) and (axis == None):
            plt.figure(figsize=[17,9])
            plt.hist(bins[:-1],bins=bins,weights=counts,histtype="step")
            if (chi2< chi2_5percent_sig):
                plt.plot(xs,algos.gauss(xs,*popt),color = "g",label="Gaussian fit, chi2: {} , ndof: {}, 5 pcnt sig level: {} , Fit Quality: {}".format(round(chi2,2),ndof,round(chi2_5percent_sig, 2),"GOOD!"))
                # plt.text(0,0,"GOOD FIT!")
                plt.legend(labelcolor="g")
            else:
                plt.plot(xs,algos.gauss(xs,*popt),color = "g",label="Gaussian fit, chi2: {} , ndof: {}, 5 pcnt sig level: {} , Fit Quality: {}".format(round(chi2,2),ndof,round(chi2_5percent_sig, 2),"BAD!"))
                plt.legend(labelcolor="r")
            plt.ylabel("Counts")
            plt.xlabel("Hist arb. units")
            plt.show()
            plt.close()

        if (Plot == True) and (axis != None):
            axis.hist(bins[:-1], bins=bins, weights=counts, histtype="step")
            fit_quality = "GOOD!" if chi2 < chi2_5percent_sig else "BAD!"
            label_text = (
                f"Gaussian fit\n"
                f"χ²: {round(chi2, 2)}, ndof: {ndof}\n"
                f"5% sig level: {round(chi2_5percent_sig, 2)}\n"
                f"Fit Quality: {fit_quality}"
            )
            color = "g" if fit_quality == "GOOD!" else "r"
            axis.plot(xs, algos.gauss(xs, *popt), color=color, label=label_text)
            axis.legend(labelcolor=color, fontsize=10, loc="upper right", frameon=False, handletextpad=0.5, labelspacing=0.4, alignment="left")
            axis.text(0.95, 0.75, f"Git Tag: {GitVersion}",horizontalalignment='right', verticalalignment='top', transform=axis.transAxes, fontsize=10)
            axis.set_ylabel("Counts")
            axis.set_xlabel("Hist arb. units")
            for label in axis.get_xticklabels():
                label.set_rotation(45)

        countdensity=popt[2]/(bins[1] - bins[0])
        return popt[0], np.abs(popt[1]), mu_err, std_err, chi2, ndof, countdensity
    
    """ - OLD GAUSSIAN FIT TO DCR
    def GetGaussianDCR(self,pkt,pkp,OnePELowerThreshold,OnePEUpperThreshold, mu, sigma,nbins=200, pk_ch = None, 
                    channel_number = None,axis=None):
        DN_Roi_start_samples = self.DarkNoiseRoiStart*self.SamplingRate                         #DCR ROI Lower threshold in Samples
        DN_Roi_end_samples   = self.DarkNoiseRoiTime*self.SamplingRate + DN_Roi_start_samples   #DCR ROI upper threshold in Samples
        roi_samples = (pkt > DN_Roi_start_samples) & (pkt < DN_Roi_end_samples)
        
        Nevents=len(pkp)

        if channel_number is not None:
            ch_mask = (pk_ch == channel_number)
            mask = roi_samples & ch_mask
        else:
            mask = roi_samples
        
        #mask= mask & (pkp > OnePELowerThreshold) &(pkp<100)

        pkpROI=pkp[mask]
        pkpflat=pkpROI.flatten()

        counts, bin_edges = np.histogram(pkpflat, bins=nbins, range=(0,1967 ))
        bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
        binwidth=bin_edges[1]-bin_edges[0]

        g = np.exp(-0.5 * ((bin_centers - mu)/sigma)**2)

        #binmask = (((bin_centers - mu) <2* sigma) & ((bin_centers - mu) > -sigma))
        #binmask2=(((bin_centers - mu) < 2*sigma) & ((bin_centers - mu) > -0.5 *sigma))
        #A_est = np.maximum(np.sum(counts[binmask] * g[binmask]) / np.sum(g[binmask]**2),(np.max(counts[binmask2])) )

        A_density=(0.01*Nevents)/(OnePEUpperThreshold-OnePELowerThreshold)
        A_est=A_density * binwidth

        predicted = A_est * np.exp(-0.5 * ((bin_centers - mu) / sigma) ** 2)

        capped = np.minimum(counts.astype(float), predicted)
        gaussianDCRcount = float(np.sum(capped))  
        predictedcounts=(A_est*sigma*(2*np.pi)**1/2)/(bin_edges[1]-bin_edges[0])

        print("mean, STD:",mu,sigma)
        print("bin_centers min/max:", bin_centers.min(), bin_centers.max())
        print("bin_width:", bin_edges[1]-bin_edges[0])
        print("nbins:", len(bin_centers))
        print("bins in ±2σ:", np.sum(np.abs(bin_centers-mu) < 2*sigma))
        print("amplitude:",A_est)
        print("predictedcounts:",predictedcounts)

        plt.figure(figsize=(8, 5))
        plt.step(bin_centers, counts, where='mid',
                color='black', label='Data (pkp in ROI)')

        # Plot smooth Gaussian fit line
        x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
        y_smooth = A_est * np.exp(-0.5 * ((x_smooth - mu) / sigma)**2)

        plt.plot(x_smooth, y_smooth,
                color='red', linewidth=2,
                label=f'Gaussian Fit\nμ={mu:.3g}, σ={sigma:.3g}')

        # Optional: mark ±2σ region
        plt.axvspan(OnePELowerThreshold, OnePEUpperThreshold,
                alpha=0.15, color='blue', label='pe threshold window')

        plt.xlabel("pkp")
        plt.ylabel("Counts per bin")
        plt.title("Gaussian Fit to pkp in ROI")
        plt.legend()

        plt.tight_layout()          
        #plt.savefig(f'GaussianDCR nbins=200 channel={channel_number}.png')
        plt.close()

        return (gaussianDCRcount,pkpflat)
    """    
    def getpkproi(self,pkt,pkp,pk_ch=None,channel_number=None):
        """Making a flat pkp in the prelaser ROI for DCR"""
        DN_Roi_start_samples = self.DarkNoiseRoiStart*self.SamplingRate                         #DCR ROI Lower threshold in Samples
        DN_Roi_end_samples   = self.DarkNoiseRoiTime*self.SamplingRate + DN_Roi_start_samples   #DCR ROI upper threshold in Samples
        roi_samples = (pkt > DN_Roi_start_samples) & (pkt < DN_Roi_end_samples)        

        if channel_number is not None:
            ch_mask = (pk_ch == channel_number)
            mask = roi_samples & ch_mask
        else:
            mask = roi_samples
        
        pkpROI=pkp[mask]
        pkpflat=pkpROI.flatten()


        return pkpflat

    def GetGSDCR(self,pkp,pkpflat,OnePELowerThreshold,OnePEUpperThreshold, mu, sigma,countdensity,Plot=True,nbins=500, pk_ch = None, 
                    channel_number = None,axis=None,):
        """This function calculates the Gaussian Subtracted Dark Count Rate in the pre-trigger region, with a fitted pre-trigger 
        pedestal gaussian subtracted from the one pe amplitude region

        GSDCR := the mean number number of hits in a pre-trigger roi with pedestal fitted Gaussian subtracted

        The DCR in this function will be in units of [#/Roitime] this is then normalised -> [Hz/mm^2]
        
        Worked out as number of hits per waveform in ROI (ie [#/Roitime] )"""

        n_waveforms = pkp.shape[0]
        counts, bin_edges = np.histogram(pkpflat, bins=nbins, range=(0,OnePEUpperThreshold*3))
        bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
        binwidth=bin_edges[1]-bin_edges[0]

        countamp=countdensity*binwidth

        predicted = countamp * np.exp(-0.5 * ((bin_centers - mu) / sigma) ** 2)
                                   
        
        onePEmask = (bin_centers > OnePELowerThreshold) & (bin_centers< OnePEUpperThreshold)
        capped = np.minimum(counts.astype(float), predicted)
        total_counts = np.sum(counts[onePEmask])
        gaussian_overlap = np.sum(capped[onePEmask]) 

        gaussianDCRcount = float(total_counts - gaussian_overlap)
        GSDN_rate=gaussianDCRcount/n_waveforms

        # Rough statistical uncertainty assuming Poisson counting statistics
        gaussianDCRcount_err = np.sqrt(total_counts + gaussian_overlap)
        GSDN_rate_err = gaussianDCRcount_err / n_waveforms

        #print("mean, STD:",mu,sigma)
        #print("bin_centers min/max:", bin_centers.min(), bin_centers.max())
        #print("bin_width:", bin_edges[1]-bin_edges[0])
        #print("nbins:", len(bin_centers))
        #print("bins in ±2σ:", np.sum(np.abs(bin_centers-mu) < 2*sigma))
        #print("amplitude:",countamp)

        x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
        y_smooth = countamp * np.exp(-0.5 * ((x_smooth - mu) / sigma)**2)

        if Plot==True& (axis!=None):
            
            ax = axis
            ax.step(bin_centers, counts, where='mid', color='black', label='Data (pkp in ROI)')
            ax.plot(x_smooth, y_smooth, color='red', linewidth=2,
                    label=f'Gaussian Fit\nμ={mu:.3g}, σ={sigma:.3g}')
            ax.axvspan(OnePELowerThreshold, OnePEUpperThreshold,
                    alpha=0.15, color='blue', label='1PE threshold window')
            ax.plot([], [], ' ', label=f'DCR count = {gaussianDCRcount:.3g}')
            ax.set_xlabel("pkp")
            ax.set_ylabel("Counts per bin")
            ax.set_title(f"Gaussian Fit to 0pe pkp in ROI ({channel_number})")
            ax.legend()

        if Plot==True & (axis == None):
            plt.figure(figsize=(8, 5))
            plt.step(bin_centers, counts, where='mid',
                    color='black', label='Data (pkp in ROI)')

            plt.plot(x_smooth, y_smooth,
                    color='red', linewidth=2,
                    label=f'Gaussian Fit\nμ={mu:.3g}, σ={sigma:.3g}')

            #mark ±2σ region
            plt.axvspan(OnePELowerThreshold, OnePEUpperThreshold,
                    alpha=0.15, color='blue', label='pe threshold window')
            plt.plot([], [], ' ', label=f'DCR count={gaussianDCRcount:.3g}')
            plt.xlabel("pkp")
            plt.ylabel("Counts per bin")
            plt.title(f"Gaussian Fit to 0pe pkp in ROI({channel_number})")
            plt.legend()
            plt.show()
            plt.close
        
        if (Plot==False):
            pass


        return (GSDN_rate, gaussianDCRcount, GSDN_rate_err, gaussianDCRcount_err) 

    def GetDCR(self,pkt,pkp,OnePELowerThreshold,OnePEUpperThreshold,Plot=False,nbins=200, pk_ch = None, channel_number = None,axis=None,):
        '''
        This Function Calcualtes the Dark Count Rate in the PRe -trigger region

        DCR := the mean number number of hits in a pre-trigger roi

        The DCR in this function will be in units of [#/Roitime] this is then normalised -> [Hz/mm^2]
        '''
        # if (pk_ch == None) & (channel_number !=None):
        #     print("[Error] if you supply channel data you also need to specify a channel number")
        #     return
        # elif (pk_ch != None) & (channel_number ==None):
        #     print("[Error] if you supply channel data you also need to specify a channel number")
        #     return

        DN_Roi_start_samples = self.DarkNoiseRoiStart*self.SamplingRate                         #DCR ROI Lower threshold in Samples
        DN_Roi_end_samples   = self.DarkNoiseRoiTime*self.SamplingRate + DN_Roi_start_samples   #DCR ROI upper threshold in Samples
        print("start samples: ", DN_Roi_start_samples)
        print("end samples: ", DN_Roi_end_samples)
        print("1PEUpperThreshold", OnePEUpperThreshold)
        print("1PELowerThreshold", OnePELowerThreshold)

        # #From DAta keep only 1pe hits in DCR ROI
        # roiDN = pkt[(   (   pkt  > DN_Roi_start_samples  )  &  (  pkt < DN_Roi_end_samples  ) ) & (  (pkp > OnePELowerThreshold)  &  (pkp< OnePEUpperThreshold)  )]

        #Count the number of 1pe hits in DCR ROI per waveform
        if channel_number == None:
            DN_Hits = np.count_nonzero((   (   pkt  > DN_Roi_start_samples  )  &  (  pkt < DN_Roi_end_samples  ) ) & (  (pkp > OnePELowerThreshold)  &  (pkp < OnePEUpperThreshold)  ),axis=1)
        else:
            DN_Hits = np.count_nonzero((   (   pkt  > DN_Roi_start_samples  )  &  (  pkt < DN_Roi_end_samples  ) ) & (  (pkp > OnePELowerThreshold)  &  (pkp < OnePEUpperThreshold) & (pk_ch == channel_number) ),axis=1)
        # print(DN_Hits)
        #Return the mean number of DCR events amd std. deviation over dataset
        DN_rate, DN_std = np.mean(DN_Hits), np.std(DN_Hits) # mean, std
        DN_rate_err     = DN_std/np.sqrt(len(DN_Hits))     #Error on the mean
        print("DN_rate:", DN_rate)

        if (Plot ==True) & (axis == None):
            plt.figure(figsize=[17,9])
            if channel_number ==None:
                plt.hist(pkt.flatten()[pkt.flatten()>self.fill_value], bins=nbins, histtype="step", color = "m")
            if channel_number != None:
                plt.hist(pkt[(pkt>self.fill_value) & (pk_ch == channel_number+1)].flatten(), bins=nbins, histtype="step", color = "m")
            plt.axvspan(DN_Roi_start_samples, DN_Roi_end_samples, alpha=0.5, color='k',label = "Dark Noise ROI")
            plt.yscale("log")
            plt.xlabel("Time [sample]")
            plt.ylabel("Counts")
            plt.title("Dark Noise Roi over pkt")
            plt.legend()
            plt.show()
            plt.close
        if (Plot ==True) & (axis != None):
            if channel_number ==None:
                axis.hist(pkt.flatten()[pkt.flatten()>self.fill_value], bins=nbins, histtype="step", color = "m")
            if channel_number != None:
                axis.hist(pkt[(pkt>self.fill_value) & (pk_ch == channel_number)].flatten(), bins=nbins, histtype="step", color = "m")
            axis.axvspan(DN_Roi_start_samples, DN_Roi_end_samples, alpha=0.5, color='k',label = "Dark Noise ROI")
            axis.set_yscale("log")

            # axis.text(0.95, 0.75, "Git Tag: {}".format(subprocess.check_output(['git', 'describe','--tags']).strip().decode()),
            # horizontalalignment='right', verticalalignment='top', transform=axis.transAxes)
            axis.text(0.95, 0.75, f"Git Tag: {GitVersion}",horizontalalignment='right', verticalalignment='top', transform=axis.transAxes, fontsize=10)
            axis.set_xlabel("Time [sample]")
            axis.set_ylabel("Counts")
            axis.set_title("Channel {} Dark Noise Roi over pkt".format(channel_number+1))
            axis.legend()
        if (Plot==False):
            pass


        return DN_rate, DN_std, DN_rate_err

    def GetCDA(self,pktUnFlattened,pkpUnFlattened,OnePEUpperThreshold,OnePELowerThreshold,nbins, Plot=False, pk_ch = None, channel_number = None,axis=None):
        '''
        This function calculates the mean number of Correlated delayed avalanches
        '''

        #find position of the trigger
        (pk_t_hist,pk_t_bins) = np.histogram(pktUnFlattened.flatten()[pktUnFlattened.flatten()>0],bins=nbins)
        max_val               = np.where(pk_t_hist == max(pk_t_hist))   #This gives the index position of the maximum value in pk_t histogram i.e teh time
        TimeMaxValBin         = pk_t_bins[max_val]


        #convert the roi time in seconds to sample number
        trig_offset_left   = self.SamplingRate * self.PromptRoiLeft
        trig_offset_right  = self.SamplingRate * self.PromptRoiRight    #convert trigger offset time to number of samples
        roi_samples        = self.SamplingRate * self.CDAroiTime        #convert CDA roi_sizee in seconds to number of samples using sample_freq

        trig_window_left   = TimeMaxValBin - (trig_offset_left)         #Define left hand side of trigger window
        trig_window_right  = TimeMaxValBin + (trig_offset_right)




        #keep waveforms with a hit in the prompt roi
        if channel_number == None:
            WaveformsWithHitinPromptROI = pktUnFlattened[np.any(( (pktUnFlattened < trig_window_right) & (pktUnFlattened > trig_window_left) ) & (  ( pkpUnFlattened > OnePELowerThreshold ) & ( pkpUnFlattened < OnePEUpperThreshold )) ,axis=1)]
            CDACountPerWaveform         = np.count_nonzero((WaveformsWithHitinPromptROI> trig_window_right) & (WaveformsWithHitinPromptROI<(trig_window_right+roi_samples) ),axis=1)
        else:
            WaveformsWithHitinPromptROI = pktUnFlattened[np.any(( (pktUnFlattened < trig_window_right) & (pktUnFlattened > trig_window_left) ) & (  ( pkpUnFlattened > OnePELowerThreshold ) & ( pkpUnFlattened < OnePEUpperThreshold )) & (pk_ch == channel_number),axis=1)]
            CDACountPerWaveform         = np.count_nonzero((WaveformsWithHitinPromptROI> trig_window_right) & (WaveformsWithHitinPromptROI<(trig_window_right+roi_samples) ),axis=1)
        CDA_Mean, CDA_std           = np.mean(CDACountPerWaveform), np.std(CDACountPerWaveform)
        CDA_mean_Err                = CDA_std/np.sqrt(len(CDACountPerWaveform))


        if (Plot ==True) and (axis == None):
            plt.figure(figsize=[17,9])
            if channel_number == None:
                plt.hist(pktUnFlattened.flatten()[pktUnFlattened.flatten()>self.fill_value],bins=250,histtype="step",color="m")
            else:
                plt.hist(pktUnFlattened[(pktUnFlattened>self.fill_value) & (pk_ch==channel_number)].flatten(),bins=250,histtype="step",color="m")
            plt.axvspan(trig_window_left[0], trig_window_right[0], alpha=0.5, color='k',label = "Trigger ROI")
            plt.axvspan(trig_window_right[0], trig_window_right[0]+roi_samples, alpha=0.5, color='red',label = "CDA ROI")
            plt.title("CDA ROI's over pkt")
            plt.xlabel("time [Sample]")
            plt.ylabel("Counts")
            plt.yscale("log")
            plt.legend()
            plt.show()
            plt.close()
        if (Plot ==True) and (axis!= None):
            if channel_number == None:
                axis.hist(pktUnFlattened.flatten()[pktUnFlattened.flatten()>self.fill_value],bins=250,histtype="step",color="m")
            else:
                axis.hist(pktUnFlattened[(pktUnFlattened>self.fill_value) & (pk_ch==channel_number)].flatten(),bins=250,histtype="step",color="m")
            axis.axvspan(trig_window_left[0], trig_window_right[0], alpha=0.5, color='k',label = "Trigger ROI")
            axis.axvspan(trig_window_right[0], trig_window_right[0]+roi_samples, alpha=0.5, color='red',label = "CDA ROI")
            axis.set_title("Channel {} CDA ROI's over pkt".format(channel_number+1))
            axis.set_xlabel("Time [Sample]")
            axis.set_ylabel("Counts")
            axis.set_yscale("log")
            axis.text(0.95, 0.75, f"Git Tag: {GitVersion}",horizontalalignment='right', verticalalignment='top', transform=axis.transAxes, fontsize=10)
            axis.legend()
        if (Plot==False):
            pass


        return CDA_Mean, CDA_std, CDA_mean_Err

    def GetMeanAPA(self,pkp,pkt,Mean1pe,std1pe, laser = False , NAPA_ROI_left = None, NAPA_ROI_right = None,pk_ch = None, channel_number = None, Plot=False,axis=None):
        '''
        if laser = True correct for poisson variation of the laser pulse
        for laser to be used you will also need to define the left and right roi's
        '''


        if channel_number ==None:
            pkp_flattened    = pkp.flatten()
            pkt_flattened    = pkt.flatten()
        else:
            pkp_flattened    = pkp[(pk_ch == channel_number) & (pkp > self.fill_value)].flatten()
            pkt_flattened    = pkt[(pk_ch == channel_number) & (pkt > self.fill_value)].flatten()
        Darknoisestart   = self.DarkNoiseRoiStart*self.SamplingRate #start 125 samples in
        DarknoiseSamples = self.DarkNoiseRoiTime*self.SamplingRate + Darknoisestart


        ###cut pkp in Darknoise region
        pkp_DarkNoiseROi            = pkp_flattened[(pkt_flattened>Darknoisestart) & (pkt_flattened<DarknoiseSamples) ]
        pkp_DarkNoiseROi_normalised = pkp_DarkNoiseROi/Mean1pe
        pkp_0_5_pe                  = pkp_DarkNoiseROi_normalised[pkp_DarkNoiseROi_normalised  > 0.66]


        if laser == False:
            pkp_normStdError_0_5pe      = np.std(pkp_0_5_pe) / np.sqrt( len( pkp_0_5_pe )  )
            N_APA                       = np.mean(pkp_0_5_pe) - 1
            if Plot == True and axis == None:
                plt.figure(figsize=[17,9])
                if channel_number == None:
                    plt.hist(pkt.flatten()[pkt.flatten()>self.fill_value],bins=250,histtype="step",color="m")
                else:
                    plt.hist(pkt[(pkt>self.fill_value) & (pk_ch==channel_number)].flatten(),bins=250,histtype="step",color="m")
                plt.axvspan(Darknoisestart, DarknoiseSamples, alpha=0.5, color='k',label = "NAPA ROI")
                plt.axvspan(trig_window_right[0], trig_window_left[0], alpha=0.5, color='red',label = "Trigger ROI")
                plt.title("NAPA (Data Driven) ROI's over pkt")
                plt.xlabel("Time [Sample]")
                plt.ylabel("Counts")
                plt.yscale("log")
                plt.legend()
                plt.show()
                plt.close()
            if Plot == True and axis != None:
                if channel_number == None:
                    axis.hist(pkt.flatten()[pkt.flatten()>self.fill_value],bins=250,histtype="step",color="m")
                else:
                    axis.hist(pkt[(pkt>self.fill_value) & (pk_ch==channel_number)].flatten(),bins=250,histtype="step",color="m")
                axis.axvspan(Darknoisestart, DarknoiseSamples, alpha=0.5, color='k',label = "NAPA ROI")
                axis.axvspan(trig_window_right[ 0], trig_window_left[0], alpha=0.5, color='red',label = "Trigger ROI")
                axis.set_title("Channel {} NAPA (Data Driven) ROI's over pkt".format(channel_number))
                axis.text(0.95, 0.75, f"Git Tag: {GitVersion}",horizontalalignment='right', verticalalignment='top', transform=axis.transAxes, fontsize=10)
                axis.set_xlabel("Time [Sample]")
                axis.set_ylabel("Counts")
                axis.set_yscale("log")
                axis.legend()
            if Plot == False:
                pass



        if laser == True:

            (pk_t_hist,pk_t_bins) = np.histogram(pkt.flatten()[pkt.flatten()>0],bins=200)
            max_val               = np.where(pk_t_hist == max(pk_t_hist))   #This gives the index position of the maximum value in pk_t histogram i.e teh time
            TimeMaxValBin         = pk_t_bins[max_val]

            trig_offset_left   = self.SamplingRate * self.PromptRoiLeft
            trig_offset_right  = self.SamplingRate * self.PromptRoiRight    #convert trigger offset time to number of samples
            trig_window_left   = TimeMaxValBin - (trig_offset_left)         #Define left hand side of trigger window
            trig_window_right  = TimeMaxValBin + (trig_offset_right)

            if (NAPA_ROI_left == None):
                #if no NAPA ROIs are given use the darknoise region
                NAPA_ROI_left = Darknoisestart
            if (NAPA_ROI_right == None):
                #if no NAPA ROIs are given use the darknoise region
                NAPA_ROI_right = DarknoiseSamples


            pkp_NAPA = pkp[((pkt>NAPA_ROI_left) & (pkt<NAPA_ROI_right))] / Mean1pe
            no_pulse=0
            for wf,pt in zip(pkp,pkt):
                if np.sum(wf[((pt>trig_window_left) & (pt<trig_window_right))]) == 0:
                    no_pulse +=1

            mean_poisson = -1*np.log(no_pulse/len(pkp))
            print("Poisson Mean: ", mean_poisson)
            N_APA                    = (np.mean(pkp_NAPA) / mean_poisson) - 1
            pkp_normStdError_0_5pe   = np.std(pkp_NAPA) / np.sqrt( len( pkp_NAPA )  )
            if Plot==True and axis == None:
                plt.figure(figsize=[17,9])
                if channel_number == None:
                    plt.hist(pkt.flatten()[pkt.flatten()>self.fill_value],bins=250,histtype="step",color="m")
                else:
                    plt.hist(pkt[(pkt>self.fill_value) & (pk_ch==channel_number)].flatten(),bins=250,histtype="step",color="m")
                plt.axvspan(NAPA_ROI_left, NAPA_ROI_right, alpha=0.5, color='k',label = "NAPA ROI")
                plt.axvspan(trig_window_right[0], trig_window_left[0], alpha=0.5, color='red',label = "Trigger ROI")
                plt.title("NAPA (Laser) ROI's over pkt")
                plt.xlabel("Time [Sample]")
                plt.ylabel("Counts")
                plt.yscale("log")
                plt.legend()
                plt.show()
                plt.close()
            if Plot == True and axis !=None:
                if channel_number == None:
                    axis.hist(pkt.flatten()[pkt.flatten()>self.fill_value],bins=250,histtype="step",color="m")
                else:
                    axis.hist(pkt[(pkt>self.fill_value) & (pk_ch==channel_number)].flatten(),bins=250,histtype="step",color="m")
                axis.axvspan(NAPA_ROI_left, NAPA_ROI_right, alpha=0.5, color='k',label = "NAPA ROI")
                axis.axvspan(trig_window_right[0], trig_window_left[0], alpha=0.5, color='red',label = "Trigger ROI")
                axis.set_title("Channel {} NAPA (Laser) ROI's over pkt".format(channel_number))
                axis.set_xlabel("Time [Sample]")
                axis.set_ylabel("Counts")
                axis.yscale("log")
                axis.text(0.95, 0.75, f"Git Tag: {GitVersion}",horizontalalignment='right', verticalalignment='top', transform=axis.transAxes, fontsize=10)
                axis.legend()
            if Plot == False:
                pass


        return N_APA, pkp_normStdError_0_5pe


    def GetDiCT(self,pkp,pkt,Mean1pe,plot=False,nbins=250,pk_ch = None, channel_number = None,axis1 = None, axis2 = None):
        '''
        This function calcuates the mean Number of Additional Prompt Avalanches.
        IT is defined as the number of avalanches larger than 1.5pe divided by the number of avalanches bigger
        than 0.5 pe. since this essentially a coutning exercise the error on the measurement is given as the sqrt
        of the counts, error propogation is used for the fraction.
        '''
        if channel_number ==None:
            pkp_flattened    = pkp.flatten()
            pkt_flattened    = pkt.flatten()
        else:
            pkp_flattened    = pkp[(pk_ch == channel_number) & (pkp > self.fill_value)].flatten()
            pkt_flattened    = pkt[(pk_ch == channel_number) & (pkt > self.fill_value)].flatten()

        Darknoisestart   = self.DarkNoiseRoiStart*self.SamplingRate
        DarknoiseSamples = self.DarkNoiseRoiTime*self.SamplingRate + Darknoisestart


        ###cut pkp in Darknoise region
        pkp_DarkNoiseROi            = pkp_flattened[(pkt_flattened>Darknoisestart) & (pkt_flattened<DarknoiseSamples) ]
        pkp_DarkNoiseROi_normalised = pkp_DarkNoiseROi/Mean1pe
        pkp_0_5_pe                  = pkp_DarkNoiseROi_normalised[pkp_DarkNoiseROi_normalised  > 0.5]
        pkp_1_5_pe                  = pkp_DarkNoiseROi_normalised[pkp_DarkNoiseROi_normalised  > 1.5]
        NA_0p5pe                    = len(pkp_0_5_pe)
        NA_1p5pe                    = len(pkp_1_5_pe)
        print("# Avalanches > 0.5pe  :" ,len(pkp_0_5_pe))
        print("# Avalanches > 1.5pe  :" ,len(pkp_1_5_pe))

        DiCTProb       = NA_1p5pe/NA_0p5pe
        DiCTProb_Error = algos.ErrorDivide(NA_1p5pe,NA_0p5pe,np.sqrt(NA_1p5pe),np.sqrt(NA_0p5pe))

        if plot ==True and axis1 == None and axis2 ==None:
            plt.figure(figsize=[17,9])
            plt.hist(pkp_DarkNoiseROi_normalised,bins=nbins,range=[0,7])
            plt.xlabel("Normalised pe")
            plt.ylabel("Counts")
            plt.title("Normalised SPE")
            plt.yscale("log")

            plt.figure(figsize=[17,9])
            plt.hist(pkp_0_5_pe,bins=nbins, range=[0,7])
            plt.xlabel("Amplitude > 0.5p.e.")
            plt.ylabel("Counts")
            plt.title("APA spectrum")
        if plot == True and axis1 !=None and axis2 != None:
            axis1.hist(pkp_DarkNoiseROi_normalised,bins=nbins,range=[0,7])
            axis1.set_xlabel("Normalised pe")
            axis1.set_ylabel("Counts")
            axis1.set_title("Channel {} Normalised SPE".format(channel_number+1))
            axis1.text(0.95, 0.75, f"Git Tag: {GitVersion}",horizontalalignment='right', verticalalignment='top', transform=axis1.transAxes, fontsize=10)
            axis1.set_yscale("log")

            axis2.hist(pkp_0_5_pe,bins=nbins, range=[0,7])
            axis2.set_xlabel("Amplitude > 0.5p.e.")
            axis2.set_ylabel("Counts")
            axis2.set_title("Channel {} APA spectrum".format(channel_number+1))

            axis2.text(0.95, 0.75, f"Git Tag: {GitVersion}",horizontalalignment='right', verticalalignment='top', transform=axis2.transAxes, fontsize=10)
        if plot == False:
            pass


        return DiCTProb, DiCTProb_Error
