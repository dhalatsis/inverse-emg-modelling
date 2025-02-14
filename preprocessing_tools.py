import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import scipy
from scipy import signal
from numpy import linalg
from scipy.fft import fft
import matplotlib.pyplot as plt
import sklearn 
from sklearn.metrics import silhouette_samples, r2_score
from sklearn.cluster import KMeans
# import numba
from copy import deepcopy
# from numba import jit

##################################### FILTERING TOOLS #######################################################



def single_diff_montage_grid(data, r_map, c_map):
    """
    Spatial first order differentiation on channel data, aligned w.r.t montage columns.
    This is written only for 64 channel grids atm, where the 0th channel is non existent (so need -1 compensation).
    """
    reformatted_data = []
 
    for i in range(c_map):  # for now 5 columns
        if i == 0:
            single_diff_data = np.diff(data[i * r_map: (i + 1) * r_map - 1, :], n=1, axis=0)
        else:
            single_diff_data = np.diff(data[(i * r_map) - 1: (i + 1) * r_map -1, :], n=1, axis=0)
 
        reformatted_data.append(single_diff_data)
 
    # Concatenate the reformatted_data along the row axis
    reformatted_data = np.concatenate(reformatted_data, axis=0)
 
    return reformatted_data

 
def double_diff_montage_grid(data, r_map, c_map):
    """
    Spatial second order differentiation on channel data, aligned w.r.t montage columns.
    (ch2-ch1) + (ch2-ch3)
    This is written only for 64 channels atm, where the 0th channel is non existent (so need -1 compensation).
    """
 
    reformatted_data = []
 
    for i in range(c_map):  # for now 5 columns
        if i == 0:
            first_diff = np.diff(data[i * r_map: (i + 1) * r_map - 1, :], n=1, axis=0)
        else:
            first_diff = np.diff(data[(i * r_map) - 1: (i + 1) * r_map - 1, :], n=1, axis=0)
 
        # Perform second order differentiation as per the formula (ch2-ch1) + (ch2-ch3)
        second_diff = first_diff[:-1, :] - first_diff[1:, :]
 
        reformatted_data.append(second_diff)
 
    # Concatenate the reformatted_data along the row axis
    reformatted_data = np.concatenate(reformatted_data, axis=0)
 
    return reformatted_data
 

def notch_filter(signal,fsamp,to_han = False):

    """ Implementation of a notch filter, where the frequencies of the line interferences are unknown. Therefore, interference is defined
    as frequency components with magnitudes greater than 5 stds away from the median frequency component magnitude in a window of the signal
    - assuming you will iterate this function over each grid 
    
    IMPORTANT!!! There is a small difference in the output of this function and that in MATLAB, best guess currently is differences in inbuilt FFT implementation."""

    bandwidth_as_index = int(round(4*(np.shape(signal)[1]/fsamp)))
    # width of the notch filter's effect, when you intend for it to span 4Hz, but converting to indices using the frequency resolution of FT
    filtered_signal = np.zeros([np.shape(signal)[0],np.shape(signal)[1]])


    for chan in range(np.shape(signal)[0]):

        if to_han:
            hwindow = scipy.signal.hann(np.shape(signal[chan,:])[0])
            final_signal = signal[chan,:]* hwindow
        else:
            final_signal = signal[chan,:]

        fourier_signal = np.fft.fft(final_signal)
        fourier_interf = np.zeros(len(fourier_signal),dtype = 'complex_')
        #interf2remove = np.zeros(len(fourier_signal),dtype=np.int)
        interf2remove = np.zeros(len(fourier_signal), dtype=np.int32)
        window = fsamp
        tracker = 0
    
        for interval in range(0,len(fourier_signal)-window+ 1,window): # so the last interval will start at len(fourier_emg) - window
            
            # range(start, stop, step)
            median_freq = np.median(abs(fourier_signal[interval+1:interval+window+1])) # so interval + 1: interval + window + 1
            std_freq = np.std(abs(fourier_signal[interval+1:interval+window+1]))
            # interference is defined as when the magnitude of a given frequency component in the fourier spectrum
            # is greater than 5 times the std, relative to the median magnitude
            label_interf = list(np.where(abs(fourier_signal[interval+1:interval+window+1]) > median_freq+5*std_freq)[0]) # np.where gives tuple, element access to the array
            # need to shift these labels to make sure they are not relative to the window only, but to the whole signal
            label_interf = [x + interval + 1 for x in label_interf] # + 1 since we are already related to a +1 shifted array?
    
            if label_interf: # if a list exists
                for i in range(int(-np.floor(bandwidth_as_index/2)),int(np.floor(bandwidth_as_index/2)+1)): # so as to include np.floor(bandwidth_as_index/2)
                    
                    temp_shifted_list = [x + i for x in label_interf]
                    interf2remove[tracker: tracker + len(label_interf)] = temp_shifted_list
                    tracker = tracker + len(label_interf)
        
        # we only take the first half of the signal, we need a compensatory step for the second half given we haven't wrapped the FT yet
        indexf2remove = np.where(np.logical_and(interf2remove >= 0 , interf2remove <= len(fourier_signal)/2))[0]
        fourier_interf[interf2remove[indexf2remove]] = fourier_signal[interf2remove[indexf2remove]]
        corrector = int(len(fourier_signal) - np.floor(len(fourier_signal)/2)*2)  # will either be 0 or 1 (0 if signal length is even, 1 if signal length is odd)
        # wrapping FT
        fourier_interf[int(np.ceil(len(fourier_signal)/2)):] = np.flip(np.conj(fourier_interf[1: int(np.ceil(len(fourier_signal)/2)+1- corrector)])) # not indexing first because this is 0Hz, not to be repeated
        filtered_signal[chan,:] = signal[chan,:] - np.fft.ifft(fourier_interf).real

    return filtered_signal


   
   
def bandpass_filter(signal,fsamp, emg_type = 'surface'):

    """ Generic band-pass filter implementation and application to EMG signal  - assuming that you will iterate this function over each electrode """

    """IMPORTANT!!! There is a difference in the default padding length between Python and MATLAB. For MATLAB -> 3*(max(len(a), len(b)) - 1),
    for Python scipy -> 3*max(len(a), len(b)). So I manually adjusted the Python filtfilt to pad by the same amount as in MATLAB, if you don't the results will not match across
    lanugages. """   

    if emg_type == 'surface':
        lowfreq = 20
        highfreq = 500
        order = 2
    elif emg_type == 'intra':
        lowfreq = 100
        highfreq = 4400
        order = 3

    # get the coefficients for the bandpass filter
    nyq = fsamp/2
    lowcut = lowfreq/nyq
    highcut = highfreq/nyq
    [b,a] = scipy.signal.butter(order, [lowcut,highcut],'bandpass') # the cut off frequencies should be inputted as normalised angular frequencies

    filtered_signal = np.zeros([np.shape(signal)[0],np.shape(signal)[1]])
    # construct and apply filter
    for chan in range(np.shape(signal)[0]):
        
        filtered_signal[chan,:] = scipy.signal.filtfilt(b,a,signal[chan,:],padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    
    return filtered_signal



def moving_mean1d(v,w):
    """ Moving average filter that replicates the method of movmean in MATLAB
    v is a 1 dimensional vector to be filtered via a moving average
    w is the window length of this filter """

    u = v.copy()
    w_temp = w
    n = len(v)-1

    if w_temp % 2 != 0:

        w = int(np.ceil(w_temp/2))
        for i in range(w):
            u[i] = np.mean(v[0:w+i])
            u[n-i] = np.mean(v[n-(w-1)-i:])

        n1 = 1 + w
        n2 = n - w

        for i in range(n1-1,n2+1):
        
            u[i] = np.mean(v[i - w + 1:i + w])

    else:

        w = int(w_temp/2)
        for i in range(w):
            u[i] = np.mean(v[0:w+i])
            u[n-i] = np.mean(v[n-(w-1)-(i+1):])

        n1 = 1 + w
        n2 = n - w

        for i in range(n1-1,n2+1):
            u[i] = np.mean(v[i - w:i + w ])

    return u
    

################################# CONVOLUTIVE SPHERING TOOLS ##########################################################

def extend_emg(extended_template, signal, ext_factor):

    """ Extension of EMG signals, for a given window, and a given grid. For extension, R-1 versions of the original data are stacked, with R-1 timeshifts.
    Structure: [channel1(k), channel2(k),..., channelm(k); channel1(k-1), channel2(k-1),...,channelm(k-1);...;channel1(k - (R-1)),channel2(k-(R-1)), channelm(k-(R-1))] """

    # signal = self.signal_dict['batched_data'][tracker][0:] (shape is channels x temporal observations)
    from tqdm import tqdm
    nchans = np.shape(signal)[0]
    nobvs = np.shape(signal)[1]
    for i in tqdm(range(ext_factor)):

        extended_template[nchans*i :nchans*(i+1), i:nobvs +i] = signal
  
    return extended_template


def whiten_emg(signal):
    
    """ Whitening the EMG signal imposes a signal covariance matrix equal to the identity matrix at time lag zero. Use to shrink large directions of variance
    and expand small directions of variance in the dataset. With this, you decorrelate the data. """

    # get the covariance matrix of the extended EMG observations

    cov_mat = np.cov(np.squeeze(signal),bias=True)
    print('FINISHED GETTING COVARIANCE MATRIX...')
    # get the eigenvalues and eigenvectors of the covariance matrix
    evalues, evectors  = scipy.linalg.eigh(cov_mat)
    print('FINISHED GETTING EIGENDECOMPOSITION...')
    # in MATLAB: eig(A) returns diagonal matrix D of eigenvalues and matrix V whose columns are the corresponding right eigenvectors, so that A*V = V*D
    # sort the eigenvalues in descending order, and then find the regularisation factor = "average of the smallest half of the eigenvalues of the correlation matrix of the extended EMG signals" (Negro 2016)
    sorted_evalues = np.sort(evalues)[::-1]
    penalty = np.mean(sorted_evalues[len(sorted_evalues)//2:]) # int won't wokr for odd numbers
    penalty = max(0, penalty)


    rank_limit = np.sum(evalues > penalty)-1
    if rank_limit < np.shape(signal)[0]:

        hard_limit = (np.real(sorted_evalues[rank_limit]) + np.real(sorted_evalues[rank_limit + 1]))/2

    # use the rank limit to segment the eigenvalues and the eigenvectors
    evectors = evectors[:,evalues > hard_limit]
    evalues = evalues[evalues>hard_limit]
    diag_mat = np.diag(evalues)
    # np.dot is faster than @, since it's derived from C-language
    # np.linalg.solve can be faster than np.linalg.inv
    whitening_mat = evectors @ np.linalg.inv(np.sqrt(diag_mat)) @ np.transpose(evectors)
    # whitening_mat = evectors @ np.diag(1 / np.diagonal(np.sqrt(diag_mat))) @ np.transpose(evectors)
    print('FINISHED INVERTING DIAGONAL MATRIX...')
    dewhitening_mat = evectors @ np.sqrt(diag_mat) @ np.transpose(evectors)
    whitened_emg =  np.matmul(whitening_mat, signal).real 

    return whitened_emg, whitening_mat, dewhitening_mat



###################################### DECOMPOSITION TOOLS ##################################################################

# @numba.njit
def ortho_gram_schmidt(w,B):

    """ This is the recommended method of orthogonalisation in Negro et.al 2016,
    documented in Hyvärinen et.al 2000 (fast ICA) """
    basis_projection = 0
    for i in range(B.shape[1]):
        w_history = B[:, i]
        if np.all(w_history == 0):
            continue
        # Estimate the independent components one by one. When we have estimated p independent components, 
        # or p vectors w1; ...; wp; we run the one-unit fixed- point algorithm for wp11; 
        # and after every iteration step subtract from wp11 the “projections”  of the previously estimated p vectors, and then
        # renormalize 
        basis_projection = basis_projection + (np.dot(w, w_history) / np.dot(w_history,w_history)) * w_history
    w = w - basis_projection
    return w


##### CONTRAST FUNCTIONS ########
# Using g' and g'' in the iterations of fast ICA (see Hyvarinen et.al 'Independent Component Analysis: Algorithms and Applications, 
# and Negro et. al 'Multichannel intramuscular and surface EMG decomposition by convolutive blind source separation)

# @numba.njit
# def square(x):
#     return np.square(x)

# @numba.njit
# def skew(x):
#     return np.square(x)

# @numba.njit
# def kurtosis(x):
#     return x**3

# @numba.njit
# def exp(x):
#     return np.exp(-np.square(x)/2)

# @numba.njit
# def logcosh(x):
#     return np.log(np.cosh(x))

# @numba.njit
# def dot_square(x):
#     return 2*x

# @numba.njit
# def dot_skew(x):
#     return 2*x

# @numba.njit
# def dot_kurtosis(x):
#     return 3*(np.square(x))

# @numba.njit
# def dot_exp(x):
#     return -1*(np.exp(-np.square(x)/2)) + np.dot((np.square(x)), np.exp(-np.square(x)/2))

# @numba.njit
# def dot_logcosh(x):
#     return np.tanh(x)

"""
def _logcosh(da, xp, x):
    # As opposed to scikit-learn here we fix alpha = 1 and we vectorize the derivation
    gx = da.tanh(x, x)  # apply the tanh inplace
    g_x = (1 - gx ** 2).mean(axis=-1)
    return gx, g_x
"""

# @numba.njit(fastmath=True)
def fixed_point_alg(w_n, B, Z,cf, dot_cf, its = 500):

    """ Update function for source separation vectors. The code user can select their preferred contrast function using a string input:
    1) square --> x^2
    2) logcosh --> log(cosh(x))
    3) exp  --> e(-x^2/2)
    4) skew --> -x^3/3 
    e.g. skew is faster, but less robust to outliers relative to logcosh and exp
    Upon meeting a threshold difference between iterations of the algorithm, separation vectors are discovered 
    
    The maximum number of iterations (its) and the contrast function type (cf) are already specified, unless alternative input is provided. """
   
    assert B.ndim == 2
    assert Z.ndim == 2
    assert w_n.ndim == 1
    assert its in [500]
    counter = 0
    its_tolerance = 0.0001
    sep_diff = np.ones(its)
    
    B_T_B = B @ B.T
    Z_meaner = Z.shape[1]

    while sep_diff[counter] > its_tolerance and counter < its:

        # transfer current separation vector as the previous arising separation vector
        w_n_1 = w_n.copy()
        # use update function to get new, current separation vector
        wTZ = w_n_1.T @ Z 
        A = dot_cf(wTZ).mean()
        w_n = Z @ cf(wTZ).T / Z_meaner  - A * w_n_1

        # orthogonalise separation vectors
        w_n -= np.dot(B_T_B, w_n)
        
        # normalise separation vectors
        w_n /= np.linalg.norm(w_n)
        counter += 1
        sep_diff[counter] = np.abs(w_n @ w_n_1 - 1)
        # print(counter)

    #print('Exited ')
    return w_n

def get_spikes (w_n,Z, fsamp):

    """ Based on gradient convolutive kernel compensation. Aim to remove spurious discharges to improve the source separation
    vector estimate. Results in a reduction in ISI vairability (by seeking to minimise the covariation in MU discharges)"""

    # Step 4a: 
    source_pred = np.dot(np.transpose(w_n), Z).real # element-wise square of the input to estimate the ith source
    source_pred = np.multiply(source_pred,abs(source_pred)) # keep the negatives 
    # Step 4b:
    peaks, _ = scipy.signal.find_peaks(np.squeeze(source_pred), distance = np.round(fsamp*0.02)+1) # peaks variable holds the indices of all peaks
    source_pred /=  np.mean(maxk(source_pred[peaks], 10))
    
    if len(peaks) > 1:

        kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(source_pred[peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
        spikes_ind = np.argmax(kmeans.cluster_centers_)
        spikes = peaks[np.where(kmeans.labels_ == spikes_ind)]
        # remove outliers from the spikes cluster with a std-based threshold
        spikes = spikes[source_pred[spikes] <= np.mean(source_pred[spikes]) + 3*np.std(source_pred[spikes])]
    else:
        spikes = peaks

    return source_pred, spikes

def min_cov_isi(w_n,B,Z,fsamp,cov_n,spikes_n): 
    
    cov_n_1 = cov_n + 0.1
    
    while cov_n < cov_n_1:

        cov_n_1 = cov_n.copy()
        spikes_n_1 = spikes_n.copy()
        # w_n = np.expand_dims(w_n,axis=1)
        w_n_1 = w_n.copy()
        _ , spikes_n = get_spikes(w_n,Z,fsamp)
        # determine the interspike interval
        ISI = np.diff(spikes_n/fsamp)
        # determine the coefficient of variation
        cov_n = np.std(ISI)/np.mean(ISI)
        # update the sepearation vector by summing all the spikes
        w_n = np.sum(Z[:,spikes_n],axis=1) # summing the spiking across time, leaving an array that is channels x 1 
       

    # if you meet the CoV minimisation condition, but with a single-spike-long train, use the updated
    # separation vector and recompute the spikes
    if len(spikes_n_1) < 2:
        _ , spikes_n_1 = get_spikes(w_n,Z,fsamp)

    return w_n_1, spikes_n_1, cov_n_1


################################ VALIDATION TOOLS ########################################

def maxk(signal, k): 
    return np.partition(signal, -k, axis=-1)[..., -k:]

def get_silohuette(w_n,Z,fsamp):

    # Step 4a: 
    source_pred = np.dot(np.transpose(w_n), Z).real # element-wise square of the input to estimate the ith source
    source_pred = np.multiply(source_pred,abs(source_pred)) # keep the negatives 
    
    # Step 4b:
    peaks, _ = scipy.signal.find_peaks(np.squeeze(source_pred), distance = np.round(fsamp*0.002)+1 ) # this is approx a value of 20, which is in time approx 10ms
    source_pred /=  np.mean(maxk(source_pred[peaks], 10))
    if len(peaks) > 1:

        kmeans = KMeans(n_clusters = 2,init = 'k-means++',n_init = 1).fit(source_pred[peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
        # indices of the spike and noise clusters (the spike cluster should have a larger value)
        spikes_ind = np.argmax(kmeans.cluster_centers_)
        noise_ind = np.argmin(kmeans.cluster_centers_)
        # get the points that correspond to each of these clusters
        spikes = peaks[np.where(kmeans.labels_ == spikes_ind)]
        noise = peaks[np.where(kmeans.labels_ == noise_ind)]
        # calculate the centroids
        spikes_centroid = kmeans.cluster_centers_[spikes_ind]
        noise_centroid = kmeans.cluster_centers_[noise_ind]
        # difference between the within-cluster sums of point-to-centroid distances 
        intra_sums = (((source_pred[spikes]- spikes_centroid)**2).sum()) 
        # difference between the between-cluster sums of point-to-centroid distances
        inter_sums = (((source_pred[spikes] - noise_centroid)**2).sum())
        sil = (inter_sums - intra_sums) / max(intra_sums, inter_sums)  

    else:
        sil = 0

    return source_pred, spikes, sil


def peel_off(Z,spikes,fsamp,peel_off_win):

    """
    Removes the contribution of motor unit action potentials (MUAPs) from an EMG signal by 
    peeling off the source vector related to a specific motor unit's firing.

    Parameters
    ----------
    Z : numpy.ndarray
        The EMG signal matrix with dimensions (channels x timepoints). Each row corresponds 
        to a different EMG channel, and each column corresponds to a time point.
        
    spikes : array-like
        The array of spike times (indices) where the motor unit fires. This array marks the 
        locations in the time series where the motor unit activity occurs.
        
    fsamp : int
        The sampling frequency (in Hz) of the EMG signal. Used to convert the `peel_off_win`
        from seconds to samples.
        
    peel_off_win : float
        The window length (in seconds) around each spike from which to extract the motor unit 
        action potential (MUAP) waveform. The total window used will be `2 * peel_off_win` 
        around the spike.
    
    Returns
    -------
    Z : numpy.ndarray
        The modified EMG signal matrix after removing the contribution of the motor unit 
        associated with the spikes. This is achieved by subtracting the reconstructed EMG 
        signal (generated from the motor unit's firing and waveform) from the original signal.
    
    Notes
    -----
    - This function operates by first extracting the average MUAP waveform centered around 
      each spike in the provided `spikes` array.
    - The average MUAP waveform is then convolved with the firing pattern of the motor unit to 
      reconstruct the contribution of this motor unit to the signal.
    - Finally, this contribution is subtracted from the original EMG signal `Z` to remove the 
      motor unit’s influence, allowing further processing to focus on the remaining components.
    - The `cutMUAP` function (called within) extracts the MUAP waveform for the 
      provided spike times and window length.
    
    Example
    -------
    Z_cleaned = peel_off(Z, spikes, fsamp=1000, peel_off_win=0.01)
    """

    windowl = round(peel_off_win*fsamp)
    waveform = np.zeros([windowl*2+1])
    firings = np.zeros([np.shape(Z)[1]])
    firings[spikes] = 1 # make the firings binary
    EMGtemp = np.empty(Z.shape) # intialise a temporary EMG signal

    for i in range(np.shape(Z)[0]): # iterating through the (extended) channels
        temp = cutMUAP(spikes,windowl,Z[i,:])
        waveform = np.mean(temp,axis=0)
        EMGtemp[i,:] =  scipy.signal.convolve(firings, waveform, mode = 'same',method='auto')

    Z -= EMGtemp; # removing the EMG representation of the source spearation vector from the signal, avoid picking up replicate content in future iterations
    return Z
############################## POST PROCESSING #####################################################

def xcorr(x, y, max_lag=None): 

    # asssume no lag limitation unless specificied
    fft_size = 2 ** (len(x) + len(y) - 1).bit_length()
    xcorrvalues = np.fft.ifft(np.fft.fft(x, fft_size) * np.conj(np.fft.fft(y, fft_size))) 
    xcorrvalues = np.fft.fftshift(xcorrvalues)
    
    if max_lag is not None:
        max_lag = min(max_lag, len(xcorrvalues) // 2)
        lags = np.arange(-max_lag, max_lag + 1)
        xcorrvalues = xcorrvalues[len(xcorrvalues)//2 - max_lag : len(xcorrvalues)//2 + max_lag + 1]
    else:
        lags = np.arange(-(len(xcorrvalues) // 2), len(xcorrvalues) // 2 + 1)
    
    xcorrvalues /= np.max(xcorrvalues)  
    
    return xcorrvalues, lags

def gausswin(M, alpha=2.5):
    
    """ Python equivalent of the in-built gausswin function MATLAB (since there is no open-source Python equivalent) """
    
    n = np.arange(-(M-1) / 2, (M-1) / 2 + 1,dtype=np.float64)
    w = np.exp((-1/2) * (alpha * n / ((M-1) / 2)) ** 2)
    return w

def cutMUAP(MUPulses, length, Y):

    """
    Extracts consecutive motor unit action potentials (MUAPs) from a given signal, `Y`, based on 
    specified trigger positions (`MUPulses`). Each extracted MUAP is aligned and stored row-wise 
    in the output matrix.

    Parameters
    ----------
    MUPulses : array-like
        Array of trigger positions (in samples) where motor unit firings occur. These are the centers 
        of the rectangular windows used to extract MUAPs.
        
    length : int
        Radius of the rectangular window around each trigger position. The total window length for 
        extraction is `2 * length + 1`.
        
    Y : array-like
        The single-channel EMG signal (a 1D array) from which the MUAPs are extracted.

    Returns
    -------
    MUAPs : numpy.ndarray
        A 2D array where each row contains the extracted MUAP aligned with its trigger position in 
        the original signal. The extracted MUAPs have a length of `2 * length + 1` samples.

    Notes
    -----
    - This function applies a Gaussian window to taper the edges of the extracted MUAPs to minimize 
      artifacts near the edges.
    - If the extraction window exceeds the bounds of the signal `Y`, zeros are padded to ensure the 
      MUAP remains the correct length.
    - The Gaussian window is generated using `gausswin` (a direct replacement for MATLAB’s `gausswin`).
    """

 
    while len(MUPulses) > 0 and MUPulses[-1] + 2 * length > len(Y):
        MUPulses = MUPulses[:-1]

    c = len(MUPulses)
    edge_len = round(length / 2)
    tmp = gausswin(2 * edge_len) # gives the same output as the in-built gausswin function in MATLAB
    # create the filtering window 
    win = np.ones(2 * length + 1)
    win[:edge_len] = tmp[:edge_len]
    win[-edge_len:] = tmp[edge_len:]
    MUAPs = np.empty((c, 1 + 2 * length))
    for k in range(c):
        start = max(MUPulses[k] - length, 1) - (MUPulses[k] - length)
        end = MUPulses[k] + length- min(MUPulses[k] + length, len(Y))
        MUAPs[k, :] = win * np.concatenate((np.zeros(start), Y[max(MUPulses[k] - length, 1):min(MUPulses[k] + length, len(Y))+1], np.zeros(end)))

    return MUAPs


# FOR POST PROCESSING WHEN COMBINING WITH BIOFEEDBACK
def get_pulse_trains(data, rejected_channels, mu_filters, chans_per_grid, fsamp,g):
     
    # channel rejection again, but on the PRE-FILTERED data
    # OR: if filtering was not used in the pre processing, the batched data could be used (?)
    data_slice = data[chans_per_grid[g]*(g):(g+1)* chans_per_grid[g],:] # will need to be generalised
    rejected_channels_slice = rejected_channels[g] == 1
    cleaned_data = np.delete(data_slice, rejected_channels_slice, 0)

    # get the first estimate of pulse trains using the previously derived mu filters, applied to the emg data
    ext_factor = int(np.round(1000/np.shape(cleaned_data)[0]))
    extended_data = np.zeros([1, np.shape(cleaned_data)[0]*(ext_factor), np.shape(cleaned_data)[1] + ext_factor -1]) # no differential mode used here (?)
    extended_data =  extend_emg(extended_data,cleaned_data,ext_factor)

    # get the real and inverted versions
    sq_extended_data = (extended_data @ extended_data.T)/np.shape(extended_data)[1]
    inv_extended_data = np.linalg.pinv(extended_data)
    
    # initialisations for extracting pulse trains in clustering
    mu_count =  np.shape(mu_filters)[1]
    pulse_trains = np.zeros([mu_count, np.shape(data)[1]]) 
    discharge_times = [None] * mu_count # do not know size yet, so can only predefine as a list
    
    for mu in range(mu_count):

        pulse_temp = (mu_filters[:,mu].T @ inv_extended_data) @ extended_data # what does this do?
        # step 4a 
        pulse_trains[mu,:] = pulse_temp[:np.shape(data)[1]]
        # source_pred = np.dot(np.transpose(w_n), Z).real # element-wise square of the input to estimate the ith source
        pulse_trains[mu,:] = np.multiply(pulse_trains[mu,:],abs(pulse_trains[mu,:])) # keep the negatives 
        # Step 4b:
        peaks, _ = scipy.signal.find_peaks(np.squeeze(pulse_trains[mu,:]), distance = np.round(fsamp*0.02)+1) # peaks variable holds the indices of all peaks
        pulse_trains[mu,:] /=  np.mean(maxk(pulse_trains[mu,peaks], 10))
    
        if len(peaks) > 1:

            kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(pulse_trains[mu,peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
            spikes_ind = np.argmax(kmeans.cluster_centers_)
            spikes = peaks[np.where(kmeans.labels_ == spikes_ind)]
            # remove outliers from the spikes cluster with a std-based threshold
            discharge_times[mu] = spikes[pulse_trains[mu,spikes] <= np.mean(pulse_trains[mu,spikes]) + 3*np.std(pulse_trains[mu,spikes])]
        else:
            discharge_times[mu] = peaks

    return pulse_trains, discharge_times, ext_factor

##########################################################################################################
############################# ADDITIONAL ONLINE PROCESSING FUNCTIONS #####################################
##########################################################################################################

def get_MU_filters(signal, signal_mask, discharge_times):
    """
    Takes raw data, removes noisy channels, and calculates MU filters for online decomposition,
    based on the best possible discharge times, identified in a prior offline decomposition.

    Parameters:
        - signal: raw data, no. channels x time
        - signal_mask: binary array, where 1 highlights a noisy channel, 0 highlights a noise-stable channel
        - discharge_times: the discharge times for only one electrode

    Returns:
        - mu_filters: calculated as the cross-correlation between the extended, demeaned observations and
          the current estimated spike trains
    """

    print("Refining MU pulse trains...")

    # Channel rejection on the pre-filtered data
    rejected_channels = signal_mask == 1
    cleaned_data = np.delete(signal, np.where(rejected_channels), axis=0)
    
    # Calculate the extension factor
    ext_factor = int(np.round(1000 / cleaned_data.shape[0]))

    # Initialize the extended data array
    extended_data = np.zeros((cleaned_data.shape[0] * ext_factor, cleaned_data.shape[1] + ext_factor - 1))
    
    # Extend and demean the EMG data
    extended_data = extend_emg(extended_data, cleaned_data, ext_factor)
    extended_data = scipy.signal.detrend(extended_data, axis=-1, type='constant', bp=0)

    # Initialize MU filters array
    num_units = np.shape(discharge_times)[0]
    mu_filters = np.zeros((extended_data.shape[0], num_units))

    # Calculate MU filters
    for mu in range(num_units):
        times = discharge_times[mu]
        if isinstance(times, np.ndarray):
            times = times.astype(int)  # Ensure discharge_times are integers
        else:
            times = np.array(times, dtype=int)  # Convert to numpy array and ensure integers
        mu_filters[:, mu] = np.sum(extended_data[:, times], axis=1)

    return mu_filters


def get_online_parameters(signal, signal_mask, mu_filters, fsamp):

    """
        Uses the most recent MU filters to calculate the spike-noise clustering space, ready for 
        online decomposition

        Parameters:

            - data: raw data, no. channels x time
            - signal_mask: binary array, where 1 highlights a noisy channel, 0 highlights a noise-stable channel
            - mu_filters: (previously) calculated as the cross-corrleation between the extended, demeaned observations and
            the current estimated spike trains
            - fsamp: sampling frequency of data
            - g: electrode index

        Returns:

            - extension factor
            - inv_extended_data: inverted matrix of extended data
            - norm: normalised pulse trains (using a subgroup of maxima)
            - centroids: spike and noise centroid coordinates in knn space
    
    """

    # channel rejection again, but on the PRE-FILTERED data
    # OR: if filtering was not used in the pre processing, the batched data could be used (?)
    # channel rejection again, but on the PRE-FILTERED data
    rejected_channels = signal_mask == 1
    cleaned_data = np.delete(signal, rejected_channels, 0)

    # get the first estimate of pulse trains using the previously derived mu filters, applied to the emg data
    ext_factor = int(np.round(1000/np.shape(cleaned_data)[0]))
    extended_data = np.zeros([np.shape(cleaned_data)[0]*(ext_factor), np.shape(cleaned_data)[1] + ext_factor - 1])
    extended_data =  extend_emg(extended_data,cleaned_data,ext_factor)
    print(np.shape(extended_data))
    # demean
    extended_data = scipy.signal.detrend(extended_data, axis=- 1, type='constant', bp=0)
    # get squared version
    sq_extended_data = (extended_data @ extended_data.T)/ np.shape(extended_data)[1]
    # get inverted version
    inv_extended_data = np.linalg.pinv(sq_extended_data)

    # initialisations for extracting pulse trains and centroids in clustering
    mu_count =  np.shape(mu_filters)[1]
    pulse_trains = np.zeros([mu_count, np.shape(signal)[1]]) # nmus x time
    discharge_times = [None] * mu_count # do not know size yet, so can only predefine as a list
    norm = np.zeros(mu_count)
    centroids = np.zeros([mu_count,2])# first column is the spike centroids, second column is the noise centroids
    nMU = 0
    spikes2check = []
    
    for mu in range(mu_count):

        IPTtemp = (mu_filters[:,mu].T @ inv_extended_data) @ extended_data # what does this do? decorrelate coarsely
        # step 4a 
        pulse_trains[mu,:] = IPTtemp[:np.shape(signal)[1]]
        # source_pred = np.dot(np.transpose(w_n), Z).real # element-wise square of the input to estimate the ith source
        pulse_trains[mu,:] = np.multiply(pulse_trains[mu,:],abs(pulse_trains[mu,:])) # keep the negatives 
        # Step 4b:
        peaks, _ = scipy.signal.find_peaks(np.squeeze(pulse_trains[mu,:]),distance = np.round(fsamp*0.02)+1) # peaks variable holds the indices of all peaks
    
        if len(peaks) > 1:

            kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(pulse_trains[mu,peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
            # need both spikes and noise to determine cluster centres for the online decomposition
            # spikes
            spikes_ind = np.argmax(kmeans.cluster_centers_)
            spikes = peaks[np.where(kmeans.labels_ == spikes_ind)]
            spikes = spikes[pulse_trains[mu,spikes] <= np.mean(pulse_trains[mu,spikes]) + 3*np.std(pulse_trains[mu,spikes])]
            # noise
            noise_ind = np.argmin(kmeans.cluster_centers_)
            noise = peaks[np.where(kmeans.labels_ == noise_ind)]
            norm[mu] = np.mean(maxk(pulse_trains[mu,peaks], 10))
            pulse_trains[mu,:] /=  norm[mu]
            centroids[mu,0] = KMeans(n_clusters = 1, init = 'k-means++',n_init = 1).fit(pulse_trains[mu,spikes].reshape(-1,1)).cluster_centers_
            centroids[mu,1] = KMeans(n_clusters = 1, init = 'k-means++',n_init = 1).fit(pulse_trains[mu,noise].reshape(-1,1)).cluster_centers_
            nMU += 1
            spikes2check.append(spikes)

    return ext_factor, inv_extended_data, norm, centroids, spikes2check




def get_spikes_online(data, ext_factor, ext_addon, MUfilters, norm, noise_centroids, spike_centroids, nsamp, fsamp):


    """
    Uses the most recent MU filters to calculate the discharge times from the received data packet.
    Requires an adjustment in the extension procedure, to avoid skewing the calculation of the pulse trains.
    Therefore, the 0's of the extension matrix (across columns) are filled with prior data.

    Parameters:

        - data: raw data, no. channels x time
        - signal_mask: binary array, where 1 highlights a noisy channel, 0 highlights a noise-stable channel
        - mu_filters: (previously) calculated as the cross-corrleation between the extended, demeaned observations and
        the current estimated spike trains
        - fsamp: sampling frequency of data
        - g: electrode index

    Returns:

        - extension factor
        - inv_extended_data: inverted matrix of extended data
        - norm: normalised pulse trains (using a subgroup of maxima)
        - centroids: spike and noise centroid coordinates in knn space


    function [PulseT, Distime, esample2] = getspikesonline(EMGtmp, extensionfactor, esample2, MUfilt, norm, noise_centroids, spike_centroids, nsamp, fsamp)
    esample1 = demean(extend(EMGtmp, extensionfactor));
    esample = esample1(:,1:nsamp);
    esample(:,1:extensionfactor-1) = esample(:,1:extensionfactor-1) + esample2;
    esample2 = esample1(:,nsamp+1:end);

    PulseT = ((MUfilt * esample) .* abs(MUfilt * esample)) ./ norm';
    [spikes1, ~] = islocalmax(PulseT', 1, 'MinSeparation', round(fsamp*0.02));
    Distime = (abs(PulseT' .* spikes1 - noise_centroids) > abs(PulseT' .* spikes1 - spike_centroids));
    
    """
    # data packet extension
    ext_noaddon = np.zeros([np.shape(data)[0]*ext_factor, np.shape(data)[1] + ext_factor -1])
    ext_noaddon = extend_emg(ext_noaddon,data,ext_factor)
    ext_noaddon = scipy.signal.detrend(ext_noaddon, axis=- 1, type='constant', bp=0)
    ext_obvs = ext_noaddon[:,:nsamp] # only keep up to the size of the packet
    ext_obvs[:, 0: ext_factor-1] = ext_obvs[:, 0: ext_factor-1] + ext_addon # filler for the zeros, byproduct of extension

       

    print('extension works')
   
    # update the add on for the next interation
    ext_addon = ext_noaddon[:, nsamp:]
    multipulse = np.dot(MUfilters, ext_obvs)
    PulseT = (multipulse * np.abs(multipulse)) / norm[:, np.newaxis]
    Distime = np.zeros([MUfilters.shape[0], data.shape[1]])
    print('set up of array works')

    spikes_boolean = np.zeros_like(PulseT, dtype=bool)

    for i_mu in range(MUfilters.shape[0]):

        peaks, _ = scipy.signal.find_peaks(PulseT[i_mu, :], distance =np.round(fsamp*0.02)+1)
        spikes_boolean[i_mu,peaks] = True
    
    
    PulseT[~spikes_boolean] = 0
    Distime = np.abs(PulseT - noise_centroids) > np.abs(PulseT - spike_centroids)
    print('Found spikes')
    
    return PulseT, Distime.T, ext_addon


##################################################################################################################


# FOR POST PROCESSING DURING OFFLINE PROCEDURES
#### ONLY FILE UPDATED FROM THE ORIGINAL SOURCE CODE
def batch_process_filters(whit_sig, mu_filters,plateau,extender,diff,orig_sig_size,fsamp):

    """ dis_time: the distribution of spiking times for every identified motor unit, but at this point we don't check to see
    whether any of these MUs are repeats"""
    # NOTE!!! the mu_filters variable is a list NOT an np.array, given thresholding cannot guaranteed consistent array dimensions

    # Pulse trains has shape no_mus x original signal duration
    # dewhitening matrix has shape no_windows x exten chans x exten chans
    # mu filters has size no_windows x exten chans x (maximum of) no iterations  --> less if iterations failed to reach SIL threshold

    mu_count = mu_filters.shape[1]
    pulse_trains = np.zeros([mu_count, orig_sig_size]) 
    discharge_times = [None] * mu_count # do not know size yet, so can only predefine as a list

    for mu_candidate in range(np.shape(mu_filters)[1]):
            
        pulse_trains[mu_candidate, int(plateau[0]):int(plateau[1])+ extender - diff] = np.transpose(mu_filters[:,mu_candidate]) @ whit_sig
                
        # Step 4a: 
        pulse_trains[mu_candidate,:] = np.multiply(pulse_trains[mu_candidate,:],abs(pulse_trains[mu_candidate,:])) # keep the negatives 
        # Step 4b:
        peaks, _ = scipy.signal.find_peaks(np.squeeze(pulse_trains[mu_candidate,:]), distance = np.round(fsamp*0.005)+1) # peaks variable holds the indices of all peaks
        pulse_trains[mu_candidate,:] /=  np.mean(maxk(pulse_trains[mu_candidate,:], 10))
        kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(pulse_trains[mu_candidate,peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
        spikes_ind = np.argmax(kmeans.cluster_centers_)
        discharge_times[mu_candidate] = peaks[np.where(kmeans.labels_ == spikes_ind)]
        print(f"Batch processing MU#{mu_candidate+1} out of {mu_count} MUs")

    
    return pulse_trains, discharge_times

def remove_duplicates(pulse_trains, discharge_times, discharge_times2, mu_filters, maxlag, jitter_val, tol, fsamp):

    """
    Removes duplicate motor unit (MU) activity by identifying and eliminating redundant or highly correlated
    motor units from the EMG data. It compares the spike trains of each MU and removes duplicates based 
    on correlation, jitter, and coefficient of variation (CoV).

    Parameters
    ----------
    pulse_trains : numpy.ndarray
        Binary matrix representing the pulse trains of motor units. Each row corresponds to a motor unit, 
        and each column corresponds to time points where spikes occur.
    
    discharge_times : list of arrays
        List containing the discharge times (in sample indices) for each motor unit.
    
    discharge_times2 : list of arrays
        Similar to `discharge_times`, but with an additional jitter margin for each motor unit's spikes. 
        This helps in identifying duplicates with temporal variance.
    
    mu_filters : numpy.ndarray
        A 3D matrix of motor unit filters or source separation vectors. The first two dimensions represent 
        time or space, and the third dimension corresponds to different motor units.
    
    maxlag : int
        The maximum lag (in samples) allowed when computing cross-correlations between motor unit spike trains.
    
    jitter_val : float
        The amount of jitter (in seconds) allowed when comparing discharge times to account for temporal 
        variability between motor units.
    
    tol : float
        The tolerance value for identifying duplicates. If the ratio of common discharges between two motor units 
        exceeds this value, they are considered duplicates.
    
    fsamp : int
        The sampling frequency of the EMG signal, used to convert the jitter value from seconds to samples.
    
    Returns
    -------
    discharge_times_new : list of arrays
        Updated list of discharge times for the surviving motor units, after duplicates have been removed.
    
    pulse_trains_new : list of arrays
        Updated list of pulse trains corresponding to the surviving motor units.
    
    mu_filters : numpy.ndarray
        Updated set of motor unit filters, with duplicates removed.
    
    Notes
    -----
    - The function first generates binary spike trains from the discharge times and applies jitter to account 
      for small variations in firing times.
    - It then calculates the cross-correlation between spike trains to identify highly correlated motor units 
      (indicating potential duplicates).
    - Common discharges are compared, and a tolerance threshold is applied to filter out duplicates. The motor 
      unit with the lowest coefficient of variation (CoV) of inter-spike intervals (ISI) is retained as the 
      "survivor".
    - The spike trains, pulse trains, and motor unit filters are updated iteratively to remove duplicates while 
      keeping the most representative motor unit.
    """
  
    jitter_thr = int(np.round(jitter_val * fsamp))
    spike_trains = np.zeros([np.shape(pulse_trains)[0], np.shape(pulse_trains)[1]])
    
    discharge_jits = []
    discharge_times_new = []  # do not know yet how many non-duplicates MUs there will be
    pulse_trains_new = []  # do not know yet how many non-duplicates MUs there will be

    for i in range(np.shape(pulse_trains)[0]):
        spike_trains[i, discharge_times[i]] = 1
        discharge_jits.append([])  # append an empty list to be extended with jitters
        # adding jitter
        for j in range(jitter_thr):
            discharge_jits[i].extend(discharge_times2[i] - j)
            discharge_jits[i].extend(discharge_times2[i] + j)  # adding varying extents of jitter, based on computed jitter threshold

        discharge_jits[i].extend(discharge_times2[i])

    i = 1
    # With the binary trains generated above, you can readily identify duplicate MUs
    while discharge_jits:
        discharge_temp = []
        num_mu_candidates = len(discharge_jits)
        
        for mu_candidate in range(num_mu_candidates):
            corr, lags = xcorr(spike_trains[0, :], spike_trains[mu_candidate, :], int(maxlag))
            ind_max = np.argmax(corr)
            corr_max = np.real(corr[ind_max])

            if corr_max > 0.2:
                discharge_temp.append(discharge_jits[mu_candidate] + lags[ind_max])
            else:
                discharge_temp.append(discharge_jits[mu_candidate])

        comdis = np.zeros(np.shape(pulse_trains)[0])
        
        for j in range(1, np.shape(pulse_trains)[0]):  # skip the first since it is used for the baseline comparison
            com = np.intersect1d(discharge_jits[0], discharge_temp[j])

            if len(com) > 1:
                # Calculate the differences between consecutive elements
                diff_com = np.diff(com)
                # Apply the jitter threshold to determine where the differences are within acceptable range
                boolean_index = np.insert(diff_com > jitter_thr, 0, False)

                # Ensure boolean_index matches the length of com
                if len(boolean_index) != len(com):
                    raise ValueError(f"Length mismatch: boolean_index length is {len(boolean_index)}, but com length is {len(com)}")

                # Filter com based on the jitter threshold
                com_filtered = com[boolean_index]
            else:
                # If com has 0 or 1 elements, com_filtered is just com
                com_filtered = com

            # Use the filtered common discharges to calculate the ratio
            comdis[j] = len(com_filtered) / max(len(discharge_times[0]), len(discharge_times[j]))
            com = None

        duplicates = np.where(comdis >= tol)[0]
        duplicates = np.insert(duplicates, 0, 0)  # insert
        CoV = np.zeros(len(duplicates))

        for j in range(len(duplicates)):
            ISI = np.diff(discharge_times[duplicates[j]])
            CoV[j] = np.std(ISI) / np.mean(ISI)

        survivor = np.argmin(CoV)  # the surviving MU has the lowest CoV

        # delete all duplicates, but save the surviving MU
        discharge_times_new.append(discharge_times[duplicates[survivor]])
        pulse_trains_new.append(pulse_trains[duplicates[survivor]])

        # update firings and discharge times
        for j in range(len(duplicates)):
            discharge_times[duplicates[-(j + 1)]] = []
            discharge_times2[duplicates[-(j + 1)]] = []
            discharge_jits[duplicates[-(j + 1)]] = []

        # if it is not empty, assign it back to the list, otherwise remove the empty element i.e. only keep the non duplicate MUs that were not emptied in previous loop
        discharge_times = [mu for mu in discharge_times if len(mu) > 0]
        discharge_times2 = [mu for mu in discharge_times2 if len(mu) > 0]
        discharge_jits = [mu for mu in discharge_jits if len(mu) > 0]


        print('Duplicates')
        print(duplicates)
        # Clean the spike and pulse train arrays based on identified duplicates
        # all duplicates removed so we identify different duplicate groups on the next iteration of the while loop
        spike_trains = np.delete(spike_trains, duplicates, axis=0)
        pulse_trains = np.delete(pulse_trains, duplicates, axis=0)

        # Remove duplicates excluding duplicates
        mu_filters = np.delete(mu_filters, [duplicates[mdx] for mdx in range(len(duplicates)) if mdx != survivor], axis=1)

        i += 1

    print('Duplicates removed')
    print('Final shape of mu filters')
    print(np.shape(mu_filters))
    return discharge_times_new, pulse_trains_new, mu_filters


def remove_duplicates_between_arrays(pulse_trains, discharge_times, muscle, maxlag, jitter_val, tol, fsamp):

    jitter_thr = int(np.round(jitter_val*fsamp))
    spike_trains = np.zeros([np.shape(pulse_trains)[0],np.shape(pulse_trains)[1]])
    
    discharge_jits = []
    discharge_times_new = [] # do not know yet how many non-duplicates MUs there will be
    pulse_trains_new = [] # do not know yet how many non-duplicates MUs there will be
    muscle_new = []

    # generating binary spike trains for each MU extracted so far
    for i in range(np.shape(pulse_trains)[0]):

        spike_trains[i,discharge_times[i]] = 1
        discharge_jits.append([]) # append an empty list to be extended with jitters
        # adding jitter
        for j in range(jitter_thr):
            discharge_jits[i].extend(discharge_times[i]-j)
            discharge_jits[i].extend(discharge_times[i]+j) # adding varying extents of jitter, based on computed jitter threshold

        discharge_jits[i].extend(discharge_times[i])

    #mu_count = np.shape(discharge_times)[0] # the total number of MUs extracted so far
    mu_count = len(discharge_times)

    i = 1
    # With the binary trains generated above, you can readily identify duplicate MUs
    while discharge_jits:
        
        discharge_temp = []
        for mu_candidate in range(np.shape(discharge_jits)[0]):

            # in matlab: [c, lags] = xcorr(firings(1,:), firings(j,:), maxlag*2,'normalized')
            # calculating the cross correlation between the firings of two cnadidate MUs, within a limited range of maxlag*2
            # then, normalise the resulting correlation values between 0 and 1

            corr, lags = xcorr(spike_trains[0,:], spike_trains[mu_candidate,:],int(maxlag))
            ind_max = np.argmax(corr)
            corr_max = np.real(corr[ind_max])

            if corr_max > 0.2:
                discharge_temp.append(discharge_jits[mu_candidate] + lags[ind_max])
            else:
                discharge_temp.append(discharge_jits[mu_candidate])


        # discharge_temp is the lag-shifted version of discharge_jits if the correlation is sufficiently large
        # so if they are quite misaligned in time, we shift them to be aligned, ready to maximise the count of the common discharges
        # otherwise, we assume their temporal alignment is okay enough to just go ahead and count common discharges below
        
        # Now, we count the common discharge times
        comdis = np.zeros(np.shape(pulse_trains)[0])
        
        for j in range(1, np.shape(pulse_trains)[0]): # skip the first since it is used for the baseline comparison   
            com = np.intersect1d(discharge_jits[0], discharge_temp[j])
            com = com[np.insert(np.diff(com) != 1, 0, False)] # shouldn't this be based on the jitter threshold
            comdis[j] = len(com) / max(len(discharge_times[0]), len(discharge_times[j]))
            com = None

        # now left with an array of common discharges
        # use this establish the duplicate MUs, and keep only the MU that has the most stable, regular firing behaviour i.e. low ISI
        
        duplicates = np.where(comdis >= tol)[0]
        duplicates = np.insert(duplicates, 0, 0) # insert 
        CoV = np.zeros(len(duplicates))
        
        for j in range(len(duplicates)):
            ISI = np.diff(discharge_times[duplicates[j]])
            CoV[j] = np.std(ISI) / np.mean(ISI)

        survivor = np.argmin(CoV) # the surviving MU has the lowest CoV

        # delete all duplicates, but save the surviving MU
        discharge_times_new.append(discharge_times[duplicates[survivor]])
        pulse_trains_new.append(pulse_trains[duplicates[survivor]])
        muscle_new.append(muscle[duplicates[survivor]])

        # update firings and discharge times

        for j in range(len(duplicates)):

            discharge_times[duplicates[-(j+1)]] = []
            discharge_jits[duplicates[-(j+1)]] = []

        # if it is not empty, assign it back to the list, otherwise remove the empty element i.e. only keep the non duplicate MUs that were not emptied in previous loop
        discharge_times = [mu for mu in discharge_times if len(mu) > 0] 
        discharge_jits = [mu for mu in discharge_jits if len(mu) > 0]

        # Clean the spike and pulse train arrays based on identificed duplicates
        # all duplicates removed so we identify different duplicate groups on the next iteration of the while loop
        spike_trains = np.delete(spike_trains, duplicates, axis=0)
        pulse_trains = np.delete(pulse_trains, duplicates, axis=0)
        muscle = np.delete(muscle, duplicates, axis=0)

        i += 1
    muscle_new = np.array(muscle_new)
    pulse_trains_new = np.array(pulse_trains_new)
    print('Duplicates across arrays removed')
    return discharge_times_new, pulse_trains_new, muscle_new
  

def remove_outliers(pulse_trains, discharge_times, fsamp, threshold=0.4, max_its=30):

    """
    Removes outlier motor unit discharges by identifying and eliminating abnormal interspike intervals (ISIs) 
    that cause excessive variability in discharge rates.

    Parameters
    ----------
    pulse_trains : numpy.ndarray
        A binary matrix representing pulse trains of motor units. Each row corresponds to a motor unit, 
        and each column corresponds to time points where spikes occur.
    
    discharge_times : list of arrays
        A list where each element is an array of discharge times (in sample indices) for a given motor unit. 
        These represent the times at which each motor unit fires.
    
    fsamp : int
        The sampling frequency of the EMG signal, used to convert discharge times into actual time intervals 
        for calculating discharge rates.
    
    threshold : float, optional, default=0.4
        The coefficient of variation (CoV) threshold for discharge rates. If the CoV of discharge rates exceeds 
        this value, the function will iteratively remove outliers until the CoV is below the threshold or the 
        maximum number of iterations is reached.
    
    max_its : int, optional, default=30
        The maximum number of iterations allowed for removing outliers. The process will stop either when the 
        CoV of discharge rates falls below the threshold or when this iteration limit is reached.

    Returns
    -------
    discharge_times : list of arrays
        The updated list of discharge times for each motor unit, with outliers removed. Each element in the list 
        corresponds to a motor unit and contains its cleaned discharge times.

    Notes
    -----
    - The function calculates discharge rates by taking the inverse of the interspike intervals (ISIs) for each 
      motor unit.
    - It then iteratively removes outliers by identifying ISIs that exceed a threshold based on the mean and 
      standard deviation of discharge rates.
    - Outliers are removed based on their discharge times, and the iteration continues until the CoV of discharge 
      rates is within the specified `threshold` or the maximum number of iterations (`max_its`) is reached.
    - The function ensures that for any identified outlier, the discharge time contributing most to the abnormal 
      rate is removed.
    """
    
    for mu in range(len(discharge_times)):
        discharge_rates = 1 / (np.diff(discharge_times[mu]) / fsamp)
        it = 1
        while (np.std(discharge_rates) / np.mean(discharge_rates)) > threshold and it < max_its:
            artifact_limit = np.mean(discharge_rates) + 2 * np.std(discharge_rates)
            artifact_inds = np.argwhere(discharge_rates > artifact_limit).flatten().tolist()

            if len(artifact_inds) > 0:
                if max(artifact_inds) < len(discharge_times[mu]) - 1:
                    artifact_inds = np.array(artifact_inds)
                    diff_artifact_comp = pulse_trains[mu][discharge_times[mu][artifact_inds]] < pulse_trains[mu][discharge_times[mu][artifact_inds + 1]]
                    less_or_more = np.argmax([diff_artifact_comp, ~diff_artifact_comp], axis=0)
                    discharge_times[mu] = np.delete(discharge_times[mu], artifact_inds + less_or_more)

            discharge_rates = 1 / (np.diff(discharge_times[mu]) / fsamp)
            it += 1

    return discharge_times

def refine_mus(signal, pulse_trains_n_1, discharge_times_n_1, fsamp, extension_factor=17):

    print("Refining MU pulse trains...")
    
    # nbextchan = 1000
    # extension_factor = round(nbextchan/np.shape(signal)[0])
    extend_obvs = np.zeros([np.shape(signal)[0]*(extension_factor), np.shape(signal)[1] + extension_factor -1 ])
    extend_obvs = extend_emg(extend_obvs,signal,extension_factor)
    re_obvs = np.matmul(extend_obvs, extend_obvs.T)/np.shape(extend_obvs)[1]
    invre_obvs = np.linalg.pinv(re_obvs)
    pulse_trains_n = np.zeros(np.shape(pulse_trains_n_1))
    discharge_times_n = [None] * len(pulse_trains_n_1)

    # recalculating the mu filters

    for mu in range(len(pulse_trains_n_1)):

        mu_filters = np.sum(extend_obvs[:,discharge_times_n_1[mu]],axis=1)
        IPTtmp = np.dot(np.dot(mu_filters.T,invre_obvs),extend_obvs)
        pulse_trains_n[mu,:] = IPTtmp[:np.shape(signal)[1]]

        #pulse_trains_n[mu,:] = pulse_trains_n[mu,:]/ np.max(pulse_trains_n[mu,:])
        pulse_trains_n[mu,:] = np.multiply( pulse_trains_n[mu,:],abs(pulse_trains_n[mu,:])) 
        peaks, _ = scipy.signal.find_peaks(np.squeeze(pulse_trains_n[mu,:]),distance = np.round(fsamp*0.02)+1)  
        pulse_trains_n[mu,:] /=  np.mean(maxk(pulse_trains_n[mu,peaks], 10))
        kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(pulse_trains_n[mu,peaks].reshape(-1,1)) 
        spikes_ind = np.argmax(kmeans.cluster_centers_)
        discharge_times_n[mu] = peaks[np.where(kmeans.labels_ == spikes_ind)] 
  
   
    print(f"Refined {len(pulse_trains_n_1)} MUs")

    return pulse_trains_n, discharge_times_n


# for mvc calculation
def find_peak_indices(signal, peak_value):
  
    changes = np.diff(signal)
    starts = np.where(changes == peak_value)[0] + 1
    ends = np.where(changes == -peak_value)[0] + 1
    peak_indices = [list(range(start, end)) for start, end in zip(starts, ends)]
    
    return peak_indices


#######################################################################################################################################################################################################
############################################### TOOLS FOR VALIDATION AND APPROXIMATION ################################################################################################################
#######################################################################################################################################################################################################

def parameterise_real_time(decomposed_output, data4biofeedback,nfirings = 2):

    tracker = 0
    # Reducing the dimensions of the mu filters based on the mus that have been removed as duplicates from the pulse trains
    matching_indices = []
    # iterate over the newer pulse trains
    for i, col1 in enumerate(decomposed_output["pt1"][0]):
        # compare with the older pulse trains
        for j, col2 in enumerate(decomposed_output["pt0"][0]):
            # check for match
            if np.allclose(col1, col2):  
                matching_indices.append(j)

    sorted_indices_to_keep = sorted(matching_indices, key=lambda idx: matching_indices.index(idx))
    mu_filters_reduced = decomposed_output["mu_filters"][:, sorted_indices_to_keep]
    ######################################################################################################

    # for each electrode available in the dictionary, import the EMG signal and the discharge times
    for i in range(int(decomposed_output["nelectrodes"])):

        # structure of nested lists is inhomogeneous, does not allow for easy indexing
        # therefore, need to change to an array
        data4biofeedback["discharge_times"][i] = np.array(data4biofeedback["discharge_times"][i].copy(), dtype=object)
        rates = []

        # get the filters and parameters for online decomposition
        # note that the discharge times is now an array of objects
        data4biofeedback['MU_filters'].append(get_MU_filters(decomposed_output['data'][tracker : data4biofeedback['chans_per_electrode'][i] + tracker ,: ], data4biofeedback['emg_mask'][i], data4biofeedback['discharge_times'][i]))
        data4biofeedback['MU_filters_reduced'] = []
        data4biofeedback['MU_filters_reduced'].append(np.copy(mu_filters_reduced))
        # get online parameters, for the ith electrode
        data4biofeedback["ext_factor"][i], data4biofeedback["inv_extend_obvs"][i], data4biofeedback["norms"][i], data4biofeedback["centroids"][i], proxy_spikes = get_online_parameters(decomposed_output["data"][tracker : data4biofeedback["chans_per_electrode"][i]+ tracker,:], data4biofeedback["emg_mask"][i], data4biofeedback["MU_filters"][i],data4biofeedback["fsamp"])

        # ensure parsing out relevant part of the data by using electrode-independent counter
        tracker += data4biofeedback["chans_per_electrode"][i]

        print("Analysing for MU recruitment thresholds ...")

        rel_rt = []
        rel_dert = []
        abs_rt = []
        abs_dert = []

        nmus, _ = np.shape(decomposed_output["pulse_trains"][i])

        for mu in range(nmus):

            # this is cleaning the discharge times array from any erroneous spiking at the beginning and ending of the array
            data4biofeedback["discharge_times"][i][mu] = [dt for dt in data4biofeedback["discharge_times"][0][mu] if data4biofeedback["fsamp"] <= dt <= 500000 - data4biofeedback["fsamp"]]
            mu_rec = data4biofeedback["discharge_times"][i][mu][0:nfirings]
            mu_derec = data4biofeedback["discharge_times"][i][mu][-nfirings:]
            rel_rt.append(np.mean(decomposed_output["target"][mu_rec]))
            rel_dert.append(np.mean(decomposed_output["target"][mu_derec]))

        sorted_indices = np.argsort(rel_rt)
        rel_rt = np.array(rel_rt)[sorted_indices]
        rel_dert = np.array(rel_dert)[sorted_indices]

        print(sorted_indices)

    ##### TO DO ##### add the rest of the things that need to be reordered #########
    
    data4biofeedback["pulse_trains"][i] = np.array(data4biofeedback["pulse_trains"][i])[sorted_indices]
    data4biofeedback['norms'][i] = np.array(data4biofeedback["norms"][i])[sorted_indices]
    data4biofeedback['centroids'][i] = np.array(data4biofeedback['centroids'][i][sorted_indices])
    data4biofeedback["discharge_times"][i] = [data4biofeedback["discharge_times"][i][j] for j in sorted_indices]
    data4biofeedback["MU_filters"][i] = data4biofeedback["MU_filters"][i][:, sorted_indices]
    proxy_spikes = [proxy_spikes[j] for j in sorted_indices]

    return data4biofeedback, proxy_spikes


def rate_of_agreement(discharges_ref, firings_ref, discharges_test, firings_test, maxlag, jitter_val, fsamp, duration):

    """
    Computing the rate of agreement (RoA) for each motor unit in the test set compared to 
    the reference set by introducing jitter in the discharge timings and aligning the signals using 
    cross-correlation. The RoA is calculated based on the common discharge times after realigning 
    the units and accounting for jitter. This method ensures that the comparison is insensitive to 
    small timing misalignments between the test and reference discharges.

    Parameters:
    -----------
    discharges_ref : list of np.ndarray
        A list where each element is an array of discharge times for each motor unit in the reference set.
    
    firings_ref : np.ndarray
        A 2D array where each row corresponds to the firing sequence (binary or continuous firing) of a motor unit in the reference set.
    
    discharges_test : list of np.ndarray
        A list where each element is an array of discharge times for each motor unit in the test set.
    
    firings_test : np.ndarray
        A 2D array where each row corresponds to the firing sequence (binary or continuous firing) of a motor unit in the test set.
    
    maxlag : int
        The maximum lag, in number of samples, to be considered when performing the cross-correlation for realigning motor unit discharges.
    
    jitter_val : float
        The jitter threshold in seconds. This determines how much timing variation is tolerated when simulating jittered discharges.
    
    fsamp : int
        The sampling frequency in Hz.
    
    duration : float
        The duration of the recording in number of samples.

    Returns:
    --------
    all_roas : list of float
        A list where each element corresponds to the maximum rate of agreement for each motor unit in the test set, calculated against all reference motor units.
    
    ind_roas : list of int
        A list of indices corresponding to the reference motor unit that yielded the maximum RoA for each motor unit in the test set.

    Notes:
    ------
    - The function first applies a jitter simulation to the test and reference motor units, extending the list of discharges to account for possible timing variation within the given jitter threshold.
    - Cross-correlation is used to realign the test and reference motor units, and the RoA is calculated based on the number of common discharge times.
    - The RoA is defined as the number of common discharges (true positives) divided by the sum of true positives, false positives, and false negatives.
    - This method accounts for shifts between the test and reference motor units, ensuring that the RoA calculation is robust to timing differences.
    """

    jitter_thr = int(np.round(jitter_val * fsamp))
    discharges_test_jits = []
    discharges_ref_jits = []
    all_roas = []
    ind_roas = []

    # jitter simulation for the test MUs
    for i in range(np.shape(firings_test)[0]): 

        discharges_test_jits.append([])
        for jit in range(jitter_thr + 1):  # so if the threshold is 3, you add the original, plus 1,2,3 jitters

            discharges_test_jits[i].extend(discharges_test[i] - jit)
            discharges_test_jits[i].extend(discharges_test[i] + jit)

   
    # jitter simulation for the reference MUs
    for j in range(np.shape(firings_ref)[0]):

        discharges_ref_jits.append([])
        for jit in range(jitter_thr + 1):

            discharges_ref_jits[j].extend(discharges_ref[j] - jit)
            discharges_ref_jits[j].extend(discharges_ref[j] + jit)


    for i in range(np.shape(firings_test)[0]):

        print('MU')
        print(i)
        
        roa = []

        for j in range(np.shape(firings_ref)[0]):

            # first, before a binary cross correlation between the reference and test MUs
            corr,lags =  xcorr(firings_test[i,:],firings_ref[j,:], int(maxlag*2))
            ind_max = np.argmax(corr)
            corr_max =  np.real(corr[ind_max])

            # realign the reference and test units to ensure that the rate of agreement etc. measures are accurate + insenstive to time lag
            if corr_max > 0.2:
                discharges_alignment = discharges_ref[j] + lags[ind_max]
            else:
                discharges_alignment = discharges_ref[j].copy()


            # find common discharge times
            com = np.intersect1d(discharges_test_jits[i], discharges_alignment)

            
            if len(com) > 1:
                # calculate the differences between consecutive elements
                diff_com = np.diff(com)
                # apply the jitter threshold to determine whether the differences are within an acceptable range
                boolean_index = np.insert(diff_com > jitter_thr, 0, False)
                # ensure boolean_index matches the length of com
                if len(boolean_index) != len(com):
                    raise ValueError(f"Length mismatch: boolean_index length is {len(boolean_index)}, but com length is {len(com)}")
                
                # filter com based on the jitter threshold
                com_filtered = com[boolean_index]

            else:

                com_filter = com
        
            # use the common discharges to calculate the RoA (and other useful metrics)
            true_positive = len(com)
            
            false_positive = len(discharges_test[i]) - true_positive
            false_negative = len(discharges_ref[j]) - true_positive
            roa.append(true_positive/(true_positive + false_positive + false_negative))

        
        print('Rate agreement for for MU {i}')
        print(max(roa))
        all_roas.append(max(roa))
        print('MU index')
        print(np.argmax(roa))
        ind_roas.append(np.argmax(roa))

    return all_roas, ind_roas



def getMUAPs(discharge_times,nmus,duration,data,chans2use,fsamp,target = None, thres = 0.8):

    """ Includes spike-triggered averaging 
        Chans2use ensures that you are only using the data from channels that were not rejected in a prior offline decomposition
    """

    firings_temp = np.zeros([nmus,duration])
    MUAP = []

    for i in range(nmus):

        firings_temp[i,discharge_times[i]] = 1
        MUAPacrosschans = []

        for j in chans2use:

            MUAPacrosschans.append(cutMUAP(discharge_times[i], int(fsamp*0.02),data[j,:]))
            
        MUAP.append(MUAPacrosschans)


    if target is not None:
        #parsing out the plateau region
        crossing_indices = np.where(target >= thres*max(target))[0]
        point1 = crossing_indices[0]  # First crossing point
        point2 = crossing_indices[-1]  # Last crossing point
        firings = np.zeros(np.shape(firings_temp))
        firings[:,point1:point2]= firings_temp[:,point1:point2]

    else:

        firings = firings_temp.copy()

    distimes = []
    MUAPa = []
  
    for i in range(nmus):

        MUAPacrosschans = []
        distimes.append(np.where(firings[i,:] == 1)[0])

        for j in range(len(chans2use)):

            MUAPacrosschans.append(np.mean(MUAP[i][j],axis=0).T)

        MUAPa.append(MUAPacrosschans)

    print('MUAPs determined')

    return MUAPa


def alignMUAPs(MUAP1,MUAP2):

    correlation = np.correlate(MUAP1, MUAP2, mode='full')
    shift = correlation.argmax() - (len(MUAP2) - 1)
    if shift > 0:
        MUAP1 = MUAP1[shift:]
        MUAP2 = MUAP2[:len(MUAP1)]
    elif shift < 0:
        MUAP2 = MUAP2[-shift:]
        MUAP1 = MUAP1[:len(MUAP2)]
    return MUAP1, MUAP2



def rsquaredMUAPs(MUAPs1, MUAPs2):

    """ Calculating the R-squared value for each MU AND for all the channels"""

    if  np.shape(MUAPs1) != np.shape(MUAPs2):
        raise ValueError(f"Shape mismatch: muap1 has shape {MUAPs1.shape} and muap2 has shape {MUAPs2.shape}. The arrays must have the same shape.")

    r_squared = np.zeros((np.shape(MUAPs1)[0],np.shape(MUAPs1)[1]))

    for i in range(np.shape(MUAPs1)[0]):  # iterate over motor units
        for j in range(np.shape(MUAPs1)[1]):
       
            # signal alignment, ensuring the r-squared value is insenstive to shifts
            aligned_muap1, aligned_muap2 = alignMUAPs(MUAPs1[i][j], MUAPs2[i][j])

            
            if len(aligned_muap1) > 1 and len(aligned_muap2) > 1:
                r_squared[i][j] = r2_score(aligned_muap1, aligned_muap2)
            else:
                r_squared[i][j] = np.nan  # when comparison is invalid

    return r_squared
    

def coarse_firing_rate(firing_matrix, sample_rate, bin_size):
    """
    Calculate the coarse firing rate using a specified bin size.
    
    Parameters:
    firing_matrix (ndarray): Binary matrix where each row corresponds to a neuron and each column to a time point.
    sample_rate (int): The sampling rate (samples per second).
    bin_size (int): The bin size in milliseconds.
    
    Returns:
    ndarray: A matrix where each row corresponds to the firing rate (spikes per second) for each bin for each neuron.
    """
    
    num_units = firing_matrix.shape[0]
    samples_per_bin = int(sample_rate * (bin_size / 1000.0))
    total_bins = firing_matrix.shape[1] // samples_per_bin
    firing_rate_per_bin = np.zeros((num_units, total_bins))
    
   
    for unit_idx in range(num_units):
        for bin_idx in range(total_bins):
            
            spike_count = np.sum(firing_matrix[unit_idx, bin_idx * samples_per_bin:(bin_idx + 1) * samples_per_bin])
            
            
            bin_duration_sec = bin_size / 1000.0  
            firing_rate_per_bin[unit_idx, bin_idx] = spike_count / bin_duration_sec
    
    return firing_rate_per_bin



