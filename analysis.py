import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.signal


def autocorr(x):
	# https://stackoverflow.com/a/47369584
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
    lag = np.abs(acorr).argmax() + 1
    r = acorr[lag-1]        
    if np.abs(r) > 0.5:
      print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
    else: 
      print('Appears to be not autocorrelated')
    return r, lag, acorr

#Finds the fundamendamental freq of a series using autocorrelation
def fund_freq(x, sample_rate):
    pass #TODO: Autocorrelation

x = np.load('short_reds.npy')
frate = 30

acorr = autocorr(x)[2]
np.abs(acorr)
# plt.plot(1/np.arange(0,len(acorr)) * 30, np.abs(acorr))
# plt.show()
norm = (x-x.mean())/x.std()

npoints = 300
w = np.abs(np.fft.fft(norm, npoints))[1:]
cycle_freqs = np.fft.fftfreq(npoints)[1:]
real = cycle_freqs > 0
hz = np.array([f * frate for f in cycle_freqs[real]])

fund_freq = hz[np.argmax(w[real])]
print("The fundamental frequency is", round(fund_freq,3), "Hz aka", round(fund_freq*60,1),"bpm")
plt.plot(hz, w[real])
plt.xlim(0,3)
plt.xlabel('Hz')
plt.show()
