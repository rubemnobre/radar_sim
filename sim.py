import numpy as np
import scipy.interpolate
import scipy.signal
import matplotlib.pyplot as plt

class Target:
    def __init__(self, ang, vel):
        pass

    def update(self, peaks):
        pass

ds_x = [-np.pi, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, np.pi]
ds_y =  [np.power(10, -30/10), np.power(10, -9/10), np.power(10, -3/10), 1, np.power(10, -3/10), np.power(10, -9/10), np.power(10, -30/10)]
antenna_gain_interp = scipy.interpolate.CubicSpline(ds_x, ds_y)
antenna_gain = lambda x: antenna_gain_interp(np.mod(x + np.pi, 2*np.pi) - np.pi)

rf = 10e9
targets = [(0, 1), (np.pi/4, 2)]

freq_res = 1
n_window = 512
sampling_freq = freq_res*n_window    # freq_res = sampling_freq / (n_samples)
print("Sampling frequency is %f Hz. Frequency resolution is %f Hz." % (sampling_freq, sampling_freq/n_window))

sweep_time = 10
print("Sweep time is %f s, angle resolution is %f °" % (sweep_time, 180*freq_res/sweep_time))
n_samples = int(sampling_freq*sweep_time)
angles = np.linspace(-np.pi/2, np.pi/2, n_samples)
time = np.linspace(0, sweep_time, n_samples)

samples = np.zeros(len(time))
for t in targets:
    target_angle, target_vel = t
    target_freq = 2*target_vel*rf/3e8 
    print("Target frequency is %f Hz" % (target_freq))
    gains = antenna_gain(angles - target_angle)
    samples = samples + np.multiply(gains, np.cos(2*np.pi*target_freq*time))

freqs = np.linspace(0, sampling_freq, n_window)

angle_res = sweep_time/freq_res

angs = []
ffts = []
targets = []
for i in range(0, len(angles), n_window):
    fft = np.abs(np.fft.fft(samples[i:i+n_window])[:int(n_window/2)])
    ang = np.average(angles[i:i+n_window])
    peaks = scipy.signal.find_peaks(fft, height=15)
    print(ang,[freqs[p] for p in peaks[0]])


    plt.plot(freqs[:int(n_window/2)], fft)
    plt.show()