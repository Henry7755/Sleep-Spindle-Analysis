# Loading necessary libraries
import numpy as np
import pandas as pd
import scipy.signal
from numpy.fft import fft, fftfreq
from matplotlib import pyplot as plt
from scipy.signal import butter, hilbert, filtfilt


data = pd.read_csv("extrait_wSleepPage01.csv", delimiter = ";") # loadind and storing data

print(f"Shape of the data is{data.shape}\n")
print(f"The Data has {data.shape[1]} columns\n")
print(data.head(10))

numerical_features = data.iloc[:,1:9]


# Taking out the commas and replacing with dots
for column in numerical_features.columns:
    numerical_features[column] = numerical_features[column].astype(str)  # Ensure the column is treated as a string
    numerical_features[column] = numerical_features[column].str.replace(",", ".", regex=True)
    numerical_features[column] = pd.to_numeric(numerical_features[column], errors='coerce')

numerical_features['Time (s)'] = np.arange(0,data.count()[0]*0.005,0.005)
df = numerical_features.iloc[:,3:-1]


# Spindle Characteristics
fs = 1 /0.005
lowcut = 11.0
highcut = 16.0
window_size = int(fs*1)


# filtering out the signal to contain a band of spindle feature frequency
def bandpass_filter (signal, sampling_freq, low_freq, high_freq, order = 3, padlen = None):
    nyquist = 0.5 * sampling_freq
    low = low_freq / nyquist
    high = high_freq/ nyquist
    b,a = butter(order,[low, high], btype = 'band')

    # Apply the filter to each column (assuming each column is a separate signal)
    filtered_signals = []
    for column in signal.columns:
        sig = signal[column]
        if padlen is not None and len(sig) <= padlen:
            raise ValueError("Signal length must be greater than padlen.")
        y = filtfilt(b, a, sig, padlen=padlen)
        filtered_signals.append(y)

    return pd.DataFrame(filtered_signals).transpose() # Transpose to get columns as signals
 

filtered_signal = bandpass_filter(df, fs,lowcut, highcut, 2,15) # Pass the DataFrame to filter each column

# Signal Windowing
window_size = int(fs*1) # 1 second window
window = scipy.signal.windows.hamming(window_size)

windowed_signals = []
for column in filtered_signal.columns:
    signal = filtered_signal[column]
    windowed_signal = scipy.signal.convolve(signal, window, mode='same') / window.sum()
    windowed_signals.append(windowed_signal)

windowed_signal = pd.DataFrame(windowed_signals).transpose() # Transpose to get columns as signals


######################### Error: challenge look at the reason why the graph are that way. Contact P.E Adjei

# Conversion of Signal from the Time Domain to Frequency Domain
fft_result = np.abs(fft(filtered_signal))

# Calculate frequencies
frequencies = fftfreq(window_size, d=1/fs) 

# Plotting frequencies by the magnitude of each column signal
plt.plot(frequencies, fft_result[:len(frequencies),0])


print(f" Shape of filtered signal: {filtered_signal.shape}")

