### Worked Example 1: Fourier Transform and Power Spectrum

#### Problem Statement:
Given a time series composed of two sine waves of frequencies 5 Hz and 15 Hz, with added Gaussian noise, compute and interpret the power spectrum using the Fourier Transform.

#### Step-by-Step Solution:

1. **Generate the Time Series:**
   Let's create a synthetic time series, \( x(t) \), which is the sum of two sine waves and Gaussian noise:

   \[
   x(t) = \sin(2\pi \times 5t) + 0.5\sin(2\pi \times 15t) + \epsilon(t)
   \]
   where \( \epsilon(t) \) is Gaussian noise with mean 0 and standard deviation 0.1.

   ```python
   import numpy as np
   import matplotlib.pyplot as plt

   # Parameters
   t = np.linspace(0, 1, 500)  # 1 second of data at 500 Hz sampling rate
   noise = np.random.normal(0, 0.1, t.shape)
   signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t) + noise

   plt.plot(t, signal)
   plt.title('Time Series')
   plt.xlabel('Time [s]')
   plt.ylabel('Amplitude')
   plt.show()
   ```

2. **Compute the Fourier Transform:**
   Apply the Fast Fourier Transform (FFT) to the time series to obtain the frequency components:

   ```python
   X = np.fft.fft(signal)
   freqs = np.fft.fftfreq(len(t), t[1] - t[0])

   plt.plot(freqs[:len(freqs)//2], np.abs(X[:len(X)//2])**2)
   plt.title('Power Spectrum')
   plt.xlabel('Frequency [Hz]')
   plt.ylabel('Power')
   plt.show()
   ```

   Here, `np.fft.fft` computes the FFT, and `np.fft.fftfreq` returns the corresponding frequencies.

3. **Interpret the Results:**
   The power spectrum should display two prominent peaks corresponding to the sine wave frequencies (5 Hz and 15 Hz). The magnitude of these peaks reflects the strength of the sinusoidal components, and any additional noise should manifest as a more spread-out background level in the spectrum.

   **Key Insight:**
   This example demonstrates how Fourier analysis can decompose a complex time series into its constituent frequency components, providing a clear view of the periodic behavior within the data.

### Worked Example 2: Spectral Leakage and Windowing

#### Problem Statement:
Explore spectral leakage in a time series and apply window functions to mitigate this effect.

#### Step-by-Step Solution:

1. **Generate the Time Series:**
   Consider a pure sine wave at 10 Hz sampled at 100 Hz for 1 second:

   ```python
   t = np.linspace(0, 1, 100)  # 1 second of data at 100 Hz
   signal = np.sin(2 * np.pi * 10 * t)

   plt.plot(t, signal)
   plt.title('10 Hz Sine Wave')
   plt.xlabel('Time [s]')
   plt.ylabel('Amplitude')
   plt.show()
   ```

2. **Compute the FFT Without Windowing:**
   Calculate the FFT and observe spectral leakage:

   ```python
   X = np.fft.fft(signal)
   freqs = np.fft.fftfreq(len(t), t[1] - t[0])

   plt.plot(freqs[:len(freqs)//2], np.abs(X[:len(X)//2])**2)
   plt.title('Power Spectrum without Windowing')
   plt.xlabel('Frequency [Hz]')
   plt.ylabel('Power')
   plt.show()
   ```

   **Observation:**
   Notice that the power spectrum might show spread energy around the 10 Hz frequency, which is an artifact of spectral leakage due to the abrupt beginning and end of the signal.

3. **Apply Window Functions:**
   Now, apply a Hamming window to reduce spectral leakage:

   ```python
   from scipy.signal import get_window

   window = get_window('hamming', len(signal))
   windowed_signal = signal * window

   X_windowed = np.fft.fft(windowed_signal)

   plt.plot(freqs[:len(freqs)//2], np.abs(X_windowed[:len(X_windowed)//2])**2)
   plt.title('Power Spectrum with Hamming Window')
   plt.xlabel('Frequency [Hz]')
   plt.ylabel('Power')
   plt.show()
   ```

   **Interpretation:**
   The application of the window should reduce the spectral leakage, leading to a sharper peak at 10 Hz.

   **Key Insight:**
   This example illustrates the importance of windowing in Fourier analysis to minimize spectral leakage and obtain more accurate frequency estimates.

### Worked Example 3: Spectral Density Estimation

#### Problem Statement:
Estimate the spectral density of an AR(2) process using both classical and Bayesian approaches.

#### Step-by-Step Solution:

1. **Simulate an AR(2) Process:**
   The AR(2) process is defined as:

   \[
   X_t = 0.75X_{t-1} - 0.5X_{t-2} + \epsilon_t
   \]
   where \( \epsilon_t \) is white noise.

   ```python
   import statsmodels.api as sm

   np.random.seed(42)
   ar_params = np.array([0.75, -0.5])
   ma_params = np.array([0])
   ar = np.r_[1, -ar_params]  # add lag-0 coefficient
   ma = np.r_[1, ma_params]
   ar2_process = sm.tsa.ArmaProcess(ar, ma)
   X = ar2_process.generate_sample(nsample=500)

   plt.plot(X)
   plt.title('AR(2) Process')
   plt.xlabel('Time')
   plt.ylabel('Amplitude')
   plt.show()
   ```

2. **Estimate Spectral Density Using the Periodogram:**
   Compute the periodogram as a non-parametric estimate of the spectral density:

   ```python
   f, Pxx = plt.psd(X, NFFT=256, Fs=1)
   plt.show()
   ```

   **Observation:**
   The periodogram should show peaks corresponding to the dominant frequencies of the AR(2) process.

3. **Bayesian Spectral Estimation:**
   Use Bayesian methods with the Whittle likelihood and a Gaussian process prior for spectral estimation:

   ```python
   import pymc3 as pm
   import numpy as np

   def spectral_gp_model(x):
       N = len(x)
       f = np.fft.rfftfreq(N)
       I = np.abs(np.fft.rfft(x))**2 / N

       with pm.Model() as model:
           # Prior on the mean log spectral density
           μ = pm.Normal('μ', 0, 10)

           # GP prior on the log spectral density
           ℓ = pm.Gamma('ℓ', alpha=2, beta=0.1)
           η = pm.HalfNormal('η', sigma=1)
           cov = η**2 * pm.gp.cov.ExpQuad(1, ℓ)
           gp = pm.gp.Latent(cov_func=cov)

           # Log spectral density
           log_S = gp.prior('log_S', X=f[:, None]) + μ

           # Whittle likelihood
           pm.Potential('likelihood', -0.5 * np.sum(log_S + I / np.exp(log_S)))

           # Sample from the posterior
           trace = pm.sample(1000, tune=1000)

       return trace

   trace = spectral_gp_model(X)
   pm.plot_posterior(trace)
   ```

   **Interpretation:**
   Compare the posterior distribution of the spectral density to the classical periodogram. Discuss how Bayesian methods provide a more flexible and probabilistic view of the spectral density.

   **Key Insight:**
   This example emphasizes the strengths of Bayesian spectral estimation, particularly in incorporating uncertainty and prior knowledge into the analysis.

### Worked Example 4: Wavelet Transform and Time-Frequency Analysis

#### Problem Statement:
Use wavelet analysis to analyze a non-stationary signal with time-varying frequency components.

#### Step-by-Step Solution:

1. **Generate a Non-Stationary Time Series:**
   Create a signal where the frequency changes over time, e.g., a chirp:

   ```python
   from scipy.signal import chirp

   t = np.linspace(0, 10, 500)
   signal = chirp(t, f0=5, f1=50, t1=10, method='linear')

   plt.plot(t, signal)
   plt.title('Chirp Signal')
   plt.xlabel('Time [s]')
   plt.ylabel('Amplitude')
   plt.show()
   ```

2. **Perform Continuous Wavelet Transform (CWT):**
   Apply the CWT using the Morlet wavelet:

   ```python
   import pywt

   scales = np.arange(1, 128)
   coeffs, freqs = pywt.cwt(signal, scales, 'morlet')

   plt.imshow(np.abs(coeffs)**2, aspect='auto', extent=[t[0], t[-1], freqs[-1], freqs[0]])
   plt.colorbar(label='Power')
   plt.ylabel('Frequency')
   plt.xlabel('Time

')
   plt.title('Wavelet Power Spectrum')
   plt.show()
   ```

   **Interpretation:**
   The wavelet power spectrum reveals how the frequency content of the signal changes over time, highlighting the chirp's increasing frequency.

   **Key Insight:**
   Wavelet analysis provides a powerful tool for analyzing non-stationary signals, capturing both frequency and time localization.

### Worked Example 5: Hilbert-Huang Transform and EMD

#### Problem Statement:
Decompose a complex signal using EMD and perform Hilbert spectral analysis to determine the instantaneous frequency.

#### Step-by-Step Solution:

1. **Generate a Complex Signal:**
   Create a signal with multiple non-linear and non-stationary components:

   ```python
   t = np.linspace(0, 10, 500)
   signal = np.sin(2 * np.pi * 1 * t) + np.sin(2 * np.pi * 5 * t) * np.sin(2 * np.pi * 0.5 * t)

   plt.plot(t, signal)
   plt.title('Complex Signal')
   plt.xlabel('Time [s]')
   plt.ylabel('Amplitude')
   plt.show()
   ```

2. **Apply EMD to Extract IMFs:**
   Use EMD to decompose the signal into intrinsic mode functions (IMFs):

   ```python
   from PyEMD import EMD

   emd = EMD()
   IMFs = emd(signal)

   for i, imf in enumerate(IMFs):
       plt.plot(t, imf, label=f'IMF {i+1}')
   plt.title('IMFs')
   plt.xlabel('Time [s]')
   plt.ylabel('Amplitude')
   plt.legend()
   plt.show()
   ```

3. **Perform Hilbert Spectral Analysis:**
   Compute the instantaneous frequency and amplitude using the Hilbert transform on each IMF:

   ```python
   from scipy.signal import hilbert

   for imf in IMFs:
       analytic_signal = hilbert(imf)
       amplitude_envelope = np.abs(analytic_signal)
       instantaneous_phase = np.unwrap(np.angle(analytic_signal))
       instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * 500  # assuming 500 Hz sampling rate

       plt.plot(t[:-1], instantaneous_frequency, label='Instantaneous Frequency')
       plt.title('Instantaneous Frequency')
       plt.xlabel('Time [s]')
       plt.ylabel('Frequency [Hz]')
       plt.show()
   ```

   **Interpretation:**
   The instantaneous frequency plot provides insights into how the frequency content of each IMF varies over time, highlighting non-linear and non-stationary dynamics.

   **Key Insight:**
   The Hilbert-Huang Transform, combining EMD and Hilbert spectral analysis, is particularly effective for analyzing complex signals with non-linear and non-stationary characteristics.
