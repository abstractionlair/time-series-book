### Exercise 1: Fourier Transform and Power Spectrum
**Objective:** Compute and analyze the power spectrum of a time series using the Fourier Transform.

1. **Generate a synthetic time series** consisting of a sum of two sine waves with different frequencies and add Gaussian noise.
2. **Compute the Fourier Transform** of the time series using the Fast Fourier Transform (FFT).
3. **Plot the power spectrum** and identify the frequencies present in the signal.
4. **Interpret the results**: Discuss how the power spectrum reflects the underlying sine waves and the effect of noise.

*Hint:* Use Python's `numpy.fft` module to compute the FFT and `matplotlib` for plotting.

---

### Exercise 2: Spectral Leakage and Windowing
**Objective:** Explore the concept of spectral leakage and the use of windowing to mitigate it.

1. **Generate a time series** consisting of a single sine wave of known frequency.
2. **Compute the Fourier Transform** without any windowing and plot the power spectrum.
3. **Apply different window functions** (e.g., Hamming, Hanning, and Blackman) to the time series before computing the Fourier Transform.
4. **Plot the power spectra** for each windowed signal and compare them to the unwindowed spectrum.
5. **Discuss the effects** of windowing on spectral leakage and frequency resolution.

*Hint:* Use `scipy.signal.get_window` to generate the window functions.

---

### Exercise 3: Spectral Density Estimation
**Objective:** Estimate and compare the spectral density of a time series using classical and Bayesian methods.

1. **Simulate a time series** with a known spectral density (e.g., AR(2) process).
2. **Estimate the spectral density** using the periodogram and Welch’s method.
3. **Implement a Bayesian spectral estimation** using the Whittle likelihood and Gaussian process priors.
4. **Compare the spectral density estimates** from the classical and Bayesian approaches.
5. **Interpret the results**: Discuss the strengths and limitations of each method.

*Hint:* Use `scipy.signal.welch` for Welch’s method and `pymc3` for Bayesian estimation.

---

### Exercise 4: Wavelet Transform and Time-Frequency Analysis
**Objective:** Perform wavelet analysis to investigate the time-frequency characteristics of a non-stationary time series.

1. **Generate a non-stationary time series** that contains a sine wave with a frequency that changes over time.
2. **Compute the Continuous Wavelet Transform (CWT)** using a Morlet wavelet.
3. **Plot the wavelet power spectrum** and identify how the frequency content of the time series evolves over time.
4. **Interpret the results**: Discuss how wavelet analysis captures the time-varying frequency characteristics.

*Hint:* Use `pywt.cwt` from the PyWavelets library.

---

### Exercise 5: Hilbert-Huang Transform and Empirical Mode Decomposition (EMD)
**Objective:** Decompose a complex signal into its intrinsic mode functions (IMFs) and analyze the instantaneous frequency using the Hilbert-Huang Transform.

1. **Generate a complex signal** consisting of multiple non-linear and non-stationary components.
2. **Apply Empirical Mode Decomposition (EMD)** to the signal to extract IMFs.
3. **Perform Hilbert Spectral Analysis** on the IMFs to compute the instantaneous frequency.
4. **Plot the instantaneous frequency** over time for each IMF and interpret the results.
5. **Discuss the advantages** of using the Hilbert-Huang Transform for analyzing non-linear and non-stationary time series.

*Hint:* Use Python libraries like `PyEMD` for EMD and custom implementations for the Hilbert Transform.
