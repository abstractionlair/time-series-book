Here are the worked examples for Chapter 9. These examples aim to bridge the gap between the main text and the exercises, providing a clear path for students to tackle the exercises with confidence.

### Worked Example 9.1: Extracting Statistical Features from Time Series

**Objective:** To demonstrate the process of extracting basic statistical features from a time series dataset.

**Problem Statement:**
Given a time series dataset representing daily temperatures over a year, extract the following features: mean, variance, skewness, kurtosis, and the rolling mean and standard deviation with a window size of 30.

**Solution:**

1. **Load the Dataset:**
   ```python
   import pandas as pd
   
   # Assuming the dataset is a CSV file
   df = pd.read_csv('daily_temperatures.csv')
   temperature_series = df['temperature']
   ```

2. **Calculate Basic Statistical Features:**
   ```python
   mean_temp = temperature_series.mean()
   var_temp = temperature_series.var()
   skew_temp = temperature_series.skew()
   kurt_temp = temperature_series.kurtosis()
   
   print(f"Mean: {mean_temp}, Variance: {var_temp}, Skewness: {skew_temp}, Kurtosis: {kurt_temp}")
   ```

3. **Compute Rolling Mean and Standard Deviation:**
   ```python
   rolling_mean = temperature_series.rolling(window=30).mean()
   rolling_std = temperature_series.rolling(window=30).std()
   
   # Plotting the rolling statistics
   temperature_series.plot(label='Original', alpha=0.5)
   rolling_mean.plot(label='Rolling Mean')
   rolling_std.plot(label='Rolling Std')
   plt.legend()
   plt.show()
   ```

4. **Interpretation:**
   - The mean and variance give a sense of the central tendency and dispersion of the temperature series.
   - Skewness and kurtosis provide insights into the asymmetry and tail behavior of the distribution.
   - The rolling statistics show how the mean and variability of temperatures change over time, revealing potential seasonal patterns.

This example prepares students for Exercise 9.1 by illustrating the extraction and interpretation of basic statistical features.

---

### Worked Example 9.2: Autocorrelation Analysis

**Objective:** To perform an autocorrelation analysis on a time series and interpret the results.

**Problem Statement:**
Analyze the autocorrelation of a daily stock price series to identify any significant periodic patterns or relationships between lags.

**Solution:**

1. **Load the Stock Price Series:**
   ```python
   import pandas as pd
   from statsmodels.graphics.tsaplots import plot_acf
   
   df = pd.read_csv('daily_stock_prices.csv')
   price_series = df['price']
   ```

2. **Calculate and Plot the Autocorrelation Function (ACF):**
   ```python
   plot_acf(price_series, lags=10)
   plt.show()
   ```

3. **Interpretation:**
   - The ACF plot shows the correlation of the series with its own lagged values.
   - Significant spikes at certain lags indicate a strong relationship at those points. For example, a significant spike at lag 1 suggests that today’s price is strongly related to yesterday’s price.
   - If there are significant spikes at multiple lags, this might indicate the presence of cyclic behavior in the stock prices.

This example helps students understand how to conduct autocorrelation analysis, setting the foundation for Exercise 9.2.

---

### Worked Example 9.3: Frequency Domain Analysis using Fourier Transform

**Objective:** To transform a time series into the frequency domain and identify dominant frequencies.

**Problem Statement:**
Using a monthly sales dataset, apply the Fourier Transform and identify any dominant frequencies.

**Solution:**

1. **Load the Sales Data:**
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   
   df = pd.read_csv('monthly_sales.csv')
   sales_series = df['sales']
   ```

2. **Apply Fourier Transform:**
   ```python
   fft_values = np.fft.fft(sales_series)
   frequencies = np.fft.fftfreq(len(fft_values))
   
   # Only consider the positive frequencies
   positive_freqs = frequencies[np.where(frequencies >= 0)]
   power_spectrum = np.abs(fft_values[np.where(frequencies >= 0)])
   
   # Plot the power spectrum
   plt.plot(positive_freqs, power_spectrum)
   plt.xlabel('Frequency')
   plt.ylabel('Power')
   plt.show()
   ```

3. **Interpretation:**
   - The power spectrum highlights the contribution of various frequencies to the overall time series.
   - Peaks in the spectrum represent dominant cycles. For example, a peak at a frequency corresponding to 1/12 might indicate an annual cycle in the sales data.

This example is directly relevant to Exercise 9.3 and familiarizes students with frequency domain analysis.

---

### Worked Example 9.4: Nonlinear Feature Extraction via Sample Entropy

**Objective:** To compute and interpret sample entropy as a measure of time series complexity.

**Problem Statement:**
Calculate the sample entropy of a heart rate variability (HRV) time series to assess its complexity.

**Solution:**

1. **Load the HRV Data:**
   ```python
   df = pd.read_csv('hrv_data.csv')
   hrv_series = df['HRV']
   ```

2. **Define a Function to Calculate Sample Entropy:**
   ```python
   import numpy as np
   
   def sample_entropy(time_series, m, r):
       N = len(time_series)
       B = 0.0
       A = 0.0
       
       # Split the time series into m-length subsequences
       for i in range(N - m):
           x = time_series[i:i+m]
           for j in range(i+1, N-m):
               y = time_series[j:j+m]
               if np.all(np.abs(x - y) < r):
                   B += 1
                   if np.all(np.abs(time_series[i:i+m+1] - time_series[j:j+m+1]) < r):
                       A += 1
       return -np.log(A / B)
   
   # Calculate sample entropy
   sampen = sample_entropy(hrv_series, m=2, r=0.2*np.std(hrv_series))
   print(f"Sample Entropy: {sampen}")
   ```

3. **Interpretation:**
   - Sample entropy provides a measure of the predictability or regularity of a time series.
   - Lower entropy values suggest more regularity (less complexity), while higher values indicate higher complexity.
   - In the context of HRV, a higher sample entropy might indicate healthy variability, while lower entropy could suggest a pathological condition.

This example prepares students for Exercise 9.4 by explaining the calculation of sample entropy.

---

### Worked Example 9.5: Feature Extraction Using CNN

**Objective:** To automatically extract features from a time series using a Convolutional Neural Network (CNN).

**Problem Statement:**
Train a CNN on a dataset of time series representing different speech signals and extract the learned features.

**Solution:**

1. **Prepare the Data:**
   ```python
   import tensorflow as tf
   from tensorflow.keras import layers, models
   
   # Assuming `X_train` is your time series data and `y_train` are the labels
   X_train = X_train.reshape(-1, X_train.shape[1], 1)  # Reshape for CNN input
   ```

2. **Build and Train the CNN:**
   ```python
   model = models.Sequential()
   model.add(layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
   model.add(layers.MaxPooling1D(pool_size=2))
   model.add(layers.Flatten())
   model.add(layers.Dense(100, activation='relu'))
   model.add(layers.Dense(10, activation='softmax'))
   
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   model.fit(X_train, y_train, epochs=10, batch_size=32)
   ```

3. **Extract and Visualize Features:**
   ```python
   feature_model = models.Model(inputs=model.input, outputs=model.layers[-2].output)
   features = feature_model.predict(X_train)
   
   # Visualize the learned features
   plt.plot(features[0])
   plt.show()
   ```

4. **Interpretation:**
   - The CNN automatically extracts hierarchical features from the time series that are useful for classification.
   - The visualized features can be compared to hand-crafted features to understand the differences and advantages of automatic feature extraction.

This example supports Exercise 9.5 by showing how to use CNNs for feature extraction from time series.

---

### Worked Example 9.6: Applying Kernel Methods with SVM

**Objective:** To classify time series data using a Support Vector Machine (SVM) with a Gaussian RBF kernel.

**Problem Statement:**
Classify a set of time series representing different ECG signals using an SVM with an RBF kernel.

**Solution:**

1. **Prepare the Data:**
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from sklearn.svm import SVC
   
   X_train, X_test, y_train, y_test = train_test_split(ecg_data, labels, test_size=0.2)
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

2. **Train the SVM:**
   ```python
   svm_model = SVC(kernel='rbf', gamma='scale')
