### Worked Examples for Chapter 3: Time Series Components and Decomposition

---

#### **Worked Example 1: Testing for Stationarity**

**Objective:** To demonstrate how to apply and interpret the results of stationarity tests on a time series dataset.

**Problem:**
You are given a dataset containing monthly airline passenger numbers from 1949 to 1960. The task is to determine whether the series is stationary and to explore methods to achieve stationarity if it is not.

**Step-by-Step Solution:**

1. **Plot the Time Series:**
   - Start by visualizing the data to identify any obvious trends or seasonal patterns.
   - **Python Code:**
     ```python
     import pandas as pd
     import matplotlib.pyplot as plt

     data = pd.read_csv('AirPassengers.csv', index_col='Month', parse_dates=True)
     plt.plot(data)
     plt.title('Monthly Airline Passengers (1949-1960)')
     plt.show()
     ```
   - **Interpretation:** The plot shows a clear upward trend and seasonal fluctuations, indicating that the series is likely non-stationary.

2. **Augmented Dickey-Fuller (ADF) Test:**
   - Apply the ADF test to check for stationarity.
   - **Python Code:**
     ```python
     from statsmodels.tsa.stattools import adfuller

     result = adfuller(data['#Passengers'])
     print('ADF Statistic:', result[0])
     print('p-value:', result[1])
     ```
   - **Interpretation:** If the p-value is greater than 0.05, we fail to reject the null hypothesis, suggesting the series has a unit root and is non-stationary.

3. **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test:**
   - Perform the KPSS test to cross-check the stationarity results.
   - **Python Code:**
     ```python
     from statsmodels.tsa.stattools import kpss

     kpss_result = kpss(data['#Passengers'], regression='c')
     print('KPSS Statistic:', kpss_result[0])
     print('p-value:', kpss_result[1])
     ```
   - **Interpretation:** A low p-value (< 0.05) suggests the series is trend stationary.

4. **Transformation for Stationarity:**
   - Apply a differencing technique to remove the trend and make the series stationary.
   - **Python Code:**
     ```python
     data_diff = data.diff().dropna()
     plt.plot(data_diff)
     plt.title('Differenced Series')
     plt.show()

     result_diff = adfuller(data_diff['#Passengers'])
     print('ADF Statistic (Differenced):', result_diff[0])
     print('p-value (Differenced):', result_diff[1])
     ```
   - **Interpretation:** After differencing, the ADF test should show a low p-value, indicating the series is now stationary.

---

#### **Worked Example 2: Dealing with Non-Stationarity**

**Objective:** To demonstrate the techniques used to handle non-stationary time series.

**Problem:**
You are working with a dataset containing daily closing prices of a stock over several years. The task is to identify non-stationarity, apply transformations, and explore suitable models.

**Step-by-Step Solution:**

1. **Identify Non-Stationarity:**
   - Plot the time series and perform ADF and KPSS tests as in Worked Example 1 to confirm non-stationarity.

2. **Apply Differencing:**
   - To handle the trend, first difference the series.
   - **Python Code:**
     ```python
     stock_data_diff = stock_data.diff().dropna()
     plt.plot(stock_data_diff)
     plt.title('First Differenced Stock Prices')
     plt.show()
     ```

3. **Log Transformation:**
   - Apply a logarithmic transformation to stabilize the variance.
   - **Python Code:**
     ```python
     log_data = np.log(stock_data)
     plt.plot(log_data)
     plt.title('Log-transformed Stock Prices')
     plt.show()
     ```

4. **Check for Stationarity Again:**
   - After transformations, repeat the stationarity tests to confirm that the series is stationary.
   - **Interpretation:** Ideally, both the ADF and KPSS tests should indicate that the transformed series is now stationary.

5. **ARIMA Modeling:**
   - Use the transformed, stationary series to identify ARIMA model parameters.
   - **Python Code:**
     ```python
     from statsmodels.tsa.arima.model import ARIMA

     model = ARIMA(stock_data_diff, order=(1,1,1))
     model_fit = model.fit()
     print(model_fit.summary())
     ```
   - **Interpretation:** Review the model summary to confirm the adequacy of the ARIMA model in capturing the dynamics of the series.

---

#### **Worked Example 3: Understanding Ergodicity**

**Objective:** To explore the concept of ergodicity and how it applies to time series analysis.

**Problem:**
Consider a time series representing the daily temperature of a city over 50 years. The task is to determine whether the series is ergodic.

**Step-by-Step Solution:**

1. **Define Ergodicity:**
   - Ergodicity implies that the time averages and ensemble averages of a process are equivalent.

2. **Estimate Time Averages:**
   - Compute the mean temperature over different periods (e.g., each decade) and compare.
   - **Python Code:**
     ```python
     mean_1970s = temperature_data['1970-01-01':'1979-12-31'].mean()
     mean_1980s = temperature_data['1980-01-01':'1989-12-31'].mean()
     print(mean_1970s, mean_1980s)
     ```
   - **Interpretation:** If the means across different periods are similar, the process may be ergodic.

3. **Compare to Ensemble Average:**
   - If possible, compare with the ensemble average (average across multiple realizations of the process). If unavailable, discuss the implications.

4. **Implications of Non-Ergodicity:**
   - Reflect on what non-ergodicity would mean for modeling, such as difficulties in using past data to predict future behavior.

---

#### **Worked Example 4: Empirical Mode Decomposition (EMD)**

**Objective:** To implement EMD and interpret its results on a non-linear time series.

**Problem:**
Given a time series of daily wind speed data, the task is to decompose it using EMD and analyze the resulting Intrinsic Mode Functions (IMFs).

**Step-by-Step Solution:**

1. **Apply EMD:**
   - Decompose the series into IMFs.
   - **Python Code:**
     ```python
     from PyEMD import EMD
     emd = EMD()
     IMFs = emd.emd(wind_speed_data)
     plt.plot(IMFs)
     plt.title('Intrinsic Mode Functions')
     plt.show()
     ```

2. **Interpret IMFs:**
   - Analyze the physical meaning of each IMF. For example, some IMFs may correspond to short-term oscillations, while others may represent longer-term trends.

3. **Statistical Properties of IMFs:**
   - Test each IMF for stationarity using the ADF test, similar to previous examples.
   - **Python Code:**
     ```python
     for i, imf in enumerate(IMFs):
         print(f'IMF {i+1} ADF Test:')
         adf_result = adfuller(imf)
         print(adf_result)
     ```

4. **Forecast Using IMFs:**
   - Select one or more IMFs for forecasting and compare the performance with an ARIMA model.
   - **Interpretation:** Discuss the advantages of EMD in isolating different components of the series for forecasting.

---

#### **Worked Example 5: Bayesian and Frequentist Approaches to Stationarity and Ergodicity**

**Objective:** To compare and contrast Bayesian and frequentist perspectives in analyzing time series.

**Problem:**
You have a time series of annual rainfall over 100 years. The task is to analyze the series for stationarity and ergodicity from both Bayesian and frequentist perspectives.

**Step-by-Step Solution:**

1. **Frequentist Approach:**
   - Perform ADF and KPSS tests to check for stationarity.
   - **Python Code:** (As used in previous examples)
   - **Interpretation:** Based on test results, make a frequentist decision regarding the stationarity and ergodicity.

2. **Bayesian Approach:**
   - Use Bayesian inference to update beliefs about stationarity and ergodicity as more data is observed.
   - **Python Code:** (This might require more advanced statistical packages or custom Bayesian models)
   - **Interpretation:** Compare posterior distributions and discuss how they evolve with new data.

3. **Compare Results:**
   - Summarize the differences in conclusions reached by the Bayesian and frequentist approaches.

4. **Reflection:**
   - Discuss the scenarios where one approach might be preferred, considering the nature of the data and the goals of the analysis.

---
