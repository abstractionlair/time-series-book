### **Worked Example 4.1: Fitting an AR Model Using a Bayesian Approach**

**Context:**
In Section 4.1, the chapter introduces Autoregressive (AR) models, specifically through a Bayesian lens. We discussed how AR models predict the current value of a time series using its past values and how Bayesian methods allow us to incorporate prior knowledge and quantify uncertainty in our parameter estimates.

**Problem:**
Consider a time series representing the monthly average temperature in a city over the last 10 years. We wish to model this data using an AR(2) model, where the current temperature depends on the previous two months' temperatures. Additionally, we want to use a Bayesian approach to estimate the parameters.

**Solution:**
1. **Visual Inspection:** Begin by plotting the time series to visually inspect any patterns or trends.
   
   ```python
   import matplotlib.pyplot as plt
   
   plt.plot(temperature_data)
   plt.title("Monthly Average Temperature Over 10 Years")
   plt.xlabel("Month")
   plt.ylabel("Temperature")
   plt.show()
   ```

   The plot shows some seasonal fluctuations, but we'll focus on modeling the short-term dependencies using an AR(2) model.

2. **Specify the AR(2) Model:** Mathematically, the AR(2) model is given by:
   
   \[
   X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \epsilon_t
   \]
   
   where \( X_t \) is the temperature at month \( t \), \( c \) is the constant term, \( \phi_1 \) and \( \phi_2 \) are the autoregressive coefficients, and \( \epsilon_t \) is the white noise term.

3. **Bayesian Approach:** 
   - **Priors:** Assume normal priors for \( \phi_1 \) and \( \phi_2 \) (e.g., \( \phi_1, \phi_2 \sim N(0, 1) \)) and an inverse-gamma prior for the noise variance \( \sigma^2 \) (e.g., \( \sigma^2 \sim \text{InvGamma}(2, 0.5) \)).
   - **Likelihood:** The likelihood is based on the assumption of normally distributed errors.
   - **Posterior:** Use Markov Chain Monte Carlo (MCMC) methods to sample from the posterior distribution.

   ```python
   import pymc3 as pm
   
   with pm.Model() as model:
       phi1 = pm.Normal('phi1', mu=0, sigma=1)
       phi2 = pm.Normal('phi2', mu=0, sigma=1)
       c = pm.Normal('c', mu=0, sigma=1)
       sigma2 = pm.InverseGamma('sigma2', alpha=2, beta=0.5)
       
       likelihood = pm.Normal('X_t', mu=c + phi1 * X_t_1 + phi2 * X_t_2, sigma=sigma2**0.5, observed=temperature_data[2:])
       
       trace = pm.sample(1000, return_inferencedata=True)
   ```

4. **Results and Interpretation:**
   - Analyze the posterior distributions of \( \phi_1 \), \( \phi_2 \), and \( \sigma^2 \).
   - Use trace plots to assess the convergence of the MCMC chains.
   - Summarize the results with credible intervals for the parameters.

   ```python
   pm.plot_posterior(trace, var_names=['phi1', 'phi2', 'c', 'sigma2'])
   plt.show()
   ```

   The plots show the posterior distributions, indicating the most likely values for the parameters and their associated uncertainties.

**Conclusion:**
This worked example has demonstrated how to fit an AR(2) model using a Bayesian approach, linking the theory discussed in the text to the practical steps of model implementation. This foundation prepares you for Exercise 4.1, where you'll apply similar techniques to new data.

---

### **Worked Example 4.2: Determining the Order of an AR Model**

**Context:**
In Section 4.1, the chapter explains how AR models can have different orders \( p \), where \( p \) is the number of lagged observations used to predict the current value. Selecting the correct order is crucial for accurate modeling and forecasting.

**Problem:**
Given a time series of quarterly GDP growth rates over 20 years, determine the appropriate order \( p \) for an AR model.

**Solution:**
1. **Plot the Time Series:** Start by visualizing the data to detect any patterns or significant changes.
   
   ```python
   plt.plot(gdp_growth_data)
   plt.title("Quarterly GDP Growth Rates")
   plt.xlabel("Quarter")
   plt.ylabel("Growth Rate (%)")
   plt.show()
   ```

2. **Fit AR Models of Different Orders:** Fit AR models with orders ranging from 1 to 5 and calculate the AIC and BIC for each model.
   
   ```python
   from statsmodels.tsa.ar_model import AutoReg
   import statsmodels.api as sm
   
   aic_values = []
   bic_values = []
   for p in range(1, 6):
       model = AutoReg(gdp_growth_data, lags=p).fit()
       aic_values.append(model.aic)
       bic_values.append(model.bic)
   
   print("AIC values:", aic_values)
   print("BIC values:", bic_values)
   ```

3. **Analyze the Results:**
   - Compare the AIC and BIC values across the different models.
   - The model with the lowest AIC and BIC values is generally preferred.
   
   ```python
   plt.plot(range(1, 6), aic_values, label='AIC')
   plt.plot(range(1, 6), bic_values, label='BIC')
   plt.xlabel("AR Order (p)")
   plt.ylabel("Criterion Value")
   plt.legend()
   plt.show()
   ```

   The plot shows how AIC and BIC change with different values of \( p \). The optimal order is where both criteria reach their minimum values.

4. **Cross-Validation:** Perform cross-validation to validate the selected order, ensuring that it generalizes well to unseen data.

   ```python
   from sklearn.model_selection import TimeSeriesSplit
   from sklearn.metrics import mean_squared_error
   
   tscv = TimeSeriesSplit(n_splits=5)
   for p in range(1, 6):
       model = AutoReg(gdp_growth_data, lags=p)
       errors = []
       for train_index, test_index in tscv.split(gdp_growth_data):
           model_fit = model.fit()
           predictions = model_fit.predict(start=test_index[0], end=test_index[-1])
           error = mean_squared_error(gdp_growth_data[test_index], predictions)
           errors.append(error)
       print(f"Order {p}, CV MSE: {np.mean(errors)}")
   ```

   The results from cross-validation help confirm the optimal model order suggested by AIC/BIC.

**Conclusion:**
This worked example illustrates the process of determining the appropriate order for an AR model, combining both model selection criteria and cross-validation. This approach directly leads into Exercise 4.2, where you'll apply these techniques to a different dataset.

---

### **Worked Example 4.3: Estimating MA Models**

**Context:**
Section 4.2 introduces Moving Average (MA) models, which model a time series as a function of past forecast errors. Estimating the parameters of MA models requires careful consideration of invertibility and the autocorrelation structure.

**Problem:**
Given a time series of monthly sales data for a retail store, fit both MA(1) and MA(2) models and compare their performance.

**Solution:**
1. **Visual Inspection:** Begin by plotting the sales data to observe any obvious patterns.
   
   ```python
   plt.plot(sales_data)
   plt.title("Monthly Sales Data")
   plt.xlabel("Month")
   plt.ylabel("Sales")
   plt.show()
   ```

2. **Fit MA(1) and MA(2) Models:**
   - Use maximum likelihood estimation to fit the MA(1) and MA(2) models.
   
   ```python
   from statsmodels.tsa.arima.model import ARIMA
   
   model_ma1 = ARIMA(sales_data, order=(0, 0, 1)).fit()
   model_ma2 = ARIMA(sales_data, order=(0, 0, 2)).fit()
   
   print("MA(1) AIC:", model_ma1.aic)
   print("MA(2) AIC:", model_ma2.aic)
   ```

3. **Analyze Residuals:**
   - Check the residuals from each model for any remaining autocorrelation.
   - Use the ACF plot to assess the residuals.
   
   ```python
   sm.graphics.tsa.plot_acf(model_ma1.resid)
   plt.title("ACF of Residuals for MA(1) Model")
   plt.show()
   
   sm.graphics.tsa.plot_acf(model_ma2.resid)
   plt.title("ACF of Residuals for MA(2) Model")
   plt.show()
   ```

   Ideally, the ACF of the residuals should resemble white noise, indicating a good model fit.

4. **Model Comparison:**
   - Compare the AIC values

 of both models to determine which one provides a better fit.
   - Assess the residual diagnostics to ensure that the chosen model adequately captures the data's structure.

**Conclusion:**
This worked example demonstrates the process of fitting and comparing MA models, preparing you for Exercise 4.3, where you will apply similar techniques to new time series data.
