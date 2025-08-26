### Worked Example 1: Bayesian Inference for an AR(1) Model
**Objective:** Understand how to perform Bayesian inference on a simple AR(1) model.

**Problem:** Suppose we have a time series \( X_t \) following an AR(1) model:
\[
X_t = \phi X_{t-1} + \epsilon_t,
\]
where \( \epsilon_t \sim \mathcal{N}(0, \sigma^2) \) is white noise with unknown variance \(\sigma^2\).

We assume the following prior distributions:
- \( \phi \sim \mathcal{N}(0, 1) \)
- \( \sigma^2 \sim \text{Inverse-Gamma}(2, 2) \)

**Solution:**

1. **Likelihood Function:**
   The likelihood function based on \(n\) observations is:
   \[
   p(X | \phi, \sigma^2) = \prod_{t=2}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(X_t - \phi X_{t-1})^2}{2\sigma^2}\right)
   \]

2. **Posterior Distribution:**
   The posterior distribution is obtained by combining the likelihood with the prior:
   \[
   p(\phi, \sigma^2 | X) \propto p(X | \phi, \sigma^2) \cdot p(\phi) \cdot p(\sigma^2)
   \]
   Due to the non-conjugate nature of this problem, the posterior distribution is typically sampled using MCMC methods.

3. **MCMC Sampling:**
   Implementing a simple Gibbs sampler:
   - **Step 1:** Sample \(\phi\) given \(\sigma^2\) and data \(X\).
   - **Step 2:** Sample \(\sigma^2\) given \(\phi\) and data \(X\).

4. **Interpretation:**
   After running the MCMC algorithm, plot the posterior distribution of \(\phi\) and compare it to the prior. This provides insight into how the data has updated our beliefs about \(\phi\).

This example directly leads into Exercise 1, where students will extend this approach and apply it to a dataset using MCMC to estimate the posterior distributions.

### Worked Example 2: Conducting the ADF Test
**Objective:** Learn how to perform the Augmented Dickey-Fuller (ADF) test to check for stationarity.

**Problem:** Given a time series \( Y_t \), you want to test for the presence of a unit root to determine if the series is non-stationary.

**Solution:**

1. **ADF Test Setup:**
   The ADF test tests the null hypothesis that the time series has a unit root (i.e., is non-stationary):
   \[
   H_0: \text{The series } Y_t \text{ has a unit root.}
   \]
   \[
   H_1: \text{The series } Y_t \text{ is stationary.}
   \]

2. **Test Statistic:**
   The test involves estimating the following regression:
   \[
   \Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta Y_{t-i} + \epsilon_t
   \]
   The test statistic is the t-statistic for \(\gamma = 0\).

3. **Performing the Test:**
   Using statistical software, apply the ADF test to \(Y_t\) and obtain the test statistic and p-value. Compare the p-value to a significance level (e.g., 0.05) to make a decision about the null hypothesis.

4. **Result Interpretation:**
   If the p-value is below the threshold, reject the null hypothesis and conclude that the series is stationary.

This example prepares students for Exercise 2, where they will apply the ADF test to real data and interpret the results.

### Worked Example 3: Mutual Information in Time Series
**Objective:** Explore the concept of mutual information to detect dependencies between two time series.

**Problem:** You have two time series \( X_t \) and \( Y_t \). You want to investigate whether \( X_t \) influences \( Y_t \) with a certain lag.

**Solution:**

1. **Mutual Information Definition:**
   Mutual information quantifies the amount of information obtained about one random variable through another. For time series, it can be calculated as:
   \[
   I(X_{t-k}; Y_t) = \sum_{x, y} p(x, y) \log \left( \frac{p(x, y)}{p(x)p(y)} \right)
   \]
   where \( k \) is the lag.

2. **Calculating Mutual Information:**
   Compute the mutual information between \( X_t \) and \( Y_t \) for different lags. This requires estimating the joint and marginal probabilities, which can be done using histograms or kernel density estimation.

3. **Lag Selection:**
   Identify the lag \( k \) that maximizes the mutual information, suggesting the most significant dependency between \( X_t \) and \( Y_t \).

4. **Interpretation:**
   High mutual information at a certain lag indicates a strong dependency, potentially suggesting a causal relationship.

This worked example leads to Exercise 3, where students will calculate mutual information for different lags and explore the dependencies in a time series context.

### Worked Example 4: Model Comparison Using AIC and BIC
**Objective:** Learn how to compare time series models using information criteria.

**Problem:** You have fitted two models to a time series \( Z_t \): an AR(1) model and an AR(2) model. You want to decide which model is more appropriate using AIC and BIC.

**Solution:**

1. **AIC and BIC Definition:**
   - **AIC:** \( \text{AIC} = 2k - 2\log(L) \), where \( k \) is the number of parameters and \( L \) is the likelihood of the model.
   - **BIC:** \( \text{BIC} = k\log(n) - 2\log(L) \), where \( n \) is the number of observations.

2. **Calculating AIC and BIC:**
   Compute the AIC and BIC for both the AR(1) and AR(2) models using the likelihood values and the number of parameters.

3. **Comparison:**
   The model with the lower AIC or BIC is preferred. Discuss the implications of choosing one model over the other based on these criteria, considering the trade-off between model fit and complexity.

4. **Example Calculation:**
   Perform the calculations on an example time series and interpret the results.

This example provides the foundation for Exercise 4, where students will apply these criteria to their models and decide on the best fit.

### Worked Example 5: Evaluating Assumptions in ARIMA Models
**Objective:** Understand the assumptions behind ARIMA models and how to evaluate them.

**Problem:** Fit an ARIMA model to a non-stationary time series, ensure stationarity, and evaluate the model's residuals.

**Solution:**

1. **Checking Stationarity:**
   Use visual inspection (e.g., plotting) and statistical tests (e.g., ADF test) to check for stationarity. If the series is non-stationary, apply differencing until stationarity is achieved.

2. **Fitting the ARIMA Model:**
   Once stationarity is ensured, fit the ARIMA model to the transformed series. Estimate the model parameters using Maximum Likelihood Estimation (MLE).

3. **Residual Analysis:**
   After fitting, analyze the residuals to check for independence using tools like the autocorrelation function (ACF) of the residuals.

4. **Discussion:**
   Reflect on how violations of stationarity and other assumptions impact the model’s accuracy and forecast reliability.

This example connects with Exercise 5, guiding students through the practical steps of fitting and validating ARIMA models while highlighting the importance of underlying assumptions.
