### Exercise 1: Bayesian Inference with Simple Time Series Data
**Objective:** Apply Bayesian inference to estimate parameters of a simple time series model.

1. **Problem Statement:** Consider a simple autoregressive model of order 1, AR(1), given by:
   \[
   X_t = \phi X_{t-1} + \epsilon_t,
   \]
   where \(\epsilon_t\) is a white noise process with variance \(\sigma^2\).
   
   Assume a prior distribution for \(\phi\) as \( \phi \sim \text{Normal}(0, 1) \) and for \(\sigma^2\) as \( \sigma^2 \sim \text{Inverse-Gamma}(2, 2)\).

   1. Derive the posterior distribution for \(\phi\) given a set of observed data \(X_1, X_2, \dots, X_n\).
   2. Use a Markov Chain Monte Carlo (MCMC) method to estimate the posterior distribution of \(\phi\) and \(\sigma^2\).
   3. Plot the posterior distribution of \(\phi\) and compare it with the prior distribution.

### Exercise 2: Frequentist Hypothesis Testing in Time Series
**Objective:** Perform hypothesis testing on time series data using frequentist methods.

1. **Problem Statement:** You are given a time series \(Y_t\) suspected to be stationary. Conduct a hypothesis test using the Augmented Dickey-Fuller (ADF) test to determine whether \(Y_t\) contains a unit root (i.e., whether it is non-stationary).

   1. State the null and alternative hypotheses for the ADF test.
   2. Perform the ADF test on the provided dataset.
   3. Interpret the results of the test, and discuss what the presence or absence of a unit root implies about the stationarity of the series.

### Exercise 3: Information Theory and Mutual Information
**Objective:** Apply information-theoretic measures to identify dependencies in time series data.

1. **Problem Statement:** Consider two time series \(X_t\) and \(Y_t\). You suspect that \(X_t\) has a lagged influence on \(Y_t\). Use mutual information to quantify the dependency between \(X_t\) and \(Y_t\).

   1. Define mutual information and explain how it can be used to detect dependencies between time series.
   2. Calculate the mutual information between \(X_t\) and \(Y_t\) for different lags \(k\) (e.g., \(k = 1, 2, \dots, 10\)).
   3. Determine the lag \(k\) that maximizes the mutual information, and discuss what this implies about the relationship between \(X_t\) and \(Y_t\).

### Exercise 4: Model Selection Using Information Criteria
**Objective:** Use information criteria to compare different time series models.

1. **Problem Statement:** You have fitted two models to a time series \(Z_t\): an AR(1) model and an AR(2) model. Compare the models using the Akaike Information Criterion (AIC) and the Bayesian Information Criterion (BIC).

   1. Calculate the AIC and BIC for both models.
   2. Interpret the values of AIC and BIC. Which model is preferred according to each criterion?
   3. Discuss the trade-offs between model complexity and goodness of fit as reflected by AIC and BIC.

### Exercise 5: Understanding Assumptions in Time Series Models
**Objective:** Evaluate the assumptions underlying different time series models and the impact of their violation.

1. **Problem Statement:** Choose a time series dataset and fit an ARIMA model to it. Discuss the assumptions behind the ARIMA model, specifically focusing on stationarity and the independence of errors.

   1. Test whether the time series is stationary. If not, apply a transformation (e.g., differencing) to achieve stationarity.
   2. Fit the ARIMA model to the transformed series and assess the independence of the residuals.
   3. Discuss how violations of the stationarity assumption might affect the model's forecasts.
