### **Exercise 4.1: Fitting an AR Model**
1. **Objective:** Fit an AR(2) model to a given time series dataset.
   
2. **Data:** Use a simulated or real-world time series dataset (e.g., daily temperature data).
   
3. **Tasks:**
   - Plot the time series to visually inspect for trends or patterns.
   - Fit an AR(2) model using the method of moments or Yule-Walker equations.
   - Use a Bayesian approach to fit the AR(2) model, specifying appropriate prior distributions for the parameters.
   - Compare the frequentist and Bayesian estimates of the model parameters.
   - Assess the residuals for autocorrelation using the autocorrelation function (ACF).
   
4. **Questions:**
   - How do the parameter estimates from the frequentist and Bayesian methods compare?
   - What does the residual analysis suggest about the model fit?
   - How would you modify the model if the residuals showed significant autocorrelation?

### **Exercise 4.2: Exploring Model Order in AR Models**
1. **Objective:** Determine the appropriate order \( p \) for an AR model.
   
2. **Data:** Use a real-world financial time series (e.g., stock prices or exchange rates).
   
3. **Tasks:**
   - Fit AR models with varying orders (e.g., AR(1) to AR(5)).
   - Use criteria such as the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) to select the optimal model order.
   - Validate the chosen model order using cross-validation.
   - Perform a posterior predictive check on the selected model.
   
4. **Questions:**
   - What order does AIC suggest? What about BIC?
   - How does cross-validation confirm or challenge the AIC/BIC selection?
   - What does the posterior predictive check reveal about model adequacy?

### **Exercise 4.3: Estimating MA Models**
1. **Objective:** Fit an MA(1) and MA(2) model to a given time series and compare them.
   
2. **Data:** Use a time series that exhibits short-term correlations (e.g., monthly sales data).
   
3. **Tasks:**
   - Fit MA(1) and MA(2) models to the dataset using maximum likelihood estimation.
   - Implement a Bayesian approach to fit the same models.
   - Compare the models using AIC, BIC, and posterior predictive checks.
   - Analyze the residuals for remaining autocorrelation.
   
4. **Questions:**
   - Which model (MA(1) or MA(2)) provides a better fit according to the criteria used?
   - How do the frequentist and Bayesian estimates differ?
   - What steps would you take if the residuals from both models showed significant autocorrelation?

### **Exercise 4.4: Forecasting with ARMA Models**
1. **Objective:** Perform multi-step forecasting using an ARMA(1,1) model.
   
2. **Data:** Use a time series with clear autoregressive and moving average components (e.g., inflation rates).
   
3. **Tasks:**
   - Fit an ARMA(1,1) model to the data.
   - Generate point forecasts and predictive intervals for the next 10 periods.
   - Use a Bayesian approach to generate a full posterior predictive distribution for the forecasts.
   - Compare the forecasts to actual future values (if available) or assess forecast accuracy using out-of-sample data.
   
4. **Questions:**
   - How do the point forecasts compare to the Bayesian predictive intervals?
   - What insights can be drawn from the posterior predictive distribution regarding forecast uncertainty?
   - How would you assess the accuracy of the ARMA model's forecasts?

### **Exercise 4.5: Model Selection and Diagnostics**
1. **Objective:** Evaluate and select the best model among AR, MA, and ARMA for a given time series.
   
2. **Data:** Use a complex time series dataset (e.g., economic indicators).
   
3. **Tasks:**
   - Fit AR, MA, and ARMA models to the dataset.
   - Use model selection criteria (AIC, BIC) and cross-validation to determine the best model.
   - Perform residual diagnostics on the chosen model to check for autocorrelation and model adequacy.
   - Interpret the results in terms of the underlying data generating process.
   
4. **Questions:**
   - Which model is selected as the best according to the criteria?
   - What do the residual diagnostics reveal about the chosen model?
   - How would you improve the model if diagnostics indicated problems?

---
