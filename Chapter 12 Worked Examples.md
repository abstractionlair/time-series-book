Here are the worked examples for Chapter 12, designed to bridge the gap between the main text and the exercises. These examples aim to provide a detailed walkthrough of the key forecasting concepts and methods introduced in the chapter, preparing students to tackle the exercises effectively.

### Worked Example 1: Understanding Probabilistic Forecasting with ARIMA Models
**Context:** Before diving into Exercise 12.2 on optimal forecasting theory, it's essential to understand how probabilistic forecasts work in practice and why they contain more information than point predictions.

1. **Theoretical Background:**
   - ARIMA models provide not just point forecasts but full probability distributions for future values
   - The uncertainty naturally increases with forecast horizon due to information decay
   - Optimal forecasts under squared loss are conditional expectations, but we also need uncertainty quantification

2. **Example:**
   Let's implement a complete ARIMA forecasting system with proper uncertainty quantification.
   
   **Step 1:** Generate synthetic data with known properties
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from statsmodels.tsa.arima.model import ARIMA
   from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
   from scipy import stats
   
   # Generate AR(1) process with known parameters
   np.random.seed(42)
   n = 200
   phi = 0.7  # AR parameter
   sigma = 1.0  # Innovation variance
   
   y = np.zeros(n)
   y[0] = np.random.normal(0, sigma / np.sqrt(1 - phi**2))
   for t in range(1, n):
       y[t] = phi * y[t-1] + np.random.normal(0, sigma)
   
   # Split into train/test
   train_size = 150
   y_train, y_test = y[:train_size], y[train_size:]
   ```
   
   **Step 2:** Fit ARIMA model and generate probabilistic forecasts
   ```python
   # Fit ARIMA(1,0,0) model
   model = ARIMA(y_train, order=(1, 0, 0))
   fitted_model = model.fit()
   
   # Generate forecasts with prediction intervals
   forecast_steps = len(y_test)
   forecast_result = fitted_model.get_forecast(steps=forecast_steps)
   
   # Extract point forecasts and confidence intervals
   point_forecast = forecast_result.predicted_mean
   conf_int = forecast_result.conf_int()
   
   # Theoretical forecast variance for AR(1)
   theoretical_var = np.zeros(forecast_steps)
   for h in range(forecast_steps):
       theoretical_var[h] = sigma**2 * (1 - phi**(2*(h+1))) / (1 - phi**2)
   
   print(f"Estimated AR parameter: {fitted_model.params[0]:.3f} (true: {phi})")
   print(f"Estimated variance: {fitted_model.sigma2:.3f} (true: {sigma**2})")
   ```
   
   **Step 3:** Evaluate probabilistic forecast quality
   ```python
   # Plot results
   plt.figure(figsize=(12, 8))
   
   # Time series plot
   plt.subplot(2, 2, 1)
   plt.plot(range(train_size), y_train, label='Training', color='blue')
   plt.plot(range(train_size, n), y_test, label='True', color='red')
   plt.plot(range(train_size, n), point_forecast, label='Forecast', color='green', linestyle='--')
   plt.fill_between(range(train_size, n), conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                    alpha=0.3, color='green', label='95% CI')
   plt.axvline(train_size, color='black', linestyle=':', alpha=0.7)
   plt.legend()
   plt.title('ARIMA Forecasts with Uncertainty')
   
   # Forecast error analysis
   plt.subplot(2, 2, 2)
   forecast_errors = y_test - point_forecast
   plt.hist(forecast_errors, bins=15, alpha=0.7, density=True)
   plt.axvline(0, color='red', linestyle='--')
   plt.xlabel('Forecast Error')
   plt.ylabel('Density')
   plt.title('Forecast Error Distribution')
   
   # Theoretical vs empirical forecast variance
   plt.subplot(2, 2, 3)
   empirical_var = np.cumsum(forecast_errors**2) / np.arange(1, len(forecast_errors)+1)
   plt.plot(theoretical_var[:len(empirical_var)], label='Theoretical')
   plt.plot(empirical_var, label='Empirical')
   plt.xlabel('Forecast Horizon')
   plt.ylabel('Forecast Variance')
   plt.title('Forecast Uncertainty Growth')
   plt.legend()
   
   # Probability integral transform (PIT) for calibration
   plt.subplot(2, 2, 4)
   forecast_std = np.sqrt(forecast_result.var_pred_mean)
   pit_values = stats.norm.cdf(forecast_errors, loc=0, scale=forecast_std)
   plt.hist(pit_values, bins=10, alpha=0.7, density=True)
   plt.axhline(1.0, color='red', linestyle='--', label='Perfect calibration')
   plt.xlabel('PIT Values')
   plt.ylabel('Density')
   plt.title('Forecast Calibration')
   plt.legend()
   
   plt.tight_layout()
   plt.show()
   
   # Quantitative evaluation
   mse = np.mean(forecast_errors**2)
   mae = np.mean(np.abs(forecast_errors))
   coverage = np.mean((y_test >= conf_int.iloc[:, 0]) & (y_test <= conf_int.iloc[:, 1]))
   
   print(f"Mean Squared Error: {mse:.3f}")
   print(f"Mean Absolute Error: {mae:.3f}")
   print(f"95% Coverage Probability: {coverage:.3f} (should be ~0.95)")
   ```

3. **Connection to Exercise 12.2:**
   This example demonstrates the key concepts needed for Exercise 12.2: how forecast uncertainty grows with horizon, why conditional expectations are optimal under squared loss, and how to evaluate probabilistic forecast quality through calibration.

### Worked Example 2: Ensemble Forecasting Methods
**Context:** Exercise 12.4 involves implementing ensemble methods. This example shows how to combine multiple forecasting approaches to improve robustness and accuracy.

1. **Theoretical Background:**
   - Ensemble methods reduce forecast variance by averaging multiple predictors
   - Different models capture different aspects of the time series dynamics
   - Model averaging can be done with equal weights or using Bayesian model averaging

2. **Example:**
   Let's build a comprehensive ensemble forecasting system.
   
   **Step 1:** Implement multiple forecasting methods
   ```python
   from statsmodels.tsa.holtwinters import ExponentialSmoothing
   from statsmodels.tsa.seasonal import seasonal_decompose
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.linear_model import Ridge
   
   class ForecastEnsemble:
       def __init__(self):
           self.models = {}
           self.weights = None
           
       def add_arima_model(self, order=(1,1,1)):
           """Add ARIMA model to ensemble"""
           def fit_predict(y_train, steps):
               model = ARIMA(y_train, order=order).fit()
               forecast = model.get_forecast(steps=steps)
               return forecast.predicted_mean.values, np.sqrt(forecast.var_pred_mean.values)
           
           self.models['ARIMA'] = fit_predict
           
       def add_ets_model(self):
           """Add Exponential Smoothing model"""
           def fit_predict(y_train, steps):
               model = ExponentialSmoothing(y_train, trend='add', seasonal=None).fit()
               forecast = model.forecast(steps)
               # Approximate prediction intervals
               residuals = model.resid[~np.isnan(model.resid)]
               pred_std = np.std(residuals) * np.sqrt(np.arange(1, steps+1))
               return forecast, pred_std
           
           self.models['ETS'] = fit_predict
           
       def add_ml_model(self, model_type='rf', lags=5):
           """Add machine learning model with lagged features"""
           def fit_predict(y_train, steps):
               # Create lagged features
               X, y = self._create_features(y_train, lags)
               
               if model_type == 'rf':
                   model = RandomForestRegressor(n_estimators=100, random_state=42)
               else:
                   model = Ridge(alpha=1.0)
               
               model.fit(X[lags:], y[lags:])
               
               # Multi-step ahead forecasting
               forecasts = []
               last_obs = y_train[-lags:].copy()
               
               for step in range(steps):
                   next_pred = model.predict([last_obs])[0]
                   forecasts.append(next_pred)
                   last_obs = np.append(last_obs[1:], next_pred)
               
               # Rough uncertainty estimate from residuals
               train_pred = model.predict(X[lags:])
               residuals = y[lags:] - train_pred
               pred_std = np.std(residuals) * np.ones(steps)
               
               return np.array(forecasts), pred_std
           
           self.models[f'ML_{model_type}'] = fit_predict
           
       def _create_features(self, y, lags):
           """Create lagged features for ML models"""
           n = len(y)
           X = np.zeros((n, lags))
           for i in range(lags):
               X[i:, i] = y[:n-i]
           return X, y
           
       def fit_weights(self, y_train, validation_split=0.2):
           """Learn optimal ensemble weights using validation set"""
           split_idx = int(len(y_train) * (1 - validation_split))
           train_data = y_train[:split_idx]
           val_data = y_train[split_idx:]
           val_steps = len(val_data)
           
           # Get predictions from each model
           predictions = {}
           for name, model_func in self.models.items():
               pred_mean, pred_std = model_func(train_data, val_steps)
               predictions[name] = pred_mean
           
           # Optimize weights to minimize validation error
           from scipy.optimize import minimize
           
           def objective(weights):
               weights = weights / np.sum(weights)  # Normalize
               ensemble_pred = sum(w * pred for w, pred in zip(weights, predictions.values()))
               return np.mean((val_data - ensemble_pred)**2)
           
           n_models = len(self.models)
           initial_weights = np.ones(n_models) / n_models
           bounds = [(0, 1) for _ in range(n_models)]
           constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
           
           result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
           self.weights = result.x
           
           # Report weights
           print("Optimal ensemble weights:")
           for name, weight in zip(self.models.keys(), self.weights):
               print(f"  {name}: {weight:.3f}")
               
       def forecast(self, y_train, steps, return_individual=False):
           """Generate ensemble forecast"""
           individual_forecasts = {}
           individual_stds = {}
           
           for name, model_func in self.models.items():
               pred_mean, pred_std = model_func(y_train, steps)
               individual_forecasts[name] = pred_mean
               individual_stds[name] = pred_std
           
           # Weighted ensemble
           if self.weights is None:
               weights = np.ones(len(self.models)) / len(self.models)
           else:
               weights = self.weights
               
           ensemble_mean = sum(w * pred for w, pred in zip(weights, individual_forecasts.values()))
           
           # Ensemble uncertainty (simplified)
           ensemble_var = sum(w**2 * std**2 for w, std in zip(weights, individual_stds.values()))
           ensemble_std = np.sqrt(ensemble_var)
           
           if return_individual:
               return ensemble_mean, ensemble_std, individual_forecasts, individual_stds
           else:
               return ensemble_mean, ensemble_std
   ```
   
   **Step 2:** Test ensemble on real data
   ```python
   # Generate more complex synthetic data
   np.random.seed(123)
   n = 300
   t = np.arange(n)
   
   # Trend + seasonality + noise + regime change
   trend = 0.02 * t
   seasonal = 2 * np.sin(2 * np.pi * t / 12) + np.sin(2 * np.pi * t / 4)
   noise = np.random.normal(0, 0.5, n)
   regime_change = np.where(t > 200, 1.5, 0)  # Structural break
   
   y_complex = trend + seasonal + noise + regime_change
   
   # Split data
   train_size = 250
   y_train = y_complex[:train_size]
   y_test = y_complex[train_size:]
   
   # Build ensemble
   ensemble = ForecastEnsemble()
   ensemble.add_arima_model(order=(2,1,2))
   ensemble.add_ets_model()
   ensemble.add_ml_model('rf')
   ensemble.add_ml_model('ridge')
   
   # Fit ensemble weights
   ensemble.fit_weights(y_train)
   
   # Generate forecasts
   forecast_steps = len(y_test)
   ensemble_mean, ensemble_std, individual_forecasts, individual_stds = ensemble.forecast(
       y_train, forecast_steps, return_individual=True
   )
   ```
   
   **Step 3:** Compare ensemble vs individual models
   ```python
   # Plot results
   plt.figure(figsize=(15, 10))
   
   # Main forecast plot
   plt.subplot(2, 2, 1)
   plt.plot(range(train_size), y_train, label='Training', alpha=0.7)
   plt.plot(range(train_size, train_size + forecast_steps), y_test, 
            label='True', color='red', linewidth=2)
   
   # Individual model forecasts
   colors = ['green', 'blue', 'orange', 'purple']
   for i, (name, forecast) in enumerate(individual_forecasts.items()):
       plt.plot(range(train_size, train_size + forecast_steps), forecast, 
                label=name, color=colors[i], alpha=0.6)
   
   # Ensemble forecast
   plt.plot(range(train_size, train_size + forecast_steps), ensemble_mean,
            label='Ensemble', color='black', linewidth=2)
   plt.fill_between(range(train_size, train_size + forecast_steps),
                    ensemble_mean - 1.96*ensemble_std,
                    ensemble_mean + 1.96*ensemble_std,
                    alpha=0.2, color='black')
   
   plt.axvline(train_size, color='gray', linestyle='--', alpha=0.7)
   plt.legend()
   plt.title('Ensemble Forecasting Results')
   
   # Error comparison
   plt.subplot(2, 2, 2)
   errors = {}
   for name, forecast in individual_forecasts.items():
       errors[name] = np.mean((y_test - forecast)**2)
   errors['Ensemble'] = np.mean((y_test - ensemble_mean)**2)
   
   names = list(errors.keys())
   mse_values = list(errors.values())
   plt.bar(names, mse_values)
   plt.ylabel('Mean Squared Error')
   plt.title('Model Comparison')
   plt.xticks(rotation=45)
   
   # Error evolution over forecast horizon
   plt.subplot(2, 2, 3)
   cumulative_errors = {}
   for name, forecast in individual_forecasts.items():
       cumulative_errors[name] = np.cumsum((y_test - forecast)**2) / np.arange(1, len(y_test)+1)
   cumulative_errors['Ensemble'] = np.cumsum((y_test - ensemble_mean)**2) / np.arange(1, len(y_test)+1)
   
   for name, cum_error in cumulative_errors.items():
       plt.plot(cum_error, label=name)
   plt.xlabel('Forecast Horizon')
   plt.ylabel('Cumulative MSE')
   plt.title('Error Evolution')
   plt.legend()
   
   # Residual analysis
   plt.subplot(2, 2, 4)
   ensemble_residuals = y_test - ensemble_mean
   plt.scatter(ensemble_mean, ensemble_residuals, alpha=0.6)
   plt.axhline(0, color='red', linestyle='--')
   plt.xlabel('Fitted Values')
   plt.ylabel('Residuals')
   plt.title('Residual Plot')
   
   plt.tight_layout()
   plt.show()
   
   # Print summary
   print("\nForecast Performance Summary:")
   for name, mse in errors.items():
       print(f"{name:10s}: MSE = {mse:.4f}")
   ```

3. **Connection to Exercise 12.4:**
   This example provides the foundation for implementing sophisticated ensemble methods, showing how to combine different model types, optimize weights, and evaluate ensemble performance - all key components of Exercise 12.4.

### Worked Example 3: Long-term Scenario Analysis and Forecasting
**Context:** This example prepares students for exercises involving long-term forecasting under uncertainty, scenario analysis, and the challenges of forecasting beyond the typical horizon.

1. **Theoretical Background:**
   - Long-term forecasts face fundamental challenges due to information decay and structural changes
   - Scenario analysis provides alternative futures rather than point predictions
   - Fan charts visualize forecast uncertainty over multiple horizons

2. **Example:**
   Let's implement a comprehensive long-term forecasting system with scenario analysis.
   
   **Step 1:** Implement scenario generation framework
   ```python
   import numpy as np
   from scipy import stats
   from statsmodels.tsa.statespace.sarimax import SARIMAX
   
   class ScenarioForecaster:
       def __init__(self, base_model=None):
           self.base_model = base_model
           self.scenarios = {}
           
       def fit_base_model(self, y_train, model_type='arima', **kwargs):
           """Fit base forecasting model"""
           if model_type == 'arima':
               order = kwargs.get('order', (2,1,2))
               self.base_model = ARIMA(y_train, order=order).fit()
           elif model_type == 'sarima':
               order = kwargs.get('order', (1,1,1))
               seasonal_order = kwargs.get('seasonal_order', (1,1,1,12))
               self.base_model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order).fit()
           
       def add_scenario(self, name, shock_function, probability=None):
           """Add a scenario with specific shocks"""
           self.scenarios[name] = {
               'shock_function': shock_function,
               'probability': probability
           }
           
       def generate_long_term_forecasts(self, steps, n_simulations=1000, confidence_levels=[0.1, 0.25, 0.4]):
           """Generate long-term forecasts with scenarios"""
           
           # Base model simulation
           base_simulations = []
           for _ in range(n_simulations):
               sim = self.base_model.simulate(nsimulations=steps, initial_state=self.base_model.states.filtered[-1])
               base_simulations.append(sim)
           
           base_simulations = np.array(base_simulations)
           
           # Scenario simulations
           scenario_results = {}
           for scenario_name, scenario_info in self.scenarios.items():
               scenario_sims = []
               shock_func = scenario_info['shock_function']
               
               for _ in range(n_simulations):
                   # Start with base simulation
                   base_sim = self.base_model.simulate(nsimulations=steps, 
                                                      initial_state=self.base_model.states.filtered[-1])
                   
                   # Apply scenario shocks
                   shocked_sim = shock_func(base_sim, steps)
                   scenario_sims.append(shocked_sim)
               
               scenario_results[scenario_name] = np.array(scenario_sims)
           
           # Compute forecast quantiles
           def compute_quantiles(simulations, levels):
               quantiles = {}
               quantiles['mean'] = np.mean(simulations, axis=0)
               quantiles['median'] = np.median(simulations, axis=0)
               
               for level in levels:
                   lower_q = level / 2
                   upper_q = 1 - level / 2
                   quantiles[f'lower_{int(level*100)}'] = np.quantile(simulations, lower_q, axis=0)
                   quantiles[f'upper_{int(level*100)}'] = np.quantile(simulations, upper_q, axis=0)
               
               return quantiles
           
           results = {
               'baseline': compute_quantiles(base_simulations, confidence_levels),
               'scenarios': {name: compute_quantiles(sims, confidence_levels) 
                           for name, sims in scenario_results.items()}
           }
           
           return results, base_simulations, scenario_results
   ```
   
   **Step 2:** Define realistic scenarios
   ```python
   # Create synthetic economic data with trend and cycles
   np.random.seed(42)
   n = 120  # 10 years of monthly data
   t = np.arange(n)
   
   # Economic time series with trend, cycle, and shocks
   trend = 0.02 * t  # Long-term growth
   cycle = 0.5 * np.sin(2 * np.pi * t / 48) + 0.3 * np.sin(2 * np.pi * t / 24)  # Business cycles
   irregular = np.random.normal(0, 0.2, n)
   
   # Add some historical shocks
   shocks = np.zeros(n)
   shocks[40:45] = -1.5  # Recession
   shocks[80:82] = -0.8  # Market correction
   
   y_economic = 100 + np.cumsum(trend + cycle + irregular + shocks)
   
   # Split data
   train_size = 100
   y_train = y_economic[:train_size]
   y_test = y_economic[train_size:]
   
   # Initialize scenario forecaster
   forecaster = ScenarioForecaster()
   forecaster.fit_base_model(y_train, model_type='arima', order=(2,1,1))
   
   # Define scenarios
   def recession_scenario(base_forecast, steps):
       """Recession hits in months 6-18"""
       shocked = base_forecast.copy()
       recession_start = min(6, len(shocked))
       recession_end = min(18, len(shocked))
       
       # Gradual decline and recovery
       for i in range(recession_start, recession_end):
           intensity = np.sin(np.pi * (i - recession_start) / (recession_end - recession_start))
           shocked[i:] -= 2.0 * intensity  # Cumulative negative shock
       
       return shocked
   
   def productivity_boom_scenario(base_forecast, steps):
       """Productivity boom leads to sustained higher growth"""
       shocked = base_forecast.copy()
       boost_start = min(3, len(shocked))
       
       for i in range(boost_start, len(shocked)):
           shocked[i:] += 0.1 * (i - boost_start + 1)  # Cumulative positive shock
       
       return shocked
   
   def volatility_regime_scenario(base_forecast, steps):
       """Increased volatility regime"""
       shocked = base_forecast.copy()
       # Add time-varying volatility shocks
       vol_shocks = np.random.normal(0, 0.5, len(shocked))  # Higher volatility
       shocked += np.cumsum(vol_shocks)
       return shocked
   
   # Add scenarios
   forecaster.add_scenario('Recession', recession_scenario, probability=0.3)
   forecaster.add_scenario('Productivity Boom', productivity_boom_scenario, probability=0.2)
   forecaster.add_scenario('High Volatility', volatility_regime_scenario, probability=0.15)
   
   # Generate long-term forecasts
   forecast_steps = 24  # 2 years ahead
   results, base_sims, scenario_sims = forecaster.generate_long_term_forecasts(
       forecast_steps, n_simulations=500
   )
   ```
   
   **Step 3:** Create comprehensive visualization and analysis
   ```python
   # Create fan chart visualization
   plt.figure(figsize=(16, 12))
   
   # Main forecast plot with fan chart
   plt.subplot(2, 3, 1)
   forecast_index = range(train_size, train_size + forecast_steps)
   
   # Plot historical data
   plt.plot(range(train_size), y_train, 'b-', label='Historical', linewidth=2)
   if len(y_test) > 0:
       plt.plot(range(train_size, train_size + len(y_test)), y_test, 
                'r-', label='Actual (test)', linewidth=2)
   
   # Fan chart for baseline
   baseline = results['baseline']
   plt.plot(forecast_index, baseline['mean'], 'g-', label='Baseline Mean', linewidth=2)
   
   # Confidence bands
   colors = ['lightblue', 'lightgreen', 'lightyellow']
   alphas = [0.3, 0.4, 0.5]
   for i, level in enumerate([10, 25, 40]):
       plt.fill_between(forecast_index,
                       baseline[f'lower_{level}'], baseline[f'upper_{level}'],
                       color=colors[i], alpha=alphas[i], 
                       label=f'{100-level}% CI')
   
   plt.axvline(train_size, color='gray', linestyle='--', alpha=0.7)
   plt.legend()
   plt.title('Long-term Forecast Fan Chart')
   plt.ylabel('Value')
   
   # Scenario comparison
   plt.subplot(2, 3, 2)
   plt.plot(forecast_index, baseline['mean'], 'k-', label='Baseline', linewidth=2)
   
   scenario_colors = ['red', 'green', 'orange']
   for i, (scenario_name, scenario_data) in enumerate(results['scenarios'].items()):
       plt.plot(forecast_index, scenario_data['mean'], 
                color=scenario_colors[i], label=scenario_name, linewidth=2)
       plt.fill_between(forecast_index,
                       scenario_data['lower_25'], scenario_data['upper_25'],
                       color=scenario_colors[i], alpha=0.2)
   
   plt.axvline(train_size, color='gray', linestyle='--', alpha=0.7)
   plt.legend()
   plt.title('Scenario Analysis')
   plt.ylabel('Value')
   
   # Probability distributions at different horizons
   horizons = [6, 12, 24]
   for i, horizon in enumerate(horizons):
       plt.subplot(2, 3, 4 + i)
       
       # Distribution at specific horizon
       if horizon <= forecast_steps:
           # Baseline distribution
           baseline_values = base_sims[:, horizon-1]
           plt.hist(baseline_values, bins=30, alpha=0.5, density=True, 
                   label='Baseline', color='blue')
           
           # Scenario distributions
           for j, (scenario_name, scenario_data) in enumerate(scenario_sims.items()):
               scenario_values = scenario_data[:, horizon-1]
               plt.hist(scenario_values, bins=30, alpha=0.5, density=True,
                       label=scenario_name, color=scenario_colors[j])
           
           plt.axvline(y_train[-1] if horizon == 6 else baseline['mean'][horizon-1], 
                      color='red', linestyle='--', label='Current/Expected')
           plt.legend()
           plt.title(f'Forecast Distribution at {horizon} months')
           plt.xlabel('Value')
           plt.ylabel('Density')
   
   # Risk analysis
   plt.subplot(2, 3, 3)
   
   # Value at Risk analysis
   var_levels = [0.05, 0.1, 0.25]
   baseline_var = []
   recession_var = []
   
   for level in var_levels:
       baseline_var.append(np.quantile(base_sims, level, axis=0))
       if 'Recession' in scenario_sims:
           recession_var.append(np.quantile(scenario_sims['Recession'], level, axis=0))
   
   for i, level in enumerate(var_levels):
       plt.plot(forecast_index, baseline_var[i], 
               label=f'Baseline VaR {int(level*100)}%', linestyle='--')
       if recession_var:
           plt.plot(forecast_index, recession_var[i],
                   label=f'Recession VaR {int(level*100)}%', linestyle=':')
   
   plt.axvline(train_size, color='gray', linestyle='--', alpha=0.7)
   plt.legend()
   plt.title('Value at Risk Analysis')
   plt.ylabel('VaR Level')
   
   plt.tight_layout()
   plt.show()
   
   # Quantitative scenario analysis
   print("Long-term Forecast Analysis Summary:")
   print("=" * 50)
   
   # Expected values and ranges
   final_horizon = forecast_steps - 1
   print(f"\nForecast at {forecast_steps} months:")
   print(f"Baseline expectation: {baseline['mean'][final_horizon]:.2f}")
   print(f"50% confidence interval: [{baseline['lower_25'][final_horizon]:.2f}, {baseline['upper_25'][final_horizon]:.2f}]")
   print(f"90% confidence interval: [{baseline['lower_10'][final_horizon]:.2f}, {baseline['upper_10'][final_horizon]:.2f}]")
   
   print("\nScenario Analysis:")
   for scenario_name, scenario_data in results['scenarios'].items():
       expected = scenario_data['mean'][final_horizon]
       lower = scenario_data['lower_25'][final_horizon]
       upper = scenario_data['upper_25'][final_horizon]
       print(f"{scenario_name:20s}: {expected:8.2f} [{lower:8.2f}, {upper:8.2f}]")
   
   # Tail risk analysis
   print(f"\nTail Risk Analysis (at {forecast_steps} months):")
   baseline_final = base_sims[:, final_horizon]
   print(f"Probability of decline > 5%: {np.mean(baseline_final < y_train[-1] * 0.95):.3f}")
   print(f"Probability of decline > 10%: {np.mean(baseline_final < y_train[-1] * 0.90):.3f}")
   
   if 'Recession' in scenario_sims:
       recession_final = scenario_sims['Recession'][:, final_horizon]
       print(f"Under recession scenario:")
       print(f"  Probability of decline > 5%: {np.mean(recession_final < y_train[-1] * 0.95):.3f}")
       print(f"  Probability of decline > 10%: {np.mean(recession_final < y_train[-1] * 0.90):.3f}")
   ```

3. **Connection to Chapter Exercises:**
   This example provides the framework for tackling complex forecasting scenarios involving uncertainty quantification, multiple scenarios, and long-term projections - essential skills for exercises dealing with real-world forecasting challenges.

### Worked Example 4: Advanced Forecasting Diagnostics and Model Selection
**Context:** This example prepares students for sophisticated model evaluation and selection procedures, crucial for exercises involving forecast comparison and diagnostic checking.

1. **Theoretical Background:**
   - Forecast evaluation requires multiple criteria beyond simple accuracy measures
   - Diagnostic tests check model assumptions and identify areas for improvement
   - Information criteria help balance model fit against complexity

2. **Example:**
   Let's implement a comprehensive forecasting diagnostic system.
   
   **Step 1:** Comprehensive diagnostic framework
   ```python
   from scipy import stats
   from statsmodels.stats.diagnostic import acorr_ljungbox
   from statsmodels.stats.stattools import jarque_bera
   from arch.unitroot import ADF
   
   class ForecastDiagnostics:
       def __init__(self):
           self.results = {}
           
       def evaluate_model_fit(self, model, y_train):
           """Comprehensive model fit evaluation"""
           diagnostics = {}
           
           # Basic model statistics
           diagnostics['aic'] = model.aic
           diagnostics['bic'] = model.bic
           diagnostics['log_likelihood'] = model.llf
           
           # Residual analysis
           residuals = model.resid
           standardized_residuals = residuals / np.std(residuals)
           
           # Normality tests
           jb_stat, jb_pvalue = jarque_bera(residuals)
           diagnostics['jarque_bera_stat'] = jb_stat
           diagnostics['jarque_bera_pvalue'] = jb_pvalue
           diagnostics['residuals_normal'] = jb_pvalue > 0.05
           
           # Autocorrelation in residuals
           ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
           diagnostics['ljung_box_pvalue'] = ljung_box['lb_pvalue'].iloc[-1]
           diagnostics['residuals_uncorrelated'] = ljung_box['lb_pvalue'].iloc[-1] > 0.05
           
           # Heteroskedasticity test (simplified)
           abs_residuals = np.abs(residuals)
           het_corr, het_pvalue = stats.pearsonr(abs_residuals[1:], abs_residuals[:-1])
           diagnostics['heteroskedasticity_pvalue'] = het_pvalue
           diagnostics['homoskedastic'] = het_pvalue > 0.05
           
           # Residual statistics
           diagnostics['residual_mean'] = np.mean(residuals)
           diagnostics['residual_std'] = np.std(residuals)
           diagnostics['residual_skewness'] = stats.skew(residuals)
           diagnostics['residual_kurtosis'] = stats.kurtosis(residuals)
           
           return diagnostics
           
       def evaluate_forecast_accuracy(self, forecasts, actuals, forecast_std=None):
           """Multiple forecast accuracy measures"""
           errors = actuals - forecasts
           abs_errors = np.abs(errors)
           pct_errors = 100 * errors / actuals
           
           accuracy = {}
           
           # Basic accuracy measures
           accuracy['mae'] = np.mean(abs_errors)
           accuracy['mse'] = np.mean(errors**2)
           accuracy['rmse'] = np.sqrt(accuracy['mse'])
           accuracy['mape'] = np.mean(np.abs(pct_errors))
           
           # Scale-independent measures
           naive_forecast = np.full_like(forecasts, actuals[0])
           naive_errors = actuals - naive_forecast
           accuracy['mase'] = accuracy['mae'] / np.mean(np.abs(naive_errors))
           
           # Directional accuracy
           if len(actuals) > 1:
               actual_directions = np.sign(np.diff(actuals))
               forecast_directions = np.sign(np.diff(forecasts))
               accuracy['directional_accuracy'] = np.mean(actual_directions == forecast_directions)
           
           # Probabilistic measures (if uncertainty provided)
           if forecast_std is not None:
               # Continuous Ranked Probability Score (simplified)
               z_scores = errors / forecast_std
               accuracy['crps'] = np.mean(forecast_std * (z_scores * (2 * stats.norm.cdf(z_scores) - 1) + 
                                                         2 * stats.norm.pdf(z_scores)))
               
               # Calibration check
               pit_values = stats.norm.cdf(errors, loc=0, scale=forecast_std)
               ks_stat, ks_pvalue = stats.kstest(pit_values, 'uniform')
               accuracy['calibration_ks_pvalue'] = ks_pvalue
               accuracy['well_calibrated'] = ks_pvalue > 0.05
           
           return accuracy
           
       def rolling_forecast_evaluation(self, y_data, model_func, window_size=50, 
                                     forecast_horizon=1, step_size=1):
           """Rolling window forecast evaluation"""
           n = len(y_data)
           results = []
           
           for start in range(0, n - window_size - forecast_horizon, step_size):
               end = start + window_size
               train_data = y_data[start:end]
               test_data = y_data[end:end + forecast_horizon]
               
               try:
                   # Fit model and forecast
                   model = model_func(train_data)
                   if hasattr(model, 'get_forecast'):
                       forecast_result = model.get_forecast(steps=forecast_horizon)
                       forecast = forecast_result.predicted_mean.values
                       forecast_std = np.sqrt(forecast_result.var_pred_mean.values)
                   else:
                       forecast = model.forecast(forecast_horizon)
                       forecast_std = None
                   
                   # Evaluate
                   accuracy = self.evaluate_forecast_accuracy(forecast, test_data, forecast_std)
                   fit_diagnostics = self.evaluate_model_fit(model, train_data)
                   
                   result = {
                       'start_date': start,
                       'end_date': end,
                       'forecast': forecast,
                       'actual': test_data,
                       **accuracy,
                       **{f'fit_{k}': v for k, v in fit_diagnostics.items()}
                   }
                   results.append(result)
                   
               except Exception as e:
                   print(f"Error at window {start}: {e}")
                   continue
           
           return pd.DataFrame(results)
           
       def model_comparison(self, models_dict, y_train, y_test, forecast_steps):
           """Compare multiple models"""
           comparison = {}
           
           for name, model_func in models_dict.items():
               try:
                   # Fit model
                   model = model_func(y_train)
                   
                   # Generate forecasts
                   if hasattr(model, 'get_forecast'):
                       forecast_result = model.get_forecast(steps=forecast_steps)
                       forecast = forecast_result.predicted_mean.values
                       forecast_std = np.sqrt(forecast_result.var_pred_mean.values)
                   else:
                       forecast = model.forecast(forecast_steps)
                       forecast_std = None
                   
                   # Evaluate
                   fit_diag = self.evaluate_model_fit(model, y_train)
                   forecast_acc = self.evaluate_forecast_accuracy(forecast[:len(y_test)], 
                                                                 y_test, 
                                                                 forecast_std[:len(y_test)] if forecast_std is not None else None)
                   
                   comparison[name] = {
                       'model': model,
                       'forecast': forecast,
                       'forecast_std': forecast_std,
                       **fit_diag,
                       **forecast_acc
                   }
                   
               except Exception as e:
                   print(f"Error with model {name}: {e}")
                   comparison[name] = {'error': str(e)}
           
           return comparison
   ```
   
   **Step 2:** Apply comprehensive diagnostics
   ```python
   # Generate test data with known issues
   np.random.seed(42)
   n = 200
   
   # Create data with multiple characteristics to test diagnostics
   t = np.arange(n)
   trend = 0.05 * t
   seasonal = np.sin(2 * np.pi * t / 12) + 0.5 * np.sin(2 * np.pi * t / 4)
   
   # Add heteroskedasticity
   volatility = 0.2 + 0.1 * np.abs(seasonal)
   noise = np.random.normal(0, volatility)
   
   # Add structural break
   structural_break = np.where(t > 120, 2.0, 0)
   
   y_complex = trend + seasonal + noise + structural_break
   
   # Split data
   train_size = 150
   y_train = y_complex[:train_size]
   y_test = y_complex[train_size:]
   
   # Initialize diagnostics
   diagnostics = ForecastDiagnostics()
   
   # Define models for comparison
   def fit_arima_211(y):
       return ARIMA(y, order=(2,1,1)).fit(disp=False)
   
   def fit_arima_312(y):
       return ARIMA(y, order=(3,1,2)).fit(disp=False)
   
   def fit_ets(y):
       return ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=12).fit(disp=False)
   
   models = {
       'ARIMA(2,1,1)': fit_arima_211,
       'ARIMA(3,1,2)': fit_arima_312,
       'ETS(A,A,A)': fit_ets
   }
   
   # Model comparison
   comparison_results = diagnostics.model_comparison(models, y_train, y_test, len(y_test))
   
   # Rolling evaluation for best model
   rolling_results = diagnostics.rolling_forecast_evaluation(
       y_complex, fit_arima_211, window_size=60, forecast_horizon=5
   )
   ```
   
   **Step 3:** Comprehensive visualization and reporting
   ```python
   # Create diagnostic plots
   fig, axes = plt.subplots(3, 4, figsize=(20, 15))
   
   # Model comparison summary
   ax = axes[0, 0]
   model_names = list(comparison_results.keys())
   model_metrics = ['rmse', 'mae', 'mape', 'aic']
   
   for i, metric in enumerate(model_metrics):
       values = [comparison_results[name].get(metric, np.nan) for name in model_names]
       ax.bar([f"{name}\n{metric.upper()}" for name in model_names], values, alpha=0.7)
   ax.set_title('Model Comparison - Key Metrics')
   ax.tick_params(axis='x', rotation=45)
   
   # Forecast comparison
   ax = axes[0, 1]
   ax.plot(range(train_size), y_train, label='Train', color='blue', alpha=0.7)
   ax.plot(range(train_size, train_size + len(y_test)), y_test, label='Actual', color='red', linewidth=2)
   
   colors = ['green', 'orange', 'purple']
   for i, (name, results) in enumerate(comparison_results.items()):
       if 'error' not in results:
           forecast = results['forecast'][:len(y_test)]
           ax.plot(range(train_size, train_size + len(forecast)), forecast,
                  label=name, color=colors[i % len(colors)], linestyle='--')
   
   ax.axvline(train_size, color='gray', linestyle=':', alpha=0.7)
   ax.legend()
   ax.set_title('Forecast Comparison')
   
   # Residual diagnostics for best model
   best_model_name = min(comparison_results.keys(), 
                        key=lambda x: comparison_results[x].get('aic', float('inf')))
   best_model = comparison_results[best_model_name]['model']
   residuals = best_model.resid
   
   # Q-Q plot
   ax = axes[0, 2]
   stats.probplot(residuals, dist="norm", plot=ax)
   ax.set_title(f'Q-Q Plot - {best_model_name}')
   
   # ACF of residuals
   ax = axes[0, 3]
   from statsmodels.graphics.tsaplots import plot_acf
   plot_acf(residuals, ax=ax, lags=20)
   ax.set_title('Residual Autocorrelation')
   
   # Rolling performance metrics
   ax = axes[1, 0]
   rolling_results['rmse'].rolling(window=10).mean().plot(ax=ax, label='RMSE (10-period MA)')
   rolling_results['mae'].rolling(window=10).mean().plot(ax=ax, label='MAE (10-period MA)')
   ax.legend()
   ax.set_title('Rolling Forecast Performance')
   ax.set_xlabel('Forecast Period')
   
   # Model stability
   ax = axes[1, 1]
   if 'fit_aic' in rolling_results.columns:
       rolling_results['fit_aic'].plot(ax=ax)
   ax.set_title('Model Fit Stability (AIC)')
   ax.set_xlabel('Forecast Period')
   
   # Forecast error distribution
   ax = axes[1, 2]
   best_forecast = comparison_results[best_model_name]['forecast'][:len(y_test)]
   forecast_errors = y_test - best_forecast
   ax.hist(forecast_errors, bins=15, alpha=0.7, density=True)
   ax.axvline(0, color='red', linestyle='--')
   
   # Overlay normal distribution for comparison
   mu, sigma = stats.norm.fit(forecast_errors)
   x = np.linspace(forecast_errors.min(), forecast_errors.max(), 100)
   ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label=f'Normal fit (μ={mu:.2f}, σ={sigma:.2f})')
   ax.legend()
   ax.set_title('Forecast Error Distribution')
   
   # Calibration plot (if probabilistic forecasts available)
   ax = axes[1, 3]
   if 'forecast_std' in comparison_results[best_model_name] and comparison_results[best_model_name]['forecast_std'] is not None:
       forecast_std = comparison_results[best_model_name]['forecast_std'][:len(y_test)]
       pit_values = stats.norm.cdf(forecast_errors, loc=0, scale=forecast_std)
       ax.hist(pit_values, bins=10, alpha=0.7, density=True)
       ax.axhline(1.0, color='red', linestyle='--', label='Perfect calibration')
       ax.legend()
       ax.set_title('Forecast Calibration (PIT)')
   else:
       ax.text(0.5, 0.5, 'Probabilistic forecasts\nnot available', 
              ha='center', va='center', transform=ax.transAxes)
       ax.set_title('Calibration Analysis')
   
   # Directional accuracy over time
   ax = axes[2, 0]
   if len(rolling_results) > 0 and 'directional_accuracy' in rolling_results.columns:
       rolling_results['directional_accuracy'].rolling(window=10).mean().plot(ax=ax)
       ax.axhline(0.5, color='red', linestyle='--', label='Random guess')
       ax.legend()
   ax.set_title('Rolling Directional Accuracy')
   ax.set_xlabel('Forecast Period')
   
   # Information criteria evolution
   ax = axes[2, 1]
   if len(rolling_results) > 0:
       if 'fit_aic' in rolling_results.columns:
           rolling_results['fit_aic'].plot(ax=ax, label='AIC')
       if 'fit_bic' in rolling_results.columns:
           rolling_results['fit_bic'].plot(ax=ax, label='BIC')
       ax.legend()
   ax.set_title('Information Criteria Evolution')
   ax.set_xlabel('Forecast Period')
   
   # Forecast uncertainty bands
   ax = axes[2, 2]
   ax.plot(range(train_size, train_size + len(y_test)), y_test, 'r-', label='Actual', linewidth=2)
   ax.plot(range(train_size, train_size + len(best_forecast)), best_forecast, 'g--', label='Forecast')
   
   if 'forecast_std' in comparison_results[best_model_name] and comparison_results[best_model_name]['forecast_std'] is not None:
       forecast_std = comparison_results[best_model_name]['forecast_std'][:len(y_test)]
       ax.fill_between(range(train_size, train_size + len(best_forecast)),
                      best_forecast - 1.96*forecast_std,
                      best_forecast + 1.96*forecast_std,
                      alpha=0.3, color='green')
   
   ax.legend()
   ax.set_title('Forecast with Uncertainty Bands')
   
   # Model diagnostic summary
   ax = axes[2, 3]
   ax.axis('off')
   
   # Create diagnostic summary text
   summary_text = f"Model Diagnostic Summary\n{'-'*25}\n\n"
   summary_text += f"Best Model: {best_model_name}\n\n"
   
   best_results = comparison_results[best_model_name]
   summary_text += f"Fit Diagnostics:\n"
   summary_text += f"  AIC: {best_results.get('aic', 'N/A'):.2f}\n"
   summary_text += f"  Residuals Normal: {best_results.get('residuals_normal', 'N/A')}\n"
   summary_text += f"  Residuals Uncorr: {best_results.get('residuals_uncorrelated', 'N/A')}\n"
   summary_text += f"  Homoskedastic: {best_results.get('homoskedastic', 'N/A')}\n\n"
   
   summary_text += f"Forecast Accuracy:\n"
   summary_text += f"  RMSE: {best_results.get('rmse', 'N/A'):.3f}\n"
   summary_text += f"  MAE: {best_results.get('mae', 'N/A'):.3f}\n"
   summary_text += f"  MAPE: {best_results.get('mape', 'N/A'):.1f}%\n"
   summary_text += f"  Dir. Accuracy: {best_results.get('directional_accuracy', 'N/A'):.3f}\n"
   
   if 'well_calibrated' in best_results:
       summary_text += f"  Well Calibrated: {best_results.get('well_calibrated', 'N/A')}\n"
   
   ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
          verticalalignment='top', fontfamily='monospace')
   
   plt.tight_layout()
   plt.show()
   
   # Print detailed comparison table
   print("\nDetailed Model Comparison:")
   print("=" * 80)
   
   metrics_to_compare = ['aic', 'bic', 'rmse', 'mae', 'mape', 'directional_accuracy']
   
   print(f"{'Model':<15s}", end="")
   for metric in metrics_to_compare:
       print(f"{metric.upper():<12s}", end="")
   print()
   print("-" * 80)
   
   for name, results in comparison_results.items():
       if 'error' not in results:
           print(f"{name:<15s}", end="")
           for metric in metrics_to_compare:
               value = results.get(metric, np.nan)
               if not np.isnan(value):
                   if metric in ['aic', 'bic']:
                       print(f"{value:<12.1f}", end="")
                   elif metric in ['rmse', 'mae']:
                       print(f"{value:<12.3f}", end="")
                   elif metric in ['mape']:
                       print(f"{value:<12.1f}", end="")
                   else:
                       print(f"{value:<12.3f}", end="")
               else:
                   print(f"{'N/A':<12s}", end="")
           print()
   ```

3. **Connection to Chapter Exercises:**
   This comprehensive diagnostic framework prepares students for advanced exercises involving model selection, forecast evaluation, and diagnostic checking - essential skills for real-world forecasting applications.

These worked examples provide hands-on experience with the key forecasting concepts covered in Chapter 12, preparing students to tackle the more challenging exercises that follow while demonstrating best practices in probabilistic forecasting, ensemble methods, scenario analysis, and diagnostic evaluation.