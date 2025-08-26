Here are the worked examples for Chapter 13, designed to bridge the gap between the main text and the exercises. These examples provide detailed implementations across different application domains, preparing students to tackle domain-specific challenges while maintaining the pedagogical style of Feynman, Gelman, Jaynes, and Murphy.

### Worked Example 1: Financial Time Series - GARCH Modeling with Regime Detection
**Context:** Before tackling Exercise 13.2 on real financial data complexities, this example demonstrates how to implement and diagnose GARCH models while handling structural breaks and regime changes.

1. **Theoretical Background:**
   - GARCH models capture volatility clustering in financial returns
   - Real financial data often exhibits regime switching and structural breaks
   - Proper model diagnostics are crucial for detecting when standard GARCH assumptions fail

2. **Example:**
   Let's implement a comprehensive GARCH analysis system with regime detection.
   
   **Step 1:** Generate realistic financial return data with regimes
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from arch import arch_model
   from scipy import stats
   from sklearn.mixture import GaussianMixture
   from datetime import datetime, timedelta
   
   class FinancialDataGenerator:
       def __init__(self, seed=42):
           np.random.seed(seed)
           
       def generate_garch_returns(self, n=1000, omega=0.01, alpha=0.1, beta=0.85):
           """Generate GARCH(1,1) returns"""
           returns = np.zeros(n)
           sigma2 = np.zeros(n)
           sigma2[0] = omega / (1 - alpha - beta)
           
           for t in range(1, n):
               sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
               returns[t] = np.sqrt(sigma2[t]) * np.random.standard_normal()
           
           return returns, sigma2
       
       def add_regime_switches(self, returns, regime_dates, regime_params):
           """Add regime switching to return series"""
           enhanced_returns = returns.copy()
           
           for i, (start_date, end_date) in enumerate(regime_dates):
               regime_mult = regime_params[i]
               enhanced_returns[start_date:end_date] *= regime_mult
           
           return enhanced_returns
       
       def add_fat_tails(self, returns, df=5):
           """Convert normal returns to t-distributed (fat tails)"""
           # Convert to uniform then to t-distribution
           uniform_vals = stats.norm.cdf(returns)
           t_returns = stats.t.ppf(uniform_vals, df=df)
           
           # Scale to maintain similar variance
           return t_returns * np.std(returns) / np.std(t_returns)
   
   # Generate synthetic data with realistic features
   generator = FinancialDataGenerator()
   base_returns, true_volatility = generator.generate_garch_returns(n=2000)
   
   # Add regime switches (crisis periods)
   regime_dates = [(400, 500), (1200, 1350)]  # Two crisis periods
   regime_params = [2.5, 3.0]  # Higher volatility multipliers
   
   synthetic_returns = generator.add_regime_switches(base_returns, regime_dates, regime_params)
   synthetic_returns = generator.add_fat_tails(synthetic_returns, df=4)
   
   # Create time index
   start_date = datetime(2010, 1, 1)
   dates = [start_date + timedelta(days=i) for i in range(len(synthetic_returns))]
   returns_series = pd.Series(synthetic_returns, index=dates)
   ```
   
   **Step 2:** Implement comprehensive GARCH analysis framework
   ```python
   class GARCHAnalyzer:
       def __init__(self):
           self.models = {}
           self.diagnostics = {}
           
       def fit_garch_models(self, returns, models_to_fit=None):
           """Fit multiple GARCH specifications"""
           if models_to_fit is None:
               models_to_fit = {
                   'GARCH(1,1)': {'p': 1, 'q': 1},
                   'GARCH(1,2)': {'p': 1, 'q': 2},
                   'GARCH(2,1)': {'p': 2, 'q': 1},
                   'EGARCH(1,1)': {'p': 1, 'q': 1, 'egarch': True},
                   'GJR-GARCH(1,1)': {'p': 1, 'q': 1, 'gjr': True}
               }
           
           for name, params in models_to_fit.items():
               try:
                   if params.get('egarch', False):
                       model = arch_model(returns, vol='EGARCH', 
                                        p=params['p'], q=params['q'])
                   elif params.get('gjr', False):
                       model = arch_model(returns, vol='GARCH', 
                                        p=params['p'], o=1, q=params['q'])
                   else:
                       model = arch_model(returns, vol='GARCH', 
                                        p=params['p'], q=params['q'])
                   
                   fitted_model = model.fit(disp='off')
                   self.models[name] = fitted_model
                   print(f"Successfully fitted {name}")
                   
               except Exception as e:
                   print(f"Failed to fit {name}: {e}")
           
           return self.models
       
       def compute_diagnostics(self, model_name):
           """Comprehensive GARCH model diagnostics"""
           if model_name not in self.models:
               raise ValueError(f"Model {model_name} not found")
           
           model = self.models[model_name]
           diagnostics = {}
           
           # Basic model fit statistics
           diagnostics['aic'] = model.aic
           diagnostics['bic'] = model.bic
           diagnostics['log_likelihood'] = model.loglikelihood
           
           # Standardized residuals
           std_resid = model.resid / model.conditional_volatility
           
           # Normality tests
           jb_stat, jb_pvalue = stats.jarque_bera(std_resid)
           diagnostics['jarque_bera_pvalue'] = jb_pvalue
           diagnostics['residuals_normal'] = jb_pvalue > 0.05
           
           # ARCH test on standardized residuals
           arch_test = model.arch_lm_test(lags=10)
           diagnostics['arch_test_pvalue'] = arch_test.pvalue
           diagnostics['no_remaining_arch'] = arch_test.pvalue > 0.05
           
           # Ljung-Box test on standardized residuals
           from statsmodels.stats.diagnostic import acorr_ljungbox
           lb_test = acorr_ljungbox(std_resid, lags=10, return_df=True)
           diagnostics['ljung_box_pvalue'] = lb_test['lb_pvalue'].iloc[-1]
           diagnostics['no_serial_correlation'] = lb_test['lb_pvalue'].iloc[-1] > 0.05
           
           # Ljung-Box test on squared standardized residuals
           lb_test_sq = acorr_ljungbox(std_resid**2, lags=10, return_df=True)
           diagnostics['ljung_box_sq_pvalue'] = lb_test_sq['lb_pvalue'].iloc[-1]
           diagnostics['no_remaining_volatility_clustering'] = lb_test_sq['lb_pvalue'].iloc[-1] > 0.05
           
           # Sign bias test (simplified)
           negative_shocks = (model.resid < 0).astype(int)
           sign_bias_stat, sign_bias_pvalue = stats.pearsonr(negative_shocks[1:], std_resid[1:]**2)
           diagnostics['sign_bias_pvalue'] = sign_bias_pvalue
           diagnostics['no_sign_bias'] = sign_bias_pvalue > 0.05
           
           self.diagnostics[model_name] = diagnostics
           return diagnostics
       
       def detect_regime_changes(self, returns, method='gaussian_mixture', n_regimes=2):
           """Detect regime changes in volatility"""
           if method == 'gaussian_mixture':
               # Use absolute returns as proxy for volatility
               abs_returns = np.abs(returns.values).reshape(-1, 1)
               
               # Fit Gaussian mixture model
               gmm = GaussianMixture(n_components=n_regimes, random_state=42)
               regime_probs = gmm.fit_predict(abs_returns)
               
               # Get regime probabilities
               regime_probabilities = gmm.predict_proba(abs_returns)
               
               return regime_probs, regime_probabilities, gmm
           
           elif method == 'rolling_volatility':
               # Simple regime detection based on rolling volatility
               window = 50
               rolling_vol = returns.rolling(window).std()
               vol_threshold = rolling_vol.quantile(0.75)
               
               high_vol_regime = (rolling_vol > vol_threshold).astype(int)
               return high_vol_regime, None, None
       
       def rolling_garch_analysis(self, returns, window_size=500, step_size=50):
           """Rolling window GARCH parameter estimation"""
           results = []
           
           for start in range(0, len(returns) - window_size, step_size):
               end = start + window_size
               window_data = returns.iloc[start:end]
               
               try:
                   # Fit GARCH(1,1) to window
                   model = arch_model(window_data, vol='GARCH', p=1, q=1)
                   fitted = model.fit(disp='off')
                   
                   result = {
                       'start_date': window_data.index[0],
                       'end_date': window_data.index[-1],
                       'omega': fitted.params['omega'],
                       'alpha[1]': fitted.params['alpha[1]'],
                       'beta[1]': fitted.params['beta[1]'],
                       'persistence': fitted.params['alpha[1]'] + fitted.params['beta[1]'],
                       'log_likelihood': fitted.loglikelihood,
                       'aic': fitted.aic
                   }
                   results.append(result)
                   
               except Exception as e:
                   print(f"Error in window {start}: {e}")
                   continue
           
           return pd.DataFrame(results)
   ```
   
   **Step 3:** Apply comprehensive analysis
   ```python
   # Initialize analyzer
   analyzer = GARCHAnalyzer()
   
   # Fit multiple GARCH models
   fitted_models = analyzer.fit_garch_models(returns_series)
   
   # Compute diagnostics for all models
   all_diagnostics = {}
   for model_name in fitted_models.keys():
       all_diagnostics[model_name] = analyzer.compute_diagnostics(model_name)
   
   # Detect regime changes
   regimes, regime_probs, gmm_model = analyzer.detect_regime_changes(returns_series)
   
   # Rolling parameter analysis
   rolling_results = analyzer.rolling_garch_analysis(returns_series)
   
   # Create comprehensive visualization
   fig, axes = plt.subplots(3, 3, figsize=(18, 15))
   
   # Returns time series with regime coloring
   ax = axes[0, 0]
   colors = ['blue', 'red', 'green']
   for regime in range(len(np.unique(regimes))):
       mask = regimes == regime
       ax.scatter(returns_series.index[mask], returns_series.values[mask], 
                 c=colors[regime], alpha=0.6, s=1, label=f'Regime {regime+1}')
   ax.set_title('Returns with Detected Regimes')
   ax.legend()
   ax.tick_params(axis='x', rotation=45)
   
   # Volatility estimation comparison
   ax = axes[0, 1]
   best_model = min(fitted_models.keys(), key=lambda x: fitted_models[x].aic)
   estimated_vol = fitted_models[best_model].conditional_volatility
   true_vol_scaled = np.sqrt(true_volatility[:len(estimated_vol)]) * np.std(returns_series)
   
   ax.plot(returns_series.index[:len(estimated_vol)], estimated_vol, 
           label='GARCH Estimated', alpha=0.8)
   ax.plot(returns_series.index[:len(true_vol_scaled)], true_vol_scaled, 
           label='True Volatility', alpha=0.8)
   ax.set_title(f'Volatility Estimation - {best_model}')
   ax.legend()
   ax.tick_params(axis='x', rotation=45)
   
   # Model comparison (AIC/BIC)
   ax = axes[0, 2]
   model_names = list(fitted_models.keys())
   aic_values = [fitted_models[name].aic for name in model_names]
   bic_values = [fitted_models[name].bic for name in model_names]
   
   x = np.arange(len(model_names))
   width = 0.35
   ax.bar(x - width/2, aic_values, width, label='AIC', alpha=0.8)
   ax.bar(x + width/2, bic_values, width, label='BIC', alpha=0.8)
   ax.set_xlabel('Model')
   ax.set_ylabel('Information Criterion')
   ax.set_title('Model Comparison')
   ax.set_xticks(x)
   ax.set_xticklabels(model_names, rotation=45)
   ax.legend()
   
   # Standardized residuals Q-Q plot
   ax = axes[1, 0]
   best_model_obj = fitted_models[best_model]
   std_resid = best_model_obj.resid / best_model_obj.conditional_volatility
   stats.probplot(std_resid, dist="norm", plot=ax)
   ax.set_title(f'Q-Q Plot - {best_model}')
   
   # ACF of squared standardized residuals
   ax = axes[1, 1]
   from statsmodels.graphics.tsaplots import plot_acf
   plot_acf(std_resid**2, ax=ax, lags=20)
   ax.set_title('ACF of Squared Std. Residuals')
   
   # Rolling GARCH parameters
   ax = axes[1, 2]
   if len(rolling_results) > 0:
       ax.plot(rolling_results['end_date'], rolling_results['alpha[1]'], 
              label='Alpha (ARCH effect)', marker='o', markersize=3)
       ax.plot(rolling_results['end_date'], rolling_results['beta[1]'], 
              label='Beta (GARCH effect)', marker='s', markersize=3)
       ax.plot(rolling_results['end_date'], rolling_results['persistence'], 
              label='Persistence (α + β)', marker='^', markersize=3)
       ax.axhline(1.0, color='red', linestyle='--', label='Unit persistence')
       ax.legend()
       ax.set_title('Rolling GARCH Parameters')
       ax.tick_params(axis='x', rotation=45)
   
   # Regime probability evolution
   ax = axes[2, 0]
   if regime_probs is not None:
       for regime in range(regime_probs.shape[1]):
           ax.plot(returns_series.index, regime_probs[:, regime], 
                  label=f'Regime {regime+1} Probability', alpha=0.8)
       ax.set_title('Regime Probabilities Over Time')
       ax.legend()
       ax.tick_params(axis='x', rotation=45)
   
   # Diagnostics heatmap
   ax = axes[2, 1]
   diagnostic_metrics = ['residuals_normal', 'no_remaining_arch', 'no_serial_correlation', 
                        'no_remaining_volatility_clustering', 'no_sign_bias']
   
   # Create diagnostic matrix
   diag_matrix = np.zeros((len(model_names), len(diagnostic_metrics)))
   for i, model_name in enumerate(model_names):
       for j, metric in enumerate(diagnostic_metrics):
           diag_matrix[i, j] = int(all_diagnostics[model_name].get(metric, False))
   
   im = ax.imshow(diag_matrix, cmap='RdYlGn', aspect='auto')
   ax.set_xticks(range(len(diagnostic_metrics)))
   ax.set_xticklabels([m.replace('_', '\n') for m in diagnostic_metrics], rotation=45)
   ax.set_yticks(range(len(model_names)))
   ax.set_yticklabels(model_names)
   ax.set_title('Model Diagnostics\n(Green=Pass, Red=Fail)')
   
   # Add text annotations
   for i in range(len(model_names)):
       for j in range(len(diagnostic_metrics)):
           text = ax.text(j, i, 'Pass' if diag_matrix[i, j] else 'Fail',
                         ha="center", va="center", color="black", fontsize=8)
   
   # VaR comparison
   ax = axes[2, 2]
   confidence_levels = [0.01, 0.05, 0.1]
   
   # Empirical VaR
   empirical_vars = [np.quantile(returns_series, level) for level in confidence_levels]
   
   # Model-based VaR (using best GARCH model)
   best_model_obj = fitted_models[best_model]
   forecasts = best_model_obj.forecast(horizon=1, reindex=False)
   forecast_vol = np.sqrt(forecasts.variance.iloc[-1, 0])
   
   # Assuming normal distribution for VaR calculation
   normal_vars = [stats.norm.ppf(level, loc=0, scale=forecast_vol) for level in confidence_levels]
   
   # Assuming t-distribution (better for fat tails)
   df_estimated = 4  # Could estimate from data
   t_vars = [stats.t.ppf(level, df=df_estimated, loc=0, scale=forecast_vol) for level in confidence_levels]
   
   x = np.arange(len(confidence_levels))
   width = 0.25
   ax.bar(x - width, empirical_vars, width, label='Empirical VaR', alpha=0.8)
   ax.bar(x, normal_vars, width, label='Normal VaR', alpha=0.8)
   ax.bar(x + width, t_vars, width, label='t-dist VaR', alpha=0.8)
   
   ax.set_xlabel('Confidence Level')
   ax.set_ylabel('Value at Risk')
   ax.set_title('VaR Comparison')
   ax.set_xticks(x)
   ax.set_xticklabels([f'{int(level*100)}%' for level in confidence_levels])
   ax.legend()
   
   plt.tight_layout()
   plt.show()
   
   # Print summary results
   print("\nGARCH Analysis Summary:")
   print("=" * 60)
   
   print(f"\nBest Model (by AIC): {best_model}")
   print(f"AIC: {fitted_models[best_model].aic:.2f}")
   print(f"BIC: {fitted_models[best_model].bic:.2f}")
   
   print(f"\nModel Parameters ({best_model}):")
   for param_name, param_value in fitted_models[best_model].params.items():
       print(f"  {param_name}: {param_value:.6f}")
   
   print(f"\nDiagnostic Tests ({best_model}):")
   best_diagnostics = all_diagnostics[best_model]
   for test_name, result in best_diagnostics.items():
       if isinstance(result, bool):
           print(f"  {test_name}: {'PASS' if result else 'FAIL'}")
       elif 'pvalue' in test_name:
           print(f"  {test_name}: {result:.4f}")
   
   print(f"\nRegime Analysis:")
   unique_regimes, regime_counts = np.unique(regimes, return_counts=True)
   for regime, count in zip(unique_regimes, regime_counts):
       percentage = count / len(regimes) * 100
       print(f"  Regime {regime + 1}: {count} observations ({percentage:.1f}%)")
   
   if len(rolling_results) > 0:
       print(f"\nParameter Stability:")
       print(f"  Alpha range: [{rolling_results['alpha[1]'].min():.4f}, {rolling_results['alpha[1]'].max():.4f}]")
       print(f"  Beta range: [{rolling_results['beta[1]'].min():.4f}, {rolling_results['beta[1]'].max():.4f}]")
       print(f"  Persistence range: [{rolling_results['persistence'].min():.4f}, {rolling_results['persistence'].max():.4f}]")
       
       # Check for unit root in volatility process
       unstable_windows = np.sum(rolling_results['persistence'] >= 0.99)
       print(f"  Windows with near-unit persistence (≥0.99): {unstable_windows}/{len(rolling_results)}")
   ```

3. **Connection to Exercise 13.2:**
   This example provides the comprehensive framework needed for Exercise 13.2, including rolling parameter estimation, regime detection, diagnostic testing, and crisis period analysis - all essential for handling real financial data complexities.

### Worked Example 2: Climate Time Series - Trend Detection and Attribution
**Context:** This example prepares students for Exercise 13.3 by demonstrating how to detect and attribute climate trends while handling natural variability and potential confounding factors.

1. **Theoretical Background:**
   - Climate data contains multiple time scales: seasonal, annual, decadal, and long-term trends
   - Trend detection must account for natural variability and autocorrelation
   - Attribution involves separating anthropogenic signals from natural variation

2. **Example:**
   Let's implement a comprehensive climate trend analysis system.
   
   **Step 1:** Generate realistic climate data with known trends
   ```python
   import numpy as np
   import pandas as pd
   from scipy import signal
   from statsmodels.tsa.seasonal import seasonal_decompose
   from statsmodels.tsa.stattools import adfuller
   from sklearn.linear_model import LinearRegression
   
   class ClimateDataGenerator:
       def __init__(self, seed=42):
           np.random.seed(seed)
           
       def generate_temperature_series(self, n_years=120, start_year=1900):
           """Generate realistic temperature time series"""
           n_months = n_years * 12
           
           # Time variables
           months = np.arange(n_months)
           years = start_year + months / 12
           
           # Seasonal cycle
           seasonal = 10 * np.cos(2 * np.pi * months / 12) + 3 * np.cos(4 * np.pi * months / 12)
           
           # Long-term natural variability (AMO, PDO-like oscillations)
           amo_like = 0.8 * np.sin(2 * np.pi * months / (60 * 12))  # ~60 year cycle
           pdo_like = 0.6 * np.sin(2 * np.pi * months / (20 * 12))  # ~20 year cycle
           
           # ENSO-like variation
           enso_like = 1.2 * np.sin(2 * np.pi * months / (3.5 * 12)) * np.random.exponential(0.7, n_months)
           
           # Volcanic effects (random large negative anomalies)
           volcanic = np.zeros(n_months)
           volcanic_events = np.random.choice(n_months, size=5, replace=False)
           for event in volcanic_events:
               duration = np.random.randint(6, 24)  # 6-24 months
               end_idx = min(event + duration, n_months)
               volcanic[event:end_idx] = -np.random.uniform(1.5, 3.0) * np.exp(-np.arange(end_idx - event) / 6)
           
           # Anthropogenic warming (nonlinear acceleration)
           warming_start_year = 1950
           warming_start_idx = int((warming_start_year - start_year) * 12)
           
           anthropogenic = np.zeros(n_months)
           for i in range(warming_start_idx, n_months):
               years_since_warming = (i - warming_start_idx) / 12
               # Accelerating warming trend
               anthropogenic[i] = 0.01 * years_since_warming + 0.0002 * years_since_warming**2
           
           # Natural year-to-year variability
           natural_noise = np.random.normal(0, 0.8, n_months)
           
           # Combine all components
           temperature = (15 +  # baseline temperature
                         seasonal +
                         amo_like + pdo_like + enso_like +
                         volcanic +
                         anthropogenic +
                         natural_noise)
           
           # Create DataFrame
           dates = pd.date_range(start=f'{start_year}-01', periods=n_months, freq='M')
           
           return pd.DataFrame({
               'temperature': temperature,
               'seasonal': seasonal,
               'amo_like': amo_like,
               'pdo_like': pdo_like,
               'enso_like': enso_like,
               'volcanic': volcanic,
               'anthropogenic': anthropogenic,
               'natural_noise': natural_noise
           }, index=dates)
   
   class ClimateAnalyzer:
       def __init__(self):
           self.results = {}
           
       def detect_trends(self, data, methods=None):
           """Multiple trend detection methods"""
           if methods is None:
               methods = ['linear', 'mann_kendall', 'sen_slope', 'changepoint']
           
           trends = {}
           
           # Linear trend
           if 'linear' in methods:
               trends['linear'] = self._linear_trend(data)
           
           # Mann-Kendall trend test
           if 'mann_kendall' in methods:
               trends['mann_kendall'] = self._mann_kendall_test(data)
           
           # Sen's slope estimator
           if 'sen_slope' in methods:
               trends['sen_slope'] = self._sen_slope(data)
           
           # Change point detection
           if 'changepoint' in methods:
               trends['changepoint'] = self._detect_changepoints(data)
           
           return trends
       
       def _linear_trend(self, data):
           """Simple linear trend estimation"""
           x = np.arange(len(data)).reshape(-1, 1)
           y = data.values
           
           model = LinearRegression()
           model.fit(x, y)
           
           # Calculate confidence intervals
           predictions = model.predict(x)
           residuals = y - predictions
           mse = np.mean(residuals**2)
           
           # Standard error of slope
           x_centered = x.flatten() - np.mean(x)
           se_slope = np.sqrt(mse / np.sum(x_centered**2))
           
           # 95% confidence interval for slope
           t_critical = 1.96  # approximately for large samples
           slope_ci = [model.coef_[0] - t_critical * se_slope,
                      model.coef_[0] + t_critical * se_slope]
           
           return {
               'slope': model.coef_[0],
               'intercept': model.intercept_,
               'r_squared': model.score(x, y),
               'slope_ci': slope_ci,
               'significant': 0 not in slope_ci,
               'fitted': predictions
           }
       
       def _mann_kendall_test(self, data):
           """Mann-Kendall trend test"""
           n = len(data)
           s = 0
           
           for i in range(n-1):
               for j in range(i+1, n):
                   if data.iloc[j] > data.iloc[i]:
                       s += 1
                   elif data.iloc[j] < data.iloc[i]:
                       s -= 1
           
           # Variance of S
           var_s = n * (n - 1) * (2 * n + 5) / 18
           
           # Standardized test statistic
           if s > 0:
               z = (s - 1) / np.sqrt(var_s)
           elif s < 0:
               z = (s + 1) / np.sqrt(var_s)
           else:
               z = 0
           
           # p-value (two-tailed)
           p_value = 2 * (1 - stats.norm.cdf(abs(z)))
           
           return {
               's_statistic': s,
               'z_statistic': z,
               'p_value': p_value,
               'trend_direction': 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no trend',
               'significant': p_value < 0.05
           }
       
       def _sen_slope(self, data):
           """Sen's slope estimator (robust trend estimate)"""
           n = len(data)
           slopes = []
           
           for i in range(n-1):
               for j in range(i+1, n):
                   if j != i:
                       slope = (data.iloc[j] - data.iloc[i]) / (j - i)
                       slopes.append(slope)
           
           slopes = np.array(slopes)
           sen_slope = np.median(slopes)
           
           # Confidence interval (simplified)
           slopes_sorted = np.sort(slopes)
           n_slopes = len(slopes)
           ci_lower_idx = int(0.025 * n_slopes)
           ci_upper_idx = int(0.975 * n_slopes)
           
           return {
               'sen_slope': sen_slope,
               'slope_ci': [slopes_sorted[ci_lower_idx], slopes_sorted[ci_upper_idx]],
               'all_slopes': slopes
           }
       
       def _detect_changepoints(self, data, min_segment_length=24):
           """Simple change point detection using binary segmentation"""
           def compute_cost(segment):
               if len(segment) < 2:
                   return 0
               return np.var(segment) * len(segment)
           
           def find_best_split(segment, start_idx):
               if len(segment) < 2 * min_segment_length:
                   return None, float('inf')
               
               best_split = None
               best_cost = float('inf')
               
               for split in range(min_segment_length, len(segment) - min_segment_length):
                   left_cost = compute_cost(segment[:split])
                   right_cost = compute_cost(segment[split:])
                   total_cost = left_cost + right_cost
                   
                   if total_cost < best_cost:
                       best_cost = total_cost
                       best_split = start_idx + split
               
               return best_split, best_cost
           
           # Simple implementation - find one major change point
           original_cost = compute_cost(data.values)
           best_changepoint, best_cost = find_best_split(data.values, 0)
           
           improvement = original_cost - best_cost
           
           return {
               'changepoint': best_changepoint,
               'changepoint_date': data.index[best_changepoint] if best_changepoint else None,
               'cost_improvement': improvement,
               'significant_change': improvement > original_cost * 0.1  # 10% improvement threshold
           }
       
       def decompose_series(self, data, model='additive', period=12):
           """Enhanced time series decomposition"""
           # Standard decomposition
           decomposition = seasonal_decompose(data, model=model, period=period)
           
           # Additional analysis on trend component
           trend_data = decomposition.trend.dropna()
           trend_analysis = self.detect_trends(trend_data)
           
           # Analysis of residuals
           residuals = decomposition.resid.dropna()
           residual_stats = {
               'mean': residuals.mean(),
               'std': residuals.std(),
               'autocorr_lag1': residuals.autocorr(lag=1),
               'ljung_box_pvalue': acorr_ljungbox(residuals, lags=10, return_df=True)['lb_pvalue'].iloc[-1]
           }
           
           return {
               'decomposition': decomposition,
               'trend_analysis': trend_analysis,
               'residual_stats': residual_stats
           }
       
       def attribution_analysis(self, observed_data, natural_components, anthropogenic_component):
           """Climate attribution analysis"""
           # Multiple regression approach
           X = np.column_stack([comp.values for comp in natural_components.values()])
           X = np.column_stack([X, anthropogenic_component.values])
           
           y = observed_data.values
           
           # Fit attribution model
           model = LinearRegression()
           model.fit(X, y)
           
           # Component names
           component_names = list(natural_components.keys()) + ['anthropogenic']
           
           # Calculate explained variance by component
           predictions = model.predict(X)
           total_var = np.var(observed_data)
           
           component_contributions = {}
           for i, name in enumerate(component_names):
               # Partial contribution
               X_partial = X.copy()
               X_partial[:, i] = 0
               pred_without_component = model.predict(X_partial)
               contribution_var = np.var(predictions - pred_without_component)
               component_contributions[name] = {
                   'coefficient': model.coef_[i],
                   'variance_explained': contribution_var,
                   'percent_variance': contribution_var / total_var * 100
               }
           
           return {
               'model': model,
               'r_squared': model.score(X, y),
               'component_contributions': component_contributions,
               'fitted_values': predictions,
               'residuals': y - predictions
           }
   ```
   
   **Step 2:** Apply comprehensive climate analysis
   ```python
   # Generate synthetic climate data
   generator = ClimateDataGenerator()
   climate_data = generator.generate_temperature_series(n_years=120, start_year=1900)
   
   # Initialize analyzer
   analyzer = ClimateAnalyzer()
   
   # Perform trend detection on observed temperature
   temp_trends = analyzer.detect_trends(climate_data['temperature'])
   
   # Decompose the temperature series
   decomp_results = analyzer.decompose_series(climate_data['temperature'])
   
   # Attribution analysis
   natural_components = {
       'AMO': climate_data['amo_like'],
       'PDO': climate_data['pdo_like'],
       'ENSO': climate_data['enso_like'],
       'Volcanic': climate_data['volcanic'],
       'Noise': climate_data['natural_noise']
   }
   
   attribution_results = analyzer.attribution_analysis(
       climate_data['temperature'],
       natural_components,
       climate_data['anthropogenic']
   )
   
   # Create comprehensive visualization
   fig, axes = plt.subplots(3, 3, figsize=(18, 15))
   
   # Original temperature time series with trends
   ax = axes[0, 0]
   ax.plot(climate_data.index, climate_data['temperature'], 'b-', alpha=0.6, label='Observed')
   
   # Add linear trend
   linear_trend = temp_trends['linear']
   trend_line = linear_trend['intercept'] + linear_trend['slope'] * np.arange(len(climate_data))
   ax.plot(climate_data.index, trend_line, 'r--', linewidth=2, label='Linear Trend')
   
   # Add confidence intervals
   se_pred = np.sqrt(np.var(climate_data['temperature'] - linear_trend['fitted']))
   ax.fill_between(climate_data.index, 
                   trend_line - 1.96*se_pred, trend_line + 1.96*se_pred,
                   alpha=0.2, color='red', label='95% CI')
   
   ax.set_title('Temperature Time Series with Linear Trend')
   ax.set_ylabel('Temperature (°C)')
   ax.legend()
   
   # Decomposition plot
   ax = axes[0, 1]
   decomp = decomp_results['decomposition']
   ax.plot(climate_data.index, decomp.trend, 'g-', label='Trend')
   ax.plot(climate_data.index, climate_data['anthropogenic'] + np.mean(climate_data['temperature']), 
           'r--', label='True Anthropogenic', alpha=0.8)
   ax.set_title('Extracted Trend vs True Anthropogenic Signal')
   ax.legend()
   
   # Seasonal component
   ax = axes[0, 2]
   # Plot seasonal component for one year
   one_year = decomp.seasonal[:12]
   months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
   ax.plot(months, one_year, 'b-', marker='o')
   ax.set_title('Seasonal Pattern')
   ax.set_ylabel('Temperature Anomaly (°C)')
   ax.tick_params(axis='x', rotation=45)
   
   # Attribution results
   ax = axes[1, 0]
   contributions = attribution_results['component_contributions']
   components = list(contributions.keys())
   percentages = [contributions[comp]['percent_variance'] for comp in components]
   
   colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
   bars = ax.bar(components, percentages, color=colors)
   ax.set_title('Variance Explained by Components')
   ax.set_ylabel('Percent Variance Explained')
   ax.tick_params(axis='x', rotation=45)
   
   # Add percentage labels on bars
   for bar, pct in zip(bars, percentages):
       height = bar.get_height()
       ax.text(bar.get_x() + bar.get_width()/2., height,
              f'{pct:.1f}%', ha='center', va='bottom')
   
   # Model fit quality
   ax = axes[1, 1]
   observed = climate_data['temperature']
   fitted = attribution_results['fitted_values']
   ax.scatter(fitted, observed, alpha=0.5)
   ax.plot([fitted.min(), fitted.max()], [fitted.min(), fitted.max()], 'r--')
   ax.set_xlabel('Fitted Values')
   ax.set_ylabel('Observed Values')
   ax.set_title(f'Model Fit (R² = {attribution_results["r_squared"]:.3f})')
   
   # Residual analysis
   ax = axes[1, 2]
   residuals = attribution_results['residuals']
   ax.plot(climate_data.index, residuals, 'b-', alpha=0.6)
   ax.axhline(0, color='r', linestyle='--')
   ax.set_title('Attribution Model Residuals')
   ax.set_ylabel('Residual (°C)')
   
   # Trend significance tests
   ax = axes[2, 0]
   test_names = ['Linear', 'Mann-Kendall', 'Sen Slope']
   test_results = [
       temp_trends['linear']['significant'],
       temp_trends['mann_kendall']['significant'],
       temp_trends['sen_slope']['slope_ci'][0] > 0  # Sen slope CI doesn't contain 0
   ]
   
   colors = ['green' if result else 'red' for result in test_results]
   bars = ax.bar(test_names, [1 if result else 0 for result in test_results], color=colors)
   ax.set_title('Trend Significance Tests')
   ax.set_ylabel('Significant Trend Detected')
   ax.set_ylim(0, 1.2)
   
   for i, (bar, result) in enumerate(zip(bars, test_results)):
       ax.text(bar.get_x() + bar.get_width()/2., 0.5,
              'YES' if result else 'NO', ha='center', va='center', 
              fontweight='bold', color='white')
   
   # Change point analysis
   ax = axes[2, 1]
   changepoint_results = temp_trends['changepoint']
   
   ax.plot(climate_data.index, climate_data['temperature'], 'b-', alpha=0.6)
   
   if changepoint_results['changepoint']:
       cp_date = changepoint_results['changepoint_date']
       ax.axvline(cp_date, color='red', linestyle='--', linewidth=2, label='Detected Change Point')
       
       # Add true change point (when anthropogenic warming started)
       true_cp = pd.Timestamp('1950-01-01')
       ax.axvline(true_cp, color='green', linestyle=':', linewidth=2, label='True Change Point')
       
       ax.legend()
   
   ax.set_title('Change Point Detection')
   ax.set_ylabel('Temperature (°C)')
   
   # Decadal trends
   ax = axes[2, 2]
   decades = []
   decade_trends = []
   
   for decade_start in range(1900, 2020, 10):
       decade_end = decade_start + 10
       decade_mask = ((climate_data.index.year >= decade_start) & 
                     (climate_data.index.year < decade_end))
       decade_data = climate_data.loc[decade_mask, 'temperature']
       
       if len(decade_data) > 12:  # At least one year of data
           decade_trend = analyzer._linear_trend(decade_data)
           decades.append(f"{decade_start}s")
           decade_trends.append(decade_trend['slope'] * 120)  # Convert to °C/decade
   
   ax.bar(decades, decade_trends, alpha=0.7)
   ax.set_title('Decadal Temperature Trends')
   ax.set_ylabel('Trend (°C/decade)')
   ax.tick_params(axis='x', rotation=45)
   ax.axhline(0, color='black', linestyle='-', alpha=0.3)
   
   plt.tight_layout()
   plt.show()
   
   # Print comprehensive results
   print("Climate Trend Analysis Summary:")
   print("=" * 50)
   
   print(f"\nLinear Trend Analysis:")
   linear = temp_trends['linear']
   print(f"  Slope: {linear['slope']:.6f} °C/month ({linear['slope']*120:.3f} °C/decade)")
   print(f"  95% CI: [{linear['slope_ci'][0]*120:.3f}, {linear['slope_ci'][1]*120:.3f}] °C/decade")
   print(f"  R²: {linear['r_squared']:.4f}")
   print(f"  Significant: {'YES' if linear['significant'] else 'NO'}")
   
   print(f"\nMann-Kendall Test:")
   mk = temp_trends['mann_kendall']
   print(f"  Z-statistic: {mk['z_statistic']:.3f}")
   print(f"  p-value: {mk['p_value']:.6f}")
   print(f"  Trend direction: {mk['trend_direction']}")
   print(f"  Significant: {'YES' if mk['significant'] else 'NO'}")
   
   print(f"\nSen's Slope Estimator:")
   sen = temp_trends['sen_slope']
   print(f"  Sen slope: {sen['sen_slope']:.6f} °C/month ({sen['sen_slope']*120:.3f} °C/decade)")
   print(f"  95% CI: [{sen['slope_ci'][0]*120:.3f}, {sen['slope_ci'][1]*120:.3f}] °C/decade")
   
   print(f"\nChange Point Detection:")
   cp = temp_trends['changepoint']
   if cp['changepoint']:
       print(f"  Detected change point: {cp['changepoint_date'].strftime('%Y-%m')}")
       print(f"  Significant change: {'YES' if cp['significant_change'] else 'NO'}")
   else:
       print(f"  No significant change point detected")
   
   print(f"\nAttribution Analysis (R² = {attribution_results['r_squared']:.3f}):")
   for component, contrib in attribution_results['component_contributions'].items():
       print(f"  {component:>12s}: {contrib['percent_variance']:5.1f}% of variance "
             f"(coeff = {contrib['coefficient']:7.4f})")
   
   print(f"\nModel Diagnostics:")
   residual_stats = decomp_results['residual_stats']
   print(f"  Residual autocorr (lag 1): {residual_stats['autocorr_lag1']:.4f}")
   print(f"  Ljung-Box p-value: {residual_stats['ljung_box_pvalue']:.4f}")
   print(f"  Residuals white noise: {'YES' if residual_stats['ljung_box_pvalue'] > 0.05 else 'NO'}")
   
   # Calculate warming acceleration
   early_period = climate_data.loc[climate_data.index.year < 1980, 'temperature']
   late_period = climate_data.loc[climate_data.index.year >= 1980, 'temperature']
   
   early_trend = analyzer._linear_trend(early_period)['slope'] * 120
   late_trend = analyzer._linear_trend(late_period)['slope'] * 120
   
   print(f"\nWarming Acceleration:")
   print(f"  Pre-1980 trend: {early_trend:.3f} °C/decade")
   print(f"  Post-1980 trend: {late_trend:.3f} °C/decade")
   print(f"  Acceleration factor: {late_trend/early_trend if early_trend != 0 else 'undefined':.1f}x")
   ```

3. **Connection to Exercise 13.3:**
   This example provides the complete toolkit for Exercise 13.3, including multiple trend detection methods, attribution analysis, change point detection, and proper handling of natural variability - all essential for convincing climate change analysis.

### Worked Example 3: Biomedical Time Series - Heart Rate Variability Analysis
**Context:** This example demonstrates advanced time series techniques applied to physiological signals, preparing students for complex biomedical data analysis.

1. **Theoretical Background:**
   - HRV analysis requires both time-domain and frequency-domain methods
   - Physiological signals often exhibit nonlinear dynamics and multiscale behavior
   - Clinical interpretation requires domain-specific feature extraction

2. **Example:**
   Let's implement comprehensive HRV analysis with clinical relevance.
   
   **Step 1:** Generate realistic HRV data
   ```python
   import numpy as np
   from scipy import signal, stats
   from scipy.integrate import odeint
   
   class HRVGenerator:
       def __init__(self, seed=42):
           np.random.seed(seed)
           
       def generate_rr_intervals(self, duration_minutes=5, base_hr=70, 
                               respiratory_rate=0.25, stress_level=0.5):
           """Generate realistic R-R interval time series"""
           
           # Sampling parameters
           fs = 4  # 4 Hz sampling for R-R intervals
           n_samples = int(duration_minutes * 60 * fs)
           t = np.linspace(0, duration_minutes * 60, n_samples)
           
           # Base heart rate to R-R interval conversion
           base_rr = 60.0 / base_hr  # seconds
           
           # Respiratory sinus arrhythmia (RSA)
           rsa = 0.05 * (1 - stress_level) * np.sin(2 * np.pi * respiratory_rate * t)
           
           # Low frequency component (sympathetic/parasympathetic balance)
           lf_component = 0.03 * np.sin(2 * np.pi * 0.1 * t + np.random.uniform(0, 2*np.pi))
           
           # Very low frequency component
           vlf_component = 0.02 * np.sin(2 * np.pi * 0.04 * t + np.random.uniform(0, 2*np.pi))
           
           # Nonlinear component (chaos-like)
           def lorenz_system(state, t):
               x, y, z = state
               return [10.0 * (y - x), x * (28.0 - z) - y, x * y - 8.0/3.0 * z]
           
           # Generate chaotic component
           lorenz_t = np.linspace(0, 10, n_samples)
           lorenz_sol = odeint(lorenz_system, [1, 1, 1], lorenz_t)
           chaos_component = 0.01 * (lorenz_sol[:, 0] - np.mean(lorenz_sol[:, 0])) / np.std(lorenz_sol[:, 0])
           
           # Stress-related changes
           stress_component = stress_level * 0.02 * np.random.randn(n_samples)
           
           # High-frequency noise
           noise = 0.01 * np.random.randn(n_samples)
           
           # Combine components
           rr_variations = rsa + lf_component + vlf_component + chaos_component + stress_component + noise
           rr_intervals = base_rr + rr_variations
           
           # Ensure realistic bounds
           rr_intervals = np.clip(rr_intervals, 0.4, 1.5)  # 40-150 bpm range
           
           return t, rr_intervals
   
   class HRVAnalyzer:
       def __init__(self):
           self.results = {}
           
       def time_domain_analysis(self, rr_intervals):
           """Comprehensive time-domain HRV analysis"""
           
           # Convert to milliseconds if in seconds
           if np.mean(rr_intervals) < 10:
               rr_ms = rr_intervals * 1000
           else:
               rr_ms = rr_intervals
           
           # Basic statistics
           mean_rr = np.mean(rr_ms)
           std_rr = np.std(rr_ms, ddof=1)  # SDNN
           
           # R-R differences
           rr_diff = np.diff(rr_ms)
           rmssd = np.sqrt(np.mean(rr_diff**2))  # Root mean square of successive differences
           
           # pNN50: percentage of successive RR differences > 50ms
           pnn50 = np.sum(np.abs(rr_diff) > 50) / len(rr_diff) * 100
           
           # Triangular index
           hist, bins = np.histogram(rr_ms, bins=50)
           triangular_index = len(rr_ms) / np.max(hist)
           
           # TINN (Triangular Interpolation of NN interval histogram)
           # Simplified implementation
           tinn = (bins[-1] - bins[0])
           
           # Geometric measures
           # HRV index (simplified as mode of histogram)
           hrv_index = bins[np.argmax(hist)]
           
           return {
               'mean_rr': mean_rr,
               'sdnn': std_rr,
               'rmssd': rmssd,
               'pnn50': pnn50,
               'triangular_index': triangular_index,
               'tinn': tinn,
               'hrv_index': hrv_index,
               'cv': (std_rr / mean_rr) * 100  # Coefficient of variation
           }
       
       def frequency_domain_analysis(self, rr_intervals, fs=4):
           """Frequency-domain HRV analysis using Welch's method"""
           
           # Resample to uniform time grid if needed
           if len(rr_intervals) > 1:
               # Simple resampling (in practice, more sophisticated interpolation needed)
               t_uniform = np.linspace(0, len(rr_intervals)/fs, len(rr_intervals))
               rr_uniform = rr_intervals
           else:
               return {}
           
           # Remove trend
           rr_detrended = signal.detrend(rr_uniform)
           
           # Apply window
           window = signal.windows.hann(len(rr_detrended))
           rr_windowed = rr_detrended * window
           
           # Compute power spectral density
           frequencies, psd = signal.welch(rr_windowed, fs=fs, nperseg=len(rr_windowed)//2)
           
           # Frequency band definitions (Hz)
           vlf_band = (0.003, 0.04)
           lf_band = (0.04, 0.15)
           hf_band = (0.15, 0.4)
           
           # Compute power in each band
           vlf_power = np.trapz(psd[(frequencies >= vlf_band[0]) & 
                                   (frequencies < vlf_band[1])],
                               frequencies[(frequencies >= vlf_band[0]) & 
                                         (frequencies < vlf_band[1])])
           
           lf_power = np.trapz(psd[(frequencies >= lf_band[0]) & 
                                  (frequencies < lf_band[1])],
                              frequencies[(frequencies >= lf_band[0]) & 
                                        (frequencies < lf_band[1])])
           
           hf_power = np.trapz(psd[(frequencies >= hf_band[0]) & 
                                  (frequencies < hf_band[1])],
                              frequencies[(frequencies >= hf_band[0]) & 
                                        (frequencies < hf_band[1])])
           
           total_power = np.trapz(psd, frequencies)
           
           # Normalized powers
           lf_norm = lf_power / (total_power - vlf_power) * 100
           hf_norm = hf_power / (total_power - vlf_power) * 100
           
           # LF/HF ratio
           lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.inf
           
           return {
               'frequencies': frequencies,
               'psd': psd,
               'vlf_power': vlf_power,
               'lf_power': lf_power,
               'hf_power': hf_power,
               'total_power': total_power,
               'lf_norm': lf_norm,
               'hf_norm': hf_norm,
               'lf_hf_ratio': lf_hf_ratio
           }
       
       def nonlinear_analysis(self, rr_intervals):
           """Nonlinear HRV analysis methods"""
           
           # Convert to milliseconds
           if np.mean(rr_intervals) < 10:
               rr_ms = rr_intervals * 1000
           else:
               rr_ms = rr_intervals
               
           results = {}
           
           # Poincaré plot analysis
           rr1 = rr_ms[:-1]  # RR(n)
           rr2 = rr_ms[1:]   # RR(n+1)
           
           # SD1 and SD2
           diff_rr = rr2 - rr1
           sum_rr = rr2 + rr1
           
           sd1 = np.std(diff_rr, ddof=1) / np.sqrt(2)
           sd2 = np.std(sum_rr, ddof=1) / np.sqrt(2)
           
           results['sd1'] = sd1
           results['sd2'] = sd2
           results['sd1_sd2_ratio'] = sd1 / sd2 if sd2 > 0 else 0
           
           # Approximate entropy (simplified implementation)
           def _maxdist(xi, xj, N):
               return max([abs(ua - va) for ua, va in zip(xi, xj)])
           
           def _phi(m, r, data):
               N = len(data)
               patterns = np.array([data[i:i+m] for i in range(N - m + 1)])
               C = np.zeros(N - m + 1)
               
               for i in range(N - m + 1):
                   template = patterns[i]
                   matches = [_maxdist(template, patterns[j], m) <= r 
                            for j in range(N - m + 1)]
                   C[i] = sum(matches) / (N - m + 1.0)
               
               phi = np.mean([np.log(c) for c in C if c > 0])
               return phi
           
           # Approximate entropy
           m = 2
           r = 0.2 * np.std(rr_ms, ddof=1)
           
           try:
               phi_m = _phi(m, r, rr_ms)
               phi_m1 = _phi(m + 1, r, rr_ms)
               results['approximate_entropy'] = phi_m - phi_m1
           except:
               results['approximate_entropy'] = np.nan
           
           # Sample entropy (simplified)
           def _sample_entropy(data, m=2, r=None):
               if r is None:
                   r = 0.2 * np.std(data, ddof=1)
               
               N = len(data)
               
               def _get_matches(m, r):
                   patterns = [data[i:i+m] for i in range(N - m + 1)]
                   matches = 0
                   comparisons = 0
                   
                   for i in range(len(patterns)):
                       for j in range(i + 1, len(patterns)):
                           comparisons += 1
                           if max(abs(a - b) for a, b in zip(patterns[i], patterns[j])) <= r:
                               matches += 1
                   
                   return matches, comparisons
               
               try:
                   A, B = _get_matches(m, r)
                   C, D = _get_matches(m + 1, r)
                   
                   if A > 0 and C > 0:
                       return -np.log(C / A)
                   else:
                       return np.nan
               except:
                   return np.nan
           
           results['sample_entropy'] = _sample_entropy(rr_ms)
           
           # Detrended Fluctuation Analysis (DFA) - simplified
           def _dfa(data, n_scales=10):
               N = len(data)
               scales = np.logspace(1, np.log10(N//4), n_scales).astype(int)
               fluctuations = []
               
               # Integrate the signal
               y = np.cumsum(data - np.mean(data))
               
               for scale in scales:
                   # Divide into non-overlapping segments
                   n_segments = N // scale
                   segments = [y[i*scale:(i+1)*scale] for i in range(n_segments)]
                   
                   # Detrend each segment and compute fluctuation
                   F = 0
                   for segment in segments:
                       if len(segment) == scale:
                           # Linear detrending
                           coeffs = np.polyfit(range(scale), segment, 1)
                           trend = np.polyval(coeffs, range(scale))
                           F += np.mean((segment - trend)**2)
                   
                   fluctuations.append(np.sqrt(F / n_segments))
               
               # Fit power law: F(n) ~ n^α
               log_scales = np.log(scales)
               log_fluct = np.log(fluctuations)
               
               alpha = np.polyfit(log_scales, log_fluct, 1)[0]
               return alpha, scales, fluctuations
           
           try:
               dfa_alpha, dfa_scales, dfa_fluctuations = _dfa(rr_ms)
               results['dfa_alpha'] = dfa_alpha
               results['dfa_scales'] = dfa_scales
               results['dfa_fluctuations'] = dfa_fluctuations
           except:
               results['dfa_alpha'] = np.nan
           
           return results
       
       def clinical_interpretation(self, time_domain, frequency_domain, nonlinear):
           """Clinical interpretation of HRV parameters"""
           
           interpretations = {}
           
           # SDNN interpretation (ms)
           sdnn = time_domain['sdnn']
           if sdnn < 50:
               interpretations['sdnn'] = 'Severely decreased HRV (high risk)'
           elif sdnn < 100:
               interpretations['sdnn'] = 'Moderately decreased HRV'
           else:
               interpretations['sdnn'] = 'Normal HRV'
           
           # RMSSD interpretation (ms)
           rmssd = time_domain['rmssd']
           if rmssd < 27:
               interpretations['rmssd'] = 'Low parasympathetic activity'
           elif rmssd > 39:
               interpretations['rmssd'] = 'High parasympathetic activity'
           else:
               interpretations['rmssd'] = 'Normal parasympathetic activity'
           
           # pNN50 interpretation (%)
           pnn50 = time_domain['pnn50']
           if pnn50 < 3:
               interpretations['pnn50'] = 'Low vagal tone'
           elif pnn50 > 7:
               interpretations['pnn50'] = 'High vagal tone'
           else:
               interpretations['pnn50'] = 'Normal vagal tone'
           
           # LF/HF ratio interpretation
           if 'lf_hf_ratio' in frequency_domain:
               lf_hf = frequency_domain['lf_hf_ratio']
               if lf_hf < 1.5:
                   interpretations['autonomic_balance'] = 'Parasympathetic dominance'
               elif lf_hf > 4:
                   interpretations['autonomic_balance'] = 'Sympathetic dominance'
               else:
                   interpretations['autonomic_balance'] = 'Balanced autonomic tone'
           
           # DFA alpha interpretation
           if 'dfa_alpha' in nonlinear and not np.isnan(nonlinear['dfa_alpha']):
               dfa_alpha = nonlinear['dfa_alpha']
               if dfa_alpha < 0.75:
                   interpretations['correlation_properties'] = 'Anti-correlated (may indicate pathology)'
               elif dfa_alpha < 1.0:
                   interpretations['correlation_properties'] = 'Healthy fractal scaling'
               elif dfa_alpha < 1.5:
                   interpretations['correlation_properties'] = 'Persistent correlations'
               else:
                   interpretations['correlation_properties'] = 'Random walk-like (may indicate pathology)'
           
           return interpretations
   ```
   
   **Step 3:** Apply comprehensive HRV analysis
   ```python
   # Generate HRV data for different conditions
   generator = HRVGenerator()
   analyzer = HRVAnalyzer()
   
   # Generate data for different physiological states
   conditions = {
       'Healthy Rest': {'stress_level': 0.2, 'base_hr': 65},
       'Mild Stress': {'stress_level': 0.5, 'base_hr': 75},
       'High Stress': {'stress_level': 0.8, 'base_hr': 85},
       'Athletic': {'stress_level': 0.1, 'base_hr': 50}
   }
   
   results = {}
   
   for condition, params in conditions.items():
       t, rr_intervals = generator.generate_rr_intervals(
           duration_minutes=5,
           base_hr=params['base_hr'],
           stress_level=params['stress_level']
       )
       
       # Perform comprehensive analysis
       time_domain = analyzer.time_domain_analysis(rr_intervals)
       frequency_domain = analyzer.frequency_domain_analysis(rr_intervals)
       nonlinear = analyzer.nonlinear_analysis(rr_intervals)
       clinical = analyzer.clinical_interpretation(time_domain, frequency_domain, nonlinear)
       
       results[condition] = {
           'time_domain': time_domain,
           'frequency_domain': frequency_domain,
           'nonlinear': nonlinear,
           'clinical': clinical,
           'raw_data': {'time': t, 'rr_intervals': rr_intervals}
       }
   
   # Create comprehensive visualization
   fig, axes = plt.subplots(4, 4, figsize=(20, 16))
   
   # R-R interval time series for each condition
   for i, (condition, data) in enumerate(results.items()):
       ax = axes[0, i]
       t = data['raw_data']['time']
       rr = data['raw_data']['rr_intervals'] * 1000  # Convert to ms
       ax.plot(t, rr, 'b-', alpha=0.7)
       ax.set_title(f'{condition}\nR-R Intervals')
       ax.set_ylabel('R-R Interval (ms)')
       if i == 3:
           ax.set_xlabel('Time (s)')
   
   # Power spectral density comparison
   ax = axes[1, 0]
   colors = ['blue', 'red', 'green', 'purple']
   for i, (condition, data) in enumerate(results.items()):
       if 'frequencies' in data['frequency_domain']:
           freq = data['frequency_domain']['frequencies']
           psd = data['frequency_domain']['psd']
           ax.semilogy(freq, psd, color=colors[i], label=condition, alpha=0.8)
   
   ax.set_xlabel('Frequency (Hz)')
   ax.set_ylabel('PSD (ms²/Hz)')
   ax.set_title('Power Spectral Density Comparison')
   ax.legend()
   ax.set_xlim(0, 0.5)
   
   # Poincaré plot comparison
   for i, (condition, data) in enumerate(results.items()):
       ax = axes[1, i] if i > 0 else axes[1, 1]
       if i == 0:
           continue  # Skip first position (used for PSD)
           
       rr = data['raw_data']['rr_intervals'] * 1000
       rr1 = rr[:-1]
       rr2 = rr[1:]
       
       ax.scatter(rr1, rr2, alpha=0.6, s=10)
       ax.set_xlabel('RR(n) (ms)')
       ax.set_ylabel('RR(n+1) (ms)')
       ax.set_title(f'{condition}\nPoincaré Plot')
       
       # Add ellipse fit
       sd1 = data['nonlinear']['sd1']
       sd2 = data['nonlinear']['sd2']
       
       center = (np.mean(rr1), np.mean(rr2))
       angle = 45  # degrees
       
       from matplotlib.patches import Ellipse
       ellipse = Ellipse(center, 2*sd2, 2*sd1, angle=angle, 
                        fill=False, color='red', linewidth=2)
       ax.add_patch(ellipse)
   
   # Time-domain parameters comparison
   ax = axes[2, 0]
   conditions_list = list(results.keys())
   time_metrics = ['sdnn', 'rmssd', 'pnn50']
   
   x = np.arange(len(conditions_list))
   width = 0.25
   
   for i, metric in enumerate(time_metrics):
       values = [results[cond]['time_domain'][metric] for cond in conditions_list]
       ax.bar(x + i*width, values, width, label=metric.upper(), alpha=0.8)
   
   ax.set_xlabel('Condition')
   ax.set_ylabel('Value')
   ax.set_title('Time-Domain Parameters')
   ax.set_xticks(x + width)
   ax.set_xticklabels(conditions_list, rotation=45)
   ax.legend()
   
   # Frequency-domain parameters
   ax = axes[2, 1]
   freq_metrics = ['lf_power', 'hf_power', 'lf_hf_ratio']
   
   # Normalize powers and handle ratio separately
   lf_powers = [results[cond]['frequency_domain'].get('lf_power', 0) for cond in conditions_list]
   hf_powers = [results[cond]['frequency_domain'].get('hf_power', 0) for cond in conditions_list]
   lf_hf_ratios = [results[cond]['frequency_domain'].get('lf_hf_ratio', 0) for cond in conditions_list]
   
   ax2 = ax.twinx()
   
   bars1 = ax.bar(x - width/2, lf_powers, width, label='LF Power', alpha=0.8)
   bars2 = ax.bar(x + width/2, hf_powers, width, label='HF Power', alpha=0.8)
   line = ax2.plot(x, lf_hf_ratios, 'ro-', label='LF/HF Ratio', markersize=8)
   
   ax.set_xlabel('Condition')
   ax.set_ylabel('Power (ms²)', color='blue')
   ax2.set_ylabel('LF/HF Ratio', color='red')
   ax.set_title('Frequency-Domain Parameters')
   ax.set_xticks(x)
   ax.set_xticklabels(conditions_list, rotation=45)
   
   # Combine legends
   lines1, labels1 = ax.get_legend_handles_labels()
   lines2, labels2 = ax2.get_legend_handles_labels()
   ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
   
   # Nonlinear parameters
   ax = axes[2, 2]
   nonlinear_metrics = ['sd1', 'sd2', 'dfa_alpha']
   
   # Create separate scales for different metrics
   sd1_values = [results[cond]['nonlinear']['sd1'] for cond in conditions_list]
   sd2_values = [results[cond]['nonlinear']['sd2'] for cond in conditions_list]
   dfa_values = [results[cond]['nonlinear'].get('dfa_alpha', np.nan) for cond in conditions_list]
   
   ax.bar(x - width, sd1_values, width, label='SD1', alpha=0.8)
   ax.bar(x, sd2_values, width, label='SD2', alpha=0.8)
   
   ax2 = ax.twinx()
   ax2.bar(x + width, dfa_values, width, label='DFA α', alpha=0.8, color='green')
   
   ax.set_xlabel('Condition')
   ax.set_ylabel('SD1/SD2 (ms)', color='blue')
   ax2.set_ylabel('DFA Alpha', color='green')
   ax.set_title('Nonlinear Parameters')
   ax.set_xticks(x)
   ax.set_xticklabels(conditions_list, rotation=45)
   
   lines1, labels1 = ax.get_legend_handles_labels()
   lines2, labels2 = ax2.get_legend_handles_labels()
   ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
   
   # Clinical interpretation summary
   ax = axes[2, 3]
   ax.axis('off')
   
   # Create clinical summary text
   clinical_summary = "Clinical Interpretations:\n" + "="*25 + "\n\n"
   
   for condition, data in results.items():
       clinical_summary += f"{condition}:\n"
       for param, interpretation in data['clinical'].items():
           clinical_summary += f"  {param}: {interpretation}\n"
       clinical_summary += "\n"
   
   ax.text(0.05, 0.95, clinical_summary, transform=ax.transAxes, 
          fontsize=8, verticalalignment='top', fontfamily='monospace')
   
   # DFA scaling plots
   for i, (condition, data) in enumerate(results.items()):
       ax = axes[3, i]
       
       if 'dfa_scales' in data['nonlinear'] and data['nonlinear']['dfa_scales'] is not None:
           scales = data['nonlinear']['dfa_scales']
           fluctuations = data['nonlinear']['dfa_fluctuations']
           alpha = data['nonlinear']['dfa_alpha']
           
           ax.loglog(scales, fluctuations, 'bo-', markersize=4, alpha=0.7)
           
           # Plot fitted line
           fitted_line = np.power(scales, alpha) * fluctuations[0] / np.power(scales[0], alpha)
           ax.loglog(scales, fitted_line, 'r--', 
                    label=f'α = {alpha:.3f}', linewidth=2)
           
           ax.set_xlabel('Scale (beats)')
           ax.set_ylabel('Fluctuation')
           ax.set_title(f'{condition}\nDFA Scaling')
           ax.legend()
           ax.grid(True, alpha=0.3)
       else:
           ax.text(0.5, 0.5, 'DFA analysis\nfailed', ha='center', va='center',
                  transform=ax.transAxes)
   
   plt.tight_layout()
   plt.show()
   
   # Print comprehensive results
   print("Heart Rate Variability Analysis Summary:")
   print("=" * 60)
   
   for condition, data in results.items():
       print(f"\n{condition.upper()}:")
       print("-" * len(condition))
       
       td = data['time_domain']
       print(f"Time Domain:")
       print(f"  Mean R-R: {td['mean_rr']:.1f} ms")
       print(f"  SDNN: {td['sdnn']:.1f} ms")
       print(f"  RMSSD: {td['rmssd']:.1f} ms")
       print(f"  pNN50: {td['pnn50']:.1f}%")
       
       fd = data['frequency_domain']
       if fd:
           print(f"Frequency Domain:")
           print(f"  LF Power: {fd.get('lf_power', 0):.1f} ms²")
           print(f"  HF Power: {fd.get('hf_power', 0):.1f} ms²")
           print(f"  LF/HF Ratio: {fd.get('lf_hf_ratio', 0):.2f}")
       
       nl = data['nonlinear']
       print(f"Nonlinear:")
       print(f"  SD1: {nl['sd1']:.1f} ms")
       print(f"  SD2: {nl['sd2']:.1f} ms")
       print(f"  SD1/SD2: {nl['sd1_sd2_ratio']:.3f}")
       if not np.isnan(nl.get('dfa_alpha', np.nan)):
           print(f"  DFA α: {nl['dfa_alpha']:.3f}")
       
       print(f"Clinical Interpretation:")
       for param, interp in data['clinical'].items():
           print(f"  {param}: {interp}")
   ```

3. **Connection to Chapter Exercises:**
   This comprehensive HRV analysis provides the foundation for biomedical time series exercises, demonstrating how to extract clinically relevant features from physiological signals using multiple analytical approaches.

### Worked Example 4: Industrial Time Series - Anomaly Detection in Sensor Networks
**Context:** This example demonstrates real-time anomaly detection in industrial settings, preparing students for complex monitoring and fault detection scenarios.

1. **Theoretical Background:**
   - Industrial systems generate multivariate time series from sensor networks
   - Anomalies can indicate equipment failures, process deviations, or security breaches
   - Real-time detection requires efficient algorithms and proper false alarm control

2. **Example:**
   Let's implement a comprehensive industrial anomaly detection system.
   
   **Step 1:** Generate realistic industrial sensor data
   ```python
   import numpy as np
   import pandas as pd
   from sklearn.ensemble import IsolationForest
   from sklearn.preprocessing import StandardScaler
   from sklearn.decomposition import PCA
   
   class IndustrialDataGenerator:
       def __init__(self, seed=42):
           np.random.seed(seed)
           
       def generate_sensor_data(self, n_hours=24*7, n_sensors=10, 
                              anomaly_probability=0.02):
           """Generate multivariate sensor data with anomalies"""
           
           n_samples = n_hours * 60  # Minute-level data
           timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='T')
           
           # Normal operating conditions
           data = {}
           
           # Temperature sensors (correlated)
           temp_base = 75 + 5 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 60))  # Daily cycle
           
           for i in range(3):
               sensor_drift = 0.1 * i  # Small sensor-specific offsets
               noise = np.random.normal(0, 1, n_samples)
               data[f'temp_{i+1}'] = temp_base + sensor_drift + noise
           
           # Pressure sensors (process-related)
           pressure_base = 100 + 2 * np.random.randn(n_samples).cumsum() / np.sqrt(n_samples)  # Random walk
           
           for i in range(2):
               sensor_noise = np.random.normal(0, 0.5, n_samples)
               data[f'pressure_{i+1}'] = pressure_base + sensor_noise
           
           # Vibration sensors (machinery health)
           vibration_base = 0.5 + 0.1 * np.sin(2 * np.pi * np.arange(n_samples) / 60)  # Hourly variation
           
           for i in range(2):
               harmonic_noise = 0.02 * np.sin(2 * np.pi * np.arange(n_samples) * (50 + i*5) / 60)  # Harmonic frequencies
               random_noise = np.random.normal(0, 0.05, n_samples)
               data[f'vibration_{i+1}'] = vibration_base + harmonic_noise + random_noise
           
           # Flow rate sensors
           flow_base = 50 + np.random.normal(0, 2, n_samples)
           
           for i in range(2):
               measurement_noise = np.random.normal(0, 1, n_samples)
               data[f'flow_{i+1}'] = np.maximum(0, flow_base + measurement_noise)  # Non-negative flow
           
           # Power consumption sensor
           power_base = 1000 + 100 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 60))  # Daily pattern
           power_noise = np.random.normal(0, 20, n_samples)
           data['power'] = np.maximum(0, power_base + power_noise)
           
           # Create DataFrame
           df = pd.DataFrame(data, index=timestamps)
           
           # Inject anomalies
           anomaly_indices = np.random.choice(n_samples, 
                                            size=int(n_samples * anomaly_probability), 
                                            replace=False)
           
           anomaly_labels = np.zeros(n_samples)
           anomaly_labels[anomaly_indices] = 1
           
           # Different types of anomalies
           for idx in anomaly_indices:
               anomaly_type = np.random.choice(['sensor_failure', 'process_deviation', 
                                              'equipment_fault', 'cyber_attack'])
               
               if anomaly_type == 'sensor_failure':
                   # Single sensor gives erratic readings
                   faulty_sensor = np.random.choice(df.columns)
                   duration = np.random.randint(1, 30)  # 1-30 minutes
                   end_idx = min(idx + duration, len(df))
                   
                   if 'temp' in faulty_sensor:
                       df.loc[df.index[idx:end_idx], faulty_sensor] = np.random.uniform(0, 150, end_idx - idx)
                   elif 'pressure' in faulty_sensor:
                       df.loc[df.index[idx:end_idx], faulty_sensor] = np.random.uniform(0, 200, end_idx - idx)
                   elif 'vibration' in faulty_sensor:
                       df.loc[df.index[idx:end_idx], faulty_sensor] = np.random.uniform(5, 20, end_idx - idx)
                   
               elif anomaly_type == 'process_deviation':
                   # Coordinated change across multiple sensors
                   duration = np.random.randint(5, 60)
                   end_idx = min(idx + duration, len(df))
                   
                   # Temperature spike
                   for col in [c for c in df.columns if 'temp' in c]:
                       df.loc[df.index[idx:end_idx], col] += np.random.uniform(10, 30)
                   
                   # Corresponding pressure change
                   for col in [c for c in df.columns if 'pressure' in c]:
                       df.loc[df.index[idx:end_idx], col] += np.random.uniform(5, 15)
                       
               elif anomaly_type == 'equipment_fault':
                   # Gradual degradation in vibration patterns
                   duration = np.random.randint(30, 120)
                   end_idx = min(idx + duration, len(df))
                   
                   for col in [c for c in df.columns if 'vibration' in c]:
                       degradation = np.linspace(1, 3, end_idx - idx)
                       df.loc[df.index[idx:end_idx], col] *= degradation
                       
               elif anomaly_type == 'cyber_attack':
                   # Subtle but coordinated changes (data integrity attack)
                   duration = np.random.randint(10, 50)
                   end_idx = min(idx + duration, len(df))
                   
                   # Small bias in multiple sensors
                   affected_sensors = np.random.choice(df.columns, size=3, replace=False)
                   for sensor in affected_sensors:
                       bias = np.random.uniform(-2, 2)
                       df.loc[df.index[idx:end_idx], sensor] += bias
           
           df['anomaly'] = anomaly_labels
           return df
   
   class AnomalyDetector:
       def __init__(self):
           self.models = {}
           self.scalers = {}
           self.thresholds = {}
           
       def fit_statistical_detector(self, train_data, method='isolation_forest'):
           """Fit statistical anomaly detection model"""
           
           # Prepare data (exclude anomaly labels)
           X = train_data.drop('anomaly', axis=1) if 'anomaly' in train_data.columns else train_data
           
           # Standardize data
           scaler = StandardScaler()
           X_scaled = scaler.fit_transform(X)
           
           if method == 'isolation_forest':
               model = IsolationForest(contamination=0.02, random_state=42)
               model.fit(X_scaled)
               
           elif method == 'pca_reconstruction':
               # PCA-based reconstruction error
               model = PCA(n_components=min(5, X.shape[1] - 1))
               model.fit(X_scaled)
               
               # Calculate reconstruction threshold
               X_reconstructed = model.inverse_transform(model.transform(X_scaled))
               reconstruction_errors = np.mean((X_scaled - X_reconstructed)**2, axis=1)
               threshold = np.percentile(reconstruction_errors, 98)  # 98th percentile
               
               self.thresholds[method] = threshold
           
           self.models[method] = model
           self.scalers[method] = scaler
           
           return model
       
       def detect_anomalies(self, data, methods=None):
           """Detect anomalies using fitted models"""
           
           if methods is None:
               methods = list(self.models.keys())
           
           results = {}
           
           # Prepare data
           X = data.drop('anomaly', axis=1) if 'anomaly' in data.columns else data
           
           for method in methods:
               if method not in self.models:
                   continue
               
               model = self.models[method]
               scaler = self.scalers[method]
               
               # Scale data
               X_scaled = scaler.transform(X)
               
               if method == 'isolation_forest':
                   anomaly_scores = model.decision_function(X_scaled)
                   predictions = model.predict(X_scaled)
                   anomalies = predictions == -1
                   
               elif method == 'pca_reconstruction':
                   X_reconstructed = model.inverse_transform(model.transform(X_scaled))
                   reconstruction_errors = np.mean((X_scaled - X_reconstructed)**2, axis=1)
                   anomalies = reconstruction_errors > self.thresholds[method]
                   anomaly_scores = reconstruction_errors
               
               results[method] = {
                   'anomalies': anomalies,
                   'scores': anomaly_scores
               }
           
           return results
       
       def real_time_detection(self, data_stream, window_size=60, update_frequency=10):
           """Real-time anomaly detection on streaming data"""
           
           results = []
           n_samples = len(data_stream)
           
           for i in range(window_size, n_samples, update_frequency):
               # Get current window
               window_data = data_stream.iloc[i-window_size:i]
               
               # Detect anomalies
               detections = self.detect_anomalies(window_data)
               
               # Store results for current window
               window_result = {
                   'timestamp': data_stream.index[i],
                   'window_start': data_stream.index[i-window_size],
                   'window_end': data_stream.index[i-1],
                   'detections': detections
               }
               
               results.append(window_result)
           
           return results
   ```
   
   **Step 3:** Apply comprehensive anomaly detection
   ```python
   # Generate industrial sensor data
   generator = IndustrialDataGenerator()
   sensor_data = generator.generate_sensor_data(n_hours=24*7, anomaly_probability=0.015)
   
   # Split into train/test (first 5 days for training)
   split_point = int(len(sensor_data) * 0.7)
   train_data = sensor_data.iloc[:split_point]
   test_data = sensor_data.iloc[split_point:]
   
   # Initialize and train detector
   detector = AnomalyDetector()
   
   # Fit different models
   detector.fit_statistical_detector(train_data, method='isolation_forest')
   detector.fit_statistical_detector(train_data, method='pca_reconstruction')
   
   # Detect anomalies on test data
   detection_results = detector.detect_anomalies(test_data)
   
   # Real-time detection simulation
   realtime_results = detector.real_time_detection(test_data, window_size=60)
   
   # Create comprehensive visualization
   fig, axes = plt.subplots(4, 3, figsize=(18, 16))
   
   # Raw sensor data with anomalies highlighted
   ax = axes[0, 0]
   temp_sensors = [col for col in sensor_data.columns if 'temp' in col]
   for sensor in temp_sensors:
       ax.plot(sensor_data.index, sensor_data[sensor], alpha=0.7, label=sensor)
   
   # Highlight anomalies
   anomaly_mask = sensor_data['anomaly'] == 1
   ax.scatter(sensor_data.index[anomaly_mask], 
             sensor_data.loc[anomaly_mask, temp_sensors[0]], 
             color='red', s=30, alpha=0.8, label='Anomalies')
   
   ax.set_title('Temperature Sensors')
   ax.set_ylabel('Temperature (°F)')
   ax.legend()
   
   # Pressure sensors
   ax = axes[0, 1]
   pressure_sensors = [col for col in sensor_data.columns if 'pressure' in col]
   for sensor in pressure_sensors:
       ax.plot(sensor_data.index, sensor_data[sensor], alpha=0.7, label=sensor)
   
   ax.scatter(sensor_data.index[anomaly_mask], 
             sensor_data.loc[anomaly_mask, pressure_sensors[0]], 
             color='red', s=30, alpha=0.8, label='Anomalies')
   
   ax.set_title('Pressure Sensors')
   ax.set_ylabel('Pressure (PSI)')
   ax.legend()
   
   # Vibration sensors
   ax = axes[0, 2]
   vibration_sensors = [col for col in sensor_data.columns if 'vibration' in col]
   for sensor in vibration_sensors:
       ax.plot(sensor_data.index, sensor_data[sensor], alpha=0.7, label=sensor)
   
   ax.scatter(sensor_data.index[anomaly_mask], 
             sensor_data.loc[anomaly_mask, vibration_sensors[0]], 
             color='red', s=30, alpha=0.8, label='Anomalies')
   
   ax.set_title('Vibration Sensors')
   ax.set_ylabel('Vibration (g)')
   ax.legend()
   
   # Anomaly detection results comparison
   ax = axes[1, 0]
   
   # True anomalies
   true_anomalies = test_data['anomaly'] == 1
   ax.plot(test_data.index, true_anomalies.astype(int), 'r-', linewidth=2, 
           label='True Anomalies', alpha=0.8)
   
   # Detection results
   for i, (method, results) in enumerate(detection_results.items()):
       detected_anomalies = results['anomalies']
       ax.plot(test_data.index, detected_anomalies.astype(int) + i*0.1, 
              label=f'{method}', alpha=0.7, linewidth=2)
   
   ax.set_title('Anomaly Detection Results')
   ax.set_ylabel('Anomaly Detected')
   ax.legend()
   ax.set_ylim(-0.2, 1.5)
   
   # Anomaly scores
   ax = axes[1, 1]
   for method, results in detection_results.items():
       scores = results['scores']
       ax.plot(test_data.index, scores, alpha=0.7, label=method)
   
   ax.set_title('Anomaly Scores')
   ax.set_ylabel('Score')
   ax.legend()
   
   # Detection performance metrics
   ax = axes[1, 2]
   
   from sklearn.metrics import precision_score, recall_score, f1_score
   
   metrics_data = {'Method': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
   
   true_labels = test_data['anomaly'].values
   
   for method, results in detection_results.items():
       predicted_labels = results['anomalies'].astype(int)
       
       if np.sum(predicted_labels) > 0:  # Avoid division by zero
           precision = precision_score(true_labels, predicted_labels)
           recall = recall_score(true_labels, predicted_labels)
           f1 = f1_score(true_labels, predicted_labels)
       else:
           precision = recall = f1 = 0
       
       metrics_data['Method'].append(method)
       metrics_data['Precision'].append(precision)
       metrics_data['Recall'].append(recall)
       metrics_data['F1-Score'].append(f1)
   
   x = np.arange(len(metrics_data['Method']))
   width = 0.25
   
   ax.bar(x - width, metrics_data['Precision'], width, label='Precision', alpha=0.8)
   ax.bar(x, metrics_data['Recall'], width, label='Recall', alpha=0.8)
   ax.bar(x + width, metrics_data['F1-Score'], width, label='F1-Score', alpha=0.8)
   
   ax.set_xlabel('Method')
   ax.set_ylabel('Score')
   ax.set_title('Detection Performance')
   ax.set_xticks(x)
   ax.set_xticklabels(metrics_data['Method'])
   ax.legend()
   
   # Correlation matrix of sensors
   ax = axes[2, 0]
   sensor_cols = [col for col in sensor_data.columns if col != 'anomaly']
   correlation_matrix = sensor_data[sensor_cols].corr()
   
   im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
   ax.set_xticks(range(len(sensor_cols)))
   ax.set_yticks(range(len(sensor_cols)))
   ax.set_xticklabels(sensor_cols, rotation=45)
   ax.set_yticklabels(sensor_cols)
   ax.set_title('Sensor Correlation Matrix')
   
   # Add colorbar
   plt.colorbar(im, ax=ax)
   
   # PCA visualization
   ax = axes[2, 1]
   pca_model = detector.models['pca_reconstruction']
   scaler = detector.scalers['pca_reconstruction']
   
   # Transform test data
   test_features = test_data.drop('anomaly', axis=1)
   test_scaled = scaler.transform(test_features)
   test_pca = pca_model.transform(test_scaled)
   
   # Plot first two principal components
   normal_mask = test_data['anomaly'] == 0
   anomaly_mask = test_data['anomaly'] == 1
   
   ax.scatter(test_pca[normal_mask, 0], test_pca[normal_mask, 1], 
             c='blue', alpha=0.6, s=10, label='Normal')
   ax.scatter(test_pca[anomaly_mask, 0], test_pca[anomaly_mask, 1], 
             c='red', alpha=0.8, s=30, label='Anomalies')
   
   ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} var)')
   ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%} var)')
   ax.set_title('PCA Space Visualization')
   ax.legend()
   
   # Real-time detection timeline
   ax = axes[2, 2]
   
   # Extract detection timeline
   realtime_timestamps = [result['timestamp'] for result in realtime_results]
   isolation_detections = []
   pca_detections = []
   
   for result in realtime_results:
       detections = result['detections']
       
       # Count detections in window
       iso_count = np.sum(detections.get('isolation_forest', {}).get('anomalies', []))
       pca_count = np.sum(detections.get('pca_reconstruction', {}).get('anomalies', []))
       
       isolation_detections.append(iso_count)
       pca_detections.append(pca_count)
   
   ax.plot(realtime_timestamps, isolation_detections, 'b-o', 
           label='Isolation Forest', markersize=4, alpha=0.8)
   ax.plot(realtime_timestamps, pca_detections, 'g-s', 
           label='PCA Reconstruction', markersize=4, alpha=0.8)
   
   ax.set_xlabel('Time')
   ax.set_ylabel('Anomalies Detected in Window')
   ax.set_title('Real-time Detection Timeline')
   ax.legend()
   
   # Feature importance/contribution analysis
   ax = axes[3, 0]
   
   # For isolation forest, we can't directly get feature importance
   # But we can analyze which features deviate most during anomalies
   anomaly_data = test_data[test_data['anomaly'] == 1].drop('anomaly', axis=1)
   normal_data = test_data[test_data['anomaly'] == 0].drop('anomaly', axis=1)
   
   if len(anomaly_data) > 0 and len(normal_data) > 0:
       # Calculate feature deviation during anomalies
       normal_means = normal_data.mean()
       anomaly_means = anomaly_data.mean()
       
       deviations = np.abs(anomaly_means - normal_means) / normal_data.std()
       
       bars = ax.bar(range(len(deviations)), deviations.values, alpha=0.8)
       ax.set_xlabel('Sensor')
       ax.set_ylabel('Standardized Deviation')
       ax.set_title('Feature Deviations During Anomalies')
       ax.set_xticks(range(len(deviations)))
       ax.set_xticklabels(deviations.index, rotation=45)
       
       # Highlight top contributors
       top_indices = np.argsort(deviations.values)[-3:]
       for idx in top_indices:
           bars[idx].set_color('red')
   
   # Sensor health monitoring
   ax = axes[3, 1]
   
   # Calculate rolling statistics for each sensor
   window_size = 60  # 1 hour
   sensor_health = {}
   
   for col in sensor_cols:
       rolling_mean = sensor_data[col].rolling(window_size).mean()
       rolling_std = sensor_data[col].rolling(window_size).std()
       
       # Health score based on stability
       cv = rolling_std / rolling_mean  # Coefficient of variation
       health_score = 1 / (1 + cv)  # Higher score = more stable
       sensor_health[col] = health_score.fillna(1)
   
   # Plot average health scores
   avg_health = pd.DataFrame(sensor_health).mean()
   
   bars = ax.bar(range(len(avg_health)), avg_health.values, alpha=0.8)
   ax.set_xlabel('Sensor')
   ax.set_ylabel('Health Score')
   ax.set_title('Sensor Health Assessment')
   ax.set_xticks(range(len(avg_health)))
   ax.set_xticklabels(avg_health.index, rotation=45)
   ax.set_ylim(0, 1)
   
   # Color-code health scores
   for i, (bar, score) in enumerate(zip(bars, avg_health.values)):
       if score < 0.7:
           bar.set_color('red')
       elif score < 0.85:
           bar.set_color('orange')
       else:
           bar.set_color('green')
   
   # Alert summary
   ax = axes[3, 2]
   ax.axis('off')
   
   # Create alert summary
   alert_text = "Anomaly Detection Summary\n" + "="*30 + "\n\n"
   
   total_anomalies = np.sum(sensor_data['anomaly'])
   alert_text += f"Total Anomalies: {total_anomalies}\n"
   alert_text += f"Anomaly Rate: {total_anomalies/len(sensor_data)*100:.2f}%\n\n"
   
   alert_text += "Detection Performance:\n"
   for i, method in enumerate(metrics_data['Method']):
       precision = metrics_data['Precision'][i]
       recall = metrics_data['Recall'][i]
       f1 = metrics_data['F1-Score'][i]
       
       alert_text += f"\n{method}:\n"
       alert_text += f"  Precision: {precision:.3f}\n"
       alert_text += f"  Recall: {recall:.3f}\n"
       alert_text += f"  F1-Score: {f1:.3f}\n"
   
   # Critical sensors
   critical_sensors = avg_health[avg_health < 0.8].index.tolist()
   if critical_sensors:
       alert_text += f"\nCritical Sensors:\n"
       for sensor in critical_sensors:
           alert_text += f"  {sensor}: {avg_health[sensor]:.3f}\n"
   
   alert_text += f"\nRecommendations:\n"
   if critical_sensors:
       alert_text += "- Inspect critical sensors\n"
   if metrics_data['Recall'][0] < 0.8:  # Assuming first method
       alert_text += "- Tune detection sensitivity\n"
   alert_text += "- Schedule preventive maintenance\n"
   
   ax.text(0.05, 0.95, alert_text, transform=ax.transAxes, 
          fontsize=9, verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
   
   plt.tight_layout()
   plt.show()
   
   # Print comprehensive results
   print("Industrial Anomaly Detection Summary:")
   print("=" * 50)
   
   print(f"\nDataset Statistics:")
   print(f"  Total samples: {len(sensor_data):,}")
   print(f"  Training samples: {len(train_data):,}")
   print(f"  Test samples: {len(test_data):,}")
   print(f"  True anomalies in test: {np.sum(test_data['anomaly'])}")
   print(f"  Anomaly rate: {np.sum(test_data['anomaly'])/len(test_data)*100:.2f}%")
   
   print(f"\nDetection Results:")
   for method, results in detection_results.items():
       detected = np.sum(results['anomalies'])
       print(f"\n{method}:")
       print(f"  Anomalies detected: {detected}")
       print(f"  Detection rate: {detected/len(test_data)*100:.2f}%")
   
   print(f"\nPerformance Metrics:")
   for i, method in enumerate(metrics_data['Method']):
       print(f"\n{method}:")
       print(f"  Precision: {metrics_data['Precision'][i]:.3f}")
       print(f"  Recall: {metrics_data['Recall'][i]:.3f}")
       print(f"  F1-Score: {metrics_data['F1-Score'][i]:.3f}")
   
   print(f"\nSensor Health Status:")
   for sensor, health in avg_health.items():
       status = 'CRITICAL' if health < 0.7 else 'WARNING' if health < 0.85 else 'GOOD'
       print(f"  {sensor:>15s}: {health:.3f} ({status})")
   
   print(f"\nReal-time Processing:")
   print(f"  Processing windows: {len(realtime_results)}")
   avg_detections = np.mean([np.sum(r['detections']['isolation_forest']['anomalies']) 
                           for r in realtime_results])
   print(f"  Avg detections per window: {avg_detections:.2f}")
   ```

3. **Connection to Chapter Exercises:**
   This comprehensive industrial anomaly detection system provides the foundation for tackling complex monitoring scenarios, demonstrating real-time detection, multivariate analysis, and practical deployment considerations essential for industrial applications.

These worked examples provide hands-on experience with the key application domains covered in Chapter 13, preparing students to tackle domain-specific challenges while demonstrating the versatility and power of time series analysis techniques across different fields.