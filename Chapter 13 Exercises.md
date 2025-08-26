# Chapter 13 Exercises

## Exercises

### Financial Time Series

**Exercise 13.1** (Feynman Style)
Imagine you're watching the stock market. You notice that whenever there's a big drop on Monday, people say "It's just the Monday effect" or "Markets are correcting after the weekend." But when there's a big rise, they have completely different explanations.

a) Design a simple experiment to test whether Monday returns are actually different from other weekdays. What would constitute evidence of a real "Monday effect"?
b) Now imagine you discover that Mondays really do have lower returns on average. Does this mean you can make money by shorting stocks every Monday? Why or why not?
c) Think about this paradox: If everyone knew about a profitable pattern, what would happen to that pattern? Use this to explain the concept of market efficiency.

**Exercise 13.2** (Gelman Style - GARCH Reality Check)
Real financial data is messy and assumptions matter. Let's explore this:

a) Download real stock return data (e.g., S&P 500) for the last 20 years.
b) Fit GARCH(1,1) models to rolling 2-year windows. Track how the parameters evolve over time.
c) For each window, test whether the standardized residuals are actually normal (use Q-Q plots and formal tests).
d) Now add external shocks: identify major events (2008 crisis, COVID-19) and show how GARCH parameters behave around these events.
e) Implement a regime-switching GARCH model. Does it handle the crisis periods better?
f) Write a diagnostic function that automatically detects when a GARCH model is failing.

### Environmental and Climate Analysis

**Exercise 13.3** (Mixed Voices - Climate Change Detection)
[Feynman] Suppose you're given temperature data from a weather station that's been operating since 1900. How would you convince a skeptic that there's a real warming trend and not just natural variation?

[Jaynes] From an information-theoretic perspective, we want to extract the signal (trend) from the noise (natural variability) in an optimal way.

[Gelman] But we need to be careful about multiple testing and cherry-picking. If we look at enough stations, we'll find trends by chance.

Your task:
a) Download real temperature data from multiple stations (use NOAA or Berkeley Earth datasets).
b) Implement at least three different trend detection methods:
   - Simple linear regression
   - Bayesian structural time series
   - Nonparametric methods (e.g., Mann-Kendall test)
c) Account for autocorrelation in your significance tests.
d) Perform a meta-analysis across multiple stations to get a regional trend estimate.
e) Create a visualization that honestly represents both the trend and the uncertainty.

**Exercise 13.4** (Murphy Style - Extreme Event Prediction)
Build a system for predicting extreme weather events:

```python
class ExtremeEventPredictor:
    def __init__(self, event_type='heatwave'):
        """
        Initialize extreme event prediction system.
        
        Parameters:
        -----------
        event_type : str
            Type of event ('heatwave', 'drought', 'flood')
        """
        pass
    
    def fit_threshold_model(self, data, quantile=0.95):
        """Fit a Generalized Pareto Distribution to exceedances."""
        pass
    
    def predict_return_period(self, magnitude):
        """Estimate return period for event of given magnitude."""
        pass
    
    def forecast_probability(self, forecast_data, threshold):
        """Forecast probability of exceeding threshold."""
        pass
```

Requirements:
- Implement both block maxima and peak-over-threshold approaches
- Include covariates (e.g., ENSO indices for weather extremes)
- Handle non-stationarity in extreme value parameters
- Provide uncertainty quantification for all predictions

### Biomedical Applications

**Exercise 13.5** (Feynman Style - Heart Rate Mysteries)
Your heart doesn't beat like a metronome - the time between beats varies. This Heart Rate Variability (HRV) contains rich information about your health.

a) Generate synthetic heartbeat data: start with a regular rhythm, then add respiratory sinus arrhythmia (heart speeds up when you inhale).
b) Now add complexity: make the variability itself variable. What might cause this in a real heart?
c) Implement detrended fluctuation analysis (DFA) to quantify the fractal properties of HRV.
d) Create a simple model where reduced HRV predicts adverse events. How early can you detect problems?

**Exercise 13.6** (Comprehensive EEG Analysis)
Build a complete pipeline for analyzing EEG data:

a) Simulate multi-channel EEG data with:
   - Alpha waves (8-13 Hz) that increase with eyes closed
   - Beta waves (13-30 Hz) during concentration
   - Artifacts from eye blinks and muscle movement

b) Implement preprocessing:
   - Bandpass filtering
   - Independent Component Analysis (ICA) for artifact removal
   - Re-referencing (common average reference)

c) Feature extraction:
   - Spectral power in each frequency band
   - Connectivity measures (coherence, phase-locking value)
   - Entropy measures (sample entropy, permutation entropy)

d) Classification task:
   - Use the features to classify cognitive states (e.g., relaxed vs. focused)
   - Compare multiple classifiers (SVM, Random Forest, neural networks)
   - Implement proper cross-validation for time series data

e) Real-time analysis:
   - Modify your pipeline to work with streaming data
   - Implement a sliding window approach with <100ms latency

### Economic Policy Analysis

**Exercise 13.7** (Jaynes Style - Information and Monetary Policy)
Central banks must make decisions based on incomplete, noisy, and delayed economic data.

a) Formulate monetary policy as a Bayesian decision problem where the central bank has a loss function and must choose interest rates based on uncertain estimates of inflation and output gaps.
b) Show that under quadratic loss and Gaussian uncertainty, the optimal policy is certainty equivalence (act as if estimates were true values).
c) Now introduce model uncertainty. Show how robust control methods lead to more conservative policies.
d) Implement a simple New Keynesian model and demonstrate how measurement error affects optimal policy.
e) Quantify the value of information: how much would perfect foreknowledge of next quarter's inflation be worth?

**Exercise 13.8** (Policy Intervention Analysis)
Analyze the causal impact of a real economic policy:

a) Choose a specific policy intervention (e.g., minimum wage change, quantitative easing, tax reform).
b) Gather relevant time series data before and after the intervention.
c) Implement synthetic control methods to construct a counterfactual.
d) Use multiple approaches:
   - Difference-in-differences
   - Regression discontinuity (if applicable)
   - Bayesian structural time series
e) Perform sensitivity analysis:
   - How do results change with different control groups?
   - What if there are spillover effects?
   - How robust are results to model specification?
f) Write a policy brief summarizing your findings for non-technical audiences.

### Industrial Applications

**Exercise 13.9** (Murphy Style - Smart Manufacturing)
Design an end-to-end anomaly detection system for manufacturing:

```python
class ManufacturingMonitor:
    def __init__(self, process_type='continuous'):
        self.models = {}
        self.thresholds = {}
        
    def train_models(self, sensor_data, labels=None):
        """
        Train multiple anomaly detection models.
        Include: Isolation Forest, LSTM-Autoencoder, One-class SVM, 
        Statistical Process Control charts
        """
        pass
    
    def detect_anomalies(self, new_data, ensemble=True):
        """
        Detect anomalies using trained models.
        If ensemble=True, combine predictions from all models.
        """
        pass
    
    def explain_anomaly(self, anomaly_data):
        """
        Provide interpretable explanation for detected anomaly.
        Use SHAP values or similar for feature importance.
        """
        pass
    
    def update_online(self, new_normal_data):
        """
        Update models with new normal operating data.
        Handle concept drift.
        """
        pass
```

Test your system on:
- Gradual drift (e.g., sensor degradation)
- Sudden changes (e.g., raw material change)
- Rare events (e.g., equipment failure)
- Cyclic patterns (e.g., maintenance cycles)

### Cross-Domain Challenge

**Exercise 13.10** (Philosophical Integration)
[Feynman] Nature doesn't care about our disciplinary boundaries. Many patterns appear across different domains.

[Jaynes] Information theory provides a unifying framework for understanding time series across all domains.

[Gelman] But we should be skeptical of universal laws. Context always matters.

[Murphy] Let's build something that works across domains.

Your challenge:
a) Identify a pattern that appears in at least three different domains from this chapter (e.g., power laws, cycles, regime switches).
b) Implement a generic detection algorithm for this pattern.
c) Apply it to:
   - Financial data (stock volatility)
   - Environmental data (earthquake magnitudes)
   - Biomedical data (neural avalanches)
   - Economic data (firm sizes)
   - Industrial data (failure times)
d) For each domain:
   - Explain why this pattern might emerge
   - Discuss domain-specific considerations
   - Show how to adapt the generic method
e) Write a synthesis discussing:
   - What's universal vs. domain-specific
   - When cross-domain insights are valuable
   - Dangers of naive pattern-matching across domains

### Real-World Project

**Exercise 13.11** (Complete System Integration)
Build a monitoring dashboard that combines multiple data streams:

Scenario: You're monitoring a city's health using diverse time series:
- Air quality sensors (PM2.5, O3, NO2)
- Traffic flow data
- Hospital admissions
- Social media sentiment
- Weather data

Requirements:
a) Data integration:
   - Handle different sampling frequencies
   - Deal with missing data and sensor failures
   - Align data streams temporally

b) Analysis pipeline:
   - Detect anomalies in each stream
   - Identify correlations between streams
   - Forecast next 24 hours for each metric

c) Causal analysis:
   - Test Granger causality between streams
   - Account for confounders (e.g., weekday effects)
   - Identify leading indicators of health emergencies

d) Visualization:
   - Real-time dashboard with multiple panels
   - Alert system for unusual patterns
   - Interactive exploration tools

e) Decision support:
   - Recommend interventions based on forecasts
   - Quantify uncertainty in recommendations
   - Provide interpretable explanations

f) Validation:
   - Backtest on historical data
   - Compare with simpler baseline methods
   - Document limitations and failure modes