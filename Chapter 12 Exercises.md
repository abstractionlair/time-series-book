# Chapter 12: Time Series Forecasting - Exercises

### Fundamental Concepts

**Exercise 12.1** (Feynman Style)
Think about weather forecasting. You check the forecast every morning, and it says there's a 30% chance of rain. After a year of tracking, you find that on days with "30% chance of rain," it actually rained about 30% of the time.

a) What does this tell you about the quality of the probabilistic forecasts?
b) Now imagine a different forecaster who always predicts either 0% or 100% chance of rain. Could this forecaster have the same accuracy rate? What's the fundamental difference?
c) Design a simple experiment using coin flips to demonstrate why probabilistic forecasts contain more information than binary predictions.

**Exercise 12.2** (Jaynes Style)
Consider the problem of optimal forecasting from an information-theoretic perspective. Let Y_{t+h} be the value we want to forecast h steps ahead, and let I_t be all information available at time t.

a) Show that the optimal forecast minimizing expected squared error is E[Y_{t+h} | I_t].
b) Prove that for a stationary Gaussian AR(1) process with parameter φ, the h-step forecast uncertainty grows as σ²(1 - φ^{2h})/(1 - φ²).
c) Derive the limiting forecast distribution as h → ∞ and interpret this result in terms of information decay.

### Classical Methods Implementation

**Exercise 12.3** (Gelman Style)
Real forecasting is messier than textbooks suggest. Let's explore this reality.

a) Generate three time series: one with a trend, one with seasonality, and one with both. Add realistic features like outliers and level shifts.
b) Apply moving average, exponential smoothing, and ARIMA forecasting to each series.
c) Now break each series: add missing values, introduce measurement error, and create a structural break halfway through. How do your forecasts degrade?
d) Implement diagnostic checks: residual analysis, forecast error distributions, and calibration plots.
e) Write a function that automatically detects which type of forecasting method is most appropriate for a given series.

**Exercise 12.4** (Murphy Style)
Implement an efficient forecasting system that can handle streaming data:

```python
class OnlineForecaster:
    def __init__(self, method='exponential', window_size=100):
        """
        Initialize online forecasting system.
        
        Parameters:
        -----------
        method : str
            Forecasting method ('exponential', 'arima', 'kalman')
        window_size : int
            Size of sliding window for model updates
        """
        pass
    
    def update(self, new_observation):
        """Update model with new observation."""
        pass
    
    def forecast(self, horizon):
        """Generate forecast for given horizon."""
        pass
    
    def get_prediction_intervals(self, horizon, alpha=0.05):
        """Get prediction intervals."""
        pass
```

Requirements:
- O(1) time complexity for updates
- Automatic detection and handling of concept drift
- Support for missing values and irregular timestamps
- Include at least three different forecasting methods

### Bayesian Forecasting

**Exercise 12.5** (Mixed Voices)
[Jaynes] Consider a Bayesian approach to forecasting with model uncertainty. We have M competing models for our time series, each with prior probability P(M_i).

[Gelman] But in practice, all models are wrong. Some are just useful in different contexts.

Your task:
a) Implement Bayesian Model Averaging (BMA) for time series forecasting using PyMC3.
b) Include at least: AR(1), AR(2), MA(1), and a structural time series model.
c) Use the Watanabe-Akaike Information Criterion (WAIC) for model comparison.
d) Apply your method to real economic data (e.g., GDP growth) and show how model weights evolve over time.
e) Compare BMA forecasts with selecting a single "best" model. When does each approach win?

**Exercise 12.6** (Probabilistic Forecasting Deep Dive)
Implement a complete probabilistic forecasting framework:

a) Create a function that generates forecast distributions using:
   - Parametric bootstrap for ARIMA models
   - MCMC for Bayesian models
   - Conformal prediction for distribution-free intervals

b) Implement multiple scoring rules for probabilistic forecasts:
   - Continuous Ranked Probability Score (CRPS)
   - Log score
   - Interval score
   - Quantile score

c) Create visualization tools:
   - Fan charts showing prediction intervals
   - Reliability diagrams (PIT histograms)
   - Murphy diagrams for scoring rule comparison

d) Test on the M5 competition data and compare your results with the winning solutions.

### Advanced Techniques

**Exercise 12.7** (Ensemble Methods)
[Murphy] Design a meta-learning system for forecast combination:

a) Implement at least 5 base forecasters (e.g., ARIMA, ETS, Prophet, Random Forest, LSTM).
b) Create three combination strategies:
   - Simple averaging
   - Weighted averaging based on recent performance
   - Stacked generalization using a meta-model

c) Implement online learning for weight updates as new data arrives.
d) Add a feature that detects when certain models should be excluded from the ensemble.
e) Compare computational cost vs. accuracy improvement for different ensemble sizes.

**Exercise 12.8** (Long-term Scenario Analysis)
[Feynman] Imagine you're trying to forecast global temperature 50 years into the future. You know this is like predicting where a specific molecule in a gas will be after billions of collisions - essentially impossible in detail, but perhaps possible in aggregate.

Implement a scenario-based forecasting system:
a) Create a structural model with multiple sources of uncertainty:
   - Parameter uncertainty (use Bayesian inference)
   - Model uncertainty (use multiple model forms)
   - Scenario uncertainty (different assumption sets)

b) Generate 1000 scenarios and cluster them into representative narratives.
c) Implement sensitivity analysis to identify which uncertainties matter most.
d) Create an interactive visualization showing how scenarios diverge over time.
e) Apply to a real problem: forecast renewable energy adoption, population growth, or economic development.

### Theoretical Challenges

**Exercise 12.9** (Information-Theoretic Limits)
[Jaynes] Explore the fundamental limits of predictability:

a) For an AR(p) process, derive the mutual information between the present state and future values as a function of forecast horizon.
b) Implement the Grassberger-Procaccia algorithm to estimate the correlation dimension of a chaotic time series.
c) Show empirically that for chaotic systems (use the Lorenz system), forecast error grows exponentially with horizon.
d) Implement the permutation entropy to quantify predictability in real-world time series.
e) Create a function that estimates the "predictability horizon" - beyond which forecasts are no better than climatology.

### Real-World Project

**Exercise 12.10** (Complete Forecasting Pipeline)
Build a production-ready forecasting system for electricity demand:

a) Data ingestion and preprocessing:
   - Handle multiple data sources (weather, calendar, historical demand)
   - Detect and impute anomalies
   - Create appropriate features (see Chapter 9.1)

b) Model development:
   - Implement hierarchical forecasting for multiple geographic levels
   - Include weather-based covariates
   - Handle special events (holidays, sports events)

c) Forecast generation:
   - Generate probabilistic forecasts at multiple horizons (1 hour to 1 week)
   - Implement forecast reconciliation to ensure hierarchical coherence
   - Include extreme event warnings

d) Evaluation and monitoring:
   - Implement proper time series cross-validation
   - Create automated reports with forecast performance metrics
   - Set up alerts for forecast degradation

e) Optimization:
   - Use your forecasts to solve a simple economic dispatch problem
   - Show how forecast uncertainty affects optimal decisions
   - Quantify the economic value of improved forecasts

Requirements:
- Must handle at least 1 year of hourly data efficiently
- Include both point and probabilistic forecasts
- Provide interpretable outputs for non-technical users
- Document all modeling choices and assumptions