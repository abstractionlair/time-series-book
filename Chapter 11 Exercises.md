# Chapter 11: Causal Inference in Time Series - Exercises

### Conceptual Understanding

**Exercise 11.1** (Feynman Style)
Let's think about causality in a simple everyday scenario. You notice that every time your neighbor starts their old car in the morning, your kitchen radio gets static. Later, you observe that on days when the neighbor doesn't drive to work, there's no static, even though other cars pass by your house.

a) Is the neighbor's car Granger-causing the radio static? Explain why or why not.
b) Design a simple experiment to test whether this is true causation or just predictive association.
c) What hidden confounders might explain both phenomena without direct causation?

**Exercise 11.2** (Jaynes Style)
Consider the logical foundations of Granger causality. We define X to Granger-cause Y if P(Y_t|Y_{t-1}, X_{t-1}) ≠ P(Y_t|Y_{t-1}).

a) Prove that Granger causality is not symmetric: X Granger-causing Y does not imply Y Granger-causes X.
b) Show that if X and Y are jointly Gaussian and X Granger-causes Y, then the transfer entropy from X to Y is positive.
c) Demonstrate that Granger causality can exist even when there is no true causal relationship. Provide a concrete example with a hidden common cause.

### Implementation Exercises

**Exercise 11.3** (Murphy Style)
Implement a comprehensive Granger causality testing framework:

```python
def granger_causality_suite(data, var_names, max_lag=10):
    """
    Test for Granger causality between all pairs of variables.
    
    Parameters:
    -----------
    data : array-like, shape (T, n_vars)
        Time series data
    var_names : list of str
        Variable names
    max_lag : int
        Maximum lag to test
    
    Returns:
    --------
    results : dict
        Dictionary with test statistics, p-values, and optimal lag selection
    """
    # Your implementation here
    pass
```

a) Implement the function using both F-tests and information criteria (AIC/BIC).
b) Add functionality to handle non-stationary series using differencing or detrending.
c) Implement a bootstrap procedure to compute confidence intervals for the test statistics.
d) Profile your code and optimize the most computationally expensive operations.

**Exercise 11.4** (Gelman Style)
Real data is messy. Let's explore how violations of assumptions affect causal inference:

a) Generate synthetic data where X causes Y with a time-varying effect β_t = β_0 + β_1 * sin(2πt/T).
b) Apply standard Granger causality tests. Do they detect the relationship?
c) Implement a rolling-window Granger test and plot how the detected causality changes over time.
d) What happens when you add measurement error to X? How does the signal-to-noise ratio affect your conclusions?
e) Write a diagnostic function that checks whether your data meets the assumptions for Granger causality testing.

### Advanced Problems

**Exercise 11.5** (Bayesian Causal Discovery)
Implement a Bayesian network structure learning algorithm for time series:

a) Define a prior over directed acyclic graphs (DAGs) that respects temporal ordering.
b) Implement a Metropolis-Hastings sampler to explore the space of possible causal structures.
c) Use your implementation to discover causal relationships in a simulated dataset with known ground truth.
d) How does your method's performance degrade as you add more variables? Propose and implement a strategy for scaling to high dimensions.

**Exercise 11.6** (Causal Impact with Model Uncertainty)
Extend the causal impact analysis to handle model uncertainty:

a) Implement causal impact using three different models: ARIMA, state space, and Gaussian Process.
b) Use Bayesian model averaging to combine the predictions.
c) Apply your method to detect the impact of a known intervention in stock market data (e.g., a major policy announcement).
d) How sensitive are your results to the choice of control series? Implement a procedure to automatically select relevant controls.

### Philosophical Challenges

**Exercise 11.7** (Mixed Voices)
[Feynman] Here's a puzzle that'll make you think: You're studying a complex system - let's say it's the relationship between social media activity, news coverage, and stock prices for a particular company.

[Jaynes] From an information-theoretic perspective, we want to understand the flow of information between these variables. But there's a fundamental challenge: these variables likely influence each other in complex, possibly circular ways.

[Gelman] And here's the thing - in practice, we never measure everything. There are always unmeasured confounders like insider information, coordinated trading strategies, or external events affecting all three variables.

[Murphy] Your task is to design and implement a causal discovery algorithm that can:

a) Handle potential feedback loops (social media → news → stocks → social media).
b) Account for unmeasured confounders using latent variable models.
c) Detect regime changes where causal relationships might shift.
d) Provide not just point estimates but full posterior distributions over causal structures.
e) Scale to handle hourly data over several years.

### Data Analysis Project

**Exercise 11.8** (Comprehensive Causal Analysis)
Perform a complete causal analysis on real-world climate data:

a) Download monthly data for temperature, CO2 levels, solar radiation, and oceanic indices (e.g., ENSO).
b) Test for Granger causality between all pairs of variables at multiple time scales.
c) Use at least two different causal discovery algorithms to infer the causal structure.
d) Identify a major volcanic eruption in the data and quantify its causal impact on temperature.
e) Write a report discussing:
   - Which causal relationships are robust across methods?
   - How do your results change with different preprocessing choices?
   - What are the main limitations of your analysis?
   - How would you design an ideal study to definitively establish causality in this system?

### Computational Challenges

**Exercise 11.9** (High-Performance Causal Discovery)
[Murphy] Implement a GPU-accelerated version of the PC algorithm for time series:

a) Profile the standard implementation to identify computational bottlenecks.
b) Rewrite the conditional independence tests using matrix operations suitable for GPU computation.
c) Implement an efficient parallel search strategy for exploring the space of possible graphs.
d) Compare performance (both speed and accuracy) with the standard implementation on datasets of increasing size.
e) Can you achieve real-time causal discovery for streaming data? What approximations are necessary?

**Exercise 11.10** (Online Intervention Detection)
[Murphy] Design an online algorithm for detecting and quantifying interventions in streaming time series:

a) Implement a sliding-window version of the causal impact method.
b) Add an anomaly detection component to automatically identify potential intervention points.
c) Handle multiple simultaneous interventions with potentially overlapping effects.
d) Optimize for minimal memory usage and constant time updates.
e) Test on high-frequency financial data to detect market interventions or major trades.