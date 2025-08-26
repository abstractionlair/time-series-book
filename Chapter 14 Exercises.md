# Chapter 14 Exercises

## Exercises

### Algorithmic Efficiency

**Exercise 14.1** (Feynman Style)
Imagine you're trying to find patterns in a heartbeat recording that's 24 hours long, sampled at 1000 Hz. That's about 86 million data points!

a) If you wanted to compute the correlation between every pair of points (the full correlation matrix), how many operations would that require? Why is this impractical?
b) Now suppose you only care about correlations up to 1 second lag. How does this change the problem?
c) Design a clever sampling strategy that could give you a good approximation of the long-range correlations without computing everything.
d) What fundamental trade-off are you making between accuracy and computation time?

**Exercise 14.2** (Murphy Style - FFT Optimization)
Implement and benchmark different FFT approaches:

```python
def compare_fft_methods(signal_length):
    """
    Compare different FFT implementations.
    
    Returns:
    --------
    dict with timing results for:
    - Naive DFT O(N^2)
    - Recursive FFT O(N log N)
    - NumPy FFT (optimized C implementation)
    - GPU FFT (if available)
    """
    pass

def fft_memory_analysis(max_length):
    """
    Analyze memory usage patterns of FFT.
    Show when out-of-core algorithms become necessary.
    """
    pass
```

Requirements:
- Implement naive DFT and Cooley-Tukey FFT from scratch
- Profile both time and memory usage
- Identify the crossover point where FFT becomes faster than DFT
- If GPU is available, implement cuFFT version
- Analyze cache performance using appropriate tools

### Parallel Computing

**Exercise 14.3** (Mixed Voices - The Parallel Paradox)
[Feynman] Here's a puzzle: You have a time series forecasting task that takes 1 hour on a single CPU. You have access to 100 CPUs. How long will it take?

[Gelman] The naive answer is 36 seconds, but that's almost never right in practice.

[Murphy] Let's build a realistic model of parallel execution.

Your task:
a) Implement Amdahl's Law for time series computation: If fraction p of the algorithm is parallelizable, what's the maximum speedup with n processors?
b) Add communication overhead: Each processor needs time t_comm to send/receive data.
c) Model load imbalance: Some chunks of work take longer than others (use a log-normal distribution).
d) Implement a simulation showing actual vs. theoretical speedup for:
   - Moving average calculation
   - ARIMA model fitting
   - Cross-validation
   - Ensemble forecasting
e) Find the optimal number of processors for each task. Why isn't it always "use all available CPUs"?

**Exercise 14.4** (Distributed Time Series)
Build a distributed time series processing system:

```python
class DistributedTSProcessor:
    def __init__(self, n_workers):
        """Initialize distributed processor with n_workers."""
        pass
    
    def partition_strategy(self, ts_data, method='temporal'):
        """
        Partition time series for distribution.
        Methods: 'temporal', 'frequency', 'feature', 'hybrid'
        """
        pass
    
    def map_reduce_forecast(self, data, model_func):
        """
        Map: fit local models
        Reduce: combine forecasts
        """
        pass
    
    def handle_failure(self, worker_id):
        """Implement fault tolerance."""
        pass
```

Test your system on:
- 1GB of high-frequency trading data
- Multi-location weather data (spatial-temporal)
- Hierarchical retail sales data

### Streaming and Online Learning

**Exercise 14.5** (Jaynes Style - Information Flow)
Consider a streaming time series where each observation costs C to process and provides information I.

a) Derive the optimal sampling rate that maximizes information per unit cost.
b) Show that for a stationary AR(1) process, the information gain from observation n+1 given n observations is I_{n+1} = -0.5 * log(1 - ρ²) where ρ is the AR coefficient.
c) Implement an adaptive sampling algorithm that increases sampling rate when the process becomes less predictable.
d) Prove that your algorithm converges to the optimal sampling rate under stationarity.

**Exercise 14.6** (Real-time Anomaly Detection)
Build a complete streaming anomaly detection system:

```python
class StreamingAnomalyDetector:
    def __init__(self, window_size, update_frequency):
        self.reservoir = []  # Reservoir sampling for history
        self.model = None
        self.threshold = None
        
    def update(self, new_point, timestamp):
        """
        Process new point in O(1) time.
        Return anomaly score.
        """
        pass
    
    def adapt_threshold(self):
        """
        Dynamically adjust threshold based on recent false positive rate.
        """
        pass
    
    def concept_drift_detection(self):
        """
        Detect when the underlying distribution has changed.
        """
        pass
```

Requirements:
- Maintain constant memory usage regardless of stream length
- Sub-millisecond latency per point
- Handle seasonal patterns without storing full seasons
- Adapt to concept drift without full retraining

### Software Engineering

**Exercise 14.7** (Gelman Style - Testing Time Series Code)
Real time series code has subtle bugs. Let's catch them:

a) Create a test suite for a forecasting function that checks:
   - Forecast values are in reasonable range
   - Prediction intervals widen with horizon
   - Backtesting gives consistent results
   - No data leakage from future to past

b) Implement property-based testing for time series:
```python
def property_test_stationarity(transformation_func):
    """
    Test that transformation preserves/creates stationarity.
    Generate random TS, apply transformation, verify properties.
    """
    pass

def property_test_forecast_coherence(forecast_func):
    """
    Test that forecasts are coherent:
    - Hierarchical forecasts sum correctly
    - Confidence intervals contain point forecast
    - Multi-step ahead = iterated one-step
    """
    pass
```

c) Create a benchmark suite that tracks:
   - Execution time vs. data size
   - Memory usage patterns
   - Numerical stability for edge cases

### Best Practices Integration

**Exercise 14.8** (Complete Pipeline Optimization)
Build an auto-optimizing time series pipeline:

```python
class AutoOptimizer:
    def __init__(self):
        self.profiler = {}
        self.optimization_history = []
        
    def profile_stage(self, stage_name, func, *args):
        """Profile a pipeline stage."""
        pass
    
    def identify_bottleneck(self):
        """Find the slowest stage in pipeline."""
        pass
    
    def suggest_optimization(self, bottleneck):
        """
        Suggest optimization based on bottleneck type:
        - CPU bound → parallelization
        - Memory bound → chunking/streaming
        - I/O bound → async/caching
        """
        pass
    
    def auto_optimize(self, pipeline):
        """
        Automatically apply optimizations and measure improvement.
        """
        pass
```

Test on a complete forecasting pipeline:
1. Data loading (CSV, parquet, database)
2. Preprocessing (missing values, outliers)
3. Feature engineering
4. Model fitting (multiple models)
5. Cross-validation
6. Forecast generation
7. Visualization

### Theoretical Challenges

**Exercise 14.9** (Information-Theoretic Limits)
[Jaynes] Every algorithm has fundamental limits. Let's find them.

a) For a Gaussian time series with known autocovariance, derive the minimum number of operations required to compute an optimal linear forecast.
b) Show that for a general nonlinear time series, the Kolmogorov complexity provides a lower bound on the computational resources needed for prediction.
c) Implement the "speed prior" - prefer faster algorithms when multiple algorithms give similar predictive performance.
d) Create a meta-algorithm that trades off between:
   - Computational cost
   - Predictive accuracy
   - Model interpretability

### Scaling Challenge

**Exercise 14.10** (The Billion Point Challenge)
You have 1 billion points of minutely data (about 2000 years of minutes). Build a system that can:

a) Load and visualize any time window in <1 second
b) Compute rolling statistics with any window size in <10 seconds
c) Fit an ARIMA model to any subset in <1 minute
d) Generate forecasts at multiple horizons in <5 seconds

Constraints:
- Maximum 16GB RAM
- Data stored in compressed format
- Must handle irregular timestamps and missing values

Your solution should include:
- Hierarchical indexing strategy
- Approximate algorithms where exact computation is too slow
- Caching strategy for frequently accessed windows
- Progressive refinement (show approximate results quickly, refine as computation continues)

Document your design decisions and trade-offs.

### Research Project

**Exercise 14.11** (Pushing the Boundaries)
Choose one of the following open problems and make progress on it:

a) **Optimal Parallelization**: Develop an algorithm that automatically determines the optimal parallelization strategy for any time series algorithm based on data characteristics and hardware constraints.

b) **Streaming Spectral Analysis**: Design an algorithm that maintains an accurate spectral density estimate for a streaming time series using O(log N) space.

c) **Distributed Bayesian Inference**: Implement a distributed MCMC algorithm for time series models that provably converges to the correct posterior distribution.

d) **Quantum Time Series**: Explore how quantum computing could accelerate time series analysis. Implement a simulator and show theoretical speedup for specific operations.

Write a report including:
- Literature review of existing approaches
- Your proposed solution
- Theoretical analysis (complexity, convergence)
- Empirical evaluation
- Limitations and future work