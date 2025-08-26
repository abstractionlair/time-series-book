### Worked Example 8.1: Implementing Approximate Bayesian Computation (ABC) for an AR(1) Model

#### Problem:
Suppose you are given an AR(1) time series model:
\[
x_t = \phi x_{t-1} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)
\]
You are tasked with estimating the parameter \(\phi\) using Approximate Bayesian Computation (ABC).

#### Step 1: Simulating Data
First, simulate a time series of length 100 with \(\phi = 0.6\) and \(\sigma = 1.0\).

```python
import numpy as np

np.random.seed(42)
n = 100
phi = 0.6
sigma = 1.0
x = np.zeros(n)
epsilon = np.random.normal(0, sigma, n)

for t in range(1, n):
    x[t] = phi * x[t-1] + epsilon[t]

print(x[:10])  # Display the first 10 values of the simulated time series
```

#### Step 2: Choosing Summary Statistics
In ABC, you need to reduce the data into summary statistics. For an AR(1) model, appropriate choices might include:
- Sample mean: \(\bar{x}\)
- Sample variance: \(s^2\)

These statistics summarize the central tendency and dispersion of the time series, which are influenced by \(\phi\) and \(\sigma\).

#### Step 3: Implementing ABC Rejection Algorithm
Next, implement the ABC rejection algorithm:

1. **Prior**: Assume a uniform prior for \(\phi\) over \([-1, 1]\).
2. **Simulation**: For each candidate \(\phi'\), simulate a time series, calculate summary statistics, and compare them to the observed statistics.
3. **Rejection**: Retain the candidate \(\phi'\) if the simulated statistics are within a small tolerance of the observed ones.

```python
phi_prior = np.random.uniform(-1, 1, 10000)  # 10,000 candidates

def simulate_ar1(phi, n=100):
    x_sim = np.zeros(n)
    epsilon = np.random.normal(0, sigma, n)
    for t in range(1, n):
        x_sim[t] = phi * x_sim[t-1] + epsilon[t]
    return x_sim

observed_mean = np.mean(x)
observed_var = np.var(x)

tolerance = 0.1
accepted_phi = []

for phi_cand in phi_prior:
    x_sim = simulate_ar1(phi_cand)
    sim_mean = np.mean(x_sim)
    sim_var = np.var(x_sim)
    if abs(sim_mean - observed_mean) < tolerance and abs(sim_var - observed_var) < tolerance:
        accepted_phi.append(phi_cand)

print("Posterior mean of accepted phi:", np.mean(accepted_phi))
```

#### Step 4: Analyzing Results
The accepted \(\phi\) values form an approximate posterior distribution. Plot this distribution and discuss how the tolerance and choice of summary statistics impact the results.

```python
import matplotlib.pyplot as plt

plt.hist(accepted_phi, bins=30, density=True)
plt.title("Posterior Distribution of φ")
plt.xlabel("φ")
plt.ylabel("Density")
plt.show()
```

#### Conclusion:
The resulting posterior distribution provides insight into the parameter \(\phi\). As the tolerance decreases, the approximation improves, but fewer candidates are accepted, highlighting a trade-off between accuracy and computational efficiency.

---

### Worked Example 8.2: Sequential Monte Carlo (SMC) ABC for an AR(1) Model

#### Problem:
Extend the ABC method to use Sequential Monte Carlo (SMC) for the AR(1) model from Worked Example 8.1 to improve efficiency.

#### Step 1: Overview of SMC-ABC
Sequential Monte Carlo allows for gradually refining the tolerance over several iterations, using the posterior from one iteration as the prior for the next.

#### Step 2: Implementing SMC-ABC

1. **Initial Sampling**: Begin with a broad prior and large tolerance, similar to the ABC rejection method.
2. **Iterative Refinement**: Gradually reduce the tolerance, re-weighting particles (accepted \(\phi\) values) to favor those closer to the observed summary statistics.

```python
def smc_abc(n_particles, n_iter, initial_tolerance, observed_mean, observed_var):
    phi_particles = np.random.uniform(-1, 1, n_particles)
    weights = np.ones(n_particles) / n_particles
    
    for iteration in range(n_iter):
        tolerance = initial_tolerance / (iteration + 1)
        new_particles = []
        new_weights = []
        
        for i in range(n_particles):
            phi_cand = np.random.choice(phi_particles, p=weights)
            x_sim = simulate_ar1(phi_cand)
            sim_mean = np.mean(x_sim)
            sim_var = np.var(x_sim)
            
            if abs(sim_mean - observed_mean) < tolerance and abs(sim_var - observed_var) < tolerance:
                new_particles.append(phi_cand)
                new_weights.append(1.0)
        
        new_weights = np.array(new_weights)
        new_weights /= new_weights.sum()  # Normalize weights
        phi_particles = np.array(new_particles)
        weights = new_weights
    
    return phi_particles, weights

phi_smc, weights_smc = smc_abc(1000, 10, 0.5, observed_mean, observed_var)
```

#### Step 3: Posterior Analysis
After running the SMC-ABC, analyze the posterior distribution:

```python
plt.hist(phi_smc, bins=30, weights=weights_smc, density=True)
plt.title("SMC-ABC Posterior Distribution of φ")
plt.xlabel("φ")
plt.ylabel("Density")
plt.show()
```

#### Conclusion:
SMC-ABC provides a more efficient approach to approximate the posterior distribution, especially when the parameter space is complex or high-dimensional. The re-weighting mechanism allows for better exploration of the parameter space, leading to more accurate estimates with fewer simulations.

---

### Worked Example 8.3: Hamiltonian Monte Carlo (HMC) for a Non-linear Time Series Model

#### Problem:
Use Hamiltonian Monte Carlo (HMC) to estimate parameters in the non-linear time series model:
\[
x_t = 0.5x_{t-1} + \frac{25x_{t-1}}{1 + x_{t-1}^2} + 8\cos(1.2t) + w_t
\]
\[
y_t = \frac{x_t^2}{20} + v_t
\]

#### Step 1: Simulating Data
Simulate data from the given model.

```python
import numpy as np

np.random.seed(42)
n = 100
x = np.zeros(n)
y = np.zeros(n)
w = np.random.normal(0, 1, n)
v = np.random.normal(0, 1, n)

for t in range(1, n):
    x[t] = 0.5 * x[t-1] + (25 * x[t-1]) / (1 + x[t-1]**2) + 8 * np.cos(1.2 * t) + w[t]
    y[t] = x[t]**2 / 20 + v[t]

print(y[:10])  # Display the first 10 values of y
```

#### Step 2: Implementing HMC
Use a standard HMC implementation, leveraging a Python library like PyMC3 or Stan, to estimate the parameters of the model.

```python
import pymc3 as pm

with pm.Model() as model:
    phi = pm.Normal('phi', mu=0, sigma=1)
    x_sim = pm.Deterministic('x_sim', phi * x[:-1] + (25 * x[:-1]) / (1 + x[:-1]**2) + 8 * np.cos(1.2 * np.arange(1, n)))
    y_obs = pm.Normal('y_obs', mu=x_sim**2 / 20, sigma=1, observed=y)
    
    trace = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=False)

pm.traceplot(trace)
```

#### Step 3: Diagnostics and Posterior Analysis
Analyze the trace plots and other diagnostics to ensure the HMC sampler has converged.

```python
pm.summary(trace)
```

#### Conclusion:
HMC provides an efficient way to sample from the posterior distribution of complex models, leveraging gradient information to navigate the parameter space. The effective sample size and other diagnostics confirm the reliability of the estimates obtained.

---

### Worked Example 8.4: Bayesian Model Comparison Using ABC

#### Problem:
Given two competing models for a time series (AR(1) and ARMA(1,1)), use ABC to compare their posterior probabilities and select the most appropriate model.

#### Step 1: Simulate Data
Simulate data from both an AR(1) and an ARMA(1,1) model.

```python
from statsmodels.tsa.arima_process import ArmaProcess

np.random.seed(42)
ar1 = np.array([1, -0.6])
ma1 = np.array([1])
arma_process = ArmaProcess(ar=ar1, ma

=ma1)
y_arma = arma_process.generate_sample(nsample=100)

arma_process = ArmaProcess(ar=ar1, ma=[1, 0.3])
y_arma = arma_process.generate_sample(nsample=100)

print(y_arma[:10])  # Display the first 10 values of the ARMA(1,1) series
```

#### Step 2: ABC Model Comparison
Implement ABC for each model, calculating the posterior probability of each.

```python
# Implement ABC as shown in Worked Example 8.1 and 8.2 for both models

# Example comparison: if model 1 posterior is higher, select AR(1); otherwise, select ARMA(1,1)
posterior_ar1 = ...
posterior_arma11 = ...

if posterior_ar1 > posterior_arma11:
    print("Select AR(1) Model")
else:
    print("Select ARMA(1,1) Model")
```

#### Conclusion:
ABC allows for model comparison by estimating the posterior probabilities of competing models. By comparing these probabilities, you can select the model that most likely generated the observed data, taking into account the uncertainty in the model parameters.
