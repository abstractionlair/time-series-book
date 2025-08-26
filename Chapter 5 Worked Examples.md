### **Worked Example 5.1: Representing an AR(1) Model in State Space Form**

**Context:** The main text introduced the concept of state space models and how various time series models, including ARMA models, can be represented in this framework. This worked example will demonstrate how to represent a simple AR(1) model as a state space model.

**Problem:**

Consider the AR(1) process given by:

\[
y_t = 0.7y_{t-1} + \epsilon_t
\]

where \(\epsilon_t \sim \mathcal{N}(0, 1)\).

**Solution:**

To represent this model in state space form, we need to identify the state transition and observation equations:

1. **State Equation:** The state vector \(x_t\) is simply the previous value of the time series, so:

\[
x_t = y_{t-1}
\]

The state equation can be written as:

\[
x_t = F_t x_{t-1} + w_t
\]

Where \(F_t = 0.7\) and \(w_t\) is the process noise. In this case, \(w_t = \epsilon_t\) and is normally distributed with variance \(\sigma^2_w = 1\).

2. **Observation Equation:** The observation at time \(t\) is directly given by the state:

\[
y_t = H_t x_t + v_t
\]

Here, \(H_t = 1\) and \(v_t\) (the observation noise) is zero because the observation is a direct measurement of the state.

Thus, the state space representation is:

\[
x_t = 0.7x_{t-1} + \epsilon_t
\]
\[
y_t = x_t
\]

Where \(\epsilon_t \sim \mathcal{N}(0, 1)\) and \(\sigma^2 = 1\).

This example illustrates the basic process of translating a simple time series model into state space form, a key step before applying techniques such as the Kalman filter.

---

### **Worked Example 5.2: Implementing the Kalman Filter for Tracking**

**Context:** The Kalman filter is introduced in the text as a recursive algorithm to estimate the state of a linear dynamic system. This example will demonstrate how to implement the Kalman filter for a simple tracking problem where we estimate the position and velocity of an object.

**Problem:**

Given the state space model:

\[
x_t = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} x_{t-1} + \begin{bmatrix} 0.5 \\ 1 \end{bmatrix} w_t
\]
\[
y_t = \begin{bmatrix} 1 & 0 \end{bmatrix} x_t + v_t
\]

where \(w_t \sim \mathcal{N}(0, 0.1)\) and \(v_t \sim \mathcal{N}(0, 1)\), implement the Kalman filter.

**Solution:**

The Kalman filter proceeds in two steps: prediction and update.

1. **Initialization:**
   - Initial state estimate: \(\hat{x}_0 = [0, 1]^T\)
   - Initial covariance estimate: \(P_0 = I\) (identity matrix)

2. **Prediction Step:**

\[
\hat{x}_{t|t-1} = F \hat{x}_{t-1|t-1}
\]
\[
P_{t|t-1} = F P_{t-1|t-1} F^T + Q
\]

Where \(F = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}\) and \(Q = \begin{bmatrix} 0.25 & 0.5 \\ 0.5 & 1 \end{bmatrix}\).

3. **Update Step:**

\[
\tilde{y}_t = y_t - H \hat{x}_{t|t-1}
\]
\[
S_t = H P_{t|t-1} H^T + R
\]
\[
K_t = P_{t|t-1} H^T S_t^{-1}
\]
\[
\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t \tilde{y}_t
\]
\[
P_{t|t} = (I - K_t H) P_{t|t-1}
\]

Where \(H = \begin{bmatrix} 1 & 0 \end{bmatrix}\) and \(R = 1\).

**Python Implementation:**

```python
import numpy as np

# Define system matrices
F = np.array([[1, 1], [0, 1]])
H = np.array([[1, 0]])
Q = np.array([[0.25, 0.5], [0.5, 1]])
R = np.array([[1]])

# Initial estimates
x_est = np.array([[0], [1]])  # Initial state estimate
P_est = np.eye(2)  # Initial covariance estimate

# Arrays to store results
positions = []
velocities = []

# Simulate data and apply Kalman filter
for t in range(50):
    # Prediction step
    x_pred = F @ x_est
    P_pred = F @ P_est @ F.T + Q
    
    # Update step
    y_t = H @ x_pred  # Simulated observation
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    x_est = x_pred + K @ (y_t - H @ x_pred)
    P_est = (np.eye(2) - K @ H) @ P_pred
    
    positions.append(x_est[0, 0])
    velocities.append(x_est[1, 0])

# Plot the results
import matplotlib.pyplot as plt
plt.plot(positions, label='Estimated Position')
plt.plot(velocities, label='Estimated Velocity')
plt.legend()
plt.show()
```

This example provides a detailed walkthrough of implementing the Kalman filter, preparing the reader to tackle Exercise 5.2.

---

### **Worked Example 5.3: Handling Nonlinear State Space Models with Particle Filters**

**Context:** In the text, the limitations of the Kalman filter for nonlinear models are discussed, leading to the introduction of particle filters. This example will demonstrate the use of particle filters for a nonlinear state space model.

**Problem:**

Given the nonlinear model:

\[
x_t = \sin(0.1x_{t-1}) + w_t
\]
\[
y_t = x_t^2 + v_t
\]

where \(w_t \sim \mathcal{N}(0, 0.1)\) and \(v_t \sim \mathcal{N}(0, 1)\), use a particle filter to estimate the state \(x_t\).

**Solution:**

1. **Initialize Particles:**

   Initialize \(N\) particles \(x_t^{(i)}\) from the prior distribution. Typically, this could be done by sampling from the known distribution of the initial state.

2. **Prediction Step:**

   For each particle, propagate the state according to the model:

   \[
   x_t^{(i)} = \sin(0.1x_{t-1}^{(i)}) + w_t^{(i)}
   \]

3. **Update Step:**

   Calculate the weight of each particle based on the likelihood of the observed data:

   \[
   w_t^{(i)} = p(y_t | x_t^{(i)})
   \]

   Normalize the weights:

   \[
   \tilde{w}_t^{(i)} = \frac{w_t^{(i)}}{\sum_{j=1}^{N} w_t^{(j)}}
   \]

4. **Resampling Step:**

   Resample the particles according to the weights to prevent degeneracy (where only a few particles have significant weight).

**Python Implementation:**

```python
import numpy as np

# Parameters
N = 1000  # Number of particles
Q = 0.1  # Process noise variance
R = 1.0  # Measurement noise variance

# Initialize particles and weights
particles = np.random.normal(0, 1, N)
weights = np.ones(N) / N

# Arrays to store estimates
state_estimates = []

# Simulate data and apply particle filter
for t in range(50):
    # Propagate particles
    particles = np.sin(0.1 * particles) + np.random.normal(0, np.sqrt(Q), N)
    
    # Calculate weights
    observation = particles**2 + np.random.normal(0, np.sqrt(R))
    weights *= np.exp(-0.5 * (observation - particles**2)**2 / R)
    weights /= np.sum(weights)
    
    # Resample particles
    indices = np.random.choice(N, N, p=weights)
    particles = particles[indices]
    weights = np.ones(N) / N
    
    # Estimate the state
    state_estimates.append(np.mean(particles))

# Plot results
plt.plot(state_estimates, label='Estimated State')
plt.legend()
plt.show()
```

This worked example walks through the implementation of a particle

 filter, providing a foundation for tackling Exercise 5.3.

---

### **Worked Example 5.4: Modeling with Bayesian Structural Time Series**

**Context:** The text discusses Bayesian Structural Time Series (BSTS) models as a flexible approach for modeling time series data with trends, seasonality, and regression effects. This example will walk through the setup and estimation of a BSTS model.

**Problem:**

Consider monthly sales data with a known trend and seasonal component, and a regression effect from advertising spending. The sales data and advertising spending are given as follows:

- Monthly sales: \([100, 110, 120, 130, 140, 145, 150, 160, 170, 180, 190, 200]\)
- Monthly advertising spend: \([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]\)

**Solution:**

1. **Model Specification:**

   The BSTS model can be defined as:

   \[
   y_t = \mu_t + \tau_t + \beta x_t + \epsilon_t
   \]

   where \(\mu_t\) is the local linear trend, \(\tau_t\) is the seasonal component, and \(\beta x_t\) is the regression effect of advertising.

2. **Priors:**

   Use weakly informative priors for the trend and seasonality components, and a normal prior for the regression coefficient \(\beta\).

3. **Estimation:**

   Use a Bayesian inference method, such as MCMC, to estimate the model parameters.

**Python Implementation:**

```python
import pymc3 as pm
import numpy as np

# Data
sales = np.array([100, 110, 120, 130, 140, 145, 150, 160, 170, 180, 190, 200])
advertising = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])

# Define the model
with pm.Model() as model:
    # Priors for trend components
    level = pm.GaussianRandomWalk('level', sigma=0.1, shape=len(sales))
    slope = pm.GaussianRandomWalk('slope', sigma=0.1, shape=len(sales))
    trend = level + slope
    
    # Prior for seasonal component
    seasonal_effect = pm.GaussianRandomWalk('seasonal', sigma=0.1, shape=12)
    seasonality = pm.Deterministic('seasonality', seasonal_effect[:len(sales)])
    
    # Prior for regression coefficient
    beta = pm.Normal('beta', mu=0, sigma=10)
    
    # Observation model
    mu = trend + seasonality + beta * advertising
    sigma = pm.HalfNormal('sigma', sigma=10)
    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=sales)
    
    # Inference
    trace = pm.sample(2000, tune=1000)

# Extract and plot results
pm.plot_posterior(trace, var_names=['beta'])
```

This example shows how to specify and estimate a BSTS model, laying the groundwork for Exercise 5.4.
