### Exercises for Chapter 5: State Space Models and Filtering

---

#### **Exercise 5.1: Basic State Space Model Representation**

Consider the following time series model:

\[
y_t = 0.7y_{t-1} + \epsilon_t
\]

where \(\epsilon_t\) is white noise with variance \(\sigma^2 = 1\).

1. Represent this AR(1) model in state space form. Specify the state transition matrix \(F_t\), the observation matrix \(H_t\), and the noise covariance matrices \(Q_t\) and \(R_t\).

2. Assume that the initial state is \(x_0 \sim \mathcal{N}(0, 1)\). Compute the first two predicted states and observations using the state space representation.

---

#### **Exercise 5.2: Kalman Filter Implementation**

You are given a simple state space model for tracking the position and velocity of a moving object:

\[
x_t = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} x_{t-1} + \begin{bmatrix} 0.5 \\ 1 \end{bmatrix} w_t
\]
\[
y_t = \begin{bmatrix} 1 & 0 \end{bmatrix} x_t + v_t
\]

where \(w_t\) and \(v_t\) are independent Gaussian noises with variances \(Q = 0.1\) and \(R = 1\), respectively.

1. Implement the Kalman filter for this model in Python or MATLAB. 
2. Simulate a time series of length 50 using the given model, starting from an initial state \(x_0 = [0, 1]^T\).
3. Apply the Kalman filter to your simulated data and plot the true states and the filtered estimates for both position and velocity.

---

#### **Exercise 5.3: Nonlinear State Space Models and Particle Filters**

Consider a nonlinear state space model:

\[
x_t = \sin(0.1x_{t-1}) + w_t
\]
\[
y_t = x_t^2 + v_t
\]

where \(w_t \sim \mathcal{N}(0, 0.1)\) and \(v_t \sim \mathcal{N}(0, 1)\).

1. Explain why the Kalman filter is not suitable for this model.
2. Implement a particle filter to estimate the states of this system. Use 1000 particles for your filter.
3. Generate synthetic data from the model and apply your particle filter. Compare the true states to the estimated states and discuss the performance of the particle filter.

---

#### **Exercise 5.4: Bayesian Structural Time Series Models**

A company wishes to model its sales data using a Bayesian Structural Time Series (BSTS) model, incorporating trend, seasonality, and the effect of advertising spending.

1. Write down a state space representation of the BSTS model that includes:
   - A local linear trend component.
   - A seasonal component (assume monthly data).
   - A regression component to capture the impact of advertising.

2. Using a Bayesian software package of your choice (e.g., PyMC3, Stan), fit this model to the following hypothetical dataset:
   - Monthly sales: \([100, 110, 120, 130, 140, 145, 150, 160, 170, 180, 190, 200]\)
   - Monthly advertising spend: \([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]\)

3. Provide credible intervals for the trend and the effect of advertising. Discuss how these intervals can be interpreted.

---

#### **Exercise 5.5: Model Comparison and Selection**

You are given two state space models for a dataset:

**Model 1:**
\[
x_t = F_t x_{t-1} + w_t
\]
\[
y_t = H_t x_t + v_t
\]

**Model 2:**
\[
x_t = G_t x_{t-1} + u_t
\]
\[
y_t = J_t x_t + z_t
\]

where \(F_t\), \(H_t\), \(G_t\), and \(J_t\) are known matrices and \(w_t\), \(v_t\), \(u_t\), and \(z_t\) are independent Gaussian noises.

1. Implement the Kalman filter for both models.
2. Simulate data using Model 1 and fit both models to this data.
3. Compare the log-likelihoods and the AIC values for the two models. Which model provides a better fit, and why?
