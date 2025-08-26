### Exercise 7.1: Logistic Map and Bifurcation
Consider the logistic map, a classic example of a simple nonlinear dynamical system:

\[ x_{t+1} = r x_t (1 - x_t) \]

where \( r \) is a parameter and \( x_t \) represents the state of the system at time \( t \).

1. **Bifurcation Analysis**:
   - Simulate the logistic map for 1000 iterations with initial condition \( x_0 = 0.5 \) and different values of \( r \) ranging from 2.5 to 4.0.
   - Plot the bifurcation diagram for the logistic map, showing how the long-term behavior of \( x_t \) changes as \( r \) varies.
   - Identify and discuss the different dynamical regimes (fixed points, periodic behavior, and chaos) observed in the bifurcation diagram.

2. **Sensitivity to Initial Conditions**:
   - Simulate the logistic map for \( r = 3.9 \) with two slightly different initial conditions \( x_0 = 0.5 \) and \( x_0 = 0.5001 \).
   - Plot the trajectories of \( x_t \) for these initial conditions over 100 iterations and discuss the sensitivity of the system to initial conditions.

---

### Exercise 7.2: Nonlinear Autoregressive Models
Consider a nonlinear autoregressive model (NAR) of order 2 given by:

\[ X_t = 0.5 X_{t-1} - 0.4 X_{t-2} + 0.6 X_{t-1}^2 + \epsilon_t \]

where \( \epsilon_t \) is white noise with zero mean and variance \( \sigma^2 = 0.1 \).

1. **Simulation**:
   - Simulate 200 observations from this model with \( \epsilon_t \sim N(0, 0.1) \) and initial conditions \( X_0 = 0 \) and \( X_1 = 0 \).
   - Plot the time series and discuss the observed behavior.

2. **Model Estimation**:
   - Estimate the parameters of the nonlinear autoregressive model using least squares or maximum likelihood estimation.
   - Compare the estimated parameters to the true parameters and discuss any discrepancies.

3. **Prediction**:
   - Use the estimated model to predict the next 10 observations based on the simulated data.
   - Plot the predicted values along with the actual simulated values and compute the prediction error.

---

### Exercise 7.3: Threshold Autoregressive (TAR) Model
A Threshold Autoregressive (TAR) model is defined as:

\[
X_t =
\begin{cases}
\phi_1^{(1)} X_{t-1} + \epsilon_t, & \text{if } X_{t-d} \leq r \\
\phi_1^{(2)} X_{t-1} + \epsilon_t, & \text{if } X_{t-d} > r
\end{cases}
\]

where \( r \) is the threshold and \( d \) is the delay parameter.

1. **Simulation**:
   - Simulate a time series of 300 observations from a TAR model with \( \phi_1^{(1)} = 0.5 \), \( \phi_1^{(2)} = -0.5 \), \( r = 0.2 \), and \( d = 1 \).
   - Plot the time series and identify the switching points between regimes.

2. **Parameter Estimation**:
   - Estimate the parameters \( \phi_1^{(1)} \), \( \phi_1^{(2)} \), and \( r \) from the simulated data.
   - Use the estimated model to classify the data into the two regimes and compare the classification to the true regime states.

3. **Model Comparison**:
   - Fit a linear autoregressive model of the same order to the data and compare its performance to the TAR model in terms of in-sample fit and out-of-sample prediction.

---

### Exercise 7.4: Chaos and the Lyapunov Exponent
The Lyapunov exponent is a measure of the rate at which nearby trajectories in a dynamical system diverge. A positive Lyapunov exponent is indicative of chaos.

1. **Lyapunov Exponent Calculation**:
   - For the logistic map with \( r = 3.9 \), calculate the Lyapunov exponent by iterating the map and computing the average logarithmic rate of separation of nearby trajectories.
   - Interpret the resulting Lyapunov exponent and discuss whether the system exhibits chaotic behavior.

2. **Impact of Noise**:
   - Add small Gaussian noise to the logistic map and re-calculate the Lyapunov exponent.
   - Discuss how noise affects the predictability and stability of the system.

---

### Exercise 7.5: Bayesian Estimation in Nonlinear Models
Consider the following nonlinear autoregressive model:

\[ X_t = \frac{0.7 X_{t-1}}{1 + X_{t-1}^2} + \epsilon_t \]

where \( \epsilon_t \sim N(0, \sigma^2) \).

1. **Bayesian Estimation**:
   - Use a Bayesian approach to estimate the parameters \( \sigma \) and the coefficient \( 0.7 \) from simulated data.
   - Implement the estimation using MCMC techniques, and obtain posterior distributions for the parameters.

2. **Uncertainty Quantification**:
   - Compute and plot the posterior predictive distribution of \( X_t \) for the next 10 time steps.
   - Discuss how the Bayesian framework allows for uncertainty quantification in the predictions.
