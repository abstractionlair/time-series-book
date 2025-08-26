### Worked Example 7.1: Exploring the Logistic Map and Chaos

**Objective:**  
To explore the dynamics of the logistic map and understand how changes in the parameter \( r \) affect the behavior of the system, leading to bifurcations and chaos.

**Problem:**  
Consider the logistic map:

\[ x_{t+1} = r x_t (1 - x_t) \]

Simulate the logistic map for different values of \( r \) and observe how the system transitions from stable fixed points to periodic behavior and eventually to chaos.

**Solution:**

1. **Simulation Setup:**
   - Choose initial conditions: \( x_0 = 0.5 \).
   - Select a range of \( r \) values: \( r = 2.5 \), \( r = 3.2 \), \( r = 3.5 \), and \( r = 3.9 \).
   - Number of iterations: 1000.

2. **Implementation:**
   - For each value of \( r \), generate the time series using the logistic map equation.
   - Plot the time series for each value of \( r \).

   ```python
   import numpy as np
   import matplotlib.pyplot as plt

   def logistic_map(x0, r, n):
       x = np.zeros(n)
       x[0] = x0
       for i in range(1, n):
           x[i] = r * x[i-1] * (1 - x[i-1])
       return x

   r_values = [2.5, 3.2, 3.5, 3.9]
   n = 1000
   x0 = 0.5

   fig, axs = plt.subplots(2, 2, figsize=(12, 10))
   for i, r in enumerate(r_values):
       x = logistic_map(x0, r, n)
       axs[i//2, i%2].plot(range(n), x)
       axs[i//2, i%2].set_title(f'r = {r}')
       axs[i//2, i%2].set_xlabel('Time')
       axs[i//2, i%2].set_ylabel('x')

   plt.tight_layout()
   plt.show()
   ```

3. **Observation:**
   - For \( r = 2.5 \), the system quickly settles into a fixed point.
   - As \( r \) increases to 3.2, the system exhibits periodic behavior with a period of 2.
   - For \( r = 3.5 \), the system enters a period-doubling cascade, indicating the onset of chaos.
   - At \( r = 3.9 \), the system becomes chaotic, with seemingly random fluctuations in \( x_t \).

**Discussion:**
- This example illustrates how simple nonlinear systems like the logistic map can exhibit a rich variety of behaviors, from stability to chaos, depending on the parameter \( r \). The transition from periodic behavior to chaos is a hallmark of nonlinear dynamics and is a central concept in the study of complex systems.

---

### Worked Example 7.2: Fitting a Nonlinear Autoregressive Model

**Objective:**  
To fit a nonlinear autoregressive (NAR) model to simulated data and compare the results to a linear autoregressive model.

**Problem:**  
Consider a nonlinear autoregressive model of order 2:

\[ X_t = 0.5 X_{t-1} - 0.4 X_{t-2} + 0.6 X_{t-1}^2 + \epsilon_t \]

where \( \epsilon_t \sim N(0, 0.1) \). Simulate data from this model and estimate the parameters using least squares.

**Solution:**

1. **Simulation Setup:**
   - Simulate 200 observations with initial conditions \( X_0 = 0 \) and \( X_1 = 0 \).

2. **Implementation:**
   - Generate the data using the specified NAR model.
   - Use a nonlinear regression technique to estimate the parameters.

   ```python
   import numpy as np
   from scipy.optimize import curve_fit

   def nar_model(X, a, b, c):
       X_t1, X_t2 = X
       return a * X_t1 + b * X_t2 + c * X_t1**2

   # Simulate data
   np.random.seed(0)
   n = 200
   X = np.zeros(n)
   for t in range(2, n):
       X[t] = 0.5 * X[t-1] - 0.4 * X[t-2] + 0.6 * X[t-1]**2 + np.random.normal(0, 0.1)

   # Prepare lagged data for regression
   X_t1 = X[1:-1]
   X_t2 = X[:-2]
   Y = X[2:]

   # Fit the model
   popt, pcov = curve_fit(nar_model, (X_t1, X_t2), Y)
   a_est, b_est, c_est = popt

   print(f"Estimated parameters: a = {a_est}, b = {b_est}, c = {c_est}")
   ```

3. **Results:**
   - The estimated parameters \( a \), \( b \), and \( c \) should be close to the true values (0.5, -0.4, and 0.6, respectively).

**Discussion:**
- The nonlinear autoregressive model captures the quadratic relationship between \( X_t \) and \( X_{t-1} \), which would be missed by a purely linear model. This example demonstrates the power of nonlinear models in capturing complex dynamics that cannot be represented by linear relationships.

---

### Worked Example 7.3: Analyzing a Threshold Autoregressive Model

**Objective:**  
To simulate and estimate a Threshold Autoregressive (TAR) model and understand its behavior.

**Problem:**  
Consider a TAR model defined as:

\[
X_t =
\begin{cases}
0.5 X_{t-1} + \epsilon_t, & \text{if } X_{t-1} \leq 0.2 \\
-0.5 X_{t-1} + \epsilon_t, & \text{if } X_{t-1} > 0.2
\end{cases}
\]

Simulate a time series of 300 observations and estimate the model parameters.

**Solution:**

1. **Simulation Setup:**
   - Define the TAR model with the given parameters.
   - Simulate 300 observations starting with \( X_0 = 0 \).

2. **Implementation:**
   - Generate the data and estimate the model parameters using maximum likelihood estimation.

   ```python
   import numpy as np

   def tar_model(n, r, phi1, phi2, sigma=0.1):
       X = np.zeros(n)
       for t in range(1, n):
           if X[t-1] <= r:
               X[t] = phi1 * X[t-1] + np.random.normal(0, sigma)
           else:
               X[t] = phi2 * X[t-1] + np.random.normal(0, sigma)
       return X

   # Simulate data
   np.random.seed(42)
   n = 300
   r = 0.2
   X = tar_model(n, r, 0.5, -0.5)

   # Plot the simulated data
   import matplotlib.pyplot as plt
   plt.plot(X)
   plt.title("Simulated TAR Model")
   plt.xlabel("Time")
   plt.ylabel("X_t")
   plt.show()

   # Estimate the parameters (simple heuristic approach for demonstration)
   X_lagged = X[:-1]
   X_current = X[1:]

   threshold = 0.2  # Assuming known
   regime1 = X_lagged <= threshold
   regime2 = X_lagged > threshold

   phi1_est = np.polyfit(X_lagged[regime1], X_current[regime1], 1)[0]
   phi2_est = np.polyfit(X_lagged[regime2], X_current[regime2], 1)[0]

   print(f"Estimated parameters: phi1 = {phi1_est}, phi2 = {phi2_est}")
   ```

3. **Results:**
   - The estimated parameters \( \phi1 \) and \( \phi2 \) should be close to 0.5 and -0.5, respectively.

**Discussion:**
- The TAR model allows for different behaviors in different regimes, which is useful in capturing phenomena like asymmetric responses to shocks. This example illustrates how regime-switching models can better capture the dynamics of time series with non-linearities.

---

### Worked Example 7.4: Understanding Lyapunov Exponents

**Objective:**  
To compute the Lyapunov exponent for the logistic map and understand its implication for chaos.

**Problem:**  
For the logistic map with \( r = 3.9 \), calculate the Lyapunov exponent and determine whether the system exhibits chaotic behavior.

**Solution:**

1. **Simulation Setup:**
   - Use the logistic map equation with \( r = 3.9 \) and initial condition \( x_0 = 0.5 \).
   - Simulate 1000 iterations.

2. **Lyapunov Exponent Calculation:**
   - Compute

 the separation between two nearby trajectories at each step and calculate the average rate of divergence.

   ```python
   import numpy as np

   def logistic_map(x, r):
       return r * x * (1 - x)

   def lyapunov_exponent(r, n=1000, x0=0.5, delta=1e-5):
       x = x0
       y = x0 + delta
       sum_log_diff = 0
       for _ in range(n):
           x = logistic_map(x, r)
           y = logistic_map(y, r)
           diff = np.abs(x - y)
           sum_log_diff += np.log(diff / delta)
           y = x + np.sign(y - x) * delta  # Rescale
       return sum_log_diff / n

   r = 3.9
   lyap_exp = lyapunov_exponent(r)
   print(f"Lyapunov exponent for r = {r}: {lyap_exp}")
   ```

3. **Results:**
   - The Lyapunov exponent should be positive, indicating that the system is chaotic.

**Discussion:**
- A positive Lyapunov exponent is a strong indicator of chaos, meaning that small differences in initial conditions will lead to exponentially diverging outcomes. This example shows how Lyapunov exponents can be used to diagnose chaos in nonlinear systems.
