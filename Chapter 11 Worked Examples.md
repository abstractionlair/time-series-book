Here are the worked examples for Chapter 11, designed to bridge the gap between the main text and the exercises. These examples aim to provide a detailed walkthrough of the key concepts and methods introduced in the chapter, preparing students to tackle the exercises effectively.

### Worked Example 1: Understanding Granger Causality
**Context:** Before diving into the exercises on causal inference, let's work through a complete example of testing for Granger causality between two time series.

1. **Theoretical Background:**
   - Granger causality tests whether past values of one time series (X) help predict future values of another (Y) beyond what Y's own past provides
   - It's about predictive causation, not true causation - a crucial distinction
   - The test is based on comparing restricted and unrestricted VAR models

2. **Example:**
   Let's analyze whether temperature Granger-causes ice cream sales.
   
   **Step 1:** Generate synthetic data with known causal structure
   ```python
   import numpy as np
   import pandas as pd
   from statsmodels.tsa.stattools import grangercausalitytests
   
   np.random.seed(42)
   n = 200
   
   # Temperature follows an AR(1) process
   temp = np.zeros(n)
   temp[0] = 20
   for t in range(1, n):
       temp[t] = 0.5 * temp[t-1] + 10 + np.random.normal(0, 2)
   
   # Ice cream sales depend on temperature with a lag
   sales = np.zeros(n)
   sales[0] = 100
   for t in range(1, n):
       sales[t] = 0.3 * sales[t-1] + 2 * temp[t-1] + 50 + np.random.normal(0, 5)
   
   data = pd.DataFrame({'temp': temp, 'sales': sales})
   ```
   
   **Step 2:** Test for Granger causality
   ```python
   # Test if temperature Granger-causes sales
   results_temp_to_sales = grangercausalitytests(data[['sales', 'temp']], maxlag=5)
   
   # Test if sales Granger-cause temperature (should be insignificant)
   results_sales_to_temp = grangercausalitytests(data[['temp', 'sales']], maxlag=5)
   ```
   
   **Step 3:** Interpret the results
   - Look at p-values for different lag orders
   - The first test should show significant causality (low p-values)
   - The second test should show no causality (high p-values)
   - This demonstrates the asymmetry of Granger causality

3. **Connection to Exercise 11.1-11.2:**
   This example provides the foundation for understanding predictive vs. true causation, preparing you to think critically about the radio static scenario in Exercise 11.1 and prove the theoretical properties in Exercise 11.2.

### Worked Example 2: Implementing Transfer Entropy
**Context:** Transfer entropy provides a model-free approach to measuring directed information flow between time series.

1. **Theoretical Background:**
   - Transfer entropy quantifies the reduction in uncertainty about Y's future given X's past
   - Unlike Granger causality, it captures nonlinear relationships
   - It's based on conditional mutual information

2. **Example:**
   Let's compute transfer entropy for a nonlinear system.
   
   **Step 1:** Create a nonlinear coupled system
   ```python
   def coupled_logistic_map(n=1000, r1=3.8, r2=3.7, coupling=0.4):
       x = np.zeros(n)
       y = np.zeros(n)
       x[0], y[0] = 0.4, 0.6
       
       for t in range(n-1):
           x[t+1] = r1 * x[t] * (1 - x[t])
           y[t+1] = r2 * y[t] * (1 - y[t]) * (1 - coupling) + coupling * x[t]
       
       return x[500:], y[500:]  # Discard transient
   
   x, y = coupled_logistic_map()
   ```
   
   **Step 2:** Estimate transfer entropy
   ```python
   from scipy.stats import entropy as scipy_entropy
   
   def transfer_entropy(x, y, k=1, bins=10):
       # Discretize the data
       x_binned = np.digitize(x, np.linspace(x.min(), x.max(), bins))
       y_binned = np.digitize(y, np.linspace(y.min(), y.max(), bins))
       
       # Create lagged versions
       y_future = y_binned[k:]
       y_past = y_binned[:-k]
       x_past = x_binned[:-k]
       
       # Estimate joint probabilities
       hist_yfuture_ypast_xpast, _ = np.histogramdd(
           [y_future, y_past, x_past], bins=[bins, bins, bins]
       )
       hist_yfuture_ypast, _ = np.histogram2d(y_future, y_past, bins=bins)
       hist_ypast_xpast, _ = np.histogram2d(y_past, x_past, bins=bins)
       hist_ypast = np.histogram(y_past, bins=bins)[0]
       
       # Normalize to probabilities
       p_yfuture_ypast_xpast = hist_yfuture_ypast_xpast / hist_yfuture_ypast_xpast.sum()
       p_yfuture_ypast = hist_yfuture_ypast / hist_yfuture_ypast.sum()
       p_ypast_xpast = hist_ypast_xpast / hist_ypast_xpast.sum()
       p_ypast = hist_ypast / hist_ypast.sum()
       
       # Compute transfer entropy (simplified calculation)
       te = 0
       for i in range(bins):
           for j in range(bins):
               for k in range(bins):
                   if p_yfuture_ypast_xpast[i,j,k] > 0:
                       te += p_yfuture_ypast_xpast[i,j,k] * np.log(
                           p_yfuture_ypast_xpast[i,j,k] * p_ypast[j] /
                           (p_yfuture_ypast[i,j] * p_ypast_xpast[j,k] + 1e-10)
                       )
       
       return te
   
   te_x_to_y = transfer_entropy(x, y)
   te_y_to_x = transfer_entropy(y, x)
   print(f"Transfer entropy X→Y: {te_x_to_y:.4f}")
   print(f"Transfer entropy Y→X: {te_y_to_x:.4f}")
   ```
   
   **Step 3:** Interpret the results
   - X→Y should show higher transfer entropy (X drives Y)
   - This captures the nonlinear coupling that linear methods might miss

3. **Connection to Exercises:**
   This example prepares you for implementing more sophisticated causal discovery methods in Exercises 11.3-11.5, especially for handling nonlinear relationships.

### Worked Example 3: Causal Impact Analysis
**Context:** Let's analyze the causal impact of an intervention using synthetic control methods.

1. **Theoretical Background:**
   - Causal impact analysis estimates what would have happened without an intervention
   - It constructs a synthetic control from similar units that didn't receive treatment
   - The difference between actual and synthetic is the causal effect

2. **Example:**
   Analyze the impact of a marketing campaign on sales.
   
   **Step 1:** Create synthetic data with intervention
   ```python
   import matplotlib.pyplot as plt
   
   # Generate pre-intervention data
   np.random.seed(123)
   n_pre = 50
   n_post = 30
   
   # Multiple stores (control units)
   n_stores = 10
   stores = np.zeros((n_stores, n_pre + n_post))
   
   for i in range(n_stores):
       trend = np.linspace(100, 120, n_pre + n_post)
       seasonal = 10 * np.sin(np.arange(n_pre + n_post) * 2 * np.pi / 7)
       noise = np.random.normal(0, 5, n_pre + n_post)
       stores[i] = trend + seasonal + noise
   
   # Add intervention effect to first store
   intervention_effect = 15
   stores[0, n_pre:] += intervention_effect
   
   # Split data
   treated = stores[0]
   controls = stores[1:]
   ```
   
   **Step 2:** Construct synthetic control
   ```python
   from sklearn.linear_model import LinearRegression
   
   # Use pre-intervention period to learn weights
   X_pre = controls[:, :n_pre].T
   y_pre = treated[:n_pre]
   
   # Fit model with non-negative constraints (simplified)
   model = LinearRegression(positive=True, fit_intercept=False)
   model.fit(X_pre, y_pre)
   weights = model.coef_
   weights = weights / weights.sum()  # Normalize
   
   # Construct synthetic control
   synthetic = np.sum(controls * weights[:, np.newaxis], axis=0)
   ```
   
   **Step 3:** Estimate causal impact
   ```python
   # Plot results
   plt.figure(figsize=(12, 6))
   plt.axvline(x=n_pre, color='red', linestyle='--', label='Intervention')
   plt.plot(treated, label='Treated Unit', linewidth=2)
   plt.plot(synthetic, label='Synthetic Control', linewidth=2)
   plt.fill_between(range(n_pre, n_pre + n_post), 
                    treated[n_pre:], synthetic[n_pre:], 
                    alpha=0.3, label='Causal Impact')
   plt.xlabel('Time')
   plt.ylabel('Sales')
   plt.legend()
   plt.title('Causal Impact Analysis')
   plt.show()
   
   # Calculate average treatment effect
   ate = np.mean(treated[n_pre:] - synthetic[n_pre:])
   print(f"Average Treatment Effect: {ate:.2f}")
   ```

3. **Connection to Exercise 11.6:**
   This example provides the foundation for implementing causal impact with model uncertainty, showing how to construct synthetic controls and estimate treatment effects.

### Worked Example 4: PC Algorithm for Causal Discovery
**Context:** The PC algorithm discovers causal structure from observational data using conditional independence tests.

1. **Theoretical Background:**
   - PC algorithm starts with a complete graph and removes edges based on conditional independence
   - It then orients edges using causal reasoning rules
   - Assumes causal sufficiency (no hidden confounders)

2. **Example:**
   Discover causal structure in a simple system.
   
   **Step 1:** Generate data from known DAG
   ```python
   # True DAG: A → B → C, A → C
   n = 1000
   A = np.random.normal(0, 1, n)
   B = 2 * A + np.random.normal(0, 1, n)
   C = 1.5 * B + 0.5 * A + np.random.normal(0, 1, n)
   
   data = np.column_stack([A, B, C])
   var_names = ['A', 'B', 'C']
   ```
   
   **Step 2:** Implement simplified PC algorithm
   ```python
   from scipy.stats import pearsonr
   from itertools import combinations
   
   def conditional_independence_test(X, Y, Z_set, data, alpha=0.05):
       """Test if X ⊥ Y | Z using partial correlation"""
       if len(Z_set) == 0:
           corr, p_value = pearsonr(data[:, X], data[:, Y])
       else:
           # Use linear regression to compute partial correlation
           from sklearn.linear_model import LinearRegression
           
           # Regress X on Z
           model_x = LinearRegression()
           model_x.fit(data[:, list(Z_set)], data[:, X])
           residual_x = data[:, X] - model_x.predict(data[:, list(Z_set)])
           
           # Regress Y on Z
           model_y = LinearRegression()
           model_y.fit(data[:, list(Z_set)], data[:, Y])
           residual_y = data[:, Y] - model_y.predict(data[:, list(Z_set)])
           
           corr, p_value = pearsonr(residual_x, residual_y)
       
       return p_value > alpha  # True if independent
   
   def pc_algorithm_skeleton(data, alpha=0.05):
       n_vars = data.shape[1]
       
       # Start with complete graph
       graph = np.ones((n_vars, n_vars)) - np.eye(n_vars)
       
       # Remove edges based on conditional independence
       for level in range(n_vars):
           for i, j in combinations(range(n_vars), 2):
               if graph[i, j] == 0:
                   continue
               
               # Find potential conditioning sets
               neighbors = [k for k in range(n_vars) if k != i and k != j and graph[i, k] == 1]
               
               for Z_set in combinations(neighbors, min(level, len(neighbors))):
                   if conditional_independence_test(i, j, Z_set, data, alpha):
                       graph[i, j] = 0
                       graph[j, i] = 0
                       break
       
       return graph
   
   skeleton = pc_algorithm_skeleton(data)
   print("Discovered skeleton:")
   for i in range(3):
       for j in range(3):
           if skeleton[i, j] == 1:
               print(f"{var_names[i]} -- {var_names[j]}")
   ```
   
   **Step 3:** Orient edges (simplified)
   ```python
   # Apply orientation rules (simplified version)
   # Rule: If i--j--k and i not connected to k, orient as i→j←k
   oriented = skeleton.copy()
   for j in range(3):
       for i, k in combinations(range(3), 2):
           if (skeleton[i, j] == 1 and skeleton[j, k] == 1 and skeleton[i, k] == 0):
               oriented[j, i] = 0  # Remove j→i
               oriented[j, k] = 0  # Remove j→k
               print(f"Oriented: {var_names[i]} → {var_names[j]} ← {var_names[k]}")
   ```

3. **Connection to Exercise 11.7-11.9:**
   This example provides the foundation for implementing more sophisticated causal discovery algorithms, including handling feedback loops and latent variables.

These worked examples provide hands-on experience with the key causal inference methods covered in Chapter 11, preparing you to tackle the more challenging exercises that follow.