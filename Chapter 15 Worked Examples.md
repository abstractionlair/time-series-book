Here are the worked examples for Chapter 15, designed to bridge the gap between the main text and the exercises. These examples aim to provide a detailed walkthrough of the key concepts and methods introduced in the chapter, preparing students to tackle the exercises effectively.

### Worked Example 1: Building a Simple VAE for Time Series
**Context:** Before diving into the advanced deep probabilistic models in the exercises, let's build intuition with a simple Variational Autoencoder for time series data.

1. **Theoretical Background:**
   - VAEs learn a probabilistic mapping between data and a latent space
   - The encoder maps data to a distribution over latent variables
   - The decoder reconstructs data from samples in the latent space
   - The loss balances reconstruction accuracy with latent space regularity

2. **Example:**
   Let's build a VAE for sine waves with varying frequency and amplitude.
   
   **Step 1:** Generate synthetic time series data
   ```python
   import numpy as np
   import tensorflow as tf
   from tensorflow import keras
   import matplotlib.pyplot as plt
   
   def generate_sine_waves(n_samples=1000, seq_length=50):
       data = []
       for _ in range(n_samples):
           freq = np.random.uniform(0.1, 0.5)
           amplitude = np.random.uniform(0.5, 2.0)
           phase = np.random.uniform(0, 2*np.pi)
           t = np.linspace(0, 10, seq_length)
           wave = amplitude * np.sin(2*np.pi*freq*t + phase)
           data.append(wave)
       return np.array(data).reshape(n_samples, seq_length, 1)
   
   data = generate_sine_waves()
   ```
   
   **Step 2:** Build the VAE architecture
   ```python
   from tensorflow.keras import layers, Model
   
   seq_length = 50
   latent_dim = 2
   
   # Encoder
   encoder_inputs = keras.Input(shape=(seq_length, 1))
   x = layers.LSTM(32, return_sequences=True)(encoder_inputs)
   x = layers.LSTM(16)(x)
   z_mean = layers.Dense(latent_dim)(x)
   z_log_var = layers.Dense(latent_dim)(x)
   
   # Sampling layer
   def sampling(args):
       z_mean, z_log_var = args
       batch = tf.shape(z_mean)[0]
       dim = tf.shape(z_mean)[1]
       epsilon = tf.random.normal(shape=(batch, dim))
       return z_mean + tf.exp(0.5 * z_log_var) * epsilon
   
   z = layers.Lambda(sampling)([z_mean, z_log_var])
   encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
   
   # Decoder
   latent_inputs = keras.Input(shape=(latent_dim,))
   x = layers.RepeatVector(seq_length)(latent_inputs)
   x = layers.LSTM(16, return_sequences=True)(x)
   x = layers.LSTM(32, return_sequences=True)(x)
   decoder_outputs = layers.TimeDistributed(layers.Dense(1))(x)
   decoder = Model(latent_inputs, decoder_outputs, name='decoder')
   
   # VAE Model
   outputs = decoder(encoder(encoder_inputs)[2])
   vae = Model(encoder_inputs, outputs, name='vae')
   ```
   
   **Step 3:** Define the VAE loss and train
   ```python
   # VAE loss
   reconstruction_loss = tf.reduce_mean(
       tf.reduce_sum(keras.losses.mse(encoder_inputs, outputs), axis=1)
   )
   kl_loss = -0.5 * tf.reduce_mean(
       tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
   )
   vae_loss = reconstruction_loss + kl_loss
   vae.add_loss(vae_loss)
   vae.compile(optimizer='adam')
   
   # Train
   history = vae.fit(data, epochs=50, batch_size=32, verbose=0)
   ```
   
   **Step 4:** Explore the latent space
   ```python
   # Encode data to latent space
   z_mean, _, _ = encoder.predict(data[:100])
   
   # Visualize latent space
   plt.figure(figsize=(8, 6))
   plt.scatter(z_mean[:, 0], z_mean[:, 1], alpha=0.5)
   plt.xlabel('Latent dimension 1')
   plt.ylabel('Latent dimension 2')
   plt.title('VAE Latent Space Representation')
   plt.show()
   
   # Generate new time series
   random_latent = np.random.normal(size=(5, latent_dim))
   generated = decoder.predict(random_latent)
   
   plt.figure(figsize=(12, 6))
   for i in range(5):
       plt.subplot(2, 3, i+1)
       plt.plot(generated[i, :, 0])
       plt.title(f'Generated series {i+1}')
   plt.tight_layout()
   plt.show()
   ```

3. **Connection to Exercise 15.1:**
   This example provides hands-on experience with VAEs, preparing you to explore how they learn dynamics vs. compression and to implement more sophisticated architectures.

### Worked Example 2: Causal Discovery with Convergent Cross Mapping
**Context:** Let's implement CCM to discover causal relationships in nonlinear dynamical systems.

1. **Theoretical Background:**
   - CCM exploits Takens' theorem: if X causes Y, the attractor of Y contains information about X
   - We test if we can predict X from the attractor reconstruction of Y
   - Convergence with library size indicates causation

2. **Example:**
   Analyze causality in coupled Rössler systems.
   
   **Step 1:** Generate coupled Rössler systems
   ```python
   from scipy.integrate import odeint
   
   def coupled_rossler(state, t, a=0.2, b=0.2, c=5.7, coupling=0.3):
       x1, y1, z1, x2, y2, z2 = state
       
       # First Rössler system
       dx1 = -y1 - z1
       dy1 = x1 + a*y1
       dz1 = b + z1*(x1 - c)
       
       # Second Rössler system (coupled to first)
       dx2 = -y2 - z2 + coupling*(x1 - x2)
       dy2 = x2 + a*y2
       dz2 = b + z2*(x2 - c)
       
       return [dx1, dy1, dz1, dx2, dy2, dz2]
   
   # Simulate
   t = np.linspace(0, 500, 5000)
   initial = [1, 1, 1, 2, 2, 2]
   trajectory = odeint(coupled_rossler, initial, t)
   x1, x2 = trajectory[:, 0], trajectory[:, 3]
   ```
   
   **Step 2:** Implement CCM
   ```python
   from sklearn.neighbors import NearestNeighbors
   
   def embed_series(x, E, tau):
       """Create shadow manifold using time-delay embedding"""
       n = len(x) - (E-1)*tau
       embedded = np.zeros((n, E))
       for i in range(E):
           embedded[:, i] = x[i*tau:i*tau+n]
       return embedded
   
   def ccm(x, y, E=3, tau=1, lib_sizes=None):
       """Convergent Cross Mapping"""
       if lib_sizes is None:
           lib_sizes = np.arange(10, len(x)//2, 50)
       
       # Embed both series
       Mx = embed_series(x, E, tau)
       My = embed_series(y, E, tau)
       n = len(Mx)
       
       correlations = []
       for L in lib_sizes:
           if L > n:
               break
           
           # Use manifold of Y to predict X
           library_idx = np.random.choice(n, L, replace=False)
           
           # Find nearest neighbors in Y manifold
           knn = NearestNeighbors(n_neighbors=E+1)
           knn.fit(My[library_idx])
           
           # Predict X from Y's manifold
           predictions = []
           for i in range(L):
               distances, indices = knn.kneighbors([My[library_idx[i]]])
               # Exclude self
               dist = distances[0, 1:]
               idx = library_idx[indices[0, 1:]]
               
               # Weighted average
               weights = np.exp(-dist / (dist[0] + 1e-8))
               weights /= weights.sum()
               pred = np.sum(weights * x[idx])
               predictions.append(pred)
           
           # Correlation between actual and predicted
           corr = np.corrcoef(x[library_idx], predictions)[0, 1]
           correlations.append(corr)
       
       return lib_sizes[:len(correlations)], correlations
   
   # Test causality in both directions
   lib_sizes1, corr_x1_to_x2 = ccm(x1, x2, E=3, tau=10)
   lib_sizes2, corr_x2_to_x1 = ccm(x2, x1, E=3, tau=10)
   ```
   
   **Step 3:** Visualize and interpret results
   ```python
   plt.figure(figsize=(10, 5))
   plt.plot(lib_sizes1, corr_x1_to_x2, 'o-', label='X1 → X2 (using X2 to predict X1)')
   plt.plot(lib_sizes2, corr_x2_to_x1, 's-', label='X2 → X1 (using X1 to predict X2)')
   plt.xlabel('Library Size')
   plt.ylabel('Correlation')
   plt.legend()
   plt.title('Convergent Cross Mapping Results')
   plt.grid(True, alpha=0.3)
   plt.show()
   
   # Interpretation: X1→X2 should show convergence (coupling direction)
   ```

3. **Connection to Exercise 15.3-15.4:**
   This example provides the foundation for implementing causal discovery in complex systems like flocks and chaotic attractors.

### Worked Example 3: Transfer Learning for Time Series
**Context:** Let's implement domain adaptation for time series using adversarial training.

1. **Theoretical Background:**
   - Transfer learning leverages knowledge from source domain for target domain
   - Domain adaptation aligns feature distributions between domains
   - Adversarial training can learn domain-invariant representations

2. **Example:**
   Transfer learning from synthetic to real-world patterns.
   
   **Step 1:** Create source and target domains
   ```python
   # Source domain: clean synthetic patterns
   def generate_source_data(n=1000):
       data = []
       labels = []
       for _ in range(n):
           if np.random.rand() > 0.5:
               # Pattern A: increasing trend
               x = np.linspace(0, 1, 50) + 0.1*np.random.randn(50)
               labels.append(0)
           else:
               # Pattern B: seasonal
               x = np.sin(np.linspace(0, 4*np.pi, 50)) + 0.1*np.random.randn(50)
               labels.append(1)
           data.append(x)
       return np.array(data).reshape(n, 50, 1), np.array(labels)
   
   # Target domain: noisy real-world-like patterns
   def generate_target_data(n=500):
       data = []
       labels = []
       for _ in range(n):
           if np.random.rand() > 0.5:
               # Noisy trend with drift
               x = np.linspace(0, 1, 50) + 0.3*np.random.randn(50)
               x += 0.2*np.sin(np.linspace(0, 2*np.pi, 50))
               labels.append(0)
           else:
               # Noisy seasonal with trend
               x = np.sin(np.linspace(0, 4*np.pi, 50)) + 0.3*np.random.randn(50)
               x += 0.1*np.linspace(0, 1, 50)
               labels.append(1)
           data.append(x)
       return np.array(data).reshape(n, 50, 1), np.array(labels)
   
   X_source, y_source = generate_source_data()
   X_target, y_target = generate_target_data()
   ```
   
   **Step 2:** Build domain-adversarial neural network
   ```python
   from tensorflow.keras import layers, Model
   
   def build_dann(input_shape=(50, 1), n_classes=2):
       # Shared feature extractor
       inputs = keras.Input(shape=input_shape)
       x = layers.LSTM(32, return_sequences=True)(inputs)
       x = layers.LSTM(16)(x)
       features = layers.Dense(32, activation='relu', name='features')(x)
       
       # Task classifier
       task_classifier = layers.Dense(n_classes, activation='softmax', 
                                     name='task_classifier')(features)
       
       # Domain discriminator (with gradient reversal)
       domain_discriminator = layers.Dense(1, activation='sigmoid',
                                          name='domain_discriminator')(features)
       
       model = Model(inputs, [task_classifier, domain_discriminator])
       return model
   
   # Custom training loop would include gradient reversal
   model = build_dann()
   ```
   
   **Step 3:** Train with domain adaptation
   ```python
   # Simplified training (full implementation would include gradient reversal)
   def train_step(model, X_source, y_source, X_target):
       # Combine source and target data
       X_combined = np.vstack([X_source, X_target])
       
       # Domain labels (0=source, 1=target)
       domain_labels = np.hstack([
           np.zeros(len(X_source)),
           np.ones(len(X_target))
       ])
       
       # In practice, implement gradient reversal layer
       # Here we show the concept
       with tf.GradientTape() as tape:
           # Forward pass
           task_pred, domain_pred = model(X_combined, training=True)
           
           # Task loss (only on source data)
           task_loss = keras.losses.sparse_categorical_crossentropy(
               y_source, task_pred[:len(X_source)]
           )
           
           # Domain loss (on all data)
           domain_loss = keras.losses.binary_crossentropy(
               domain_labels, domain_pred[:, 0]
           )
           
           # Total loss (with reversed gradient for domain)
           total_loss = tf.reduce_mean(task_loss) - 0.1 * tf.reduce_mean(domain_loss)
       
       # Update weights
       # gradients = tape.gradient(total_loss, model.trainable_variables)
       # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
   
   print("Domain adaptation training concept demonstrated")
   ```

3. **Connection to Exercise 15.5-15.6:**
   This example introduces the concepts of transfer learning and domain adaptation, preparing you for implementing more sophisticated transfer learning systems.

### Worked Example 4: Quantum Computing Concepts for Time Series
**Context:** Let's explore quantum computing concepts applied to time series pattern matching.

1. **Theoretical Background:**
   - Quantum superposition allows exploring multiple states simultaneously
   - Grover's algorithm provides quadratic speedup for search problems
   - Quantum amplitude encodes probability information

2. **Example:**
   Simulate quantum pattern search in time series.
   
   **Step 1:** Classical pattern matching baseline
   ```python
   def classical_pattern_search(time_series, pattern):
       """Classical sliding window search"""
       n = len(time_series)
       m = len(pattern)
       matches = []
       
       for i in range(n - m + 1):
           window = time_series[i:i+m]
           distance = np.sum((window - pattern)**2)
           if distance < 0.1:  # Threshold
               matches.append(i)
       
       return matches
   
   # Test data
   time_series = np.sin(np.linspace(0, 10*np.pi, 1000))
   pattern = np.sin(np.linspace(0, 2*np.pi, 100))
   
   matches = classical_pattern_search(time_series, pattern)
   print(f"Classical search: {len(matches)} matches found")
   ```
   
   **Step 2:** Simulate quantum-inspired search
   ```python
   def quantum_inspired_search(time_series, pattern, iterations=None):
       """
       Simplified simulation of Grover's algorithm concept.
       Real quantum implementation would provide quadratic speedup.
       """
       n = len(time_series) - len(pattern) + 1
       
       # Initialize uniform superposition (all states equally likely)
       amplitudes = np.ones(n) / np.sqrt(n)
       
       # Number of Grover iterations (optimal: ~π/4 * sqrt(n))
       if iterations is None:
           iterations = int(np.pi/4 * np.sqrt(n))
       
       for _ in range(iterations):
           # Oracle: mark states that match pattern
           oracle_marked = np.zeros(n)
           for i in range(n):
               window = time_series[i:i+len(pattern)]
               distance = np.sum((window - pattern)**2)
               if distance < 0.1:
                   oracle_marked[i] = 1
           
           # Grover operator (simplified)
           # 1. Apply oracle (flip phase of marked states)
           amplitudes[oracle_marked == 1] *= -1
           
           # 2. Inversion about average
           avg = np.mean(amplitudes)
           amplitudes = 2 * avg - amplitudes
       
       # Measure (highest probability states)
       probabilities = amplitudes ** 2
       likely_positions = np.where(probabilities > 1/(2*n))[0]
       
       return likely_positions
   
   quantum_matches = quantum_inspired_search(time_series, pattern)
   print(f"Quantum-inspired search: {len(quantum_matches)} likely matches")
   ```
   
   **Step 3:** Visualize quantum advantage concept
   ```python
   # Compare scaling
   sizes = [100, 500, 1000, 5000]
   classical_steps = []
   quantum_steps = []
   
   for n in sizes:
       # Classical: O(n) comparisons
       classical_steps.append(n)
       
       # Quantum: O(sqrt(n)) iterations
       quantum_steps.append(int(np.pi/4 * np.sqrt(n)))
   
   plt.figure(figsize=(10, 6))
   plt.plot(sizes, classical_steps, 'o-', label='Classical O(n)')
   plt.plot(sizes, quantum_steps, 's-', label='Quantum O(√n)')
   plt.xlabel('Time Series Length')
   plt.ylabel('Number of Operations')
   plt.legend()
   plt.title('Quantum Speedup for Pattern Search')
   plt.yscale('log')
   plt.grid(True, alpha=0.3)
   plt.show()
   ```

3. **Connection to Exercise 15.9-15.10:**
   This example introduces quantum computing concepts and demonstrates potential quantum advantages, preparing you to explore more sophisticated quantum algorithms for time series.

These worked examples provide hands-on experience with the cutting-edge techniques discussed in Chapter 15, from deep probabilistic models to quantum computing, preparing you to tackle the challenging research-oriented exercises that follow.