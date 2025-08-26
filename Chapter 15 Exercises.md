# Chapter 15 Exercises

## Exercises

### Deep Probabilistic Models

**Exercise 15.1** (Feynman Style - Understanding Through Building)
Let's build intuition about deep probabilistic models by starting simple and adding complexity.

a) Start with a deterministic autoencoder for time series. Train it on a sine wave. What does the latent space look like?
b) Now add noise to the latent space (make it a VAE). How does this change what the model learns?
c) Replace the sine wave with a chaotic system (like the Lorenz attractor). Can your VAE learn a meaningful latent representation?
d) Here's the key insight question: What is the VAE really learning - the dynamics or just a compression? Design an experiment to tell the difference.

**Exercise 15.2** (Deep Time Series Generation)
Implement a complete deep generative model for time series:

```python
class DeepTimeSeriesGenerator:
    def __init__(self, architecture='vae'):
        """
        Initialize generator.
        architectures: 'vae', 'gan', 'flow', 'diffusion'
        """
        pass
    
    def fit(self, time_series_dataset):
        """Train the generative model."""
        pass
    
    def generate(self, n_samples, length, conditional=None):
        """Generate new time series."""
        pass
    
    def interpolate(self, ts1, ts2, steps=10):
        """Interpolate between two time series in latent space."""
        pass
    
    def evaluate_quality(self, generated, real):
        """
        Evaluate generated time series quality.
        Metrics: discriminative score, MMD, signature MMD
        """
        pass
```

Requirements:
- Implement at least VAE and GAN architectures
- Handle multivariate time series
- Include attention mechanisms for long-range dependencies
- Compare quality of generated samples across architectures

### Causal Discovery

**Exercise 15.3** (Mixed Voices - The Causal Challenge)
[Feynman] Imagine you're watching a flock of birds. Each bird adjusts its position based on its neighbors. Can you figure out who's influencing whom just by watching?

[Jaynes] Information theory tells us that causal influence should manifest as directed information flow.

[Gelman] But correlation isn't causation, and in complex systems, everything correlates with everything else.

[Murphy] Let's build algorithms to tackle this.

Your challenge:
a) Simulate a flock using a Boids model (3 rules: separation, alignment, cohesion).
b) Record time series of bird positions.
c) Apply three causal discovery methods:
   - Convergent Cross Mapping (CCM)
   - Transfer Entropy
   - Neural Granger Causality
d) Compare what each method discovers about the causal structure.
e) Now add a "leader" bird that others follow more strongly. Can your methods detect this asymmetry?
f) What happens if you add noise or missing data? Which method is most robust?

**Exercise 15.4** (Causal Discovery in Chaos)
Implement causal discovery for chaotic systems:

```python
def discover_chaos_causality(system='lorenz_coupled'):
    """
    Discover causal structure in coupled chaotic systems.
    
    Systems:
    - lorenz_coupled: Two coupled Lorenz attractors
    - rossler_lorenz: Coupled Rössler-Lorenz system
    - custom: User-defined coupled ODEs
    """
    # Generate data from coupled chaotic systems
    # Apply multiple causal discovery methods
    # Handle the challenges of chaos:
    #   - Sensitive dependence on initial conditions
    #   - Fractal attractors
    #   - Multiple time scales
    pass
```

Requirements:
- Implement CCM with automatic embedding dimension selection
- Use recurrence plots for causal analysis
- Compare with deep learning approaches (e.g., TCDF)
- Analyze how coupling strength affects detectability

### Transfer Learning

**Exercise 15.5** (Jaynes Style - Information Transfer)
From an information-theoretic perspective, transfer learning is about extracting invariant information that generalizes across domains.

a) Define a measure of "transferable information" between two time series domains.
b) Prove that for stationary Gaussian processes with similar correlation structures, the transferable information is maximized by matching the spectral densities.
c) Implement a method that identifies which components of a model (features, layers, parameters) contain the most transferable information.
d) Design an experiment with synthetic data where you control exactly what should transfer, and verify your theory.

**Exercise 15.6** (Cross-Domain Time Series Transfer)
Build a comprehensive transfer learning system:

```python
class TimeSeriesTransferLearning:
    def __init__(self, base_model):
        self.base_model = base_model
        self.domain_discriminator = None
        
    def adversarial_adaptation(self, source_data, target_data):
        """
        Use adversarial training to align domains.
        """
        pass
    
    def progressive_adaptation(self, source_data, intermediate_domains, target_data):
        """
        Gradually adapt through intermediate domains.
        """
        pass
    
    def meta_learning_adaptation(self, task_distribution):
        """
        Use MAML or similar for few-shot adaptation.
        """
        pass
    
    def zero_shot_transfer(self, source_data, domain_description):
        """
        Transfer to new domain using only description.
        """
        pass
```

Test on real transfer scenarios:
- Financial markets (US → emerging markets)
- Weather patterns (temperate → tropical regions)
- Human activity (young adults → elderly)
- Industrial sensors (one machine → different model)

### Interpretable AI

**Exercise 15.7** (Gelman Style - Trust but Verify)
Interpretability methods can be misleading. Let's test them thoroughly:

a) Create a "pathological" time series model that appears to use certain features but actually ignores them (use careful initialization and architecture design).
b) Apply SHAP, LIME, and Integrated Gradients. Do they detect the deception?
c) Design a suite of "sanity checks" for time series interpretability:
   - Sensitivity to random permutation
   - Consistency across similar inputs
   - Agreement with known ground truth
d) Create a diagnostic tool that warns when interpretability methods might be unreliable.

**Exercise 15.8** (Building Interpretable Deep Models)
Design architectures that are inherently interpretable:

```python
class InterpretableTimeSeriesModel:
    def __init__(self):
        self.trend_module = None
        self.seasonal_module = None
        self.attention_module = None
        
    def build_decomposable_architecture(self):
        """
        Architecture where each module has clear interpretation.
        """
        pass
    
    def extract_learned_patterns(self):
        """
        Extract and visualize what each module learned.
        """
        pass
    
    def generate_explanations(self, prediction):
        """
        Generate human-readable explanations.
        E.g., "Prediction is high because of strong weekly pattern 
        and upward trend detected in last 30 days"
        """
        pass
```

Requirements:
- Modules should correspond to interpretable time series components
- Use attention weights to show which time points matter
- Generate both local and global explanations
- Compare accuracy vs. interpretability trade-offs

### Quantum Computing

**Exercise 15.9** (Feynman Style - Quantum Intuition)
Let's build intuition about quantum computing for time series:

a) Classical bit strings can represent discretized time series values. How would quantum superposition change this?
b) Implement a classical simulation of Grover's algorithm to search for patterns in a time series. When would the quantum version provide speedup?
c) The Heisenberg uncertainty principle limits simultaneous knowledge of position and momentum. Is there an analogous trade-off in time series between knowing exact values and trends?
d) Design a thought experiment: If you could put a time series "in superposition," what would that mean and what could you compute that you couldn't classically?

**Exercise 15.10** (Quantum Time Series Algorithms)
Implement quantum algorithms for time series (using simulators):

```python
from qiskit import QuantumCircuit, execute, Aer

class QuantumTimeSeriesProcessor:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        
    def quantum_pattern_matching(self, time_series, pattern):
        """
        Use Grover's algorithm for pattern search.
        """
        pass
    
    def quantum_anomaly_detection(self, time_series):
        """
        Use quantum amplitude estimation for outlier detection.
        """
        pass
    
    def quantum_forecast(self, time_series):
        """
        Implement quantum version of ARIMA or similar.
        """
        pass
    
    def benchmark_vs_classical(self, algorithm, input_sizes):
        """
        Compare quantum advantage for different problem sizes.
        """
        pass
```

Analyze:
- At what problem size does quantum advantage appear?
- How does noise affect the results?
- Which time series problems are most amenable to quantum speedup?

### Integration Challenge

**Exercise 15.11** (The Grand Unification)
[Murphy] Let's combine everything we've learned about future directions.

Build a system that:
a) Uses deep probabilistic models to learn representations
b) Discovers causal structures in the latent space
c) Transfers knowledge across domains
d) Provides interpretable explanations
e) Identifies components suitable for quantum acceleration

Specifically:
```python
class NextGenTimeSeriesSystem:
    def __init__(self):
        self.deep_model = DeepProbabilisticTSModel()
        self.causal_engine = CausalDiscoveryEngine()
        self.transfer_module = TransferLearningModule()
        self.interpreter = InterpretabilityModule()
        self.quantum_accelerator = QuantumAccelerator()
        
    def end_to_end_pipeline(self, multi_domain_data):
        """
        Process multiple related time series datasets:
        1. Learn shared representations
        2. Discover causal structures
        3. Transfer patterns across domains
        4. Explain predictions
        5. Identify quantum-accelerable components
        """
        pass
```

Apply to a real complex system:
- Climate data from multiple regions
- Global financial markets
- Multi-site industrial IoT sensors
- Biomedical data from multiple patients

### Research Frontiers

**Exercise 15.12** (Push the Boundaries)
Choose one open problem and make a contribution:

a) **Continual Learning for Non-Stationary Time Series**: Design a system that never stops learning, adapting to distribution shifts without forgetting past patterns.

b) **Time Series Foundation Models**: Can we create a "GPT for time series" that works across domains? Design the architecture and training strategy.

c) **Quantum-Classical Hybrid Algorithms**: Develop algorithms that optimally combine quantum and classical processing for time series.

d) **Causal Representation Learning**: Learn representations that explicitly encode causal structures, not just correlations.

e) **Interpretable Deep Time Series Models by Design**: Create architectures where every component has a clear time series interpretation.

Write a research paper including:
- Literature review and positioning
- Theoretical contributions
- Experimental validation
- Limitations and future work
- Code and reproducible experiments