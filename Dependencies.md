graph TD
    %% Chapter 1: Introduction to Time Series
    C1_1["1.1 What is a Time Series?"]
    C1_2["1.2 Historical Context and Applications"] --> C1_1
    C1_3["1.3 Key Concepts and Terminology"] --> C1_1
    C1_4["1.4 Computational Thinking in Time Series Analysis"] --> C1_1

    %% Chapter 2: Foundations of Probability and Inference in Time Series
    C2_1["2.1 Probability Theory: A Measure of Knowledge"] --> C1_3
    C2_2["2.2 Bayesian Inference in Time Series"] --> C2_1
    C2_3["2.3 Frequentist Perspectives and Their Role"] --> C2_1
    C2_4["2.4 Information Theory in Time Series Analysis"] --> C2_1
    C2_5["2.5 Synthesis: Choosing Appropriate Methods for Time Series Problems"] --> C2_2
    C2_5 --> C2_3
    C2_5 --> C2_4

    %% Chapter 3: Time Series Components and Decomposition
    C3_1["3.1 Trend, Seasonality, and Cyclical Components"] --> C1_3
    C3_2["3.2 Noise and Irregularities: An Information-Theoretic View"] --> C3_1
    C3_3["3.3 Stationarity and Ergodicity: Bayesian and Frequentist Perspectives"] --> C3_1
    C3_4["3.4 Modern Decomposition Techniques: Empirical Mode Decomposition and Beyond"] --> C3_3

    %% Chapter 4: Linear Time Series Models
    C4_1["4.1 Autoregressive (AR) Models: A Bayesian Approach"] --> C3_3
    C4_2["4.2 Moving Average (MA) Models"] --> C4_1
    C4_3["4.3 ARMA and ARIMA Models: Bayesian vs. Frequentist Estimation"] --> C4_2
    C4_3 --> C2_5
    C4_4["4.4 Seasonal Models and Their Bayesian Treatment"] --> C4_3
    C4_5["4.5 Vector Autoregressive (VAR) Models"] --> C4_4

    %% Chapter 5: State Space Models and Filtering
    C5_1["5.1 State Space Representation: A Unifying Framework"] --> C4_3
    C5_2["5.2 Kalman Filter and Its Bayesian Interpretation"] --> C5_1
    C5_3["5.3 Particle Filters and Sequential Monte Carlo Methods"] --> C5_2
    C5_4["5.4 Bayesian Structural Time Series Models"] --> C5_3

    %% Chapter 6: Spectral Analysis and Filtering
    C6_1["6.1 Fourier Analysis and the Frequency Domain"] --> C4_2
    C6_2["6.2 Spectral Density Estimation: Bayesian and Classical Approaches"] --> C6_1
    C6_3["6.3 Wavelet Analysis for Time Series"] --> C6_2
    C6_4["6.4 Hilbert-Huang Transform and Empirical Mode Decomposition"] --> C6_3
    C6_4 --> C3_4

    %% Chapter 7: Nonlinear Time Series Analysis
    C7_1["7.1 Introduction to Nonlinear Dynamics in Time Series"] --> C4_5
    C7_2["7.2 Nonlinear Autoregressive Models"] --> C7_1
    C7_3["7.3 Threshold and Regime-Switching Models"] --> C7_2
    C7_4["7.4 Chaos Theory and Strange Attractors in Time Series"] --> C7_3

    %% Chapter 8: Bayesian Computation for Time Series
    C8_1["8.1 Markov Chain Monte Carlo (MCMC) Methods for Time Series"] --> C5_3
    C8_2["8.2 Hamiltonian Monte Carlo and Its Applications"] --> C8_1
    C8_3["8.3 Variational Inference for Time Series Models"] --> C8_2
    C8_4["8.4 Approximate Bayesian Computation in Time Series Analysis"] --> C8_3

    %% Chapter 9: Machine Learning Approaches to Time Series
    C9_1["9.1 Feature Engineering for Time Series"] --> C8_3
    C9_2["9.2 Kernel Methods for Time Series"] --> C9_1
    C9_3["9.3 Decision Trees and Random Forests for Time Series"] --> C9_1
    C9_4["9.4 Support Vector Machines for Time Series"] --> C9_2
    C9_5["9.5 Deep Learning for Time Series: CNNs, RNNs, and LSTMs"] --> C9_3
    C9_6["9.6 Attention Mechanisms and Transformers for Time Series"] --> C9_5
    C9_7["9.7 Gaussian Processes for Time Series"] --> C9_6
    C9_8["9.8 Probabilistic Graphical Models for Time Series"] --> C9_7

    %% Chapter 10: Advanced Topics in Time Series Analysis
    C10_1["10.1 Long Memory Processes and Fractional Differencing"] --> C9_7
    C10_2["10.2 Time Series on Networks"] --> C10_1
    C10_3["10.3 Multivariate and High-Dimensional Time Series Analysis"] --> C10_2
    C10_4["10.4 Functional Time Series"] --> C10_3
    C10_5["10.5 Point Processes and Temporal Point Patterns"] --> C10_3
    C10_6["10.6 Bayesian Nonparametric Methods for Time Series"] --> C10_4

    %% Chapter 11: Causal Inference in Time Series
    C11_1["11.1 Granger Causality and Its Limitations"] --> C10_3
    C11_2["11.2 Bayesian Approaches to Causal Inference in Time Series"] --> C11_1
    C11_3["11.3 Structural Causal Models for Time Series"] --> C11_2
    C11_4["11.4 Causal Discovery in Time Series"] --> C11_3
    C11_5["11.5 Intervention Analysis and Causal Impact"] --> C11_4

    %% Chapter 12: Time Series Forecasting
    C12_1["12.1 Traditional Forecasting Methods: Moving Averages to ARIMA"] --> C4_3
    C12_2["12.2 Bayesian Forecasting"] --> C12_1
    C12_3["12.3 Machine Learning for Time Series Forecasting"] --> C9_8
    C12_4["12.4 Ensemble Methods and Forecast Combination"] --> C12_3
    C12_5["12.5 Probabilistic Forecasting and Uncertainty Quantification"] --> C12_4
    C12_6["12.6 Long-term Forecasting and Scenario Analysis"] --> C12_5

    %% Chapter 13: Applications and Case Studies
    C13_1["13.1 Financial Time Series Analysis"] --> C12_6
    C13_2["13.2 Environmental and Climate Time Series"] --> C13_1
    C13_3["13.3 Biomedical Time Series Analysis"] --> C13_1
    C13_4["13.4 Economic Time Series and Policy Analysis"] --> C13_2
    C13_5["13.5 Industrial and Engineering Applications"] --> C13_4

    %% Chapter 14: Computational Efficiency and Practical Considerations
    C14_1["14.1 Efficient Algorithms for Time Series Analysis"] --> C13_5
    C14_2["14.2 Parallel and Distributed Computing for Time Series"] --> C14_1
    C14_3["14.3 Online Learning and Streaming Data Analysis"] --> C14_2
    C14_4["14.4 Software Tools and Libraries for Time Series Analysis"] --> C14_3
    C14_5["14.5 Best Practices in Time Series Modeling and Forecasting"] --> C14_4

    %% Chapter 15: Future Directions and Open Problems
    C15_1["15.1 Deep Probabilistic Models for Time Series"] --> C14_5
    C15_2["15.2 Causal Discovery in Complex, Nonlinear Time Series"] --> C15_1
    C15_3["15.3 Transfer Learning and Meta-Learning for Time Series"] --> C15_2
    C15_4["15.4 Interpretable AI for Time Series Analysis"] --> C15_3
    C15_5["15.5 Quantum Computing Approaches to Time Series Modeling"] --> C15_4
