# Chapter Dependencies

## Purpose
This dependency graph helps with selective content loading when working on specific chapters, keeping token usage manageable for LLM context windows.

## Design Principles
- Maximum dependency depth of 3 chapters
- Topic-based grouping rather than sequential
- Core foundations (Ch 1-3) referenced by many chapters
- Target: Keep dependency chains under 40,000 tokens

## Dependency Graph

```mermaid
graph TD
    %% Core Foundations
    C1[Chapter 1: Introduction]
    C2[Chapter 2: Probability & Inference]
    C3[Chapter 3: Components & Decomposition]
    
    %% Basic Models
    C4[Chapter 4: Linear Models]
    C5[Chapter 5: State Space Models]
    C6[Chapter 6: Spectral Analysis]
    C7[Chapter 7: Nonlinear Models]
    
    %% Advanced Methods
    C8[Chapter 8: Bayesian Computation]
    C9[Chapter 9: Classical ML]
    C9A[Chapter 9A: Deep Learning]
    C9B[Chapter 9B: Probabilistic ML]
    C10[Chapter 10: Advanced Topics]
    
    %% Applications & Theory
    C11[Chapter 11: Causal Inference]
    C12[Chapter 12: Forecasting]
    C13[Chapter 13: Applications]
    
    %% Meta Topics
    C14[Chapter 14: Computational Efficiency]
    C15[Chapter 15: Future Directions]
    
    %% Core Dependencies (most chapters need these)
    C2 --> C1
    C3 --> C1
    
    %% Basic Models Dependencies
    C4 --> C2
    C4 --> C3
    
    C5 --> C3
    C5 --> C2
    
    C6 --> C3
    %% C6 is relatively independent, just needs decomposition concepts
    
    C7 --> C3
    C7 --> C4
    %% Nonlinear extends linear concepts
    
    %% Advanced Methods Dependencies  
    C8 --> C2
    %% Bayesian computation mainly needs probability foundations
    
    C9 --> C3
    C9 --> C2
    %% Classical ML needs basic time series concepts and probability
    
    C9A --> C9
    %% Deep learning builds on classical ML concepts
    
    C9B --> C2
    C9B --> C8
    %% Probabilistic ML needs probability and Bayesian computation
    
    C10 --> C4
    C10 --> C5
    %% Advanced topics builds on basic models
    
    %% Applications & Theory Dependencies
    C11 --> C4
    C11 --> C2
    %% Causality needs VAR models and probability
    
    C12 --> C4
    C12 --> C9
    %% Forecasting uses classical methods and ML
    
    C13 --> C1
    %% Applications just needs basic understanding
    %% Specific sections can load what they need
    
    %% Meta Topics Dependencies
    C14 --> C1
    %% Computational efficiency is mostly standalone
    
    C15 --> C1
    %% Future directions is mostly standalone
    
    %% Optional Cross-References (dotted lines)
    C8 -.-> C5
    %% State space models benefit from MCMC
    
    C9 -.-> C6
    %% ML can use spectral features
    
    C9A -.-> C9
    %% Deep learning may reference classical ML
    
    C12 -.-> C9A
    %% Forecasting may use deep learning
    
    C11 -.-> C7
    %% Causality in nonlinear systems
    
    C11 -.-> C9B
    %% Causality may use graphical models
    
    C12 -.-> C8
    %% Bayesian forecasting
```

## Loading Guide

### Chapter Sizes (for token estimation)
- Chapter 1: 3,606 words (~4,800 tokens)
- Chapter 2: 7,155 words (~9,500 tokens)
- Chapter 3: 5,005 words (~6,700 tokens)
- Chapter 4: 5,959 words (~7,900 tokens)
- Chapter 5: 4,339 words (~5,800 tokens)
- Chapter 6: 4,295 words (~5,700 tokens)
- Chapter 7: 5,833 words (~7,800 tokens)
- Chapter 8: 3,797 words (~5,100 tokens)
- Chapter 9: 4,996 words (~6,600 tokens)
- Chapter 9A: 3,298 words (~4,400 tokens)
- Chapter 9B: 2,656 words (~3,500 tokens)
- Chapter 10: 7,641 words (~10,200 tokens)
- Chapter 11: 5,923 words (~7,900 tokens)
- Chapter 12: 7,066 words (~9,400 tokens)
- Chapter 13: 7,815 words (~10,400 tokens)
- Chapter 14: 6,632 words (~8,800 tokens)
- Chapter 15: 7,792 words (~10,400 tokens)

### Example Loading Instructions

**Working on Chapter 12 (Forecasting):**
- Load: Chapters 1, 2, 3, 4, 9, 12
- Tokens: ~39,000
- Optional: Chapter 9A (deep learning), Chapter 8 (Bayesian forecasting)

**Working on Chapter 15 (Future Directions):**
- Load: Chapters 1, 15
- Tokens: ~15,000
- Very efficient for isolated work

**Working on Chapter 11 (Causal Inference):**
- Load: Chapters 1, 2, 3, 4, 11
- Tokens: ~37,000
- Optional: Chapter 7 (for nonlinear causality)

## Notes
- Dotted lines (-.->)indicate optional dependencies
- Load optional chapters only when working on related specific topics
- Applications chapters (13-15) are intentionally independent for flexibility