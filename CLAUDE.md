# Instructions for AI Assistants Working on This Book

## Project Overview
This is a collaborative book on Time Series Analysis written in the imagined styles of four renowned scientists: Richard Feynman, Andrew Gelman, E.T. Jaynes, and Kevin Murphy. The book presents time series concepts from multiple complementary perspectives.

## Core Principle
The book embodies the authentic thinking styles and approaches of these scientists. Rather than following a rigid template, let their natural voices emerge through the content. Each brings their unique perspective:
- Physics and intuition (Feynman)
- Statistical pragmatism (Gelman)  
- Probability as logic (Jaynes)
- Computational machine learning (Murphy)

## Writing Approach
- Allow the authors' voices to flow naturally from the subject matter
- Let them engage in genuine dialogue and even disagreement
- Maintain mathematical rigor while ensuring accessibility
- Use code to illuminate concepts, not just implement them

## Remember
This is an experiment in scientific communication - showing how different brilliant minds would approach the same fundamental questions about time in data. The goal is not to mimic surface-level quirks but to channel their deeper ways of thinking about problems.

## Dependency Management

When working on a specific chapter or section:
1. Check Dependencies.md for what chapters to load as context
2. Load only the necessary dependencies to stay within token limits
3. Typical load: Core chapters (1-3) + specific dependencies
4. See Dependencies.md for specific loading instructions

## Current Status

### Completed
- Chapters 1-15: Main content
- Chapters 1-10: Exercises and worked examples
- Appendices A-E: Mathematical foundations
- Restructured Chapter 9 into 9, 9A, 9B for better modularity

### TODO (High Priority)
- Exercises for Chapters 11-15
- Worked Examples for Chapters 11-15
- Add Philosophical Interludes throughout
- Add Computational Challenges throughout

### Writing Priorities
1. Maintain author voices consistently
2. Ensure mathematical correctness
3. Provide practical value
4. Make complex topics accessible

## Example Passage Showing Mixed Voices

```
[Feynman] Let's think about stationarity like this: imagine you're recording the temperature in your backyard. If you record it for a year, you'll see it go up and down with the seasons. That's non-stationary - the average temperature in January is different from July.

[Jaynes] From an information-theoretic perspective, stationarity represents a powerful constraint on our probability distributions. It tells us that the information content of our observations is time-invariant, which dramatically reduces the space of possible models.

[Gelman] But here's the thing - perfect stationarity never exists in real data. What matters is whether the departures from stationarity will mess up your analysis. Sometimes a little non-stationarity is fine, sometimes it ruins everything.

[Murphy] Computationally, we can test for stationarity using the Augmented Dickey-Fuller test or KPSS test. Here's how to implement a simple check...
```

## Remember
This book is an experiment in collaborative scientific writing. The goal is not just to teach time series analysis, but to show how different scientific perspectives can illuminate the same concepts in complementary ways.