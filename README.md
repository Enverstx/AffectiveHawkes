# AffectiveHawkes
AffectiveHawkes is a Python package for simulating Spatiotemporal Marked Hawkes Processes to model emotional contagion in continuous Valence-Arousal space.

The simulation utilizes the **exact cluster (branching) process representation** of Hawkes processes. This approach avoids the computational overhead of the thinning algorithm ($O(n^2)$) by iteratively generating offspring events from immigrant (background) events. To mitigate boundary effects inherent in finite-domain simulations, the package implements a **buffer method**, simulating events on an enlarged domain before filtering for the target observation window.

The methods implemented in this package support two model specifications detailed in:  
Qiu, Z., Sornette, D. and Lera, S. C. (2025). *Spatiotemporal Emotional Contagion in Continuous Affect Space*.

## Installation

You can install the dependencies and run the package locally:

```bash
git clone [https://github.com/Enverstx/AffectiveHawkes.git](https://github.com/Enverstx/AffectiveHawkes.git)
cd AffectiveHawkes
pip install numpy pandas seaborn matplotlib
```

## Usage
Here is a simple example to get you started with AffectiveHawkes:
```python
import numpy as np
import AffectiveHawkes.simulate_hawkes as ah

# ------------------------------------------------------------------------------
# Example 1: Version A (Separable Structure)
# ------------------------------------------------------------------------------
# In this version, the strength function is mark-dependent (based on extremity), 
# while the spatial kernel is an isotropic Gaussian.

# Define model parameters
params_a = {
    'mu0': 0.5,                 # Exogenous background intensity
    'beta': 1.0,                # Temporal decay rate
    'alpha': (0.6, 0.4, 0.2),   # Strength parameters (Valence, Arousal, Interaction)
    'sigma': 0.1                # Standard deviation of spatial diffusion
}

# Run simulation
# We simulate over [0, T] but internally use a buffer (T_e, S_e) to handle edge effects.
df_a = ah.simulate_spatiotemporal_hawkes(
    T=100.0,
    model_type='A',
    params=params_a,
    T_e=2.0,    # Temporal buffer extension
    S_e=0.5,    # Spatial buffer extension
    seed=42     # For reproducibility
)

print(f"Version A: Generated {len(df_a)} events.")
print(df_a.head())


# ------------------------------------------------------------------------------
# Example 2: Version B (Non-Separable Structure)
# ------------------------------------------------------------------------------
# In this version, the strength and spatial propagation are coupled via an 
# anisotropic Gaussian kernel, allowing for correlation between Valence and Arousal.

# Define model parameters
params_b = {
    'mu0': 0.5,
    'beta': 1.0,
    'nu_v': 0.05,   # Scale parameter for Valence
    'nu_a': 0.05,   # Scale parameter for Arousal
    'rho':  0.5     # Correlation between Valence and Arousal propagation
}

# Run simulation
df_b = ah.simulate_spatiotemporal_hawkes(
    T=100.0, 
    model_type='B', 
    params=params_b, 
    seed=101
)

print(f"Version B: Generated {len(df_b)} events.")
```

You can also run the included example script directly to visualize the results:
```python
python examples.py
```
