"""
Example script to demonstrate the simulation of Spatiotemporal Hawkes Processes.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from simulate_hawkes import simulate_spatiotemporal_hawkes

def run_example_version_a():
    """
    Runs a simulation for Model Version A (Separable) and plots results.
    """
    print("Running Simulation for Version A (Separable)...")
    
    # Define parameters (Stability check: 0.5*0.6 + 0.5*0.4 + 0.25*0.2 = 0.55 < 1)
    params_a = {
        'mu0':   0.5,
        'beta':  1.0,
        'alpha': (0.6, 0.4, 0.2), 
        'sigma': 0.1
    }
    
    df_a = simulate_spatiotemporal_hawkes( T=100.0, 
                                           model_type='A', 
                                           params=params_a, 
                                           seed=1 )
    
    print(f"Generated {len(df_a)} events.")
    print(df_a.head())

    # Plotting
    ####################################################################################################################
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot in V-A space
    sns.scatterplot( data=df_a, x='valence', y='arousal', hue='generation', 
                     palette='viridis', alpha=0.6, ax=ax[0] )
    ax[0].set_title("Version A: V-A Space")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
    
    # Time series
    ax[1].plot(df_a['time'], df_a['valence'], '.', markersize=2, label='Valence')
    ax[1].set_title("Temporal Dynamics")
    ax[1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()


def run_example_version_b():
    """
    Runs a simulation for Model Version B (Non-Separable).
    """
    print("\nRunning Simulation for Version B (Non-Separable)...")
    
    # Define parameters (Stability check: gamma approx 0.018 < 1)
    params_b = {
        'mu0':   0.5,
        'beta':  1.0,
        'nu_v':  0.05,
        'nu_a':  0.05,
        'rho':   0.5 
    }
    
    df_b = simulate_spatiotemporal_hawkes( T=100.0, 
                                           model_type='B', 
                                           params=params_b, 
                                           seed=101 )
    
    print(f"Generated {len(df_b)} events.")


if __name__ == "__main__":
    run_example_version_a()
    run_example_version_b()