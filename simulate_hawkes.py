"""
Main function is simulate_spatiotemporal_hawkes, which generates synthetic events 
based on the Branching Process representation (Algorithm 1 & 2).
"""

import numpy  as np
import pandas as pd
from typing   import Dict, Any, Optional

from utils import compute_branching_ratio_version_a, get_covariance_matrix_version_b


def simulate_spatiotemporal_hawkes( T: float,
                                    model_type: str,
                                    params: Dict[str, Any],
                                    T_e: float=2.0,
                                    S_e: float=0.5,
                                    seed: Optional[int]=None
                                    ) -> pd.DataFrame:
    """
    Simulates a Spatiotemporal Marked Hawkes Process using the Cluster (Branching) method.
    Implements Version A (Separable) and Version B (Non-Separable).

    parameters:
    ----------
    T:          End time of the simulation (observation window [0, T]).
    model_type: 'A' for Separable/Linear, 'B' for Non-Separable.
    params:     Dictionary containing model parameters.
                - Common: 'mu0', 'beta'
                - Version A: 'alpha' (tuple), 'sigma'
                - Version B: 'nu_v', 'nu_a', 'rho'
    T_e:        Temporal buffer extension magnitude.
    S_e:        Spatial buffer extension magnitude.
    seed:       Random seed for reproducibility.

    return values:
    -------------
    df:         Pandas DataFrame with columns ['time', 'valence', 'arousal', 'generation', 'parent_id'].
    """

    # Check input.
    ####################################################################################################################
    assert T > 0,               f'Time horizon T must be positive'
    assert model_type in ['A', 'B'], f'model_type must be A or B'
    assert params['beta'] > 0,  f'Decay rate beta must be positive'
    assert params['mu0'] >= 0,  f'Background rate mu0 must be non-negative'

    if seed is not None: np.random.seed(seed)

    # Setup Simulation Window (Buffer Method).
    # We simulate in [-T_e, T + T_e] x [-S_e, 1 + S_e]^2 to avoid edge effects.
    ####################################################################################################################
    t_start_sim, t_end_sim = -T_e, T + T_e
    space_min, space_max   = -S_e, 1 + S_e
    
    # Define lists to store event data.
    ####################################################################################################################
    events        = []  # list of dicts
    queue         = []  # list of dicts for processing offspring
    event_counter = 0   # unique ID tracker

    # Step 1: Generate Immigrant (Background) Events.
    # Modeled as a Homogeneous Poisson Process over the enlarged volume.
    ####################################################################################################################
    mu0        = params['mu0']
    area_space = (space_max - space_min) ** 2
    duration   = t_end_sim - t_start_sim
    lambda_bg  = mu0 * duration * area_space
    n_bg       = np.random.poisson(lambda_bg)
    
    for _ in range(n_bg):
        t_i = np.random.uniform(t_start_sim, t_end_sim)
        v_i = np.random.uniform(space_min, space_max)
        a_i = np.random.uniform(space_min, space_max)
        
        event = {
            'id':        event_counter,
            'time':      t_i,
            'valence':   v_i,
            'arousal':   a_i,
            'generation': 0,
            'parent_id': -1  # -1 indicates background/immigrant
        }
        
        events.append(event)
        queue.append(event)
        event_counter += 1

    # Step 2: Iterative Offspring Generation.
    # Process the queue until no more offspring are generated.
    ####################################################################################################################
    beta = params['beta']
    
    while queue:
        parent = queue.pop(0)
        
        # 2.1 Calculate Branching Ratio (Expected Offspring).
        ################################################################################################################
        n_offspring = 0
        
        if model_type == 'A':
            # Version A: Mark-dependent branching ratio based on extremity.
            alpha           = params['alpha']
            branching_ratio = compute_branching_ratio_version_a( parent['valence'], parent['arousal'], alpha )
            n_offspring     = np.random.poisson(branching_ratio)
            
        elif model_type == 'B':
            # Version B: Constant infinite-domain branching ratio.
            nu_v  = params['nu_v']
            nu_a  = params['nu_a']
            rho   = params['rho']
            # Derived in Appendix A.3.2
            gamma = (2 * np.pi * nu_v * nu_a) / np.sqrt(1 - rho**2)
            n_offspring = np.random.poisson(gamma)
        
        # 2.2 Generate Offspring Events.
        ################################################################################################################
        for _ in range(n_offspring):
            
            # Temporal Decay: Exponential(beta)
            dt      = np.random.exponential(1.0 / beta)
            t_child = parent['time'] + dt
            
            # Spatial Dispersion
            dv, da = 0.0, 0.0
            
            if model_type == 'A':
                # Isotropic Gaussian
                sigma  = params['sigma']
                dv, da = np.random.normal(0, sigma, 2)
                
            elif model_type == 'B':
                # Anisotropic Gaussian with correlation
                Sigma  = get_covariance_matrix_version_b( params['nu_v'], params['nu_a'], params['rho'] )
                dv, da = np.random.multivariate_normal([0, 0], Sigma)
            
            v_child = parent['valence'] + dv
            a_child = parent['arousal'] + da
            
            # 2.3 Window-based Acceptance.
            # Keep events in buffer zone to produce their own children, filter later.
            ############################################################################################################
            in_time  = t_start_sim <= t_child <= t_end_sim
            in_space = (space_min <= v_child <= space_max) and (space_min <= a_child <= space_max)

            if in_time and in_space:
                child_event = {
                    'id':         event_counter,
                    'time':       t_child,
                    'valence':    v_child,
                    'arousal':    a_child,
                    'generation': parent['generation'] + 1,
                    'parent_id':  parent['id']
                }
                
                events.append(child_event)
                queue.append(child_event)
                event_counter += 1

    # Step 3: Final Filtering & Formatting.
    # Crop to observation window [0, T] x [0, 1]^2 and return as DataFrame.
    ####################################################################################################################
    df = pd.DataFrame(events)
    
    if df.empty: return df
    
    df   = df.sort_values('time').reset_index(drop=True)
    
    mask = ( (df['time']    >= 0) & (df['time']    <= T) &
             (df['valence'] >= 0) & (df['valence'] <= 1) &
             (df['arousal'] >= 0) & (df['arousal'] <= 1) )
    
    final_df = df[mask].copy()
    
    # Store metadata
    final_df.attrs['model_type'] = model_type
    final_df.attrs['params']     = params
    
    return final_df