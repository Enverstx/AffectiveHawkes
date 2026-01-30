"""
Collection of helper functions for the Spatiotemporal Hawkes Process simulation.
"""

import numpy as np
from typing import Tuple

def compute_branching_ratio_version_a( m_v: float, 
                                       m_a: float, 
                                       alpha: Tuple[float, float, float] ) -> float:
    """
    Computes the branching ratio (expected number of offspring) for Version A.
    Based on Equation (76) in the manuscript.

    parameters:
    ----------
    m_v:    Valence of the parent event (can be outside [0,1] due to buffer).
    m_a:    Arousal of the parent event (can be outside [0,1] due to buffer).
    alpha:  Tuple of (alpha1, alpha2, alpha3).

    return values:
    -------------
    n*:     Expected number of offspring.
    """
    
    # Check input.
    ####################################################################################################################
    # REMOVED: Assertions for [0,1] range to allow buffer zone events to generate offspring.
    # assert 0 <= m_v <= 1, f'Valence must be in [0,1]'
    # assert 0 <= m_a <= 1, f'Arousal must be in [0,1]'
    assert len(alpha) == 3, f'Alpha must have 3 components'

    # Compute deviations and linear combination.
    ####################################################################################################################
    alpha1, alpha2, alpha3 = alpha
    dev_v                  = abs(m_v - 0.5)
    dev_a                  = abs(m_a - 0.5)
    
    return alpha1 * dev_v + alpha2 * dev_a + alpha3 * dev_v * dev_a


def get_covariance_matrix_version_b( nu_v: float, 
                                     nu_a: float, 
                                     rho: float ) -> np.ndarray:
    """
    Constructs the covariance matrix for Version B sampling.
    
    parameters:
    ----------
    nu_v:   Scale parameter for Valence.
    nu_a:   Scale parameter for Arousal.
    rho:    Correlation coefficient (-1 to 1).

    return values:
    -------------
    Sigma:  2x2 Covariance matrix as numpy array.
    """

    # Check input.
    ####################################################################################################################
    assert nu_v > 0,        f'nu_v must be positive'
    assert nu_a > 0,        f'nu_a must be positive'
    assert -1 < rho < 1,    f'rho must be in (-1, 1)'

    # Construct Matrix.
    # Note: We use positive covariance for sampling to match the positive correlation definition.
    ####################################################################################################################
    factor   = 1.0 / (1.0 - rho**2)
    cov_term = rho * nu_v * nu_a
    
    Sigma    = factor * np.array([ [nu_v**2,  cov_term],
                                   [cov_term, nu_a**2] ])
    
    return Sigma