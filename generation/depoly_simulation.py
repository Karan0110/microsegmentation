import numpy as np

def simulate_depoly(mt_points : np.ndarray,
                    config : dict) -> np.ndarray:
    depoly_displacement_std = config['microtubules']['depoly_displacement_std']

    depoly_mt_points = mt_points.copy()

    displacement = np.random.normal(loc=0., 
                                    scale=depoly_displacement_std / np.sqrt(3), 
                                    size=mt_points.shape)
    depoly_mt_points += displacement

    return depoly_mt_points
