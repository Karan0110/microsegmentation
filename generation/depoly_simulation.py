from typing import Union, Tuple

import numpy as np

def simulate_depoly(mt_points : np.ndarray,
                    mt_ids : np.ndarray,
                    config : dict,
                    proportion : float,
                    unique_mt_ids : Union[np.ndarray, None] = None) -> Tuple[np.ndarray, np.ndarray]:
    depoly_displacement_std = config['microtubules']['depoly_displacement_std']

    if mt_points.shape[0] != mt_ids.shape[0]:
        raise ValueError(f"List of MTs and MT ids aren't aligned properly!\n(Shapes {mt_points.shape} and {mt_ids.shape} respectively)")

    if unique_mt_ids is None:
        unique_mt_ids = np.unique(mt_ids)

    num_mts = unique_mt_ids.shape[0]
    sample_size = int(num_mts * proportion)
    depoly_id_choices = np.random.choice(unique_mt_ids, size=sample_size, replace=False)
    depoly_indices = np.where(np.isin(mt_ids, depoly_id_choices))[0]
    num_depoly_mt_points = depoly_indices.shape[0]

    tubulin_points = mt_points.copy()
    displacement = np.random.normal(loc=0., 
                                    scale=depoly_displacement_std / np.sqrt(3), 
                                    size=(num_depoly_mt_points, 3))
    tubulin_points[depoly_indices] += displacement

    mt_points = np.delete(mt_points, depoly_indices, axis=0)

    return tubulin_points, mt_points
