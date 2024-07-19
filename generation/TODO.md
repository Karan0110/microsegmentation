# Areas for Improvement

## My Code 

* Check over the parameters with biologist (especially where we used typical value readouts from ChatGPT)
* The fluorophores are currently modelled as "blobs" (as there are too many), is this the best way? (Maybe use DEs and a concentration field)
* We aren't using z-position for the PSF
* Make the z-slice position variable (instead of fixed proportion)
* Use a Brownian motion (reflecting against the cell wall) instead of normal perturbation for depolymerisation modelling (Easy analytic solution when the boundary is cuboid, and sphercylinder is close enough)
* Modify `generate_config.json5` to accomodate random parameters (supply distribution and parameters instead of constant value)
* Actually use mesh map to get boundary of cell instead of treating it as a cuboid and taking min/max of intensities (very hacky)

## Tubulaton

* Change parameter names in config.ini
* The MTs shouldn't permeat the vacuole or the root tip
