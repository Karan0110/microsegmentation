## To Do List

### Work

* Modify `tubulaton_post_process.py` to model HyD detector instead of CCD
 * <span style="color: red;"> I don't think there is any difference in the model except parameter-tuning </span>
 * Actually we should account for non-linearity of HyD
* Integrate `tubulaton_post_process.py` into generation pipeline and remove deprecated programs
* Go over the various scales (lengths, numbers of particles, etc) in the problem to check everything adds up
* Generate the more realistic training data (**NB: Reuse the tubulaton .vtk files from last run**)
* Train classifier on new data, validate against French data again
* Implement U-Net segmentation
* Check the intensity spectrum of the French data (does it fill up the higher intensity *at all*? - if it does perhaps the low intensity is just shot noise dominating due to weak fluorescence)

### ISIT

* ISIT Travel Grant forms
* See VJ email