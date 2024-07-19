# Tasks

## Main Sequence

* Modify `tubulaton_post_process.py` to model HyD detector instead of CCD
 * <span style="color: red;"> I don't think there is any difference in the model except parameter-tuning </span>
 * Actually we should account for non-linearity of HyD
* Integrate `tubulaton_post_process.py` into generation pipeline and remove deprecated programs
 * <span style="color: red;"> Done! </span>

* Generate the more realistic training data (**NB: Reuse the tubulaton .vtk files from last run**)
 * <span style="color: red;"> HPC job is pending... </span>
* Pre-processing (classification.py)
 * Change resolution (crop and upsample)
 * What else?
* Track underlying MT structure to avoid problems of feeding black square as "polymerised"
* Train classifier on new data, validate against French data again
* Implement U-Net segmentation

## Side Quests

* Go over the various scales (lengths, numbers of particles, etc) in the problem to check everything adds up
 * <span style="color: red;"> A work in progress </span>
* Check the intensity spectrum of the French data (does it fill up the higher intensity *at all*? - if it does perhaps the low intensity is just shot noise dominating due to weak fluorescence)

## Errands

* ISIT Travel Grant forms
* See VJ email about registration reimbursement
* Swap out QM book at library