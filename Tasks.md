# Tasks

### Big Question: How does the nucleus, via the cytoskeleton, effect RH cell morphogenesis in Arab Thal?

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
 * <span style="color: red;"> Side-stepped by going straight to converting training data to be ready for a U-Net
* Implement U-Net segmentation (Just copy/paste from that skin lesion practice project you did)
* Train classifier on new data, validate against French data again

## Side Quests

* Go over the various scales (lengths, numbers of particles, etc) in the problem to check everything adds up
 * <span style="color: red;"> A work in progress </span>
* Check the intensity spectrum of the French data (does it fill up the higher intensity *at all*? - if it does perhaps the low intensity is just shot noise dominating due to weak fluorescence)
* Git: Check how to deal with removing files retroactively in .gitignore and how to pull a project (for copying onto HPC nicely)
* Make a personal website/blog

## Errands

* ISIT Reimbursements:
 * Hotel: 531 EUR = £447
 * Airfare: 562 GBP (101 USD = £78 left after ISIT Travel Grant)
 * Hotel+Airfare = £1009
* Swap out QM book at library