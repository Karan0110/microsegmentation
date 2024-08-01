## Model History

**REMEMBER:** The augmentation pipeline has been manually turned off in the code (see TODO)

* V1: U-Net model
* V2: Changed the convolution padding mode from 'zeros' --> 'reflect' (goal: stop the edge artifacts)
* V3: Used globally weighted CE to address class imbalance (but accidentally changed the CPM back to 'zeros'!)
* V4: Reduced learning rate from 1e-4 to 2e-5 (Model wasn't converging anymore)
* V5: Changed to focal loss (gamma = 5.0, no weighting)