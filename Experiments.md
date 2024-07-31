## Model History

**REMEMBER:** The augmentation pipeline has been manually turned off in the code (see TODO)

* V1: U-Net model
* V2: Changed the convolution padding mode from 'zeros' --> 'reflect' (goal: stop the edge artifacts)
* V3: Used globally weighted CE to address class imbalance
* V4: Added data augmentation