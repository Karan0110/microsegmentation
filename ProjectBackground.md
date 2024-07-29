# Project Background

### How does the Nucleus effect RH cell morphogenesis in Arab. Thal. via the cytoskeleton?

## The Teams

* French (Marie-Edith)
	* Main driver
	* Made the microchannels
* SLCU (Henrik) 
	* Botanicon model
* Japanese:
	* Investigated genetic mutants
* French (Atef)
	* Did further experiements with the microchannels     

 
## Drugs used to Effect Cytoskeleton Network

* Oryzalin (depoly MTs)
* Taxol (stab MTs)
* Latrunculin B (depoly Actin)
* Jasplakinolide (stab Actin)


## Theorized "cross-talk" between MT and Actin

* OZ Treatment causes bulging at the tip
* In Botanicon model this corresponds to changing `L` (size of growth zone in tip)
* But just changing `L` causes faster growth in model than in reality
* The model params related to growth are linked to the Actin network


## Observing the (de)polymerization under OZ

**Q:** Why did the French do this at all to begin with?
**A:** Destroying the cytoskeleton arrests growth, so we can observe the "oscillation" of the nucleus more accurately (SNR higher when no net movement)

Depoly:

* If the microchannel setup works as intended, the depoly should be uniform

Repoly:

* Marie-Edith thinks she observes repoly faster at the shank (between tip/nucleus)
* (Opposed to uniform repoly)
* If she's right (as the ML segmentation would confirm) this means there is more gamma-tubulin at shank and also links into the WIP paper (somehow, haven't read the paper)

## CRWN-1 Mutant

Somehow effects nucleus interaction with cytoskeleton.

## Botanicon Model

Degrees of freedom:

* x1, x2 to track back/front of nucleus
* y to track tip position
* The surface is the surface of revolution of a curve.

See paper "An anisotropic..." for the inspiration.
