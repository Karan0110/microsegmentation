## Useful commands/information for HPC

* To check HPC usage:
`mybalance`

* To make job a test: `#SBATCH --qos=intr`

* Account names in job script for billing:
 * JONSSON-SL3-GPU
 * JONSSON-SL3-CPU

* Mounted dir paths:
 * `/rds/user/ke330/hpc-work/`
 * `/Users/karan/mount/hpc_uni/`

* The website says to use module load python, but you should run:
	* Running  
	* `module load python/3.8` 

* Before doing module load calls you should check using
	* `module avail [keyword]` 
	* A lot of the version numbers of university HPC docs are out of date.

* Check what version of cuda is used by installed pytorch with
	* `torch.version.cuda` (run in python)
	* Then run `module load cuda/[version]`
 