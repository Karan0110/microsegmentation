* Include a note about the `demo_segmentation_cache`


* When compiling tubulaton on HPC you must run
`module load vtk/7.1.1`
before running `cmake ..`

* When converting to test job for HPC remember:
	* Wall clock time
	* Time steps (for tubulaton)
	* qos=intr

* To run tubulaton on HPC (using python randomisation of parameters):
	* `sbatch --array=1-1000 hpc_run_tubulaton.sh` 

* The virtualenvs need to be installed as .venv_generation and .venv_segmentation in the right folders if you want to use HPC scripts out of the box
	* And you need to load the latest version of python (find with module avail python) before making the venv: `module load python/3.11.0-icl` 

	
	
# The things we've changed for the testing on HPC:
* `model-config.json5`
* `training-config.json5`
* We need to do something about the hard-coded file paths in `generation`