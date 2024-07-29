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

