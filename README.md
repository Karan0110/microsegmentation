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
	* You need to set the various environmental variables (PYTHONPATH, etc)

* Look for "TODO TESTING" in config files/scripts when switching between test and production mode
	
# The things we've changed for the testing on HPC:
* `model-config.json5`
* `training-config.json5`
* We need to do something about the hard-coded file paths in `generation`


# Micro(tubule) Segmentation

## Installation

The project is made of 2 separate modules:

* `generation` (generate synthetic data)
* `segmentation` (train/run inference with a model to segment images)

This project was tested on `Python 3.11.8`.

To set up the project:

* Create *separate* virtual environments for each module by running (within the module folder):
	* `python -m venv .venv_generation` 
	* `python -m venv .venv_segmentation`
* You can (de)activate a virtual environment by running the below commands. Ensure you have the correct environment activated when installing/running.
	* `source [virtual environment name]/bin/activate` 
	* `deactivate`
* Install the required libraries by running
	* `pip install -r requirements.txt`
	* There are separate `requirements.txt` files for the 2 modules, and the above command must be run twice, ensuring you have the correct venv active at the time.
* If you want to use the `generation` module, you will need to have [tubulaton](https://gitlab.com/slcu/teamHJ/tubulaton) installed

## Generation

The `generation` module consists of 2 executable files:

* `generate.py`
* `create_tubulaton_config.py` (If you are on Mac/Linux then you can just use `run_tubulaton.sh` to get tubulaton .vtk output files)

<span style='color: red;'> TODO: I haven't included any further explanation of parameters, etc since things might change further (I will do this properly at the very end of the project)

## Segmentation

The `segmentation` module consists of 2 executable files:

* `train.py` (The file `run_train.sh` in `local_scripts` is just a shortcut so you don't have to keep copying the same directory/file paths) 
* `demo.py`

You can see a list of parameters and explanations of them by running `python [insert file] -h`


Some notes on parameters for `train.py`:

* The model name can be any string that forms a valid filename. Some example names:
	* `model-v4`
	* `model-focal-loss` 
* All of the filepaths can be wherever you want (the program will create them if they don't exist already) with the exception of `--datadir`:
	* This must be the location of the synthetic data produced by the `generation` pipeline
	
**Note**: If you want to use the `run_train.sh` - make sure to change all the directories to reflect your local filesystem!

Some notes on the config files:

* The variables marked `//TESTING` are the those that should be changed to small numbers if you want to run a quick test:
	* They can all just be 1 except for `patch_size` which must be a multiple of `2**depth` (see `config/model.json5`) and `depth` which should be at least `2`.
* Many of the objects in the config files are dictionaries that contain a `name` and `params` attribute.
	* The `name` attribute can be anything within the default pytorch/python namespace for the object in question (or one of the custom classes defined in the code)
	* The `params` attribute is directly fed into the constructor for the class by `Class(**params)`
	* For example you coud use the criterion `CrossEntropyLoss` in `config/criterions.json5` as that is contained in pytorch's `torch.nn` namespace.



