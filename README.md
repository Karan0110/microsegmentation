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
* If you want to use the `generation` module, you will need to have [tubulaton](https://gitlab.com/slcu/teamHJ/tubulaton/-/tree/ke330) installed on your computer. Ensure you are using the `ke330` branch.
* You need to set the environment variable `PYTHONPATH` to the path of your `microsegmentation` directory. This can be achieved either by:
	* Running`export PYTHONPATH="/Users/karan/microsegmentation"` (but replace the path to reflect your local directory structure)
	* Inserting this line into *both* of your virtual environment activation scripts at the end, i.e. `[virtual environment name]/bin/activate`(Recommended)
* **Optional**: For ease of use you can store all the filepaths, etc that rarely change between runs in a `.env` file. However all the same functionalities are available without this step (at the cost of having to include all file paths as command line arguments!)
	* See "Environment Files" section for more information 

## Environment Files

Both modules can automatically read specific values (mainly file paths) stored in a `.env` file. There should be 2 separate `.env` files at the root of both modules.

**Note**: The cleanest way to use this functionality is by making `.env` files. But if for some reason you wish to directly export the variables to your working environment, they can still be detected by the program.

See below for example `.env` files that show all the variables you can define and what they represent. You can use these on your own machine once you edit the file paths to reflect your local directory structure.

### Generation example `.env` file

	# Root path of compiled tubulaton installation 
	TUBULATON_PATH=/Users/karan/tubulaton
	
	# The parent directory of the various config files you would like to use for generation 
	# (Note: you specify the specific config file within the directory as a CL argument)
	# You can use the config files provided by the project verbatim
	GENERATE_CONFIG_DIR=/Users/karan/microsegmentation/generation/generate_configs/
	
	# The parent directory of the various config files you would like to use for running tubulaton 
	# (Note: you specify the specific config file within the directory as a CL argument)
	# You can use the config files provided by the project verbatim
	TUBULATON_CONFIG_DIR=/Users/karan/microsegmentation/generation/tubulaton_configs/
	
	# Where should the .vtk file outputs of tubulaton be stored?
	# This will be created by the program if it does not exist already, so you can use any sensisble location 
	# (But note the .vtk files take up quite a lot of memory - on the order of GBs!)
	TUBULATON_OUTPUT_DIR=/Users/karan/tubulaton/tubulaton-run
	
	# Where should the final segmented 2D synthetic data be stored?
	# This will be created by the program if it does not exist already, so you can use any sensisble location 
	GENERATE_OUTPUT_DIR=/Users/karan/microsegmentation/SyntheticData
	
### Segementation example `.env` file

	# The root directory of your microsegmentation installation
	BASE_DIR=/Users/karan/microsegmentation
	
	# NOTE: Relative paths are taken relative to BASE_DIR
	# You can also just specify an absolute directory 
	# (e.g. if one of the below directories doesn't belong to BASE_DIR)
	
	# Location of the synthetic training data (from the generation module)
	DATA_PATH=SyntheticData_TRIVIAL_DENSE
	
	# Path to config for training, model, etc 
	# It may either be a single .json5 file or a folder of .json5 files that will be stitched together
	# by the program (as is done in the provided config files) 
	CONFIG_PATH=segmentation/configs/test_config
	
	# Path to demo config file
	# It may either be a single .json5 file or a folder of .json5 files that will be stitched together
	# by the program (as is done in the provided config files) 
	DEMO_CONFIG_PATH=segmentation/demo_config.json5
	
	# Where should demos (from demo.py) be stored?
	# Directory will be created by program, so any sensible path is fine
	DEMO_SAVE_PATH=segmentation_demos
	
	# For which data should a demo be produced (when running demo.py)
	DEMO_INPUT_PATH=DemoData
	
	# Where should the tensorboard log files be stored?
	# Directory will be created by program, so any sensible path is fine
	LOG_PATH=runs
	
	# Where should the model save files be stored?
	# Directory will be created by program, so any sensible path is fine
	# Note: The model save files will be very large (on the order of GBs)
	MODELS_PATH=Models
	
	# Uncomment to manually set num workers to use in DataLoader (rule of thumb: number of CPUs)
	# Otherwise program automatically decides the value (recommended unless you know what you're doing!)
	# NUM_WORKERS=4


## Generation

The `generation` module consists of 2 executable files:

* `generate.py`
* `create_tubulaton_config.py` (If you are on Mac/Linux then you can just use `run_tubulaton.sh` to get tubulaton .vtk output files)

# <span style='color: red;'> TODO </span>

## Segmentation

The `segmentation` module consists of 3 executable files:

* `train.py` (The file `run_train.sh` in `local_scripts` is just a shortcut so you don't have to keep copying the same directory/file paths) 
* `demo.py`
* `timeseries_inference.py`

You can see a list of parameters and explanations of them by running `python [insert file] -h`

### Train

* If you have set-up an `.env` file (recommended) then you can just run:
	* `python train.py --name Heather -v ` 
	* Since the `--epochs` argument is not set, the program will run until you perform a keyboard interrupt
	* Note: "Heather" is just an arbitrary example, any string that you could use as a name for a folder on your computer will work just fine here!

* If you are running on Mac OS X and want to train for a long time, away from keyboard (e.g. overnight), then make sure to prepend `caffeinate -i` to avoid idling:
	* `caffeinate -i python train.py --name Heather -v`
* After/during training your model you can view the train/test loss curves using TensorBoard:
	* `tensorboard --logdir=runs` 
	* Make sure you run this whilst in the root `microsegmentation` directory (or just replace `runs` with the appropriate relative/absolute file path)
	* Replace `runs` with one of its subfolders (e.g. `runs/Heather`) to only view metrics for a specific model.

* In practice it is helpful to modify various parameters (e.g. learning rate) after a few epochs after observing the loss curves.
	* Terminate training if still in progress (You can always use a KeyboardInterrupt - all relevant checkpoints, etc are saved at the end of each epoch) 
	* Navigate to the model folder (if you have used all the default relative paths then this will be: `[your local file structure]/microsegmentation/Models/Heather`)
	* Modify the file `config.json5` within the model folder
		* Unlike the config folder provided with the project, the json5 files are all stitched together into one single file.
		* You cannot modify any of the values in the model attribute of the config file (the weights have been saved for that specific architecture - you need to make a new model if you wish to modify the architecture)
	* Resume training by running `train.py` with the same model name (when resuming from a checkpoint the program will disregard any config files passed through environment variables or CL args)  


### Config Files

* The variables marked `//TESTING` are the those that should be changed (to small numbers) if you want to run a quick test that `train.py` works:
	* They can all just be 1 except for `patch_size` which must be a multiple of `2**depth`  and `depth` which should be at least `2` (see `model.json5`)
* Many of the objects in the config files are dictionaries that contain a `name` and `params` attribute.
	* The `name` attribute can be anything within the default pytorch/python namespace for the object in question (or one of the custom classes defined in the code)
	* The `params` attribute is directly fed into the constructor for the class by `Class(**params)`
	* For example you coud use the criterion `CrossEntropyLoss` in `config/criterions.json5` as that is contained in pytorch's `torch.nn` namespace.


## Running on the Cambridge HPC

# <span style='color: red;'> TODO </span>

* To run tubulaton on HPC (using python randomisation of parameters):
	* `sbatch --array=1-1000 hpc_run_tubulaton.sh` 

* When compiling tubulaton on HPC you must run `module load vtk/7.1.1` before running `cmake ..`

* The virtualenvs need to be installed as `.venv_generation` and `.venv_segmentation` in the right folders if you want to use HPC scripts out of the box
	* And you need to load the latest version of python (find with module avail python) before making the venv: `module load python/3.11.0-icl` 
	* You need to set the various environmental variables (PYTHONPATH, etc)

* Look for "TODO TESTING" in config files/scripts when switching between test and production mode
