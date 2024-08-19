#!/bin/zsh

generate_path="/Users/karan/microsegmentation/generation/generate.py"

python $generate_path -ids=1-100    -od /Users/karan/microsegmentation/SyntheticData_EASY_DENSE -c easy -v > easy_1_100.log 2>&1 &
python $generate_path -ids=101-200  -od /Users/karan/microsegmentation/SyntheticData_EASY_DENSE -c easy -v > easy_101_200.log 2>&1 &
python $generate_path -ids=201-300  -od /Users/karan/microsegmentation/SyntheticData_EASY_DENSE -c easy -v > easy_201_300.log 2>&1 &
python $generate_path -ids=301-400  -od /Users/karan/microsegmentation/SyntheticData_EASY_DENSE -c easy -v > easy_301_400.log 2>&1 &
python $generate_path -ids=401-500  -od /Users/karan/microsegmentation/SyntheticData_EASY_DENSE -c easy -v > easy_401_500.log 2>&1 
echo "Finished Batch 1"

python $generate_path -ids=501-600  -od /Users/karan/microsegmentation/SyntheticData_EASY_DENSE -c easy -v > easy_501_600.log 2>&1 &
python $generate_path -ids=601-700  -od /Users/karan/microsegmentation/SyntheticData_EASY_DENSE -c easy -v > easy_601_700.log 2>&1 &
python $generate_path -ids=701-800  -od /Users/karan/microsegmentation/SyntheticData_EASY_DENSE -c easy -v > easy_701_800.log 2>&1 &
python $generate_path -ids=801-900  -od /Users/karan/microsegmentation/SyntheticData_EASY_DENSE -c easy -v > easy_801_900.log 2>&1 &
python $generate_path -ids=901-1000  -od /Users/karan/microsegmentation/SyntheticData_EASY_DENSE -c easy -v > easy_901_1000.log 2>&1 
echo "Finished Batch 2"

python $generate_path -ids=1-100 -od /Users/karan/microsegmentation/SyntheticData_TRIVIAL_DENSE -c trivial -v > trivial_1_100.log 2>&1 &
python $generate_path -ids=101-200 -od /Users/karan/microsegmentation/SyntheticData_TRIVIAL_DENSE -c trivial -v > trivial_101_200.log 2>&1 &
python $generate_path -ids=201-300 -od /Users/karan/microsegmentation/SyntheticData_TRIVIAL_DENSE -c trivial -v > trivial_201_300.log 2>&1 &
python $generate_path -ids=301-400 -od /Users/karan/microsegmentation/SyntheticData_TRIVIAL_DENSE -c trivial -v > trivial_301_400.log 2>&1 &
python $generate_path -ids=401-500 -od /Users/karan/microsegmentation/SyntheticData_TRIVIAL_DENSE -c trivial -v > trivial_401_500.log 2>&1 
echo "Finished Batch 3"

python $generate_path -ids=501-600 -od /Users/karan/microsegmentation/SyntheticData_TRIVIAL_DENSE -c trivial -v > trivial_501_600.log 2>&1 &
python $generate_path -ids=601-700 -od /Users/karan/microsegmentation/SyntheticData_TRIVIAL_DENSE -c trivial -v > trivial_601_700.log 2>&1 &
python $generate_path -ids=701-800 -od /Users/karan/microsegmentation/SyntheticData_TRIVIAL_DENSE -c trivial -v > trivial_701_800.log 2>&1 &
python $generate_path -ids=801-900 -od /Users/karan/microsegmentation/SyntheticData_TRIVIAL_DENSE -c trivial -v > trivial_801_900.log 2>&1 &
python $generate_path -ids=901-1000 -od /Users/karan/microsegmentation/SyntheticData_TRIVIAL_DENSE -c trivial -v > trivial_901_1000.log 2>&1 
echo "Finished Batch 4"
