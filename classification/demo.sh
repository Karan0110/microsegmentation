#!/bin/zsh

echo "Simulated Control"
python3 demo.py default /Users/karan/MTData/SimulatedData/Control/ 6

echo "Simulated Depoly"
python3 demo.py default /Users/karan/MTData/SimulatedData/Depoly/ 6

echo "Real-World Control"
python3 demo.py default /Users/karan/MTData/ExperimentalIndividualImages/Control/ 6

echo "Real-World Depoly"
python3 demo.py default /Users/karan/MTData/ExperimentalIndividualImages/Depolymerised 6
