#!/bin/bash

n=laplace
for loc in -3 -2 -1 0 1 2 3
do
    python ./exp.py -c True -cn 20 -nt 1 -wp 'results/cutoff_20_loc_'$loc'_scale_1e-06_'${n}'_noise.txt' -np 1e-06 2 $loc
done