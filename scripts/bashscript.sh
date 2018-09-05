#!/bin/bash

#SBATCH -n 1
#SBATCH -p compute
#SBATCH -t 30:00:00
#SBATCH -J test:
###SBATCH --constraint=LargeMem
###SBATCH --mem=128000

echo "~~~~~~~~~~~~~~~n = 07~~~~~~~~~~~~~~~~~~~~~~~~" >> test.txt
export OMP_NUM_THREADS=1
python snnt.py
echo "~~~~~~~~~~~~~~~DONE~~~~~~~~~~~~~~~~~~~~~~~~" >> test.txt

