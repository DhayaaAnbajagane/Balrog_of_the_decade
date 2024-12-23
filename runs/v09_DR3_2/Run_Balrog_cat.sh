#!/bin/bash
#SBATCH --job-name balrog_concat
##SBATCH --partition=broadwl
#SBATCH --partition=chihway
#SBATCH --account=pi-chihway
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=10:00:00
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/dhayaa/Desktop/DECADE/Balrog_of_the_decade/runs/v09_DR3_2/log_MakeBalrog

if [ "$USER" == "dhayaa" ]
then
    cd /home/dhayaa/Desktop/DECADE/Balrog_of_the_decade/
    #module load python
    #conda activate shearDM
    #source /home/dhayaa/Desktop/DECADE/Balrog_of_the_decade/bash_profile.sh
fi

cd /home/dhayaa/Desktop/DECADE/Balrog_of_the_decade/

pwd

python -u ./runs/v09_DR3_2/Make_balrog_cat.py
