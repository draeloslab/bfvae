# GREATLAKES CHEATSHEET
### get started
conda activate u19_env
module load cuda cudnn
cd src/bfvae0/

### delete job
scancel <job-id>

### see all jobs
sq <user-id>

### file transfer
scp '/home/sachinks/Code/data-core-u19/ROSMAP_data/ROSMAP_normalize_train.csv' sachinks@greatlakes-xfer.arc-ts.umich.edu:'/nfs/turbo/umms-adraelos/sachinks/u19/'

### load modules
module load cuda cudnn
module load python3.9-anaconda

conda env remove --name myenv
conda remove --name u19 --all

# LINUX COMMANDS
## check ram memory
free -h

## check gpu
nvidia-smi

## check cores
lscpu
nproc

# MISC COMMANDS
## convert jupyter to py
jupyter nbconvert --to script data_preprocessing.ipynb