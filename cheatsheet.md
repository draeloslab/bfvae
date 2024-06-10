# GREATLAKES CHEATSHEET
### get started
conda activate bfvae_env
module load cuda cudnn
cd src/bfvae0/

### delete job
scancel <job-id>

### see all jobs
sq <user-id>

### file transfer
scp '<local-file-path>' <uniqname>@greatlakes-xfer.arc-ts.umich.edu:'<hpc-path>'

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

## remove conda envs
conda env remove --name myenv
conda remove --name u19 --all