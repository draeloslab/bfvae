# bfvae

**This is the project page for Bayes-Factor-VAE: Hierarchical Bayesian Deep Auto-Encoder Models forFactor Disentanglement.
The work was accepted by ICCV 2019 Oral.**
[[Paper Link]]( NA ).
<br>

## Setup instructions
(Added by Sachin Salim)
FIRST: INSTALL CORRECT DRIVER for GPU, see https://stackoverflow.com/questions/30820513/what-is-the-correct-version-of-cuda-for-my-nvidia-driver/30820690

AFTER THIS FILE IS INSTALLED, if you have a GPU be sure to install `conda forge cudnn`

install: `conda env create -f reqs.yaml`

## Execute instructions
* Activate conda env: `conda activate bfvae_env`
* If in Greatlakes HPC, load cuda module: `module load cuda cudnn`
* Navigate to the model folder: `cd src/bfvae<id>/`
* Read `cmdlines.txt` for specific instructions on running various scripts
** Eg: `python main.py --dataset dsprites --dset_dir <dset_dir>`

To run custom dataset:
* Navigate to the src folder: `cd src/`
* Generate custom dataset by `python synthesize_data.py --dataset <dataset> --output_dir <save_dir>`
* Execute: `python main.py --dataset <dataset> --output_dir <save_dir>`


