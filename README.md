# bfvae

**This is the project page for Bayes-Factor-VAE: Hierarchical Bayesian Deep Auto-Encoder Models forFactor Disentanglement.
The work was accepted by ICCV 2019 Oral.**
[[Paper Link]]( NA ).
<br>

_Further instructions added by Sachin Salim_
## Setup instructions
### GPU drivers
If you don't have the GPU drivers installed:
* Install correct driver for the GPU, see [here](https://stackoverflow.com/questions/30820513/what-is-the-correct-version-of-cuda-for-my-nvidia-driver/30820690)
* After this file is installed, if you have a GPU be sure to install `conda forge cudnn`
### Dependencies
After installing the drivers:
* Install dependencies: `conda env create -f reqs.yaml`
* Activate conda env: `conda activate bfvae_env`
### Datasets
* Download/synthesize dataset by `python src/setup_dataset.py --dataset <dataset> --save_dir <save_dir>`
* Currently supported datasets to be automatically downloaded/synthesized are:
  1. dsprites (Downloaded by script from [here](https://github.com/google-deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz))
  2. latent2_0 (Synthesized)
  3. latent2_1 (Synthesized)

## Execute instructions
* If in Greatlakes HPC, load cuda module: `module load cuda cudnn`
* Navigate to the model folder: `cd src/bfvae<id>/`
* Read `cmdlines.txt` for specific instructions on running various scripts
* Eg: To train the model on dsprites, execute `python main.py --dataset dsprites --dset_dir <dset_dir>`

### Analyse training
* After the training, The training loss is saved in `records/`
* Navigate to `cd src/` and Execute `python analyse_training.py --model bfvae<id> --train_filename <train_filename>`
* Pass the respective file inside `records/` folder as the `train_filename`