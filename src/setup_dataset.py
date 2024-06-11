import argparse
import os
import requests
import numpy as np

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument( '--save_dir', type=str, required=True,
      help='directory path to save dataset' )
    parser.add_argument( '--dataset', choices=['dsprites', 'latent2_0', 'latent2_1'], type=str, required=True,
      help='dataset name' )

    return parser

def download_data(url, save_path):    
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print("Download completed successfully.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

def extract_npz(npz_file_path, output_dir):
    npz_data = np.load(npz_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in npz_data.files[1:]:
        npy_file_path = os.path.join(output_dir, f"{file_name}.npy")
        np.save(npy_file_path, npz_data[file_name])
        print(f"Saved {npy_file_path}")

def main(args):
    if args.dataset == 'dsprites':
        data_path = os.path.join(args.save_dir, "dsprites-dataset.npz")
        os.makedirs(args.save_dir, exist_ok=True)
        print("Downloading dsprites dataset...")
        download_data(url = "https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
            save_path=data_path)
        print("Extracting dataset...")
        extract_npz(data_path, os.path.join(args.save_dir, "dsprites-dataset"))
        os.remove(data_path)

    elif args.dataset == 'latent2_0':
        # create latent class and values
        n_c1 = 2
        n_c2 = 6

        class1 = np.arange(n_c1)
        class2 = np.arange(n_c2)

        mesh1, mesh2 = np.meshgrid(class1, class2)
        latents_classes = np.vstack([mesh1.ravel(), mesh2.ravel()]).T

        latents_values = latents_classes.astype(float)
        latents_values[:, 1] = 0.5+latents_values[:, 1]/10

        # create dataset
        num_samples = latents_values.shape[0]
        data = np.zeros((num_samples, 4))

        for i in range(num_samples):
            t0, t1 = latents_values[i,0], latents_values[i,1]
            data[i,0] = t0 * t1
            data[i,1] = t0 * (1.5-t1)
            data[i,2] = (1-t0) * t1
            data[i,3] = (1-t0) * (1.5-t1)

    elif args.dataset == 'latent2_1':
        # create latent class and values
        n_c1 = 10
        n_c2 = 10

        class1 = np.arange(n_c1)
        class2 = np.arange(n_c2)

        mesh1, mesh2 = np.meshgrid(class1, class2)
        latents_classes = np.vstack([mesh1.ravel(), mesh2.ravel()]).T

        latents_values = latents_classes.astype(float)
        latents_values = latents_values/n_c1

        # create dataset
        num_samples = latents_values.shape[0]
        data = np.zeros((num_samples, 4))

        for i in range(num_samples):
            t0, t1 = latents_values[i,0], latents_values[i,1]
            data[i,0] = 0.5*(t0+t1)
            data[i,1] = 0.5*(t0**5 + (1-t1)**2)
            data[i,2] = t0/(1+t1)
            data[i,3] = (t0+1)**t1 - 1

    else:
        raise NotImplementedError

    if args.dataset.startswith('latent2'):
        save_dir = f'{args.save_dir}/{args.dataset}'
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(f'{save_dir}/latents_classes.npy', latents_classes)
        np.save(f'{save_dir}/latents_values.npy', latents_values)
        np.save(f'{save_dir}/data.npy', data)
        print("Dataset synthesized successfully.")


if __name__ == '__main__':    
	parser = create_parser()
	args = parser.parse_args()    
	main(args)