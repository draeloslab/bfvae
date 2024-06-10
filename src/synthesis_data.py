import argparse
import os
import numpy as np

def create_parser():
    
    '''
    Create a parser for command-line arguments
    '''
    
    parser = argparse.ArgumentParser()

    parser.add_argument( '--rseed', default=0, type=int, 
      help='random seed (default=0)' )

    # dataset
    parser.add_argument( '--output_dir', type=str, required=True,
      help='directory path to save dataset' )
    parser.add_argument( '--dataset', choices=['latent2_0', 'latent2_1'], type=str, required=True,
      help='dataset name' )

    return parser

def main(args):
    SEED = args.rseed
    np.random.seed(SEED)
    
    if args.dataset == 'latent2_0':
        # create latent class and values
        n_c1 = 2
        n_c2 = 6

        class1 = np.arange(n_c1)
        class2 = np.arange(n_c2)

        mesh1, mesh2 = np.meshgrid(class1, class2)
        latent_classes = np.vstack([mesh1.ravel(), mesh2.ravel()]).T

        latent_values = latent_classes.astype(float)
        latent_values[:, 1] = 0.5+latent_values[:, 1]/10

        # create dataset
        num_samples = latent_values.shape[0]
        data = np.zeros((num_samples, 4))

        for i in range(num_samples):
            t0, t1 = latent_values[i,0], latent_values[i,1]
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
        latent_classes = np.vstack([mesh1.ravel(), mesh2.ravel()]).T

        latent_values = latent_classes.astype(float)
        latent_values = latent_values/n_c1

        # create dataset
        num_samples = latent_values.shape[0]
        data = np.zeros((num_samples, 4))

        for i in range(num_samples):
            t0, t1 = latent_values[i,0], latent_values[i,1]
            data[i,0] = 0.5*(t0+t1)
            data[i,1] = 0.5*(t0**5 + (1-t1)**2)
            data[i,2] = t0/(1+t1)
            data[i,3] = (t0+1)**t1 - 1

    else:
        raise NotImplementedError

    output_dir = f'{args.output_dir}/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f'{output_dir}/latent_classes.npy', latent_classes)
    np.save(f'{output_dir}/latent_values.npy', latent_values)
    np.save(f'{output_dir}/data.npy', data)


if __name__ == '__main__':    
    parser = create_parser()
    args = parser.parse_args()    
    main(args)
    