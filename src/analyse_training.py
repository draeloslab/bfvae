import re
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument( '--model', type=str, choices=['bfvae0', 'bfvae1', 'bfvae2'], required=True,
      help='model used' )
    parser.add_argument( '--train_filename', type=str, required=True,
      help='filename of training loss' )
    parser.add_argument( '--smooth', default=1, type=float, 
      help='smoothen the plot' )

    return parser

def main(args):
    # os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-sachinks'

    # Define the file path
    file_path = f'{args.model}/records/{args.train_filename}.txt'

    output_dir = f'{args.model}/outputs/{args.train_filename}/figs'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize lists to hold the values
    iter_list = []
    vae_loss_list = []
    dis_loss_list = []
    recon_list = []
    kl_list = []
    tc_list = []
    pv_reg_list = []
    pv_values = []
    metric1_values = []
    metric2_values = []

    # Regular expression patterns to extract the losses
    iter_pattern = re.compile(r'\[iter (\d+)')
    vae_loss_pattern = re.compile(r'vae_loss:\s([\d\.]+)')
    dis_loss_pattern = re.compile(r'dis_loss:\s([\d\.]+)')
    recon_pattern = re.compile(r'recon:\s([\d\.]+)')
    kl_pattern = re.compile(r'kl:\s([\d\.]+)')
    tc_pattern = re.compile(r'tc:\s([-]*[\d\.]+)')
    pv_reg_pattern = re.compile(r'pv_reg:\s([\d\.]+)')
    pv_pattern = re.compile(r'pv = \[([0-9\.\s]+)\]')
    metric1_pattern = re.compile(r'metric1\s*=\s*([\d\.]+)')
    metric2_pattern = re.compile(r'metric2\s*=\s*([\d\.]+)')

    # Read the file and extract the values
    with open(file_path, 'r') as file:
        for line in file:
            iter_match = iter_pattern.search(line)
            if iter_match:
                iter_list.append(int(iter_match.group(1)))
                
            vae_loss_match = vae_loss_pattern.search(line)
            if vae_loss_match:
                vae_loss_list.append(float(vae_loss_match.group(1)))

            dis_loss_match = dis_loss_pattern.search(line)
            if dis_loss_match:
                dis_loss_list.append(float(dis_loss_match.group(1)))

            recon_match = recon_pattern.search(line)
            if recon_match:
                recon_list.append(float(recon_match.group(1)))

            kl_match = kl_pattern.search(line)
            if kl_match:
                kl_list.append(float(kl_match.group(1)))

            tc_match = tc_pattern.search(line)
            if tc_match:
                tc_list.append(float(tc_match.group(1)))

            pv_reg_match = pv_reg_pattern.search(line)
            if pv_reg_match:
                pv_reg_list.append(float(pv_reg_match.group(1)))

            pv_match = pv_pattern.search(line)
            if pv_match:
                pv_str = pv_match.group(1)
                pv_list = [float(val) for val in pv_str.split()]
                pv_values.append(pv_list)
                
            metric1_match = metric1_pattern.search(line)
            if metric1_match:
                metric1_values.append(float(metric1_match.group(1)))

            metric2_match = metric2_pattern.search(line)
            if metric2_match:
                metric2_values.append(float(metric2_match.group(1)))


    loss_iter_intv = iter_list[0]
    loss_iter_max = loss_iter_intv*len(iter_list)
    smooth_scale = args.smooth

    for is_log in [True, False]:
        plt.figure(figsize=(12, 8))

        plt.plot(gaussian_filter1d(vae_loss_list, smooth_scale), label='Total Loss', color='blue')
        plt.plot(gaussian_filter1d(recon_list, smooth_scale), label='Reconstruction Loss', color='green')
        plt.plot(gaussian_filter1d(kl_list, smooth_scale), label='KL Divergence', color='purple')
        plt.plot(6.4*gaussian_filter1d(tc_list, smooth_scale), label='TC Loss', color='orange')
        plt.plot(gaussian_filter1d(pv_reg_list, smooth_scale), label='PV Regularization', color='brown')
        # plt.plot(gaussian_filter1d(dis_loss_list, smooth_scale), label='Discriminator Loss', color='red')
        plt.xlabel('Iteration')

        if is_log:
            plt.yscale('log')
        log_suffix = ' (logscale)' if is_log else ''
        plt.ylabel(f'Loss Value {log_suffix}')
        plt.title(f'Losses over Iterations {log_suffix}')
        plt.legend()
        plt.grid(True, which="both", ls="--")

        plt.xticks(ticks=np.linspace(0, loss_iter_max//loss_iter_intv, 5), labels=[str(x * loss_iter_intv) for x in np.linspace(0, loss_iter_max//loss_iter_intv, 5,dtype=int)])
        plt.savefig(os.path.join(output_dir, f'loss{log_suffix}.png'))


    plt.figure(figsize=(12, 8))
    for pv_i in range(len(pv_values[0])):
        smoothed_sig = gaussian_filter1d(np.array(pv_values)[:,pv_i], smooth_scale)
        plt.plot(smoothed_sig, label=f'pv[{pv_i+1}]')
        # plt.ylim([0,1.2])

    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.title('PV losses over Iterations')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.xticks(ticks=np.linspace(0, loss_iter_max//loss_iter_intv, 5), labels=[str(x * loss_iter_intv) for x in np.linspace(0, loss_iter_max//loss_iter_intv, 5,dtype=int)])
    plt.savefig(os.path.join(output_dir, f'pv.png'))
    print("Figures saved in:", output_dir)

if __name__ == '__main__':    
    parser = create_parser()
    args = parser.parse_args()    
    main(args)