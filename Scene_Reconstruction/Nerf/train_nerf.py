"""
Author: dizhong zhu
Date: 08/04/2022
"""

import torch
import torch.multiprocessing as mp
from nerf_func import train_nerf
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed training for Nerf')
    parser.add_argument('--object', default='lego', type=str, help='object to train, antinous, benin, lego, matthew, rubik, trex')
    parser.add_argument('--epochs', default=1000, type=int, help='how many epochs to train')
    parser.add_argument('--save_epochs', default=1, type=int, help='Save in each epochs')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--model_sel', default='nerf', type=str, help='type of nerf model: mlp, fourier, nerf')
    parser.add_argument('--output', default='output', type=str, help='output directory')
    parser.add_argument('--fine', default=1, type=int, help='whether to do fine refinement')
    args = parser.parse_args()

    epochs = args.epochs
    learning_rate = args.learning_rate
    save_epochs = args.save_epochs
    model_sel = args.model_sel
    batch_size = args.batch_size
    object = args.object
    output = args.output
    bFine = True if args.fine > 0 else False

    world_size = torch.cuda.device_count()
    mp.spawn(train_nerf, nprocs=world_size, args=(world_size, epochs, learning_rate, save_epochs, batch_size, bFine, model_sel, object, output))

    # train_nerf(device=0, world_size=1, learning_rate=learning_rate, model_sel=model_sel, save_epochs=save_epochs, bFine=bFine, batch_size=batch_size, epochs=epochs, object=object, output_dir=output)
