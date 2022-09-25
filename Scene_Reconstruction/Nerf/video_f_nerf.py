"""
Author: dizhong zhu
Date: 24/09/2022
"""

from nerf_func import evaluate_nerf
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed training for Nerf')
    parser.add_argument('--checkpoint', type=str, help='The check point path')
    parser.add_argument('--write_video', type=str, help='The path to save the video')
    # parser.add_argument('--object', default='lego', type=str, help='object to train, antinous, benin, lego, matthew, rubik, trex')
    # parser.add_argument('--model_sel', default='nerf', type=str, help='model for evaluation. mlp, fourier, nerf')
    # parser.add_argument('--save_folder', default='output_2', type=str, help='path to model')

    args = parser.parse_args()

    # object = args.object
    # model_sel = args.model_sel
    # save_folder = args.save_folder

    checkpoint_path = args.checkpoint
    save_path = args.write_video

    # path = f'{save_folder}/{model_sel}/{object}'
    evaluate_nerf(checkpoint_path, save_path)
