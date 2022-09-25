#!/bin/bash

export PYTHONPATH=:./:../:../../

echo "start training nerf"

num_epochs=20000
num_save=5000
lr=0.001
declare -a objectArray=("lego" "benin" "antinous" "matthew" "rubik" "trex")

for object in "${objectArray[@]}"; do
  echo "=============================================>   testing $object  <================================================================"
  #  python3 nueral_radiance/train_nerf_entry.py --epochs $num_epochs --save_epochs $num_save --batch_size=10 --learning_rate 1e-3 --model_sel mlp --object $object
  #  python3 nueral_radiance/evaluate_nerf.py --model_sel mlp --object $object
  #
  #  python3 nueral_radiance/train_nerf_entry.py --epochs $num_epochs --save_epochs $num_save --batch_size=10 --learning_rate 1e-3 --model_sel fourier --object $object
  #  python3 nueral_radiance/evaluate_nerf.py --model_sel fourier --object $object

  python3 train_nerf.py --epochs $num_epochs --save_epochs $num_save --batch_size=5 --learning_rate 1e-3 --model_sel nerf --object $object --output output
#  python3 nueral_radiance/evaluate_nerf.py --model_sel nerf --object $object --save_folder output_2
done
