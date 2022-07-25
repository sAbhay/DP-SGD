#!/usr/bin/env bash
source ../../../miniconda3/etc/profile.d/conda.sh
conda activate jax-privacy-dm

for i in 3 4 6 8 10 12 14 16
do
   echo "Running experiment with $i depth"
   python experiment.py --dpsgd=True --noise_multiplier=1.3 --l2_norm_clip=1.5 --epochs=20 --learning_rate=.25 --overparameterised=True --loss=cross-entropy --groups=0 --batch_size=256 --weight_standardisation=False --param_averaging=False --ema_coef=0.999 --ema_start_step=0 --polyak_start_step=0 --augmult=0 --random_flip=True --random_crop=True --norm_dir=norms --plot_dir=plots --dataset=mnist --model=cnn --augmult_batch_size=8192 --checkpoint=20 --depth=$i
done