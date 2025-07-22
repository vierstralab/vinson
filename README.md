# vinson
ML sequence to function models



srun  --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=8 --mem 256G --partition=gpuAll --pty bash


conda activate tf-2.13.0

python /home/jvierstra/proj/vinson/train_lit.py --nodes 1 --devices 4 --outdir /home/jvierstra/proj/vinson/models/23.2M_jitter_noise

