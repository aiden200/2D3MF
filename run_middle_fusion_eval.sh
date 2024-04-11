#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32g
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=yzhao010_1246

export OPENBLAS_NUM_THREADS=4

module purge
eval "$(conda shell.bash hook)"
conda activate mamenv

python evaluate.py --config config/celebvhq_marlin_deepfake_ft.yaml --data_path /project/yzhao010_1246/deepfake_datasets/2D3MF_Dataset --num_workers 4 --batch_size 256 --marlin_ckpt pretrained/marlin_vit_small_ytf.encoder.pt --epochs 500 --dataset Forensics++

# get features
#python preprocess/extract_features.py --data_dir ./FaceForensics++ --video_backbone marlin_vit_small_ytf --audio_backbone eat

# eval
# python evaluate.py --config config/celebvhq_marlin_deepfake_ft.yaml --data_path . --num_workers 4     --batch_size 256 --marlin_ckpt pretrained/marlin_vit_small_ytf.encoder.pt --epochs 500 --dataset Forensics++


# trash
#python3 evaluate.py --config /home1/kevinhop/deepfake/2D3MF/config/celebv_hq/appearance/celebvhq_marlin_deepfake_ft.yaml --data_path ./yt_av_mixed --num_workers 4 --batch_size 16 --marlin_ckpt pretrained/marlin_vit_base_ytf.encoder.pt

#python3 evaluate.py --config config/celebvhq_marlin_deepfake_ft.yaml --dataset Forensics++ --data_path . --num_workers 4 --batch_size 16 --marlin_ckpt pretrained/marlin_vit_base_ytf.encoder.pt --epochs 1

# python3 evaluate.py --config config/celebvhq_marlin_deepfake_ft.yaml --data_path ./Forensics++ --num_workers 4 --batch_size 16 --marlin_ckpt pretrained/marlin_vit_base_ytf.encoder.pt --epochs 1

# python3 evaluate.py --config config/celebvhq_marlin_deepfake_ft.yaml --data_path ./yt_av_mixed --num_workers 4 --batch_size 16 --marlin_ckpt pretrained/marlin_vit_base_ytf.encoder.pt --epochs 1

# python evaluate.py --config config/celebvhq_marlin_deepfake_ft.yaml --data_path ../2D3MF_Datasets --num_workers 4 --batch_size 16 --marlin_ckpt pretrained/marlin_vit_small_ytf.encoder.pt --epochs 1