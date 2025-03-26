# 2D3MF: Deepfake Detection using Multi Modal Middle Fusion

<div align="center">
    <!-- <img src="assets/github_visualization.png" width="500" height="500"> -->
    <img src="assets/github_visualization.png">
</div>

<div align="center">
   <a href="https://github.com/aiden200/2D3MF/stargazers">
       <img src="https://img.shields.io/github/stars/aiden200/2D3MF?style=flat-square">
   </a>
   <a href="https://github.com/aiden200/2D3MF/issues">
       <img src="https://img.shields.io/github/issues/aiden200/2D3MF?style=flat-square">
   </a>
   <a href="https://github.com/aiden200/2D3MF/blob/master/LICENSE">
       <img src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-97ca00?style=flat-square">
   </a>
    <a href="https://pytorch.org/">
      <img src="https://img.shields.io/badge/PyTorch-%3E%3D1.8.0-EE4C2C?style=flat-square&logo=pytorch">
    </a>

   <!-- <a href="https://arxiv.org/abs/2211.06627">
       <img src="https://img.shields.io/badge/arXiv-2211.06627-b31b1b.svg?style=flat-square">
   </a> -->
</div>

> [!CAUTION]
> This repo is under development. No hyper parameter tuning is presented yet here; hence, the current architecture is not optimal for deepfake detection.

<!-- <div align="center">    -->
<!--    <a href="https://pypi.org/project/marlin-pytorch/">-->
<!--        <img src="https://img.shields.io/pypi/v/marlin-pytorch?style=flat-square">-->
<!--    </a>-->
<!--    <a href="https://pypi.org/project/marlin-pytorch/">-->
<!--        <img src="https://img.shields.io/pypi/dm/marlin-pytorch?style=flat-square">-->
<!--    </a>-->
<!--    <a href="https://www.python.org/"><img src="https://img.shields.io/pypi/pyversions/marlin-pytorch?style=flat-square"></a>-->
<!-- </div> -->

<!--<div align="center">-->
<!--    <a href="https://github.com/aiden200/2D3MF/actions"><img src="https://img.shields.io/github/actions/workflow/status/ControlNet/MARLIN/unittest.yaml?branch=dev&label=unittest&style=flat-square"></a>-->
<!--    <a href="https://github.com/aiden200/2D3MF/actions"><img src="https://img.shields.io/github/actions/workflow/status/ControlNet/MARLIN/release.yaml?branch=master&label=release&style=flat-square"></a>-->
<!--    <a href="https://coveralls.io/github/ControlNet/MARLIN"><img src="https://img.shields.io/coverallsCoverage/github/ControlNet/MARLIN?style=flat-square"></a>-->
<!--</div>-->

This repo is the implementation for the paper
[2D3MF: Deepfake Detection using Multi Modal Middle Fusion](https://drive.google.com/file/d/10gjCXD-Bkpe5J_U6OYoDl57_HDTo6K4Q/view?usp=sharing).

## Repository Structure

```
.
├── assets                # Images for README.md
├── LICENSE
├── README.md
├── MODEL_ZOO.md
├── CITATION.cff
├── .gitignore
├── .github

# below is for the PyPI package marlin-pytorch
├── src                   # Source code for marlin-pytorch and audio feature extractors
├── tests                 # Unittest
├── requirements.lib.txt
├── setup.py
├── init.py
├── version.txt

# below is for the paper implementation
├── configs              # Configs for experiments settings
├── TD3MF                # 2D3MF model code
├── preprocess           # Preprocessing scripts
├── dataset              # Dataloaders
├── utils                # Utility functions
├── train.py             # Training script
├── evaluate.py          # Evaluation script
├── requirements.txt

```

## Installing and running our model
### Feature Extraction - 2D3MF

Install 2D3MF from pypi

```bash
pip install 2D3MF
```

Sample code snippet for feature extraction

``` python
from TD3MF.classifier import TD3MF
ckpt = "ckpt/celebvhq_marlin_deepfake_ft/last-v72.ckpt"
model = TD3MF.load_from_checkpoint(ckpt)
features = model.feature_extraction("2D3MF_Datasets/test/SampleVideo_1280x720_1mb.mp4")
```

<!-- TODO -->
We have some pretrained marlin checkpoints and configurations [here]()

## Paper Implementation

Requirements:

- Python >= 3.7, < 3.12
- PyTorch ~= 1.11
- Torchvision ~= 0.12
- ffmpeg



## Installation

Install PyTorch from the [official website](https://pytorch.org/get-started/locally/)

Clone the repo and install the requirements:

```bash
git clone https://github.com/aiden200/2D3MF
cd 2D3MF
pip install -e .
```

## Training

### 1. Download Datasets

<details>
  <summary>Forensics++</summary>
We cannot offer the direct script in our repository due to their terms on using the dataset. Please follow the instructions on the [Forensics++](https://github.com/ondyari/FaceForensics?tab=readme-ov-file) page to obtain the download script.

#### Storage

```bash
- FaceForensics++
    - The original downladed source videos from youtube: 38.5GB
    - All h264 compressed videos with compression rate factor
        - raw/0: ~500GB
        - 23: ~10GB (Which we use)
```

#### Downloading the data

Please download the [Forensics++](https://github.com/ondyari/FaceForensics?tab=readme-ov-file) dataset. We used the all light compressed original & altered videos of three manipulation methods. It's the script in the Forensics++ repository that ends with: `<output path> -d all -c c23 -t videos`

The script offers two servers which can be selected by add `--server <EU or CA>`. If the `EU` server is not working for you, you can also try `EU2` which has been reported to work in some of those instances.

#### Audio download

Once the first two steps are executed, you should have a structure of

```bash
-- Parent_dir
|-- manipulated_sequences
|-- original_sequences
```

Since the Forensics++ dataset doesn't provide audio data, we need to extract the data ourselves. Please run the script in the Forensics++ repository that ends with: `<Parent_dir from last step> -d original_youtube_videos_info`

Now you should have a directory with the following structure:

```bash
-- Parent_dir
|-- manipulated_sequences
|-- original_sequences
|-- downloaded_videos_info
```

Please run the script from our repository:
`python3 preprocess/faceforensics_scripts/extract_audio.py --dir [Parent_dir]`

After this, you should have a directory with the following structure:

```bash
-- Parent_dir
|-- manipulated_sequences
|-- original_sequences
|-- downloaded_videos_info
|-- audio_clips
```

#### References

- Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, Matthias Nießner. "FaceForensics++: Learning to Detect Manipulated Facial Images." In _International Conference on Computer Vision (ICCV)_, 2019.

</details>

<details>
  <summary>DFDC</summary>
  Kaggle provides a nice and easy way to download the [DFDC dataset](https://www.kaggle.com/c/deepfake-detection-challenge/data)
</details>

<details>                    
  <summary>DeepFakeTIMIT</summary>
  We recommend downloading the data from the [DeepfakeTIMIT Zenodo Record](https://zenodo.org/records/4068245)
</details>

<details>                    
  <summary>FakeAVCeleb</summary>
  We recommend requesting access to FakeAVCeleb via their [repo README](https://github.com/DASH-Lab/FakeAVCeleb)
</details>

<details>
  <summary>RAVDESS</summary>
  We recommend downloading the data from the [RAVDESS Zenodo Record](https://zenodo.org/records/1188976)
</details>

### 2. Preprocess the dataset

We recommend using the following unified dataset structure

```
2D3MF_Dataset/
├── DeepfakeTIMIT
│   ├── audio/*.wav
│   └── video/*.mp4
├── DFDC
│   ├── audio/*.wav
│   └── video/*.mp4
├── FakeAVCeleb
│   ├── audio/*.wav
│   └── video/*.mp4
├── Forensics++
│   ├── audio/*.wav
│   └── video/*.mp4
├── RAVDESS
    ├── audio/*.wav
    └── video/*.mp4
```

Crop the face region from the raw video.
Run:

```bash
python3 preprocess/preprocess_clips.py --data_dir [Dataset_Dir]
```

### 3. Extract features from pretrained models


<details>
  <summary>EfficientFace</summary>

Download the pre-trained EfficientFace from [here](https://github.com/zengqunzhao/EfficientFace) under 'Pre-trained models'. In our experiments, we use the model pre-trained on AffectNet7, i.e., EfficientFace_Trained_on_AffectNet7.pth.tar. Please place it under the `pretrained` directory
</details>


Run:

```bash
python preprocess/extract_features.py --data_dir /path/to/data --video_backbone [VIDEO_BACKBONE] --audio_backbone [AUDIO_BACKBONE]
```

[VIDEO_BACKBONE] can be replaced with one of the following: 
- marlin_vit_small_ytf
- marlin_vit_base_ytf
- marlin_vit_large_ytf
- efficientface

[AUDIO_BACKBONE] can be replaced with one of the following:
- MFCC
- xvectors
- resnet
- emotion2vec
- eat


Optionally add the `--Forensics` flag in the end if Forensics++ is the dataset being processed.

From our paper, we found that `eat` works the best as the audio backbone. 

Split the train val and test sets.
Run:

```bash
python preprocess/gen_split.py --data_dir /path/to/data --test 0.1 --val 0.1 --feat_type [AUDIO_BACKBONE]
```

<!-- TODO add the rest -->
Note that the pre-trained `video_backbone` and `audio_backbone` can be downloaded from [MODEL_ZOO.md](MODEL_ZOO.md)

### 4. Train and evaluate

Train and evaluate the 2D3MF model..

Please use the configs in `config/*.yaml` as the config file.

```bash
python evaluate.py \
    --config /path/to/config \
    --data_path /path/to/CelebV-HQ
    --num_workers 4
    --batch_size 16


python evaluate.py \
    --config /path/to/config  \
    --data_path /path/to/dataset \
    --num_workers 4 \
    --batch_size 8 \
    --marlin_ckpt pretrained/marlin_vit_base_ytf.encoder.pt \
    --epochs 300


python evaluate.py --config config/celebvhq_marlin_deepfake_ft.yaml --data_path 2D3MF_Datasets --num_workers 4     --batch_size 1 --marlin_ckpt pretrained/marlin_vit_small_ytf.encoder.pt --epochs 300
```

Optionally, add 
```bash
--skip_train --resume /path/to/checkpoint
```

To skip the training.

### 5. Configuration File

Set a configuration file based on your hyperparameters and backbones. You can find a example config file under `config/`

Explanation:
- `training_datasets` - list, can contain one or more datasets within `"DeepfakeTIMIT"`, `"RAVDESS"`, `"Forensics++"`, `"DFDC"`, `"FakeAVCeleb"`
- `eval_datasets`- list, can contain one or more datasets within `"DeepfakeTIMIT"`, `"RAVDESS"`, `"Forensics++"`, `"DFDC"`, `"FakeAVCeleb"`
- `learning_rate` - int, ex: `1.00e-3`
- `num_heads` - int, Number of attention heads
- `fusion` - str, Choice of fusion type: `"mf"` for middle fusion and `"lf"` for late fusion.
- `audio_positional_encoding` - bool, add audio positional encoding
- `hidden_layers` - int, hidden layers
- `lp_only` - bool, setting this to be true will perform inference from the video features only
- `audio_backbone`- str, select one of the following options: `"MFCC"`, `"eat"`, `"xvectors"`, `"resnet"`, `"emotion2vec"`
- `middle_fusion_type`- str, select one of the following options: `"default"`, `"audio_refuse"`, `"video_refuse"`, `"self_attention"`, `"self_cross_attention"`
- `modality_dropout` - float, modality dropout rate
- `video_backbone` - str, select one of the following options: `"efficientface"`, `"marlin"`

### 6. Performing Grid Search

- config/grid_search_config.py
- --grid_search


### 7. Monitoring Performance:

Run

```bash
tensorboard --logdir=lightning_logs/
```

Should be hosted on http://localhost:6006/

</details>

## License

This project is under the CC BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## References

Please cite our work!

```bibtex

```

## Acknowledgements

Some code about model is based on [ControlNet/MARLIN](https://github.com/ControlNet/MARLIN). The code related to middle fusion
is from [Self-attention fusion for audiovisual emotion recognition with incomplete data](https://arxiv.org/abs/2201.11095).

Our Audio Feature Extraction Models:

- [EAT: Self-Supervised Pre-Training with Efficient Audio Transformer](https://github.com/cwx-worst-one/EAT)
- [X-Vectors](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf)
- [Resnet](https://arxiv.org/pdf/1512.03385.pdf)
- [Emotion2vec](https://github.com/ddlBoJack/emotion2vec)

Our Video Feature Extraction Models:

- [MARLIN](https://github.com/ControlNet/MARLIN)
- [EfficientFace](https://github.com/zengqunzhao/EfficientFace)
