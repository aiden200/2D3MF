# 2D3MF: Deepfake Detection using Multi Modal Middle Fusion
<div align="center">
    <img src="assets/architecture.png" width="500" height="500">
</div>

<!--<div>-->
<!--    <img src="assets/teaser.svg">-->
<!--    <p></p>-->
<!--</div>-->

<!--<div align="center">-->
<!--    <a href="https://github.com/ControlNet/MARLIN/network/members">-->
<!--        <img src="https://img.shields.io/github/forks/ControlNet/MARLIN?style=flat-square">-->
<!--    </a>-->
<!--    <a href="https://github.com/ControlNet/MARLIN/stargazers">-->
<!--        <img src="https://img.shields.io/github/stars/ControlNet/MARLIN?style=flat-square">-->
<!--    </a>-->
<!--    <a href="https://github.com/ControlNet/MARLIN/issues">-->
<!--        <img src="https://img.shields.io/github/issues/ControlNet/MARLIN?style=flat-square">-->
<!--    </a>-->
<!--    <a href="https://github.com/ControlNet/MARLIN/blob/master/LICENSE">-->
<!--        <img src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-97ca00?style=flat-square">-->
<!--    </a>-->
<!--    <a href="https://arxiv.org/abs/2211.06627">-->
<!--        <img src="https://img.shields.io/badge/arXiv-2211.06627-b31b1b.svg?style=flat-square">-->
<!--    </a>-->
<!--</div>-->

<!--<div align="center">    -->
<!--    <a href="https://pypi.org/project/marlin-pytorch/">-->
<!--        <img src="https://img.shields.io/pypi/v/marlin-pytorch?style=flat-square">-->
<!--    </a>-->
<!--    <a href="https://pypi.org/project/marlin-pytorch/">-->
<!--        <img src="https://img.shields.io/pypi/dm/marlin-pytorch?style=flat-square">-->
<!--    </a>-->
<!--    <a href="https://www.python.org/"><img src="https://img.shields.io/pypi/pyversions/marlin-pytorch?style=flat-square"></a>-->
<!--    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%3E%3D1.8.0-EE4C2C?style=flat-square&logo=pytorch"></a>-->
<!--</div>-->

<!--<div align="center">-->
<!--    <a href="https://github.com/ControlNet/MARLIN/actions"><img src="https://img.shields.io/github/actions/workflow/status/ControlNet/MARLIN/unittest.yaml?branch=dev&label=unittest&style=flat-square"></a>-->
<!--    <a href="https://github.com/ControlNet/MARLIN/actions"><img src="https://img.shields.io/github/actions/workflow/status/ControlNet/MARLIN/release.yaml?branch=master&label=release&style=flat-square"></a>-->
<!--    <a href="https://coveralls.io/github/ControlNet/MARLIN"><img src="https://img.shields.io/coverallsCoverage/github/ControlNet/MARLIN?style=flat-square"></a>-->
<!--</div>-->

This repo is the implementation for the paper
[2D3MF: Deepfake Detection using Multi Modal Middle Fusion](https://).

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
├── src                   # Source code for marlin-pytorch
├── tests                 # Unittest
├── requirements.lib.txt
├── setup.py
├── init.py
├── version.txt

# below is for the paper implementation
├── configs              # Configs for experiments settings
├── model                # 2D3MF & Marlin models
├── preprocess           # Preprocessing scripts
├── dataset              # Dataloaders
├── utils                # Utility functions
├── train.py             # Training script
├── evaluate.py          # Evaluation script
├── requirements.txt

```

## Installing and running our model


## Paper Implementation

### Feature Extraction - MARLIN

Requirements:

- Python >= 3.7, < 3.12
- PyTorch ~= 1.11
- Torchvision ~= 0.12
- ffmpeg

Install MARLIN (our feature extractor) from PyPI:

```bash
pip install marlin-pytorch
```

For more details, see [MARLIN_MODEL_ZOO.md](MARLIN_MODEL_ZOO.md).


### Installation

Install PyTorch from the [official website](https://pytorch.org/get-started/locally/)

Clone the repo and install the requirements:

```bash
git clone https://github.com/aiden200/2D3MF
cd 2D3MF
pip install -r requirements.txt
```

### Training

#### 1. Download the Faceforensics ++ dataset

#### 2. Preprocess the dataset

Crop the face region from the raw video and split the train val and test sets.

```bash
python3 preprocess/preprocess_clips.py --data_dir . --yt /path/to/yt_data
```

Create split
```bash
python create_split.py --data_dir /path/to/data
```

#### 3. Extract MARLIN features (Optional, if linear probing)

Extract MARLIN features from the cropped video and saved to `<backbone>` directory in `CelebV-HQ` directory.

```bash
python preprocess/extract_features.py --data_dir /path/to/data --backbone marlin_vit_base_ytf

ex:
python preprocess/extract_features.py --data_dir yt_av_mixed --backbone marlin_vit_base_ytf
```



#### 4. Train and evaluate

Train and evaluate the 2D3MF model..

Please use the configs in `config/*.yaml` as the config file.

```bash
python evaluate.py \
    --config /path/to/config \
    --data_path /path/to/CelebV-HQ
    --num_workers 4
    --batch_size 16


python evaluate.py \
    --config config/celebv_hq/appearance/celebvhq_marlin_deepfake_ft.yaml \
    --data_path new_yt_sequences \
    --num_workers 4 \
    --batch_size 8 \
    --marlin_ckpt pretrained/marlin_vit_base_ytf.encoder.pt

python evaluate.py     --config config/celebv_hq/appearance/celebvhq_marlin_deepfake_ft.yaml     --data_path yt_av_mixed     --num_workers 4     --batch_size 256     --marlin_ckpt pretrained/marlin_vit_base_ytf.encoder.pt --epochs 500


--skip_train --resume ckpt/celebvhq_marlin_deepfake_ft/celebvhq_marlin_deepfake_ft-epoch=121-val_auc=0.587.ckpt

```

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
