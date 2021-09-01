# MultiModal-InfoMax

This repository contains the official implementation code of the paper [Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis](), accepted to ``EMNLP 2021``.

:fire:  If you would be interested in other multimodal works in our DeCLaRe Lab, welcome to visit the [clustered repository](https://github.com/declare-lab/multimodal-deep-learning)

## Introduction
Multimodal-informax (MMIM) synthesizes fusion results from multi-modality input through a two-level mutual information (MI) maximization. We use BA (Barber-Agakov) lower bound and contrastive predictive coding as the target function to be maximized. To facilitate the computation, we design an entropy estimation module with associated history data memory to facilitate the computation of BA lower bound and the training process.

![Alt text](img/ModelFigSingle.png?raw=true "Model")

## Usage
1. Download the CMU-MOSI and CMU-MOSEI dataset from [Google Drive]() or [Baidu Disk](). Please them under the folder `Multimodal-Infomax/datasets`

2. Set up the environment (need conda prerequisite)
```
conda env create -f environment.yml
conda activate MMIM
```

3. Start training
```
python main.py --dataset mosi --contrast
```

## Citation
Please cite our paper if you find our work useful for your research:
```bibtex
@article{han2021improving,
  title={Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis},
  author={Han, Wei and Chen, Hui and Poria, Soujanya},
  journal={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2021}
}
```

## Contact 
Should you have any question, feel free to contact me through [henryhan88888@gmail.com](henryhan88888@gmail.com)