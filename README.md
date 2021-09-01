# MultiModal-InfoMax

This repository contains the official implementation code of the paper [Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis](), accepted to ``EMNLP 2021``.

`:fire:` code and paper are coming soon!!

## Introduction
Multimodal-informax (MMIM) synthesizes fusion results from multi-modality input through a two-level mutual information (MI) maximization. We use BA (Barber-Agakov) lower bound and contrastive predictive coding as the target function to be maximized. To facilitate the computation, we design an entropy estimation module with associated history data memory to facilitate the computation of BA lower bound and the training process.

![Alt text](img/ModelFigSingle.png?raw=true "Model")

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
