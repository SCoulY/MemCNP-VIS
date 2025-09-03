# MemCNP-VIS for YouTube-VIS and OVIS

This repo serves as the official code release of the MemCNP-VIS model in the paper:

## [A Memory-based Conditional Neural Process for Video Instance Segmentation](https://doi.org/10.1016/j.neucom.2025.131439)


Some results can be seen below:

<table style="display:flex;justify-content:center;border:0" rules=none frame=void >
<tr>
<td><img src="visualisations/vis_0.gif" width="160" height="90" />
</td>
<td><img src="visualisations/vis_1.gif" width="160" height="90">
</td>
<td><img src="visualisations/vis_2.gif" width="160" height="90">
</td>
<td><img src="visualisations/vis_3.gif"  width="160" height="90">
</td>

<tr>
<td><img src="visualisations/vis_4.gif" width="160" height="90" />
</td>
<td><img src="visualisations/vis_5.gif" width="160" height="90">
</td>
<td><img src="visualisations/vis_6.gif" width="160" height="90">
</td>
<td><img src="visualisations/vis_7.gif" width="160" height="90">
</td>
</tr>
<tr>

</tr>
</table>

For more details about the dataset, please refer to our paper.

## Model training and evaluation

### Installation

This repo is built based on [MaskTrackRCNN](https://github.com/youtubevos/MaskTrackRCNN), [SipMask](https://github.com/JialeCao001/SipMask) and a customized [FCOS](https://github.com/tianzhi0549/FCOS).

You can use following commands to create conda env with all dependencies. 

Modify ``cuda`` and corresponding ``torch`` version.
```
conda env create -f env_conda.yml
conda activate memcnp

pip install -r env_pip.txt
git clone https://github.com/qjy981010/cocoapi.git /tmp/cocoapi 
cd /tmp/cocoapi/PythonAPI 
python setup.py install 
cd -
conda install six matplotlib
```

Change the ``cuda`` path in ``compile.sh`` to your correct directory
```
bash compile.sh
```

### Data preparation
1. Download [YIVS](https://youtube-vos.org/dataset/vis/) and [OVIS](http://songbai.site/ovis/).
2. Symlink the train/validation dataset to `data/OVIS/` folder. Put COCO-style json annotations under `data/annotations`.

```
mmdetection
├── ...
├── mmdet
├── tools
├── configs
├── data
│   ├── OVIS
│   │   ├── train_images
│   │   ├── valid_images
│   │   ├── annotations
│   │   │   ├── annotations_train.json
│   │   │   ├── annotations_valid.json
│   │   │   ├── annotations_test.json
├── ├── YVIS
│   │   ├── train_images
│   │   ├── valid_images
│   │   ├── annotations
│   │   │   ├── instances_train.json
│   │   │   ├── instances_valid.json
│   │   │   ├── instances_test.json
```

### Training
Our default model is based on ViT-b_SipMask. The model is pretrained on ImageNet using MAE. [model link](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth).

Run the command below to train the model.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py configs/vitmae_cnp_contrast_ms_1x.py --work_dir 
./workdir/vit-b_memcnp_yvis --gpus 4
```
To change training configurations such as learning rate, model parameters, and dataset, please refer to `vitmae_cnp_contrast_ms_1x.py`.

### Evaluation
Our pretrained model is available for download at [Google Drive](https://drive.google.com/file/d/1fUgaEtMw2H-KGVQqrOUnHPvLiOWqYle8/view?usp=drive_link).
Run the following command to evaluate the model on YVIS.
```
CUDA_VISIBLE_DEVICES=0 python test_video.py configs/vitmae_cnp_contrast_ms_1x.py [MODEL_PATH] --out [OUTPUT_PATH.pkl] --eval segm
```

A json file containing the predicted result will be generated as `OUTPUT_PATH.pkl.json`. YVIS currently only allows evaluation on the codalab server. Please upload the generated result to [codalab server](https://codalab.lisn.upsaclay.fr/competitions/7680) to see actual performances.

For visualisation purpose, please use:
```
CUDA_VISIBLE_DEVICES=0 python test_video.py configs/vitmae_cnp_contrast_ms_1x.py [MODEL_PATH] --eval segm --show True --save_vis_path [VISUALISATION_PATH] --thresh 0.2
```
Note that when working in visualisation mode, the result file `OUTPUT_PATH.pkl.json` will not be properly generated even though you specify `--out [OUTPUT_PATH.pkl]`.

## License
This project is released under the [Apache 2.0 license](LICENSE), while the correlation ops is under [MIT license](mmdet/ops/correlation/LICENSE).

## Acknowledgement

This project is based on [MaskTrack-RCNN](https://github.com/youtubevos/MaskTrackRCNN), [SipMask](https://github.com/JialeCao001/SipMask), and [FCOS](https://github.com/tianzhi0549/FCOS). Thanks for their wonderful works.

## Citation
If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝 :

```
@article{YUANMemCNP-VIS
title = {A memory-based conditional neural process for video instance segmentation},
journal = {Neurocomputing},
volume = {655},
pages = {131439},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2025.131439},
url = {https://www.sciencedirect.com/science/article/pii/S0925231225021113},
author = {Kunhao Yuan and Gerald Schaefer and Yu-Kun Lai and Xiyao Liu and Lin Guan and Hui Fang}
}
```
