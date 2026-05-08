# GEM3R: Geometry-Guided Memory for Long-Horizon Streaming 3D Reconstruction
## Experiments logs
```
Tab1: logs/recon
Tab2: logs/video_depth
Tab3,4: logs/wo
```

## Environment
``` bash
conda create -n GEM3R python=3.11 cmake=3.14.0
conda activate GEM3R 
pip install -r requirements.txt
conda install 'llvm-openmp<16'
```

## CKPT
Please download pretrained model from https://huggingface.co/lch01/StreamVGGT.

## Run Inference
``` bash
python demo_viser.py --seq_path examples/whiteroom
```

## Run Eval

``` bash
cd src
bash eval/mv_recon/run.sh # mv recon
bash eval/video_depth/run.sh # video depth
```