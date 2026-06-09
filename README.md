# Paper Title

## 📋 Experiment Logs

| Tab | Path | Description |
|-----|------|-------------|
| Tab 1 | `logs/recon` | Multi-view Reconstruction |
| Tab 2 | `logs/video_depth` | Video Depth Estimation |
| Tab 3 & 4 | `logs/wo` | Ablation Studies |

---

## 🛠️ Environment Setup

```bash
conda create -n GEM3R python=3.11 cmake=3.14.0
conda activate GEM3R
pip install -r requirements.txt
conda install 'llvm-openmp<16'
```

---

## 📦 Pretrained Checkpoint

Download `checkpoint.pth` from the [HuggingFace URL🤗](https://huggingface.co/lch01/StreamVGGT) and place it under `ckpt/`.

---

## 🚀 Run Inference

```bash
python demo_viser.py --seq_path examples/whiteroom
```

---

## 📊 Run Evaluation

```bash
cd src
bash eval/mv_recon/run.sh       # Multi-view reconstruction
bash eval/video_depth/run.sh    # Video depth estimation
```
