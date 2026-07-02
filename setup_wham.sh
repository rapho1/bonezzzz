#!/bin/bash
# Sets up WHAM (SMPL pose estimation) inside WSL2 Ubuntu. Automates everything
# EXCEPT the SMPL body models, which are license-gated and must be downloaded
# with YOUR OWN account (see fetch_smpl.sh, run after this script).
#
# Run this INSIDE WSL Ubuntu:
#   bash setup_wham.sh
#
# Requirements: WSL2 Ubuntu, an NVIDIA GPU with 8GB+ VRAM and drivers that
# expose CUDA to WSL (works out of the box on recent Windows/driver versions —
# check with `nvidia-smi` inside WSL before running this).
#
# Idempotent: safe to re-run if it fails partway through.
set -e

WHAM_DIR="/root/WHAM"
CONDA="/root/miniconda3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== 1/8: checking GPU =="
if ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi failed. WHAM needs GPU access from WSL."
    echo "Check: Windows NVIDIA driver is up to date, and you're on WSL2 (not WSL1)."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== 2/8: apt dependencies =="
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq git wget curl unzip build-essential ffmpeg

echo "== 3/8: Miniconda =="
if [ ! -d "$CONDA" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA"
    rm /tmp/miniconda.sh
else
    echo "already installed"
fi
source "$CONDA/etc/profile.d/conda.sh"

echo "== 4/8: conda env 'wham' (python 3.9) =="
if ! conda env list | grep -q "^wham "; then
    # conda-forge avoids the default-channel ToS prompt some conda versions require
    conda create -y -n wham -c conda-forge --override-channels python=3.9
else
    echo "already exists"
fi
conda activate wham
conda install -y -c conda-forge --override-channels pip setuptools wheel >/dev/null

echo "== 5/8: PyTorch 1.11 + cu113 =="
if ! python -c "import torch" 2>/dev/null; then
    pip install -q torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113
fi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in torch'; print('torch', torch.__version__, 'cuda ok')"

echo "== 6/8: clone WHAM + Python deps =="
if [ ! -d "$WHAM_DIR" ]; then
    git clone --recursive --depth 1 https://github.com/yohanshin/WHAM.git "$WHAM_DIR"
fi
cd "$WHAM_DIR"
if ! python -c "import mmcv" 2>/dev/null; then
    # numpy/setuptools/cython must land BEFORE requirements.txt so chumpy and
    # mmcv==1.3.9 (no prebuilt wheel — compiles from source) build correctly.
    pip install -q numpy==1.22.3 "setuptools==59.5.0" wheel cython
    pip install -r requirements.txt
fi
if ! python -c "import mmpose" 2>/dev/null; then
    pip install -e third-party/ViTPose
fi

echo "== 7/8: checkpoints (public, no login needed) =="
mkdir -p checkpoints dataset/body_models/smpl
declare -A CKPTS=(
    [checkpoints/wham_vit_w_3dpw.pth.tar]=1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB
    [checkpoints/wham_vit_bedlam_w_3dpw.pth.tar]=19qkI-a6xuwob9_RFNSPWf1yWErwVVlks
    [checkpoints/yolov8x.pt]=1zJ0KP23tXD42D47cw1Gs7zE2BA_V_ERo
    [checkpoints/vitpose-h-multi-coco.pth]=1xyF7F3I7lWtdq82xmEPVQ5zl4HaasBso
)
for f in "${!CKPTS[@]}"; do
    if [ ! -f "$f" ]; then
        gdown "${CKPTS[$f]}" -O "$f"
    else
        echo "  $f already present"
    fi
done
if [ ! -f dataset/body_models/smpl_mean_params.npz ]; then
    gdown 1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ -O dataset/body_models.tar.gz
    tar -xf dataset/body_models.tar.gz -C dataset/
    rm dataset/body_models.tar.gz
fi

echo "== 8/8: copy Bonezzzz integration scripts into the WHAM checkout =="
cp "$SCRIPT_DIR/wham_to_bvh.py" "$WHAM_DIR/wham_to_bvh.py"
cp "$SCRIPT_DIR/fetch_smpl.sh" "$WHAM_DIR/fetch_smpl.sh"
chmod +x "$WHAM_DIR/fetch_smpl.sh"

echo ""
echo "============================================================"
echo " Automated setup complete."
echo ""
echo " ONE MANUAL STEP LEFT: download the SMPL body models (license-"
echo " gated — you need your own account). Run:"
echo ""
echo "     cd $WHAM_DIR && bash fetch_smpl.sh"
echo ""
echo " Register first if you haven't: https://smpl.is.tue.mpg.de and"
echo " https://smplify.is.tue.mpg.de"
echo "============================================================"
