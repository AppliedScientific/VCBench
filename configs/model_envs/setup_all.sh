#!/usr/bin/env bash
# Consolidated model environment setup
# Creates 3 base environments grouped by compatible PyTorch versions,
# then installs model-specific packages into each.
#
# Environments:
#   vcbench-pt118  (Python 3.10, PyTorch+CUDA 11.8) -> Geneformer, UCE
#   vcbench-pt212  (Python 3.9,  PyTorch 2.1.2)     -> scGPT
#   vcbench-pt25   (Python 3.11, PyTorch <=2.5.1)   -> TranscriptFormer, State
#
# Usage:
#   bash configs/model_envs/setup_all.sh           # Setup all
#   bash configs/model_envs/setup_all.sh pt118     # Setup only pt118 env
#   bash configs/model_envs/setup_all.sh pt25      # Setup only pt25 env
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

TARGET="${1:-all}"

create_env_if_missing() {
    local name="$1" python_ver="$2"
    if conda info --envs | grep -q "$name"; then
        echo "Environment $name already exists."
    else
        conda create -n "$name" "python=$python_ver" -y
    fi
}

verify_cuda() {
    python -c "
import torch
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
}

# ── vcbench-pt118: Geneformer + UCE (Python 3.10, PyTorch CUDA 11.8) ──
setup_pt118() {
    echo ""
    echo "=== vcbench-pt118 (Geneformer + UCE) ==="
    create_env_if_missing vcbench-pt118 3.10

    eval "$(conda shell.bash hook)"
    conda activate vcbench-pt118
    conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
    pip install scanpy anndata pybiomart

    # Geneformer
    if ! command -v git-lfs &> /dev/null; then
        conda install git-lfs -c conda-forge -y
    fi
    git lfs install

    if [ ! -d "models/geneformer/Geneformer" ]; then
        mkdir -p models/geneformer
        cd models/geneformer
        git clone https://huggingface.co/ctheodoris/Geneformer
        cd Geneformer && pip install .
        cd "$PROJECT_DIR"
    else
        echo "Geneformer already downloaded."
    fi

    # UCE
    if [ ! -d "models/uce/UCE" ]; then
        mkdir -p models/uce
        cd models/uce
        git clone https://github.com/snap-stanford/UCE.git
        cd UCE && pip install -r requirements.txt
        cd "$PROJECT_DIR"
    else
        echo "UCE repo already cloned."
    fi

    if [ ! -f "models/uce/33l_8ep_1024t_1280.torch" ]; then
        echo "NOTE: Download UCE 33-layer weights from https://figshare.com/articles/dataset/24320806"
        echo "Place at: models/uce/33l_8ep_1024t_1280.torch"
    fi

    echo "Verifying vcbench-pt118..."
    verify_cuda
    python -c "from geneformer import TranscriptomeTokenizer; print('  Geneformer: OK')"
    echo "=== vcbench-pt118 ready ==="
}

# ── vcbench-pt212: scGPT (Python 3.9, PyTorch 2.1.2) ──
setup_pt212() {
    echo ""
    echo "=== vcbench-pt212 (scGPT) ==="
    create_env_if_missing vcbench-pt212 3.9

    eval "$(conda shell.bash hook)"
    conda activate vcbench-pt212
    conda install pytorch==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
    pip install "scgpt==0.1.7" "numpy<2" "flash-attn<1.0.5" wandb gears scanpy anndata

    mkdir -p models/scgpt
    if [ ! -d "models/scgpt/scGPT_human" ]; then
        echo "NOTE: Download scGPT_human checkpoint from bowang-lab/scGPT GitHub README"
        echo "Place in: models/scgpt/scGPT_human/"
    fi

    echo "Verifying vcbench-pt212..."
    verify_cuda
    python -c "import scgpt; print('  scGPT: OK'); from gears import PertData; print('  GEARS: OK')"
    echo "=== vcbench-pt212 ready ==="
}

# ── vcbench-pt25: TranscriptFormer + State (Python 3.11, PyTorch <=2.5.1) ──
setup_pt25() {
    echo ""
    echo "=== vcbench-pt25 (TranscriptFormer + State) ==="
    create_env_if_missing vcbench-pt25 3.11

    eval "$(conda shell.bash hook)"
    conda activate vcbench-pt25
    pip install "torch<=2.5.1" scanpy anndata

    # TranscriptFormer
    pip install transcriptformer
    echo "Downloading TranscriptFormer checkpoints..."
    transcriptformer download tf-sapiens
    transcriptformer download all-embeddings

    # Arc State
    pip install uv
    uv tool install arc-state

    echo "Verifying vcbench-pt25..."
    verify_cuda
    python -c "import transcriptformer; print('  TranscriptFormer: OK')"
    state --help > /dev/null 2>&1 && echo "  State CLI: OK"
    echo "=== vcbench-pt25 ready ==="
}

# ── Dispatch ──
echo "============================================"
echo "  VCBench: Model Environment Setup"
echo "============================================"

case "$TARGET" in
    pt118)   setup_pt118 ;;
    pt212)   setup_pt212 ;;
    pt25)    setup_pt25 ;;
    all)
        setup_pt118
        setup_pt212
        setup_pt25
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: $0 [pt118|pt212|pt25|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  Environment setup complete!"
echo ""
echo "  Env mapping:"
echo "    vcbench-pt118 -> Geneformer, UCE"
echo "    vcbench-pt212 -> scGPT"
echo "    vcbench-pt25  -> TranscriptFormer, State"
echo "============================================"
