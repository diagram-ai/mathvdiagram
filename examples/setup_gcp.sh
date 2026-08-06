#!/bin/bash
# setup_gcp.sh -- One-shot GCP VM environment setup for the benchmark experiment.
# Run from the repository root.
set -euo pipefail

echo "=== [1/5] Installing system dependencies ==="
sudo apt-get update -qq
sudo apt-get install -y -qq \
    texlive-full \
    poppler-utils \
    libcairo2-dev pkg-config python3-dev python3-venv \
    git tmux

echo "=== [2/5] Creating Python venv ==="
python3 -m venv venv
source venv/bin/activate

echo "=== [3/5] Installing Python dependencies ==="
pip install --upgrade pip -q
pip install -e ".[benchmark]" -q

echo "=== [4/5] Adding swap (2 GB safety margin for evaluation) ==="
if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo "Swap enabled."
else
    echo "Swap already exists."
fi

echo "=== [5/5] Verifying installation ==="
python -c "import openai, PIL, tqdm, numpy, torch, piq, cv2, transformers, cairosvg, matplotlib, scipy; print('All Python deps: OK')"
pdflatex --version | head -1
pdftoppm -v 2>&1 | head -1

echo ""
echo "=== Setup complete ==="
echo "Next: source API keys and run the experiment:"
echo ""
echo '  source venv/bin/activate'
echo '  export DEEPSEEK_API_KEY=...'
echo '  export OPENAI_API_KEY=...'
echo '  export OPENROUTER_API_KEY=...'
echo '  export GROQ_API_KEY=...'
echo '  bash examples/run_experiment.sh                               # all 11 models'
echo '  MODELS="deepseek-v3 gpt-5.4" bash examples/run_experiment.sh   # specific models'
