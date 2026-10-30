#!/bin/bash

# ==========================================
# Edge Computing Device (Orange Pi) LLaMA Environment Automated Setup Script
# Target Phase: Phase 1 (PoC & Environment Initialization)
# ==========================================

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== [1/6] Cleaning up system and freeing disk space ==="
sudo journalctl --vacuum-time=2d
sudo apt clean
sudo apt autoclean
sudo apt autoremove --purge -y

echo "echo \"=== [2/6] Updating system and installing dependencies ===\""
sudo apt update && sudo apt upgrade -y
sudo apt install -y git build-essential cmake curl wget python3-pip htop

echo "=== [3/6] Creating and mounting 16GB Swap memory (Preventing OOM) ==="
# Check if /swapfile already exists; if not, create it
if [ ! -f /swapfile ]; then
    sudo fallocate -l 16G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "16GB Swap created and mounted successfully!"
else
    echo "Swap file already exists, skipping creation."
fi

echo "=== [4/6] Cloning and compiling llama.cpp from GitHub ==="
if [ ! -d "$HOME/llama.cpp" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git
fi

cd "$HOME/llama.cpp"
cmake -B build
cmake --build build --config Release -j $(nproc)

echo "=== [5/6] Creating models directory and downloading LLaMA-3.1-8B quantized model ==="
mkdir -p models

MODEL_PATH="./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading model (~4.5GB, supports resume)..."
    curl -L -C - --retry 5 --retry-delay 10 "$MODEL_URL" -o "$MODEL_PATH"
else
    echo "Model file already exists, skipping download."
fi

echo "=== [6/6] Phase 1 environment setup completed successfully! ==="
echo "You can run the following command to test single-instance inference:"
echo "cd ~/llama.cpp && ./build/bin/llama-cli -m ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -p \"Please briefly introduce Edge Computing in Traditional Chinese.\" -n 128 -t 6"