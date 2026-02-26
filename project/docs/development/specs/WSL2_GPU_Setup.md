# WSL2 + GPU Setup Guide for Splendor RL Project

**Date**: 2026-02-24  
**Platform**: Windows Subsystem for Linux 2 (WSL2)  
**GPU**: NVIDIA RTX 4090  

---

## Why WSL2 is Better for This Project

✅ **Advantages**:
- Native Linux environment (better for RL/ML workflows)
- Better package compatibility (no Windows DLL issues)
- Easier debugging (Linux error messages clearer)
- Integration with VS Code Remote
- Can still access Windows files

✅ **GPU Support**: WSL2 supports CUDA through Windows driver passthrough

---

## Prerequisites Check

### 1. Verify WSL2 (not WSL1)

```powershell
# Run in Windows PowerShell
wsl --list --verbose
```

Expected output:
```
  NAME            STATE           VERSION
* Ubuntu          Running         2        ← Must be 2!
```

If VERSION is 1, upgrade:
```powershell
wsl --set-version Ubuntu 2
```

### 2. Verify Windows NVIDIA Driver

The GPU driver on **Windows side** automatically works for WSL2.

```powershell
# Check driver version (Windows side)
nvidia-smi
```

Should show RTX 4090. **Do NOT install NVIDIA drivers inside WSL2!**

---

## Installation Steps (Inside WSL2)

### Step 1: Enter WSL2

```powershell
# From Windows PowerShell
wsl
```

Now you're in Linux! All commands below run **inside WSL**.

---

### Step 2: Install Miniconda (Linux version)

```bash
# Inside WSL2
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Initialize conda
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Verify
conda --version
```

---

### Step 3: Create Conda Environment

```bash
# Create environment with Python 3.10
conda create -n splendor python=3.10 -y
conda activate splendor
```

---

### Step 4: Install PyTorch with CUDA

```bash
# Install PyTorch (CUDA 12.1 for RTX 4090)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**Important**: Use `pytorch-cuda`, NOT the Linux CUDA toolkit. WSL2 uses Windows driver.

---

### Step 5: Test GPU

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output:
```
PyTorch: 2.5.1
CUDA Available: True
GPU: NVIDIA GeForce RTX 4090
```

✅ If you see this, GPU is working!

---

### Step 6: Install RL Dependencies

```bash
# Still in splendor environment
pip install stable-baselines3[extra]
pip install gym pyyaml tensorboard
pip install numpy pandas matplotlib seaborn tqdm
```

---

### Step 7: Navigate to Project

```bash
# WSL2 can access Windows files at /mnt/c/
cd /mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759

# Or set up symlink for convenience
ln -s /mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759 ~/splendor-project
cd ~/splendor-project
```

---

### Step 8: Run Environment Check

```bash
cd project
python tests/test_environment.py
```

Expected output:
```
============================================================
IFT6759 Splendor RL - Environment Check
============================================================

1. Python Environment:
✅ Python 3.10.x

2. PyTorch & GPU:
✅ PyTorch 2.5.1
✅ CUDA Available: NVIDIA GeForce RTX 4090
   GPU Memory: 24.00 GB

3. Reinforcement Learning:
✅ Stable-Baselines3
✅ Gym

4. Splendor Game:
✅ Splendor environment available
   ✅ Environment reset successful

🎉 All critical dependencies satisfied!
============================================================
```

---

## VS Code Integration (Highly Recommended)

### Install VS Code Extensions

1. **WSL** extension (ms-vscode-remote.remote-wsl)
2. **Python** extension
3. **Jupyter** extension (optional)

### Open Project in WSL

```powershell
# From Windows
cd C:\Users\yehao\Documents\03Study\IFT6759\Splendor-6759
code .
```

VS Code will detect WSL and ask to "Reopen in WSL". Click Yes!

Or directly:
```bash
# From inside WSL
cd ~/splendor-project
code .
```

### Select Python Interpreter

1. Press `Ctrl+Shift+P`
2. Type: "Python: Select Interpreter"
3. Choose: `Python 3.10.x ('splendor')`

---

## Daily Workflow

### Starting a Session

```bash
# Open WSL terminal
wsl

# Activate environment
conda activate splendor

# Navigate to project
cd ~/splendor-project  # or /mnt/c/Users/yehao/.../Splendor-6759

# Start working!
python scripts/train_score_based.py
```

### Monitoring Training (TensorBoard)

```bash
# Terminal 1: Training
python scripts/train_score_based.py

# Terminal 2: TensorBoard
conda activate splendor
tensorboard --logdir logs/tensorboard/
```

Then open browser: `http://localhost:6006`

---

## Common Issues & Solutions

### Issue 1: "CUDA not available"

**Cause**: Windows NVIDIA driver not installed or outdated

**Solution**:
```powershell
# On Windows side, download latest driver from:
# https://www.nvidia.com/Download/index.aspx
```

### Issue 2: "Permission denied" accessing /mnt/c

**Solution**:
```bash
# Add to ~/.bashrc
echo 'cd /mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759' >> ~/.bashrc
```

### Issue 3: Slow file access from /mnt/c

**Recommendation**: Clone project into native WSL filesystem

```bash
# Better performance
cp -r /mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759 ~/splendor-project
cd ~/splendor-project
```

**Trade-off**: Files only accessible from WSL, not Windows Explorer.

**Hybrid approach**: Keep docs/code on Windows, run training from WSL copy.

### Issue 4: Out of memory

**Solution**: WSL2 memory limit can be adjusted

Create `C:\Users\yehao\.wslconfig`:
```ini
[wsl2]
memory=24GB      # Adjust as needed (you have 32GB total)
processors=16    # Use 16 of 32 threads
```

Then restart WSL:
```powershell
wsl --shutdown
wsl
```

---

## Performance Tips

### 1. Use Native WSL Filesystem for Training

```bash
# Copy project to WSL home
rsync -av /mnt/c/Users/yehao/.../Splendor-6759/ ~/splendor-project/

# Work here for training
cd ~/splendor-project
```

10-20x faster I/O than /mnt/c!

### 2. Share Models/Logs with Windows

```bash
# After training, copy results back
cp -r logs /mnt/c/Users/yehao/.../Splendor-6759/project/
cp -r models /mnt/c/Users/yehao/.../Splendor-6759/project/
```

### 3. Use tmux for Long Training Sessions

```bash
# Install tmux
sudo apt install tmux

# Start session
tmux new -s training

# Start training
python scripts/train_score_based.py

# Detach: Ctrl+B, then D
# Training continues even if you close terminal!

# Re-attach later
tmux attach -s training
```

---

## Comparison: WSL2 vs Windows Native

| Aspect | WSL2 | Windows Native |
|--------|------|----------------|
| Setup | Harder | Easier |
| Performance | **Better** (native Linux) | Good |
| Compatibility | **Better** (more packages) | Some issues |
| File I/O | Fast (native FS) | Slower |
| Debugging | **Easier** | Harder |
| VS Code | **Best** (Remote) | Good |
| Recommended | ✅ **Yes** | For simple tasks |

---

## Next Steps After Setup

```bash
# Verify everything works
cd ~/splendor-project/project  # or /mnt/c/.../project
python tests/test_environment.py

# If all green, tell AI:
# "WSL2 environment ready, let's continue with Task 1.2"
```

---

## Quick Reference Card

```bash
# Activate environment
conda activate splendor

# Check GPU
nvidia-smi  # Does NOT work in WSL2!
python -c "import torch; print(torch.cuda.is_available())"  # Use this!

# Project location
cd /mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759
# Or better:
cd ~/splendor-project

# Start training
python scripts/train_score_based.py

# Monitor training
tensorboard --logdir logs/tensorboard/

# Background training (detach with Ctrl+B,D)
tmux new -s training
python scripts/train_score_based.py
```

---

**Last Updated**: 2026-02-24  
**Status**: Ready for implementation
