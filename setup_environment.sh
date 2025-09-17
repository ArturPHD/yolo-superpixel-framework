#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
VENV_NAME=".venv"
REQUIRED_PYTHON_VERSION="3.11"
SPIT_REPO_URL="https://github.com/dsb-ifi/SPiT.git"
YOLOV12_REPO_URL="https://github.com/sunsmarterjie/yolov12.git"

echo "============================================"
echo "     YOLO Superpixel Project Setup (Linux)"
echo "============================================"

# --- Step 1: Create Project Structure ---
echo
echo "[1/4] Creating directory structure..."
mkdir -p src/custom_datasets src/custom_models src/training scripts models/vendor
echo "Done."

# --- Step 2: Verify Python Version ---
echo
echo "[2/4] Verifying Python version..."
# Check if python3.11 is available
if ! command -v python${REQUIRED_PYTHON_VERSION} &> /dev/null
then
    echo "ERROR: python${REQUIRED_PYTHON_VERSION} could not be found."
    echo "Please ensure Python ${REQUIRED_PYTHON_VERSION} is installed or load the appropriate module."
    exit 1
fi
echo "Found Python ${REQUIRED_PYTHON_VERSION}."

# --- Step 3: Setup Environment and Install Dependencies ---
echo
echo "[3/4] Setting up environment and installing dependencies..."
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment '$VENV_NAME'..."
    python${REQUIRED_PYTHON_VERSION} -m venv $VENV_NAME
else
    echo "Virtual environment '$VENV_NAME' already exists."
fi

echo "Activating virtual environment..."
source "${VENV_NAME}/bin/activate"

echo "Installing PyTorch with CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing remaining dependencies from requirements.txt..."
pip install -r requirements.txt
echo "Done."

# --- Step 4: Clone and Prepare Vendor Repositories ---
echo
echo "[4/4] Cloning and preparing vendor repositories..."
# Handle SPiT
if [ ! -d "models/vendor/SPiT" ]; then
    echo "Cloning SPiT repository..."
    git clone "${SPIT_REPO_URL}" "models/vendor/SPiT"
else
    echo "SPiT repository already exists."
fi

# Handle yolov12
if [ ! -d "models/vendor/yolov12" ]; then
    echo "Cloning yolov12 repository..."
    git clone "${YOLOV12_REPO_URL}" "models/vendor/yolov12"
else
    echo "yolov12 repository already exists."
fi
echo "Done."

echo
echo "============================================"
echo "     ✅ Setup Complete!"
echo "============================================"
echo
echo "Activate the environment with: source ${VENV_NAME}/bin/activate"