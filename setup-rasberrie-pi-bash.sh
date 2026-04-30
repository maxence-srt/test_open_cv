# 1. Update system packages
sudo apt update && sudo apt upgrade -y

# 2. Install system dependencies for OpenCV, matplotlib, and Jupyter
#    (libatlas-base-dev removed & replaced with libopenblas-dev)
sudo apt install -y python3-pip python3-venv libopenblas-dev libjpeg-dev zlib1g-dev

# 3. Create a directory for the virtual environment
mkdir -p ~/my_vision_env
cd ~/my_vision_env

# 4. Create a Python virtual environment
python3 -m venv venv

# 5. Activate the virtual environment
source venv/bin/activate

# 6. Increase the temporary space limit by redirecting pip's temp folder
export TMPDIR=~/my_vision_env/tmp
mkdir -p $TMPDIR

# 7. Upgrade pip and setuptools inside the venv
pip install --upgrade pip setuptools wheel

# 8. Install the required Python libraries
pip install opencv-python
pip install matplotlib
pip install ultralytics
pip install dronekit
pip install jupyter

# 9. (Optional) Verify installations
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
python -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)"
python -c "import ultralytics; print('Ultralytics version:', ultralytics.__version__)"
python -c "import dronekit; print('DroneKit version:', dronekit.__version__)"

# 10. Launch Jupyter Notebook
jupyter notebook
