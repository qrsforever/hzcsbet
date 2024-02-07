python3 -m venv .venv/pytorch
source _env
pip3 install neovim
pip3 install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cpu
pip3 install opencv-python-headless opencv-contrib-python seaborn pandas h5py scikit-learn scikit-image imageio matplotlib

## ray
# pip3 install aiohttp opencensus aiohttp_cors pandas==2.1.4 fastapi grpcio ray pyarrow
##  bytetrack
pip3 install loguru thop lap tqdm cython_bbox py-cpuinfo
