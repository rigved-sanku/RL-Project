# RL-Project
DS551 - Reinforcement Learning Project


# Instructions to Train
```
conda create --name flying_drone python=3.9
conda activate flying_drone
pip install -r requirements.txt
python3 -m pip install --extra-index-url https://pypi.nvidia.com tensorrt-bindings==8.6.1 tensorrt-libs==8.6.1
mkdir -p $HOME/.mujoco/ && wget -O - https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz | tar xzf - -C $HOME/.mujoco/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia     
python apply_patch.py
cd src/
python main.py
```