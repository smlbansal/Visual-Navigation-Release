Visual MPC
==========
### Setup A Virtual Environment
```
conda create -n venv-mpc python=3.6
source activate venv-mpc
pip install -U pip
pip install -r requirements.txt
pip install --upgrade tensorflow (make sure to install v1.10.1)
```


### Setup the Stanford Building Parser Dataset Renderer
```
1. Clone the repository https://github.com/vtolani95/mp-env
2. Checkout branch visual_mpc and follow the setup instructions
3. Install mp-env as a package ("pip install -e ." from the root directory)
```

