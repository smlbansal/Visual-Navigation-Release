WayPtNav
==========
Welcome to WayPtNav, a codebase for simulation of robot navigational scenarios in indoor-office environments! We are a team of researchers from UC Berkeley and Facebook AI Research.

In this codebase we explore ["Combining Optimal Control and Learning for Visual Navigation in Novel Environments"](https://vtolani95.github.io/WayPtNav/). We provide code to run our pretrained model based method as well as a comparable end-to-end method on geometric, point-navigation tasks in the [Stanford Building Parser Dataset](http://buildingparser.stanford.edu/dataset.html).

Additionally, we provide all code needed to generate more training data, train your own agent, and deploy it in a variety of different simulations rendered from scans of Stanford Buildings.

More information on the model-based and end-to-end methods we use is available [here](https://vtolani95.github.io/WayPtNav/).


## Setup
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

### Download the necessary data
##### 1. Download the Precomputed Control Pipeline Data
```
TODO: Put Something Here
```
##### 2. Download the meshes/traversables for the Stanford Building Parser Dataset
```
TODO: Put Something Here
```
##### 3. Download the model checkpoints for the model-based and end-to-end methods
```
TODO: Put Something Here
```
##### 4 (Optional). Download the training data used in training the model-based and end-to-end methods.
```
TODO: Put Something Here
```



## Testing Pretrained Algorithms

Along with the codebase, we provide implementations of our model-based method as well as a state-of-the-art end-to-end method trained for the task of geometric point navigation in indoor office settings. To test both of these methods on a held out set of navigational goals in a novel office building not seen at training time run the following commands.

### Test Our Model-Based Method
```
TODO: Put Something Here
```
### Test A Comparable End-to-End Method
```
TODO: Put Something Here
```
## Training Your Own Networks
We also provide the training data we used to train both the model-based and end-to-end methods. You can experiment with training your own models on this training data using the following commands:
### Train Our Model-Based Method
```
TODO: Put Something Here
```
### Train A Comparable End-to-End Method
```
TODO: Put Something Here
```

## Generating More Training Data
In addition to testing and training on our data we also offer functionality to generate your own training data using our optimal control based expert planner. You can then train and test a network on your own dataset. To generate more data run the following command:
```
TODO: Put Something Here
```
## Citing This Work
If you use the WayPtNav simulator or algorithms in your research please cite:
```
@misc{bansal2019combining,
    title={Combining Optimal Control and Learning for Visual Navigation in Novel Environments},
    author={Somil Bansal and Varun Tolani and Saurabh Gupta and Jitendra Malik and Claire Tomlin},
    year={2019},
    eprint={1903.02531},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```
