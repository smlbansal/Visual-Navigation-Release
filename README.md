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
```

#### Install Tensorflow (v 1.10.1)
For an ubuntu machine with GPU support run the following:
```
pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.1-cp36-cp36m-linux_x86_64.whl
```


### Download the necessary data
##### 1. Download the Precomputed Control Pipeline Data
```
TODO: Put Something Here
```
##### 2. Download the meshes/traversables for the Stanford Building Parser Dataset
```
1. TODO: Put Something Here
```
##### 2a. Update the directory for the building traversables and meshes
```
In ./params/renderer_params.py replace "get_traversible_dir" and "get_sbpd_data_dir" with the location of your SBPD data.

def get_traversible_dir():
    return '/home/ext_drive/somilb/data/stanford_building_parser_dataset/traversibles'


def get_sbpd_data_dir():
    return '/home/ext_drive/somilb/data/stanford_building_parser_dataset/'
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
Results will be saved in the following directory:
path/to/pretrained_weights/session_2019-01-27-23-32-01/test/checkpoint_9/reproduce_WayptNavResults/session_CURRENT_DATE_TIME/rgb_resnet50_nn_waypoint_simulator
```
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python executables/rgb/resnet50/rgb_waypoint_trainer.py test
--job-dir reproduce_WayptNavResults --params params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py -d 0
```
### Test A Comparable End-to-End Method
Results will be saved in the following directory:
path/to/pretrained_weights/session_2019-01-27-23-34-22/test/checkpoint_18/reproduce_WayptNavResults/session_CURRENT_DATE_TIME/rgb_resnet50_nn_control_simulator
```
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python executables/rgb/resnet50/rgb_control_trainer.py test
--job-dir reproduce_WayptNavResults --params
params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_control_trainer_finetune_params.py -d 0
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
