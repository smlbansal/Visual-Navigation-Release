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

#### Patch the OpenGL Installation
In the terminal from the root directory of the project run the following commands.
```
1. cd mp_env
2. bash patches/apply_patches_3.sh
```
If the script fails there are instructions in apply_patches_3.sh describing how to manually apply the patch.

### Download the necessary data (~13 GB)
```
TODO: Put Something Here
```
#### (Optional). Download the training data used in training the model-based and end-to-end methods. (~82 GB)
```
TODO: Put Something Here
```
### Configure WayptNav to look for your data installation.
##### Configure the Control Pipeline Data Directory
In ./params/control_pipeline_params.py change the following line
```
p.dir = 'PATH/TO/DATA/control_pipelines' 
```
##### Configure the Stanford Building Parser Dataset Data Directory
In ./params/renderer_params.py change the following line
```
def get_sbpd_data_dir():
	return 'PATH/TO/DATA/stanford_building_parser_dataset'
```
##### Configure the Pretrained Weights Data Directory
In ./params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py change the following line:
```
p.trainer.ckpt_path = 'PATH/TO/DATA/pretrained_weights/WayPtNav/session_2019-01-27_23-32-01/checkpoints/ckpt-9'
```
In ./params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_control_trainer_finetune_params.py change the following line:
```
p.trainer.ckpt_path = 'PATH/TO/DATA/pretrained_weights/E2E/session_2019-01-27_23-34-22/checkpoints/ckpt-18'
```
##### Configure the Expert Success Goals Data Directory
In ./params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py and ./params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_control_trainer_finetune_params.py change the following line:
```
p.test.expert_success_goals = DotMap(use=True,
							         dirname='PATH/TO/DATA/expert_success_goals/sbpd_projected_grid')
```
##### Configure the Training Data Directory (Optional)
In ./params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py and ./params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_control_trainer_finetune_params.py change the following line:
```
p.data_creation.data_dir = ['PATH/TO/DATA/sbpd_projected_grid_include_last_step_successful_goals_only/area3/full_episode_random_v1_100k',
						    'PATH/TO/DATA/sbpd_projected_grid_include_last_step_successful_goals_only/area4/full_episode_random_v1_100k',
   							'PATH/TO/DATA/sbpd_projected_grid_include_last_step_successful_goals_only/area5a/full_episode_random_v1_100k']
```


### Run the Tests
To ensure you have successfully installed the WayptNav codebase run the following command. All tests should pass.
```
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python executables/run_all_tests.py
```

## Getting Started
#### Overview
The WayptNav codebase is designed to allow you to:

	1. Create training data using an expert policy
	2. Train a network (for either model based or end-to-end navigation)  
	3. Test a trained network

Each of these 3 tasks can be run via an executable file. All of the relevant executable files are located in the ./executables subdirectory of the main project. To use an executable file the user must specify
```
1. mode (generate_data, train, or test)
2. job_dir (where to save the all relevant data from this run of the executable)
3. params (which parameter file to use)
4. device (which device to run tensorflow on. -1 forces CPU, 0+ force the program to run on the corresponding GPU device)
```
#### Generate Data, Train, and Test a Sine Function
We have provided a simple example to train a sine function for your understanding. To generate data, train and test the sine function example using GPU 0 run the following 3 commands:
```
1. PYTHONPATH='.' python executables/sine_function_trainer generate-data --job-dir JOB_DIRECTORY_NAME_HERE --params params/sine_params.py -d 0
2. PYTHONPATH='.' python executables/sine_function_trainer train --job-dir JOB_DIRECTORY_NAME_HERE --params params/sine_params.py -d 0

In ./params/sine_params.py change p.trainer.ckpt_path to point to a checkpoint from the previously run training session. For example:

3a. p.trainer.ckpt_path = 'PATH/TO/PREVIOUSLY/RUN/SESSION/checkpoints/ckpt-10'

3b. PYTHONPATH='.' python executables/sine_function_trainer test --job-dir JOB_DIRECTORY_NAME_HERE --params params/sine_params.py -d 0
```

The output of testing the sine function will be saved in 'PATH/TO/PREVIOUSLY/RUN/SESSION/TEST/ckpt-10'.

## Testing Pretrained Visual Navigation Algorithms

Along with the codebase, we provide implementations of our model-based method as well as a state-of-the-art end-to-end method trained for the task of geometric point navigation in indoor office settings. To test both of these methods on a held out set of navigational goals in a novel office building not seen at training time run the following commands.

### Test Our Model-Based Method
Example Command
```
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python executables/rgb/resnet50/rgb_waypoint_trainer.py test --job-dir reproduce_WayptNavResults --params params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py -d 0
```
Results will be saved in the following directory:

```
path/to/pretrained_weights/session_2019-01-27-23-32-01/test/checkpoint_9/reproduce_WayptNavResults/session_CURRENT_DATE_TIME/rgb_resnet50_nn_waypoint_simulator
```

### Test A Comparable End-to-End Method
Example Command
```
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python executables/rgb/resnet50/rgb_control_trainer.py test --job-dir reproduce_WayptNavResults --params params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_control_trainer_finetune_params.py -d 0
```
Results will be saved in the following directory:
```
path/to/pretrained_weights/session_2019-01-27-23-34-22/test/checkpoint_18/reproduce_WayptNavResults/session_CURRENT_DATE_TIME/rgb_resnet50_nn_control_simulator
```
## Training Your Own Networks
We also provide the training data we used to train both the model-based and end-to-end methods. You can experiment with training your own models on this training data using the following commands:
### Train Our Model-Based Method
```
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python executables/rgb/resnet50/rgb_waypoint_trainer.py train --job-dir PATH/TO/LOG/DIR --params params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py -d 0
```
### Train A Comparable End-to-End Method
```
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python executables/rgb/resnet50/rgb_control_trainer.py train --job-dir PATH/TO/LOG/DIR --params params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_control_trainer_finetune_params.py -d 0
```

## Generating More Training Data
In addition to testing and training on our data we also offer functionality to generate your own training data using our optimal control based expert planner. You can then train and test a network on your own dataset. To generate more data run the following command:  
#### Change the data_dir to reflect the desired directory for your new data
In params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py
```
p.data_creation.data_dir = 'PATH/TO/NEW/DATA'
```
Run the following command to create new data.
```
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python executables/rgb/resnet50/rgb_waypoint_trainer.py generate-data --job-dir PATH/TO/LOG/DIR --params params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py -d 0
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
