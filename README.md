# AirTreesPy
Neuro-symbolic Quadcopter Navigation using PyTrees and Microsoft AirSim

## Install Dependencies
Install and build AirSim version 1.8.1 for Ubuntu 20.04 (Note: Installing and Building Unreal Engine is not required if using the Blocks Environment Binary)
https://microsoft.github.io/AirSim/build_linux/

Download Blocks environment binary for v1.8.1 Linux from https://github.com/microsoft/AirSim/releases and unzip the folder. 

Create a new virtual environment with Python==3.10 then execute this command:
```shell
pip install -r requirements.txt
```

## Pretrained Models
To use pretrained data, as training runs can take hours, unzip the file provided. 
Scripts are compatible with using pretrained weights. If you want to use your model data, you will need to modify scripts with location of your model data.

```shell
unzip pretrained.zip
```

## Training in Small AirSim Environment
The small AirSim Environment is used to train RL policies from scratch that can then be transferred to other complex AirSim environments. The small environment consists of a 200x200 meter, where the drone spawns in the center. The boundary is outlined by a wall and the interior consists of obstacles (rectangular blocks).

First airsim must be running in the background. Copy the contents of `airsim_settings/settings.json` to your AirSim settings.json. This is located in `/home/USER/Documents/AirSim`. 

Notice that `ClockSpeed` and `NoDisplay` are set to increase the speed of simulation and prevent rendering the objects on screen. This speeds up training significantly. To run in headless mode in the Blocks environment, execute the following command:

```shell
./path_to_blocks/Blocks.sh -nullrhi
```
Then in a separate shell, in the root directory, execute the following:

```shell
./scripts/airnav.sh ${timesteps} ${gpu} ${seed}  

./scripts/airnav.sh dense 5e5 0 2
```
This will save models to `navigation/models` that will then be loaded in evaluation.

## Transfer Learning in Airsim Environment

The transfer environment is a 5x5 sq mi area consisting of tree-like objects (cylinders). Our environment only considers a 300x300 meter portion of the environment centered on the drone's spawn location. The size of the environment can be modified in `airmap/blocks_tree_generator.py` in line 78.

There are two types of learning we can accomplish when transfering to this environment. We can continue to learn the RL policies in the new environment, learn landmarks, or both.

Always ensure AirSim is running before executing these scripts.

### Learn Landmarks only
```shell
./scripts/collect_landmarks.sh 1e5 0 2
```

### Learn Landmarks and Transfer Learn RL policies

```shell
./scripts/load_airnav.sh 25e4 0 2
```


## Evaluation
Copy the contents of `airsim_settings/settings.json` to your AirSim settings.json. `ClockSpeed` can be reduced and `NoDisplay` removed if you want to visualize the drone. This is located in `/home/USER/Documents/AirSim`.

Run Blocks environment:
```shell
./path_to_blocks/Blocks.sh
```
Then in a separate shell, in the root directory, execute the following:

### EBT
There are multiple configurations for running the EBT. Training and Transfer are the two environments supported. FPS is the learned landmarks method, while Grid is a simple grid based landmark sampler. Expert is the EBT created by our knowledge and GP is one created with prior knowledge and genetic programming. Goal_Change_Expert supports traveling to multiple goals. Recommended to use grid landmarks with goal_change.

```shell
./scripts/bt_eval_airnav.sh ${ENV_TYPE} ${LANDMARK_SAMPLING_METHOD} ${BT_TYPE} ${GPU} ${SEED}

./scripts/bt_eval_airnav.sh ['training', 'transfer'] ['fps', 'grid'] ['expert', 'goal_change_expert', 'gp'] 0 2
```

#### Multi-Goal Navigation Example

```shell
./scripts/bt_eval_airnav.sh transfer grid goal_change_expert 0 2
```

Relevant Files to Look at:  
* `navigation/bt_eval.py` - Inside evaluate_policy function, you can modify `blackboard.goal_list` to the list of goals/waypoints you prefer. You can modify `blackboard.env.set_init_pos` parameter to change initial position of drone. Both input meters/10 units (i.e. 5.5 -> 55 meters).
* `navigation/bt_eval_nodes.py` - Implementation of the behavior tree nodes