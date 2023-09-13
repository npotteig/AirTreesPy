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

## Training
First airsim must be running in the background. Copy the contents of `airsim_settings/settings_train.json` to your AirSim settings.json. This is located in `/home/USER/Documents/AirSim`. 

Notice that `ClockSpeed` and `NoDisplay` are set to increase the speed of simulation and prevent rendering the objects on screen. This speeds up training significantly. To run in headless mode in the Blocks environment, execute the following command:

```shell
./path_to_blocks/Blocks.sh -nullrhi
```
Then in a separate shell, in the root directory, execute the following:

```shell
./scripts/airnav.sh ${reward_shaping} ${timesteps} ${gpu} ${seed}  

./scripts/airnav.sh dense 5e5 0 2
```
This will save models to `navigation/models` that will then be loaded in evaluation.

## Evaluation
Copy the contents of `airsim_settings/settings_eval.json` to your AirSim settings.json. You can use the method to run AirSim from the training section if you do not want to visualize the results. This is located in `/home/USER/Documents/AirSim`.

Run Blocks environment:
```shell
./path_to_blocks/Blocks.sh
```
Then in a separate shell, in the root directory, execute the following:

### EBT
```shell
./scripts/bt_eval_airnav.sh dense 0 2
```

### Default (w/o BT components)
```shell
./scripts/eval_airnav.sh dense 0 2
```

