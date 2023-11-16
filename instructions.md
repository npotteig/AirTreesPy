### Instructions for how to run EBT with multiple goals/waypoints in ANSR environment

Clone repo  
 ```shell
 git clone https://git.isis.vanderbilt.edu/ansr/ebt.git
```

Switch to ansr_branch
```shell
git checkout ansr_branch
```

Install dependencies
```shell
pip install -r requirements.txt
```

Download `safe_fast.zip` and move files to `./navigation/`, then unzip
```shell
unzip safe_fast.zip
```

Navigate back to root folder and execute the following script with the Blocks environment running.
```shell
./scripts/bt_eval_airnav.sh ansr grid goal_change_expert 0 2
```

Relevant Files to Look at:  
* `navigation/bt_eval.py` - Inside evaluate_policy function, you can modify `blackboard.goal_list` to the list of goals/waypoints you prefer.
* `navigation/bt_eval_nodes.py` - Implementation of the behavior tree nodes

