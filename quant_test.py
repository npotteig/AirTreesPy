import numpy as np

import torch
import torch.nn as nn
from torch.quantization import quantize_fx
import copy
from higl.models import ControllerActor
import higl.utils as utils

device = torch.device("cpu")


def var(tensor):
    return tensor.to(device)


def get_tensor(z):
    if z is None:
        return None
    if z[0].dtype == np.dtype("O"):
        return None
    if len(z.shape) == 1:
        return var(torch.FloatTensor(z.copy())).unsqueeze(0)
    else:
        return var(torch.FloatTensor(z.copy()))

def clean_obs(state, dims=2):
    with torch.no_grad():
        mask = torch.ones_like(state)
        if len(state.shape) == 3:
            mask[:, :, :dims] = 0
        elif len(state.shape) == 2:
            mask[:, :dims] = 0
        elif len(state.shape) == 1:
            mask[:dims] = 0

        return state*mask
    
def select_action(actor, state, sg):
    state = get_tensor(state)
    sg = get_tensor(sg)
    state = clean_obs(state)

    return actor(state, sg).cpu().data.numpy().squeeze()
        

if __name__ == '__main__':
    backend = "fbgemm"
    
    control_actor = ControllerActor(13, 2, 2, scale=10)
    controller_buffer = utils.ReplayBuffer(maxsize=60000,
                                           reward_func=None,
                                           reward_scale=None)
    control_actor.load_state_dict(torch.load('navigation/safe_fast/models/AirSimEnv-v0_higl_dense_2_ControllerActor.pth'))
    controller_buffer.load("navigation/_controller.npz")
    
    state = np.zeros(13)
    goal = np.array([0.1, 0.03])
    action = select_action(control_actor, state, goal)
    print(action)
    
    q_ctrl_actor = copy.deepcopy(control_actor)
    q_ctrl_actor.eval()
    
    qconfig_dict = {"": torch.quantization.get_default_qconfig(backend)}
    q_ctrl_prepared = quantize_fx.prepare_fx(q_ctrl_actor, qconfig_dict)
    
    with torch.inference_mode():
        for _ in range(20):
            # Get mini-batch
            # Called forward
            pass
            
    ctrl_quantized = quantize_fx.convert_fx(q_ctrl_prepared)
    
    
    
    
    

    
    
    
    
