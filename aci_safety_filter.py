# Author: Sacha Huriot
# Project: Safe Navigation with Adaptive Conformal Predictions in a World Model

'''
/dino_wm/train_HJ_dubinslatent.py   (train HJ) code
 
 wandb.ai/i-k-tabbara-washington-university-in-st-louis/ddpg-hj-latent-dubins/workspace?nw=nwuseriktabbara checkpoint logs for HJ to pick what you want
 
 /dino_wm/runs/ddpg_hj_latent HJ checkpoints (u can match the name of checkpoint with the name of run on wandb website)
 
 /checkpt_dino/outputs2 world model checkpoints
 
 /dino_wm/env/dubins dubins env
 
 /dino_wm/train_HJ_dubinstruth_BRS.py the function function _get_avoidable_dubins is safest controller
 
 /dino_wm/train_HJ_dubinslatent_withfinetune.py collect_trajectories() How to use the env in latent space
 
 /dino_wm/train_HJ_dubinslatent_withfinetune.py collect_trajectories() How to use the HJ safe controller
 '''

import bisect
import math

from train_HJ_dubinstruth_BRS import compute_hj_value
from train_HJ_dubinslatent_withfinetune import load_shared_world_model

def dubins_safety_function(state, safe_policy):
    x, y, theta = state
    return compute_hj_value(x, y, theta, policy, None)


class FilterController():

    def __init__(self, ckpt_dir: str, device: str, value_function, value_function_args, pi_task, pi_safe):
        self.learning_rate = 0.05
        self.target_miscoverage = 0.003
        self.wm = load_shared_world_model(ckpt_dir, device)
        self.V = (lambda s => value_function(s, value_function_args))
        self.pi_task = pi_task
        self.pi_safe = pi_safe

    def safety_function(state):
        '''Retrieve the HJ reachability value of the state'''
        return self.V(state)

    def expected_value(state, control):
        '''Sample the (stochastic) next state and average their safety value'''
        return 0

    def task_policy(state):
        '''Retrieve the control which maximizes expected task completion'''
        return self.pi_task(state)

    def safe_policy(state):
        '''Retrieve the control which maximizes expected safety value'''
        return self.pi_safe(state)

    def aci_safety_filter(state, prev_prediction, score_history, prev_quantile, prev_miscoverage_rate):
        '''
        (In paper) Algorithm 1 Adaptive Safety-Filtering Controller
        input: Current model state, previous prediction of the safety value, 
            history of scores, previous quantile, previous miscoverage rate
        global parameters: learning rate, target miscoverage rate
        effect: modifies history of scores in place
        output: control decision, new prediction of the safety value,
            new quantile, new miscoverage rate
        '''
        state_safety_value = self.safety_function(state)
        if prev_quantile == '-Infinity' or 
            (prev_quantile != '+Infinity' and state_safety_value < prev_prediction - prev_quantile):
            error = 1
        else:
            error = 0
        miscoverage_rate = prev_miscoverage_rate + self.learning_rate * (self.target_miscoverage - error)
        score = max(prev_prediction - state_safety_value, 0)
        bisect.insort(score_history, score)
        if miscoverage_rate < 0:
            quantile = '+Infinity'
        elif miscoverage_rate > 1:
            quantile = '-Infinity'
        else:
            quantile = score_history[math.ceil((1 - miscoverage_rate) * len(score_history)) - 1]
        task_control = self.task_policy(state)
        candidate_safety_value_prediction = self.expected_value(state, task_control)
        if candidate_safety_value > 0 and 
            (quantile == '-Infinity' or 
            (quantile != '+Infinity' and candidate_safety_value > quantile)):
            return task_control, candidate_safety_value_prediction, quantile, miscoverage_rate
        else:
            safe_control = self.safe_policy(state)
            safety_value_prediction = self.expected_value(state, safe_control)
            return safe_control, safety_value_prediction, quantile, miscoverage_rate


def main():
    dubinsFilter = FilterController()