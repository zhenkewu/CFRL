import numpy as np
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from .environment import sample_trajectory, sample_counterfactual_trajectories
from .environment import SimulatedEnvironment
from .environment import estimate_counterfactual_trajectories_from_data
from .fqe import FQE

def f_ux(N, state_dim):
    return np.random.normal(0, 1, size=[N, state_dim])

def f_ua(N):
    return np.random.uniform(0, 1, size=[N])

def f_ur(N):
    return np.random.normal(0, 1, size=[N, 1])



def evaluate_reward_through_simulation(env, z_eval_levels, state_dim, N, T, policy, 
                                       f_ux=f_ux, f_ua=f_ua, f_ur=f_ur, 
                                       z_probs=None, gamma=0.9, seed=1):
    z_levels = np.array(z_eval_levels)
    np.random.seed(seed)

    # generate the sensitive attribute for each simulated individual
    #Z = np.random.binomial(1, p=0.5, size=[N, 1])
    Z = np.zeros((N, z_levels.shape[1]))
    if z_probs is None:
        Z_idx = np.random.choice(range(len(z_levels)), size=N, replace=True)
    else:
        Z_idx = np.random.choice(range(len(z_levels)), size=N, p=z_probs, replace=True)
    for i in range(N):
        Z[i] = z_levels[Z_idx[i]]

    # simulate a trajectory and compute the discounted cumulative reward
    _, _, _, rewards = sample_trajectory(env=env, zs=Z, state_dim=state_dim, T=T, 
                                         policy=policy, f_ux=f_ux, f_ua=f_ua, f_ur=f_ur, seed=seed)
    discounted_factor = np.repeat(
        np.array([[gamma**i for i in range(T)]]), repeats=N, axis=0
    )
    discounted_cumulative_reward = np.mean(discounted_factor * rewards) * T
    
    return discounted_cumulative_reward



# REQUIRES: z_eval_levels must be the one used to generate the counterfactual trajectories
def _compute_cf_metric(trajectories, z_eval_levels):
    z_eval_levels = np.array(z_eval_levels)

    max_cf_metric = 0
    for i in range(len(z_eval_levels)):
        for j in range(i, len(z_eval_levels)):
            ai = trajectories[tuple(z_eval_levels[i].flatten())]['A']
            aj = trajectories[tuple(z_eval_levels[j].flatten())]['A']
            cf_metric = np.mean(np.mean(np.abs(ai != aj), axis=0))
            if cf_metric > max_cf_metric:
                max_cf_metric = cf_metric

    return max_cf_metric



def evaluate_fairness_through_simulation(env, z_eval_levels, state_dim, N, T, 
                                         policy, f_ux=f_ux, f_ua=f_ua, f_ur=f_ur, 
                                         z_probs=None, seed=1):
    z_eval_levels = np.array(z_eval_levels)
    np.random.seed(seed)

    # generate the sensitive sttribute for each simulated individual
    zs = np.random.binomial(n=1, p=1/2, size=[N, z_eval_levels.shape[1]])
    '''zs = np.zeros((N, z_eval_levels.shape[1]))
    if z_probs is None:
        Z_idx = np.random.choice(range(len(z_eval_levels)), size=N, replace=True)
    else:
        Z_idx = np.random.choice(range(len(z_eval_levels)), size=N, p=z_probs, replace=True)
    for i in range(N):
        zs[i] = z_eval_levels[Z_idx[i]]'''

    # generate the simulated counterfactual trajectories and compute the CF metric
    trajectories = sample_counterfactual_trajectories(env=env, zs=zs, z_eval_levels=z_eval_levels, 
                                                      state_dim=state_dim, T=T, policy=policy, 
                                                      f_ux=f_ux, f_ua=f_ua, f_ur=f_ur, seed=seed)
    cf_metric = _compute_cf_metric(trajectories=trajectories, z_eval_levels=z_eval_levels)

    return cf_metric



def evaluate_fairness_through_model(env, zs, states, actions, 
                                    policy, f_ua=f_ua, seed=1):
    zs = np.array(zs)
    z_eval_levels = np.unique(zs, axis=0)
    states = np.array(states)
    actions = np.array(actions)
    np.random.seed(seed)

    # generate the simulated counterfactual trajectories
    trajectories = estimate_counterfactual_trajectories_from_data(env=env, zs=zs, states=states, 
                                                                  actions=actions, policy=policy, 
                                                                  f_ua=f_ua, seed=seed)

    # compute the CF metric
    cf_metric = _compute_cf_metric(trajectories=trajectories, z_eval_levels=z_eval_levels)

    return cf_metric



def evaluate_reward_through_fqe(
    zs, states, actions, rewards, model_type, policy, f_ua=f_ua, 
    hidden_dims=[32], lr=0.1, epochs=500, gamma=0.9, max_iter=200, seed=1, **kwargs
):
    np.random.seed(seed)
    zs = np.array(zs)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    action_size = len(np.unique(actions.flatten(), axis=0))

    fqe = FQE(model_type=model_type, action_size=action_size, policy=policy, 
              hidden_dims=hidden_dims, lr=lr, epochs=epochs, gamma=gamma)
    fqe.fit(
        states=states, zs=zs, actions=actions, rewards=rewards, 
        max_iter=max_iter, f_ua=f_ua
    )

    return np.mean(fqe.evaluate(zs=zs, states=states, actions=actions, f_ua=f_ua))