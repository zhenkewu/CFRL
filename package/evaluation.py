import numpy as np
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from environment import sample_trajectory, sample_counterfactual_trajectories
from environment import SimulatedEnvironment
from environment import estimate_counterfactual_trajectories_from_data
from fqe import FQE

def evaluate_reward_through_simulation(env, z_levels, state_dim, N, T, policy, 
                                       sigma=1, seed=1, z_probs=None, gamma=0.9):
    z_levels = np.array(z_levels)
    #discounted_cumulative_rewards = np.zeros(repeats)
  
    #for iter in range(repeats):
    np.random.seed(seed)
    #Z = np.random.binomial(1, p=0.5, size=[N, 1])
    Z = np.zeros((N, z_levels.shape[1]))
    if z_probs is None:
        Z_idx = np.random.choice(range(len(z_levels)), size=N, replace=True)
    else:
        Z_idx = np.random.choice(range(len(z_levels)), size=N, p=z_probs, replace=True)
    for i in range(N):
        Z[i] = z_levels[Z_idx[i]]
    _, _, _, rewards = sample_trajectory(env, Z, state_dim, N, T, sigma, seed, policy)

    discounted_factor = np.repeat(
        np.array([[gamma**i for i in range(T)]]), repeats=N, axis=0
    )
    discounted_cumulative_reward = np.mean(discounted_factor * rewards) * T
    #discounted_cumulative_rewards[iter] = discounted_cumulative_reward
    
    return discounted_cumulative_reward



# REQUIRES: z_eval_levels must be the one used to generate the counterfactual trajectories
def _compute_cf_metric(trajectories, z_eval_levels, N, T):
    z_eval_levels = np.array(z_eval_levels)

    max_cf_metric = 0
    for i in range(len(z_eval_levels)):
        for j in range(i, len(z_eval_levels)):
            #ai = trajectories[tuple([i])]['A']
            #aj = trajectories[tuple([j])]['A']
            ai = trajectories[tuple(z_eval_levels[i].flatten())]['A']
            aj = trajectories[tuple(z_eval_levels[j].flatten())]['A']
            #cf_metric = np.mean(np.mean(np.abs(ai - aj), axis=0))
            cf_metric = np.mean(np.mean(np.abs(ai != aj), axis=0))
            #print('i =', i, 'j =', j, ':')
            #print(ai)
            #print(aj)
            #print('CF metric:', cf_metric)
            if cf_metric > max_cf_metric:
                max_cf_metric = cf_metric

    return max_cf_metric



'''def evaluate_fairness_through_simulation(env, z_eval_levels, state_dim, N, T, 
                                         policy, sigma=1, seed=1):
    z_eval_levels = np.array(z_eval_levels)
    #cf_metrics = np.zeros(repeats)

    #for iter in range(repeats):
    # generate the simulated counterfactual trajectories
    trajectories = sample_counterfactual_trajectories(env, z_eval_levels, state_dim, N, T, 
                                                    policy, sigma, seed)
    # compute the CF metric
    cf_metric = _compute_cf_metric(trajectories, z_eval_levels, N, T)
    #cf_metrics[iter] = cf_metric

    return cf_metric'''



def evaluate_fairness_through_simulation(env, z_eval_levels, state_dim, N, T, 
                                         policy, sigma=1, z_probs=None, seed=1):
    z_eval_levels = np.array(z_eval_levels)
    #cf_metrics = np.zeros(repeats)
    np.random.seed(seed)

    #zs = np.random.binomial(n=1, p=1/2, size=[N, z_eval_levels.shape[1]])
    zs = np.zeros((N, z_eval_levels.shape[1]))
    if z_probs is None:
        Z_idx = np.random.choice(range(len(z_eval_levels)), size=N, replace=True)
    else:
        Z_idx = np.random.choice(range(len(z_eval_levels)), size=N, p=z_probs, replace=True)
    for i in range(N):
        zs[i] = z_eval_levels[Z_idx[i]]
    #zs = np.array([[0], [0], [1], [1], [0]])

    #for iter in range(repeats):
    # generate the simulated counterfactual trajectories
    trajectories = sample_counterfactual_trajectories(env, zs, z_eval_levels, state_dim, N, T, 
                                                    policy, sigma, seed)
    # compute the CF metric
    cf_metric = _compute_cf_metric(trajectories, z_eval_levels, N, T)
    #cf_metrics[iter] = cf_metric

    return cf_metric



def evaluate_fairness_through_model(env, zs, states, actions, 
                                    policy, sigma_a=1, seed=1):
    z_eval_levels = np.unique(zs, axis=0)
    z_eval_levels = np.array(z_eval_levels)
    #cf_metrics = np.zeros(repeats)
    np.random.seed(seed)

    #for iter in range(repeats):
    # generate the simulated counterfactual trajectories
    trajectories = estimate_counterfactual_trajectories_from_data(env, zs, states, 
                                                                  actions, 
                                                                  policy, sigma_a, seed)
    # compute the CF metric
    cf_metric = _compute_cf_metric(trajectories, z_eval_levels, states.shape[0], states.shape[1])
    #cf_metrics[iter] = cf_metric

    return cf_metric



def evaluate_reward_through_fqe(
    zs, states, actions, rewards, model_type, policy, uat=None, 
    hidden_dims=[32], lr=0.1, epochs=500, gamma=0.9, max_iter=200, seed=1, **kwargs
):
    np.random.seed(seed)
    N, T, xdim = states.shape
    action_size = len(np.unique(actions.flatten(), axis=0))

    fqe = FQE(model_type=model_type, action_size=action_size, policy=policy, 
              hidden_dims=hidden_dims, lr=lr, epochs=epochs, gamma=gamma)
    fqe.fit(
        states=states, zs=zs, actions=actions, rewards=rewards, 
        max_iter=max_iter, uat=uat
    )

    return np.mean(fqe.evaluate(zs=zs, states=states, actions=actions, uat=uat))