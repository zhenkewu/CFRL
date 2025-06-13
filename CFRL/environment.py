import gymnasium as gym
import numpy as np
import copy
import torch
from sklearn.preprocessing import OneHotEncoder
from .utils.base_models import NeuralNetRegressor, LinearRegressor
from typing import Union, Callable, Literal, Dict
from .agents import Agent

def f_x0(
        zs: list | np.ndarray, 
        ux0: list | np.ndarray, 
        z_coef: int | float = 1
    ) -> np.ndarray:
    zs = np.array(zs)
    ux0 = np.array(ux0)
    gamma0 = np.array([-0.3, 1 * z_coef, 1])
    n = zs.shape[0]
    M = np.concatenate(
        [
            np.ones([n, 1]),
            zs,
            ux0,
        ],
        axis=1,
    )
    x0 = M @ gamma0
    x0 = x0.reshape(-1, 1)
    return x0

def f_xt(
        zs: list | np.ndarray, 
        xtm1: list | np.ndarray, 
        atm1: list | np.ndarray, 
        uxt: list | np.ndarray, 
        z_coef: int | float = list | np.ndarray
    ) -> np.ndarray:
    zs = np.array(zs)
    xtm1 = np.array(xtm1)
    atm1 = np.array(atm1)
    uxt = np.array(uxt)
    gamma = np.array([-0.3, 1 * z_coef, 0.5, 0.4, 0.3, 0.3 * z_coef, 0.4 * z_coef, 1]) #-0.3
    n = xtm1.shape[0]
    M = np.concatenate(
        [
            np.ones([n, 1]),
            (zs - 0.5),
            xtm1,
            atm1.reshape(-1, 1) - 0.5,
            xtm1 * (atm1.reshape(-1, 1) - 0.5),
            xtm1 * (zs - 0.5),
            (zs - 0.5) * (atm1.reshape(-1, 1) - 0.5),
            uxt,
        ],
        axis=1,
    )
    xt = M @ gamma
    xt = xt.reshape(-1, 1)
    return xt

def f_rt(
        zs: list | np.ndarray, 
        xt: list | np.ndarray, 
        at: list | np.ndarray, 
        urtm1: list | np.ndarray, 
        z_coef: int | float =1
    ) -> np.ndarray:
    zs = np.array(zs)
    xt = np.array(xt)
    at = np.array(at)
    urtm1 = np.array(urtm1)
    lmbda = np.array([-0.3, 0.3, 0.5 * z_coef, 0.5, 0.2 * z_coef, 0.7, -1.0 * z_coef])
    n = xt.shape[0]
    at = at.reshape(-1, 1)
    M = np.concatenate(
        [np.ones([n, 1]), xt, zs, at, xt * zs, xt * at, zs * at], axis=1
    )
    rt = M @ lmbda
    return rt



class SyntheticEnvironment(gym.Env):
    def __init__(
            self, 
            state_dim: int = 1, 
            z_coef: int | float = 1, 
            f_x0: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = f_x0, 
            f_xt: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray] = f_xt, 
            f_rt: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray] = f_rt
        ) -> None:
        self.state_dim = state_dim
        self.z_coef = z_coef
        self.f_x0 = f_x0
        self.f_xt = f_xt
        self.f_rt = f_rt

        # temporarily store the state, action, and reward data of the environment
        self.xt = np.zeros(state_dim)
        self.atm1 = None
        self.rtm1 = 0
    
    '''def get_z_coef(self):
        return self.z_coef
    
    def set_z_coef(self, new_z_coef):
        self.z_coef = new_z_coef'''
    
    # ux0 needs to be 2D
    def reset(
            self, 
            z: list | np.ndarray, 
            ux0: list | np.ndarray
        ) -> tuple[np.ndarray, None]:
        zs = np.array(z)
        ux0 = np.array(ux0)
        self.N = zs.shape[0] # number of samples/individuals
        self.zs = zs.reshape(self.N, -1)
        #np.random.seed(self.seed)

        # generate initial state
        self.xt = self.f_x0(zs=self.zs, ux0=ux0, z_coef=self.z_coef)

        # form the current observation
        observation = self.xt

        return observation, None
    
    # urtm1 needs to be 2D
    def step(
            self, 
            action: list | np.ndarray, 
            uxt: list | np.ndarray, 
            urtm1: list | np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, Literal[False], Literal[False]]:
        action = np.array(action)
        uxt = np.array(uxt)
        urtm1 = np.array(urtm1)
        self.atm1 = action
        #np.random.seed(self.seed) # ALSO NEED SEED HERE OR NOT?

        # generate next state and reward
        self.rtm1 = self.f_rt(zs=self.zs, xt=self.xt, at=self.atm1, urtm1=urtm1, z_coef=self.z_coef)
        self.xt = self.f_xt(zs=self.zs, xtm1=self.xt, atm1=self.atm1, uxt=uxt, z_coef=self.z_coef)

        # form the current observation and compute the reward
        observation = self.xt
        reward = self.rtm1

        # the synthetic enviornment never terminates or gets truncated
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated



def f_ux(N: int, state_dim: int) -> np.ndarray:
    return np.random.normal(0, 1, size=[N, state_dim])

def f_ua(N: int) -> np.ndarray:
    return np.random.uniform(0, 1, size=[N])

def f_ur(N: int) -> np.ndarray:
    return np.random.normal(0, 1, size=[N, 1])

# REQUIRES: zs should be the same as the zs passed to the environment
def sample_trajectory(
        env: SyntheticEnvironment, 
        zs: list | np.ndarray, 
        state_dim: int, 
        T: int, 
        policy: Agent, 
        f_ux: Callable[[int, int], np.ndarray] = f_ux, 
        f_ua: Callable[[int], np.ndarray] = f_ua, 
        f_ur: Callable[[int], np.ndarray] = f_ur, 
        seed: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    zs = np.array(zs)
    N = zs.shape[0]

    # initialize containers to store the trajectory
    Z = zs.reshape(N, -1)
    X = np.zeros([N, T + 1, state_dim], dtype=float)
    A = np.zeros([N, T], dtype=int)
    R = np.zeros([N, T], dtype=float)
    np.random.seed(seed)

    # SHOULD DELETE RELATED COMPONENTS
    policy_c = copy.deepcopy(policy)
    X_c = np.zeros([N, T + 1, state_dim], dtype=float)
    A_c = np.zeros([N, T], dtype=int)
    R_c = np.zeros([N, T], dtype=float)
    L_c = np.zeros([N, T, 2], dtype=float)

    # generate the initial state
    ux0 = f_ux(N=N, state_dim=state_dim)
    #ux0 = np.random.normal(0, sigma, [N, 1])
    #ux0 = np.ones((N, 1))
    X[:, 0], _ = env.reset(z=Z, ux0=ux0)
    
    # take the first step
    ua0 = f_ua(N=N)
    #ua0 = np.random.uniform(0, 1, size=[N])
    #ua0 = np.zeros(N)
    A[:, 0] = policy.act(
            z=Z,
            xt=X[:, 0],
            xtm1=None,
            atm1=None,
            uat=ua0,
        )
    A_c[:, 0] = policy_c.act(
            z=Z,
            xt=X_c[:, 0],
            #atm=A_c[:, :0],
            #t=0,
            xtm1=None,
            atm1=None,
            uat=ua0,
            is_return_prob=False,
        ) # SHOULD DELETE LATER
    ur0 = f_ur(N=N)
    ux1 = f_ux(N=N, state_dim=state_dim)
    #ur0 = np.random.normal(0, sigma, [N, 1])
    #ux1 = np.random.normal(0, sigma, [N, 1])
    X[:, 1], R[:, 0], _, _ = env.step(action=A[:, 0], uxt=ux1, urtm1=ur0)

    # take subsequent steps
    for t in range(1, T):
        uat = f_ua(N=N)
        #uat = f_ua([N])
        #uat = np.random.uniform(0, 1, size=[N])
        #uat = np.ones(N)
        A[:, t] = policy.act(
                z=Z,
                xt=X[:, t],
                xtm1=X[:, t - 1],
                atm1=A[:, t - 1],
                uat=uat,
            )
        A_c[:, t] = policy_c.act(
                z=Z,
                xt=X_c[:, t],
                #atm=A_c[:, :t],
                #t=t,
                xtm1=X_c[:, t - 1],
                atm1=A_c[:, t - 1],
                uat=uat,
                is_return_prob=False,
            ) # SHOULD DELETE LATER
        urtm1 = f_ur(N=N)
        uxt = f_ux(N=N, state_dim=state_dim)
        #urtm1 = np.random.normal(0, sigma, [N, 1])
        #uxt = np.random.normal(0, sigma, [N, 1])
        X[:, t + 1], R[:, t], _, _ = env.step(action=A[:, t], uxt=uxt, urtm1=urtm1)

    # prepare and return the output
    #out = [Z, X, A, R]
    return Z, X, A, R



# z_eval_levels should have shape (N, zdim)
def sample_counterfactual_trajectories(
        env: SyntheticEnvironment, 
        zs: list | np.ndarray, 
        z_eval_levels: list | np.ndarray, 
        state_dim: int, 
        T: int, 
        policy: Agent, 
        f_ux: Callable[[int, int], np.ndarray] = f_ux, 
        f_ua: Callable[[int], np.ndarray] = f_ua, 
        f_ur: Callable[[int], np.ndarray] = f_ur, 
        seed: int = 1
    ) -> dict[tuple[Union[int, float], ...], dict[str, Union[np.ndarray, SyntheticEnvironment, Agent]]]:
    np.random.seed(seed)
    zs = np.array(zs)
    z_eval_levels = np.array(z_eval_levels)
    N = zs.shape[0]

    # dictionary to contain the counterfactual trajectories; key = z and value = trajectory
    trajectories = {}
    for z_level in z_eval_levels:
        env_z = copy.deepcopy(env)
        policy_z = copy.deepcopy(policy)
        Z = np.repeat(np.array([z_level]), repeats=N, axis=0) # RECENTLY CHANGED
        X = np.zeros([N, T + 1, state_dim], dtype=float)
        A = np.zeros([N, T], dtype=int)
        R = np.zeros([N, T], dtype=float)
        trajectories[tuple(z_level.flatten())] = {'Z': Z, 'X': X, 'A': A, 'R': R, 
                                        'env_z': env_z, 'policy_z': policy_z}

    # generate the initial state
    ux0 = f_ux(N=N, state_dim=state_dim)
    #ux0 = np.random.normal(0, sigma, [N, 1])
    for z_level in z_eval_levels:
        e = trajectories[tuple(z_level.flatten())]['env_z']
        trajectories[tuple(z_level.flatten())]['X'][:, 0], _ = e.reset(z=np.tile(z_level, (N, 1)), ux0=ux0)
    
    # take the first step
    ua0 = f_ua(N=N)
    #ua0 = np.random.uniform(0, 1, size=[N])
    for z_level in z_eval_levels:
        p = trajectories[tuple(z_level.flatten())]['policy_z']
        trajectories[tuple(z_level.flatten())]['A'][:, 0] = p.act(
                z=trajectories[tuple(z_level.flatten())]['Z'],
                xt=trajectories[tuple(z_level.flatten())]['X'][:, 0],
                xtm1=None,
                atm1=None,
                uat=ua0,
            )
    ur0 = f_ur(N=N)
    ux1 = f_ux(N=N, state_dim=state_dim)
    #ur0 = np.random.normal(0, sigma, [N, 1])
    #ux1 = np.random.normal(0, sigma, [N, 1])
    for z_level in z_eval_levels:
        e = trajectories[tuple(z_level.flatten())]['env_z']
        actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, 0] for i in range(N)]) 
        (
            trajectories[tuple(z_level.flatten())]['X'][:, 1], 
            trajectories[tuple(z_level.flatten())]['R'][:, 0], 
            _, 
            _
        ) = e.step(action=actions, uxt=ux1, urtm1=ur0)

    # take subsequent steps
    for t in range(1, T):
        uat = f_ua(N=N)
        #uat = np.random.uniform(0, 1, size=[N])
        for z_level in z_eval_levels:
            p = trajectories[tuple(z_level.flatten())]['policy_z']
            actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, t - 1] for i in range(N)]) 
            trajectories[tuple(z_level)]['A'][:, t] = p.act(
                    z=trajectories[tuple(z_level.flatten())]['Z'],
                    xt=trajectories[tuple(z_level.flatten())]['X'][:, t],
                    xtm1=trajectories[tuple(z_level.flatten())]['X'][:, t - 1],
                    atm1=actions,
                    uat=uat,
                )
        urtm1 = f_ur(N=N)
        uxt = f_ux(N=N, state_dim=state_dim)
        #urtm1 = np.random.normal(0, sigma, [N, 1])
        #uxt = np.random.normal(0, sigma, [N, 1])
        for z_level in z_eval_levels:
            e = trajectories[tuple(z_level.flatten())]['env_z']
            actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, t] for i in range(N)]) 
            (
                trajectories[tuple(z_level.flatten())]['X'][:, t + 1], 
                trajectories[tuple(z_level.flatten())]['R'][:, t], 
                _, 
                _
            ) = e.step(action=actions, uxt=uxt, urtm1=urtm1)

    # return the output
    return trajectories



class SimulatedEnvironment(gym.Env):
    def __init__(
        self,
        num_actions: int, 
        #reward_multiplication_factor: list | np.ndarray = [1.0, 1.0, 1.0],
        state_variance_factor: int | float = 1.0,
        z_factor: int | float = 0.0,
        trans_model_type: Literal["lm", "nn"] = "nn",
        trans_model_hidden_dims: list[int] = [32, 32],
        reward_model_type: Literal["lm", "nn"] = "nn",
        reward_model_hidden_dims: list[int] = [32, 32], 
        is_action_onehot: bool = True, 
        epochs: int = 1000,
        batch_size: int = 128,
        learning_rate: int | float = 0.001,
        is_early_stopping: bool = True,
        test_size: int | float = 0.2,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: int | float = 0.01,
        enforce_min_max: bool = False,
    ) -> None:
        self.action_space = np.array([a for a in range(num_actions)]).reshape(-1, 1)
        self.num_actions = num_actions
        #self.reward_multiplication_factor = reward_multiplication_factor
        self.state_variance_factor = state_variance_factor
        self.z_factor = z_factor
        self.trans_model_type = trans_model_type
        self.trans_model_hidden_dims = trans_model_hidden_dims
        self.reward_model_type = reward_model_type
        self.reward_model_hidden_dims = reward_model_hidden_dims
        self.is_action_onehot = is_action_onehot
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_early_stopping = is_early_stopping
        self.test_size = test_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.enforce_min_max = enforce_min_max
        self.is_trained = False
    
    @staticmethod
    def standardize(
            x: np.ndarray, 
            mean: np.ndarray | None = None, 
            std: np.ndarray |  None = None
        ) -> np.ndarray:
        if mean is None and std is None:
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            return (x - mean) / std, mean, std
        else:
            return (x - mean) / std

    @staticmethod
    def destandardize(
            x: np.ndarray, 
            mean: np.ndarray, 
            std: np.ndarray
        ) -> np.ndarray:
        return x * std + mean
    
    def encode_a(self, a: np.ndarray) -> np.ndarray:
        enc = OneHotEncoder(categories=[self.action_space.flatten()], drop=None)
        return enc.fit_transform(a.reshape(-1, 1)).toarray()

    def fit(
            self, 
            zs: list | np.ndarray, 
            states: list | np.ndarray, 
            actions: list | np.ndarray, 
            rewards: list | np.ndarray
        ) -> None:
        z = np.array(zs)
        xt = np.array(states)
        at = np.array(actions)
        rt = np.array(rewards)
        N, T, state_dim = xt.shape
        self.N = N
        self.T = T
        self.state_dim = state_dim

        # learn the initial state model i.e. t = 0
        # prepare data
        training_y = xt[:, 0, :]
        training_x_cat = zs
        training_x = training_x_cat
        self.initial_y_min = np.min(training_y, axis=0)
        self.initial_y_max = np.max(training_y, axis=0)

        self.initial_model_mean = {}
        self.initial_model_var = {}
        for z_ in np.unique(zs, axis=0):
            idx = np.all(training_x == z_, axis=1).flatten()
            training_y_z = training_y[idx]
            self.initial_model_mean[tuple(z_)] = np.mean(training_y_z, axis=0)
            self.initial_model_var[tuple(z_)] = np.var(training_y_z, axis=0)
        # REMARK: MIGHT NEED TO ADD BACK THE Z_FACTOR LINE SOMETIME LATER.
        #model_mean[1] = model_mean[1] + self.z_factor

        # learn the transition and reward models for t > 0
        # prepare input
        if self.is_action_onehot:
            actions = self.encode_a(actions.reshape(-1, 1)).reshape(self.N, T - 1, -1)
            dim_a = actions.shape[-1]
        else:
            actions = at
            dim_a = 1 # we require actions to be 1-dimensional w/out one-hot encoding
        actions = actions.reshape(-1, dim_a)
        next_states = xt[:, 1:T, :].reshape(N * (T - 1), -1)
        states = xt[:, 0 : (T - 1), :].reshape(N * (T - 1), -1)
        sensitives = np.repeat(z[:, np.newaxis, :], repeats=T - 1, axis=1).reshape(
            N * (T - 1), -1
        )
        rewards = rt.reshape(-1, 1)

        # fit the transition model
        #glogger.info("fitting transition model")
        if self.trans_model_type == "lm":
            self.trans_models = LinearRegressor()
            # THE FOLLOWING 2 LINES ARE NEWLY ADDED; FIT() HAS ALSO BEEN MODIFIED
            X = np.concatenate([states, sensitives, actions.reshape(-1, dim_a)], axis=1)
            y = next_states
            self.trans_models.fit(
                #states,
                #sensitives,
                #actions.reshape(-1, dim_a),
                #next_states,
                X,
                y, 
            )

        elif self.trans_model_type == "nn":
            # distinct models for different sensitive groups
            self.trans_models = {}
            for z_ in np.unique(zs, axis=0):
                idxs = np.all(sensitives == z_, axis=1).flatten()
                states_z_ = states[idxs]
                actions_z_ = actions[idxs]

                next_states_z_ = next_states[idxs]
                torch.manual_seed(1) # NEWLY ADDED
                np.random.seed(1) # NEWLY ADDED
                self.trans_models[tuple(z_)] = NeuralNetRegressor(
                    in_dim=states_z_.shape[1] + dim_a,
                    out_dim=next_states_z_.shape[1],
                    hidden_dims=self.trans_model_hidden_dims,
                )
                X = np.concatenate([states_z_, actions_z_.reshape(-1, dim_a)], axis=1)
                y = next_states_z_
                self.trans_models[tuple(z_)].fit(
                    X,
                    y,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    is_early_stopping=self.is_early_stopping,
                    test_size=self.test_size,
                    early_stopping_patience=self.early_stopping_patience,
                    early_stopping_min_delta=self.early_stopping_min_delta,
                )
            var_mean = 0 # CHANGED; NOT SURE ABOUT CORRECTNESS
            for z_ in np.unique(zs, axis=0):
                var_mean += self.trans_models[tuple(z_)].var
            var_mean = var_mean / np.unique(zs, axis=0).shape[0]
            for z_ in np.unique(zs, axis=0):
                self.trans_models[tuple(z_)].var = var_mean * self.state_variance_factor
                self.trans_models[tuple(z_)].mse = self.trans_models[tuple(z_)].var

        # reward model
        # input: x_t (weekly_pain_score, weekly_pain_interference), z (baseline characteristic), a_t
        # output: r_t
        #glogger.info("fitting reward model")
        if self.reward_model_type == "lm":
            self.reward_models = LinearRegressor()
            # THE FOLLOWING 2 LINES ARE NEWLY ADDED; FIT() HAS ALSO BEEN MODIFIED
            X = np.concatenate([states, sensitives, actions.reshape(-1, dim_a)], axis=1)
            y = rewards.reshape(-1, 1)
            self.reward_models.fit(
                #states,
                #sensitives,
                #actions.reshape(-1, dim_a),
                #rewards,
                X, 
                y, 
            )
        elif self.reward_model_type == "nn":
            self.reward_models = {}
            for z_ in np.unique(zs, axis=0):
                idxs = np.all(sensitives == z_, axis=1).flatten()
                states_z_ = states[idxs]
                actions_z_ = actions[idxs]
                rewards_z_ = rewards[idxs]
                torch.manual_seed(1) # NEWLY ADDED
                np.random.seed(1) # NEWLY ADDED
                self.reward_models[tuple(z_)] = NeuralNetRegressor(
                    in_dim=states_z_.shape[1] + dim_a,
                    out_dim=1,
                    hidden_dims=self.reward_model_hidden_dims,
                )
                X = np.concatenate([states_z_, actions_z_.reshape(-1, dim_a)], axis=1)
                y = rewards_z_.reshape(-1, 1)
                self.reward_models[tuple(z_)].fit(
                    X,
                    y,
                    epochs=self.epochs,
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    is_early_stopping=self.is_early_stopping,
                    test_size=self.test_size,
                    early_stopping_patience=self.early_stopping_patience,
                    early_stopping_min_delta=self.early_stopping_min_delta,
                )
            var_mean = 0 # CHANGED; NOT SURE ABOUT CORRECTNESS
            for z_ in np.unique(zs, axis=0):
                var_mean += self.reward_models[tuple(z_)].var
            var_mean = var_mean / np.unique(zs, axis=0).shape[0]
            for z_ in np.unique(zs, axis=0):
                self.reward_models[tuple(z_)].var = var_mean
                self.reward_models[tuple(z_)].mse = self.trans_models[tuple(z_)].var

        # force min and max for states and rewards
        if self.enforce_min_max:
            self.states_max = np.max(states, axis=0)
            self.states_min = np.min(states, axis=0)
            self.rewards_max = np.max(rewards)
            self.rewards_min = np.min(rewards)
            self.next_states_max = np.max(next_states, axis=0)
            self.next_states_min = np.min(next_states, axis=0)
        
        self.is_trained = True
        # temporarily store the state, action, and reward data of the environment
        self.xt = np.zeros(xt.shape[-1])
        self.atm1 = None
        self.rtm1 = 0
    
    def reset(
        self, 
        z: list | np.ndarray, 
        seed: int = 1,
        errors_states: np.ndarray | None = None, 
        enforce_min_max: bool = False,
        #z_factor=0.0,
    ) -> tuple[np.ndarray, None]:
        zs = np.array(z)
        if errors_states is not None:
            errors_states = np.array(errors_states)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if not self.is_trained:
            print('Cannot make transitions because the environment is not yet trained.')
            exit(1)

        testing_x_cat = zs
        testing_x = testing_x_cat
        self.zs = zs

        testing_y = np.zeros([testing_x.shape[0], self.state_dim])
        if errors_states is None:
            errors_states = np.random.multivariate_normal(
                mean=np.zeros(self.state_dim), 
                cov=np.diag(np.ones(self.state_dim)), 
                size=testing_x.shape[0]
            )
        for z_ in np.unique(zs, axis=0):
            idx = np.all(testing_x == z_, axis=1).flatten()
            testing_y[idx] = (
                np.repeat(self.initial_model_mean[tuple(z_)][np.newaxis, :], np.sum(idx), axis=0) + errors_states[idx]
            )

        # cliping
        if enforce_min_max:
            for i in range(self.state_dim):
                testing_y[:, :, i] = np.clip(
                    testing_y[:, :, i], a_min=self.initial_y_min[i], a_max=self.initial_y_max[i]
                )
        
        self.xt = testing_y
        observation = self.xt

        return observation, None
    
    # helper function
    def _next_state_reward_mean(
            self, 
            z: list | np.ndarray, 
            xt: list | np.ndarray, 
            at: list | np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
        zs = np.array(z)
        xt = np.array(xt)
        at = np.array(at)
        N = xt.shape[0]
        states = xt.reshape(N, -1)
        actions = at.flatten()
        next_states = np.zeros([N, xt.shape[-1]])
        rewards = np.zeros([N])

        if self.is_action_onehot:
            actions = self.encode_a(actions.reshape(-1, 1)).reshape(N, -1)
            dim_a = actions.shape[-1]
        else:
            dim_a = 1

        # next state
        if self.trans_model_type == "lm":
            #next_states = self.trans_models.predict(states, sensitives, actions)
            # THE FOLLOWING 2 LINES ARE NEWLY ADDED
            X = np.concatenate(
                    [states, zs, actions.reshape(-1, dim_a)], axis=1
                )
            next_states = self.trans_models.predict(X)
            #next_states = self.trans_models.predict(states, zs, actions)

        elif self.trans_model_type == "nn":
            for z_ in np.unique(zs, axis=0):
                idx = np.all(zs == z_, axis=1).flatten()
                states_z_ = states[idx]
                actions_z_ = actions[idx]
                X = np.concatenate(
                    [states_z_, actions_z_.reshape(-1, dim_a)], axis=1
                )

                torch.manual_seed(1) # NEWLY ADDED
                np.random.seed(1) # NEWLY ADDED
                next_states_z_mean = self.trans_models[tuple(z_)].predict(X)
                # MAY NEED TO UNCOMMENT BACK
                #if z_ == 1:
                #if z_ == np.array([1]):
                #    next_states_z_mean = next_states_z_mean + self.z_factor
                next_states_z_ = next_states_z_mean
                next_states[idx] = next_states_z_

        # reward
        if self.reward_model_type == "lm":
            #rewards = self.reward_models.predict(states, sensitives, actions).flatten()
            # THE FOLLOWING 2 LINES ARE NEWLY ADDED
            X = np.concatenate(
                    [states, zs, actions.reshape(-1, dim_a)], axis=1
                )
            rewards = self.reward_models.predict(X).flatten()
            #rewards = self.reward_models.predict(states, zs, actions).flatten()
        else:
            for z_ in np.unique(zs, axis=0):
                idx = np.all(zs == z_, axis=1).flatten()
                states_z_ = states[idx]
                actions_z_ = actions[idx]
                X = np.concatenate(
                    [states_z_, actions_z_.reshape(-1, dim_a)], axis=1
                )
                torch.manual_seed(1) # NEWLY ADDED
                np.random.seed(1) # NEWLY ADDED
                rewards_z_ = self.reward_models[tuple(z_)].predict(X).flatten()
                rewards[idx] = rewards_z_

        if self.enforce_min_max:
            next_states = np.clip(
                next_states, a_min=self.next_states_min, a_max=self.next_states_max
            )
            rewards = np.clip(rewards, a_min=self.rewards_min, a_max=self.rewards_max)

        return next_states, rewards
    
    def step(
            self, 
            action: list | np.ndarray, 
            errors_states: np.ndarray | None = None, 
            errors_rewards: np.ndarray | None = None, 
            seed: int = 1
        ) -> tuple[np.ndarray, np.ndarray, Literal[False], Literal[False]]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if not self.is_trained:
            print('Cannot make transitions because the environment is not yet trained.')
            exit(1)

        at = np.array(action)
        xt = self.xt
        next_states_mean, rewards_mean = self._next_state_reward_mean(
            z=self.zs, xt=xt, at=at
        )

        if errors_states is not None:
            errors_states = np.array(errors_states)
        if errors_rewards is not None:
            errors_rewards = np.array(errors_rewards)
        
        if errors_states is None:
            errors_states = np.random.multivariate_normal(
                mean=np.zeros(xt.shape[-1]),
                cov=np.diag(np.ones(xt.shape[-1])), 
                size=xt.shape[0],
            )
        if errors_rewards is None:
            errors_rewards = np.random.normal(
                loc=0, scale=1, size=xt.shape[0]
            )

        next_states = next_states_mean + errors_states
        rewards = rewards_mean + errors_rewards

        self.atm1 = action
        self.xt = next_states
        self.rtm1 = rewards

        # form the current observation and compute the reward
        observation = self.xt
        reward = self.rtm1

        # the simulated enviornment never terminates or gets truncated
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated
    


def f_errors_states(size: int) -> np.ndarray:
    return np.random.multivariate_normal(
                mean=np.zeros(size[1]),
                cov=np.diag(np.ones(size[1])), 
                size=size[0],
            )

def f_errors_rewards(size: int) -> np.ndarray:
    return np.random.normal(
                loc=0, scale=1, size=size[0]
            )

# REQUIRES: zs should be the same as the zs passed to the environment
def sample_simulated_env_trajectory(
        env: SimulatedEnvironment, 
        zs: list | np.ndarray, 
        state_dim: int, 
        T: int, 
        policy: Agent, 
        f_errors_states: Callable[[int], np.ndarray] = f_errors_states, 
        f_errors_rewards: Callable[[int], np.ndarray] = f_errors_rewards, 
        seed: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # initialize containers to store the trajectory
    zs = np.array(zs)
    N = zs.shape[0]
    Z = zs.reshape(N, -1)
    X = np.zeros([N, T + 1, state_dim], dtype=float)
    A = np.zeros([N, T], dtype=int)
    R = np.zeros([N, T], dtype=float)
    np.random.seed(seed)

    '''# SHOULD DELETE RELATED COMPONENTS
    policy_c = copy.deepcopy(policy)
    X_c = np.zeros([N, T + 1, state_dim], dtype=float)
    A_c = np.zeros([N, T], dtype=int)
    R_c = np.zeros([N, T], dtype=float)
    L_c = np.zeros([N, T, 2], dtype=float)'''

    # generate the initial state
    errors_states = f_errors_states([N, state_dim])
    X[:, 0], _ = env.reset(z=Z, errors_states=errors_states)
    
    # take the first step
    ua0 = np.random.uniform(0, 1, size=[N])
    A[:, 0] = policy.act(
            z=Z,
            xt=X[:, 0],
            xtm1=None,
            atm1=None,
            uat=ua0,
        )
    '''A_c[:, 0] = policy_c.act(
            z=Z,
            xt=X_c[:, 0],
            #atm=A_c[:, :0],
            #t=0,
            xtm1=None,
            atm1=None,
            uat=ua0,
            is_return_prob=False,
        ) # SHOULD DELETE LATER'''
    errors_states = f_errors_states([N, state_dim])
    errors_rewards = f_errors_rewards([N])
    X[:, 1], R[:, 0], _, _ = env.step(action=A[:, 0], 
                                      errors_states=errors_states, 
                                      errors_rewards=errors_rewards)

    # take subsequent steps
    for t in range(1, T):
        uat = np.random.uniform(0, 1, size=[N])
        #uat = np.ones(N)
        A[:, t] = policy.act(
                z=Z,
                xt=X[:, t],
                xtm1=X[:, t - 1],
                atm1=A[:, t - 1],
                uat=uat,
            )
        '''A_c[:, t] = policy_c.act(
                z=Z,
                xt=X_c[:, t],
                #atm=A_c[:, :t],
                #t=t,
                xtm1=X_c[:, t - 1],
                atm1=A_c[:, t - 1],
                uat=uat,
                is_return_prob=False,
            ) # SHOULD DELETE LATER'''
        #urtm1 = np.random.normal(0, sigma, [N, 1])
        #uxt = np.random.normal(0, sigma, [N, 1])
        #uxt = np.zeros((N, 1))
        #urtm1 = np.zeros((N, 1))
        errors_states = f_errors_states([N, state_dim])
        errors_rewards = f_errors_rewards([N])
        X[:, t + 1], R[:, t], _, _ = env.step(action=A[:, t], 
                                              errors_states=errors_states, 
                                              errors_rewards=errors_rewards)

    # prepare and return the output
    #out = [Z, X, A, R]
    return Z, X, A, R



def estimate_counterfactual_trajectories_from_data(
        env: SimulatedEnvironment, 
        zs: list | np.ndarray, 
        states: list | np.ndarray, 
        actions: list | np.ndarray, 
        policy: Agent, 
        f_ua: Callable[[int], np.ndarray] = f_ua, 
        seed: int = 1
    ) -> dict[tuple[Union[int, float], ...], dict[str, Union[np.ndarray, SyntheticEnvironment, Agent]]]:
    zs = np.array(zs)
    N = actions.shape[0]
    T = actions.shape[1]
    state_dim = states.shape[2]
    np.random.seed(seed)
    z_eval_levels = np.unique(zs, axis=0)
    z_eval_levels = np.array(z_eval_levels)

    # dictionary to contain the counterfactual trajectories; key = z and value = trajectory
    trajectories = {}
    for z_level in z_eval_levels:
        env_z = copy.deepcopy(env)
        policy_z = copy.deepcopy(policy)
        Z = np.repeat(np.array([z_level]), repeats=N, axis=0)
        X = np.zeros([N, T + 1, state_dim], dtype=float)
        A = np.zeros([N, T + 1], dtype=int)
        R = np.zeros([N, T], dtype=float)
        trajectories[tuple(z_level.flatten())] = {'Z': Z, 'X': X, 'A': A, 'R': R, 
                                        'env_z': env_z, 'policy_z': policy_z}

    # generate the initial state
    for z_level in z_eval_levels:
        e = trajectories[tuple(z_level.flatten())]['env_z']
        for z_obs in np.unique(zs, axis=0):
            idx = np.all(zs == z_obs, axis=1).flatten()
            (
                trajectories[tuple(z_level.flatten())]['X'][idx, 0]
            ) = (states[idx, 0] 
                 - e.initial_model_mean[tuple(z_obs.flatten())] 
                 + e.initial_model_mean[tuple(z_level.flatten())])
    
    # take the first step
    ua0 = f_ua(N=N)
    #ua0 = np.random.uniform(0, sigma_a, size=[N])
    for z_level in z_eval_levels:
        p = trajectories[tuple(z_level.flatten())]['policy_z']
        trajectories[tuple(z_level.flatten())]['A'][:, 0] = p.act(
                z=trajectories[tuple(z_level.flatten())]['Z'],
                xt=trajectories[tuple(z_level.flatten())]['X'][:, 0],
                xtm1=None,
                atm1=None,
                uat=ua0,
            )
    for z_level in z_eval_levels:
        e = trajectories[tuple(z_level.flatten())]['env_z']
        for z_obs in np.unique(zs, axis=0):
            idx = np.all(zs == z_obs, axis=1).flatten()
            obs_mean, _ = e._next_state_reward_mean(
                        z=zs[idx],
                        xt=states[idx, 0, :],
                        at=actions[idx, 0],
                        )
            cf_mean, _ = e._next_state_reward_mean(
                        z=trajectories[tuple(z_level.flatten())]['Z'][idx],
                        xt=trajectories[tuple(z_level.flatten())]['X'][idx, 0, :],
                        at=actions[idx, 0],
                        )
            (
                trajectories[tuple(z_level.flatten())]['X'][idx, 1]
            ) = (states[idx, 1] - obs_mean + cf_mean)

    # take subsequent steps
    for t in range(1, T):
        uat = f_ua(N=N)
        #uat = np.random.uniform(0, sigma_a, size=[N])
        for z_level in z_eval_levels:
            p = trajectories[tuple(z_level.flatten())]['policy_z']
            trajectories[tuple(z_level.flatten())]['A'][:, t] = p.act(
                    z=trajectories[tuple(z_level.flatten())]['Z'],
                    xt=trajectories[tuple(z_level.flatten())]['X'][:, t],
                    xtm1=trajectories[tuple(z_level.flatten())]['X'][:, t - 1],
                    atm1=actions[:, t - 1],
                    uat=uat,
                )
        for z_level in z_eval_levels:
            e = trajectories[tuple(z_level.flatten())]['env_z']
            for z_obs in np.unique(zs, axis=0):
                idx = np.all(zs == z_obs, axis=1).flatten()
                obs_mean, _ = e._next_state_reward_mean(
                        z=zs[idx],
                        xt=states[idx, t, :],
                        at=actions[idx, t],
                        )
                cf_mean, _ = e._next_state_reward_mean(
                        z=trajectories[tuple(z_level.flatten())]['Z'][idx],
                        xt=trajectories[tuple(z_level.flatten())]['X'][idx, t, :],
                        at=actions[idx, t],
                        )
                (
                    trajectories[tuple(z_level.flatten())]['X'][idx, t + 1]
                ) = (states[idx, t + 1] - obs_mean + cf_mean)
    
    # take the final step
    uaT = f_ua(N=N)
    #ua0 = np.random.uniform(0, sigma_a, size=[N])
    for z_level in z_eval_levels:
        p = trajectories[tuple(z_level.flatten())]['policy_z']
        trajectories[tuple(z_level.flatten())]['A'][:, T] = p.act(
                z=trajectories[tuple(z_level.flatten())]['Z'],
                xt=trajectories[tuple(z_level.flatten())]['X'][:, T],
                xtm1=trajectories[tuple(z_level.flatten())]['X'][:, T - 1],
                atm1=actions[:, T - 1],
                uat=uaT,
            )
    
    return trajectories