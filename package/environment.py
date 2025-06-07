import gymnasium as gym
import numpy as np
import copy
import torch
from sklearn.preprocessing import OneHotEncoder
from utils.base_models import NeuralNetRegressor, LinearRegressor

# Jitao's original
def f_x0(zs, ux0, z_coef=1):
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

def f_xt(zs, xtm1, atm1, uxt, z_coef=1):
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

def f_rt(zs, xt, at, urt, z_coef=1):
    lmbda = np.array([-0.3, 0.3, 0.5 * z_coef, 0.5, 0.2 * z_coef, 0.7, -1.0 * z_coef])
    n = xt.shape[0]
    at = at.reshape(-1, 1)
    M = np.concatenate(
        [np.ones([n, 1]), xt, zs, at, xt * zs, xt * at, zs * at], axis=1
    )
    rt = M @ lmbda
    return rt



class SyntheticEnvironment(gym.Env):
    def __init__(self, state_dim=1, z_coef=1, f_x0=f_x0, f_xt=f_xt, f_rt=f_rt):
        # initialize member variables from the parameter list
        #self.N = zs.shape[0]
        #self.zs = zs.reshape(self.N, -1)
        self.state_dim = state_dim
        #self.seed = seed
        #self.sigma = sigma
        #self.num_z_levels = num_z_levels
        #self.num_action_levels = num_action_levels
        self.z_coef = z_coef
        self.f_x0 = f_x0
        self.f_xt = f_xt
        self.f_rt = f_rt
        #self.N = 1 # will be updated in reset()

        # store the sensitive attribute of each individual
        #self.zs = None # will be updated in reset()

        # temporarily store the state, action, and reward data of the environment
        self.xt = np.zeros(state_dim)
        self.atm1 = None
        self.rtm1 = 0
        #self.cumulative_reward = 0

        '''# define the observation and action spaces
        self.observation_space = gym.spaces.Dict(
            {
                "zs": gym.spaces.Discrete(self.num_z_levels),
                "xt": gym.spaces.Box(-np.inf, np.inf, shape=self.state_dim, dtype=int),
            }
        )
        self.action_space = gym.spaces.Discrete(self.num_action_levels)'''

    '''def get_sensitive_attribute(self):
        return self.zs'''
    
    '''def get_current_state(self):
        return self.xt'''
    
    '''def _get_obs(self):
        return {'zs': self.zs, 'xt': self.xt}'''
    
    '''def get_cumulative_reward(self):
        return self.cumulative_reward'''
    
    '''def reset_seed(self):
        np.random.seed(self.seed)'''
    
    def get_z_coef(self):
        return self.z_coef
    
    def set_z_coef(self, new_z_coef):
        self.z_coef = new_z_coef
    
    def reset(self, zs, ux0):
        self.N = zs.shape[0] # number of samples/individuals
        self.zs = zs.reshape(self.N, -1)
        #np.random.seed(self.seed)

        # generate initial state
        #ux0 = np.random.normal(0, self.sigma, [self.N, 1])
        self.xt = self.f_x0(zs=self.zs, ux0=ux0, z_coef=self.z_coef)

        # form the current observation
        observation = self.xt

        return observation, None
    
    def step(self, action, uxt, urt):
        self.atm1 = action
        #np.random.seed(self.seed) # ALSO NEED SEED HERE OR NOT?

        # generate next state and reward
        #urt = np.random.normal(0, self.sigma, [self.N, 1])
        self.rtm1 = self.f_rt(zs=self.zs, xt=self.xt, at=self.atm1, urt=urt, z_coef=self.z_coef)

        #uxt = np.random.normal(0, self.sigma, [self.N, 1])
        self.xt = self.f_xt(zs=self.zs, xtm1=self.xt, atm1=self.atm1, uxt=uxt, z_coef=self.z_coef)
        
        '''#urt = np.random.normal(0, self.sigma, [self.N, 1])
        self.rtm1 = self.f_rt(self.zs, self.xt, self.atm1, urt, self.z_coef)'''

        #self.atm1 = action

        # form the current observation and compute the reward
        #self.cumulative_reward += self.rtm1
        observation = self.xt
        reward = self.rtm1

        # the synthetic enviornment never terminates or gets truncated
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated



# REQUIRES: zs should be the same as the zs passed to the environment
def sample_trajectory(env, zs, state_dim, N, T, sigma=1, seed=1, policy=None):
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
    ux0 = np.random.normal(0, sigma, [N, 1])
    #ux0 = np.ones((N, 1))
    X[:, 0], _ = env.reset(Z, ux0)
    
    # take the first step
    ua0 = np.random.uniform(0, 1, size=[N])
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
    ur0 = np.random.normal(0, sigma, [N, 1])
    ux1 = np.random.normal(0, sigma, [N, 1])
    #ux1 = np.zeros((N, 1))
    #ur0 = np.ones((N, 1))
    X[:, 1], R[:, 0], _, _ = env.step(A[:, 0], ux1, ur0)

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
        urtm1 = np.random.normal(0, sigma, [N, 1])
        uxt = np.random.normal(0, sigma, [N, 1])
        #uxt = np.zeros((N, 1))
        #urtm1 = np.zeros((N, 1))
        X[:, t + 1], R[:, t], _, _ = env.step(A[:, t], uxt, urtm1)

    # prepare and return the output
    out = [Z, X, A, R]
    return out



def sample_counterfactual_trajectories(env, zs, z_eval_levels, state_dim, N, T, 
                                       policy, sigma=1, seed=1):
    np.random.seed(seed)
    z_eval_levels = np.array(z_eval_levels)
    #print('Observed Z: ', zs, sep='\n')
    #print('\n')

    # dictionary to contain the counterfactual trajectories; key = z and value = trajectory
    trajectories = {}
    for z_level in z_eval_levels:
        env_z = copy.deepcopy(env)
        policy_z = copy.deepcopy(policy)
        #Z = np.full(N, [z_level]).reshape(-1, 1)
        Z = np.repeat(z_level, repeats=N, axis=0).reshape(N, -1) # RECENTLY CHANGED
        X = np.zeros([N, T + 1, state_dim], dtype=float)
        A = np.zeros([N, T], dtype=int)
        R = np.zeros([N, T], dtype=float)
        trajectories[tuple(z_level.flatten())] = {'Z': Z, 'X': X, 'A': A, 'R': R, 
                                        'env_z': env_z, 'policy_z': policy_z}
    #print('Initialized dict:', trajectories, sep='\n')
    #print('\n')

    # generate the initial state
    #print('t = 0: ')
    ux0 = np.random.normal(0, sigma, [N, 1])
    for z_level in z_eval_levels:
        e = trajectories[tuple(z_level.flatten())]['env_z']
        trajectories[tuple(z_level.flatten())]['X'][:, 0], _ = e.reset(np.tile(z_level, (N, 1)), ux0)
    
    # take the first step
    ua0 = np.random.uniform(0, 1, size=[N])
    for z_level in z_eval_levels:
        p = trajectories[tuple(z_level.flatten())]['policy_z']
        trajectories[tuple(z_level.flatten())]['A'][:, 0] = p.act(
                z=trajectories[tuple(z_level.flatten())]['Z'],
                xt=trajectories[tuple(z_level.flatten())]['X'][:, 0],
                xtm1=None,
                atm1=None,
                uat=ua0,
            )
    ur0 = np.random.normal(0, sigma, [N, 1])
    ux1 = np.random.normal(0, sigma, [N, 1])
    for z_level in z_eval_levels:
        e = trajectories[tuple(z_level.flatten())]['env_z']
        '''actions = np.zeros(N)
        for i in range(N):
            actions[i] = trajectories[tuple(zs[i])]['A'][i, 0]
            #actions[i] = trajectories[tuple(z_level)]['A'][i, 0]'''
        actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, 0] for i in range(N)]) 
        #print('a0: ', actions, sep='\n')
        (
            trajectories[tuple(z_level.flatten())]['X'][:, 1], 
            trajectories[tuple(z_level.flatten())]['R'][:, 0], 
            _, 
            _
        ) = e.step(actions, ux1, ur0)
    #print(trajectories)
    #print('\n')

    # take subsequent steps
    for t in range(1, T):
        #print('t =', t, sep=' ')
        uat = np.random.uniform(0, 1, size=[N])
        for z_level in z_eval_levels:
            p = trajectories[tuple(z_level.flatten())]['policy_z']
            '''actions = np.zeros(N)
            for i in range(N):
                actions[i] = trajectories[tuple(zs[i])]['A'][i, t - 1]
                #actions[i] = trajectories[tuple(z_level)]['A'][i, t - 1]'''
            actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, t - 1] for i in range(N)]) 
            trajectories[tuple(z_level)]['A'][:, t] = p.act(
                    z=trajectories[tuple(z_level.flatten())]['Z'],
                    xt=trajectories[tuple(z_level.flatten())]['X'][:, t],
                    xtm1=trajectories[tuple(z_level.flatten())]['X'][:, t - 1],
                    #atm1=trajectories[tuple(z_level)]['A'][:, t - 1],
                    atm1=actions,
                    uat=uat,
                )
        urtm1 = np.random.normal(0, sigma, [N, 1])
        uxt = np.random.normal(0, sigma, [N, 1])
        for z_level in z_eval_levels:
            #print('t =', t, 'z_level =', z_level, sep=' ')
            e = trajectories[tuple(z_level.flatten())]['env_z']
            '''actions = np.zeros(N)
            for i in range(N):
                actions[i] = trajectories[tuple(zs[i])]['A'][i, t]
                #actions[i] = trajectories[tuple(z_level)]['A'][i, t]'''
            actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, t] for i in range(N)]) 
            #print('atm1: ', actions, sep='\n')
            (
                trajectories[tuple(z_level.flatten())]['X'][:, t + 1], 
                trajectories[tuple(z_level.flatten())]['R'][:, t], 
                _, 
                _
            ) = e.step(actions, uxt, urtm1)
        #print(trajectories)
        #print('\n')
    
    '''out = {}
    for z_level in z_eval_levels:
        t = Trajectory()
        t.read_trajectory_from_arrays()'''

    # return the output
    return trajectories



class SimulatedEnvironment(gym.Env):
    def __init__(
        self,
        #zvars_cat,
        #z_levels, 
        #xvars,
        action_space, 
        reward_multiplication_factor=[1.0, 1.0, 1.0],
        state_variance_factor=1.0,
        z_factor=0.0,
        trans_model_type="nn",
        trans_model_hidden_dims=[32, 32],
        reward_model_type="nn",
        reward_model_hidden_dims=[32, 32], 
        is_action_onehot=True, 
        epochs=1000,
        batch_size=128,
        learning_rate=0.001,
        is_early_stopping=True,
        test_size=0.2,
        early_stopping_patience=10,
        early_stopping_min_delta=0.01,
        enforce_min_max=False,
        #include_week=False,
        #encode_action=False,
        ):
        self.action_space = action_space
        self.reward_multiplication_factor = reward_multiplication_factor
        self.state_variance_factor = state_variance_factor
        self.z_factor = z_factor
        #self.zvars_cat = zvars_cat
        #self.z_levels = z_levels
        #self.xvars = xvars
        self.trans_model_type = trans_model_type
        self.trans_model_hidden_dims = trans_model_hidden_dims
        self.reward_model_type = reward_model_type
        self.reward_model_hidden_dims = reward_model_hidden_dims
        self.is_action_onehot = is_action_onehot
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_early_stopping = is_early_stopping
        self.test_size = test_size # IS THIS PARAMETER USED?
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.enforce_min_max = enforce_min_max
        self.is_trained = False
        #self.include_week = include_week
        #self.encode_action = encode_action
        '''if self.trans_model_type == "lm":
            self.enc = self.make_encoder(self.zvars_cat, drop_first=True)
        else:
            self.enc = self.make_encoder(self.zvars_cat, drop_first=True)'''
    
    @staticmethod
    def standardize(x, mean=None, std=None):
        if mean is None and std is None:
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            return (x - mean) / std, mean, std
        else:
            return (x - mean) / std

    @staticmethod
    def destandardize(x, mean, std):
        return x * std + mean
    
    def encode_a(self, a):
        enc = OneHotEncoder(categories=[self.action_space.flatten()], drop=None)
        return enc.fit_transform(a.reshape(-1, 1)).toarray()

    def fit(self, zs, states, actions, rewards):
        #xt, at, rt, z, weeks = self.load_training_data()
        z = zs
        xt = states
        at = actions
        rt = rewards
        N, T, state_dim = xt.shape
        self.N = N
        self.T = T
        self.state_dim = state_dim
        #dim_a = at.shape[-1]
        #dim_a = 1 # we require actions to be 1-dimensional

        # learn the initial state model i.e. t = 0
        # prepare data
        '''training_y = impute_data[xvars].to_numpy()
        training_x_cat = impute_data[zvars_cat]'''
        training_y = xt[:, 0, :]
        training_x_cat = zs

        #enc = PowerEDEnv.make_encoder(zvars_cat, drop_first=True)
        # training_x_cat_encoded = enc.fit_transform(training_x_cat).toarray()
        #training_x = training_x_cat.to_numpy().reshape(-1, 1)
        training_x = training_x_cat
        #testing_x_cat = baseline_characteristics[zvars_cat]
        # testing_x_cat_encoded = enc.fit_transform(testing_x_cat).toarray()
        #testing_x = testing_x_cat.to_numpy().reshape(-1, 1)
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
        #actions = at[:, 1:T].reshape(-1, dim_a)
        if self.is_action_onehot:
            actions = self.encode_a(actions.reshape(-1, 1)).reshape(self.N, T - 1, -1)
            dim_a = actions.shape[-1]
        else:
            actions = at
            dim_a = 1 # we require actions to be 1-dimensional w/out one-hot encoding
        actions = actions.reshape(-1, dim_a)
        next_states = xt[:, 1:T, :].reshape(N * (T - 1), -1)
        '''if self.include_week:
            states = np.concatenate(
                [
                    xt[:, 0 : (T - 1), :],
                    weeks[:, 0 : (T - 1), np.newaxis],
                ],
                axis=2,
            ).reshape(N * (T - 1), -1)
        else:
            states = xt[:, 0 : (T - 1), :].reshape(N * (T - 1), -1)'''
        states = xt[:, 0 : (T - 1), :].reshape(N * (T - 1), -1)
        sensitives = np.repeat(z[:, np.newaxis, :], repeats=T - 1, axis=1).reshape(
            N * (T - 1), -1
        )
        # print(z)
        #rewards = rt[:, 1:T].reshape(-1, 1)
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
                # print(sensitives)
                # print(sensitives.shape, idxs.shape, states_normalized.shape)
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

                # save ,load and train the model
                '''if folder is not None:
                    model_filename = os.path.join(
                        folder,
                        "models/powered_trans_model_{}_z{}.pt".format(
                            self.trans_model_type, z_
                        ),
                    )

                    if os.path.exists(model_filename):
                        self.trans_models[z_] = torch.load(model_filename)
                    else:
                        self.trans_models[z_].train(
                            X,
                            y,
                            epochs=epochs,
                            batch_size=128,
                            learning_rate=lr,
                            is_early_stopping=True,
                            test_size=0.2,
                            early_stopping_patience=10,
                            early_stopping_min_delta=0.01,
                        )
                        torch.save(self.trans_models[z_], model_filename)
                else:
                    self.trans_models[z_].train(
                        X,
                        y,
                        epochs=epochs,
                        batch_size=128,
                        learning_rate=lr,
                        is_early_stopping=True,
                        test_size=0.2,
                        early_stopping_patience=10,
                        early_stopping_min_delta=0.01,
                    )'''
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
            # print(self.trans_models[0].var, self.trans_models[1].var)
            # raise
            '''var_mean = (self.trans_models[0].var + self.trans_models[1].var) / 2
            self.trans_models[0].var = var_mean * self.state_variance_factor
            self.trans_models[1].var = var_mean * self.state_variance_factor
            self.trans_models[0].mse = self.trans_models[0].var
            self.trans_models[1].mse = self.trans_models[1].var'''
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
                '''if folder is not None:
                    model_filename = os.path.join(
                        folder,
                        "models/powered_reward_model_{}_{}.pt".format(
                            self.reward_model_type, z_
                        ),
                    )
                    if os.path.exists(model_filename):
                        self.reward_models[z_] = torch.load(model_filename)
                    else:
                        self.reward_models[z_].train(
                            X,
                            y,
                            epochs=epochs,
                            learning_rate=lr,
                            batch_size=128,
                            is_early_stopping=True,
                            test_size=0.2,
                            early_stopping_patience=10,
                            early_stopping_min_delta=0.01,
                        )
                        torch.save(self.reward_models[z_], model_filename)
                else:
                    self.reward_models[z_].train(
                        X,
                        y,
                        epochs=epochs,
                        learning_rate=lr,
                        batch_size=128,
                        is_early_stopping=True,
                        test_size=0.2,
                        early_stopping_patience=10,
                        early_stopping_min_delta=0.01,
                    )'''
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
            '''var_mean = (self.reward_models[0].var + self.reward_models[1].var) / 2
            self.reward_models[0].var = var_mean
            self.reward_models[1].var = var_mean
            self.reward_models[0].mse = self.reward_models[0].var
            self.reward_models[1].mse = self.reward_models[1].var'''
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
        zs, 
        seed=1,
        enforce_min_max=False,
        errors=None, 
        #z_factor=0.0,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if not self.is_trained:
            print('Cannot make transitions because the environment is not yet trained.')
            exit(1)

        #enc = PowerEDEnv.make_encoder(zvars_cat, drop_first=True)
        # training_x_cat_encoded = enc.fit_transform(training_x_cat).toarray()
        testing_x_cat = zs
        # testing_x_cat_encoded = enc.fit_transform(testing_x_cat).toarray()
        testing_x = testing_x_cat
        self.zs = zs

        # print(training_y_z)
        testing_y = np.zeros([testing_x.shape[0], self.state_dim])
        if errors is None:
            errors = np.random.multivariate_normal(
                mean=np.zeros(self.state_dim), 
                #cov=np.diag(self.initial_model_var[tuple(zs[0].flatten())]), 
                cov=np.diag(np.ones(self.state_dim)), 
                size=testing_x.shape[0]
            )
        for z_ in np.unique(zs, axis=0):
            idx = np.all(testing_x == z_, axis=1).flatten()
            testing_y[idx] = (
                np.repeat(self.initial_model_mean[tuple(z_)][np.newaxis, :], np.sum(idx), axis=0) + errors[idx]
            )

        '''if len(xvars) >= 1:
            weekly_pain_score = testing_y[:, 0]
        if len(xvars) >= 2:
            weekly_pain_interference = testing_y[:, 1]'''

        # cliping
        if enforce_min_max:
            '''weekly_pain_score = np.clip(weekly_pain_score, a_min=y_min[0], a_max=y_max[0])
            weekly_pain_interference = np.clip(
                weekly_pain_interference, a_min=y_min[1], a_max=y_max[1]
            )'''
            for i in range(self.state_dim):
                testing_y[:, :, i] = np.clip(
                    testing_y[:, :, i], a_min=self.initial_y_min[i], a_max=self.initial_y_max[i]
                )
        '''if len(xvars) == 1:
            state0 = pd.DataFrame({"weekly_pain_score": weekly_pain_score})
        elif len(xvars) == 2:
            state0 = pd.DataFrame(
                {
                    "weekly_pain_score": weekly_pain_score,
                    "weekly_pain_interference": weekly_pain_interference,
                }
            )'''
        
        self.xt = testing_y
        observation = self.xt

        return observation, None
    
    # helper function
    def _next_state_reward_mean(self, zs, xt, at):
        N = xt.shape[0]

        #sensitives = self.encode_z(z_cat).reshape(N, -1)
        '''if self.include_week:
            states = np.concatenate(
                [
                    xt.reshape(N, -1),
                    weeks.reshape(N, 1),
                ],
                axis=1,
            )
        else:
            states = xt.reshape(N, -1)'''
        states = xt.reshape(N, -1)
        '''if self.encode_action:
            actions = self.encode_a(at)
        else:
            actions = at.flatten()'''
        actions = at.flatten()
        next_states = np.zeros([N, xt.shape[-1]])
        rewards = np.zeros([N])

        # standarize
        # states_normalized = self.standardize(states, self.states_mean, self.states_std)

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
                #idx = (sensitives == z_).flatten()
                idx = np.all(zs == z_, axis=1).flatten()
                states_z_ = states[idx]
                actions_z_ = actions[idx]
                #X = np.concatenate(
                #    [states_z_, actions_z_.reshape(-1, actions_z_.shape[-1])], axis=1
                #)
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
            # for a in np.unique(actions):
            #     idx = actions == a
            #     states_a = states[idx]
            #     sensitives_a = sensitives[idx]
            #     X = np.concatenate([sensitives_a, states_a], axis=1)
            #     rewards_a = self.reward_models[a].predict(X).flatten()
            #     rewards[idx] = rewards_a
            for z_ in np.unique(zs, axis=0):
                #idx = (sensitives == z_).flatten()
                idx = np.all(zs == z_, axis=1).flatten()
                states_z_ = states[idx]
                actions_z_ = actions[idx]
                #X = np.concatenate(
                #    [states_z_, actions_z_.reshape(-1, actions_z_.shape[-1])], axis=1
                #)
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
    
    def step(self, action, errors_next_states=None, errors_rewards=None, seed=1):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if not self.is_trained:
            print('Cannot make transitions because the environment is not yet trained.')
            exit(1)

        at = action
        xt = self.xt
        next_states_mean, rewards_mean = self._next_state_reward_mean(
            self.zs, xt, at
        )

        if errors_next_states is None:
            errors_next_states = np.random.multivariate_normal(
                mean=np.zeros(xt.shape[-1]),
                #cov=np.diag(self.trans_models[self.zs].var),
                cov=np.diag(np.ones(xt.shape[-1])), 
                size=xt.shape[0],
            )
        if errors_rewards is None:
            errors_rewards = np.random.normal(
                #loc=0, scale=np.sqrt(self.reward_models[0].var), size=xt.shape[0]
                loc=0, scale=1, size=xt.shape[0]
            )

        next_states = next_states_mean + errors_next_states
        rewards = rewards_mean + errors_rewards

        self.atm1 = action
        self.xt = next_states
        self.rtm1 = rewards

        # form the current observation and compute the reward
        observation = self.xt
        reward = self.rtm1

        # the synthetic enviornment never terminates or gets truncated
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated
    


# REQUIRES: zs should be the same as the zs passed to the environment
def sample_simulated_env_trajectory(env, zs, state_dim, N, T, sigma=1, seed=1, policy=None):
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
    #ux0 = np.random.normal(0, sigma, [N, 1])
    #ux0 = np.ones((N, 1))
    X[:, 0], _ = env.reset(Z)
    
    # take the first step
    ua0 = np.random.uniform(0, 1, size=[N])
    #ua0 = np.zeros(N)
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
    #ur0 = np.random.normal(0, sigma, [N, 1])
    #ux1 = np.random.normal(0, sigma, [N, 1])
    #ux1 = np.zeros((N, 1))
    #ur0 = np.ones((N, 1))
    X[:, 1], R[:, 0], _, _ = env.step(A[:, 0])

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
        X[:, t + 1], R[:, t], _, _ = env.step(A[:, t])

    # prepare and return the output
    out = [Z, X, A, R]
    return out



def estimate_counterfactual_trajectories_from_data(env, zs, states, 
                                                   actions, policy, 
                                                   sigma_a=1, seed=1):
    N = actions.shape[0]
    T = actions.shape[1]
    state_dim = states.shape[2]
    np.random.seed(seed)
    z_eval_levels = np.unique(zs, axis=0)
    z_eval_levels = np.array(z_eval_levels)
    #print('Observed Z: ', zs, sep='\n')
    #print('\n')

    # dictionary to contain the counterfactual trajectories; key = z and value = trajectory
    trajectories = {}
    for z_level in z_eval_levels:
        env_z = copy.deepcopy(env)
        policy_z = copy.deepcopy(policy)
        #Z = np.full(N, [z_level]).reshape(-1, 1)
        Z = np.repeat(z_level, repeats=N, axis=0) .reshape(N, -1)
        X = np.zeros([N, T + 1, state_dim], dtype=float)
        A = np.zeros([N, T + 1], dtype=int)
        R = np.zeros([N, T], dtype=float)
        trajectories[tuple(z_level.flatten())] = {'Z': Z, 'X': X, 'A': A, 'R': R, 
                                        'env_z': env_z, 'policy_z': policy_z}
    #print('Initialized dict:', trajectories, sep='\n')
    #print('\n')

    # generate the initial state
    #print('t = 0: ')
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
    ua0 = np.random.uniform(0, sigma_a, size=[N])
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
            '''actions = np.zeros(N)
            for i in range(N):
                actions[i] = trajectories[tuple(zs[i].flatten())]['A'][i, 0]
                #actions[i] = trajectories[tuple(z_level.flatten())]['A'][i, 0]'''
            #actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, 0] for i in range(N)]) 
            #print('a0: ', actions, sep='\n')
            obs_mean, _ = e._next_state_reward_mean(
                        zs[idx],
                        states[idx, 0, :],
                        actions[idx, 0],
                        #np.ones([xs_test.shape[0], 1]) * (t - 1),
                        )
            cf_mean, _ = e._next_state_reward_mean(
                        trajectories[tuple(z_level.flatten())]['Z'][idx],
                        trajectories[tuple(z_level.flatten())]['X'][idx, 0, :],
                        actions[idx, 0],
                        #np.ones([xs_test.shape[0], 1]) * (t - 1),
                        )
            (
                trajectories[tuple(z_level.flatten())]['X'][idx, 1]
            ) = (states[idx, 1] - obs_mean + cf_mean)
    #print(trajectories)
    #print('\n')

    # take subsequent steps
    for t in range(1, T):
        #print('t =', t, sep=' ')
        uat = np.random.uniform(0, sigma_a, size=[N])
        for z_level in z_eval_levels:
            p = trajectories[tuple(z_level.flatten())]['policy_z']
            '''actions = np.zeros(N)
            for i in range(N):
                actions[i] = trajectories[tuple(zs[i].flatten())]['A'][i, t - 1]
                #actions[i] = trajectories[tuple(z_level.flatten())]['A'][i, t - 1]'''
            #actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, t - 1] for i in range(N)]) 
            trajectories[tuple(z_level.flatten())]['A'][:, t] = p.act(
                    z=trajectories[tuple(z_level.flatten())]['Z'],
                    xt=trajectories[tuple(z_level.flatten())]['X'][:, t],
                    xtm1=trajectories[tuple(z_level.flatten())]['X'][:, t - 1],
                    #atm1=trajectories[tuple(z_level.flatten())]['A'][:, t - 1],
                    atm1=actions[:, t - 1],
                    uat=uat,
                )
        for z_level in z_eval_levels:
            #print('t =', t, 'z_level =', z_level, sep=' ')
            e = trajectories[tuple(z_level.flatten())]['env_z']
            for z_obs in np.unique(zs, axis=0):
                idx = np.all(zs == z_obs, axis=1).flatten()
                '''actions = np.zeros(N)
                for i in range(N):
                    actions[i] = trajectories[tuple(zs[i].flatten())]['A'][i, t - 1]
                    #actions[i] = trajectories[tuple(z_level.flatten())]['A'][i, t - 1]'''
                #actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, t - 1] for i in range(N)]) 
                obs_mean, _ = e._next_state_reward_mean(
                        zs[idx],
                        states[idx, t, :],
                        actions[idx, t],
                        #np.ones([xs_test.shape[0], 1]) * (t - 1),
                        )
                cf_mean, _ = e._next_state_reward_mean(
                        trajectories[tuple(z_level.flatten())]['Z'][idx],
                        trajectories[tuple(z_level.flatten())]['X'][idx, t, :],
                        actions[idx, t],
                        #np.ones([xs_test.shape[0], 1]) * (t - 1),
                        )
                (
                    trajectories[tuple(z_level.flatten())]['X'][idx, t + 1]
                ) = (states[idx, t + 1] - obs_mean + cf_mean)
    
    # take the final step
    ua0 = np.random.uniform(0, sigma_a, size=[N])
    for z_level in z_eval_levels:
        p = trajectories[tuple(z_level.flatten())]['policy_z']
        trajectories[tuple(z_level.flatten())]['A'][:, T] = p.act(
                z=trajectories[tuple(z_level.flatten())]['Z'],
                xt=trajectories[tuple(z_level.flatten())]['X'][:, T],
                xtm1=trajectories[tuple(z_level.flatten())]['X'][:, T - 1],
                atm1=actions[:, T - 1],
                uat=ua0,
            )
    
    return trajectories