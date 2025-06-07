import numpy as np
import torch
import copy
from utils.base_models import LinearRegressor, NeuralNet
#from my_utils import glogger


class FQE:
    def __init__(self, model_type, action_size, policy, 
                 hidden_dims=[32], lr=0.1, epochs=500, gamma=0.9) -> None:
        self.model_type = model_type
        self.action_size = action_size
        self.policy = policy
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.gamma = gamma

        # sanity check
        self._sanity_check()

    def _sanity_check(self):
        assert self.model_type in ["lm", "nn"], "Invalid model type"

    def get_actions(self, zs, states, actions, uat=None):
        xs = states
        del(states)
        N, T, _ = xs.shape  # xs + 1
        
        p = copy.deepcopy(self.policy) # use a deepcopy to preserve the info in original policy
        actions_taken = np.zeros([N, T])
        for t in range(T): # MIGHT NEED TO CHANGE TO T - 1
            if t == 0:
                actions_taken[:, 0] = p.act(zs, 
                                            xs[:, 0], 
                                            None, 
                                            None, 
                                            uat)
            else:
                actions_taken[:, t] = p.act(zs, 
                                            xs[:, t], 
                                            xs[:, t - 1], 
                                            actions[:, t - 1], 
                                            uat)

        return actions_taken[:, 1:].reshape(N *(T - 1), -1)

    def fit(self, zs, states, actions, rewards, max_iter, uat=None):
        torch.set_num_threads(1)
        # convenience variables
        xs = states
        del(states)
        N, T = actions.shape
        sdim = xs.shape[-1] + zs.shape[-1]

        xs_ = copy.deepcopy(xs)
        zs_ = copy.deepcopy(zs)
        actions_ = copy.deepcopy(actions)
        rewards_ = copy.deepcopy(rewards)

        # reshape data
        next_states = np.concatenate(
            [xs_[:, 1:, :], np.repeat(zs_.reshape(-1, 1, zs.shape[-1]), axis=1, repeats=T)], 
            axis=2
        ).reshape(-1, sdim)
        states = np.concatenate(
            [xs_[:, :-1, :], np.repeat(zs_.reshape(-1, 1, zs.shape[-1]), axis=1, repeats=T)],
            axis=2, # TRY TO DEAL WITH DIMENSION PROBLEMS OF ZS AND STATES: COMPLETED!
        ).reshape(-1, sdim)
        actions = actions[:, :].flatten()
        rewards = rewards[:, :].flatten()

        # init model
        if self.model_type == "lm":
            current_model = [None for _ in range(self.action_size)]
            for i in range(max_iter):
                # generate target
                if i == 0:
                    Y = rewards.reshape(-1, 1)
                else:
                    tmp = np.zeros([N * T, self.action_size])
                    for a in range(self.action_size):
                        tmp[:, a] = current_model[a].predict(next_states).flatten()
                    # selected_actions = self.policy.act(
                    #     uat=np.random.uniform(size=states.shape[0]), states=next_states
                    # ).flatten()
                    selected_actions = self.get_actions(zs_, xs_, actions_, uat).flatten()
                    Y = (
                        rewards + self.gamma * tmp[np.arange(tmp.shape[0]), selected_actions]
                    ).reshape(-1, 1)

                # generate input
                new_model = [
                    LinearRegressor(
                        featurize_method="polynomial",
                        degree=2, 
                        interaction_only=False,
                        is_standarized=False,
                    )
                    for i in range(self.action_size)
                ]
                for a in range(self.action_size):
                    idx = actions == a
                    X_a = states[idx]
                    Y_a = Y[idx]
                    new_model[a].train(X_a, Y_a)
                #glogger.info(
                #    "{}, fqe_lm mse:{}, mean_target:{}".format(
                #        i, np.mean([m.mse for m in new_model]), np.mean(Y)
                #    )
                #)
                current_model = copy.deepcopy(new_model)

        elif self.model_type == "nn":
            np.random.seed(10) # NEWLY ADDED
            torch.manual_seed(10) # NEWLY ADDED
            current_model = NeuralNet(
                in_dim=sdim, out_dim=self.action_size, hidden_dims=self.hidden_dims 
            )

            # training loop
            for i in range(max_iter):
                # generate target
                if i == 0:
                    Y = rewards.reshape(-1, 1)
                else:
                    current_model.eval()
                    tmp = (
                        current_model.forward(
                            torch.tensor(next_states, dtype=torch.float32)
                        )
                        .detach()
                        .numpy()
                    )
                    selected_actions = self.get_actions(zs_, xs_, actions_, uat).reshape(
                        -1, 1
                    )

                    Y = (
                        rewards.reshape(-1, 1)
                        + self.gamma * np.take_along_axis(tmp, selected_actions.astype(int), axis=1)
                    ).reshape(-1, 1)

                # generate input
                X = states

                # train model
                np.random.seed(10) # NEWLY ADDED
                torch.manual_seed(10) # NEWLY ADDED
                new_model = NeuralNet(
                    in_dim=sdim, out_dim=self.action_size, hidden_dims=self.hidden_dims
                )
                Y = torch.tensor(Y, dtype=torch.float32)

                optimizer = torch.optim.Adam(new_model.parameters(), lr=self.lr)
                for _ in range(self.epochs):
                    new_model.train()
                    Y_pred = new_model.forward(
                        torch.tensor(X, dtype=torch.float32)
                    ).gather(1, torch.tensor(actions.reshape(-1, 1), dtype=torch.int64))
                    optimizer.zero_grad()
                    loss = torch.nn.MSELoss()(Y, Y_pred)
                    loss.backward()
                    optimizer.step()

                #glogger.info(
                #    "{}, fqe_nn mse:{}, mean_target:{}".format(
                #        i, loss.item(), np.mean(Y.detach().numpy())
                #   )
                #)
                current_model = copy.deepcopy(new_model)

        self.model = copy.deepcopy(new_model)

    def evaluate(self, zs, states, actions, uat=None):
        xs = states
        del(states)
        sdim = xs.shape[-1] + zs.shape[-1]
        T = actions.shape[1]
        actions_taken = self.get_actions(zs, xs, actions, uat)
        xs_ = copy.deepcopy(xs)
        zs_ = copy.deepcopy(zs)

        states = np.concatenate(
            [xs_[:, 1:, :], np.repeat(zs_.reshape(-1, 1, zs.shape[-1]), axis=1, repeats=T)], axis=2
        ).reshape(
            -1, sdim
        )  # use 1: instead of 0: since get_actions is implemented using next_state

        #np.random.seed(10) # NEWLY ADDED
        #torch.manual_seed(10) # NEWLY ADDED
        if self.model_type == "lm":
            tmp = np.zeros([states.shape[0], self.action_size])
            for a in range(self.action_size):
                tmp[:, a] = (
                    self.model[a].predict(states.reshape(states.shape[0], -1)).flatten()
                )
            return np.take_along_axis(
                tmp, actions_taken.reshape(-1, 1).astype(int), axis=1
            ).flatten()
        elif self.model_type == "nn":
            self.model.eval()
            X = torch.tensor(states, dtype=torch.float32)
            actions_taken = torch.tensor(
                actions_taken.reshape(-1, 1), dtype=torch.int64
            )
            Y = self.model.forward(X).gather(1, actions_taken)
            return Y.detach().numpy().flatten()