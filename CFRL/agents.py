from abc import abstractmethod
import numpy as np
import torch
import copy
#from utils.utils import glogger
from .utils.base_models import LinearRegressor, NeuralNet
from .preprocessor import Preprocessor, SequentialPreprocessor
from typing import Literal
from tqdm import tqdm

class Agent:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def act(
            self, 
            z: list | np.ndarray, 
            xt: list | np.ndarray, 
            xtm1: list | np.ndarray | None = None, 
            atm1: list | np.ndarray | None = None, 
            uat: list | np.ndarray | None = None, 
            **kwargs
    ) -> np.ndarray:
        pass



class FQI(Agent):
    def __init__(
        self,
        model_type: Literal["nn", "lm"],
        num_actions: int,
        hidden_dims: list[int] = [32],
        preprocessor: Preprocessor | None = None,
        gamma: int | float = 0.9,
        learning_rate: int | float = 0.1,
        epochs: int = 500,
    ) -> None:
        self.model_type = model_type
        self.action_space = np.array([a for a in range(num_actions)]).reshape(-1, 1)
        self.action_size = num_actions
        self.hidden_dims = hidden_dims
        self.preprocessor = preprocessor
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.__name__ = 'FQI'
        self._sanity_check()

    def _sanity_check(self) -> None:
        assert self.model_type in ["lm", "nn"], "Invalid model type"

    def _init_model(self, state_dim: int) -> None:
        if self.model_type == "lm":
            self.model = [
                LinearRegressor(
                    featurize_method="polynomial",
                    degree=2,
                    interaction_only=False,
                    is_standarized=False,
                )
                for _ in range(self.action_size)
            ]
        elif self.model_type == "nn":
            self.model = NeuralNet(
                in_dim=state_dim,
                out_dim=self.action_size,
                hidden_dims=self.hidden_dims,
            )

    def _fit_helper(
            self, 
            states: np.ndarray, 
            actions: np.ndarray, 
            rewards: np.ndarray, 
            next_states: np.ndarray, 
            state_dim: int, 
            max_iter: int
        ) -> None:
        torch.set_num_threads(1)
        states = states.reshape(-1, state_dim)
        next_states = next_states.reshape(-1, state_dim)
        actions = actions.reshape(-1, 1)
        rewards = rewards.reshape(-1, 1)

        for iteration in tqdm(range(max_iter)):
            if self.model_type == "lm":
                self._fit_lm(states, actions, rewards, next_states, iteration)
            elif self.model_type == "nn":
                self._fit_nn(states, actions, rewards, next_states, iteration, state_dim)

    def _fit_nn(
            self, 
            states: np.ndarray, 
            actions: np.ndarray, 
            rewards: np.ndarray, 
            next_states: np.ndarray, 
            iteration: int, 
            state_dim: int
        ) -> None:

        if iteration == 0:
            Y = rewards
        else:
            self.model.eval()
            with torch.no_grad():
                next_q_values = self.model(torch.FloatTensor(next_states)).numpy()
            Y = (
                rewards + self.gamma * np.max(next_q_values, axis=1, keepdims=True)
            ).reshape(-1, 1)

        new_model = NeuralNet(
            in_dim=state_dim,
            out_dim=self.action_size,
            hidden_dims=self.hidden_dims,
        )

        X = torch.FloatTensor(states)
        Y = torch.FloatTensor(Y)
        actions_tensor = torch.LongTensor(actions.reshape(-1, 1))

        optimizer = torch.optim.Adam(new_model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()

        for _ in range(self.epochs):
            new_model.train()
            optimizer.zero_grad()
            Y_pred = new_model(X).gather(1, actions_tensor)
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.step()
        '''glogger.info(
            f"{iteration}, fqi_nn mse:{loss.item()}, mean_target:{np.mean(Y.detach().numpy())}"
        )'''
        self.model = copy.deepcopy(new_model)

    def _fit_lm(
            self, 
            states: np.ndarray, 
            actions: np.ndarray, 
            rewards: np.ndarray, 
            next_states: np.ndarray, 
            iteration: int
        ) -> None:
        if iteration == 0:
            Y = rewards
        else:
            tmp = np.array(
                [model.predict(next_states) for model in self.model]
            ).T.squeeze()
            Y = (rewards + self.gamma * np.max(tmp, axis=1, keepdims=True)).reshape(
                -1, 1
            )

        new_model = [
            LinearRegressor(
                featurize_method="polynomial",
                degree=2,
                interaction_only=False,
                is_standarized=False,
            )
            for _ in range(self.action_size)
        ]

        for a in range(self.action_size):
            idx = actions.flatten() == a
            new_model[a].fit(states[idx], Y[idx])

        mse = np.mean([m.mse for m in new_model])
        #glogger.info(f"{iteration}, fqi_lm mse:{mse}, mean_target:{np.mean(Y)}")
        self.model = copy.deepcopy(new_model)

    def _act_helper(self, states: np.ndarray) -> np.ndarray:
        if self.model_type == "lm":
            q_values = np.array(
                [model.predict(states) for model in self.model]
            ).T.squeeze()
        elif self.model_type == "nn":
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(torch.FloatTensor(states)).numpy()
        return np.argmax(q_values, axis=1)
    
    def train(
            self, 
            zs: list | np.ndarray, 
            xs: list | np.ndarray, 
            actions: list | np.ndarray, 
            rewards: list | np.ndarray, 
            max_iter: int = 1000, 
            preprocess: bool = True
        ) -> None:  
        zs = np.array(zs)
        xs = np.array(xs)
        actions = np.array(actions)
        rewards = np.array(rewards)

        if self.preprocessor is not None and preprocess:
            states_p, rewards_p = self.preprocessor.preprocess_multiple_steps(
                    zs=zs, xs=xs, actions=actions, rewards=rewards
            )
        else:
            states_p = xs
            rewards_p = rewards
        state_dim = states_p.shape[-1]
        self._init_model(state_dim)

        next_states_p = states_p[:, 1:].copy()
        states_p = states_p[:, :-1].copy()

        self._fit_helper(states_p, actions, rewards_p, next_states_p, state_dim, max_iter)

    # FOR A PARTICULAR SEQUENCE, IT SEEMS WE HAVE TO TAKE ACTIONS SEQUENTIALLY FROM THE BEGINNING 
    # TO THE END, AND WE CANNOT SKIP ANY STEP? OTHERWISE INFO IN BUFFER IS MESSED UP. SO TAKEAWAY
    # IS WE CANNOT PREPROCESS AN ARBITRARY SINGLE STEP USIN THE PREPROCESSOR, AND WE MUST 
    # PREPROCESS A WHOLE TRAJECTORY IN ORDER?
    def act(
            self, 
            z: list | np.ndarray, 
            xt: list | np.ndarray, 
            xtm1: list | np.ndarray | None = None, 
            atm1: list | np.ndarray | None = None, 
            uat: list | np.ndarray | None = None, 
            preprocess: bool = True, 
            **kwargs
        ) -> np.ndarray:
        z = np.array(z)
        xt = np.array(xt)
        if xtm1 is not None:
            xtm1 = np.array(xtm1)
        if atm1 is not None:
            atm1 = np.array(atm1)
        if uat is not None:
            uat = np.array(uat)

        if self.preprocessor is not None and preprocess:
            states = self.preprocessor.preprocess_single_step(
                xt=xt, xtm1=xtm1, z=z, atm1=atm1, rtm1=None
            )
        else:
            states = xt
        actions = self._act_helper(states)
        return actions