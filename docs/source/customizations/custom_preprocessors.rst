Custom Preprocessors
==========================

In addition to :code:`SequentialPreprocessor`, users can also define custom preprocessors and use them 
in CFRL. 

To ensure a custom preprocessor is compatible with CFRL, it must inherit from the 
:code:`Preprocessor` class provided by the :code:`preprocessor` module. That is, 

- The custom preprocessor should be a subclass of :code:`Preprocessor`.
- The custom preprocessor should have a :code:`preprocess_single_step()` method whose signature is 
  exactly as that defined in the :code:`Preprocessor` class, except that it might have some additional 
  arguments. The input and output lists or arrays should also follow the same 
  :ref:`Trajectory Array format <trajectory_arrays>` as those defined in :code:`Preprocessor`.
- The custom preprocessor should have a :code:`preprocess_multiple_steps()` method whose signature is 
  exactly as that defined in the :code:`Preprocessor` class, except that it might have some additional 
  arguments. The input and output lists or arrays should also follow the same 
  :ref:`Trajectory Array format <trajectory_arrays>` as those defined in :code:`Preprocessor`.
- :code:`preprocess_single_step()` should return only one array when :code:`rtm1=None` and two arrays 
  otherwise. :code:`preprocess_multiple_steps()` should return only one array when :code:`rewards=None` 
  and two arrays otherwise.

For example, though simple, the following :code:`ConcatenatePreprocessor` is a valid custom 
preprocessor that will be compatible with CFRL.

.. code-block:: python

    class ConcatenatePreprocessor(Preprocessor):
        def __init__(self) -> None:
            pass

        def preprocess(
                self, 
                z: list | np.ndarray, 
                xt: list | np.ndarray
            ) -> tuple[np.ndarray]:
            if xt.ndim == 1:
                xt = xt[np.newaxis, :]
                z = z[np.newaxis, :]
                xt_new = np.concatenate([xt, z], axis=1)
                return xt_new.flatten()
            elif xt.ndim == 2:
                xt_new = np.concatenate([xt, z], axis=1)
                return xt_new
            
        def preprocess_single_step(
                self, 
                z: list | np.ndarray, 
                xt: list | np.ndarray, 
                xtm1: list | np.ndarray | None = None, 
                atm1: list | np.ndarray | None = None, 
                rtm1: list | np.ndarray | None = None, 
                verbose: bool = False
            ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
            z = np.array(z)
            xt = np.array(xt)
            if verbose:
                print("Preprocessing a single step...")

            xt_new = self.preprocess(z, xt)
            if rtm1 is None:
                return xt_new
            else:
                return xt_new, rtm1
            

        def preprocess_multiple_steps(
                self, 
                zs: list | np.ndarray, 
                xs: list | np.ndarray, 
                actions: list | np.ndarray, 
                rewards: list | np.ndarray | None = None, 
                verbose: bool = False
            ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
            zs = np.array(zs)
            xs = np.array(xs)
            actions = np.array(actions)
            rewards = np.array(rewards)
            if verbose:
                print("Preprocessing multiple steps...")
        
            # some convenience variables
            N, T, xdim = xs.shape
            
            # define the returned arrays; the arrays will be filled later
            xs_tilde = np.zeros([N, T, xdim + zs.shape[-1]])
            rs_tilde = np.zeros([N, T - 1])

            # preprocess the initial step
            np.random.seed(0)
            xs_tilde[:, 0, :] = self.preprocess_single_step(zs, xs[:, 0, :])

            # preprocess subsequent steps
            if rewards is not None:
                for t in range (1, T):
                    np.random.seed(t)
                    xs_tilde[:, t, :], rs_tilde[:, t-1] = self.preprocess_single_step(zs, 
                                                                                    xs[:, t, :], 
                                                                                    xs[:, t-1, :], 
                                                                                    actions[:, t-1], 
                                                                                    rewards[:, t-1]
                                                                                    )
                return xs_tilde, rs_tilde                
            else:
                for t in range (1, T):
                    np.random.seed(t)
                    xs_tilde[:, t, :] = self.preprocess_single_step(zs, 
                                                                    xs[:, t, :], 
                                                                    xs[:, t-1, :], 
                                                                    actions[:, t-1]
                                                                    )
                return xs_tilde

On the other hand, the following preprocessor will not be compatible with CFRL 
because its :code:`preprocess_single_step()` does not have :code:`xtm1` and 
:code:`atm1` in its argument list and its :code:`preprocess_multiple_steps()` 
always returns only one array.

.. code-block:: python

    class ConcatenatePreprocessor(Preprocessor):
        def __init__(self) -> None:
            pass

        def preprocess(
                self, 
                z: list | np.ndarray, 
                xt: list | np.ndarray
            ) -> tuple[np.ndarray]:
            if xt.ndim == 1:
                xt = xt[np.newaxis, :]
                z = z[np.newaxis, :]
                xt_new = np.concatenate([xt, z], axis=1)
                return xt_new.flatten()
            elif xt.ndim == 2:
                xt_new = np.concatenate([xt, z], axis=1)
                return xt_new
            
        def preprocess_single_step(
                self, 
                z: list | np.ndarray, 
                xt: list | np.ndarray, 
                rtm1: list | np.ndarray = None, 
                verbose: bool = False
            ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
            z = np.array(z)
            xt = np.array(xt)
            if verbose:
                print("Preprocessing a single step...")

            xt_new = self.preprocess(z, xt)
            if rtm1 is None:
                return xt_new
            else:
                return xt_new, rtm1
            

        def preprocess_multiple_steps(
                self, 
                zs: list | np.ndarray, 
                xs: list | np.ndarray, 
                actions: list | np.ndarray, 
                rewards: list | np.ndarray | None = None, 
                verbose: bool = False
            ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
            zs = np.array(zs)
            xs = np.array(xs)
            if verbose:
                print("Preprocessing multiple steps...")
        
            # some convenience variables
            N, T, xdim = xs.shape
            
            # define the returned arrays; the arrays will be filled later
            xs_tilde = np.zeros([N, T, xdim + zs.shape[-1]])
            rs_tilde = np.zeros([N, T - 1])

            # preprocess the initial step
            np.random.seed(0)
            xs_tilde[:, 0, :] = self.preprocess_single_step(zs, xs[:, 0, :])

            # preprocess subsequent steps
            for t in range (1, T):
                np.random.seed(t)
                xs_tilde[:, t, :] = self.preprocess_single_step(zs, 
                                                                xs[:, t, :]
                                                                )
            return xs_tilde

If a preprocessor is a valid custom preprocessor, then it can be used wherever 
a :code:`SequentialPreprocessor` can be used. For example, it can be passed into 
a :code:`FQI` agent as an internal preprocessor.

.. code-block:: python

    # Also suppose zs, states, actions is a trajectory from our MDP of interest.
    p = ConcatenatePreprocessor()
    agent = FQI(num_actions=3, model_type="nn", preprocessor=p)
    agent.train(zs=zs, xs=states, actions=actions, rewards=rewards)