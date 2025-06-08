from package.reader import read_trajectory_from_csv, convert_trajectory_to_dataframe
import numpy as np
import pandas as pd

def test_reader_numerical_id():
    zs, states, actions, rewards, ids = read_trajectory_from_csv(
                      path='./tests/sample_inputs/sample_input_1.csv', 
                      z_labels=['gender', 'race'], 
                      state_labels=['state1', 'state2', 'state3'], 
                      action_label='decision', 
                      reward_label='reward', 
                      id_label='id', 
                      T=3)
    
    zs_correct = np.array([[0, 1], 
                           [1, 0], 
                           [1, 3]])
    states_correct = np.array([[[1, 4, 6], [5, 7, 3], [6, 7, 1]], 
                               [[5, 6, 7], [2, 7, 3], [8, 10, 1]], 
                               [[6, 7, 8], [4, 5, 6], [4, 3, 6]]])
    actions_correct = np.array([[2, 2], 
                                [2, 1], 
                                [1, 2]])
    rewards_correct = np.array([[4, 3], 
                                [4, 4], 
                                [2, 9]])
    ids_correct = np.array([[1], [2], [3]])
    assert(np.array_equal(zs, zs_correct))
    assert(np.array_equal(states, states_correct))
    assert(np.array_equal(actions, actions_correct))
    assert(np.array_equal(rewards, rewards_correct))
    assert(np.array_equal(ids, ids_correct))

    df = convert_trajectory_to_dataframe(
                        zs=zs, 
                        states=states, 
                        actions=actions, 
                        rewards=rewards, 
                        ids=ids, 
                        z_labels=['gender', 'race'], 
                        state_labels=['state1', 'state2', 'state3'], 
                        action_label='decision', 
                        reward_label='reward', 
                        id_label='id'
                        )
    df_correct = pd.read_csv('./tests/sample_inputs/sample_input_1.csv')
    assert(np.array_equal(df[['id', 'race', 'gender', 'state1', 'state2', 'state3']].to_numpy(), 
                          df_correct[['id', 'race', 'gender', 'state1', 'state2', 'state3']].to_numpy()))
    assert(np.array_equal(df[['decision', 'reward']].to_numpy()[[1, 2, 4, 5, 7, 8], :], 
                          df_correct[['decision', 'reward']].to_numpy()[[1, 2, 4, 5, 7, 8], :]))

def test_reader_text_id():
    pass



test_reader_numerical_id()