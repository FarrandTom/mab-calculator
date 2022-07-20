import random
import pandas as pd
import numpy as np
from collections import Counter


def create_simulated_reward_data(model_accuracies, no_of_rewards):
    """
    Generates a DataFrame of synthetic reward data. Columns are each of the models in the MAB setup. 
    Rows are the reward passed to each model for a given observation. 
    
    Parameters:
        model_accuracies (list): List of each of the model accuracies being compared.
        no_of_rewards (int): Number of rewards observed within the given time window.
    
    Returns: 
        
    """
    rng = np.random.default_rng()
    data = {}
    
    for i in range(len(model_accuracies)):
        data[f"Model {i}"] = rng.binomial(1, model_accuracies[i], no_of_rewards)

    data = pd.DataFrame(data)
    return data

            
def generate_ts_time_series(model_accuracies, time_series_length, no_of_rewards):
    reward_dict = {"rewards": None, "penalties": None}
    results_list = []

    init_rewards = [0] * len(model_accuracies)
    init_penalties = [0] * len(model_accuracies)

    for i in range(time_series_length):

        if i == 0:
            data = create_simulated_reward_data(model_accuracies, no_of_rewards)
            results, rewards, penalties = thompson_sampling_experiment(data, model_accuracies, init_rewards, init_penalties)

            results_list.append(results)
            reward_dict["rewards"] = rewards
            reward_dict["penalties"] = penalties

        else:
            prev_rewards = reward_dict["rewards"]
            prev_penalties = reward_dict["penalties"]

            data = create_simulated_reward_data(model_accuracies, no_of_rewards)
            results, rewards, penalties = thompson_sampling_experiment(data, model_accuracies, prev_rewards, prev_penalties)
            
            results_list.append(results)
            reward_dict["rewards"] = rewards
            reward_dict["penalties"] = penalties
    
    return results_list

            
def thompson_sampling_experiment(data, model_accuracies, rewards, penalties):
    """
    Returns the number of incorrect classifications made if observations
    were routed using the Thompson Sampling algorithm.
    
    Parameters:
        data (pd.DataFrame): DataFrame where columns are the models, and each row was 
                             the reward passed to the model for a given observation.
        model_accuracies (list): List of each of the model accuracies being compared.
                             
    
    Returns: 
        count_of_model_selected (dict): Dictionary where the keys are the model number and the values
        are the number of times that model was selected for prediction. 
    """
    
    no_of_observations = int(len(data))
    no_of_models = int(len(data.columns))
    
    assert no_of_models == int(len(model_accuracies)), \
            f"The number of models ({no_of_models}) does not match the number of accuracy values ({int(len(model_accuracies))}) provided."
    model_selected = []
    
    for n in range(0, no_of_observations):
        bandit = 0
        beta_max = 0

        for i in range(0, no_of_models):
            beta_d = random.betavariate(rewards[i] + 1, penalties[i] + 1)
            if beta_d > beta_max:
                beta_max = beta_d
                bandit = i

        model_selected.append(bandit)       
        reward = data.values[n, bandit]

        if reward == 1:
            rewards[bandit] = rewards[bandit] + 1
        else:
            penalties[bandit] = penalties[bandit] + 1
    
    count_of_model_selected = dict(Counter(model_selected))
    return count_of_model_selected, rewards, penalties


def generate_control_time_series(model_accuracies, time_series_length, no_of_rewards):
    results_list = []
    
    for i in range(time_series_length):
        data = create_simulated_reward_data(model_accuracies, no_of_rewards)
        results = control_experiment(data, model_accuracies)
        results_list.append(results)
        
    return results_list
    

def control_experiment(data, model_accuracies):
    """
    Returns the model selections made if observations
    were simply split evenly across the models.
    
    Parameters:
        model_accuracies (list): List of each of the model accuracies being compared.
        no_of_observations (int): Number of observations within time window.
        
    Returns: 
        incorrect_preds (list): Number of incorrect observations from each of the
                                models being compared. 
    """
    percentage_split = 1 / len(model_accuracies)
    count_of_model_selected = {}
    
    for i in range(len(model_accuracies)): 
        count_of_model_selected[i] = int(np.floor(len(data) * percentage_split))
    
    diff = len(data) - sum(count_of_model_selected.values())
    if  diff != 0:
        for i in range(diff):
            count_of_model_selected[i] += 1
    
    return count_of_model_selected