# Evaluate function
# Input: model to eval, num_episodes, return_episodes rewards for each episode?, baseline= True means ignore the model and just always choose 21 as action
import numpy as np

def evaluate(model,env,n_eval_episodes,return_episode_rewards = False,deterministic = True, baseline = False):
    episode_rewards, episode_lengths = [], []
    episode_costs,episode_comforts = [],[]
    for _ in range(n_eval_episodes):
       # if _ == 0:
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        episode_cost = 0.0
        episode_comfort = 0.0
        
        while not done:
            
            if not baseline:
                action, state = model.predict(obs, state=state, deterministic=deterministic)
            
            if baseline:
                obs, reward, done, _info = env.step([21]) # Ignore what the policy is predicting always set to 21
                               
            else:
                obs, reward, done, _info = env.step(action)
               
            
            episode_reward += reward
            episode_length += 1
          #  episode_cost += _info['Cost'] #Add up costs 
          #  episode_comfort += _info['Comfort']
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_costs.append(episode_cost)
        episode_comforts.append(episode_comfort)
        
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_cost = np.mean(episode_costs)
    std_cost = np.std(episode_costs)
    mean_comfort = np.mean(episode_comforts)
    std_comfort = np.std(episode_comforts)
    
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward,mean_cost,std_cost,mean_comfort,std_comfort