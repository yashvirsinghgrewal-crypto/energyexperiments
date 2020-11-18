import random
from stable_baselines import PPO2,TRPO,ACKTR
from evaluate import evaluate
# RL2 implementation using the Mlp lstm policy by stable baselines based on Learning to reinforcement learn paper -
''''At the start of a new episode, its LSTM state
is reset and a task b ∼ D is sampled. A task is defined as a set of distributions – one for
each arm – from which rewards are sampled. The agent plays in this environment for a certain
number of trials and is trained to maximize observed rewards. After training, the agent’s policy is
evaluated on a set of tasks that are drawn from a test distribution D0
, which can either be the
same as D or a slight modification of it'''
# Here the LSTM's hidden state is reset automatically at end of each episode (var mask=true when done = True) 
# Therefore the agent must update the weights of hidden state during meta-training as below, such that at start of each episode, it learns to use these hidden states from the steps in the new MDP to quickly adapt it's policy to the MDP specific policy. So during the meta-training below it just learns the optimal weight's for it's hidden states that can allow it to adapt quickly at the start of each episode to the new MDP.


def rl_squared(total_episodes,env):
    
    reward_1,dynamics =[],[]

    # Randomly sample from the training task distribution the env to be used for training
    sampled_env = random.choice(env)
    
    
    model = PPO2("MlpLstmPolicy", sampled_env, gamma=0.95, verbose=0,seed = 1211,nminibatches=1) #1#11#001extra + 10 extra
#    model = TRPO("MlpLstmPolicy", sampled_env, gamma=0.99, verbose=0,seed = 12210) 

    # Train for total_Episodes
    for i in range(total_episodes):

            # Sample a new_env for each episode
            sampled_env = random.choice(env)

            model.set_env(sampled_env)
            model.learn(total_timesteps=288,reset_num_timesteps=False) #Go on for 1day
            
            #Evaluate for 1 episode(=1day)
            mean_reward, std_reward,sum_cost,std_cost,sum_comfort,std_comfort = evaluate(model, sampled_env, n_eval_episodes=1,baseline = False)
         
            reward_1.append(mean_reward)
            dynamics.append((env.index(sampled_env)))
            
            print("Episode:",i)
                
  
    return reward_1,dynamics,model


def meta_rnn(total_episodes,env,env2,env1):
    
    reward_1,dynamics =[],[]
    
    env = DummyVecEnv([lambda: env])
    env2 =DummyVecEnv([lambda: env2])
    env1 =DummyVecEnv([lambda: env1])
    
    
    model = PPO2("MlpLstmPolicy", env, gamma=0.95, verbose=0,seed = 121,nminibatches=1) #1#11#001extra + 10 extra
  #  model = ACKTR("MlpLstmPolicy", env, gamma=0.95, verbose=0,seed = 12210) 
    
    
    for i in range(total_episodes):
            
            time_heat = round(np.random.uniform(low=8,high=26))
            
            time_cool = time_heat*2 + round(np.random.uniform(low=-4,high=8))
            
            tcl = first_temp_model(time_heat*60,time_cool*60,21,1.5)
            env = first_temp_env(tcl,prices)
            
            all_env = [env,env2,env1]
            sampled_env = random.choice(all_env)
            
            
            if sampled_env == env:
                sampled_env = DummyVecEnv([lambda: env])
            
            model.set_env(sampled_env)
            model.learn(total_timesteps=288) #Go on for 1day
            #Evaluate for 1 episode(1day)
            mean_reward, std_reward,sum_cost,std_cost,sum_comfort,std_comfort = evaluate(model, sampled_env, n_eval_episodes=1,baseline = False)
         
            reward_1.append(mean_reward)
            dynamics.append((time_heat,time_cool))
            print(i)
            
            if sampled_env == env2: 
                print('env2nd_price',mean_reward)
            elif sampled_env == env1:
                print('env2nd_fixtemp',mean_reward)
            
            else:
                
                print(('first_price',time_heat,time_cool,mean_reward))
          
        
  
    return reward_1,model