#@title Import modules.
#python3

import copy
import pyvirtualdisplay
import imageio 
import base64
import IPython


from acme import environment_loop
from acme.tf import networks
from acme.adders import reverb as adders
from acme.agents.tf import actors as actors
from acme.datasets import reverb as datasets
from acme.wrappers import gym_wrapper
from acme import specs
from acme import wrappers
from acme.agents.tf import d4pg
from acme.agents import agent
from acme.tf import utils as tf2_utils
from acme.utils import loggers

import csv
import gym 
import dm_env
import matplotlib.pyplot as plt
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf

# Import dm_control if it exists.
try:
  from dm_control import suite
except (ModuleNotFoundError, OSError):
  pass
# Set up a virtual display for rendering OpenAI gym environments.
#display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()


# Reading in the prices from the combined_csv file
import pandas as pd
df = pd.read_csv('combined_csv.csv')
prices = df[["Spot Price ($/MWh)"]]      
prices = np.array(prices)

from thermalmodels_second_price import SecondOrderModel as second_temp_model
from second_thermalenvironments_price import DirectSetpointEnv as second_temp_env
tcl1 = second_temp_model(pHouse=0)    # Calling the function firstorder model arg1,2- seconds to heat and seconds to cool, arg3-Goal, flex and noise
env1 = second_temp_env(tcl1,prices)

from thermalmodels_first_price import FirstOrderModel as first_temp_model
from first_thermalenvironments_price import DirectSetpointEnv as first_temp_env

loader = tf.saved_model.load("network_dqn/3b5b39ea-a986-11eb-8f44-0242ac110006/snapshots/network")
actor_dqn = actors.FeedForwardActor(policy_network=loader)

tcl5 = first_temp_model(15*60,40*60,21,1.5)
env5 = first_temp_env(tcl5,prices)

environment1 = gym_wrapper.GymWrapper(env5)
environment1 = wrappers.SinglePrecisionWrapper(environment1)

environment2 = gym_wrapper.GymWrapper(env1)
environment2 = wrappers.SinglePrecisionWrapper(environment2)
environment_spec = specs.make_environment_spec(environment1)

spec = specs.make_environment_spec(environment1)

import acme
from acme import specs
from acme.agents.tf import impala
from acme.tf import networks
import numpy as np
import sonnet as snt
from impala_learner_normal import IMPALALearner
#from policy_head import PolicyValueHead

for run in range(1):

    def _make_network(action_spec: specs.DiscreteArray) -> snt.RNNCore:
      return snt.DeepRNN([
          snt.Flatten(),
          snt.LSTM(256),
          snt.nets.MLP([132, 132]),
          networks.PolicyValueHead(action_spec.num_values),
      ])

    network = _make_network(spec.actions)
    num_actions = environment_spec.actions.num_values

    extra_spec = {
        'core_state': network.initial_state(1),
        'logits': tf.ones(shape=(1, num_actions), dtype=tf.float32)
    }
    # Remove batch dimensions.
    extra_spec = tf2_utils.squeeze_batch_dim(extra_spec)

    queue = reverb.Table.queue(
            name=adders.DEFAULT_PRIORITY_TABLE,
            max_size=100000,
            signature=adders.SequenceAdder.signature(
                environment_spec, extras_spec=extra_spec))
    _server = reverb.Server([queue], port=None)
    _can_sample = lambda: queue.can_sample(8)
    address = f'localhost:{_server.port}'

        # Component to add things into replay.
    adder = adders.SequenceAdder(
            client=reverb.Client(address),
            period=288,
            sequence_length=288,
        )

        # The dataset object to learn from.
    dataset = datasets.make_reverb_dataset(
            server_address=address,
            batch_size=8,
            sequence_length=288)

    tf2_utils.create_variables(network, [environment_spec.observations])

    import impala_acting_normal as acting
    from acme import types

    _actor = acting.IMPALAActor(network, adder)
    _learner = IMPALALearner(
            environment_spec=environment_spec,
            network=network,
            dataset=dataset,
            learning_rate = 0.0005

        )

    def observe_first(self, timestep: dm_env.TimeStep):
        self._actor.observe_first(timestep)

    def observe(
          self,
          action: types.NestedArray,
          next_timestep: dm_env.TimeStep,
      ):
        self._actor.observe(action, next_timestep)

    def update(self, wait: bool = False):
        # Run a number of learner steps (usually gradient steps).
        while self._can_sample():
            self._learner.step()

    def select_action(self, observation: np.ndarray) -> int:
        return self._actor.select_action(observation)

    environment = environment1
    #adder.reset()
    _actor = acting.IMPALAActor(network, adder)
    actor = _actor
    adder.reset()

    num_episodes = 50  #@param
    print("adder_filling",run)
    mean_reward=[]

    data = False
    data_train_x = None
    data_train_y = None
    for episode in range(num_episodes):

        array = [-1,-2,-3,-4,-5,-6,-7,-8,-9,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        heat = 20 + np.random.choice(array, 1)[0]
        add = np.random.choice([-1,1])
        cool = (heat*2) - add*(np.random.choice(12,1)[0])

        tcl5 = first_temp_model(heat*60,cool*60,21,1.5)
        env2 = first_temp_env(tcl5,prices) 


        environment = gym_wrapper.GymWrapper(env2)
        environment = wrappers.SinglePrecisionWrapper(environment)

        episode_return=0
        timestep = environment.reset()


        actor.observe_first(timestep)  # Note: observe_first.

        while not timestep.last():
            action = actor.select_action(timestep.observation)
          #  print(action)
            timestep = environment.step(action)
            episode_return += timestep.reward
            actor.observe(action=action, next_timestep=timestep)  # Note: observe.
  
        data = True
       # print(episode, "return",episode_return)
        mean_reward.append(episode_return)

   
    num_episodes = 410  #@param # 410

    mean_reward=[]

    # data = False
    # data_train_x,data_train_y = None,None
    print("impala_training",run)
    with open('rl2_training_results.csv', 'a', newline='') as file:
        for episode in range(num_episodes):
    
            array = [-1,-2,-3,-4,-5,-6,-7,-8,-9,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            heat = 20 + np.random.choice(array, 1)[0]
            add = np.random.choice([-1,1])
            cool = (heat*2) - add*(np.random.choice(12,1)[0])

            tcl5 = first_temp_model(heat*60,cool*60,21,1.5)
            env2 = first_temp_env(tcl5,prices) 


            environment = gym_wrapper.GymWrapper(env2)
            environment = wrappers.SinglePrecisionWrapper(environment)

            episode_return=0
            timestep = environment.reset()


            actor.observe_first(timestep)  # Note: observe_first.
            i =0
            while not timestep.last():
                action = actor.select_action(timestep.observation)
              #  print(action)
                timestep = environment.step(action)
                if i >40:
                    episode_return += timestep.reward
                #episode_return += timestep.reward
                actor.observe(action=action, next_timestep=timestep)  # Note: observe.
                i+=1
            while _can_sample():

                _learner.step()

       
            writer = csv.writer(file)
            writer.writerow([run, episode, episode_return])
            print(episode, episode_return)
            data = True
            mean_reward.append(episode_return)

    environment = gym_wrapper.GymWrapper(env1)
    environment = wrappers.SinglePrecisionWrapper(environment)

    
    from acme.tf import savers as tf2_savers
    snapshotter = tf2_savers.Snapshotter(
    objects_to_save={'network': network}, time_delta_minutes=1.,directory= 'rl2_vanilla_weigths')
    snapshotter.save()

    num_episodes = 100  #@param

    mean_reward=[]

    # data = False
    # data_train_x,data_train_y = None,None
    print("second order testing")
    with open('rl2_testing_results.csv', 'a', newline='') as file:
        for episode in range(num_episodes):


    #     array = [-1,-2,-3,-4,-5,-6,-7,-8,-9,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    #     heat = 20 + np.random.choice(array, 1)[0]
    #     add = np.random.choice([-1,1])
    #     cool = (heat*2) - add*(np.random.choice(12,1)[0])

    #     tcl5 = first_temp_model(heat*60,cool*60,21,1.5)
    #     env2 = first_temp_env(tcl5,prices) 


    #     environment = gym_wrapper.GymWrapper(env2)
    #     environment = wrappers.SinglePrecisionWrapper(environment)

            episode_return=0
            timestep = environment.reset()

            i =0
            actor.observe_first(timestep)  # Note: observe_first.

            while not timestep.last():
                action = actor.select_action(timestep.observation)
              #  print(action)
                timestep = environment.step(action)
                if i >40:
                    episode_return += timestep.reward
                
                actor.observe(action=action, next_timestep=timestep)  # Note: observe.
                i+=1
        #     while _can_sample():

        #          _learner.step()

        #     while _can_sample():

        #           _learner.step()
           # (data_train_x,data_train_y) = env2.get_data()
            data = True
            writer = csv.writer(file)
            writer.writerow([run, episode, episode_return])
            print(episode, "return",episode_return)
            mean_reward.append(episode_return)
            #     if episode %40 ==0 and episode>0 :
            #             env2.train_nn()

        print("test_reward",np.mean(mean_reward))