import time
import gym
import numpy as np
from itertools import cycle
from thermalmodels_first_price import FirstOrderModel, FirstOrderThermostat
from gym import spaces
from gym.utils import seeding
import tensorflow as tf
import tensorflow_datasets as tfds
from acme.agents.tf import actors as actors
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
from auto_encoder_state import Autoencoder as ae_state
from auto_encoder_action import Autoencoder as ae_action
from jax.experimental import optimizers

from tensorflow import keras
from datetime import datetime

csv_logger = tf.keras.callbacks.CSVLogger(
    'fixed_logs.csv', separator=',', append=True
)
#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
encoder_state = ae_state(11)
encoder_action = ae_action(1)

state_initated = np.append([np.arange(13)], [np.arange(13)], axis=0)
acion_initated = np.append([np.arange(1)], [np.arange(1)], axis=0)

x_state = encoder_state(state_initated)
x_action = encoder_action(acion_initated)

input_x = np.append(x_state, x_action, axis=1) # Join state and action horizontally
train_x = input_x
train_y = np.append([1],[1],axis=0)
print('x',train_x.shape)
print(train_y.shape)

def random_layer_params(m, n, key, scale=1e-1):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


layer_sizes = [12, 1024, 512, 512, 128, 32, 1 ]
param_scale = 0.1
step_size = 0.001
num_epochs = 30
batch_size = 132
n_targets = 1
params = init_network_params(layer_sizes, random.PRNGKey(0))
print("yp")
from jax.scipy.special import logsumexp

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=6e-4), # 1e-3
               loss=tf.keras.losses.MeanSquaredError())


def relu(x):
  return jnp.maximum(0, x)

def predict(params, image):
  # per-example predictions
  activations = image
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = relu(outputs)
  
  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b 
  return logits                   #- logsumexp(logits) # No sigmoid required
# Let's upgrade it to handle batches using `vmap`

# Make a batched version of the `predict` function
batched_predict = vmap(predict, in_axes=(None, 0))

# Make a jit version of predict as well in case
jit_predict = jit(predict)

def mse(params, x, targets):
  target = targets
  predicted = batched_predict(params, x)
  return jnp.mean((targets - preds)**2)

@jit
def loss(params, images, targets):
  preds = batched_predict(params, images)
   # jnp.mean(jnp.square(targets - preds), axis=-1)
  deviation = targets - preds
  squared_error = deviation**2
  return jnp.mean(squared_error) #-jnp.mean((preds - targets)**2)

# @jit
# def update(params, x, y):
#   grads = grad(loss)(params, x, y)
#   return [(w - step_size * dw, b - step_size * db)
#           for (w, b), (dw, db) in zip(params, grads)]

@jit
def update(params, x, y, opt_state):
    """ Compute the gradient for a batch and update the parameters """
    value, grads = value_and_grad(loss)(params, x, y)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value

random_flattened_image = random.normal(random.PRNGKey(1), (10, 28 * 28))
preds = batched_predict(params, train_x)
print('single',train_x[-1].shape)
single_pred = jit_predict(params,train_x[-1])
#model_pred = model.predict_on_batch(train_x)
#preds = predict(params, train_x)
print(preds.shape)
print(single_pred.shape)
#print(model_pred.shape)

step_size = 5e-4
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(params)

class DirectControlEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    # Outside is 0C
    # OUT = 0
    OUT = 5
    # Step size is 15 minutes.
    DELTA = 15 * 60

    # Num steps = 1 day.
    STEPS = (24 * 60 * 60) / DELTA

    def __init__(self, model):

        assert isinstance(model, FirstOrderModel)

        # Store the model.
        self.model = model

        # DirectControlEnv domain has two actions, heating OFF and heating ON.
        self.action_space = spaces.Discrete(2)

        # DirectControlEnv domain has a continuous temperature range. Gaussian noise means temp can be infinitely far.
        low = np.array([np.float32(np.NINF)])  # Low
        high = np.array([np.float32(np.PINF)])
        prices = np.array([np.float32(np.PINF)])
        self.observation_space = spaces.Box(low, high,
                                            dtype=np.float32)  # The low and high array must have 3 elements now to represent prices

        # Initialize random number generator to a random seed.
        self.seed()

        # Initialize state empty (user must call reset() before step(action)).
        self.state = None

    def step(self, action):

        # Ensure valid action chosen.
        assert self.action_space.contains(action)

        # Use the model to advance the temperature and determine the reward.
        reward = self.model.comfortScore(self.state[0])
        self.state[0] = self.model.nextTemperature(self.state[0], self.OUT, self.DELTA, action)
        # self.state[1] = 10
        self.time = self.time + 1

        # Done after STEPS time steps.
        done = bool(self.time >= self.STEPS)

        return np.array(self.state), reward, done, {}

    def reset(self):

        # Reset time.
        self.time = 0

        # Set temperature to a random value between 19 and 21 Celsius.
        self.state = self.np_random.uniform(low=19, high=21, size=(1,))

        # Return initial state.
        return np.array(self.state)

    def render(self, mode='human'):
        if mode == 'human':
            print(self.state[0])
        else:
            super(DirectControlEnv, self).render(mode=mode)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class SetpointDeltaEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    # Outside is 0C
    OUT = 0

    # Step size is 15 minutes.
    DELTA = 15 * 60

    # Num steps = 1 day.
    STEPS = (24 * 60 * 60) / DELTA

    def __init__(self, model, prices_array):

        assert isinstance(model, FirstOrderModel)

        # Convert the model to thermostat control model.
        self.model = FirstOrderThermostat(model)
        

        
        # Prices array
        self.price_array = cycle(prices_array)
        self.price = next(self.price_array)[0]

        # SetpointDeltaEnv domain has five actions, change setpoint by {-1,-0.5,0,+0.5,+1}
        self.action_space = spaces.Discrete(5)

        low = np.array([np.float32(np.NINF), np.float32(np.NINF)])
        high = np.array([np.float32(np.PINF), np.float32(np.PINF)])

        #  self.observation_space = spaces.Box(low=0, high=255, shape=(96, 96, 3), dtype=np.uint8)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)  # Temp, SET POINT, Price in next time step

        # Initialize random number generator to a random seed.
        self.seed()

    #   self.price = 10

    def _process_data(self):

        prices = cycle(self.price_array)
        # self.price = prices
        return prices

    def step(self, action):

        # Ensure valid action chosen.
        assert self.action_space.contains(action)

        # Use the model to advance the temperature and determine the reward.
        setpointDelta = (action - 2) / 2
        self.model.adjustSetpointBy(setpointDelta)
        meantemp, SECSon, reward, _, _ = self.model.advance(self.OUT, self.DELTA,
                                                            self.price)  # _ sum of secondson is the first blank
        self.time = self.time + 1

        # Done after STEPS time steps.
        done = bool(self.time >= self.STEPS)
        price_next = next(self.price_array)[0]

        #    price_now = self._getObservation(meantemp)[2]
        self.price = price_next  # Next price becomes the current price which is used in the next step to calculate the reward
        obs_return = self._getObservation(meantemp)  # Gets the reward

        #    np.insert(obs_return,1,price_next)

        return obs_return, reward, done, {'SecondsON': SECSon}

    def reset(self):

        # Reset time.
        self.time = 0

        # Change set point to a random value.
        random_setpoint = self.np_random.random_integers(low=35, high=45) / 2
        random_temp = self.np_random.uniform(low=19, high=21, size=(1,))
        random_mode = self.np_random.random_integers(low=0, high=1)

        # Adjust model.
        self.model.setSetpoint(random_setpoint)
        self.model.tIn = random_temp
        self.model.mode = random_mode

        # Return initial state.
        return self._getObservation()

    def render(self, mode='human'):
        if mode == 'human':
            print([self.model.tIn[0], self.price])  # Price #Plotting
        # print (self.price)
        else:
            super(SetpointDeltaEnv, self).render(mode=mode)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _getObservation(self, temp=None):
        if (temp == None):
            temp = self.model.tIn

        # return np.array([temp, self.model.setpoint])
        return np.array([temp[0], self.price])


class DirectSetpointEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    # Outside is 0C

    OUT = 3  # More realistic -
    OUT = np.random.normal(5, 2, 1)
    # Step size is 15 minutes.
    #  DELTA = 15 * 60
    DELTA = 5 * 60  # 5 mins
    # Num steps = 1 day.
    STEPS = (24 * 60 * 60) / DELTA
    STEPS += 40 # Exploration time 
    def __init__(self, model, prices_array, actor , data = False ,data_train_x = None, data_train_y= None):

        
        assert isinstance(model, FirstOrderModel)  # Given model is an instance of the FirstorderModel
      
        # Convert the model to thermostat control model.
        self.model = FirstOrderThermostat(model)  #
        
        # Train anf test td arrays
        #if iteration == 0:
        if data:
           
            self.train_x = data_train_x
            self.train_y = data_train_y
        else:
            self.train_x = train_x
            self.train_y = train_y
#         print('bb',self.train_x.shape)
        
#         print('bb',self.train_y.shape)
        
       # self.train_y = train_y
#         if weights:
#             self.nn_params = nn_weights[0]
#             self.opt_state = nn_weights[1]
#         else:
#             self.nn_params = params
#             self.opt_state = opt_state          
        self.env_reward = 0
        self.previous_env_reward = 0
        self.action_previous = 0
        self.min_reward = 10
        self.max_reward = -10
       # self.dqn3 = tf.saved_model.load("saved_net_transfer/7a0c52b2-5744-11eb-9d3a-0242ac110006/snapshots/network")
       # self.dqn3 = actors.FeedForwardActor(policy_network=self.dqn3)
        self.dqn3 = actor
        
    #    self.training_data = np.array([])
        
        self.status = 0 # occupation
        self.iter = 0
       # self.policy_1 = policy_1 # The base policy
       # self.policy_2 = policy_2 # another policy
     #   self.policy_3 = policy_3 # The policy another one
        self.target = 21
        self.explore =0
        self.comfort = 0
        self.cost = 0
        self.secondsON = 0
        # Prices array
        
        self.price_array = cycle(prices_array)
        self.price = next(self.price_array)[0]
        self.price_2 = next(self.price_array)[0]
        self.price_3 = next(self.price_array)[0]
        self.price_4 = next(self.price_array)[0]
        # DirectSetpointEnv domain has 40 actions, range(10,30,0.5)
        self.action_space = spaces.Discrete(40)  #

        
        self.previous_temp1 = 20
        self.previous_temp2 = 20
        self.previous_temp3 = 20
        self.previous_temp4 = 20
        
        self.task1 = 1
        self.task2 = 0
        self.task3 = 0
        self.task4 = 0
        self.task5 = 0
        
        low = np.array(
            [np.float32(np.NINF)]*13) 
        high = np.array(
            [np.float32(np.PINF)]*13)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # Initialize random number generator to a random seed.
        self.seed()

    def step(self, action): # action = action to advance the current state, outputted already

        # Ensure valid action chosen.
        assert self.action_space.contains(action)

        # Use the model to advance the temperature and determine the reward.
        setpoint = 10 + action / 2
        #   setpoint = 20                       # For baseline cost calculations
        self.model.setSetpoint(setpoint)
        meantemp, self.secondsON, reward, _, _,self.cost,self.comfort = self.model.advance(self.OUT, self.DELTA, self.price,self.time)
        env_reward = reward
        
        self.previous_env_reward = self.env_reward   # Assign the previous reward to be as input to the RNN.
        
        self.env_reward = env_reward 
        self.time = self.time + 1
        
        self.previous_temp1 = self.previous_temp2
        self.previous_temp2 = self.previous_temp3
        self.previous_temp3 = self.previous_temp4
        self.previous_temp4 = meantemp
        
        self.explore = 0
        previous = self.price
        # Done after STEPS time steps.
        done = bool(self.time >= self.STEPS)

        self.price = self.price_2
        self.price_2 = self.price_3
        self.price_3 = self.price_4
        self.price_4 = next(self.price_array)[0]
#         q_network_input = self.output_td[:11]
#         print(q_network_input.shape)
    #    print(self.output_td.shape)
        q_actions_right = self.dqn3.select_action(self.output_td) # Get the q-values of all actions on the previous state, outputted already
        
        q_right = q_actions_right[action] # Get the q-value of action taken in the previous state (outputted)
                  
        output = self._getObservation(meantemp) # output = next state, Get the next state which becomes my current state (not-outputted)
            
        action_taken = action # Store the previous action taken for rendering
        self.action_previous = action_taken
        
        action = self.dqn3.select_action(output) # Get the q-values of all actions in the next state (current state, not-outputted)
        
        q_left = np.max(action)    # Get the q_value of the max q-value action in the current new state (not-outputted)
        
        #q_max = abs(q_max/100)  # Scaling
        
        left = reward + .95*q_left  # Calculating Reward recieved in the previous state + discnt*max_reward that can be achieved
                  
        td_error = (left - q_right)**2 # was **2
        
     #   x_state = encoder_state(np.array([self.output_td])) # Previous state (np array) encoded
        x_state = np.array([self.output_td])
        x_action = np.array([[action_taken]]) 
     #   x_action = encoder_action(np.array([action_taken,action_taken])) 
   
        x = np.append(x_state,x_action,axis=1)  # Concatenate state and action embedding
        
        y_td = td_error

        self.train_x = np.append(self.train_x, x , axis=0) # Add new x to the training data
        self.train_y = np.append(self.train_y,[td_error],axis=0) # add new y to the data
        
#         if self.iter == 0:
#             self.train_x = np.delete(self.train_x,0,0)
#             self.train_y = np.delete(self.train_y,0,0)
#             self.iter = 1
       # self.train_x = jnp.reshape(self.train_x, (len(self.train_x), 32))
#         print(self.train_x[-1].shape)
#         print(self.train_y.shape)
#        print(x.shape)
#        batched_x = np.expand_dims(self.train_x[-1], axis=0)
#        batched_pred = jit_predict(self.nn_params, x[0])   
        batched_x = np.expand_dims(x, axis=0)
        single_predict = np.array(model.predict_on_batch(batched_x))
#         print(self.train_x[-1].shape)
#         print('predict',single_predict.shape)
        error_pred = jnp.square(td_error-single_predict[0])
        
#         print(td_error)
#         print(self.train_x)
#         print('error',error_pred)
        if self.time <= 40:
            reward = 0.55*error_pred[0] + 0.05*reward   # Scaler reward   (0.55,0.05)  # 0.2
            self.explore = 1
            
        self.output_td = output  # Saving the current state as a previous state for the next iteration of td(0) calculation
        
        rl2_output = self._getRL2_state(meantemp)

        return rl2_output, reward, done, {'reward': env_reward,'SecondsON': self.secondsON, 'Price': previous,'Action':action_taken,'Cost' : self.cost,'Comfort': self.comfort,'MinReward':self.min_reward,'MaxReward':self.max_reward }
    

    def reset(self):

        # Reset time.
        self.time = 0

        # Change set point to a random value.
        random_setpoint = self.np_random.random_integers(low=35, high=45) / 2
        random_temp = self.np_random.uniform(low=19, high=21, size=(1,))
        random_mode = self.np_random.random_integers(low=0, high=1)

        # Adjust model.
        self.model.setSetpoint(random_setpoint)
        self.model.tIn = random_temp
        self.model.mode = random_mode

        # Return initial state.
        self.output_td = self._getObservation()
        return self._getRL2_state()

    def render(self, mode='human'):
        if mode == 'human':
            print(self.model.tIn)
        else:
            super(SetpointDeltaEnv, self).render(mode=mode)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_train_batches(self,arr,batch_size=256):
        
        ds = np.split(arr,batch_size)
        return ds
    
    def get_data(self,b=None):
#         batch(
#         batch_size, drop_remainder=False
#                     )
      #  print(self.train_x.shape)
        return (self.train_x, self.train_y)
       # train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
     #   print("yoyoyooy")
        
    def get_train_batches(self):
      # as_supervised=True gives us the (image, label) as a tuple instead of a dict
      print(self.train_x.shape)
      ds = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y))
      # You can build up an arbitrary tf.data input pipeline
      ds = ds.batch(batch_size=512).prefetch(1)
      # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
      return tfds.as_numpy(ds)
        
        
    def train_nn(self,num_epochs=20):
        data_training = self.get_train_batches()
        BATCH_SIZE = 1024
        SHUFFLE_BUFFER_SIZE = 100
        train_dataset = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y))                
        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)   
        model.fit(train_dataset, epochs=25,callbacks=[csv_logger])
#         i = 0
#         train_dataset = tfds.as_numpy(train_dataset)
#         for epoch in range(num_epochs):
#               start_time = time.time()
              
#               for x, y in train_dataset:
#                 i+=1
#            #     self.params = update(self.params, x, y)
#                 self.nn_params, self.opt_state, loss = update(self.nn_params, x, y, self.opt_state)
# #                 print(i)
#               epoch_time = time.time() - start_time

#               train_loss = mse(self.nn_params, x, y)
#           #    test_acc = mse(self.nn_params, self.train_x, self.train_y)
#               print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
#               print("Training set loss {}".format(loss))
            # print("Test set accuracy {}".format(test_acc))

    
    def _getObservation(self, temp=None):
        if (temp == None):
            temp = self.model.tIn

           
        observation_list = np.array([temp, self.price, self.price_2, self.price_3, self.price_4, self.secondsON, self.cost,self.comfort, self.time,self.status,self.target],dtype=object) 
        #, self.action_previous, self.env_reward 
        return np.asarray(observation_list).astype('float32')
    #np.asarray(observation_list).astype('float32')
    
    def _getRL2_state(self, temp=None):
        if (temp == None):
            temp = self.model.tIn

           
        observation_list = np.array([temp, self.price, self.price_2, self.price_3, self.price_4, self.secondsON, self.cost,self.comfort, self.time, self.status, self.target, self.action_previous, self.previous_env_reward ],dtype=object) 
        #, self.action_previous, self.env_reward 
        return np.asarray(observation_list).astype('float32')  
        