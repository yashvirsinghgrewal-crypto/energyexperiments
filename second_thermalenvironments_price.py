import gym
import numpy as np
from itertools import cycle
from thermalmodels_second_price import SecondOrderModel, SecondOrderThermostat
from gym import spaces
from gym.utils import seeding


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
    OUT = 6  # More realistic -
    
   # OUT = np.random.normal(5, 4, 1)
    
    # Step size is 15 minutes.
    #  DELTA = 15 * 60
    DELTA = 5 * 60  # 5 mins
    # Num steps = 1 day.
    STEPS = (24 * 60 * 60) / DELTA
    mean_temp = []
   

    def __init__(self, model,prices_array):

        assert isinstance(model,SecondOrderModel )  # Given model is an instance of the SecondOrderModel
      
        # Convert the model to thermostat control model.
        self.model = SecondOrderThermostat(model)  #

        self.status = 0 # occupation
        
       # self.policy_1 = policy_1 # The base policy
       # self.policy_2 = policy_2 # another policy
     #   self.policy_3 = policy_3 # The policy another one
        self.target = 21
        
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
        
        self.task1 = 0
        self.task2 = 1
        self.task3 = 0
        self.task4 = 0
        self.task5 = 0
        
        low = np.array(
            [np.float32(np.NINF)]*20)
        high = np.array(
            [np.float32(np.PINF)]*20)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.seed()

    def step(self, action):

    #    if self.time in (100,230,40,150):
    #        self.OUT= np.random.normal(self.OUT, 1, 1)[0]
        
       # Ensure valid action chosen.
        assert self.action_space.contains(action)

        # Use the model to advance the temperature and determine the reward.
        setpoint = 10 + action / 2
        self.model.setSetpoint(setpoint)
        meantemp, self.secondsON, reward, _, _,self.cost,self.comfort = self.model.advance(self.OUT, self.DELTA, self.price,self.status)
        self.time = self.time + 1
        
        self.previous_temp1 = self.previous_temp2
        self.previous_temp2 = self.previous_temp3
        self.previous_temp3 = self.previous_temp4
        self.previous_temp4 = meantemp
        
        
        
        
        previous = self.price
        # Done after STEPS time steps.
        done = bool(self.time >= self.STEPS)

        self.price = self.price_2
        self.price_2 = self.price_3
        self.price_3 = self.price_4
        self.price_4 = next(self.price_array)[0]

        output = self._getObservation(meantemp)
        
        action_taken = action
        
        return output, reward, done, {'SecondsON': self.secondsON, 'Price': previous,'Action':action_taken,'Cost' : self.cost,'Comfort': self.comfort }
    
    def reset(self):

        # Reset time.
        self.time = 0

        # Change set point to a random value.
        random_setpoint = np.random.random_integers(low=35, high=45)/2
        random_temp = np.random.uniform(low=23, high=26, size=(1,))[0]
        random_mode = np.random.random_integers(low=0, high=1)

        # Adjust model.
        self.model.setSetpoint(random_setpoint)
        self.model.tIn['AIR'],self.model.tIn['MASS'] = random_temp,random_temp/2
        self.model.mode = random_mode

        # Return initial state.
        return self._getObservation()

    def render(self, mode='human'):
        if mode == 'human':
            print(self.model.tIn)
        else:
            super(SetpointDeltaEnv, self).render(mode=mode)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _getObservation(self, temp = None):
        if (temp == None):
            temp = self.model.tIn
        
        return np.array([self.model.tIn['AIR'],self.price, self.price_2, self.price_3, self.price_4, self.secondsON, self.cost,self.comfort, self.time,self.status,self.target,self.previous_temp1,self.previous_temp2,self.previous_temp3,self.previous_temp4,self.task1,self.task2,self.task3,self.task4,self.task5]) 
       # return np.array([self.model.tIn['AIR'],self.OUT])
