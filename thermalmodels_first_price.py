import math

from gym.utils import seeding
import numpy as np

"""
  First-order thermal model, approximating the temperature of a system, such as
  a house, a thermal zone, or a refrigerator by a single state variable (the
  indoor temperature).

  Model due Mortensen and Haggerty, "A stochastic computer model for heating and
  cooling loads", IEEE Transactions on Power Systems (3:3), 1988

  :by Frits de Nijs
"""


class FirstOrderModel(object):
    """
    Constructor; computes thermal parameters P, C, and R (R is always 1) on the basis
    of four quantities of interest, namely:

    :param secondsToHeat: the time it takes for the indoor temperature to increase
        from (goal - range) to (goal + range) if the outdoor temperature is 0C.
    :param secondsToCool: the time it takes for the indoor temperature to decrease
        from (goal + range) to (goal - range) if the outdoor temperature is 0C.
    :param goal: the ideal temperature of the object under control (default 20C).
    :param flex: the size of the 'deadband' range, where comfort is equal to the
        goal comfort. (default 1C wide).
    :param noise: standard deviation of the normally distributed noise term.

    """

    def __init__(self, secondsToHeat, secondsToCool, goal=20, flex=0.5, noise=None):

        # Check preconditions, strictly positive durations.
        assert secondsToHeat > 0 and secondsToCool > 0

        # Store the goal temperature and comfortable range.
        self.goal = goal
        self.deadband = flex
        self.sigma = noise
        self.np_random = None

        # Compute alpha component.
        lower = goal - flex
        upper = goal + flex
        alpha = self.__computeAlpha(lower, upper, secondsToCool)

        # Compute thermal constants.
        self.C = self.__computeCapacitance(alpha)
        self.P = self.__computePower(lower, upper, alpha, secondsToHeat)

        # Initialize random number generator, if needed
        if noise != None:
            self.seed()

    def __computeAlpha(self, lower, upper, time):
        return math.pow(lower / upper, 1 / time)

    def __computeCapacitance(self, alpha):
        return (-1 / 3600) / math.log(alpha)

    def __computePower(self, lower, upper, alpha, time):
        return (lower * math.pow(alpha, time) - upper) / (math.pow(alpha, time) - 1)

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)

    def nextTemperature(self, tempIn, tempOut, deltaSecs, power):

        # Check that the time delta is positive, and power is within 'on / off' range.
        assert deltaSecs > 0 and 0 <= power <= 1

        alpha = math.exp((-deltaSecs / 3600) / self.C);
        noise = 0;

        if self.sigma != None and self.sigma > 0:
            noise = self.np_random.normal(0, self.sigma);

        return alpha * tempIn + (1 - alpha) * (tempOut + power * self.P) + noise

    def comfortScore(self, tempIn):

        # Distance from the goal temperature?
        distance = math.fabs(tempIn - self.goal)

        # Apply comfort deadband.
        error = max(distance - self.deadband, 0)

        # Squared error penalty term.
        return -error * error
    #  return -(set_point*price)


class FirstOrderThermostat(object):
    MIN_SETPOINT = 10
    MAX_SETPOINT = 30

    def __init__(self, model, setpoint=20, deadband=0.5, tIn=20, mode=0):

        # Construct object by storing arguments.
        self.thermalmodel = model
        self.setSetpoint(setpoint)
        self.deadband = deadband
        self.tIn = tIn
        self.mode = mode

        # Guarantee controllability of thermal model.
        self.thermalmodel.sigma = None

    def adjustSetpointBy(self, delta):

        self.setSetpoint(self.setpoint + delta)

    def setSetpoint(self, newSetpoint):

        # Set.
        self.setpoint = newSetpoint

        # Clip to range.
        self.setpoint = min(self.MAX_SETPOINT, self.setpoint);
        self.setpoint = max(self.MIN_SETPOINT, self.setpoint);

    def advance(self, tOut, deltaSecs, next_price,time,noise=0):  # Next price

        max_cost = 0.0040
        min_cost = 0
        min_comfort = -2600
        max_comfort = 0

        price = max(next_price,0)  #Avoid -ve prices

        stepsize = 10
       
        alpha = 5.0  # Weight on cost level
        beta = 1
        if (next_price >= 80):
            beta = 1  # was 0.1
            alpha = alpha ** 2.0
        if (self.tIn < 17.5 or self.tIn > 25):
            beta = beta * 0.5
            aplha = alpha* 0.5
        price_kw = price / 1000
        price_kw_sec = price_kw / 3600  ### minutes

        sumSecsOn = 0
        sumComfort = 0
        meanTemp = 0;

        # SumSecson calculates the seconds it's on for
        for _ in range(0, deltaSecs, stepsize):  # Take steps 10 seconds long and for 0 to 15 min  take step of 10 secs
            # Hysteresis control, change direction when exceeded deadband.
            if (self.mode == 1 and self.tIn > self.setpoint + self.deadband):  # Mode = 1 means heating
                self.mode = 0
            elif (self.mode == 0 and self.tIn < self.setpoint - self.deadband):  # Mode = 0 means not heating
                self.mode = 1

            meanTemp += self.tIn * stepsize
            sumComfort += self.thermalmodel.comfortScore(self.tIn)
            self.tIn = self.thermalmodel.nextTemperature(self.tIn, tOut, stepsize, self.mode)
            sumSecsOn += self.mode  #

        total_cost = (price_kw_sec * 0.5) * sumSecsOn  # .5 kw consumption per second # can be tweeked as well
        cost = (total_cost - min_cost) / (max_cost - min_cost)

        comfort = (sumComfort - min_comfort) / (max_comfort - min_comfort)
       # reward = -0.017872 + beta * comfort
        reward = -alpha*cost + beta*comfort
         
        reward = min(reward,1)
      #  reward = max(reward,-1)
        #  reward = -(alpha*cost - beta*comfort)
        # reward = np.round(reward)
        # reward = cost
        
      #  if (price >= 70 and sumSecsOn == 0 and comfort == 1.0) :
      #      reward = reward*2
        
        
        return meanTemp / deltaSecs, sumSecsOn, reward, self.setpoint, self.mode,cost,comfort

#      return meanTemp / deltaSecs, sumSecsOn, sumComfort, self.setpoint, self.mode