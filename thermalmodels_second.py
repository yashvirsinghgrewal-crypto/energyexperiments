import math
import numpy as np
from gym.utils import seeding
from constants_glm import ConstantsGLM
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
    def __init__(self, secondsToHeat, secondsToCool, goal = 20, flex = 0.5, noise = None):
        
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


class SecondOrderModel(object):

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
    def __init__(self, pStepSeconds=10,pHouse=1):
        
        # Get the building constants
        self.ConstantsGLM = ConstantsGLM(pHouse)

        self.Ca    = self.ConstantsGLM.computeAirThermalMass();
        self.Cm    = self.ConstantsGLM.computeMassThermalMass();
        self.Ua    = self.ConstantsGLM.computeUA();
        self.Hm    = self.ConstantsGLM.computeHeatTransfer();

        self.fHeat = self.ConstantsGLM.computeDesignHeating(self.Ua);
        self.fAux  = self.ConstantsGLM.computeDesignAux(self.Ua);
        self.fFan  = self.ConstantsGLM.computeFanPower(self.fHeat, self.fAux) / 1000;    # W to kW.
        self.fCOP  = self.ConstantsGLM.getHeatingCOP();

        self.BTUPHPW  = (3.4120);         #// BTUPH/W
        self.BTUPHPKW = (1e3 * 3.4120);   #// BTU / H / kW
        self.KWPBTUPH = (1e-3/self.BTUPHPW);   #// kW/BTUPH


        a  = self.Cm*self.Ca/self.Hm;
        b  = self.Cm*(self.Ua+self.Hm)/self.Hm+self.Ca;
        c  = self.Ua;

        rr = math.sqrt(b*b-4*a*c)/(2*a);
        r  = -b/(2*a);

        self.c1 = -(self.Ua + self.Hm)/self.Ca;
        self.c2 = self.Hm/self.Ca;
        self.r1 = r+rr;
        self.r2 = r-rr;
        self.A3 = self.Ca/self.Hm * self.r1 + (self.Ua+self.Hm)/self.Hm;
        self.A4 = self.Ca/self.Hm * self.r2 + (self.Ua+self.Hm)/self.Hm;

        self.setStepsize(pStepSeconds);

    def setStepsize(self,stepSeconds):
        dt = stepSeconds / 3600;  #// Fraction of an hour.
        self.stepsize = stepSeconds;
        
        self.e_r1dt = math.exp(self.r1*dt);
        self.e_r2dt = math.exp(self.r2*dt);


    ''' Inputs:
     *  - outdoor temperature
     *  - indoor temperatures {'mass':, 'air':}
     *  - internal heat gain (e.g., occupants, appliances using power)
     *  - control action.
     *
     * Outputs:
     *  - Power used by HVAC system to control temperature (kW)
     * 
     * Changed state:
     *  - indoor temperatures <mass, air>
     */ '''

    def advanceTime(self,Tout, TMassAir, Qi,  action, noise=0): 

       # // Temperature conversion to use Fahrenheit internally.
        Tout = self.toFahrenheit(Tout);
        TMassAir['MASS'] = self.toFahrenheit(TMassAir['MASS']);
        TMassAir['AIR'] = self.toFahrenheit(TMassAir['AIR']);

       # // Select functionality based on the provided action.
        powerKw = 0;

        assert action in range(3)

        if action == 0: # 0 = OFF
            powerKw = self.advanceOff(Tout,TMassAir,Qi,noise)

        elif action == 1: # 1 = ON_heat
            powerKw = self.advanceHeat(Tout,TMassAir,Qi, noise)

        elif action == 2: #2 =ON_Cool
            powerKw =self.advanceCool(Tout,TMassAir,Qi,noise)

        

     #   // Restore temperature unit to Celsius after time step.
        TMassAir['MASS'] = self.toCelsius(TMassAir['MASS']);
        TMassAir['AIR'] = self.toCelsius(TMassAir['AIR']);

        return powerKw;

    def advanceHeat(self,pTout, TMassAir, Qi, noise): 
        self.advance(pTout, TMassAir, Qi, self.computeHeatingBTU(pTout), noise);

        return self.computeHeatingKW(pTout);
    

    def computeHeatingBTU(self,pTout): 
        return self.fFan * self.BTUPHPKW + self.computeHeatingCAP(pTout);
    

    def computeHeatingKW(self,pTout):
        lCOP = self.computeHeatingCOP(pTout);
        lCap = self.computeHeatingCAP(pTout);

        return ((lCap / lCOP) * self.KWPBTUPH + self.fFan);
    

    def computeHeatingCOP(self,pTout):
        lTout = min(80, pTout);

        return (self.fCOP-1) / (2.03914613 - 0.03906753*lTout + 0.00045617*lTout*lTout - 0.00000203*lTout*lTout*lTout);
    

    def computeHeatingCAP(self,pTout):
        return self.fHeat;
    
    def advanceCool(self,pTout, TMassAir, Qi, noise) :
        self.advance(pTout, TMassAir, Qi, self.computeCoolingBTU(pTout),noise);

        return self.computeCoolingKW(pTout);
    

    def computeCoolingBTU(self,pTout) :
        return self.fFan * self.BTUPHPKW - self.computeCoolingCAP(pTout);
    

    def computeCoolingKW(self,pTout):
        lCOP = self.computeCoolingCOP(pTout);
        lCap = self.computeCoolingCAP(pTout);

        return ((lCap / lCOP) * self.KWPBTUPH + self.fFan);
    

    def computeCoolingCOP(self,pTout):
        lTout = max(40, pTout);

        return self.fCOP / (-0.01363961 + 0.01066989*lTout);
    
    def computeCoolingCAP( self,pTout) :
        return self.fHeat
    

    def advanceOff(self,Tout, TMassAir, Qi, noise): 
        self.advance(Tout, TMassAir, Qi, 0,noise);

        return 0;

    def advance( self,Tout, TMassAir, Qi, Qh, noise): 

        Qa = Qh + 0.5*Qi;
        Qm = 0.5*Qi;
        Teq = (Qa+Qm)/self.Ua + Tout;

        dTa = self.c2*TMassAir['MASS'] + self.c1*TMassAir['AIR'] - (self.c1+self.c2)*Tout + Qa/self.Ca;
        k1  = (self.r2*TMassAir['AIR'] - self.r2*Teq - dTa)/(self.r2-self.r1);
        k2  = TMassAir['AIR'] - Teq - k1;

        e1 = k1*self.e_r1dt;
        e2 = k2*self.e_r2dt;

        TMassAir['AIR'] = e1 + e2 + Teq + np.random.normal(0,noise,1)[0] # Noise is the std deviation
        TMassAir['MASS'] = self.A3*e1 + self.A4*e2 + Qm/self.Hm + Teq;
      #  print('modl:',TMassAir)
        TMassAir['AIR'] = TMassAir['AIR']
        TMassAir['MASS'] =TMassAir['MASS']
        
        self.TMassAir = TMassAir
    
    def comfortScore(self, tempIn,goal,deadband):

        tempIn = tempIn['AIR']
        # Distance from the goal temperature?
        
        distance = math.fabs(tempIn - goal)

        # Apply comfort deadband.
        error = max(distance - deadband, 0)

        # Squared error penalty term.
        return -error * error

    def nextTemperature(self):


        return self.TMassAir


    def toFahrenheit(self,celsius):

        return 32+(1.8*celsius);
        

    def toCelsius(self,fahrenheit):
        return (fahrenheit-32)/1.8 





class SecondOrderThermostat(object):

    def __init__(self,model,setpoint = 25,deadband = 1.5,mode=0,goal=21,initAir=19.5,initMass=19.5/2):
    
        self.MAX_SETPOINT = 30
        self.MIN_SETPOINT = 10
    
        self.thermalmodel = model;
        self.setSetpoint(setpoint)
        self.setpoint = setpoint
        self.deadband = deadband
        self.mode = mode
       # self.temperature = {'MASS' :initMass,'AIR':initAir}
    
        self.goal = goal
        self.tIn = {'MASS' :initMass,'AIR':initAir}
    
        

  #  public SecondOrderThermostat(SecondOrderModel model, Random rnd) {
  #      this(model, 20D, 0.5D, rnd.nextBoolean(), 19.5D + rnd.nextDouble(), 19.75D + rnd.nextDouble() / 2);
  #  }

  #  def SecondOrderThermostat(SecondOrderModel model, double setpoint, double deadband, boolean initMode, double initAir, double initMass):
  #      self.model = model;
  #      this.temperature = new double[] { initMass, initAir };
  #      this.setpoint = setpoint;
  #      this.deadband = deadband;
  #      this.mode = initMode;
    

    def getTemperature(self): 
        return self.temperature['AIR'];
    

    def adjustSetpoint(self,delta):
        self.setSetpoint(self.setpoint + delta);
    

    def setSetpoint(self,newSetpoint): 
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
        #Updated and more challenging
        stepsize = 10
        alpha = 5.0  # Weight on cost level
        beta = 1
        if (next_price >= 80):
            beta = 1  # was 0.1
            alpha = alpha ** 2.0
        if (self.tIn['AIR'] < 17.5 or self.tIn['AIR'] > 25):
            beta = beta * 0.5
            aplha = alpha* 0.5
        price_kw = price / 1000
        price_kw_sec = price_kw / 3600  ### minutes

        sumSecsOn = 0
        sumComfort = 0
        meanTemp = 0;

        for t in range(0, deltaSecs, stepsize):
            # Hysteresis control, change direction when exceeded deadband. 
            if (self.mode == 1 and self.tIn['AIR'] > self.setpoint + self.deadband):
                self.mode = 0
            elif (self.mode == 0 and self.tIn['AIR'] < self.setpoint - self.deadband):
                self.mode = 1

            
            meanTemp += self.tIn['AIR'] * stepsize
            self.thermalmodel.setStepsize(min(stepsize, deltaSecs-t));
            sumComfort += self.thermalmodel.comfortScore(self.tIn,self.goal,self.deadband)
            power = self.thermalmodel.advanceTime(tOut, self.tIn, 0, self.mode,noise);

            self.tIn = self.thermalmodel.nextTemperature()
            sumSecsOn += self.mode

        
        
        total_cost = (price_kw_sec * 0.5) * sumSecsOn  # .5 kw consumption per second # can be tweeked as well
        cost = (total_cost - min_cost) / (max_cost - min_cost)

        comfort = (sumComfort - min_comfort) / (max_comfort - min_comfort)
     #   reward = -0.017872 + beta * comfort
        reward = -0.009*alpha + beta*comfort
       # print(reward) 
     #   reward = max(reward,0)
        reward = min(reward,1)
        return meanTemp / deltaSecs, sumSecsOn, reward, self.setpoint, self.mode,alpha * cost,comfort

          
    


class FirstOrderThermostat(object):

    MIN_SETPOINT = 10
    MAX_SETPOINT = 30

    def __init__(self, model, setpoint = 20, deadband = 0.5, tIn = 20, mode = 0):

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

    def advance(self, tOut, deltaSecs):

        stepsize = 10

        sumSecsOn = 0
        sumComfort = 0
        meanTemp = 0;

        for _ in range(0, deltaSecs, stepsize):
            # Hysteresis control, change direction when exceeded deadband. 
            if (self.mode == 1 and self.tIn > self.setpoint + self.deadband):
                self.mode = 0
            elif (self.mode == 0 and self.tIn < self.setpoint - self.deadband):
                self.mode = 1

            meanTemp += self.tIn * stepsize
            sumComfort += self.thermalmodel.comfortScore(self.tIn)
            self.tIn = self.thermalmodel.nextTemperature(self.tIn, tOut, stepsize, self.mode)
            sumSecsOn += self.mode

        return meanTemp / deltaSecs, sumSecsOn, sumComfort, self.setpoint, self.mode
