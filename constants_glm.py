import math as Math
import numpy as np
class ConstantsGLM(object):

	def __init__(self, p=1):

	 #	Numeric constants.
		 
		self. number_of_stories						= 1.0;							# #
		self. aspect_ratio							= 1.5;							# width / depth
		self. floor_area								= 2500.0;						# feet^2
		self. ceiling_height							= 8.0;							# feet
		self. window_wall_ratio						= 0.15;							# % / 100
		self. number_of_doors							= 4.0;							# #
		self. total_thermal_mass_per_floor_area		= 2.0;							# ?   -> lb / feet^2 ?
		self. interior_surface_heat_transfer_coeff	= 1.46;							# ?
		self. interior_exterior_wall_ratio			= 1.5;							# Based partions for six rooms per floor
		self. air_density								= 0.0735;						# density of air [lb/cf]
		self. air_heat_capacity						= 0.2402;						# heat capacity of air @ 80F [BTU/lb/F]
		self. glazing_shgc							= 0.67;
		self. window_exterior_transmission_coefficient = 0.6;
		self. over_sizing_factor						= 0;
		self. latent_load_fraction					= 0.3;
		self. cooling_design_temperature				= 95;
		self. design_cooling_setpoint					= 75;
		self. design_peak_solar						= 195.0;
		self. design_heating_setpoint					= 70;
		self. heating_design_temperature				= 0;
		self. cooling_supply_air_temp					= 50.0;
		self. heating_supply_air_temp					= 150.0;
		self. duct_pressure_drop						= 0.5;

		
		 #		Derived constants.
		 
		self. volume				   = self.ceiling_height * self.floor_area;					# volume of air [cf]
		self. air_mass			   = self.air_density    * self.volume;						# mass of air [lb]
		self. door_area              = self.number_of_doors * 3.0 * 78.0 / 12.0; 			# 3 #' wide by 78" tall
		self. gross_wall_area        = 2.0 * self.number_of_stories * (self.aspect_ratio + 1.0) * self.ceiling_height * Math.sqrt(self.floor_area/self.aspect_ratio/self.number_of_stories);
		self. window_area            = self.gross_wall_area * self.window_wall_ratio;
		self. net_exterior_wall_area = self.gross_wall_area - self.window_area - self.door_area;
		self. exterior_ceiling_area  = self.floor_area / self.number_of_stories;
		self. exterior_floor_area    = self.floor_area / self.number_of_stories;

		self. air_thermal_mass       = 3 * self.air_heat_capacity * self.air_mass;
		self. mass_thermal_mass      = self.total_thermal_mass_per_floor_area * self.floor_area - 2 * self.air_heat_capacity * self.air_mass;
		self. heat_transfer_coeff    = self.interior_surface_heat_transfer_coeff * ((self.gross_wall_area - self.window_area - self.door_area)
																 + self.gross_wall_area * self.interior_exterior_wall_ratio
																 + self.number_of_stories * self.exterior_ceiling_area);

		self. design_internal_gains  = 167.09 * Math.pow(self.floor_area, 0.442);
		self. solar_heatgain_factor  = self.window_area * self.glazing_shgc * self.window_exterior_transmission_coefficient;


		parameters = np.array([[30, 19, 11,  3, 1/0.6, 1],[30, 19, 22,  5, 1/0.47, 0.5],[48, 22, 30, 11, 1/0.31, 0.5]])
		assert p in range(3)

		self.fRroof = parameters[p,0]
		self.fRwall =  parameters[p,1]
		self.Rfloor = parameters[p,2]
		self.fRdoors = parameters[p,3]
		self.fRwindows = parameters[p,4]
		self.fAirchange = parameters[p,5]

		self.fHeatingCOP = 3.5

	def computeAirThermalMass(self):
	
		return self.air_thermal_mass;
	

	def computeMassThermalMass(self):
	
		return self.mass_thermal_mass;
	
	
	def computeUA(self):

#		

		change_UA = self.fAirchange * self.volume * self.air_density * self.air_heat_capacity

		envelope_UA = float(self.exterior_ceiling_area ) / float(self.fRroof) + self.exterior_floor_area    / self.Rfloor + self.net_exterior_wall_area / self.fRwall   + self.window_area  / self.fRwindows + self.door_area / self.fRdoors


		return change_UA + envelope_UA



	

	def computeHeatTransfer(self):
	
		return self.heat_transfer_coeff;
	

	def computeDesignHeating(self,pUA):
	
		round_value = 0.0;
		design_heating_capacity = (1.0 + self.over_sizing_factor) * (1.0 + self.latent_load_fraction) * ((pUA) * (self.cooling_design_temperature - self.design_cooling_setpoint) + self.design_internal_gains + (self.design_peak_solar * self.solar_heatgain_factor));
		round_value = (design_heating_capacity) / 6000.0;
		design_heating_capacity = Math.ceil(round_value) * 6000.0; # design_heating_capacity is rounded up to the next 6000 btu/hr

		return design_heating_capacity;
	

	def computeDesignAux( self,pUA):
	
		aux_heat_capacity = (1.0 + self.over_sizing_factor) * (pUA) * (self.design_heating_setpoint - self.heating_design_temperature);
		round_value = (aux_heat_capacity) / 10000.0;
		aux_heat_capacity = Math.ceil(round_value) * 10000.0; # aux_heat_capacity is rounded up to the next 10,000 btu/hr

		return aux_heat_capacity;
	

	def computeFanPower( self,pDesignHeat,pDesignAux):
		
		
		 design_heating_cfm = max(pDesignHeat,pDesignAux)  / (self.air_density * self.air_heat_capacity * (self.heating_supply_air_temp - self.design_heating_setpoint)) / 60.0;

		 design_cooling_cfm = pDesignHeat / (1.0 + self.latent_load_fraction) / (self.air_density * self.air_heat_capacity * (self.design_cooling_setpoint - self.cooling_supply_air_temp)) / 60.0;
		 fan_design_airflow = max(design_heating_cfm,design_cooling_cfm) ;
		 roundval = Math.ceil((0.117 * self.duct_pressure_drop * fan_design_airflow / 0.42 / 745.7)*8);
		 fan_design_power = roundval / 8.0 * 745.7 / 0.88; # fan rounds to the nearest 1/8 HP

		 return fan_design_power

	def getHeatingCOP(self):
		
		return self.fHeatingCOP
	
