import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import NamedTuple, List, Union

from simulator.utils import EulerInt, ShipDraw
from simulator.configs_ import ShipConfiguration, SimulationConfiguration, EnvironmentConfiguration, MachinerySystemConfiguration, ShipMachineryModel

class BaseShipModel:
    def __init__(
            self, ship_config: ShipConfiguration,
            simulation_config: SimulationConfiguration,
            environment_config: EnvironmentConfiguration
    ):

        payload = 0.9 * (ship_config.dead_weight_tonnage - ship_config.bunkers)
        lsw = ship_config.dead_weight_tonnage / ship_config.coefficient_of_deadweight_to_displacement \
              - ship_config.dead_weight_tonnage
        self.mass = lsw + payload + ship_config.bunkers + ship_config.ballast

        self.l_ship = ship_config.length_of_ship  # 80
        self.w_ship = ship_config.width_of_ship  # 16.0
        self.x_g = 0
        self.i_z = self.mass * (self.l_ship ** 2 + self.w_ship ** 2) / 12

        # zero-frequency added mass
        self.x_du, self.y_dv, self.n_dr = self.set_added_mass(ship_config.added_mass_coefficient_in_surge,
                                                              ship_config.added_mass_coefficient_in_sway,
                                                              ship_config.added_mass_coefficient_in_yaw)

        self.t_surge = ship_config.mass_over_linear_friction_coefficient_in_surge
        self.t_sway = ship_config.mass_over_linear_friction_coefficient_in_sway
        self.t_yaw = ship_config.mass_over_linear_friction_coefficient_in_yaw
        self.ku = ship_config.nonlinear_friction_coefficient__in_surge  # 2400.0  # non-linear friction coeff in surge
        self.kv = ship_config.nonlinear_friction_coefficient__in_sway  # 4000.0  # non-linear friction coeff in sway
        self.kr = ship_config.nonlinear_friction_coefficient__in_yaw  # 400.0  # non-linear friction coeff in yaw

        # Environmental conditions
        self.vel_c = np.array([environment_config.current_velocity_component_from_north,
                               environment_config.current_velocity_component_from_east,
                               0.0])
        self.wind_dir = environment_config.wind_direction
        self.wind_speed = environment_config.wind_speed

        # Initialize states
        self.north = simulation_config.initial_north_position_m
        self.east = simulation_config.initial_east_position_m
        self.yaw_angle = simulation_config.initial_yaw_angle_rad
        self.forward_speed = simulation_config.initial_forward_speed_m_per_s
        self.sideways_speed = simulation_config.initial_sideways_speed_m_per_s
        self.yaw_rate = simulation_config.initial_yaw_rate_rad_per_s

        # Initialize differentials
        self.d_north = 0
        self.d_east = 0
        self.d_yaw = 0
        self.d_forward_speed = 0
        self.d_sideways_speed = 0
        self.d_yaw_rate = 0

        # Set up integration
        self.int = EulerInt()  # Instantiate the Euler integrator
        self.int.set_dt(simulation_config.integration_step)
        self.int.set_sim_time(simulation_config.simulation_time)

        # Instantiate ship draw plotting
        self.drw = ShipDraw()  # Instantiate the ship drawing class
        self.ship_drawings = [[], []]  # Arrays for storing ship drawing data

        # Setup wind effect on ship
        self.rho_a = 1.2
        self.h_f = 8.0  # mean height above water seen from the front
        self.h_s = 8.0  # mean height above water seen from the side
        self.proj_area_f = self.w_ship * self.h_f  # Projected are from the front
        self.proj_area_l = self.l_ship * self.h_s  # Projected area from the side
        self.cx = 0.5
        self.cy = 0.7
        self.cn = 0.08

    def set_added_mass(self, surge_coeff, sway_coeff, yaw_coeff):
        ''' Sets the added mass in surge due to surge motion, sway due
            to sway motion and yaw due to yaw motion according to given coeffs.

            args:
                surge_coeff (float): Added mass coefficient in surge direction due to surge motion
                sway_coeff (float): Added mass coefficient in sway direction due to sway motion
                yaw_coeff (float): Added mass coefficient in yaw direction due to yaw motion
            returns:
                x_du (float): Added mass in surge
                y_dv (float): Added mass in sway
                n_dr (float): Added mass in yaw
        '''
        x_du = self.mass * surge_coeff
        y_dv = self.mass * sway_coeff
        n_dr = self.i_z * yaw_coeff
        return x_du, y_dv, n_dr

    def get_wind_force(self):
        ''' This method calculates the forces due to the relative
            wind speed, acting on the ship in surge, sway and yaw
            direction.

            :return: Wind force acting in surge, sway and yaw
        '''
        uw = self.wind_speed * np.cos(self.wind_dir - self.yaw_angle)
        vw = self.wind_speed * np.sin(self.wind_dir - self.yaw_angle)
        u_rw = uw - self.forward_speed
        v_rw = vw - self.sideways_speed
        gamma_rw = -np.arctan2(v_rw, u_rw)
        wind_rw2 = u_rw ** 2 + v_rw ** 2
        c_x = -self.cx * np.cos(gamma_rw)
        c_y = self.cy * np.sin(gamma_rw)
        c_n = self.cn * np.sin(2 * gamma_rw)
        tau_coeff = 0.5 * self.rho_a * wind_rw2
        tau_u = tau_coeff * c_x * self.proj_area_f
        tau_v = tau_coeff * c_y * self.proj_area_l
        tau_n = tau_coeff * c_n * self.proj_area_l * self.l_ship
        return np.array([tau_u, tau_v, tau_n])

    def three_dof_kinematics(self):
        ''' Updates the time differientials of the north position, east
            position and yaw angle. Should be called in the simulation
            loop before the integration step.
        '''
        vel = np.array([self.forward_speed, self.sideways_speed, self.yaw_rate])
        dx = np.dot(self.rotation(), vel)
        self.d_north = dx[0]
        self.d_east = dx[1]
        self.d_yaw = dx[2]

    def rotation(self):
        ''' Specifies the rotation matrix for rotations about the z-axis, such that
            "body-fixed coordinates" = rotation x "North-east-down-fixed coordinates" .
        '''
        return np.array([[np.cos(self.yaw_angle), -np.sin(self.yaw_angle), 0],
                         [np.sin(self.yaw_angle), np.cos(self.yaw_angle), 0],
                         [0, 0, 1]])

    def mass_matrix(self):
        return np.array([[self.mass + self.x_du, 0, 0],
                         [0, self.mass + self.y_dv, self.mass * self.x_g],
                         [0, self.mass * self.x_g, self.i_z + self.n_dr]])

    def coriolis_matrix(self):
        return np.array([[0, 0, -self.mass * (self.x_g * self.yaw_rate + self.sideways_speed)],
                         [0, 0, self.mass * self.forward_speed],
                         [self.mass * (self.x_g * self.yaw_rate + self.sideways_speed),
                          -self.mass * self.forward_speed, 0]])

    def coriolis_added_mass_matrix(self, u_r, v_r):
        return np.array([[0, 0, self.y_dv * v_r],
                        [0, 0, -self.x_du * u_r],
                        [-self.y_dv * v_r, self.x_du * u_r, 0]])

    def linear_damping_matrix(self):
        return np.array([[self.mass / self.t_surge, 0, 0],
                      [0, self.mass / self.t_sway, 0],
                      [0, 0, self.i_z / self.t_yaw]])

    def non_linear_damping_matrix(self):
        return np.array([[self.ku * self.forward_speed, 0, 0],
                       [0, self.kv * self.sideways_speed, 0],
                       [0, 0, self.kr * self.yaw_rate]])

    def three_dof_kinetics(self, *args, **kwargs):
        ''' Calculates accelerations of the ship, as a funciton
            of thrust-force, rudder angle, wind forces and the
            states in the previous time-step.
        '''
        wind_force = self.get_wind_force()
        wave_force = np.array([0, 0, 0])

        # assembling state vector
        vel = np.array([self.forward_speed, self.sideways_speed, self.yaw_rate])

        # Transforming current velocity to ship frame
        v_c = np.dot(np.linalg.inv(self.rotation()), self.vel_c)
        u_r = self.forward_speed - v_c[0]
        v_r = self.sideways_speed - v_c[1]

        # Kinetic equation
        m_inv = np.linalg.inv(self.mass_matrix())
        dx = np.dot(
            m_inv,
            -np.dot(self.coriolis_matrix(), vel)
            - np.dot(self.coriolis_added_mass_matrix(u_r=u_r, v_r=v_r), vel - v_c)
            - np.dot(self.linear_damping_matrix() + self.non_linear_damping_matrix(), vel - v_c)
            + wind_force + wave_force
        )
        self.d_forward_speed = dx[0]
        self.d_sideways_speed = dx[1]
        self.d_yaw_rate = dx[2]

    def update_differentials(self, *args, **kwargs):
        ''' This method should be called in the simulation loop. It will
            update the full differential equation of the ship.
        '''
        self.three_dof_kinematics()
        self.three_dof_kinetics()

    def integrate_differentials(self):
        ''' Integrates the differential equation one time step ahead using
            the euler intgration method with parameters set in the
            int-instantiation of the "EulerInt"-class.
        '''
        self.north = self.int.integrate(x=self.north, dx=self.d_north)
        self.east = self.int.integrate(x=self.east, dx=self.d_east)
        self.yaw_angle = self.int.integrate(x=self.yaw_angle, dx=self.d_yaw)
        self.forward_speed = self.int.integrate(x=self.forward_speed, dx=self.d_forward_speed)
        self.sideways_speed = self.int.integrate(x=self.sideways_speed, dx=self.d_sideways_speed)
        self.yaw_rate = self.int.integrate(x=self.yaw_rate, dx=self.d_yaw_rate)

    def ship_snap_shot(self):
        ''' This method is used to store a map-view snap shot of
            the ship at the given north-east position and heading.
            It uses the ShipDraw-class. To plot a map view of the
            n-th ship snap-shot, use:

            plot(ship_drawings[1][n], ship_drawings[0][n])
        '''
        x, y = self.drw.local_coords()
        x_ned, y_ned = self.drw.rotate_coords(x, y, self.yaw_angle)
        x_ned_trans, y_ned_trans = self.drw.translate_coords(x_ned, y_ned, self.north, self.east)
        self.ship_drawings[0].append(x_ned_trans)
        self.ship_drawings[1].append(y_ned_trans)


class ShipModel(BaseShipModel):
    ''' Creates a ship model object that can be used to simulate a ship in transit

        The ships model is propelled by a single propeller and steered by a rudder.
        The propeller is powered by either the main engine, an auxiliary motor
        referred to as the hybrid shaft generator, or both. The model contains the
        following states:
        - North position of ship
        - East position of ship
        - Yaw angle (relative to north axis)
        - Surge velocity (forward)
        - Sway velocity (sideways)
        - Yaw rate
        - Propeller shaft speed

        Simulation results are stored in the instance variable simulation_results
    '''
    def __init__(self, ship_config: ShipConfiguration, simulation_config: SimulationConfiguration,
                 environment_config: EnvironmentConfiguration, machinery_config: MachinerySystemConfiguration,
                 initial_propeller_shaft_speed_rad_per_s):
        super().__init__(ship_config, simulation_config, environment_config)
        self.ship_machinery_model = ShipMachineryModel(
            machinery_config=machinery_config,
            initial_propeller_shaft_speed_rad_per_sec=initial_propeller_shaft_speed_rad_per_s,
            time_step=self.int.dt
        )
        self.simulation_results = defaultdict(list)

    def three_dof_kinetics(self, thrust_force=None, rudder_angle=None, load_percentage=None, *args, **kwargs):
        ''' Calculates accelerations of the ship, as a function
            of thrust-force, rudder angle, wind forces and the
            states in the previous time-step.
        '''
        # Forces acting (replace zero vectors with suitable functions)
        f_rudder_v, f_rudder_r = self.rudder(rudder_angle)

        wind_force = self.get_wind_force()
        wave_force = np.array([0, 0, 0])
        ctrl_force = np.array([thrust_force, f_rudder_v, f_rudder_r])

        # assembling state vector
        vel = np.array([self.forward_speed, self.sideways_speed, self.yaw_rate])

        # Transforming current velocity to ship frame
        v_c = np.dot(np.linalg.inv(self.rotation()), self.vel_c)
        u_r = self.forward_speed - v_c[0]
        v_r = self.sideways_speed - v_c[1]

        # Kinetic equation
        m_inv = np.linalg.inv(self.mass_matrix())
        dx = np.dot(
            m_inv,
            -np.dot(self.coriolis_matrix(), vel)
            - np.dot(self.coriolis_added_mass_matrix(u_r=u_r, v_r=v_r), vel - v_c)
            - np.dot(self.linear_damping_matrix() + self.non_linear_damping_matrix(), vel - v_c)
            + wind_force + wave_force + ctrl_force)
        self.d_forward_speed = dx[0]
        self.d_sideways_speed = dx[1]
        self.d_yaw_rate = dx[2]

    def rudder(self, delta):
        ''' This method takes in the rudder angle and returns
            the force i sway and yaw generated by the rudder.

            args:
            delta (float): The rudder angle in radians

            returs:
            v_force (float): The force in sway-direction generated by the rudder
            r_force (float): The yaw-torque generated by the rudder
        '''
        u_c = np.dot(np.linalg.inv(self.rotation()), self.vel_c)[0]
        v_force = -self.ship_machinery_model.c_rudder_v * delta * (self.forward_speed - u_c)
        r_force = -self.ship_machinery_model.c_rudder_r * delta * (self.forward_speed - u_c)
        return v_force, r_force

    def update_differentials(self, engine_throttle=None, rudder_angle=None, *args, **kwargs):
        ''' This method should be called in the simulation loop. It will
            update the full differential equation of the ship.
        '''
        self.three_dof_kinematics()
        self.ship_machinery_model.update_shaft_equation(engine_throttle)
        self.three_dof_kinetics(thrust_force=self.ship_machinery_model.thrust(), rudder_angle=rudder_angle)

    def integrate_differentials(self):
        ''' Integrates the differential equation one time step ahead using
            the euler intgration method with parameters set in the
            int-instantiation of the "EulerInt"-class.
        '''
        self.north = self.int.integrate(x=self.north, dx=self.d_north)
        self.east = self.int.integrate(x=self.east, dx=self.d_east)
        self.yaw_angle = self.int.integrate(x=self.yaw_angle, dx=self.d_yaw)
        self.forward_speed = self.int.integrate(x=self.forward_speed, dx=self.d_forward_speed)
        self.sideways_speed = self.int.integrate(x=self.sideways_speed, dx=self.d_sideways_speed)
        self.yaw_rate = self.int.integrate(x=self.yaw_rate, dx=self.d_yaw_rate)
        self.ship_machinery_model.integrate_differentials()

    def store_simulation_data(self, load_perc):
        load_perc_me, load_perc_hsg = self.ship_machinery_model.load_perc(load_perc)
        self.simulation_results['time [s]'].append(self.int.time)
        self.simulation_results['north position [m]'].append(self.north)
        self.simulation_results['east position [m]'].append(self.east)
        self.simulation_results['yaw angle [deg]'].append(self.yaw_angle * 180 / np.pi)
        self.simulation_results['forward speed[m/s]'].append(self.forward_speed)
        self.simulation_results['sideways speed [m/s]'].append(self.sideways_speed)
        self.simulation_results['yaw rate [deg/sec]'].append(self.yaw_rate * 180 / np.pi)
        self.simulation_results['propeller shaft speed [rpm]'].append(self.ship_machinery_model.omega * 30 / np.pi)
        self.simulation_results['commanded load fraction me [-]'].append(load_perc_me)
        self.simulation_results['commanded load fraction hsg [-]'].append(load_perc_hsg)

        load_data = self.ship_machinery_model.mode.distribute_load(
            load_perc=load_perc, hotel_load=self.ship_machinery_model.hotel_load
        )
        self.simulation_results['power me [kw]'].append(load_data.load_on_main_engine / 1000)
        self.simulation_results['available power me [kw]'].append(
            self.ship_machinery_model.mode.main_engine_capacity / 1000
        )
        self.simulation_results['power electrical [kw]'].append(load_data.load_on_electrical / 1000)
        self.simulation_results['available power electrical [kw]'].append(
            self.ship_machinery_model.mode.electrical_capacity / 1000
        )
        self.simulation_results['power [kw]'].append((load_data.load_on_electrical
                                                      + load_data.load_on_main_engine) / 1000)
        self.simulation_results['propulsion power [kw]'].append(
            (load_perc * self.ship_machinery_model.mode.available_propulsion_power) / 1000)
        rate_me, rate_hsg, cons_me, cons_hsg, cons = self.ship_machinery_model.fuel_consumption(load_perc)
        self.simulation_results['fuel rate me [kg/s]'].append(rate_me)
        self.simulation_results['fuel rate hsg [kg/s]'].append(rate_hsg)
        self.simulation_results['fuel rate [kg/s]'].append(rate_me + rate_hsg)
        self.simulation_results['fuel consumption me [kg]'].append(cons_me)
        self.simulation_results['fuel consumption hsg [kg]'].append(cons_hsg)
        self.simulation_results['fuel consumption [kg]'].append(cons)
        self.simulation_results['motor torque [Nm]'].append(self.ship_machinery_model.main_engine_torque(load_perc))
        self.simulation_results['thrust force [kN]'].append(self.ship_machinery_model.thrust() / 1000)

###################################################################################################################
########################## DESCENDANT CLASS BASED ON PARENT CLASS "BaseShipModel" #################################
###################################################################################################################

class ShipModel(BaseShipModel):
    ''' Creates a ship model object that can be used to simulate a ship in transit

        The ships model is propelled by a single propeller and steered by a rudder.
        The propeller is powered by either the main engine, an auxiliary motor
        referred to as the hybrid shaft generator, or both. The model contains the
        following states:
        - North position of ship
        - East position of ship
        - Yaw angle (relative to north axis)
        - Surge velocity (forward)
        - Sway velocity (sideways)
        - Yaw rate
        - Propeller shaft speed

        Simulation results are stored in the instance variable simulation_results
    '''
    def __init__(self, ship_config: ShipConfiguration, 
                 simulation_config: SimulationConfiguration,
                 environment_config: EnvironmentConfiguration, 
                 machinery_config: MachinerySystemConfiguration,
                 initial_propeller_shaft_speed_rad_per_s):
        super().__init__(ship_config, simulation_config, environment_config)
        self.ship_machinery_model = ShipMachineryModel(
            machinery_config=machinery_config,
            initial_propeller_shaft_speed_rad_per_sec=initial_propeller_shaft_speed_rad_per_s,
            time_step=self.int.dt
        )
        self.simulation_results = defaultdict(list)

    def three_dof_kinetics(self, thrust_force=None, rudder_angle=None, load_percentage=None, *args, **kwargs):
        ''' Calculates accelerations of the ship, as a function
            of thrust-force, rudder angle, wind forces and the
            states in the previous time-step.
        '''
        # Forces acting (replace zero vectors with suitable functions)
        f_rudder_v, f_rudder_r = self.rudder(rudder_angle)

        wind_force = self.get_wind_force()
        wave_force = np.array([0, 0, 0])
        ctrl_force = np.array([thrust_force, f_rudder_v, f_rudder_r])

        # assembling state vector
        vel = np.array([self.forward_speed, self.sideways_speed, self.yaw_rate])

        # Transforming current velocity to ship frame
        v_c = np.dot(np.linalg.inv(self.rotation()), self.vel_c)
        u_r = self.forward_speed - v_c[0]
        v_r = self.sideways_speed - v_c[1]

        # Kinetic equation
        m_inv = np.linalg.inv(self.mass_matrix())
        dx = np.dot(
            m_inv,
            -np.dot(self.coriolis_matrix(), vel)
            - np.dot(self.coriolis_added_mass_matrix(u_r=u_r, v_r=v_r), vel - v_c)
            - np.dot(self.linear_damping_matrix() + self.non_linear_damping_matrix(), vel - v_c)
            + wind_force + wave_force + ctrl_force)
        self.d_forward_speed = dx[0]
        self.d_sideways_speed = dx[1]
        self.d_yaw_rate = dx[2]

    def rudder(self, delta):
        ''' This method takes in the rudder angle and returns
            the force i sway and yaw generated by the rudder.

            args:
            delta (float): The rudder angle in radians

            returs:
            v_force (float): The force in sway-direction generated by the rudder
            r_force (float): The yaw-torque generated by the rudder
        '''
        u_c = np.dot(np.linalg.inv(self.rotation()), self.vel_c)[0]
        v_force = -self.ship_machinery_model.c_rudder_v * delta * (self.forward_speed - u_c)
        r_force = -self.ship_machinery_model.c_rudder_r * delta * (self.forward_speed - u_c)
        return v_force, r_force

    def update_differentials(self, engine_throttle=None, rudder_angle=None, *args, **kwargs):
        ''' This method should be called in the simulation loop. It will
            update the full differential equation of the ship.
        '''
        self.three_dof_kinematics()
        self.ship_machinery_model.update_shaft_equation(engine_throttle)
        self.three_dof_kinetics(thrust_force=self.ship_machinery_model.thrust(), rudder_angle=rudder_angle)

    def integrate_differentials(self):
        ''' Integrates the differential equation one time step ahead using
            the euler intgration method with parameters set in the
            int-instantiation of the "EulerInt"-class.
        '''
        self.north = self.int.integrate(x=self.north, dx=self.d_north)
        self.east = self.int.integrate(x=self.east, dx=self.d_east)
        self.yaw_angle = self.int.integrate(x=self.yaw_angle, dx=self.d_yaw)
        self.forward_speed = self.int.integrate(x=self.forward_speed, dx=self.d_forward_speed)
        self.sideways_speed = self.int.integrate(x=self.sideways_speed, dx=self.d_sideways_speed)
        self.yaw_rate = self.int.integrate(x=self.yaw_rate, dx=self.d_yaw_rate)
        self.ship_machinery_model.integrate_differentials()

    def store_simulation_data(self, load_perc):
        load_perc_me, load_perc_hsg = self.ship_machinery_model.load_perc(load_perc)
        self.simulation_results['time [s]'].append(self.int.time)
        self.simulation_results['north position [m]'].append(self.north)
        self.simulation_results['east position [m]'].append(self.east)
        self.simulation_results['yaw angle [deg]'].append(self.yaw_angle * 180 / np.pi)
        self.simulation_results['forward speed[m/s]'].append(self.forward_speed)
        self.simulation_results['sideways speed [m/s]'].append(self.sideways_speed)
        self.simulation_results['yaw rate [deg/sec]'].append(self.yaw_rate * 180 / np.pi)
        self.simulation_results['propeller shaft speed [rpm]'].append(self.ship_machinery_model.omega * 30 / np.pi)
        self.simulation_results['commanded load fraction me [-]'].append(load_perc_me)
        self.simulation_results['commanded load fraction hsg [-]'].append(load_perc_hsg)

        load_data = self.ship_machinery_model.mode.distribute_load(
            load_perc=load_perc, hotel_load=self.ship_machinery_model.hotel_load
        )
        self.simulation_results['power me [kw]'].append(load_data.load_on_main_engine / 1000)
        self.simulation_results['available power me [kw]'].append(
            self.ship_machinery_model.mode.main_engine_capacity / 1000
        )
        self.simulation_results['power electrical [kw]'].append(load_data.load_on_electrical / 1000)
        self.simulation_results['available power electrical [kw]'].append(
            self.ship_machinery_model.mode.electrical_capacity / 1000
        )
        self.simulation_results['power [kw]'].append((load_data.load_on_electrical
                                                      + load_data.load_on_main_engine) / 1000)
        self.simulation_results['propulsion power [kw]'].append(
            (load_perc * self.ship_machinery_model.mode.available_propulsion_power) / 1000)
        rate_me, rate_hsg, cons_me, cons_hsg, cons = self.ship_machinery_model.fuel_consumption(load_perc)
        self.simulation_results['fuel rate me [kg/s]'].append(rate_me)
        self.simulation_results['fuel rate hsg [kg/s]'].append(rate_hsg)
        self.simulation_results['fuel rate [kg/s]'].append(rate_me + rate_hsg)
        self.simulation_results['fuel consumption me [kg]'].append(cons_me)
        self.simulation_results['fuel consumption hsg [kg]'].append(cons_hsg)
        self.simulation_results['fuel consumption [kg]'].append(cons)
        self.simulation_results['motor torque [Nm]'].append(self.ship_machinery_model.main_engine_torque(load_perc))
        self.simulation_results['thrust force [kN]'].append(self.ship_machinery_model.thrust() / 1000)