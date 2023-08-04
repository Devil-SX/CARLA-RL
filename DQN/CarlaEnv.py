import gymnasium
from gymnasium import spaces, Env
import env
import carla
from carla import Transform, Location, Rotation
import random
import time
import numpy as np
from matplotlib import pyplot as plt


class CarlaEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CarlaEnv, self).__init__()
        # For Simulation
        self.dt = 0.1
        self.v_des = 4
        self.K = 1
        self.counter = 0
        self.speed_list = []
        self.reward_list = []
        self.limit = 0.3
        self.ave_vel = [0]*100
        self.max_step = 1000


        # For Carla
        #define the location and posture for the vehicle
        self.location_x = 19.8101
        self.location_y = 1.79323
        self.location_z = 0.5
        self.rotation_pitch = 0
        self.rotation_yaw = 0
        self.rotation_roll = 0

        # Connect to CARLA server.
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        #get world function
        self.world = self.client.get_world()
        self.world = self.client.load_world('Town01')
        #get world blueprint_library
        self.blueprint_library = self.world.get_blueprint_library()

        #generate actor
        self.bp = self.blueprint_library.find('vehicle.ford.mustang')
        color = random.choice(self.bp.get_attribute('color').recommended_values)
        self.bp.set_attribute('color', color)
        self.transform = Transform(Location(x=self.location_x, y=self.location_y, z=self.location_z), 
                              Rotation(pitch=self.rotation_pitch,yaw=self.rotation_yaw,roll=self.rotation_roll))
        self.vehicle = self.world.spawn_actor(self.bp, self.transform)

        # spectator
        self.spectator = self.world.get_spectator()
        transform_ve = self.vehicle.get_transform()
        self.spectator.set_transform(carla.Transform(transform_ve.location + carla.Location(x=-30, y=0, z=75),
                                                                            carla.Rotation(pitch=-60, yaw=0, roll=0)))
        # Control
        self.control = carla.VehicleControl(throttle=0,steer=0,brake =0)

        # For Gym
        # Define action and observation space
        self.action_space = spaces.Discrete(20)
        self.observation_space = spaces.Box(0,20,dtype=np.float32) # Speed


    def step(self, action):
        # Execute the action and get the new state.
        self.counter += 1
        v = self._get_velocity()
        self.ave_vel.pop(0)
        self.speed_list.append(v)
        self.ave_vel.append(v)

        thr = self._action2thr(action)
        self._control(thr)

        
        new_state = self._get_velocity()
        reward = self._reward_fuction(new_state)
        self.reward_list.append(reward)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = {}

        print('Step: ', self.counter, 'Velocity: ', v, 'Action: ', action, 'Throttle: ', thr)
        print("RMSE ", self._rmse(self.ave_vel, self.v_des), "Reward: ", reward)


        return new_state, reward, terminated, truncated, info

    def reset(self, seed= None, options = None):
        # Reset the state of the environment to an initial state
        print('Ready to restart')
        self.vehicle.destroy()
        time.sleep(1)

        self.vehicle = self.world.spawn_actor(self.bp, self.transform)
        time.sleep(2)

        self.counter = 0
        self.speed_list = []
        self.reward_list = []
        self.ave_vel = [0]*100

        initial_state = 0
        info = {}

        return initial_state, info

    def render(self, mode='human'):
        # Render the environment to the screen
        pass

    def close(self):
        # Cleanup CARLA resources (actors, sensors, etc.)
        self.vehicle.destroy()
        self._plot()

    def _get_velocity(self):
        velocity = self.vehicle.get_velocity()
        return np.sqrt(velocity.x**2 + velocity.y**2)
    

    def _control(self,thr):
        self.control.throttle = thr
        self.control.brake = 0
        self.control.steer = 0
        self.vehicle.apply_control(self.control)
        time.sleep(self.dt)

    def _rmse(self,x,x_0):
        error = np.array(x) - x_0
        squared_error = error ** 2
        rmse = np.sqrt(np.mean(squared_error))
        return rmse
    
    def _is_truncated(self):
        x = round(self.vehicle.get_transform().location.x,2)
        return x > 378 or self.counter > self.max_step 

    def _is_terminated(self):
        return self._rmse(self.ave_vel, self.v_des) < self.limit and  self.counter > 100
        

    def _action2thr(self,action):
        thr_array = np.linspace(0,0.4,20)
        return thr_array[action]
    
    def _reward_fuction(self,v):
        distance = abs(v - self.v_des)
        reward = -self.K * distance**2
        # reward = -np.log(100*distance)
        return reward

    def _plot(self):
        font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 21,
        }

        plt.plot(self.speed_list)
        plt.xlabel("Cumulative time steps",font1)
        plt.ylabel("Velocity (m/s)",font1)
        plt.savefig('velocity.png',dpi=300)
        plt.close()

        plt.plot(self.reward_list)
        plt.xlabel("Cumulative time steps",font1)
        plt.ylabel("Reward",font1)
        plt.savefig('reward.png',dpi=300)
        plt.close()

    


        

