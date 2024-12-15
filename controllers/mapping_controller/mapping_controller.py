"""mapping_controller controller."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math
import sys
import copy
from abc import ABC, abstractmethod

from controller import Robot
from controller import Supervisor
from controller import Keyboard

MAX_SPEED = 12.3

        
############################### HELPER FUNCTIONS ##################################


# Normalizes the angle theta in range (-pi, pi)
def normalize_angle(theta):
    if (theta > np.pi):
        return theta - 2*np.pi
    if (theta < -np.pi):
        return theta + 2*np.pi
    return theta


def log_odds(p):
    return np.log(p/(1-p))
    
    
def from_log_odds(l):
    return 1.0 - 1/(1.0 + np.exp(l))
    
    
from_log_odds = np.vectorize(from_log_odds)


norm_angle_arr = np.vectorize(normalize_angle)


def get_curr_pose(trans_field, rot_field):
    values = trans_field.getSFVec3f()
    rob_theta = np.sign(rot_field.getSFRotation()[2])*rot_field.getSFRotation()[3]
    rob_x = values[0]
    rob_y = values[1]
    return [rob_x, rob_y, rob_theta]
    

def get_pose_delta(last_pose, curr_pose):
    trans_delta = np.sqrt((last_pose[0]-curr_pose[0])**2 + (last_pose[1]-curr_pose[1])**2)
    theta_delta = abs(normalize_angle(last_pose[2]-curr_pose[2]))
    return trans_delta, theta_delta
    

def velFromKeyboard(keyboard):
    turn_base = 3.0
    linear_base = 6.0
    vel_left = 0.0
    vel_right = 0.0
    key = keyboard.getKey()
    while (key != -1):
        if (key==Keyboard.UP):
            vel_left += linear_base
            vel_right += linear_base
        if (key==Keyboard.DOWN):
            vel_left += -linear_base
            vel_right += -linear_base
        if (key==Keyboard.LEFT):
            vel_left += -turn_base
            vel_right += turn_base
        if (key==Keyboard.RIGHT):
            vel_left += turn_base
            vel_right += -turn_base
        key = keyboard.getKey()
    return vel_left, vel_right


############################### GridMapBase ##################################    
    

class GridMapBase(ABC):
    
    def __init__(self, fs_x, fs_y, cell_size, p_prior):
        self.cell_size = cell_size
        self.p_prior = p_prior
        n_cells_half_x = int((fs_x/2.0)//cell_size) + 1
        n_cells_half_y = int((fs_y/2.0)//cell_size) + 1
        self.grid_size = [2*n_cells_half_x, 2*n_cells_half_y]
        self.grid_origin = [-n_cells_half_x*cell_size,-n_cells_half_y*cell_size]
        
        self.grid = np.empty((self.grid_size[0], self.grid_size[1]))
        self.grid.fill(p_prior)
        
        self.fig, ax = plt.subplots()
        self.im = ax.imshow(self.grid*255,cmap='gray', vmin = 0, vmax= 255)
        self.fig.show()
        
    @abstractmethod    
    def update_map(self, pose, scan):
        pass
        
    @abstractmethod   
    def vis_map(self):
        pass
        
        
    # adapted from http://playtechs.blogspot.com/2007/03/raytracing-on-grid.html
    # traverses cells in a 2D grid along a ray
    # start is the starting point of the ray as [x,y] in world coordinates
    # direction is the normalized direction vector of the ray
    # returns visited, intersections
    # visited is a 2D array, where each element is the index of a visited cell,
    # in order of traversal
    # intersections is a1D array that contains the distances to the visited cells 
    # along the ray
    def ray_traversal(self, start, direction):
        x0 = (start[0] - self.grid_origin[0])/self.cell_size
        y0 = (start[1] - self.grid_origin[1])/self.cell_size
        
        dx = abs(direction[0])
        dy = abs(direction[1])
        
        x = int(x0)
        y = int(y0)
        
        
        x_inc = 0
        y_inc = 0
        error = 0.0
        t_x = 0.0
        
        
        if (dx == 0):
            x_inc = 0
            error = float('inf')
            t_x = float('inf')
        elif (direction[0] > 0):
            x_inc = 1
            error = (math.floor(x0) + 1 - x0) * dy
            t_x = (math.floor(x0) + 1 - x0) / dx
        else:
            x_inc = -1
            error = (x0 - math.floor(x0)) * dy
            t_x = (x0 - math.floor(x0)) / dx
            
        t = abs(error)
        t_y = 0.0
        
        if (dy == 0):
            y_inc = 0
            error -= float('inf')
            t_y = float('inf')
            
        elif (direction[1] > 0):
            y_inc = 1
            error -= (math.floor(y0) + 1 - y0) * dx
            t_y = (math.floor(y0) + 1 - y0) / dy
        else:
            y_inc = -1
            error -= (y0 - math.floor(y0)) * dx
            t_y = (y0 - math.floor(y0)) / dy
            
        t_x = abs(t_x)
        t_y = abs(t_y)
        
        visited = []
        intersections = []
        t = 0.0
        while (x >= 0 and y >=0 and x < self.grid_size[0] and y < self.grid_size[1]):
            visited.append([y, x]);
            
            if (error > 0):
                y += y_inc
                error -= dx
                t_y += 1/dy
                t = t_y
            else:
                x += x_inc
                error += dy
                t_x += 1/dx
                t = t_x
            intersections.append(t*self.cell_size)
        
        return visited, intersections


############################### OccupancyMap ################################## 

    
class OccupancyMap(GridMapBase):

    def __init__(self, fs_x, fs_y, cell_size, p_prior):
        super().__init__(fs_x, fs_y, cell_size, p_prior)
        
        self.p_occ = 0.7
        self.p_free = 0.3
        self.r = 0.2
        
        self.grid.fill(-log_odds(p_prior))
        
     
    def vis_map(self):
        flipped_grid = np.flipud(self.grid)
        flipped_grid = 1.0 - from_log_odds(flipped_grid)
    
        self.im.set_data(flipped_grid*255)
        self.fig.canvas.draw()
            
        plt.pause(0.01)   
        
    # inverse sensor model for LiDAR measurements
    # z is the range measurement
    # val is the istance to the cell
    # for which the function is evaluated
    # returns the occupancy probability for that cell    
    def inverse_sensor_model(self, val, z):
        # your code here
        if val > self.r:
            return self.log_odds(self.p_prior)  # Beyond sensor range, keep prior belief
        elif val < z - 0.5:  # Before the obstacle
            return self.log_odds(self.p_free)  # Mark as free
        elif z - 0.5 <= val <= z + 0.5:  # Near the obstacle
            return self.log_odds(self.p_occ)  # Mark as occupied
        else:
            return self.log_odds(self.p_prior)  # Otherwise, retain prior belief
            
    # updates the occupancy grid map
    # pose is the current pose of the robot
    # scan is the current range measurement    
    def update_map(self, pose, scan):      
        # your code here
        robot_x, robot_y, robot_theta = robot_pose

        for angle, z in scan:
            # Compute global coordinates of the endpoint
            global_angle = robot_theta + angle
            endpoint_x = robot_x + z * np.cos(global_angle)
            endpoint_y = robot_y + z * np.sin(global_angle)

            # Get traversed cells using ray traversal
            traversed_cells = self.ray_traversal((robot_x, robot_y), (endpoint_x, endpoint_y))

            for cell, distance in traversed_cells:
                cell_x, cell_y = cell
                if 0 <= cell_x < self.grid.shape[0] and 0 <= cell_y < self.grid.shape[1]:
                    # Update the grid using the inverse sensor model
                    self.grid[cell_x, cell_y] += self.inverse_sensor_model(distance, z)

                
                
############################### ReflectanceMap ################################## 
        
        
class ReflectanceMap(GridMapBase):

    def __init__(self, fs_x, fs_y, cell_size, p_prior):
        super().__init__(fs_x, fs_y, cell_size, p_prior)
        
        self.hit_map = np.zeros((self.grid_size[0], self.grid_size[1]))
        self.miss_map = np.zeros((self.grid_size[0], self.grid_size[1]))
        

    def vis_map(self):
        prob_map = np.full_like(self.hit_map, self.p_prior)
        x_dim, y_dim = prob_map.shape
        for x in range(x_dim):
            for y in range(y_dim):
                h = self.hit_map[x][y]
                m = self.miss_map[x][y]
                if h + m > 0:
                    prob_map[x][y] = h / (h + m)
        flipped_grid = 1.0 - np.flipud(prob_map)
    
        self.im.set_data(flipped_grid*255)
        self.fig.canvas.draw()
         
        plt.pause(0.01)    
    
                
    # updates the reflectance grid map
    # pose is the current pose of the robot
    # scan is the current range measurement              
    def update_map(self, pose, scan):        
        # your code here
        robot_x, robot_y, robot_theta = pose

        for angle, distance in scan:
            # Compute the endpoint of the scan in the map frame
            scan_x = robot_x + distance * np.cos(robot_theta + angle)
            scan_y = robot_y + distance * np.sin(robot_theta + angle)

            # Convert the robot's position and scan endpoint to grid coordinates
            start_cell = self.world_to_grid(robot_x, robot_y)
            end_cell = self.world_to_grid(scan_x, scan_y)

            # Get the cells along the ray using Bresenham's line algorithm
            ray_cells = list(bresenham(start_cell[0], start_cell[1], end_cell[0], end_cell[1]))

            # Increment miss counts for all cells along the ray, except the last cell
            for cell in ray_cells[:-1]:
                x, y = cell
                if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                    self.miss_map[x, y] += 1

            # Increment hit count for the last cell (where the obstacle is detected)
            x, y = ray_cells[-1]
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                self.hit_map[x, y] += 1
                    
                    

########################################### main ################################


def main():
    # create the Robot instance.
    robot = Supervisor()
    robot_node = robot.getFromDef("Pioneer3dx")

    # robot pose translation and rotation objects
    trans_field = robot_node.getField("translation")
    rot_field = robot_node.getField("rotation")
    
    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # init keyboard readings
    keyboard = Keyboard()
    keyboard.enable(10)
    
    # get wheel motor controllers
    leftMotor = robot.getDevice('left wheel')
    rightMotor = robot.getDevice('right wheel')
    leftMotor.setPosition(float('inf'))
    rightMotor.setPosition(float('inf'))

    # initialize wheel velocities
    leftMotor.setVelocity(0.0)
    rightMotor.setVelocity(0.0)
    
    # get and enable lidar
    lidar = robot.getDevice('Sick LMS 291')
    lidar.enable(60)
    lidar.enablePointCloud()
    
    # get map limits
    ground_node = robot.getFromDef("RectangleArena")
    floor_size_field = ground_node.getField("floorSize")
    fs_x = floor_size_field.getSFVec2f()[0]
    fs_y = floor_size_field.getSFVec2f()[1]
    

    # last pose used for odometry calculations
    last_pose = get_curr_pose(trans_field, rot_field)
 
    # translation threshold for odometry calculation
    trans_thr = 0.1
    
    # choose which map to use
    
    #map = OccupancyMap(fs_x, fs_y, cell_size=0.05, p_prior=0.5)
    map = ReflectanceMap(fs_x, fs_y, cell_size=0.05, p_prior=0.5)
    
    
    while robot.step(timestep) != -1:
        # key controls
        vel_left, vel_right = velFromKeyboard(keyboard)
        leftMotor.setVelocity(vel_left)
        rightMotor.setVelocity(vel_right)

        # read robot pose and compute difference to last used pose
        curr_pose = get_curr_pose(trans_field, rot_field)
        trans_delta, theta_delta = get_pose_delta(last_pose, curr_pose)

        # skip until translation change is big enough
        if (trans_delta < trans_thr):
            continue
         
        # get current lidar measurements
        scan = lidar.getRangeImage()
        # we use a reversed scan order in the sensor model
        scan.reverse()
        
        # update map 
        map.update_map(curr_pose, scan)
        # visualize map
        map.vis_map()

        last_pose = curr_pose
    
    plt.show('hold')


if __name__ == "__main__":
    main()
