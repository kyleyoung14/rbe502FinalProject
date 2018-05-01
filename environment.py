import numpy as np
import csv


class Environment(object):
    #lengths of each link
    link0_len = 100
    link1_len = 100
    link2_len = 100
    env_xyz = (400, 400, 400)     
    act_limits = [-1, 1] 
    act_dime = 3 
    state_dime = 13
    dt = .1        
    get_goal = False
    thresh = 15
    goal_reach = 0


    def __init__(self):
        # node1 (l, d_rad, x, y),
        # node2 (l, d_rad, x, y)
        self.robot = np.zeros((3, 5)) 
        self.robot[0, 0] = self.link0_len 
        self.robot[1, 0] = self.link1_len
        self.robot[2, 0] = self.link2_len
        self.goal = np.array([250, 303, 200]) 
        self.goal_init = self.goal.copy()
        self.robot_base = np.array(self.env_xyz)/2
        self.angles = []

    def step(self, action):
        # action = (node1 angular v, node2 angular v, node3 angular v)
        action = np.clip(action, *self.act_limits)
        self.robot[:, 1] += (action * self.dt)
        self.robot[:, 1] %= (2 * np.pi)

        theta0 = self.robot[0, 1] 
        theta1 = self.robot[1, 1]
        theta2 = self.robot[2, 1]
        self.angles.append([theta0, theta1, theta2])

        link0dx_dy_dz = np.array([0,0,self.robot[0, 0]])
        link1dx_dy_dz = np.array([np.cos(theta0)*(self.robot[1, 0]*np.cos(theta1)), \
                                 np.sin(theta0)*(self.robot[1, 0]*np.cos(theta1)), \
                                 self.robot[1, 0]*np.sin(theta1) + self.robot[0, 0]])
        link2dx_dy_dz = np.array([np.cos(theta0)*(self.robot[1, 0]*np.cos(theta1)+self.robot[2, 0]*np.cos(theta1 + theta2)), \
                                 np.sin(theta0)*(self.robot[1, 0]*np.cos(theta1)+self.robot[2, 0]*np.cos(theta1 + theta2)), \
                                 self.robot[1, 0]*np.sin(theta1)+self.robot[2, 0]*np.sin(theta1 + theta2) + self.robot[0, 0]])

        self.robot[0, 2:5] = self.robot_base + link0dx_dy_dz  # (x1, y1)
        self.robot[1, 2:5] = self.robot_base + link1dx_dy_dz  # (x2, y2)
        self.robot[2, 2:5] = self.robot_base + link2dx_dy_dz # (x3, y3)

        s, link3_distance = self.state_generator()
        r = self.reward_function(link3_distance)

        return s, r, self.get_goal

    def reset(self):
        self.get_goal = False
        self.goal_reach = 0
        self.angles = []

        # pxyz = np.clip(np.random.rand(3) * self.env_xyz[0], 100, 300)
        # self.goal[:] = pxyz
        r = np.random.uniform() * (self.link1_len + self.link2_len)
        t1 = np.random.uniform() * 2*np.pi
        t2 = np.random.uniform() * np.pi - np.pi/2

        self.goal[0] = r*np.cos(t1)*np.cos(t2) + self.robot_base[0]
        self.goal[1] = r*np.sin(t1)*np.cos(t2) + self.robot_base[1]
        self.goal[2] = r*np.sin(t2) + self.robot_base[2] + self.link0_len
        return self.state_generator()[0]

    def action_generartor(self):
        return np.random.uniform(*self.act_limits, size=self.act_dime)


    def state_generator(self):
        # return the distance (dx, dy, dz) between link finger point with blue point
        link_end = self.robot[:, 2:5]
        joints_goal_dist = np.ravel(link_end - self.goal)
        base_dist = (self.robot_base - self.goal)/200
        goal_flag = 1 if self.goal_reach > 0 else 0
        return np.hstack([goal_flag, joints_goal_dist/200, base_dist,
                          # link1_distance_p, link1_distance_b,
                          ]), joints_goal_dist[-3:]

    def reward_function(self, distance):
        t = 30
        eucledian_distance = np.sqrt(np.sum(np.square(distance)))
        reward = -eucledian_distance/200
        if eucledian_distance < self.thresh and (not self.get_goal):
            reward = reward + 1.
            self.goal_reach += 1
            if self.goal_reach > t:
                reward = reward +  10.
                self.get_goal = True
        elif eucledian_distance > self.thresh:
            self.goal_reach = 0
            self.get_goal = False
        return reward


    def printToCSV(self, ep):
        fileName = 'traj/trajectoryEp' + str(ep) + '.csv'
        f = open(fileName,'w+')

        f.write(str(self.goal[0])+','+str(self.goal[1])+','+str(self.goal[2])+'\n')
        for i in range(len(self.angles)):
            f.write(str(self.angles[i][0])+','+str(self.angles[i][1])+','+str(self.angles[i][2])+'\n')

        f.close()