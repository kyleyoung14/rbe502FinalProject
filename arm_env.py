import numpy as np
import csv


class ArmEnv(object):
    action_bound = [-1, 1] 
    action_dim = 3 #CHANGED
    state_dim = 13
    dt = .1  # refresh rate
    arm1l = 100
    arm2l = 100
    arm0l = 100 #CHANGED
    #viewer = None
    viewer_xyz = (400, 400, 400) #CHANGED
    get_point = False
    #mouse_in = np.array([False])
    point_l = 15
    grab_counter = 0


    def __init__(self, mode='easy'):
        # node1 (l, d_rad, x, y),
        # node2 (l, d_rad, x, y)
        self.mode = mode
        self.arm_info = np.zeros((3, 5)) #CHANGED
        self.arm_info[0, 0] = self.arm0l #CHANGED
        self.arm_info[1, 0] = self.arm1l
        self.arm_info[2, 0] = self.arm2l
        self.point_info = np.array([250, 303, 200]) #CHANGED
        self.point_info_init = self.point_info.copy()
        self.center_coord = np.array(self.viewer_xyz)/2
        self.angles = []

    def step(self, action):
        # action = (node1 angular v, node2 angular v, node3 angular v)
        action = np.clip(action, *self.action_bound)
        self.arm_info[:, 1] += action * self.dt
        self.arm_info[:, 1] %= np.pi * 2

        arm0rad = self.arm_info[0, 1] #CHANGED
        arm1rad = self.arm_info[1, 1]
        arm2rad = self.arm_info[2, 1]
        self.angles.append([arm0rad, arm1rad, arm2rad])

        arm0dx_dy_dz = np.array([0,0,self.arm_info[0, 0]])
        arm1dx_dy_dz = np.array([np.cos(arm0rad)*(self.arm_info[1, 0]*np.cos(arm1rad)), \
                                 np.sin(arm0rad)*(self.arm_info[1, 0]*np.cos(arm1rad)), \
                                 self.arm_info[1, 0]*np.sin(arm1rad) + self.arm_info[0, 0]])
        arm2dx_dy_dz = np.array([np.cos(arm0rad)*(self.arm_info[1, 0]*np.cos(arm1rad)+self.arm_info[2, 0]*np.cos(arm1rad + arm2rad)), \
                                 np.sin(arm0rad)*(self.arm_info[1, 0]*np.cos(arm1rad)+self.arm_info[2, 0]*np.cos(arm1rad + arm2rad)), \
                                 self.arm_info[1, 0]*np.sin(arm1rad)+self.arm_info[2, 0]*np.sin(arm1rad + arm2rad) + self.arm_info[0, 0]])

        self.arm_info[0, 2:5] = self.center_coord + arm0dx_dy_dz  # (x1, y1)
        self.arm_info[1, 2:5] = self.center_coord + arm1dx_dy_dz  # (x2, y2)
        self.arm_info[2, 2:5] = self.center_coord + arm2dx_dy_dz # (x3, y3)

        s, arm3_distance = self._get_state()
        r = self._r_func(arm3_distance)

        return s, r, self.get_point

    def reset(self):
        self.get_point = False
        self.grab_counter = 0
        self.angles = []

        if self.mode == 'hard':
            pxyz = np.clip(np.random.rand(3) * self.viewer_xyz[0], 100, 300)
            self.point_info[:] = pxyz
        else:
            arm0rad, arm1rad, arm2rad = np.random.rand(3) * np.pi * 2
            self.arm_info[0, 1] = arm0rad
            self.arm_info[1, 1] = arm1rad
            self.arm_info[2, 1] = arm2rad
            arm0dx_dy_dz = np.array([0,0,self.arm_info[0, 0]])
            arm1dx_dy_dz = np.array([np.cos(arm0rad)*(self.arm_info[1, 0]*np.cos(arm1rad)), \
                                     np.sin(arm0rad)*(self.arm_info[1, 0]*np.cos(arm1rad)), \
                                     self.arm_info[1, 0]*np.sin(arm1rad) + self.arm_info[0, 0]])
            arm2dx_dy_dz = np.array([np.cos(arm0rad)*(self.arm_info[1, 0]*np.cos(arm1rad)+self.arm_info[2, 0]*np.cos(arm1rad + arm2rad)), \
                                     np.sin(arm0rad)*(self.arm_info[1, 0]*np.cos(arm1rad)+self.arm_info[2, 0]*np.cos(arm1rad + arm2rad)), \
                                     self.arm_info[1, 0]*np.sin(arm1rad)+self.arm_info[2, 0]*np.sin(arm1rad + arm2rad) + self.arm_info[0, 0]])

            self.arm_info[0, 2:5] = self.center_coord + arm0dx_dy_dz  # (x1, y1)
            self.arm_info[1, 2:5] = self.center_coord + arm1dx_dy_dz  # (x2, y2)
            self.arm_info[2, 2:5] = self.center_coord + arm2dx_dy_dz # (x3, y3)

            self.point_info[:] = self.point_info_init
        return self._get_state()[0]

    def sample_action(self):
        return np.random.uniform(*self.action_bound, size=self.action_dim)


    def _get_state(self):
        # return the distance (dx, dy, dz) between arm finger point with blue point
        arm_end = self.arm_info[:, 2:5]
        t_arms = np.ravel(arm_end - self.point_info)
        center_dis = (self.center_coord - self.point_info)/200
        in_point = 1 if self.grab_counter > 0 else 0
        return np.hstack([in_point, t_arms/200, center_dis,
                          # arm1_distance_p, arm1_distance_b,
                          ]), t_arms[-3:]

    def _r_func(self, distance):
        t = 20 #TUNING
        abs_distance = np.sqrt(np.sum(np.square(distance)))
        r = -abs_distance/200
        if abs_distance < self.point_l and (not self.get_point):
            r += 1.
            self.grab_counter += 1
            if self.grab_counter > t:
                r += 10.
                self.get_point = True
        elif abs_distance > self.point_l:
            self.grab_counter = 0
            self.get_point = False
        return r


    def printToCSV(self, ep):
        fileName = 'traj/trajectoryEp' + str(ep) + '.csv'
        f = open(fileName,'w+')

        f.write(str(self.point_info[0])+','+str(self.point_info[1])+','+str(self.point_info[2])+'\n')
        for i in range(len(self.angles)):
            f.write(str(self.angles[i][0])+','+str(self.angles[i][1])+','+str(self.angles[i][2])+'\n')

        f.close()








