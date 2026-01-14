#This file contains the basic set up for the co-optimization, 
#including configurable URDF file, reward, and the setting  of virtual environment 
#Author: Hao-Fang Cheng  
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
import os

class CoDesignGraspEnv(gym.Env):
    def __init__(self, render_mode=None, hw_params=None):
        super(CoDesignGraspEnv, self).__init__()
        
        # init hardware parameter
        self.hw = hw_params if hw_params else {"l1": 0.3, "l2": 0.2, "w": 0.1, "beta": 0.0, "kp": 0.03, "kd": 0.5}

        self.render_mode = render_mode
        self.left_id = None
        self.right_id = None
        self.ball_id = None
        if self.render_mode == "human":
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # define observation scale
        self.observation_space = spaces.Box(low=-20, high=20, shape=(15,), dtype=np.float32)
        
        self.current_step = 0
        self.max_steps = 500

    def _create_urdf(self, name, l1, l2, w, tip_angle):
        density = 200
        pid = os.getpid()
        filename = os.path.join(os.getcwd(), f"{name}_{pid}.urdf")
        
        def get_inertial_xml(m, l, w_val):
            ixx = (1/12) * m * (l**2 + w_val**2)
            iyy = (1/12) * m * (w_val**2 + w_val**2)
            izz = (1/12) * m * (l**2 + w_val**2)
            return f'<mass value="{m:.4f}"/><inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/>'
        
        m1 = density * (l1 * w * w)
        m2 = density * (l2 * w * w)
        m_tip = density * (0.05 * w * w)
        
        urdf_content = f"""<robot name="{name}">

        <material name="dark_gray">
        <color rgba="0.2 0.2 0.2 1.0"/>
        </material>
        <material name="orange">
            <color rgba="1.0 0.5 0.0 1.0"/>
        </material>
        <material name="blue">
            <color rgba="0.0 0.0 0.5 1.0"/>
        </material>

        <link name="base_link">
            <visual><geometry><box size="0.05 0.05 0.05"/></geometry>
            <material name="dark_gray"/> 
            </visual>
            <collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision>
            <inertial>
                <mass value="0.01"/>
                <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
            </inertial>
        </link>

        <joint name="joint1" type="revolute">
            <parent link="base_link"/><child link="link1"/>
            <axis xyz="0 0 1"/><origin xyz="0 0 0"/>
            <limit effort="100" velocity="10" lower="-3.14" upper="3.14"/>
        </joint>

        <link name="link1">
            <visual><origin xyz="0 {l1/2} 0"/><geometry><box size="{w} {l1} {w}"/></geometry>
            <material name="orange"/> 
            </visual>
            <collision><origin xyz="0 {l1/2} 0"/><geometry><box size="{w} {l1} {w}"/></geometry></collision>
            <inertial>{get_inertial_xml(m1, l1, w)}</inertial>
        </link>

        <joint name="joint2" type="revolute">
            <parent link="link1"/><child link="link2"/>
            <axis xyz="0 0 1"/><origin xyz="0 {l1} 0"/>
            <limit effort="100" velocity="10" lower="-3.14" upper="3.14"/>
        </joint>

        <link name="link2">
            <visual><origin xyz="0 {l2/2} 0"/><geometry><box size="{w} {l2} {w}"/></geometry>
            <material name="blue"/>
            </visual>
            <collision><origin xyz="0 {l2/2} 0"/><geometry><box size="{w} {l2} {w}"/></geometry></collision>
            <inertial>{get_inertial_xml(m2, l2, w)}</inertial>
        </link>

        <joint name="joint_tip" type="fixed">
            <parent link="link2"/>
            <child link="fingertip"/> <origin xyz="0 {l2} 0" rpy="0 0 {tip_angle}"/>
        </joint>

        <link name="fingertip">
            <visual><origin xyz="0 0.025 0"/><geometry><box size="{w*1.5} 0.05 {w}"/></geometry>
            <color rgba="0.0 0.0 0.4 1.0"/>
            </visual>
            <collision><origin xyz="0 0.025 0"/><geometry><box size="{w*1.5} 0.05 {w}"/></geometry></collision>
            <inertial>{get_inertial_xml(m_tip, 0.05, w)}</inertial>
        </link>
        </robot>"""
        
        with open(filename, "w") as f:
            f.write(urdf_content)
        return filename
    
    def reset(self, seed=None, options=None, new_hw=None):
        super().reset(seed=seed)
        if new_hw:
            self.hw.update(new_hw)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        left_path = self._create_urdf("left", self.hw['l1'], self.hw['l2'], self.hw['w'], self.hw['beta'])
        right_path = self._create_urdf("right", self.hw['l1'], self.hw['l2'], self.hw['w'], self.hw['beta'])

        ball_y = 0.5
        arm_orientation_left = p.getQuaternionFromEuler([-1.5708, 0, 0])
        arm_orientation_right = p.getQuaternionFromEuler([-1.5708, 0, 3.14159]) 

        self.left_id = p.loadURDF(left_path, [-0.4, 0.5, 0.8], arm_orientation_left, useFixedBase=True)

        self.right_id = p.loadURDF(right_path, [0.4, 0.5, 0.8], arm_orientation_right, useFixedBase=True)
        self.ball_id = p.loadURDF("sphere_1cm.urdf", [0, ball_y, 0.05], globalScaling=10)

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=1.2,     
                cameraYaw=0,          
                cameraPitch=-30,       
                cameraTargetPosition=[0, 0.5, 0.4] 
            )
                
        p.changeDynamics(
            self.ball_id, -1, 
            lateralFriction=1.5,    
            spinningFriction=0.1,   
            rollingFriction=0.02,   
            restitution=0.1         
        )

        for robot_id in [self.left_id, self.right_id]:
            p.changeDynamics(
                robot_id, 2,        
                lateralFriction=1.5,
                spinningFriction=0.1,
                contactStiffness=30000, 
                contactDamping=1000     
            )

        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # scaling control law
        delta_scale = 0.1
        left_states = [p.getJointState(self.left_id, i)[0] for i in range(2)]
        right_states = [p.getJointState(self.right_id, i)[0] for i in range(2)]
        current_pos = np.array(left_states + right_states)
        
        target_pos = np.clip(current_pos + action * delta_scale, -3.14, 3.14)

        kp_dynamic = self.hw['kp'] * (self.hw['w'] / 0.01)

        p.setJointMotorControlArray(
            self.left_id, 
            [0, 1], 
            p.POSITION_CONTROL, 
            targetPositions=target_pos[:2], 
            positionGains=[kp_dynamic] * 2, 
            velocityGains=[self.hw['kd']] * 2
        )

        p.setJointMotorControlArray(
            self.right_id, 
            [0, 1], 
            p.POSITION_CONTROL, 
            targetPositions=target_pos[:2], 
            positionGains=[kp_dynamic] * 2, 
            velocityGains=[self.hw['kd']] * 2
        )

        p.stepSimulation()

        # get observed state and compute reward
        obs = self._get_obs()
        reward, is_success = self._compute_reward(obs)
        
        # terminate condition
        terminated = is_success
        truncated = self.current_step >= self.max_steps
        
        if self.render_mode == "human":
            time.sleep(1./240.)

        return obs, reward, terminated, truncated, {}

    def _compute_reward(self, obs):
        # get the ball and links state
        ball_pos = obs[8:11]
        ball_vel, ball_ang_vel = p.getBaseVelocity(self.ball_id)
        ball_speed = np.linalg.norm(ball_vel)
        
        tip_l = np.array(p.getLinkState(self.left_id, 2)[0])
        tip_r = np.array(p.getLinkState(self.right_id, 2)[0])
        
        dist_l = np.linalg.norm(tip_l - ball_pos)
        dist_r = np.linalg.norm(tip_r - ball_pos)

        # approach reward
        reward_reach = np.exp(-3.0 * (dist_l + dist_r)) 

        # contact reward
        contact_l = len(p.getContactPoints(self.left_id, self.ball_id, 2)) > 0
        contact_r = len(p.getContactPoints(self.right_id, self.ball_id, 2)) > 0
        is_grasping = contact_l and contact_r

        reward_grasp = 0
        if is_grasping:
            reward_grasp = 20.0
        elif contact_l or contact_r:
            reward_grasp = 1.0

        # lift reward
        reward_lift = 0
        is_success = False
        
        if ball_pos[2] > 0.1:
            if is_grasping:
                if ball_speed < 1.0:
                    reward_lift = ball_pos[2] * 400.0
                    if ball_pos[2] > 0.4:
                        if ball_speed < 0.2:
                            is_success = True
                            reward_lift += 50000.0 # mission completed!!
                else:
                    reward_lift = -20.0 # penalize impulsive contact
            else:
                # penalize impulsive contact
                reward_lift = -50.0 

        penalty_spinning = -0.1 * np.linalg.norm(ball_ang_vel)

        # penalize arbitrary swing 
        penalty_height = 0
        if (tip_l[2] > 0.8 or tip_r[2] > 0.8) and not is_grasping:
            penalty_height = -5.0

        total_reward = (5.0 * reward_reach) + reward_grasp + reward_lift + penalty_height + penalty_spinning
        return total_reward, is_success

    def _get_obs(self):
        l_states = [p.getJointState(self.left_id, i) for i in range(2)]
        r_states = [p.getJointState(self.right_id, i) for i in range(2)]
        joint_pos = [s[0] for s in l_states + r_states]
        joint_vel = [s[1] for s in l_states + r_states]
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        
        hw_params = [self.hw['l1'], self.hw['l2'], self.hw['w'], self.hw['beta']]
        
        return np.concatenate([joint_pos, joint_vel, ball_pos, hw_params]).astype(np.float32)

    def close(self):
        try:
            if p.isConnected(): 
                p.disconnect()
                print("Successfully Disconnected")
        except Exception as e:
            print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    env = CoDesignGraspEnv(render_mode="human")
    obs, _ = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, trunc, _ = env.step(action)
        if done or trunc:
            obs, _ = env.reset()