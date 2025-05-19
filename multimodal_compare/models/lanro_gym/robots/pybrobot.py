from collections import namedtuple
import os
from gymnasium import spaces
from typing import Callable, Dict, List, Optional
import numpy as np
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.env_utils import RGBCOLORS

DEBUG = int("DEBUG" in os.environ and os.environ["DEBUG"])
JointInfo = namedtuple('JointInfo', [
    'id', 'name', 'type', 'damping', 'friction', 'lowerLimit', 'upperLimit', 'maxForce', 'maxVelocity', 'controllable'
])

GRIPPER_VEL: int = 4


class PyBulletRobot:
    NEUTRAL_JOINT_VALUES: List
    NEUTRAL_FINGER_VALUES: List
    default_arm_orn_RPY: List
    num_DOF: int
    action_space = None
    ee_link: int
    gripper_obs_left_z_offset = 0.0
    gripper_obs_right_z_offset = 0.0
    left_finger_id = -1
    right_finger_id = -1

    def __init__(self, sim: PyBulletSimulation, body_name, file_name, base_position, base_orientation, action_type,
                 full_state, fixed_gripper, finger_friction, camera_mode, **kwargs):
        """
        :param sim: Simulation class
        :param fixed_gripper: The boolean variable to lock the gripper
        :param base_position: The [x, y, z] base coordinates for the end-effector
        :param fingers_friction: The amount of finger friction of the gripper
        :param full state: If the full state should be returned
        :param action_type: How actions are calculated
            One of ['absolute_quat', 'relative_quat', 'relative_joints',
                    'absolute_joints', 'absolute_rpy', 'relative_rpy', 'end_effector']
        """
        self.sim = sim
        self.body_name = body_name
        self.action_type = action_type
        self.full_state = full_state
        self.fixed_gripper = fixed_gripper
        self.max_joint_change = sim.dt
        # gripper change is four times faster than joint changes. This in
        # combination with the force increase was necessary to achieve a
        # good success rate for pick and place.
        self.max_gripper_change = sim.dt * GRIPPER_VEL

        self.camera_mode = camera_mode
        self.action_functions: Dict[str, Callable[[np.ndarray, np.ndarray], Optional[np.ndarray]]] = {
            'absolute_quat': self.absolute_quat_step,
            'relative_quat': self.relative_quat_step,
            'relative_joints': self.relative_joint_step,
            'absolute_joints': self.absolute_joint_step,
            'absolute_rpy': self.absolute_rpy_step,
            'relative_rpy': self.relative_rpy_step,
            'end_effector': self.end_effector_step,
            'end_effector_rot': self.end_effector_step
        }
        with self.sim.no_rendering():
            self._load_robot(file_name, base_position, base_orientation, **kwargs)
            self._parse_joint_info()
            self.setup(finger_friction)

    def _load_robot(self, file_name, base_position, base_orientation, **kwargs):
        #print(file_name)
        import os
        #print(os.path.exists(file_name))
        if 'urdf' in file_name:
            self._uid = self.sim.loadURDF(body_name=self.body_name,
                                          fileName=file_name,
                                          basePosition=base_position,
                                          baseOrientation=base_orientation,
                                          useFixedBase=True,
                                          **kwargs)

        elif 'sdf' in file_name:
            self._uid = self.sim.loadSDF(body_name=self.body_name, sdfFileName=file_name)
            self.sim.set_base_pose(self.body_name, base_position, base_orientation)

    def _parse_joint_info(self):
        num_joints = self.sim.get_num_joints(self.body_name)
        self.joints = []
        self.controllable_joints = []
        for i in range(num_joints):
            info = self.sim.get_joint_info(self.body_name, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2] # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != self.sim.bclient.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
            info = JointInfo(jointID, jointName, jointType, jointDamping, jointFriction, jointLowerLimit,
                             jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            self.joints.append(info)

        self.arm_joints = self.controllable_joints[:self.num_DOF]
        self.ee_joints = self.controllable_joints[self.num_DOF:]

        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.num_DOF]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.num_DOF]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints
                                 if info.controllable][:self.num_DOF]

        self.ee_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][self.num_DOF:]
        self.ee_upper_limits = [info.upperLimit for info in self.joints if info.controllable][self.num_DOF:]
        self.ee_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints
                                if info.controllable][self.num_DOF:]

        self.arm_max_force = [self.joints[arm_id].maxForce for arm_id in self.arm_joints]
        self.ee_max_force = [self.joints[ee_id].maxForce for ee_id in self.ee_joints]

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

    def reset(self) -> None:
        self.sim.set_joint_angles(self.body_name,
                                  joints=self.arm_joints + self.ee_joints,
                                  angles=self.NEUTRAL_JOINT_VALUES + self.NEUTRAL_FINGER_VALUES)

    def setup(self, finger_friction):
        """Setup robot's action space and finger friction"""
        # XYZ relative end-effector change in position
        if self.action_type == 'end_effector':
            action_high = np.array([1] * 3)
            action_low = -action_high.copy()
        elif self.action_type == 'end_effector_rot':
            action_high = np.array([1] * 7)
            action_low = -action_high.copy()
        # relative joint change
        elif self.action_type == 'relative_joints':
            action_high = np.array([1] * self.num_DOF)
            action_low = -action_high.copy()
        # absolute joint values and
        elif self.action_type == 'absolute_joints':
            action_high = np.array(self.arm_upper_limits)
            action_low = np.array(self.arm_lower_limits)
        #  absolute rpy values
        elif self.action_type == 'absolute_rpy':
            action_high = np.array(self.arm_upper_limits[:6])
            action_low = np.array(self.arm_lower_limits[:6])
        # relative joint and rpy change
        elif self.action_type == 'relative_rpy':
            action_high = np.array([1] * 6)
            action_low = -action_high.copy()
        # relative quaternion change
        elif self.action_type == 'relative_quat':
            action_high = np.array([1] * 7)
            action_low = -action_high.copy()
        # absolute quaternion
        elif self.action_type == 'absolute_quat':
            action_high = np.array([1] * 7)
            action_low = -action_high.copy()
        else:
            raise ValueError("Unknown action type")

        # add gripper to action space
        if not self.fixed_gripper:
            action_high = np.concatenate((action_high, [1.]))
            action_low = np.concatenate((action_low, [-1.]))

        self.action_space = spaces.Box(low=action_low, high=action_high, dtype='float32')

        # set lateral and spinning friction for fingers
        self.sim.set_lateral_friction(self.body_name, self.ee_joints[0], lateral_friction=finger_friction)
        self.sim.set_lateral_friction(self.body_name, self.ee_joints[1], lateral_friction=finger_friction)
        self.sim.set_spinning_friction(self.body_name, self.ee_joints[0], spinning_friction=0.05)
        self.sim.set_spinning_friction(self.body_name, self.ee_joints[1], spinning_friction=0.05)

    def set_action(self, action) -> None:
        ''' Takes in the action and uses the appropriate function to determine the joint angles
        for execution in the environment '''
        raw_action = np.copy(action)

        if self.fixed_gripper:
            gripper = None
        else:
            action = raw_action[:-1]
            gripper = raw_action[-1]

        self.action_functions[self.action_type](action, gripper)

    def absolute_quat_step(self, action, gripper) -> None:
        """apply absolute quaternions to the joints"""
        assert len(action) == 7 # (x,y,z) (qx, qy, qz, qw)
        new_pos = action[0:3]
        new_orn = action[3:7] # as quaternions
        self.goto(pos=new_pos, orn=new_orn, gripper=gripper)

    def relative_quat_step(self, action, gripper) -> None:
        """apply relative quaternions to the joints"""
        assert len(action) == 7 # (Δx,Δy,Δz) (Δqx, Δqy, Δqz, Δqw)
        state = self.sim.get_link_state(self.body_name, self.ee_link)
        current_pos, current_orn = state[0], state[1]
        new_pos = action[0:3] * self.max_joint_change + current_pos
        new_orn = action[3:7] * self.max_joint_change + current_orn # as quaternions
        self.goto(new_pos, new_orn, gripper)

    def absolute_rpy_step(self, action, gripper) -> None:
        """apply absolute roll, pitch, and yaw to the joints"""
        assert len(action) == 6
        new_pos = action[0:3]
        new_orn = action[3:6]
        self.goto(new_pos, self.sim.get_quaternion_from_euler(new_orn), gripper)

    def relative_rpy_step(self, action, gripper) -> None:
        """apply relative action to roll, pitch, yaw, and the joints"""
        assert len(action) == 6 # (Δx,Δy,Δz) (Δr, Δp, Δy)
        state = self.sim.get_link_state(self.body_name, self.ee_link)
        current_pos, current_orn = state[0], state[1]
        current_orn = self.sim.get_euler_from_quaternion(current_orn)
        new_pos = action[0:3] * self.max_joint_change + current_pos
        new_orn = action[3:6] * self.max_joint_change + current_orn
        self.goto(new_pos, self.sim.get_quaternion_from_euler(new_orn), gripper)

    def relative_joint_step(self, action, gripper) -> None:
        """apply relative values to the joints"""
        assert len(action) == self.num_DOF # Δx_i
        current_poses = self.get_current_pos()
        jointPoses = action * self.max_joint_change + current_poses
        self.goto_joint_poses(jointPoses, gripper)

    def absolute_joint_step(self, action, gripper) -> None:
        """apply absolute values to the joints"""
        self.goto_joint_poses(action, gripper)

    def update_position_pid(self, current_position, desired_position, max_position_change):
        # Calculate the vector from current position to desired position
        error_vector = [desired_position[i] - current_position[i] for i in range(len(current_position))]

        # Calculate the proportional control term
        proportional_term = [np.clip(error_vector[i], -max_position_change, max_position_change) for i in range(len(error_vector))]
        
        # Update the current position based on the proportional control term
        updated_position = [current_position[i] + proportional_term[i] for i in range(len(current_position))]

        return updated_position

    def end_effector_step(self, action, gripper):
        #assert len(action) == 3
        orn = None
        if len(action) == 7:
            orn = action[3:]
            action = action[:3]
        ee_ctrl = action * self.max_joint_change
        ee_position = self.get_ee_position()
        ee_target_position = ee_position + ee_ctrl
        desired_pose = self.update_position_pid(ee_position, action, self.max_joint_change)
        orn = orn if orn is not None else self.default_arm_orn_RPY
        self.goto(desired_pose, orn, gripper)

    def goto(self, pos=None, orn=None, gripper=None) -> None:
        ''' Uses PyBullet IK to solve for desired joint angles '''
        joint_poses = self.sim.bclient.calculateInverseKinematics(
            bodyUniqueId=self._uid,
            endEffectorLinkIndex=self.ee_link,
            targetPosition=pos,
            targetOrientation=orn,
            #  IK requires all 4 lists (lowerLimits, upperLimits, jointRanges, restPoses).
            #  Otherwise regular IK will be used.
            lowerLimits=self.arm_lower_limits + self.ee_lower_limits,
            upperLimits=self.arm_upper_limits + self.ee_upper_limits,
            jointRanges=self.arm_joint_ranges + self.ee_joint_ranges,
            restPoses=self.NEUTRAL_JOINT_VALUES + self.NEUTRAL_FINGER_VALUES,
            maxNumIterations=100,
            residualThreshold=1e-5)
        joint_poses = list(joint_poses[0:self.num_DOF])
        self.goto_joint_poses(joint_poses, gripper)

    def goto_joint_poses(self, joint_target_angles: List, gripper: float) -> None:
        if gripper is not None:
            # call robot-specific gripper function
            finger_target_angles = self.gripper_control(gripper)
        else:
            finger_target_angles = self.gripper_control(None)
        self.control_joints(np.concatenate([joint_target_angles, finger_target_angles]))

    def gripper_control(self, amount) -> List:
        raise NotImplementedError

    def get_camera_img(self):
        raise NotImplementedError

    def get_link_position(self, link: int) -> np.ndarray:
        """Returns the position of a link as (x, y, z)"""
        return np.array(self.sim.get_link_position(self.body_name, link))

    def get_link_velocity(self, link: int) -> np.ndarray:
        """Returns the velocity of a link as (vx, vy, vz)"""
        return np.array(self.sim.get_link_velocity(self.body_name, link))

    def get_current_pos(self) -> np.ndarray:
        return np.array([self.sim.get_joint_angle(self.body_name, j) for j in self.arm_joints])

    def control_joints(self, target_angles: List) -> None:
        self.sim.bclient.setJointMotorControlArray(
            bodyUniqueId=self._uid,
            jointIndices=self.arm_joints + self.ee_joints,
            controlMode=self.sim.bclient.POSITION_CONTROL,
            targetPositions=target_angles,
            forces=self.arm_max_force + self.ee_max_force,
        )

    def get_fingers_width(self) -> float:
        """Returns the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.left_finger_id)
        finger2 = self.sim.get_joint_angle(self.body_name, self.right_finger_id)
        return finger1 + finger2

    def gripper_ray_obs(self):
        """
        This method performs a single raycast to determine which object is between
        the robot's grippers with a specific z-offset accounting for detection
        between the fingertips.
        """
        leftg = self.get_link_position(self.left_finger_id)
        rightg = self.get_link_position(self.right_finger_id)
        leftg[-1] -= self.gripper_obs_left_z_offset
        rightg[-1] -= self.gripper_obs_right_z_offset
        leftg = tuple(leftg)
        rightg = tuple(rightg)
        hit_obj_id, link_idx, hit_fraction, hit_pos, hit_normal = self.sim.bclient.rayTest(leftg, rightg)[0]
        if DEBUG:
            line_color = RGBCOLORS.PINK.value[0]
            self.sim.bclient.addUserDebugLine(leftg, rightg, line_color, 0.5, 1, replaceItemUniqueId=0)
        return hit_obj_id, link_idx, hit_fraction, hit_pos, hit_normal

    def get_obs(self):
        if self.fixed_gripper:
            gripper_state = np.concatenate((self.get_ee_position(), self.get_ee_velocity()))
        else:
            gripper_state = np.concatenate((self.get_ee_position(), self.get_ee_velocity(), [self.get_fingers_width()]))

        if self.full_state:
            state = self.sim.get_link_state(self.body_name, self.ee_link)
            orn, orn_vel = state[1], state[-1]
            current_poses = self.get_current_pos()
            return np.concatenate(
                (gripper_state, self.sim.get_euler_from_quaternion(orn), orn_vel, current_poses)).copy()
        else:
            return gripper_state.copy()

    def get_default_controls(self):
        if self.action_type == 'absolute_joints':
            default_values = {
                _key: _val
                for _key, _val in zip([str(_idx)
                                       for _idx in range(len(self.NEUTRAL_JOINT_VALUES))], self.NEUTRAL_JOINT_VALUES)
            }
        elif self.action_type == 'relative_joints':
            default_values = {
                _key: _val
                for _key, _val in zip([str(_idx) for _idx in range(len(self.NEUTRAL_JOINT_VALUES))], [0] *
                                      len(self.NEUTRAL_JOINT_VALUES))
            }
        else:
            default_values = {"X": 0.0, "Y": 0.0, "Z": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0}
        return default_values

    def get_xyz_rpy_controls(self):
        default_values = self.get_default_controls()
        controls = []
        as_low = self.action_space.low
        as_high = self.action_space.high
        if self.action_type in ['relative_joints', 'absolute_joints']:
            for _idx, _dv in enumerate(list(default_values.values())):
                controls.append(self.sim.bclient.addUserDebugParameter(str(_idx), as_low[_idx], as_high[_idx], _dv))
        else:
            ## if action_type == 'end_effector'
            controls.append(self.sim.bclient.addUserDebugParameter("X", as_low[0], as_high[0], default_values['X']))
            controls.append(self.sim.bclient.addUserDebugParameter("Y", as_low[1], as_high[1], default_values['Y']))
            controls.append(self.sim.bclient.addUserDebugParameter("Z", as_low[2], as_high[2], default_values['Z']))
            if self.action_type in ['relative_rpy', 'absolute_rpy']:
                # RPY
                controls.append(self.sim.bclient.addUserDebugParameter("Rx", as_low[3], as_high[3],
                                                                       default_values['1']))
                controls.append(self.sim.bclient.addUserDebugParameter("Px", as_low[4], as_high[4],
                                                                       default_values['2']))
                controls.append(self.sim.bclient.addUserDebugParameter("Yx", as_low[5], as_high[5],
                                                                       default_values['3']))
            elif self.action_type in ['relative_quat', 'absolute_quat']:
                # quaternions
                controls.append(self.sim.bclient.addUserDebugParameter("Qx", as_low[3], as_high[3],
                                                                       default_values['1']))
                controls.append(self.sim.bclient.addUserDebugParameter("Qy", as_low[4], as_high[4],
                                                                       default_values['2']))
                controls.append(self.sim.bclient.addUserDebugParameter("Qz", as_low[5], as_high[5],
                                                                       default_values['3']))
                controls.append(self.sim.bclient.addUserDebugParameter("Qw", as_low[6], as_high[6],
                                                                       default_values['4']))

        if not self.fixed_gripper:
            controls.append(self.sim.bclient.addUserDebugParameter("grip", as_low[-1], as_high[-1], 0))
        return controls
