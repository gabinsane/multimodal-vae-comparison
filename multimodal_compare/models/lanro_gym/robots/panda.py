from typing import List
from lanro_gym.simulation import PyBulletSimulation
import numpy as np
from lanro_gym.robots.pybrobot import PyBulletRobot
from lanro_gym.utils import gripper_camera


class Panda(PyBulletRobot):
    NEUTRAL_JOINT_VALUES: List = [0.00, 0.41, 0.00, -1.85, -0.00, 2.26, 0.79]
    NEUTRAL_FINGER_VALUES: List = [0, 0]
    ee_link: int = 11
    num_DOF: int = 7
    gripper_obs_left_z_offset = 0.026
    gripper_obs_right_z_offset = 0.026
    left_finger_id = 9
    right_finger_id = 10

    def __init__(self,
                 sim: PyBulletSimulation,
                 fixed_gripper: bool = False,
                 base_position: List[float] = [-0.6, 0, 0],
                 full_state: bool = True,
                 action_type: str = 'relative_joints',
                 finger_friction: float = 1.0,
                 camera_mode: str = 'ego'):
        super().__init__(sim,
                         body_name="panda",
                         file_name="franka_panda/panda.urdf",
                         base_position=base_position,
                         base_orientation=sim.get_quaternion_from_euler([0, 0, 0]),
                         action_type=action_type,
                         fixed_gripper=fixed_gripper,
                         full_state=full_state,
                         finger_friction=finger_friction,
                         camera_mode=camera_mode)
        self.default_arm_orn_RPY = sim.get_quaternion_from_euler([2 * np.pi, np.pi, np.pi])
        self.sim.set_orientation_lines(self._uid, 8)

        # create a constraint to keep the fingers aligned
        _c = self.sim.bclient.createConstraint(self._uid,
                                               self.ee_joints[0],
                                               self._uid,
                                               self.ee_joints[1],
                                               jointType=self.sim.bclient.JOINT_GEAR,
                                               jointAxis=[1, 0, 0],
                                               parentFramePosition=[0, 0, 0],
                                               childFramePosition=[0, 0, 0])
        self.sim.bclient.changeConstraint(_c, gearRatio=-1, erp=0.1, maxForce=50)
        # increase forces of some joints and the end effector
        self.arm_max_force[5] *= 5
        self.arm_max_force[6] *= 5
        self.ee_max_force = [85, 85]

    def gripper_control(self, amount: float) -> List:
        if amount == None:
            return self.NEUTRAL_FINGER_VALUES
        fingers_ctrl = amount * self.max_gripper_change
        fingers_width = self.get_fingers_width()
        target_finger_width = fingers_width + fingers_ctrl
        target_angles = [target_finger_width / 2, target_finger_width / 2]
        return target_angles

    def get_camera_img(self):
        ee_position = np.array(self.get_ee_position())
        projectionMatrix = self.sim.bclient.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.1, farVal=100.0)
        return gripper_camera(self.sim.bclient, projectionMatrix, ee_position, [0, 0, 0, 0], mode=self.camera_mode)
