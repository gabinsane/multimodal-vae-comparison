import os
from typing import Iterator, List, Optional, Tuple, Dict, Any
import pybullet as p
import pybullet_data as pd
from pybullet_utils import bullet_client
from contextlib import contextmanager
from lanro_gym.env_utils import RGBCOLORS
import time
from lanro_gym.utils import environment_camera
import numpy as np
import warnings
import pkgutil
import subprocess
import xml.etree.ElementTree as ET

DEBUG = int("DEBUG" in os.environ and os.environ["DEBUG"])
DEBUG_CAM = int("DEBUG_CAM" in os.environ and os.environ["DEBUG_CAM"])
PYB_GPU = int("PYB_GPU" in os.environ and os.environ["PYB_GPU"])
GRAVITY: float = -9.81
HZ: float = 500


class PyBulletSimulation:

    def __init__(self, n_substeps: int = 20, render: bool = False):
        background_color = np.array([109.0, 219.0, 145.0]) / 255
        self.render_on = render
        if render:
            options = "--background_color_red={} \
                       --background_color_green={} \
                       --background_color_blue={}".format(*background_color)
            self.bclient = bullet_client.BulletClient(connection_mode=p.GUI, options=options)
            # Enable GUI with key "g"
            self.bclient.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            # Enable shadows with key "s"
            self.bclient.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            self.bclient.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, DEBUG)
            if DEBUG_CAM:
                self._setup_camera_controls()
                if DEBUG:
                    warnings.warn(
                        "Using DEBUG and DEBUG_CAM at the same time might result in"
                        " undesired behaviors when debug user parameters overlap", UserWarning)
        else:
            self.bclient = bullet_client.BulletClient(connection_mode=p.DIRECT)

        self.time_step: float = 1. / HZ
        self.n_substeps = n_substeps
        self.bclient.setTimeStep(self.time_step)
        self.bclient.resetSimulation()
        self.bclient.setGravity(0, 0, GRAVITY)
        self.bclient.setAdditionalSearchPath(pd.getDataPath())

        # GPU support for faster rendering of camera images
        if PYB_GPU:
            os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
            os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
            # Get EGL device
            assert 'CUDA_VISIBLE_DEVICES' in os.environ
            devices = os.environ.get('CUDA_VISIBLE_DEVICES', ).split(',')
            assert len(devices) == 1, "No devices specified by CUDA_VISIBLE_DEVICES"
            out = subprocess.check_output(['nvidia-smi', '--id=' + str(devices[0]), '-q', '--xml-format'])
            tree = ET.fromstring(out)
            gpu = tree.findall('gpu')[0]
            dev_id = gpu.find('minor_number').text
            os.environ['EGL_VISIBLE_DEVICES'] = str(dev_id)
            egl = pkgutil.get_loader('eglRenderer')
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        self._bodies_idx: Dict[str, Any] = {}

    @property
    def dt(self) -> float:
        """the product of timeStep and n_substeps, dt, reflects how
        much real time it takes to execute a robot action
        dt = 1 / 500 * 20 = 0.04 seconds -> 25 Hz"""
        return self.time_step * self.n_substeps

    def get_object_id(self, body_name: str) -> int:
        """Get the id of the body.
        Args:
            body_name (str): The name of the body.
        """
        return self._bodies_idx[body_name]

    def step(self) -> None:
        """ step the simulation forward for `num_steps` steps. """
        if DEBUG_CAM:
            self.read_camera_parameters()
        for _ in range(self.n_substeps):
            self.bclient.stepSimulation()

    def close(self) -> None:
        """Close the simulation."""
        self.bclient.disconnect()

    def render(self, mode='human') -> Optional[np.ndarray]:
        if mode == 'human':
            self.bclient.configureDebugVisualizer(self.bclient.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
            time.sleep(self.dt)
        if mode == 'rgb_array':
            viewMatrix = self.bclient.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[-0.4, -0.25, 0.1],
                                                                        distance=1.,
                                                                        yaw=-90,
                                                                        pitch=-90,
                                                                        roll=0,
                                                                        upAxisIndex=2)
            projectionMatrix = self.bclient.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.1, farVal=100)
            return environment_camera(self.bclient, projectionMatrix, viewMatrix)

    def _setup_camera_controls(self):
        self._cam_controls = [
            self.bclient.addUserDebugParameter("Distance", -15, 15, 1.2),
            self.bclient.addUserDebugParameter("Yaw", -360, 360, 70),
            self.bclient.addUserDebugParameter("Pitch", -360, 360, -50),
            self.bclient.addUserDebugParameter("X", -10, 10, 0),
            self.bclient.addUserDebugParameter("Y", -10, 10, 0),
            self.bclient.addUserDebugParameter("Z", -10, 10, 0)
        ]

    def read_camera_parameters(self):
        self.bclient.resetDebugVisualizerCamera(
            self.bclient.readUserDebugParameter(self._cam_controls[0]),
            self.bclient.readUserDebugParameter(self._cam_controls[1]),
            self.bclient.readUserDebugParameter(self._cam_controls[2]),
            [self.bclient.readUserDebugParameter(self._cam_controls[i]) for i in range(3, 6)])

    @contextmanager
    def no_rendering(self) -> Iterator[None]:
        """Disable rendering within this context."""
        self.bclient.configureDebugVisualizer(self.bclient.COV_ENABLE_RENDERING, 0)
        yield
        self.bclient.configureDebugVisualizer(self.bclient.COV_ENABLE_RENDERING, 1)

    def loadURDF(self, body_name: str, **kwargs) -> str:
        """Load URDF file.
        Args:
            body_name (str): The name of the body. Must be unique in the sim.
        """
        self._bodies_idx[body_name] = self.bclient.loadURDF(**kwargs)
        return self._bodies_idx[body_name]

    def loadSDF(self, body_name: str, **kwargs) -> str:
        """Load SDF file.
        Args:
            body_name (str): The name of the body. Must be unique in the sim.
        """
        self._bodies_idx[body_name] = self.bclient.loadSDF(**kwargs)[0]
        return self._bodies_idx[body_name]

    def place_visualizer(self,
                         target: List = [-0.1, 0, -0.1],
                         distance: float = 1.1,
                         yaw: float = 45,
                         pitch: float = -30):
        """Orient the camera used for rendering.
        Args:
            target (x, y, z): Target position.
            distance (float): Distance from the target position.
            yaw (float): Yaw.
            pitch (float): Pitch.
        """
        self.bclient.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target,
        )

    def set_lateral_friction(self, body: str, link: int, lateral_friction: float, **kwargs) -> None:
        """Set the lateral friction of a link.
        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            lateral_friction (float): Lateral friction.
        """
        self.bclient.changeDynamics(bodyUniqueId=self._bodies_idx[body],
                                    linkIndex=link,
                                    lateralFriction=lateral_friction,
                                    **kwargs)

    def set_spinning_friction(self, body: str, link: int, spinning_friction: float, **kwargs):
        """Set the spinning friction of a link.
        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            spinning_friction (float): Spinning friction.
        """
        self.bclient.changeDynamics(bodyUniqueId=self._bodies_idx[body],
                                    linkIndex=link,
                                    spinningFriction=spinning_friction,
                                    **kwargs)

    def get_quaternion_from_euler(self, euler_orn: List) -> List[float]:
        """ Convert euler angles to quaternions."""
        return self.bclient.getQuaternionFromEuler(euler_orn)

    def get_euler_from_quaternion(self, quat: List) -> List[float]:
        """ Convert quaternions to euler angles."""
        return self.bclient.getEulerFromQuaternion(quat)

    def get_base_position(self, body: str) -> List[float]:
        """Get the position of the body.
        Args:
            body (str): Body unique name.
        Returns:
            (x, y, z): The cartesian position.
        """
        return self.bclient.getBasePositionAndOrientation(self._bodies_idx[body])[0]

    def get_base_orientation(self, body: str) -> List[float]:
        """Get the orientation of the body.
        Args:
            body (str): Body unique name.
        Returns:
            (x, y, z, w): The orientation as quaternion.
        """
        return self.bclient.getBasePositionAndOrientation(self._bodies_idx[body])[1]

    def get_base_rotation(self, body: str) -> List[float]:
        """Get the rotation of the body.
        Args:
            body (str): Body unique name.
        Returns:
            (rx, ry, rz): The rotation.
        """
        return self.bclient.getEulerFromQuaternion(self.get_base_orientation(body))

    def get_base_velocity(self, body: str) -> List[float]:
        """Get the velocity of the body.
        Args:
            body (str): Body unique name.
        Returns:
            (vx, vy, vz): The cartesian velocity.
        """
        return self.bclient.getBaseVelocity(self._bodies_idx[body])[0]

    def get_base_angular_velocity(self, body: str) -> List[float]:
        """Get the angular velocity of the body.
        Args:
            body (str): Body unique name.
        Returns:
            (wx, wy, wz): The angular velocity.
        """
        return self.bclient.getBaseVelocity(self._bodies_idx[body])[1]

    def get_joint_angle(self, body: str, joint: int) -> float:
        """Get the angle of the joint of the body.
        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body
        Returns:
            float: The angle.
        """
        return self.bclient.getJointState(self._bodies_idx[body], joint)[0]

    def get_link_state(self, body: str, link: int) -> Tuple:
        return self.bclient.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=1)

    def get_link_position(self, body: str, link: int) -> List:
        """Get the position of the link of the body.
        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
        Returns:
            (x, y, z): The cartesian position.
        """
        return self.bclient.getLinkState(self._bodies_idx[body], link)[0]

    def get_link_velocity(self, body: str, link: int) -> List:
        """Get the velocity of the link of the body.
        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
        Returns:
            (vx, vy, vz): The cartesian velocity.
        """
        return self.bclient.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[6]

    def get_num_joints(self, body: str) -> int:
        return self.bclient.getNumJoints(self._bodies_idx[body])

    def get_joint_info(self, body: str, link_idx: int):
        return self.bclient.getJointInfo(self._bodies_idx[body], link_idx)

    def control_joints(
        self,
        body: str,
        joints: List,
        target_angles: List,
        forces: List,
    ) -> None:
        """Control the joints motor.
        Args:
            body (str): Body unique name.
            joints (List[int]): List of joint indices.
            target_angles (List[float]): List of target angles.
            forces (List[float]): Forces to apply.
        """
        self.bclient.setJointMotorControlArray(
            self._bodies_idx[body],
            jointIndices=joints,
            controlMode=self.bclient.POSITION_CONTROL,
            targetPositions=target_angles,
            forces=forces,
        )

    def control_single_joint(self, body: str, joint: int, pos: float, force: float) -> None:
        self.bclient.setJointMotorControl2(self._bodies_idx[body],
                                           jointIndex=joint,
                                           controlMode=self.bclient.POSITION_CONTROL,
                                           targetPosition=pos,
                                           force=force)

    def set_joint_angles(self, body: str, joints: List, angles: List) -> None:
        """Set the angles of the joints of the body.
        Args:
            body (str): Body unique name.
            joints (List[int]): List of joint indices.
            angles (List[float]): List of target angles.
        """
        for joint, angle in zip(joints, angles):
            self.set_joint_angle(body, joint, angle)

    def set_joint_angle(self, body: str, joint: int, angle: float):
        """Set the angle of the joint of the body.
        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.
            angle (float): Target angle.
        """
        self.bclient.resetJointState(bodyUniqueId=self._bodies_idx[body], jointIndex=joint, targetValue=angle)

    def set_base_pose(self, body: str, position: List, orientation: List) -> None:
        """Set the position of the body.
        Args:
            body (str): Body unique name.
            position (x, y, z): The target cartesian position.
            orientation (x, y, z, w): The target orientation as quaternion.
        """
        self.bclient.resetBasePositionAndOrientation(bodyUniqueId=self._bodies_idx[body],
                                                     posObj=position,
                                                     ornObj=orientation)

    def _create_geometry(self,
                         body_name: str,
                         geom_type: Any,
                         mass: float = 0,
                         position: List = [0, 0, 0],
                         ghost: bool = False,
                         lateral_friction: Optional[float] = None,
                         spinning_friction: Optional[float] = None,
                         visual_kwargs: Optional[Dict] = {},
                         collision_kwargs: Optional[Dict] = {},
                         multi_kwargs: Optional[Dict] = {}):
        """Create a geometry.
        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            geom_type (int): The geometry type. See self.bclient.GEOM_<shape>.
            mass (float, optional): The mass in kg. Defaults to 0.
            position (x, y, z): The position of the geom. Defaults to (0, 0, 0)
            ghost (bool, optional): Whether the geometry can collide. Defaults
                to False.
            lateral_friction (float, optional): The friction coef.
            spinning_friction (float, optional): The friction coef.
            visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
            collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
        """
        baseVisualShapeIndex = self.bclient.createVisualShape(geom_type, **visual_kwargs)
        if not ghost:
            baseCollisionShapeIndex = self.bclient.createCollisionShape(geom_type, **collision_kwargs)
        else:
            baseCollisionShapeIndex = -1
        self._bodies_idx[body_name] = self.bclient.createMultiBody(baseVisualShapeIndex=baseVisualShapeIndex,
                                                                   baseCollisionShapeIndex=baseCollisionShapeIndex,
                                                                   baseMass=mass,
                                                                   basePosition=position,
                                                                   **multi_kwargs)

        if lateral_friction is not None:
            self.set_lateral_friction(body=body_name, link=-1, lateral_friction=lateral_friction)
        if spinning_friction is not None:
            self.set_spinning_friction(body=body_name, link=-1, spinning_friction=spinning_friction)

    def create_sphere(
        self,
        body_name: str,
        radius: float,
        mass: float,
        position: List,
        rgba_color: List,
        specular_color: List = [0, 0, 0, 0],
        ghost: Optional[bool] = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ):
        """Create a sphere.
        Args:
            body_name (str): The name of the box. Must be unique in the sim.
            radius (float): The radius in meter.
            mass (float): The mass in kg.
            position (x, y, z): The position of the sphere.
            specular_color (r, g, b): RGB specular color.
            rgba_color (r, g, b, a): RGBA color.
            ghost (bool, optional): Whether the sphere can collide. Defaults to False.
            friction (float, optionnal): The friction. If None, keep the pybullet default
                value. Defaults to None.
        """
        visual_kwargs = {
            "radius": radius,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius}
        self._create_geometry(
            body_name,
            geom_type=self.bclient.GEOM_SPHERE,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def create_box(
        self,
        body_name: str,
        half_extents: List,
        mass: float,
        position: List,
        rgba_color: List,
        specular_color: List = [0, 0, 0, 0],
        ghost: bool = False,
        lateral_friction: float = 1.0,
        spinning_friction: float = 0.005,
        vis_kwargs: Dict = {},
        coll_kwargs: Dict = {},
    ):
        """Create a box.
        Args:
            body_name (str): The name of the box. Must be unique in the sim.
            half_extents (x, y, z): Half size of the box in meters.
            mass (float): The mass in kg.
            position (x, y, z): The position of the box.
            specular_color (r, g, b): RGB specular color.
            rgba_color (r, g, b, a): RGBA color.
            ghost (bool, optional): Whether the box can collide. Defaults to False.
        """
        visual_kwargs = {
            "halfExtents": half_extents,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
            **vis_kwargs
        }
        collision_kwargs = {"halfExtents": half_extents, **coll_kwargs}
        return self._create_geometry(
            body_name,
            geom_type=self.bclient.GEOM_BOX,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def create_cylinder(
        self,
        body_name: str,
        radius: float,
        height: float,
        mass: float,
        position: List,
        rgba_color: List,
        specular_color: List = [0, 0, 0, 0],
        ghost: Optional[bool] = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ):
        """Create a cylinder.
        Args:
            body_name (str): The name of the box. Must be unique in the sim.
            radius (float): The radius in meter.
            height (float): The radius in meter.
            mass (float): The mass in kg.
            position (x, y, z): The position of the sphere.
            specular_color (r, g, b): RGB specular color.
            rgba_color (r, g, b, a): RGBA color.
            ghost (bool, optional): Whether the sphere can collide. Defaults to False.
        """
        visual_kwargs = {
            "radius": radius,
            "length": height,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius, "height": height}
        self._create_geometry(
            body_name,
            geom_type=self.bclient.GEOM_CYLINDER,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def get_contact_points(self, body1: str, body2: str, **kwargs) -> Tuple:
        """ Returns a tuple of contact point lists of body1 and body2 """
        return self.bclient.getContactPoints(self._bodies_idx[body1], self._bodies_idx[body2], **kwargs)

    def remove_body(self, body_name):
        """Removes a body from the simulation dictionary"""
        if body_name in self._bodies_idx:
            self.bclient.removeBody(self._bodies_idx[body_name])
            del self._bodies_idx[body_name]

    def set_orientation_lines(self, robot_uid, parent_link_index, offset=0.065):
        """ Visualize orientation lines for the robots end effector."""
        line_color = RGBCOLORS.BLUE.value[0]
        self.bclient.addUserDebugLine([-1, 0, offset], [1, 0, offset],
                                      line_color,
                                      parentObjectUniqueId=robot_uid,
                                      parentLinkIndex=parent_link_index)
        self.bclient.addUserDebugLine([0, -1, offset], [0, 1, offset],
                                      line_color,
                                      parentObjectUniqueId=robot_uid,
                                      parentLinkIndex=parent_link_index)
        self.bclient.addUserDebugLine([0, 0, -1], [0, 0, 1],
                                      line_color,
                                      parentObjectUniqueId=robot_uid,
                                      parentLinkIndex=parent_link_index)
