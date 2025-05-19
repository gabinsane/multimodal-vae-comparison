from lanro_gym.env import GoalEnv
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.robots import Panda
from lanro_gym.tasks import Reach, Push, Stack, Slide, Empty


class PandaReachEnv(GoalEnv):

    def __init__(self, render=False, reward_type="sparse", action_type='end_effector'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=True, action_type=action_type)
        task = Reach(
            sim,
            reward_type=reward_type,
            get_ee_position=robot.get_ee_position,
        )
        GoalEnv.__init__(self, sim, robot, task)

class PandaEmptyEnv(GoalEnv):

    def __init__(self, render=False, reward_type="sparse", action_type='end_effector_rot'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=False, action_type=action_type)
        task = Empty(
            sim,
            reward_type=reward_type,
            get_ee_position=robot.get_ee_position,
        )
        GoalEnv.__init__(self, sim, robot, task)


class PandaPushEnv(GoalEnv):

    def __init__(self, render=False, reward_type="sparse", action_type='end_effector'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=True, action_type=action_type)
        task = Push(sim, reward_type=reward_type)
        GoalEnv.__init__(self, sim, robot, task)


class PandaSlideEnv(GoalEnv):

    def __init__(self, render=False, reward_type="sparse", action_type='end_effector'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=True, action_type=action_type)
        task = Slide(sim, reward_type=reward_type)
        GoalEnv.__init__(self, sim, robot, task)


class PandaStackEnv(GoalEnv):

    def __init__(self, render=False, reward_type="sparse", num_obj=2, goal_z_range=0.0, action_type='end_effector'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=False, action_type=action_type)
        task = Stack(sim, reward_type=reward_type, num_obj=num_obj, goal_z_range=goal_z_range)
        GoalEnv.__init__(self, sim, robot, task)
