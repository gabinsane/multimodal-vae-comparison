from lanro_gym.env import LanguageEnv
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.robots import Panda
from lanro_gym.tasks import NLReach, NLLift, NLGrasp, NLPush, NLLeft, NLRight, NLEmpty


class PandaNLReachEnv(LanguageEnv):

    def __init__(self,
                 render=False,
                 num_obj=2,
                 obs_type="state",
                 mode='Color',
                 action_type='absolute_joints',
                 use_hindsight_instructions=False,
                 use_action_repair=False,
                 delay_action_repair=False,
                 use_negations_action_repair=False,
                 use_synonyms=False,
                 camera_mode='ego'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=False, action_type=action_type, camera_mode=camera_mode)
        task = NLReach(sim,
                       robot,
                       num_obj=num_obj,
                       mode=mode,
                       use_hindsight_instructions=use_hindsight_instructions,
                       use_action_repair=use_action_repair,
                       delay_action_repair=delay_action_repair,
                       use_negations_action_repair=use_negations_action_repair,
                       use_synonyms=use_synonyms)
        LanguageEnv.__init__(self, sim, robot, task, obs_type=obs_type)


class PandaNLGraspEnv(LanguageEnv):

    def __init__(self,
                 render=False,
                 num_obj=2,
                 obs_type="state",
                 mode='Color',
                 action_type='end_effector',
                 use_hindsight_instructions=False,
                 use_action_repair=False,
                 delay_action_repair=False,
                 use_negations_action_repair=False,
                 use_synonyms=False,
                 camera_mode='ego'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=False, action_type=action_type, camera_mode=camera_mode)
        task = NLGrasp(sim,
                       robot,
                       num_obj=num_obj,
                       mode=mode,
                       use_hindsight_instructions=use_hindsight_instructions,
                       use_action_repair=use_action_repair,
                       delay_action_repair=delay_action_repair,
                       use_negations_action_repair=use_negations_action_repair,
                       use_synonyms=use_synonyms)
        LanguageEnv.__init__(self, sim, robot, task, obs_type=obs_type)


class PandaNLLiftEnv(LanguageEnv):

    def __init__(self,
                 render=False,
                 num_obj=2,
                 obs_type="state",
                 mode='Color',
                 action_type='end_effector',
                 use_hindsight_instructions=False,
                 use_action_repair=False,
                 delay_action_repair=False,
                 use_negations_action_repair=False,
                 use_synonyms=False,
                 camera_mode='ego'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=False, action_type=action_type, camera_mode=camera_mode)
        task = NLLift(sim,
                      robot,
                      num_obj=num_obj,
                      mode=mode,
                      use_hindsight_instructions=use_hindsight_instructions,
                      use_action_repair=use_action_repair,
                      delay_action_repair=delay_action_repair,
                      use_negations_action_repair=use_negations_action_repair,
                      use_synonyms=use_synonyms)
        LanguageEnv.__init__(self, sim, robot, task, obs_type=obs_type)

class PandaNLLeftEnv(LanguageEnv):

    def __init__(self,
                 render=False,
                 num_obj=2,
                 obs_type="state",
                 mode='Color',
                 action_type='end_effector',
                 use_hindsight_instructions=False,
                 use_action_repair=False,
                 delay_action_repair=False,
                 use_negations_action_repair=False,
                 use_synonyms=False,
                 camera_mode='ego'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=False, action_type=action_type, camera_mode=camera_mode)
        task = NLLeft(sim,
                      robot,
                      num_obj=num_obj,
                      mode=mode,
                      use_hindsight_instructions=use_hindsight_instructions,
                      use_action_repair=use_action_repair,
                      delay_action_repair=delay_action_repair,
                      use_negations_action_repair=use_negations_action_repair,
                      use_synonyms=use_synonyms)
        LanguageEnv.__init__(self, sim, robot, task, obs_type=obs_type)


class PandaNLRightEnv(LanguageEnv):

    def __init__(self,
                 render=False,
                 num_obj=2,
                 obs_type="state",
                 mode='Color',
                 action_type='end_effector',
                 use_hindsight_instructions=False,
                 use_action_repair=False,
                 delay_action_repair=False,
                 use_negations_action_repair=False,
                 use_synonyms=False,
                 camera_mode='ego'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=False, action_type=action_type, camera_mode=camera_mode)
        task = NLRight(sim,
                      robot,
                      num_obj=num_obj,
                      mode=mode,
                      use_hindsight_instructions=use_hindsight_instructions,
                      use_action_repair=use_action_repair,
                      delay_action_repair=delay_action_repair,
                      use_negations_action_repair=use_negations_action_repair,
                      use_synonyms=use_synonyms)
        LanguageEnv.__init__(self, sim, robot, task, obs_type=obs_type)

class PandaNLEmptyEnv(LanguageEnv):

    def __init__(self,
                 render=False,
                 num_obj=2,
                 obs_type="state",
                 mode='Color',
                 action_type='end_effector',
                 use_hindsight_instructions=False,
                 use_action_repair=False,
                 delay_action_repair=False,
                 use_negations_action_repair=False,
                 use_synonyms=False,
                 camera_mode='ego'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=False, action_type=action_type, camera_mode=camera_mode)
        task = NLEmpty(sim,
                      robot,
                      num_obj=num_obj,
                      mode=mode,
                      use_hindsight_instructions=use_hindsight_instructions,
                      use_action_repair=use_action_repair,
                      delay_action_repair=delay_action_repair,
                      use_negations_action_repair=use_negations_action_repair,
                      use_synonyms=use_synonyms)
        LanguageEnv.__init__(self, sim, robot, task, obs_type=obs_type)


class PandaNLPushEnv(LanguageEnv):

    def __init__(self,
                 render=False,
                 num_obj=2,
                 obs_type="state",
                 mode='Color',
                 action_type='end_effector',
                 use_hindsight_instructions=False,
                 use_action_repair=False,
                 delay_action_repair=False,
                 use_negations_action_repair=False,
                 use_synonyms=False,
                 camera_mode='ego'):
        sim = PyBulletSimulation(render=render)
        robot = Panda(sim, fixed_gripper=True, action_type=action_type, camera_mode=camera_mode)
        task = NLPush(sim,
                      robot,
                      num_obj=num_obj,
                      mode=mode,
                      use_hindsight_instructions=use_hindsight_instructions,
                      use_action_repair=use_action_repair,
                      delay_action_repair=delay_action_repair,
                      use_negations_action_repair=use_negations_action_repair,
                      use_synonyms=use_synonyms)
        LanguageEnv.__init__(self, sim, robot, task, obs_type=obs_type)
