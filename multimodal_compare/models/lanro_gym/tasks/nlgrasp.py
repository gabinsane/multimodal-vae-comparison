from lanro_gym.robots import PyBulletRobot
from lanro_gym.simulation import PyBulletSimulation
from lanro_gym.tasks.nllift import NLLift


class NLGrasp(NLLift):

    def __init__(self,
                 sim: PyBulletSimulation,
                 robot: PyBulletRobot,
                 obj_xy_range: float = 0.3,
                 num_obj: int = 2,
                 min_goal_height: float = 0.0,
                 max_goal_height: float = 0.01,
                 use_hindsight_instructions: bool = False,
                 use_action_repair: bool = False,
                 delay_action_repair: bool = False,
                 use_negations_action_repair: bool = False,
                 use_synonyms: bool = False,
                 mode: str = 'Color'):
        super().__init__(sim,
                         robot,
                         obj_xy_range=obj_xy_range,
                         num_obj=num_obj,
                         min_goal_height=min_goal_height,
                         max_goal_height=max_goal_height,
                         use_hindsight_instructions=use_hindsight_instructions,
                         use_action_repair=use_action_repair,
                         delay_action_repair=delay_action_repair,
                         use_negations_action_repair=use_negations_action_repair,
                         use_synonyms=use_synonyms,
                         mode=mode)

        self.action_verbs = ["grasp", "grip", "grab"]
