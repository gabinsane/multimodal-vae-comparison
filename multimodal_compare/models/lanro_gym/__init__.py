from gymnasium.envs.registration import register

for robot in ['Panda']:
    for reward_type in ["sparse", "dense"]:
        _r_type = "Dense" if reward_type == "dense" else ""
        kwargs = {
            "reward_type": reward_type,
        }
        register(
            id=f'{robot}Reach{_r_type}-v0',
            entry_point='lanro_gym.environments:{}ReachEnv'.format(robot),
            max_episode_steps=50,
            kwargs=kwargs,
        )
        register(
            id=f'{robot}Push{_r_type}-v0',
            entry_point='lanro_gym.environments:{}PushEnv'.format(robot),
            max_episode_steps=50,
            kwargs=kwargs,
        )
        register(
            id=f'{robot}Slide{_r_type}-v0',
            entry_point='lanro_gym.environments:{}SlideEnv'.format(robot),
            max_episode_steps=50,
            kwargs=kwargs,
        )
        register(
            id=f'{robot}Empty{_r_type}-v0',
            entry_point='lanro_gym.environments:{}EmptyEnv'.format(robot),
            max_episode_steps=50,
        )
        register(
            id=f'{robot}PickAndPlace{_r_type}-v0',
            entry_point='lanro_gym.environments:{}StackEnv'.format(robot),
            max_episode_steps=50,
            kwargs={
                **kwargs,
                'num_obj': 1,
                'goal_z_range': 0.2,
            },
        )
        for num_obj in [2, 3, 4]:
            register(
                id=f'{robot}Stack{num_obj}{_r_type}-v0',
                entry_point='lanro_gym.environments:{}StackEnv'.format(robot),
                max_episode_steps=50 * num_obj,
                kwargs={
                    **kwargs, 'num_obj': num_obj
                },
            )

    for num_obj in [2, 3]:
        for _mode in [
                'Default', 'Color', 'Shape', 'Weight', 'Size', 'ColorShape', 'WeightShape', 'SizeShape',
                'ColorShapeSize', 'ColorShapeSizeWeight'
        ]:
            for _obstype in ['state', 'pixelego', 'pixelstatic']:
                _current_obstype = ''
                _cam_mode = 'ego'
                if _obstype == 'pixelego':
                    _current_obstype = 'PixelEgo'
                    _obstype = 'pixel'
                elif _obstype == 'pixelstatic':
                    _cam_mode = 'static'
                    _current_obstype = 'PixelStatic'
                    _obstype = 'pixel'
                for _h_instr in [True, False]:
                    for _a_repair in [True, False]:
                        for _negation_repair in [True, False]:
                            for _delay_a_repair in [True, False]:
                                for _use_synonyms in [True, False]:
                                    _current_mode = '' if _mode == 'Default' else _mode
                                    _current_h_instr = 'HI' if _h_instr else ''
                                    _use_syn = 'Synonyms' if _use_synonyms else ''

                                    _current_a_repair = ''
                                    if _a_repair and _negation_repair:
                                        _current_a_repair = 'ARN'
                                    elif _a_repair and not _negation_repair:
                                        _current_a_repair = 'AR'
                                    elif not _a_repair and _negation_repair:
                                        continue

                                    if _a_repair and _delay_a_repair:
                                        _current_a_repair += 'D'
                                    elif not _a_repair and not _delay_a_repair:
                                        continue
                                    # NOTE: Use 100 for action repair, as the
                                    # agent needs to solve the task for possibly 2 goals in one episode
                                    _max_episode_steps = 100 if _a_repair else 50

                                    _kwargs = {
                                        'num_obj': num_obj,
                                        'mode': _mode.lower(),
                                        'obs_type': _obstype,
                                        'use_hindsight_instructions': _h_instr,
                                        'use_action_repair': _a_repair,
                                        'delay_action_repair': _delay_a_repair,
                                        'use_negations_action_repair': _negation_repair,
                                        'camera_mode': _cam_mode,
                                        'use_synonyms': _use_synonyms
                                    }

                                    param_combination = f"{num_obj}{_current_mode}{_current_obstype}{_use_syn}{_current_h_instr}{_current_a_repair}"

                                    register(id=f'{robot}NLReach{param_combination}-v0',
                                             entry_point='lanro_gym.environments:{}NLReachEnv'.format(robot),
                                             max_episode_steps=_max_episode_steps,
                                             kwargs=_kwargs)
                                    register(id=f'{robot}NLPush{param_combination}-v0',
                                             entry_point='lanro_gym.environments:{}NLPushEnv'.format(robot),
                                             max_episode_steps=_max_episode_steps,
                                             kwargs=_kwargs)
                                    register(id=f'{robot}NLGrasp{param_combination}-v0',
                                             entry_point='lanro_gym.environments:{}NLGraspEnv'.format(robot),
                                             max_episode_steps=_max_episode_steps,
                                             kwargs=_kwargs)
                                    register(id=f'{robot}NLLift{param_combination}-v0',
                                             entry_point='lanro_gym.environments:{}NLLiftEnv'.format(robot),
                                             max_episode_steps=_max_episode_steps,
                                             kwargs=_kwargs)
                                    register(id=f'{robot}NLLeft{param_combination}-v0',
                                             entry_point='lanro_gym.environments:{}NLLeftEnv'.format(robot),
                                             max_episode_steps=_max_episode_steps,
                                             kwargs=_kwargs)
                                    register(id=f'{robot}NLRight{param_combination}-v0',
                                             entry_point='lanro_gym.environments:{}NLRightEnv'.format(robot),
                                             max_episode_steps=_max_episode_steps,
                                             kwargs=_kwargs)
