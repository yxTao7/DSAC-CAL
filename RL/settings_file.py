# TRAINING settings
# settings

settings = {}
settings['sample_time'] = 1  # hours
settings['periodic_save'] = 50  # number of save points.

settings['number_of_training_episodes'] = 2000
settings['episodes_number_test'] = 50  # Number of testing.
settings['training_seed'] = 4

# reference for the state of charge
control_settings = {}
control_settings['references'] = {}
control_settings['references']['soc'] = 0.8

# constraints
control_settings['constraints'] = {}
control_settings['constraints']['init_soc'] = 0.3  # 根据Vt计算
control_settings['constraints']['temperature'] = {}
control_settings['constraints']['voltage'] = {}
control_settings['constraints']['temperature']['init'] = 35
control_settings['constraints']['temperature']['TCmax'] = 45  # 45
control_settings['constraints']['temperature']['TSmax'] = 45
control_settings['constraints']['voltage']['max'] = 3.6

# negative score at which the episode ends
control_settings['max_negative_score'] = -1000
