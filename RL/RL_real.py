import matplotlib.pyplot as plt
import torch
from dsac3_cal import Agent
from settings_file import *
from env import *
import time
import numpy as np
import matlab.engine
import pickle


# normalize states of the system
def normalize_outputs(soc, voltage, temp_c):
    norm_soc = (soc - control_settings['constraints']['init_soc']) / (control_settings['references']['soc'] - control_settings['constraints']['init_soc'])
    norm_voltage = (voltage - 3) / (control_settings['constraints']['voltage']['max'] - 3)
    norm_temp_c = (temp_c - control_settings['constraints']['temperature']['init']) / (control_settings['constraints']['temperature']['TCmax'] - control_settings['constraints']['temperature']['init'])
    norm_output = np.array([norm_soc, norm_voltage, norm_temp_c])

    return norm_output


# get state:SOC,Vt,Temp
def get_output_observations(bat):
    return (bat.x[0]).item(), bat.V_batt.item(), (bat.x[3]).item()


# compute actual action from normalized action
def denormalize_input(input_value, min_output_value, max_output_value):
    output_value = (1 + input_value) * (max_output_value - min_output_value) / 2 + min_output_value

    return output_value


# Policy input state: SOC,Vt,Temp
agent = Agent(state_size=3, action_size=1, random_seed=1)

# Load
i_training = settings['training_seed']
agent.actor.load_state_dict(torch.load('results/testing_results/testing' + str(i_training) + '/checkpoint_actor' + '.pth',map_location='cpu'))

# Environment
env = ETM(sett=settings, cont_sett=control_settings)

# set MATLAB env
eng = matlab.engine.start_matlab()
eng.initfile(nargout=0)
controller = eng.eval('ctol')
voltage_init = eng.eval('v_init')
voltage_init = np.array(voltage_init)
print("Connected to Mega2560!")

_ = env.reset()
soc, _, temp_c = get_output_observations(env)
norm_out = normalize_outputs(soc, voltage_init, temp_c)

ACTION_VEC = []
u_VEC = []
SOC_VEC = []
TC_VEC = []
TS_VEC = []
VOLTAGE_VEC = []
RETURN_VALUE = 0
TIME_VEC = []
done = False

tt = 0
done_flag = 0
TIME_VEC.append(tt)
SOC_VEC.append(env.info['SOC'])
TC_VEC.append(env.info['Tc'])
TS_VEC.append(env.info['Ts'])
VOLTAGE_VEC.append(env.info['Vt'])

for t in range(10000):

    if done_flag == 0:
        norm_action = agent.act_d(norm_out)
        applied_action = denormalize_input(norm_action,env.action_space.low[0],env.action_space.high[0])
    else:
        eng.quit()
        break

    # send current value into MATLAB
    eng.InputCurrent(controller, applied_action, nargout=0)

    time.sleep(1)

    # Read real voltage/ts/current
    v_out, ts_out, u_real = eng.ObserveReal(controller, nargout=3)
    vt = np.array(v_out)
    ts_real = np.array(ts_out)
    u_real = np.array(abs(u_real)) * -1

    _, reward, cost, done, _ = env.step(u_real, vt, ts_real)
    next_soc, next_voltage, next_temp_c = get_output_observations(env)
    norm_next_out = normalize_outputs(next_soc, next_voltage, next_temp_c)

    RETURN_VALUE += reward

    # save the simulation vectors
    ACTION_VEC.append(applied_action)
    u_VEC.append(u_real)
    SOC_VEC.append(env.info['SOC'])
    TC_VEC.append(env.info['Tc'])
    TS_VEC.append(ts_real)

    tt += env.dt
    TIME_VEC.append(tt)
    VOLTAGE_VEC.append(vt)
    norm_out = norm_next_out

    if done:
        done_flag += 1
        # break


# store DATA
with open('results/testing_results/comp' + '/current_list.pkl', 'wb') as f:
    pickle.dump(ACTION_VEC, f)
with open('results/testing_results/comp' + '/realU_list.pkl', 'wb') as f:
    pickle.dump(u_VEC, f)
with open('results/testing_results/comp' + '/soc_list.pkl', 'wb') as f:
    pickle.dump(SOC_VEC, f)
with open('results/testing_results/comp' + '/volt_list.pkl', 'wb') as f:
    pickle.dump(VOLTAGE_VEC, f)
with open('results/testing_results/comp' + '/tc_list.pkl', 'wb') as f:
    pickle.dump(TC_VEC, f)


