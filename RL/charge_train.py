import torch
import numpy as np
import os
import ipdb
from settings_file import *
from dsac3_cal import Agent
from env import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# normalize states of the system
def normalize_outputs(soc, voltage, temp_c):
    norm_soc = (soc - control_settings['constraints']['init_soc']) / (control_settings['references']['soc'] - control_settings['constraints']['init_soc'])
    norm_voltage = (voltage - 3) / (control_settings['constraints']['voltage']['max'] - 3)
    norm_temp_c = (temp_c - 25) / (control_settings['constraints']['temperature']['TCmax'] - 25)
    norm_output = np.array([norm_soc, norm_voltage, norm_temp_c])

    return norm_output


# get state:SOC,Vt,Temp
def get_output_observations(bat):
    return (bat.x[0]).item(), bat.V_batt.item(), (bat.x[3]).item()


# compute actual action from normalized action
def denormalize_input(input_value, min_output_value, max_output_value):
    output_value = (1 + input_value) * (max_output_value - min_output_value) / 2 + min_output_value

    return output_value


# Evaluation
def evaluate(policy, eval_episodes=10):

    eval_env = ETM(sett=settings, cont_sett=control_settings)

    avg_reward = 0.
    violate = 0.
    max_temp_core = 0.
    max_temp_surface = 0.
    max_volt = 0.
    avg_chg_time = 0.

    for _ in range(eval_episodes):

        state, done = eval_env.reset(), False
        soc, voltage, temp_c = get_output_observations(eval_env)
        norm_out = normalize_outputs(soc, voltage, temp_c)

        ACTION_VEC = []
        TC_VEC = []
        TS_VEC = []
        V_VEC = []
        reward = 0
        cost = 0

        while not done:

            norm_action = policy.act_d(norm_out)
            # actual action
            applied_action = denormalize_input(norm_action, eval_env.action_space.low[0], eval_env.action_space.high[0])

            try:
                _, reward, cost, done, _ = eval_env.step(applied_action)
            except:
                ipdb.set_trace()

            next_soc, next_voltage, next_temp_c = get_output_observations(eval_env)
            norm_next_out = normalize_outputs(next_soc, next_voltage, next_temp_c)

            ACTION_VEC.append(applied_action)
            TC_VEC.append(eval_env.x[3])
            TS_VEC.append(eval_env.x[4])
            V_VEC.append(eval_env.V_batt)

            norm_out = norm_next_out

            avg_reward += reward
            violate += cost

        max_temp_core += np.max(np.array(TC_VEC))
        max_temp_surface += np.max(np.array(TS_VEC))
        max_volt += np.max(np.array(V_VEC))
        avg_chg_time += len(ACTION_VEC)

    avg_reward /= eval_episodes
    violate /= eval_episodes
    max_temp_core /= eval_episodes
    max_temp_surface /= eval_episodes
    max_volt /= eval_episodes
    avg_chg_time /= eval_episodes

    print("---------------------------------------")
    print("Evaluation rewards:{} , cost:{}".format(np.array(avg_reward, dtype='object'), np.array(violate, dtype='object')))
    print("---------------------------------------")

    return avg_reward, max_temp_core, max_temp_surface, max_volt, avg_chg_time


def train(n_episodes=600, i_training=1):
    checkpoints_list = []

    # Save the initial parameters of actor-critic networks.
    i_episode = 0
    best_reward = control_settings['max_negative_score']
    checkpoints_list.append(i_episode)
    try:
        os.makedirs('results/training_results/training' + str(i_training) + '/episode' + str(i_episode))
    except:
        pass

    torch.save(agent.actor.state_dict(), 'results/training_results/training' + str(i_training) + '/episode'
               + str(i_episode) + '/checkpoint_actor_' + str(i_episode) + '.pth')
    torch.save(agent.critic_1.state_dict(), 'results/training_results/training' + str(i_training) + '/episode'
               + str(i_episode) + '/checkpoint_critic1_' + str(i_episode) + '.pth')
    torch.save(agent.critic_2.state_dict(), 'results/training_results/training' + str(i_training) + '/episode'
               + str(i_episode) + '/checkpoint_critic2_' + str(i_episode) + '.pth')

    print('Evaluate first')
    evaluations = [evaluate(agent)]

    for i_episode in range(1, n_episodes + 1):

        _ = env.reset()
        soc, voltage, temp_c = get_output_observations(env)
        norm_out = normalize_outputs(soc, voltage, temp_c)

        score = 0
        violate = 0
        done = False
        cont = 0

        V_VEC = []
        TC_VEC = []
        TS_VEC = []
        SOC_VEC = []
        CURRENT_VEC = []

        while not done or score > control_settings['max_negative_score']:

            norm_action = agent.act(norm_out)
            # actual action
            applied_action = denormalize_input(norm_action,env.action_space.low[0],env.action_space.high[0])

            _, reward, cost, done, _ = env.step(applied_action)
            next_soc, next_voltage, next_temp_c = get_output_observations(env)
            norm_next_out = normalize_outputs(next_soc, next_voltage, next_temp_c)

            V_VEC.append(env.info['Vt'])
            TC_VEC.append(env.info['Tc'])
            TS_VEC.append(env.info['Ts'])
            SOC_VEC.append(env.info['SOC'])
            CURRENT_VEC.append(applied_action)

            # update the agent
            agent.step(norm_out, norm_action, reward, cost, norm_next_out, done)

            score += reward
            violate += cost
            cont += 1
            if done:
                break

            norm_out = norm_next_out

        print("\rEpisode number ", i_episode)
        print("Reward: ", score)
        print("Cost: ", violate)



        if (i_episode % settings['periodic_save']) == 0:
            # save the checkpoint for actor, critic and optimizer

            checkpoints_list.append(i_episode)
            try:
                os.makedirs('results/training_results/training' + str(i_training) + '/episode' + str(i_episode))
            except:
                pass

            torch.save(agent.actor.state_dict(), 'results/training_results/training' + str(i_training) + '/episode'
                       + str(i_episode) + '/checkpoint_actor_' + str(i_episode) + '.pth')
            torch.save(agent.critic_1.state_dict(), 'results/training_results/training' + str(i_training) + '/episode'
                       + str(i_episode) + '/checkpoint_critic1_' + str(i_episode) + '.pth')
            torch.save(agent.critic_2.state_dict(), 'results/training_results/training' + str(i_training) + '/episode'
                       + str(i_episode) + '/checkpoint_critic2_' + str(i_episode) + '.pth')

        if (i_episode % settings['episodes_number_test']) == 0:
            evaluations.append(evaluate(agent))

        # compare priority policy
        if score > best_reward and violate < 1 and i_episode > n_episodes * 0.6:
            best_reward = score
            print('Best model updated in {} episode!'.format(i_episode))
            try:
                os.makedirs('results/testing_results/testing' + str(i_training))
            except:
                pass

            torch.save(agent.actor.state_dict(), 'results/testing_results/testing' + str(i_training) + '/checkpoint_actor' + '.pth')

    return checkpoints_list


'''MAIN TRAIN'''

init_conditions = {}

env = ETM(sett=settings, cont_sett=control_settings)

# training order
i_seed = settings['training_seed']
i_training = i_seed
np.random.seed(i_seed)
torch.manual_seed(i_seed)

# TRAINING simulation
agent = Agent(state_size=3, action_size=1, random_seed=i_training)
returns_list, costs_list, checkpoints_list = train(n_episodes=settings['number_of_training_episodes'], i_training=i_training)
