from pettingzoo.mpe import simple_adversary_v3,simple_v3
import numpy as np
import torch
import torch.nn as nn
import os
from maddpg_agent import Agent
import time
import matplotlib.pyplot as plt

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

def multi_obs_to_state(multi_obs): #把各个智能体的观测值合并成状态
    state = np.array([])
    for agent_obs in multi_obs.values():
        state = np.concatenate([state,agent_obs])

    return state

number_episode = 3000
number_step = 100
TARGET_UPDATE_INTERVAL = 200 #目标网络采样多少步后进行更新
LR_ACTOR = 0.0001
LR_CRITIC = 0.001
HIDDEN_DIM = 64
GAMMA = 0.95
TAU = 0.01
memory_size = 500000
batch_size = 512

scenario = "simple_v3"
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/model" + scenario + '/'
timestamp = time.strftime("%Y%m%d%H%M%S")

#初始化智能体
env = simple_adversary_v3.parallel_env(max_cycles = number_step,continuous_actions = True)

multi_obs ,infos = env.reset()

num_agent = env.num_agents
agent_name_list = env.agents



#观测维度
obs_dim = []

print(multi_obs)
for agent_obs in multi_obs.values():
    obs_dim.append(agent_obs.shape[0])

state_dim = sum(obs_dim)

#动作维度
action_dim = []
for agent_name in agent_name_list:
    print(env.action_space(agent_name))
    action_dim.append(env.action_space(agent_name).sample().shape[0])



agents = []

for agent_i in range(num_agent):
    agent = Agent(memo_size=memory_size,obs_dim=obs_dim[agent_i],state_dim=state_dim,n_agent=num_agent,
                  action_dim=action_dim[agent_i],alpha=LR_ACTOR,beta=LR_CRITIC,fc1_dims=HIDDEN_DIM,fc2_dims=HIDDEN_DIM,gamma=GAMMA,tau=TAU,batch_size=batch_size)
    agents.append(agent)


ep_r_list = []
#训练
for episode_i in range(number_episode): #遍历所有ep
    multi_obs , infos = env.reset()
    episode_reward = 0
    multi_done = {agent_name: False for agent_name in agent_name_list}


    while not any(multi_done.values()):
        for step_i in range(number_step): #遍历每个step

            total_step = episode_i * number_step + step_i
            step_reward = 0
            #收集动作
            multi_actions = {}
            for agent_i , agent_name in enumerate(agent_name_list): #遍历所有智能体
                agent = agents[agent_i]
                single_obs = multi_obs[agent_name] #取出该智能体的观测
                single_action = agent.get_action(single_obs) #根据观测得到该智能体的动作
                multi_actions[agent_name] = single_action #加入到共同动作集合中

            multi_next_obs,multi_reward,multi_done,multi_truncation,infos = env.step(multi_actions)

            state = multi_obs_to_state(multi_obs) #把观测整合成当前时刻的状态
            next_state = multi_obs_to_state(multi_next_obs) #把下一个时刻的观测整合成下一时刻的状态

            if step_i >= number_step - 1:
                multi_done = {agent_name: True for agent_name in agent_name_list}

            #填充缓冲池

            for agent_i,agent_name in enumerate(agent_name_list):
                agent = agents[agent_i]
                single_obs = multi_obs[agent_name] #当前智能体的观测
                single_next_obs = multi_next_obs[agent_name] #当前智能体下一时刻的观测
                single_action = multi_actions[agent_name] #当前智能体的动作
                single_reward = multi_reward[agent_name] #当前智能体的奖励
                single_done = multi_done[agent_name] #当前智能体是否完成ep
                agent.replay_buffer.add_memo(single_obs,single_next_obs,state,next_state,single_action,single_reward,single_done) #填充缓冲池


            multi_batch_obses = []
            multi_batch_next_obses = []
            multi_batch_states = []
            multi_batch_next_states = []
            multi_batch_actions = []
            multi_batch_next_actions = []
            multi_batch_online_actions = []
            multi_batch_rewards = []
            multi_batch_dones = []

            #采样


            current_memo_size = min(memory_size,total_step + 1)
            if current_memo_size < batch_size:
                batch_idx = range(0,current_memo_size)
            else :
                batch_idx = np.random.choice(current_memo_size,batch_size)
            for agent_i in range(num_agent):
                agent = agents[agent_i]
                batch_obses,batch_next_obses,batch_states,batch_next_state,batch_actions,batch_rewards,batch_dones = agent.replay_buffer.sample(batch_idx)

                batch_obses_tensor = torch.tensor(batch_obses,dtype=torch.float).to(device)
                batch_next_obses_tensor = torch.tensor(batch_next_obses,dtype=torch.float).to(device)
                batch_states_tensor = torch.tensor(batch_states,dtype=torch.float).to(device)
                batch_next_state_tensor = torch.tensor(batch_next_state,dtype=torch.float).to(device)
                batch_actions_tensor = torch.tensor(batch_actions,dtype=torch.float).to(device)
                batch_rewards_tensor = torch.tensor(batch_rewards,dtype=torch.float).to(device)
                batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float).to(device)


                multi_batch_obses.append(batch_obses_tensor)
                multi_batch_next_obses.append(batch_next_obses_tensor)
                multi_batch_states.append(batch_states_tensor)
                multi_batch_next_states.append(batch_next_state_tensor)
                multi_batch_actions.append(batch_actions_tensor)

                single_batch_next_actions = agent.target_actor.forward(batch_next_obses_tensor) #通过下一时刻的采样观测信息来预测当前智能体下一时刻动作
                multi_batch_next_actions.append(single_batch_next_actions)

                single_batch_online_action = agent.actor.forward(batch_obses_tensor) #actor网络预测
                multi_batch_online_actions.append(single_batch_online_action)

                multi_batch_rewards.append(batch_rewards_tensor)
                multi_batch_dones.append(batch_dones_tensor)

            multi_batch_actions_tensor = torch.cat(multi_batch_actions,dim=1).to(device)
            multi_batch_next_actions_tensor = torch.cat(multi_batch_next_actions,dim=1).to(device)
            multi_batch_online_actions_tensor = torch.cat(multi_batch_online_actions,dim=1).to(device)


            if (total_step + 1) % TARGET_UPDATE_INTERVAL == 0:
                for agent_i in range(num_agent):
                    agent = agents[agent_i]
                    batch_obses_tensor = multi_batch_obses[agent_i]
                    batch_states_tensor = multi_batch_states[agent_i]
                    batch_next_states_tensor = multi_batch_next_states[agent_i]
                    batch_rewards_tensor = multi_batch_rewards[agent_i]
                    batch_dones_tensor = multi_batch_dones[agent_i]
                    batch_actions_tensor = multi_batch_actions[agent_i]

                    # 更新critic网络
                    critic_target_q = agent.target_critic.forward(batch_next_states_tensor,
                                                                  multi_batch_next_actions_tensor.detach())

                    y = batch_rewards_tensor + (1 - batch_dones_tensor) * agent.gamma * critic_target_q.flatten()

                    critic_q = agent.critic.forward(batch_states_tensor, multi_batch_actions_tensor.detach()).flatten()

                    # loss
                    critic_loss = nn.MSELoss()(y, critic_q)

                    agent.critic.optimizer.zero_grad()
                    critic_loss.backward()
                    agent.critic.optimizer.step()

                    # 更新actor网络
                    # actor_loss = agent.critic.forward(batch_states_tensor,multi_batch_online_actions_tensor.detach()).flatten()
                    actor_loss = agent.critic.forward(batch_states_tensor,
                                                      multi_batch_online_actions_tensor.detach()).flatten()
                    actor_loss = -torch.mean(actor_loss)

                    agent.actor.optimizer.zero_grad()
                    actor_loss.backward()
                    agent.actor.optimizer.step()


                    #更新target网络参数

                    for target_param,param in zip(agent.target_critic.parameters(),agent.critic.parameters()):
                        target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param)

                    #更新actor网络
                    for target_param,param in zip(agent.target_actor.parameters(),agent.actor.parameters()):
                        target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param)




            multi_obs = multi_next_obs
            step_reward = sum([single_reward for single_reward in multi_reward.values()])
            episode_reward += step_reward
        #print(f"step_reward: {step_reward}")
    print(f"episode_reward: {episode_reward}")
    ep_r_list.append(episode_reward)
#可视化

    # if (episode_i + 1) % 10 == 0:
    #     env = simple_adversary_v3.parallel_env(N = 2,max_cycles = number_step,continuous_actions=True,render_mode="human")
    #
    #     for test_epi_i in range(10):
    #         multi_obs,infos = env.reset()
    #         for step_i in range(number_step):
    #             multi_actions = {}
    #             for agent_i,agent_name in enumerate(agent_name_list):  # 遍历所有智能体
    #                 agent = agents[agent_i]
    #                 single_obs = multi_obs[agent_name]  # 取出该智能体的观测
    #                 single_action = agent.get_action(single_obs)  # 根据观测得到该智能体的动作
    #                 multi_actions[agent_name] = single_action  # 加入到共同动作集合中
    #
    #             multi_next_obs, multi_reward, multi_done, multi_truncation, infos = env.step(multi_actions)
    #             multi_obs = multi_next_obs
    #
    # if episode_i == 0:
    #     highest_reward = episode_reward
    # if episode_reward > highest_reward:
    #     highest_reward = episode_reward
    #     for agent_i in range(num_agent):
    #         agent = agents[agent_i]
    #         flag = os.path.exists(agent_path)
    #         if not flag:
    #             os.makedirs(agent_path)
    #         torch.save(agent.actor.state_dict(),f"{agent_path}" + f"agent_{agent_i}_actor_{scenario}_{timestamp}.pth")

plt.plot(range(number_episode),ep_r_list)
plt.show()
#保存模型
env.close()

