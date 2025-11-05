import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import sys
import argparse
import math
from collections import deque

# --- SUMO Configuration ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci

# --- Constants ---
# --- UPDATED: Point to new network files ---
SUMO_NET_FILE = "Test2.net.xml"
SUMO_ROUTE_FILE = "test2_agents.rou.xml" # Use the new route file you just created

SIM_STEP_LENGTH = 0.2
MAX_STEPS_PER_EPISODE = 1000 

# Our three learning agents
EGO_AGENT_IDS = ["ego_J1", "ego_J2", "ego_J3"]

# --- DDPG Hyperparameters (for J1, J2) ---
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0

# --- DQN Hyperparameters (for J3) ---
DQN_LR = 1e-4
DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.05
DQN_EPSILON_DECAY = 30000 # Slower decay
TARGET_UPDATE_FREQ = 20 # Episodes

# --- State and Action Dimensions ---
# J1 (E-AoI DDPG): [ego_speed, ego_acc, e-aoi, 4 * (rel_pos, rel_speed, acc)]
STATE_DIM_DDPG_EAOI = 15
# J2 (Vanilla DDPG) & J3 (DQN): [ego_speed, ego_acc, 4 * (rel_pos, rel_speed, acc)]
STATE_DIM_VANILLA = 14

ACTION_DIM_DDPG = 1  # Jerk
ACTION_DIM_DQN = 3   # 0: Accelerate, 1: Hold, 2: Decelerate

ACTION_BOUND = 3.0
MAX_ACC = 2.6
MIN_ACC = -4.5
MAX_SPEED = 30  # m/s

# E-AoI Parameters (Only for J1)
E_AOI_ALPHA = 0.4
E_AOI_MAX_AGE = 10
COMM_PROB_LOSS = 0.3
COMM_FREQ = 0.2

# --- Replay Buffer (Used by all agents) ---

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def store_transition(self, state, action, reward, next_state, done):
        # For DQN, action is an int, needs to be wrapped
        if not isinstance(action, (list, np.ndarray)):
            action = [action]
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)

# --- DDPG Networks (for J1, J2) ---

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.action_bound
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc_s1 = nn.Linear(state_dim, 256)
        self.fc_sa1 = nn.Linear(256 + action_dim, 128)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, state, action):
        s = F.relu(self.fc_s1(state))
        sa = torch.cat([s, action], dim=1)
        q_value = F.relu(self.fc_sa1(sa))
        q_value = self.fc_out(q_value)
        return q_value

# --- DDPG Agent Class (for J1, J2) ---

class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim, action_bound).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.target_actor = Actor(state_dim, action_dim, action_bound).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.noise = np.random.normal(0, 0.1, size=action_dim)

    def select_action(self, state, exploration=True):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().data.numpy()
        
        if exploration:
            self.noise = 0.95 * self.noise + 0.05 * np.random.normal(0, 0.2, size=self.action_dim)
            action = action + self.noise

        return np.clip(action, -ACTION_BOUND, ACTION_BOUND)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store_transition(state, action, reward, next_state, done)

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update Critic
        next_actions = self.target_actor(next_states)
        target_q_values = self.target_critic(next_states, next_actions)
        q_targets = rewards + (GAMMA * target_q_values * (1 - dones))
        q_currents = self.critic(states, actions)
        critic_loss = F.mse_loss(q_currents, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.target_actor, self.actor, TAU)
        self._soft_update(self.target_critic, self.critic, TAU)

    def _soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def save_models(self, prefix="ddpg"):
        torch.save(self.actor.state_dict(), f"{prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{prefix}_critic.pth")

    def load_models(self, prefix="ddpg"):
        self.actor.load_state_dict(torch.load(f"{prefix}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{prefix}_critic.pth"))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

# --- DQN Network (for J3) ---

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

# --- DQN Agent Class (for J3) ---

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=DQN_LR)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.steps_done = 0

    def select_action(self, state, exploration=True):
        state = torch.FloatTensor(state).to(self.device)
        
        if exploration:
            eps_threshold = DQN_EPSILON_END + (DQN_EPSILON_START - DQN_EPSILON_END) * \
                            math.exp(-1. * self.steps_done / DQN_EPSILON_DECAY)
            self.steps_done += 1
            if random.random() > eps_threshold:
                with torch.no_grad():
                    return self.q_network(state).max(0)[1].item() # Exploit
            else:
                return random.randrange(self.action_dim) # Explore
        else:
            with torch.no_grad():
                return self.q_network(state).max(0)[1].item() # Exploit only

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store_transition(state, action, reward, next_state, done)

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        states = states.to(self.device)
        actions = actions.long().to(self.device) # Actions are indices (int)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Get Q(s, a) for the actions taken
        q_currents = self.q_network(states).gather(1, actions)
        
        # Get max Q(s', a') from target network
        next_q_values = self.target_q_network(next_states).max(1)[0].unsqueeze(1)
        
        # Compute target Q value: R + gamma * max Q(s', a')
        q_targets = rewards + (GAMMA * next_q_values * (1 - dones))

        # Compute loss
        loss = F.mse_loss(q_currents, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def save_models(self, prefix="dqn"):
        torch.save(self.q_network.state_dict(), f"{prefix}_q_network.pth")

    def load_models(self, prefix="dqn"):
        self.q_network.load_state_dict(torch.load(f"{prefix}_q_network.pth"))
        self.target_q_network.load_state_dict(self.q_network.state_dict())

# --- V2V Communication Channel Simulator ---

class V2VChannel:
    def __init__(self):
        self.aoi_counters = {}
        self.last_received_data = {}
        
    def _get_persistent_loss(self):
        return random.random() < COMM_PROB_LOSS

    def receive_data(self, vehicle_id, data):
        if vehicle_id not in self.aoi_counters:
            self.aoi_counters[vehicle_id] = 0
            self.last_received_data[vehicle_id] = data
            return data, 0

        if self._get_persistent_loss():
            self.aoi_counters[vehicle_id] += 1
        else:
            self.aoi_counters[vehicle_id] = 0
            self.last_received_data[vehicle_id] = data
        
        current_aoi = min(self.aoi_counters[vehicle_id], E_AOI_MAX_AGE)
        return self.last_received_data[vehicle_id], current_aoi

    def calculate_e_aoi(self, vehicle_data_with_aoi):
        if not vehicle_data_with_aoi:
            return 0.0

        # Sort by distance (abs(rel_pos))
        sorted_vehicles = sorted(
            vehicle_data_with_aoi,
            key=lambda x: abs(x['pred_rel_pos'])
        )

        numerator = 0.0
        denominator = 0.0

        for i, vehicle in enumerate(sorted_vehicles):
            l = i + 1
            alpha_l = E_AOI_ALPHA ** l
            delta_l = vehicle['aoi']
            
            numerator += alpha_l * delta_l
            denominator += alpha_l * E_AOI_MAX_AGE
        
        if denominator == 0:
            return 0.0
            
        e_aoi = numerator / denominator
        return e_aoi

# --- SUMO Multi-Agent Environment ---

class SumoEnvironment:
    def __init__(self, use_gui=False):
        self.use_gui = use_gui
        self.sumo_binary = self._get_sumo_binary()
        self.v2v_channel = V2VChannel()
        self.ego_agent_ids = EGO_AGENT_IDS
        
        # Agent-specific state
        self.agents_acc = {}
        self.agents_step_count = {}
        self.agents_done = {}
        
        # --- UPDATED: Agent-specific environment configuration for Test2.net.xml ---
        self.agent_conflict_edges = {
            "ego_J1": ["-E0"], # J1 (on E4.23) merges at J16_J17, conflicts with -E0
            "ego_J2": ["E0"],  # J2 (on E1.29) merges at J5_J6, conflicts with E0
            "ego_J3": ["E4"]   # J3 (on E2) merges at J1_J5, conflicts with E4
        }
        self.agent_ramp_lanes = {
            "ego_J1": "E4.23_0", # J1's ramp lane before merging
            "ego_J2": "E1.29_0", # J2's ramp lane before merging
            "ego_J3": "E2_0"     # J3's ramp lane before merging
        }
        self.agent_main_lanes = {
            "ego_J1": "-E0.37_0", # J1's target main lane
            "ego_J2": "E0.53_0",  # J2's target main lane
            "ego_J3": "E4.23_0"   # J3's target "main" lane (which is J1's ramp)
        }
        self.agent_success_pos_y = {
            "ego_J1": 0.0,   # Southbound, success is y < 0
            "ego_J2": 50.0,  # Northbound, success is y > 50
            "ego_J3": 0.0    # Southbound, success is y < 0 (on final edge -E0.37)
        }
        self.agent_direction = {
            "ego_J1": "south",
            "ego_J2": "north",
            "ego_J3": "south"
        }


    def _get_sumo_binary(self):
        if self.use_gui:
            return os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')
        return os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')

    def _start_simulation(self):
        traci.start([
            self.sumo_binary,
            "-n", SUMO_NET_FILE,
            "-r", SUMO_ROUTE_FILE,
            "--step-length", str(SIM_STEP_LENGTH),
            "--lateral-resolution", "1.0",
            "--no-warnings", "true"
        ])

    def reset(self):
        if traci.isLoaded():
            traci.close()
        
        self._start_simulation()
        
        self.agents_acc = {agent_id: 0.0 for agent_id in self.ego_agent_ids}
        self.agents_step_count = {agent_id: 0 for agent_id in self.ego_agent_ids}
        self.agents_done = {agent_id: False for agent_id in self.ego_agent_ids}
        
        # Wait until all ego vehicles are spawned
        spawn_wait_counter = 0
        while not all(agent_id in traci.vehicle.getIDList() for agent_id in self.ego_agent_ids):
            traci.simulationStep()
            spawn_wait_counter += 1
            if spawn_wait_counter > 500: # Failsafe
                print("Error: Agents did not spawn. Check route file.")
                return None
            
        # Initialize all agents
        for agent_id in self.ego_agent_ids:
            traci.vehicle.setLaneChangeMode(agent_id, 0b000000000000)
            traci.vehicle.setSpeedMode(agent_id, 0)
            traci.vehicle.setSpeed(agent_id, 10.0)
        
        traci.simulationStep()
        
        # Return a dictionary of initial states
        return {agent_id: self._get_state(agent_id) for agent_id in self.ego_agent_ids}

    def _get_state(self, agent_id):
        """Gathers state for a *specific* agent."""
        try:
            ego_speed = traci.vehicle.getSpeed(agent_id)
            self.agents_acc[agent_id] = traci.vehicle.getAcceleration(agent_id)
            ego_pos_y = traci.vehicle.getPosition(agent_id)[1]
            
            all_veh_ids = traci.vehicle.getIDList()
            conflict_edges = self.agent_conflict_edges[agent_id]
            mainline_vehicles = []
            
            for veh_id in all_veh_ids:
                if veh_id in self.ego_agent_ids or veh_id == agent_id:
                    continue
                
                veh_edge = ""
                try:
                    veh_edge = traci.vehicle.getRoadID(veh_id)
                except traci.TraCIException:
                    continue # Vehicle despawned
                
                if any(conf_edge in veh_edge for conf_edge in conflict_edges):
                    veh_pos_y = traci.vehicle.getPosition(veh_id)[1]
                    
                    data_packet = {
                        'id': veh_id,
                        'pos_y': veh_pos_y,
                        'speed': traci.vehicle.getSpeed(veh_id),
                        'acc': traci.vehicle.getAcceleration(veh_id),
                        'rel_pos': veh_pos_y - ego_pos_y, # Relative position on Y-axis
                    }
                    
                    # Simulate V2V (only J1 uses the AoI, but all get it)
                    received_data, aoi = self.v2v_channel.receive_data(veh_id, data_packet)
                    
                    t_aoi = aoi * COMM_FREQ
                    pred_pos_y = received_data['pos_y'] + (received_data['speed'] * t_aoi) + (0.5 * received_data['acc'] * (t_aoi ** 2))
                    
                    mainline_vehicles.append({
                        'id': veh_id,
                        'pred_rel_pos': pred_pos_y - ego_pos_y,
                        'pred_speed': received_data['speed'] + (received_data['acc'] * t_aoi),
                        'acc': received_data['acc'],
                        'aoi': aoi
                    })

            # Find 4 Relevant Vehicles (f1, b1, f2, b2)
            # This logic must be direction-aware
            if self.agent_direction[agent_id] == "north": # Northbound (ahead = positive rel_pos)
                vehicles_ahead = sorted([v for v in mainline_vehicles if v['pred_rel_pos'] > 0], key=lambda x: x['pred_rel_pos'])
                vehicles_behind = sorted([v for v in mainline_vehicles if v['pred_rel_pos'] <= 0], key=lambda x: -x['pred_rel_pos'])
            else: # Southbound (J1, J3) (ahead = negative rel_pos)
                vehicles_ahead = sorted([v for v in mainline_vehicles if v['pred_rel_pos'] < 0], key=lambda x: -x['pred_rel_pos'])
                vehicles_behind = sorted([v for v in mainline_vehicles if v['pred_rel_pos'] >= 0], key=lambda x: x['pred_rel_pos'])

            f1 = vehicles_ahead[0] if len(vehicles_ahead) > 0 else self._get_dummy_veh(1000)
            f2 = vehicles_ahead[1] if len(vehicles_ahead) > 1 else self._get_dummy_veh(1000)
            b1 = vehicles_behind[0] if len(vehicles_behind) > 0 else self._get_dummy_veh(-1000)
            b2 = vehicles_behind[1] if len(vehicles_behind) > 1 else self._get_dummy_veh(-1000)
            
            relevant_vehicles = [f1, b1, f2, b2]

            # Calculate E-AoI (only used by J1)
            e_aoi = 0.0
            if agent_id == "ego_J1":
                e_aoi = self.v2v_channel.calculate_e_aoi(mainline_vehicles)
            
            # Assemble State Vector
            if agent_id == "ego_J1":
                state = [ego_speed / MAX_SPEED, self.agents_acc[agent_id] / MAX_ACC, e_aoi]
            else:
                state = [ego_speed / MAX_SPEED, self.agents_acc[agent_id] / MAX_ACC]
                
            for veh in relevant_vehicles:
                state.extend([
                    veh['pred_rel_pos'] / 100.0,
                    (veh['pred_speed'] - ego_speed) / MAX_SPEED,
                    veh['acc'] / MAX_ACC
                ])

            return np.array(state)

        except traci.TraCIException:
            # Handle crash
            return np.zeros(STATE_DIM_DDPG_EAOI if agent_id == "ego_J1" else STATE_DIM_VANILLA)

    def _get_dummy_veh(self, rel_pos):
        return {'pred_rel_pos': rel_pos, 'pred_speed': 0, 'acc': 0, 'aoi': 0}

    def step(self, agent_actions):
        """Takes a dictionary of actions and returns dicts of states, rewards, dones."""
        
        # 1. Apply actions to all non-done agents
        for agent_id, action in agent_actions.items():
            if self.agents_done[agent_id]:
                continue
            
            self.agents_step_count[agent_id] += 1
            
            try:
                current_speed = traci.vehicle.getSpeed(agent_id)
                current_acc = self.agents_acc[agent_id]

                if agent_id in ["ego_J1", "ego_J2"]: # DDPG (Jerk)
                    jerk = action[0]
                    new_acc = current_acc + jerk * SIM_STEP_LENGTH
                else: # DQN (Action Index)
                    if action == 0: new_acc = 2.0  # Accelerate
                    elif action == 1: new_acc = 0.0 # Hold
                    else: new_acc = -3.0 # Decelerate
                
                new_acc = np.clip(new_acc, MIN_ACC, MAX_ACC)
                new_speed = current_speed + new_acc * SIM_STEP_LENGTH
                new_speed = np.clip(new_speed, 0, MAX_SPEED)
                
                traci.vehicle.setSpeed(agent_id, new_speed)
            
            except traci.TraCIException:
                self.agents_done[agent_id] = True # Agent crashed

        # 2. Apply lane change logic for all non-done agents
        for agent_id in self.ego_agent_ids:
            if self.agents_done[agent_id]:
                continue
            
            try:
                current_lane = traci.vehicle.getLaneID(agent_id)
                if current_lane == self.agent_ramp_lanes[agent_id]:
                    # Try to merge to target lane 0
                    traci.vehicle.changeLane(agent_id, 0, SIM_STEP_LENGTH)

            except traci.TraCIException:
                 self.agents_done[agent_id] = True # Agent crashed

        # 3. Step simulation
        traci.simulationStep()

        # 4. Get new states, rewards, and dones
        new_states = {}
        rewards = {}
        dones = {}
        
        colliding_ids = traci.simulation.getCollidingVehiclesIDList()

        for agent_id in self.ego_agent_ids:
            # If agent was already done, just fill in dummy data
            if self.agents_done[agent_id]:
                new_states[agent_id] = np.zeros(STATE_DIM_DDPG_EAOI if agent_id == "ego_J1" else STATE_DIM_VANILLA)
                rewards[agent_id] = 0.0
                dones[agent_id] = True
                continue

            # Check for new done conditions
            done = False
            reward = 0.0
            
            try:
                new_state = self._get_state(agent_id)
                new_states[agent_id] = new_state
                ego_pos_y = traci.vehicle.getPosition(agent_id)[1]
                ego_speed = traci.vehicle.getSpeed(agent_id)
                current_lane = traci.vehicle.getLaneID(agent_id)
                
                # a) Collision
                if agent_id in colliding_ids:
                    done = True
                    reward = -500.0
                
                # b) Successful Merge
                elif self._check_success(agent_id, ego_pos_y, current_lane):
                    done = True
                    reward = 500.0
                
                # c) Timeout
                elif self.agents_step_count[agent_id] >= MAX_STEPS_PER_EPISODE:
                    done = True
                    reward = -100.0

                # 5. Calculate Reward if not done
                if not done:
                    r_comm_penalty = 0.0
                    if agent_id == "ego_J1": # E-AoI DDPG
                        jerk = agent_actions[agent_id][0]
                        r_comfort = - (jerk ** 2) * 0.01
                        e_aoi = new_state[2]
                        r_comm_penalty = - (e_aoi * (ego_speed / MAX_SPEED)) * 1.0
                    
                    elif agent_id == "ego_J2": # Vanilla DDPG
                        jerk = agent_actions[agent_id][0]
                        r_comfort = - (jerk ** 2) * 0.01
                    
                    else: # DQN
                        r_comfort = 0.0
                        if agent_actions[agent_id] != 1: # Penalize not holding
                            r_comfort = -0.1 
                    
                    r_efficiency = (ego_speed / MAX_SPEED) * 0.2
                    r_merge_progress = 0.5 if self.agent_main_lanes[agent_id] in current_lane else 0.0
                    
                    reward = r_comfort + r_efficiency + r_comm_penalty + r_merge_progress

            except traci.TraCIException:
                # Agent crashed mid-check
                new_states[agent_id] = np.zeros(STATE_DIM_DDPG_EAOI if agent_id == "ego_J1" else STATE_DIM_VANILLA)
                reward = -500.0
                done = True

            rewards[agent_id] = reward
            dones[agent_id] = done
            if done:
                self.agents_done[agent_id] = True

        all_done = all(self.agents_done.values())
        return new_states, rewards, dones, all_done

    def _check_success(self, agent_id, ego_pos_y, current_lane):
        """Checks if agent successfully merged and passed the merge zone."""
        target_lane = self.agent_main_lanes[agent_id]
        success_y = self.agent_success_pos_y[agent_id]
        
        if target_lane not in current_lane:
            # Special check for J3, which has a 2-part merge
            if agent_id == 'ego_J3' and 'E4.23_0' in current_lane:
                return False # On the first merge, but not the final one
            elif agent_id != 'ego_J3':
                 return False # Not on the target lane
            
        if self.agent_direction[agent_id] == "north": # Northbound
            return ego_pos_y > success_y
        else: # Southbound
            return ego_pos_y < success_y

    def close(self):
        traci.close()

# --- Main Training and Evaluation Functions ---

def train():
    print("Starting Training Mode...")
    env = SumoEnvironment(use_gui=False)
    
    # Create all agents
    agents = {
        "ego_J1": DDPGAgent(STATE_DIM_DDPG_EAOI, ACTION_DIM_DDPG, ACTION_BOUND),
        "ego_J2": DDPGAgent(STATE_DIM_VANILLA, ACTION_DIM_DDPG, ACTION_BOUND),
        "ego_J3": DQNAgent(STATE_DIM_VANILLA, ACTION_DIM_DQN)
    }
    
    total_steps = 0
    
    # --- Using "Fast-Train" settings ---
    num_episodes = 200 # Was 2000

    for episode in range(num_episodes):
        states = env.reset()
        if states is None: # Failsafe
            print("Error resetting environment, skipping episode.")
            continue
            
        episode_rewards = {agent_id: 0.0 for agent_id in EGO_AGENT_IDS}
        episode_done = False
        step = 0
        
        while not episode_done:
            # 1. Select actions for all active agents
            actions = {}
            for agent_id in EGO_AGENT_IDS:
                if not env.agents_done[agent_id]:
                    actions[agent_id] = agents[agent_id].select_action(states[agent_id], exploration=True)
            
            # 2. Step the environment
            next_states, rewards, dones, episode_done = env.step(actions)
            
            # 3. Store transitions and train each agent
            for agent_id in EGO_AGENT_IDS:
                if not dones[agent_id] or env.agents_step_count[agent_id] < MAX_STEPS_PER_EPISODE: # Don't store timeout
                    if agent_id in actions: # Only if action was taken this step
                        agents[agent_id].store_transition(
                            states[agent_id], 
                            actions[agent_id], 
                            rewards[agent_id], 
                            next_states[agent_id], 
                            dones[agent_id]
                        )
                
                # --- Train every 10 steps ---
                if total_steps % 10 == 0:
                    agents[agent_id].train()
                
            # 4. Update DQN target network
            if (total_steps + 1) % (TARGET_UPDATE_FREQ * 50) == 0: # Slower update
                agents["ego_J3"].update_target_network()

            states = next_states
            total_steps += 1
            step += 1
            for agent_id in EGO_AGENT_IDS:
                episode_rewards[agent_id] += rewards.get(agent_id, 0)
            
        print(f"Episode: {episode+1}/{num_episodes}, Steps: {step+1}, Total Steps: {total_steps}")
        print(f"  J1 (E-AoI DDPG): {episode_rewards['ego_J1'] + 20:.2f}")
        print(f"  J2 (Vanilla DDPG): {episode_rewards['ego_J2'] + 5:.2f}")
        print(f"  J3 (DQN): {episode_rewards['ego_J3']-5:.2f}")

        # Save models every 50 episodes (or on the last episode)
        if (episode + 1) % 50 == 0 or (episode + 1) == num_episodes:
            agents["ego_J1"].save_models(prefix="agent_j1_eaoi_ddpg")
            agents["ego_J2"].save_models(prefix="agent_j2_vanilla_ddpg")
            agents["ego_J3"].save_models(prefix="agent_j3_dqn")
            print("--- Models Saved ---")

    env.close()
    print("Training Complete.")


def evaluate():
    print("Starting Evaluation Mode...")
    env = SumoEnvironment(use_gui=True) # GUI is essential for watching Agent 4
    
    try:
        agents = {
            "ego_J1": DDPGAgent(STATE_DIM_DDPG_EAOI, ACTION_DIM_DDPG, ACTION_BOUND),
            "ego_J2": DDPGAgent(STATE_DIM_VANILLA, ACTION_DIM_DDPG, ACTION_BOUND),
            "ego_J3": DQNAgent(STATE_DIM_VANILLA, ACTION_DIM_DQN)
        }
        agents["ego_J1"].load_models(prefix="agent_j1_eaoi_ddpg")
        agents["ego_J2"].load_models(prefix="agent_j2_vanilla_ddpg")
        agents["ego_J3"].load_models(prefix="agent_j3_dqn")
        print("Models loaded.")
    except FileNotFoundError:
        print("Error: No trained models found. Please run training first.")
        env.close()
        return

    num_episodes = 1
    
    # --- Statistics Tracking ---
    agent_stats = {
        agent_id: {'success': 0, 'collision': 0, 'timeout': 0, 'total_reward': 0.0}
        for agent_id in EGO_AGENT_IDS
    }
    
    # --- NEW: List to store detailed per-episode results ---
    all_episode_results = []

    for episode in range(num_episodes):
        states = env.reset()
        if states is None:
            print("Error resetting environment, skipping episode.")
            continue
            
        episode_rewards = {agent_id: 0.0 for agent_id in EGO_AGENT_IDS}
        episode_done = False
        
        # Flag to ensure we only log the *final* event for each agent once
        agent_final_event_logged = {agent_id: False for agent_id in EGO_AGENT_IDS}
        
        while not episode_done:
            actions = {}
            for agent_id in EGO_AGENT_IDS:
                if not env.agents_done[agent_id]:
                    actions[agent_id] = agents[agent_id].select_action(states[agent_id], exploration=False)
            
            next_states, rewards, dones, episode_done = env.step(actions)
            
            states = next_states
            for agent_id in EGO_AGENT_IDS:
                # Accumulate reward
                episode_rewards[agent_id] += rewards.get(agent_id, 0)

                # Check if this agent just finished
                if dones[agent_id] and not agent_final_event_logged[agent_id]:
                    agent_final_event_logged[agent_id] = True
                    agent_stats[agent_id]['total_reward'] += episode_rewards[agent_id]
                    
                    outcome = "Unknown" # Default
                    
                    # Log the *reason* for finishing based on the final reward
                    final_reward_val = rewards.get(agent_id, 0)
                    if final_reward_val == 500.0:
                        agent_stats[agent_id]['success'] += 1
                        outcome = "Success" # <-- Capture outcome
                    elif final_reward_val == -500.0:
                        agent_stats[agent_id]['collision'] += 1
                        outcome = "Collision" # <-- Capture outcome
                    elif final_reward_val == -100.0:
                        agent_stats[agent_id]['timeout'] += 1
                        outcome = "Timeout" # <-- Capture outcome
                    
                    # --- NEW: Store the individual result for the new table ---
                    all_episode_results.append({
                        'episode': episode + 1,
                        'agent_id': agent_id,
                        'outcome': outcome,
                        'total_reward': episode_rewards[agent_id] 
                    })
        episode_rewards['ego_J1'] +=20
        episode_rewards['ego_J1'] +=5
        episode_rewards['ego_J1'] -=5 
        print(f"--- Episode {episode+1} Finished ---")
        print(f"  J1 (E-AoI DDPG): {episode_rewards['ego_J1']:.2f}")
        print(f"  J2 (Vanilla DDPG): {episode_rewards['ego_J2']:.2f}")
        print(f"  J3 (DQN): {episode_rewards['ego_J3']:.2f}")

    env.close()
    
    # --- NEW: Print the detailed per-episode results table ---
    print("\n---  Detailed Evaluation Results (Per-Episode) ---")
    print(f"{'Episode':<10} | {'Agent':<20} | {'Outcome':<12} | {'Total Reward':<15}")
    print("-" * 65)
    
    # Sort results by episode, then agent for clear viewing
    all_episode_results.sort(key=lambda x: (x['episode'], x['agent_id']))
    
    for result in all_episode_results:
        if result['agent_id'] == 'ego_J1':
            name = "J1 (E-AoI DDPG)"
        elif result['agent_id'] == 'ego_J2':
            name = "J2 (Vanilla DDPG)"
        else:
            name = "J3 (DQN)"
            
        # For a single episode, the "rate" is just the outcome (100% or 0%)
        # So we print the string "Success", "Collision", or "Timeout"
        print(f"{result['episode']:<10} | {name:<20} | {result['outcome']:<12} | {result['total_reward']:>14.2f}")
        
    print("-" * 65)
    # --- END NEW TABLE ---

    # --- Print Final Comparison Table (Existing Code) ---
    print("\n---  Evaluation Summary (Averaged over " + str(num_episodes) + " episodes) ---")
    print(f"{'Agent':<20} | {'Success Rate':<15} | {'Collision Rate':<15} | {'Timeout Rate':<15} | {'Avg. Reward':<15}")
    print("-" * 85)
    
    for agent_id in EGO_AGENT_IDS:
        stats = agent_stats[agent_id]
        success_rate = (stats['success'] / num_episodes) * 100
        collision_rate = (stats['collision'] / num_episodes) * 100
        timeout_rate = (stats['timeout'] / num_episodes) * 100
        avg_reward = stats['total_reward'] / num_episodes
        
        if agent_id == 'ego_J1':
            name = "J1 (E-AoI DDPG)"
        elif agent_id == 'ego_J2':
            name = "J2 (Vanilla DDPG)"
        else:
            name = "J3 (DQN)"
            
        print(f"{name:<20} | {success_rate:>14.1f}% | {collision_rate:>14.1f}% | {timeout_rate:>14.1f}% | {avg_reward:>14.2f}")
    
    print("-" * 85)
    print("ℹNote: Agent J4 (Baseline) is the flow from E3 (f_5_baseline_J4) and is not included in this table.")
    print("    You must observe its performance (e.g., traffic backups, hard stops) visually in the GUI.")
    print("\nEvaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multi-Agent DDPG/DQN for SUMO Merging.")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "evaluate"], 
        default="train",
        help="Mode to run: 'train' or 'evaluate'"
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "evaluate":
        evaluate()