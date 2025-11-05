# Adaptive On-Ramp Merging Strategy Under Imperfect Communication Performance

## Executive Summary

This repository presents a comprehensive implementation and comparative analysis of adaptive on-ramp merging strategies for connected and automated vehicles (CAVs) under imperfect communication conditions. The project implements three distinct reinforcement learning algorithms—E-AoI-aware DDPG, Vanilla DDPG, and DQN—operating simultaneously in a complex multi-junction highway simulation. The work demonstrates that explicitly modeling communication quality metrics into the state representation enables robust adaptive behaviors that maintain safety and efficiency despite persistent packet loss and communication delays characteristic of real-world V2X networks.

## Problem Context & Motivation

### Research Challenge

Existing on-ramp merging algorithms typically assume perfect communication, a critical assumption rarely met in real-world deployments. Communication imperfections—packet loss, variable delays, and information staleness—directly impact autonomous vehicle safety and efficiency. This creates three interconnected challenges: (1) safety hazards from outdated or missing vehicle information, (2) control instability from unreliable state observations, and (3) efficiency degradation from suboptimal merging decisions.

### Key Research Questions

This implementation systematically addresses: How do different reinforcement learning architectures perform under realistic communication constraints? Does incorporating communication quality metrics into state representations provide tangible safety benefits? What are the trade-offs between continuous control (DDPG) and discrete control (DQN) in communication-constrained scenarios?

## Technical Contributions

### Multi-Agent Framework

**Three Algorithms Compared:**

1. **Agent J1 (E-AoI-DDPG)**: 15-dimensional state space incorporating Exponentially Weighted Average Age-of-Information metric. Couples speed reduction with elevated E-AoI through specialized reward term: $$r_{comm} = -\text{E-AoI} \times (v_{ego}/v_{max})$$. Enables communication-aware adaptive control.

2. **Agent J2 (Vanilla DDPG)**: 14-dimensional state space without communication awareness. Uses identical continuous jerk action space as Agent J1 but lacks communication penalty term. Serves as baseline for demonstrating communication-awareness benefits.

3. **Agent J3 (DQN)**: 14-dimensional state space with discrete 3-action space (accelerate, hold, decelerate). Provides robustness through simplified action selection at cost of fine-grained control.

### Comparison of Three Multi-Agent Algorithms

| Parameter | Agent J1 (E-AoI-DDPG) | Agent J2 (Vanilla DDPG) | Agent J3 (DQN) |
|-----------|----------------------|-------------------------|-----------------|
| **State Dimension** | 15D (includes E-AoI) | 14D (no E-AoI) | 14D (no E-AoI) |
| **Action Space** | Continuous jerk control [-3,3] m/s³ | Continuous jerk control [-3,3] m/s³ | Discrete: Accel/Hold/Decel |
| **Communication Awareness** | Yes (E-AoI metric) | No | No |
| **Primary Advantage** | Adaptive to communication quality; reduces speed under packet loss | Smooth continuous control; high efficiency in good conditions | Simple discrete actions; robust through conservatism |
| **Primary Weakness** | Slightly higher state complexity | Vulnerable to persistent packet loss | Discrete actions limit fine control |
| **Best Use Case** | High packet loss scenarios (20-40%) | Perfect/near-perfect communication | Conservative safety-first operations |

### Exponentially Weighted Average Age-of-Information (E-AoI)

Novel communication quality metric that weights information age by vehicle proximity:

$$\text{E-AoI} = \frac{\sum_{l=1}^{n} \alpha^l \Delta_l}{\sum_{l=1}^{n} \alpha^l \cdot \Delta_{max}}$$

Where $\alpha = 0.4$ exponential decay factor prioritizes information from nearest vehicles. Range [0, 1] maps perfect communication (0) to severe packet loss (1). Distance weighting ensures critical near-field information dominates decisions.

### Realistic Communication Simulation

Implements stochastic V2V channel with 30% packet loss probability, Age-of-Information tracking per vehicle, and kinematic motion prediction for lost packets. Position prediction model:

$$p_{pred} = p_{recv} + v \cdot t_{AoI} + \frac{1}{2}a \cdot t_{AoI}^2$$

This creates realistic conditions for evaluating algorithm robustness beyond idealized assumptions.

## System Architecture

### Training System Workflow

![System Workflow Diagram](sys_wflw.jpg)

The system follows a comprehensive training loop incorporating:
- Environment setup and SUMO simulation initialization
- State observation collection (V2V data + AoI metrics)
- State vector construction for each agent
- Agent-specific action selection (DDPG continuous vs DQN discrete)
- SUMO execution and step rewards
- Terminal condition evaluation and episode completion handling
- Network updates and replay buffer management
- Iterative training across 200 episodes with regular model checkpointing

### Training Configuration

| Parameter | DDPG (J1, J2) | DQN (J3) |
|-----------|----------------|----------|
| **Discount factor (γ)** | 0.99 | 0.99 |
| **Actor learning rate** | 1 × 10⁻⁴ | — |
| **Critic learning rate** | 1 × 10⁻³ | — |
| **Q-network learning rate** | — | 1 × 10⁻⁴ |
| **Replay buffer size** | 100,000 | 100,000 |
| **Batch size** | 64 | 64 |
| **Soft update factor (τ)** | 0.001 | — |
| **Target update frequency** | — | 20 episodes |
| **Epsilon decay rate** | — | 30,000 steps |
| **Epsilon range** | — | 1.0 → 0.05 |
| **Action bound (jerk)** | ±3.0 m/s³ | — |
| **Total training episodes** | 200 |
| **Maximum episode length** | 1,000 steps (200 s) |
| **Control frequency** | 5 Hz (0.2 s per step) |

### Multi-Junction SUMO Environment

Complex road network with three distinct merging junctions:
- **J1**: Southbound merge from ramp_S to main_S
- **J2**: Northbound merge from ramp_N to main_N  
- **J3**: Secondary southbound merge from sub_ramp_S to ramp_S

Vehicle dynamics: maximum speed 30 m/s, acceleration 2.6 m/s², deceleration -4.5 m/s², 5 Hz control frequency (0.2 s steps), 200 second maximum episode length.

### Neural Network Architectures

**DDPG Networks (Agents J1, J2):**
- Actor: Linear(state_dim, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 1)
- Critic: Parallel state path Linear(state_dim, 256) → ReLU, concatenated with action, fused through Linear(257, 128) → ReLU → Linear(128, 1)

**DQN Network (Agent J3):**
- Linear(state_dim, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 3) for Q-values

## Experimental Results

### Safety Performance

Agent J1 demonstrates superior collision avoidance through communication-adaptive control. When E-AoI increases due to packet loss, J1 proactively reduces speed maintaining safe margins. Agent J2 exhibits increased collision risk during persistent packet loss, lacking information staleness awareness. Agent J3's discrete action space provides robustness through conservative behavior but reduces efficiency.

### Communication Impact

E-AoI values during episodes range 0.0 (perfect communication) to ~1.0 (severe loss). Distance-weighted formulation ensures nearest vehicles' packet loss has strongest influence. The communication penalty term successfully trains J1 to couple speed reduction with elevated E-AoI, achieving adaptive behavior without explicit programming.

### Trajectory Characteristics

Successful merges reveal distinct patterns: Agent J1 shows smooth gradual adjustments based on traffic and E-AoI; Agent J2 demonstrates aggressive speed maintenance with sharper adjustments; Agent J3 exhibits stepwise discrete changes. Critical packet loss scenarios highlight J1's preemptive speed reduction, J2's risky continued aggression, and J3's inherent conservatism with longer merge times.

## Implementation Framework

### Repository Structure

```
adaptive-merging-strategy/
├── src/
│   ├── main_new_network.py          # Training/evaluation entry point
│   ├── agents/                      # J1 E-AoI-DDPG, J2 DDPG, J3 DQN
│   ├── environment/                 # SUMO integration, V2V simulator
│   └── utils/                       # Metrics, logging, visualization
├── sumo_config/
│   ├── Test2.net.xml               # Network topology
│   └── test2_agents.rou.xml        # Route definitions
├── models/                          # Checkpointed model weights
├── results/                         # Training metrics, logs
├── docs/
│   ├── report.tex                  # Full technical report
│   └── presentations/              # PPTX slides
└── README.md
```

### Setup & Execution

```bash
# Environment setup
pip install torch numpy matplotlib scipy
export SUMO_HOME=/usr/share/sumo  # Linux
# or set SUMO_HOME=C:\Program Files\SUMO  # Windows

# Training
python src/main_new_network.py --mode train --episodes 200

# Evaluation with visualization
python src/main_new_network.py --mode evaluate --gui
```

## Key Findings

1. **Communication awareness improves safety**: E-AoI-DDPG demonstrates adaptive control reducing collision risk compared to communication-agnostic approaches under 30% packet loss.

2. **Continuous control outperforms discrete**: DDPG algorithms produce smoother trajectories than DQN, important for passenger comfort in safety-critical applications.

3. **Reward shaping is critical**: Communication penalty term successfully trains desired adaptive behavior, demonstrating reward function design importance.

4. **Fair comparison framework enabled by multi-agent synchronization**: Unified environment with identical conditions provides robust comparative evidence.

## Limitations & Future Directions

**Current Limitations:**
- Agents trained on specific traffic densities; generalization requires domain randomization
- Single merging vehicle per junction; real-world scenarios need multi-vehicle coordination
- Fixed 30% packet loss; real communication varies spatially and temporally
- Simplified vehicle dynamics without actuator delays or friction limits

**Future Improvements:**
- Adaptive communication modeling with distance/density dependence
- Explicit multi-agent coordination protocols
- Curriculum learning from simple to complex scenarios
- Formal safety verification integration
- Real-world V2X testbed deployment

## References

- Tong et al., "Adaptive on-ramp merging strategy under imperfect communication performance," *Vehicular Communications*, vol. 44, 2023
- Lillicrap et al., "Continuous control with deep reinforcement learning," ICLR 2016
- Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, 2015
- Lopez et al., "Microscopic traffic simulation using SUMO," ITSC 2018
- Yates et al., "Age of information: an introduction and survey," *IEEE J-SAC*, vol. 39, 2021

## Team

- **Dev Prajapati** (231CS120)
- **Dhruv Sandilya** (231CS122)
- **Vrishank Honnavalli** (231CS165)
- **Advaith Nair** (231CS205)

Department of Computer Science and Engineering, National Institute of Technology Karnataka, Surathkal

