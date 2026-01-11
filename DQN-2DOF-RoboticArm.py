# %% [code] {"execution":{"iopub.status.busy":"2026-01-11T22:03:59.949734Z","iopub.execute_input":"2026-01-11T22:03:59.949958Z","iopub.status.idle":"2026-01-11T22:05:16.524104Z","shell.execute_reply.started":"2026-01-11T22:03:59.949934Z","shell.execute_reply":"2026-01-11T22:05:16.523347Z"}}
!pip install gymnasium stable-baselines3 matplotlib pillow


# %% [code] {"execution":{"iopub.status.busy":"2026-01-11T22:05:16.525701Z","iopub.execute_input":"2026-01-11T22:05:16.525934Z","iopub.status.idle":"2026-01-11T22:05:17.282707Z","shell.execute_reply.started":"2026-01-11T22:05:16.525911Z","shell.execute_reply":"2026-01-11T22:05:17.281907Z"}}
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class Arm2DEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        super().__init__()
        self.l1 = 1.0
        self.l2 = 1.0
        self.max_steps = 60
        self.action_step = 0.15
        self.threshold = 0.08

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.pi, -2.0, -2.0]),
            high=np.array([ np.pi,  np.pi,  2.0,  2.0]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.theta1 = np.random.uniform(-np.pi, np.pi)
        self.theta2 = np.random.uniform(-np.pi, np.pi)
        self.target = np.random.uniform(-1.5, 1.5, size=2)
        self.steps = 0
        return self._get_obs(), {}

    def _forward_kinematics(self):
        x = self.l1 * np.cos(self.theta1) + self.l2 * np.cos(self.theta1 + self.theta2)
        y = self.l1 * np.sin(self.theta1) + self.l2 * np.sin(self.theta1 + self.theta2)
        return np.array([x, y])

    def _get_obs(self):
        return np.array([self.theta1, self.theta2, self.target[0], self.target[1]], dtype=np.float32)

    def step(self, action):
        if action == 0: self.theta1 += self.action_step
        elif action == 1: self.theta1 -= self.action_step
        elif action == 2: self.theta2 += self.action_step
        elif action == 3: self.theta2 -= self.action_step

        ee_pos = self._forward_kinematics()
        dist = np.linalg.norm(ee_pos - self.target)
        reward = -dist - 0.01

        terminated = dist < self.threshold
        truncated = self.steps >= self.max_steps

        if terminated:
            reward += 10.0
        if truncated:
            reward -= 5.0

        self.steps += 1
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        fig, ax = plt.subplots(figsize=(4,4))
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)

        x1 = self.l1 * np.cos(self.theta1)
        y1 = self.l1 * np.sin(self.theta1)
        x2, y2 = self._forward_kinematics()

        ax.plot([0, x1], [0, y1], "k-", lw=3)
        ax.plot([x1, x2], [y1, y2], "b-", lw=3)
        ax.plot(self.target[0], self.target[1], "ro", markersize=8)

        ax.set_title("2-DOF Arm Reaching")
        plt.close(fig)
        return fig


# %% [code] {"execution":{"iopub.status.busy":"2026-01-11T22:05:17.283460Z","iopub.execute_input":"2026-01-11T22:05:17.283702Z","iopub.status.idle":"2026-01-11T22:08:00.481046Z","shell.execute_reply.started":"2026-01-11T22:05:17.283676Z","shell.execute_reply":"2026-01-11T22:08:00.480382Z"}}
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import torch

# Wrap environment for monitoring
env = Monitor(Arm2DEnv())
eval_env = Arm2DEnv()

# Callback to save best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./",
    log_path="./",
    eval_freq=2000,
    deterministic=True,
    render=False
)

# Create DQN agent
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=500,
    train_freq=4,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    verbose=1,
)

# Train
model.learn(total_timesteps=120_000, callback=eval_callback)

# Save final model and PyTorch weights
model.save("dqn_2dof_arm")
torch.save(model.policy.state_dict(), "best.pt")


# %% [code] {"execution":{"iopub.status.busy":"2026-01-11T22:08:00.482023Z","iopub.execute_input":"2026-01-11T22:08:00.482843Z","iopub.status.idle":"2026-01-11T22:08:00.814610Z","shell.execute_reply.started":"2026-01-11T22:08:00.482813Z","shell.execute_reply":"2026-01-11T22:08:00.813852Z"}}
import pandas as pd

# Get rewards from Monitor wrapper
episode_rewards = env.get_episode_rewards()

# Save CSV
pd.DataFrame(episode_rewards, columns=["reward"]).to_csv("reward_log.csv", index=False)

# Plot and save reward curve
plt.figure(figsize=(8,4))
plt.plot(episode_rewards, alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Performance")
plt.grid(True)
plt.savefig("reward_curve.png")
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2026-01-11T22:08:00.815519Z","iopub.execute_input":"2026-01-11T22:08:00.815895Z","iopub.status.idle":"2026-01-11T22:08:02.425115Z","shell.execute_reply.started":"2026-01-11T22:08:00.815868Z","shell.execute_reply":"2026-01-11T22:08:02.424472Z"}}
import matplotlib.animation as animation

def generate_gif(env, model, filename="arm_reaching.gif"):
    frames = []
    obs, _ = env.reset()

    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

        fig = env.render()
        frames.append(fig)

        if terminated or truncated:
            break

    anim = animation.ArtistAnimation(
        plt.figure(),
        [[f.axes[0]] for f in frames],
        interval=200
    )
    anim.save(filename, writer="pillow")
    print(f"GIF saved: {filename}")

# Run GIF generation
generate_gif(Arm2DEnv(), model)
