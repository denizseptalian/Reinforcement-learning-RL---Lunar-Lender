import base64
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Constants
SEED = 0
MINIBATCH_SIZE = 64
TAU = 1e-3
E_DECAY = 0.995
E_MIN = 0.01

random.seed(SEED)

# Streamlit Interface
st.title("Q-Learning Agent with ε-Greedy Policy")
st.sidebar.header("Hyperparameters")

# Hyperparameters for the agent
epsilon = st.sidebar.slider("Initial Epsilon", 0.0, 1.0, 0.1)
tau = st.sidebar.slider("Tau (Soft Update Rate)", 0.0, 1.0, 1e-3)
minibatch_size = st.sidebar.slider("Mini-batch Size", 1, 128, 64)

st.sidebar.markdown("### Plotting Configuration")
show_rolling_mean = st.sidebar.checkbox("Show Rolling Mean", True)
rolling_window_size = st.sidebar.slider("Rolling Mean Window Size", 1, 50, 10)

# Dummy experience buffer (simulating the buffer with random data for illustration)
memory_buffer = [{'state': np.random.rand(8), 'action': random.randint(0, 3), 
                  'reward': random.random(), 'next_state': np.random.rand(8), 
                  'done': random.choice([True, False])} for _ in range(100)]

# Function to retrieve experiences (simulating experience retrieval)
def get_experiences(memory_buffer):
    experiences = random.sample(memory_buffer, k=minibatch_size)
    states = tf.convert_to_tensor([e['state'] for e in experiences], dtype=tf.float32)
    actions = tf.convert_to_tensor([e['action'] for e in experiences], dtype=tf.float32)
    rewards = tf.convert_to_tensor([e['reward'] for e in experiences], dtype=tf.float32)
    next_states = tf.convert_to_tensor([e['next_state'] for e in experiences], dtype=tf.float32)
    done_vals = tf.convert_to_tensor([e['done'] for e in experiences], dtype=tf.float32)
    return states, actions, rewards, next_states, done_vals

# Update epsilon based on decay
def get_new_eps(epsilon):
    return max(E_MIN, E_DECAY * epsilon)

# Action selection with epsilon-greedy
def get_action(q_values, epsilon=0.0):
    if random.random() > epsilon:
        return np.argmax(q_values.numpy()[0])
    else:
        return random.choice(np.arange(4))

# Plot history
def plot_history(point_history, window_size=10):
    window_size = max(1, int(window_size))
    rolling_mean = pd.Series(point_history).rolling(window=window_size).mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(point_history, label="Point History", color="blue", alpha=0.5)
    if show_rolling_mean:
        ax.plot(rolling_mean, label="Rolling Mean", color="red", linewidth=2)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Points")
    ax.legend()
    st.pyplot(fig)

# Simulated training loop to generate point history
point_history = [random.randint(0, 100) for _ in range(500)]  # Example points history

# Display Plot
st.subheader("Training History Plot")
plot_history(point_history, rolling_window_size)

# Display parameters
st.markdown("### Current Parameters:")
st.write(f"Initial Epsilon: {epsilon}")
st.write(f"Tau: {tau}")
st.write(f"Mini-batch Size: {minibatch_size}")
st.write(f"Rolling Mean Window Size: {rolling_window_size}")
