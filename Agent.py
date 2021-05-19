from ReplayBuffer import ReplayBuffer
from DQN import build_dqn
import numpy as np
from tensorflow.keras.models import load_model

class Agent:
    
    def __init__(self, lr, gamma, n_actions, epsilon,
                    batch_size, input_dims, epsilon_dec=1e-3,
                    epsilon_end=0.01, memsize=1000000, 
                    fname='dqn_model.h5'):
        
        self.action = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(memsize, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)


    def store_transition(self, state, action, reward, state_, done):    
        self.memory.store_transition(state, action, reward, state_, done)

    
    def choose_action(self, observation):
        if np.random.random() < self.epsilon : 
            # Explore
            action = np.random.choice(self.action)
        else:
            # Send state (frame of sates) into the neural net
            state = np.array([observation])
            actions = self.q_eval.predict(state)

            # Choose the best based on neural net's decision
            action = np.argmax(actions)
        return action

    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size :
            # no learning happens if we haven't finished one batch yet
            return
        
        states, actions, rewards, states_, dones = \
            self.memory.sample_buffer(self.batch_size)
    

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + self.gamma*np.max(q_next, axis=1)*dones

        self.q_eval.train_on_batch(states, q_target)

        if self.epsilon > self.eps_min :
            self.epsilon = self.epsilon-self.dec
        else :
            self.epsilon = self.eps_min
    

    def save_model(self):
        self.q_eval.save(self.model_file)
    

    def load_model(self):
        self.q_eval = load_model(self.model_file)