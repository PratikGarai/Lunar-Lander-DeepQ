from Agent import Agent
import numpy as np
import gym
import tensorflow as tf
from utils import plotLearning

if __name__=="__main__":
    tf.compat.v1.disable_eager_execution()
    env = gym.make('LunarLander-v2')
    lr = 0.001
    n_games = 500
    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, 
                    input_dims=env.observation_space.shape,
                    n_actions=env.action_space.n, mem_size=1000000,
                    batch_size=64, epsilon_end=0.10)

    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print("Episode : ", i)
        print("Score ", score)
        print("Average score : ", avg_score)
        print("Epsilon : ", agent.epsilon)
        print("-------------------------------------\n\n")


    filename = 'learning_rate.png'
    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)