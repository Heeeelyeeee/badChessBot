import gym
import gym_chess
import random

env = gym.make('Chess-v0')
print(env.render())

env.reset()
done = False

while not done:
    action = env.legal_moves[random.randrange(0,len(env.legal_moves))] #randomSample didnt work?
    state,s,done,other = env.step(action)
    print(env.render(mode='unicode'))
    print(done)
    print("--------------")

env.close()
