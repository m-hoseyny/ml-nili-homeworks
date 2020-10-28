# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 08:59:26 2018

@author: Dr-Abbasfar
"""

import numpy as np
import matplotlib.pyplot as plt

EXIT_CAPITAL = 100

gamma = 0.9
delta = 1
theta = 1e-6
p = 0.4
win_factor = 1
loss_factor = 1

states = np.arange(1, EXIT_CAPITAL)
values = np.zeros(EXIT_CAPITAL+1 + (win_factor-1) * EXIT_CAPITAL//2)
policy = np.zeros(EXIT_CAPITAL+1 + (win_factor-1) * EXIT_CAPITAL//2)
    
    
def Gambler():
    global gamma, delta, theta, p, win_factor, loss_factor, states, values, policy
    states = np.arange(1, EXIT_CAPITAL)
    values = np.zeros(EXIT_CAPITAL+1 + (win_factor-1) * EXIT_CAPITAL//2)
    policy = np.zeros(EXIT_CAPITAL+1 + (win_factor-1) * EXIT_CAPITAL//2)
    delta = 1
    
    def reward_func(r):
        if r == EXIT_CAPITAL:
            return 1
        else:
            return 0
            
    
    def action_space(s):
        return [i for i in range(min(s, EXIT_CAPITAL - s)+1)]
    
    round_ = 0
    
    while delta > theta:
        delta = 0
        print("round : ", round_)
        for state in states:
            v = values[state]
            actions = action_space(state)
            # Calculate Sigma
            sigma = []
            for action in actions:
                win = state + win_factor * action
                loss = state - loss_factor * action
                if loss < 0:
                    loss = 0
                Q = p * (reward_func(win) + gamma * values[win]) +  \
                    (1 - p) * (reward_func(loss) + gamma * values[loss])
    #            print(Q)
                sigma.append(Q)
                
            policy[state] = np.argmax(sigma)
            values[state] = np.max(sigma)
            
            delta = max(delta, abs(v - values[state]))
        print("delta : ",delta)
    
        round_ += 1
    return sum(1-values)


# plot Value Function
def plot():
    plt.plot(values[1:100])
    plt.title("Value Function for gamma = {}, win factor = {}, loss factor = {}".format(gamma, win_factor, loss_factor))
    plt.ylabel("Values")
    plt.xlabel("Capital")
    plt.savefig("ValueFunction_{}_{}_{}.jpg".format(gamma, win_factor, loss_factor))
    plt.show()
    
    plt.plot(policy[1:100])
    plt.title("policy Function for gamma = {}, win factor = {}, loss factor = {}".format(gamma, win_factor, loss_factor))
    plt.ylabel("policy")
    plt.xlabel("Capital")
    plt.savefig("PolicyFunction_{}_{}_{}.jpg".format(gamma, win_factor, loss_factor))
    plt.show()


# part a gamma = 0.9
regret = Gambler()
print("For gamma = {}, win factor = {}, loss factor = {} regre is : ".format(gamma, win_factor, loss_factor))
print("\t", regret)
plot()

# part a gamma = 1
gamma = 1
win_factor = 1
loss_factor = 1
regret = Gambler()
print("For gamma = {}, win factor = {}, loss factor = {} regre is : ".format(gamma, win_factor, loss_factor))
print("\t", regret)
plot()

#part b gamma = 0.9 , win_factor = 2
gamma = 0.9
win_factor = 2
loss_factor = 1
regret = Gambler()
print("For gamma = {}, win factor = {}, loss factor = {} regre is : ".format(gamma, win_factor, loss_factor))
print("\t", regret)
plot()


#part c gamma = 0.9 , win_factor = 2, loss_factor = 2
gamma = 0.9
win_factor = 2
loss_factor = 2
regret = Gambler()
print("For gamma = {}, win factor = {}, loss factor = {} regre is : ".format(gamma, win_factor, loss_factor))
print("\t", regret)
plot()














