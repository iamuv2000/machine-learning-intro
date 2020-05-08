#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 03:58:12 2020

@author: yuvrajsingh
"""


#Import libraries
import pandas as pd
import matplotlib.pyplot as plt

#Import dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#TSA
import random
N = 10000
d = 10
ads_selected = []
numbers_of_reward_1 = [0]*d
numbers_of_reward_0 = [0]*d
total_reward = 0
for n in range(0,N):
    ad = 0
    max_random = 0
    for i in range(0,d):
        random_data = random.betavariate(numbers_of_reward_1[i]+1 ,numbers_of_reward_0[0]+1)
        if random_data > max_random :
            max_random = random_data
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if reward == 1:
        numbers_of_reward_1[ad] = numbers_of_reward_1[ad] + 1
    else:
        numbers_of_reward_0[ad] = numbers_of_reward_0[ad] + 1
    total_reward = total_reward +reward
    
#Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selected')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()