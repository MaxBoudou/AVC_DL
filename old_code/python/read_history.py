# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:52:00 2020

@author: 20100
"""
import pickle
import matplotlib.pyplot as plt

model_index = 29
model_path = f"models/model_{model_index}"

objects = []
with (open(f"{model_path}/trainHistoryDict_{model_index}", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
        
sp = plt.subplots(1,3)
print(objects[0])
loss_plot = plt.plot(objects['loss'])