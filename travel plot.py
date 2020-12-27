#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import dates as mpl_dates
from csv import reader

# DATA READING AND STORING
temp = []

# Armenia
with open("travel.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)

data_arm = np.array(temp)

# Dates Universal
dates = data_arm[:, 0]

tot_travel_spain = data_arm[:, 4]
tot_travel_spain = np.array(tot_travel_spain).tolist()
for i in range(len(tot_travel_spain)):
    s = tot_travel_spain[i].split(",")
    te = "".join(s)
    tot_travel_spain[i] = float(te)
tot_travel_spain = np.array(tot_travel_spain)
print(tot_travel_spain)
# tot_travel_italy = data_arm[:, 7]
# tot_travel_italy = np.array(tot_travel_italy).tolist()
# for i in tot_travel_italy:
#     s = i.split(",")
#     te = "".join(s)
#     i = int(te)


# tot_travel_ge = data_arm[:, 4]
# tot_travel_ge = np.array(tot_travel_ge).tolist()
# for i in tot_travel_ge:
#     s = i.split(",")
#     te = "".join(s)
#     i = int(te)

# tot_travel_ne = data_arm[:, 8]
# tot_travel_ne = np.array(tot_travel_ne).tolist()
# for i in tot_travel_ne:
#     s = i.split(",")
#     te = "".join(s)
#     i = int(te)


barWidth = 0.25
fig = plt.subplots(figsize =(20, 20)) 

plt.bar(dates,tot_travel_spain) 
# Make the plot 
plt.title("Traveling in Germany this year")
# plt.show() 
plt.savefig('Untitled Folder/travel_ge.jpg')
plt.clf()


# In[ ]:




