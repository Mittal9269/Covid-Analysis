#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import dates as mpl_dates
from csv import reader


# DATA READING AND STORING
temp = []

# Armenia
with open("armenia.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)

data_arm = np.array(temp)

# Dates Universal
dates = data_arm[:, 0]

# Armenia data
tot_cases_arm = data_arm[:, 1]
tot_cases_arm[tot_cases_arm == ''] = '0.0'
tot_cases_arm = tot_cases_arm.astype(np.float)

new_cases_arm = data_arm[:, 3]
new_cases_arm[new_cases_arm == ''] = '0.0'
new_cases_arm = new_cases_arm.astype(np.float)

tot_deaths_arm = data_arm[:, 4]
tot_deaths_arm[tot_deaths_arm == ''] = '0.0'
tot_deaths_arm = tot_deaths_arm.astype(np.float)

new_deaths_arm = data_arm[:, 6]
new_deaths_arm[new_deaths_arm == ''] = '0.0'
new_deaths_arm = new_deaths_arm.astype(np.float)

r_arm = data_arm[:, 13]
r_arm[r_arm == ''] = '0.0'
r_arm = r_arm.astype(np.float)


temp.clear()


# Australia
with open("aus.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)

data_aus = np.array(temp)

# Australia Data
tot_cases_aus = data_aus[:, 1]
tot_cases_aus[tot_cases_aus == ''] = '0.0'
tot_cases_aus = tot_cases_aus.astype(np.float)

new_cases_aus = data_aus[:, 3]
new_cases_aus[new_cases_aus == ''] = '0.0'
new_cases_aus = new_cases_aus.astype(np.float)

tot_deaths_aus = data_aus[:, 4]
tot_deaths_aus[tot_deaths_aus == ''] = '0.0'
tot_deaths_aus = tot_deaths_aus.astype(np.float)

new_deaths_aus = data_aus[:, 6]
new_deaths_aus[new_deaths_aus == ''] = '0.0'
new_deaths_aus = new_deaths_aus.astype(np.float)

r_aus = data_aus[:, 13]
r_aus[r_aus == ''] = '0.0'
r_aus = r_aus.astype(np.float)

new_tests_aus = data_aus[:, 26]
new_tests_aus[new_tests_aus == ''] == '0.0'
new_tests_aus = new_tests_aus.astype(np.float)

stringency_aus = data_aus[:, 31]
stringency_aus[stringency_aus == ''] = '0.0'
stringency_aus = stringency_aus.astype(np.float)

temp.clear()


# China
with open("china.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)

data_ch = np.array(temp)

# China Data
tot_cases_ch = data_ch[:, 1]
tot_cases_ch[tot_cases_ch == ''] = '0.0'
tot_cases_ch = tot_cases_ch.astype(np.float)

new_cases_ch = data_ch[:, 3]
new_cases_ch[new_cases_ch == ''] = '0.0'
new_cases_ch = new_cases_ch.astype(np.float)

tot_deaths_ch = data_ch[:, 4]
tot_deaths_ch[tot_deaths_ch == ''] = '0.0'
tot_deaths_ch = tot_deaths_ch.astype(np.float)

new_deaths_ch = data_ch[:, 6]
new_deaths_ch[new_deaths_ch == ''] = '0.0'
new_deaths_ch = new_deaths_ch.astype(np.float)

r_ch = data_ch[:, 13]
r_ch[r_ch == ''] = '0.0'
r_ch = r_ch.astype(np.float)

stringency_ch = data_ch[:, 31]
stringency_ch[stringency_ch == ''] = '0.0'
stringency_ch = stringency_ch.astype(np.float)

temp.clear()


# Germany
with open("germany.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)

data_de = np.array(temp)

# Germany Data
tot_cases_de = data_de[:, 1]
tot_cases_de[tot_cases_de == ''] = '0.0'
tot_cases_de = tot_cases_de.astype(np.float)

new_cases_de = data_de[:, 3]
new_cases_de[new_cases_de == ''] = '0.0'
new_cases_de = new_cases_de.astype(np.float)

tot_deaths_de = data_de[:, 4]
tot_deaths_de[tot_deaths_de == ''] = '0.0'
tot_deaths_de = tot_deaths_de.astype(np.float)

new_deaths_de = data_de[:, 6]
new_deaths_de[new_deaths_de == ''] = '0.0'
new_deaths_de = new_deaths_de.astype(np.float)

r_de = data_de[:, 13]
r_de[r_de == ''] = '0.0'
r_de = r_de.astype(np.float)

new_tests_de = data_de[:, 26]
new_tests_de[new_tests_de == ''] == '0.0'
new_tests_de = new_tests_de.astype(np.float)

stringency_de = data_de[:, 31]
stringency_de[stringency_de == ''] = '0.0'
stringency_de = stringency_de.astype(np.float)

temp.clear()


# India
with open("ind.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)

data_in = np.array(temp)

# India Data
tot_cases_in = data_in[:, 1]
tot_cases_in[tot_cases_in == ''] = '0.0'
tot_cases_in = tot_cases_in.astype(np.float)

new_cases_in = data_arm[:, 3]
new_cases_in[new_cases_in == ''] = '0.0'
new_cases_in = new_cases_in.astype(np.float)

tot_deaths_in = data_in[:, 4]
tot_deaths_in[tot_deaths_in == ''] = '0.0'
tot_deaths_in = tot_deaths_in.astype(np.float)

new_deaths_in = data_in[:, 6]
new_deaths_in[new_deaths_in == ''] = '0.0'
new_deaths_in = new_deaths_in.astype(np.float)

r_in = data_in[:, 13]
r_in[r_in == ''] = '0.0'
r_in = r_in.astype(np.float)

new_tests_in = data_in[:, 26]
new_tests_in[new_tests_in == ''] == '0.0'
new_tests_in = new_tests_in.astype(np.float)

stringency_in = data_in[:, 31]
stringency_in[stringency_de == ''] = '0.0'
stringency_in = stringency_in.astype(np.float)

temp.clear()


# Italy
with open("italy.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)

data_it = np.array(temp)

# Italy Data
tot_cases_it = data_it[:, 1]
tot_cases_it[tot_cases_it == ''] = '0.0'
tot_cases_it = tot_cases_it.astype(np.float)

new_cases_it = data_it[:, 3]
new_cases_it[new_cases_it == ''] = '0.0'
new_cases_it = new_cases_it.astype(np.float)

tot_deaths_it = data_it[:, 4]
tot_deaths_it[tot_deaths_it == ''] = '0.0'
tot_deaths_it = tot_deaths_it.astype(np.float)

new_deaths_it = data_it[:, 6]
new_deaths_it[new_deaths_it == ''] = '0.0'
new_deaths_it = new_deaths_it.astype(np.float)

r_it = data_it[:, 13]
r_it[r_it == ''] = '0.0'
r_it = r_it.astype(np.float)

new_tests_it = data_it[:, 26]
new_tests_it[new_tests_it == ''] == '0.0'
new_tests_it = new_tests_it.astype(np.float)

stringency_it = data_it[:, 31]
stringency_it[stringency_it == ''] = '0.0'
stringency_it = stringency_it.astype(np.float)
temp.clear()


# Japan
with open("japan.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)

data_jp = np.array(temp)

# Japan Data
tot_cases_jp = data_jp[:, 1]
tot_cases_jp[tot_cases_jp == ''] = '0.0'
tot_cases_jp = tot_cases_jp.astype(np.float)

new_cases_jp = data_jp[:, 3]
new_cases_jp[new_cases_jp == ''] = '0.0'
new_cases_jp = new_cases_jp.astype(np.float)

tot_deaths_jp = data_jp[:, 4]
tot_deaths_jp[tot_deaths_jp == ''] = '0.0'
tot_deaths_jp = tot_deaths_jp.astype(np.float)

new_deaths_jp = data_jp[:, 6]
new_deaths_jp[new_deaths_jp == ''] = '0.0'
new_deaths_jp = new_deaths_jp.astype(np.float)

r_jp = data_jp[:, 13]
r_jp[r_jp == ''] = '0.0'
r_jp = r_jp.astype(np.float)

new_tests_jp = data_jp[:, 26].astype(np.float)
stringency_jp = data_jp[:, 31].astype(np.float)
# PLOTING

# Armenia
# Total Cases
plt.figure(1)
plt.plot(dates, tot_cases_arm)
plt.xticks([])
plt.yticks(np.arange(np.min(tot_cases_arm), np.max(tot_cases_arm), 10000.0))
plt.title("Total Cases Armenia")
# plt.show()
plt.savefig('Untitled Folder/tot_Ar.jpg')
plt.clf()

# New Cases
plt.figure(2)
plt.plot(dates, new_cases_arm)
plt.xticks([])
plt.yticks(np.arange(np.min(new_cases_arm), np.max(new_cases_arm), 250.0))
plt.title("Daily New Cases Armenia")
# plt.show()
plt.savefig('Untitled Folder/daily_Ar.jpg')
plt.clf()

# Total Deaths
plt.figure(3)
plt.plot(dates, tot_deaths_arm)
plt.xticks([])
plt.yticks(np.arange(np.min(tot_deaths_arm), np.max(tot_deaths_arm), 250.0))
plt.title("Total Deaths Armenia")
# plt.show()
plt.savefig('Untitled Folder/totd_Ar.jpg')
plt.clf()

# New Deaths
plt.figure(4)
plt.plot(dates, new_deaths_arm)
plt.xticks([])
plt.yticks(np.arange(np.min(new_deaths_arm), np.max(new_deaths_arm), 250.0))
plt.title("Daily New Deaths Armenia")
# plt.show()
plt.savefig('Untitled Folder/newd_Ar.jpg')
plt.clf() 

# Reproduction Rate
plt.figure(5)
fig = plt.figure(figsize =(10, 7)) 

plt.boxplot(r_arm) 
plt.title("COVID19 Reproduction Rate Armenia")
# plt.show()
plt.savefig('Untitled Folder/rr_Ar.jpg')
plt.clf() 


# Australia
# Total Cases
plt.figure(6)
plt.plot(dates, tot_cases_aus)
plt.xticks([])
plt.yticks(np.arange(np.min(tot_cases_aus), np.max(tot_cases_aus), 5000.0))
plt.title("Total Cases Australia")
# plt.show()
plt.savefig('Untitled Folder/tot_aus.jpg')
plt.clf() 

# New Cases
plt.figure(7)
plt.plot(dates, new_cases_aus)
plt.xticks([])
plt.yticks(np.arange(np.min(new_cases_aus), np.max(new_cases_aus), 50.0))
plt.title("Daily New Cases Australia")
# plt.show()
plt.savefig('Untitled Folder/daily_aus.jpg')
plt.clf() 

# Total Deaths
plt.figure(8)
plt.plot(dates, tot_deaths_aus)
plt.xticks([])
plt.yticks(np.arange(np.min(tot_deaths_aus), np.max(tot_deaths_aus), 100.0))
plt.title("Total Deaths Australia")
# plt.show()
plt.savefig('Untitled Folder/totd_aus.jpg')
plt.clf() 

# New Deaths
plt.figure(9)
plt.plot(dates, new_deaths_aus)
plt.xticks([])
plt.yticks(np.arange(0, np.max(new_deaths_aus), 2.0))
plt.title("Daily New Deaths Australia")
# plt.show()
plt.savefig('Untitled Folder/newd_aus.jpg')
plt.clf() 

# Reproduction Rate
plt.figure(10)
fig = plt.figure(figsize =(10, 7)) 

plt.boxplot(r_aus) 
plt.title("COVID19 Reproduction Rate Australia")
# plt.show()
plt.savefig('Untitled Folder/rr_aus.jpg')
plt.clf() 

# New Tests
plt.figure(11)
plt.plot(dates, new_tests_aus)
plt.xticks([])
plt.yticks(np.arange(np.min(new_tests_aus), np.max(new_tests_aus), 10000))
plt.title("Daily New Tests Australia")
# plt.show()
plt.savefig('Untitled Folder/newt_aus.jpg')
plt.clf() 

# New Cases vs Stringency
fig, ax1 = plt.subplots()
plt.title("New Cases vs Stringency Australia")
color = 'tab:red'
ax1.plot(dates, new_cases_aus, color=color)
ax1.set_xticks([])
plt.yticks(np.arange(np.min(new_cases_aus), np.max(new_cases_aus), 100.0), color=color)
ax1.set_ylabel('New Cases', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(dates, stringency_aus, color=color)
ax2.set_xticks([])
plt.yticks(np.arange(np.min(stringency_aus), np.max(stringency_aus), 10.0), color=color)
ax2.set_ylabel('Stringency', color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()

# plt.show()
plt.savefig('Untitled Folder/newVsstr_aus.jpg')
plt.clf()

# New Cases vs Stringency
fig, ax1 = plt.subplots()
plt.title("New Cases vs New tests")
color = 'tab:blue'
ax1.plot(dates, new_cases_aus, color=color)
ax1.set_xticks([])
plt.yticks(np.arange(np.min(new_tests_aus), np.max(new_tests_aus), 1000.0), color=color)
ax1.set_ylabel('New Cases', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.plot(dates, new_tests_aus, color=color)
ax2.set_xticks([])
plt.yticks(np.arange(np.min(new_tests_aus), np.max(new_tests_aus), 1000.0), color=color)
ax2.set_ylabel('New tests', color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()

plt.show()
# plt.savefig('Untitled Folder/newVsnewt_aus.jpg')
plt.clf()

# China
# Total Cases
plt.figure(12)
plt.plot(dates, tot_cases_ch)
plt.xticks([])
plt.yticks(np.arange(0, np.max(tot_cases_ch), 10000.0))
plt.title("Total Cases China")
# plt.show()
plt.savefig('Untitled Folder/tot_ch.jpg')
plt.clf()

# New Cases
plt.figure(13)
plt.plot(dates, new_cases_ch)
plt.xticks([])
plt.yticks(np.arange(0, np.max(new_cases_ch), 500.0))
plt.title("Daily New Cases China")
# plt.show()
plt.savefig('Untitled Folder/daily_ch.jpg')
plt.clf()

# Total Deaths
plt.figure(14)
plt.plot(dates, tot_deaths_ch)
plt.xticks([])
plt.yticks(np.arange(0, np.max(tot_deaths_ch), 500.0))
plt.title("Total Deaths China")
# plt.show()
plt.savefig('Untitled Folder/totd_ch.jpg')
plt.clf()

# New Deaths
plt.figure(15)
plt.plot(dates, new_deaths_ch)
plt.xticks([])
plt.yticks(np.arange(np.min(new_deaths_ch), np.max(new_deaths_ch), 20.0))
plt.title("Daily New Deaths China")
# plt.show()
plt.savefig('Untitled Folder/newd_ch.jpg')
plt.clf()

# Reproduction Rate
plt.figure(16)
fig = plt.figure(figsize =(10, 7)) 

plt.boxplot(r_ch) 
plt.title("COVID19 Reproduction Rate China")
# plt.show()
plt.savefig('Untitled Folder/rr_ch.jpg')
plt.clf()

# New Cases vs Stringency
fig2, ax1 = plt.subplots()
plt.title("New Cases vs Stringency China")
color = 'tab:red'
ax1.plot(dates, new_cases_ch, color=color)
ax1.set_xticks([])
plt.yticks(np.arange(0, np.max(new_cases_ch), 500.0), color=color)
ax1.set_ylabel('New Cases', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(dates, stringency_ch, color=color)
ax2.set_xticks([])
plt.yticks(np.arange(0, np.max(stringency_ch), 4), color=color)
ax2.set_ylabel('Stringency', color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig2.tight_layout()

# plt.show()
plt.savefig('Untitled Folder/newVsstr_ch.jpg')
plt.clf()


# Germany
# Total Cases
plt.figure(17)
plt.plot(dates, tot_cases_de)
plt.xticks([])
plt.yticks(np.arange(np.min(tot_cases_de), np.max(tot_cases_de), 50000.0))
plt.title("Total Cases Germany")
# plt.show()
plt.savefig('Untitled Folder/tot_de.jpg')
plt.clf() 

# New Cases
plt.figure(7,figsize=(10,18))
plt.plot(dates, new_cases_de)
plt.xticks([])
plt.yticks(np.arange(np.min(new_cases_de), np.max(new_cases_de), 250.0))
plt.title("Daily New Cases Germany")
# plt.show()
plt.savefig('Untitled Folder/daily_de.jpg')
plt.clf() 

# Total Deaths
plt.figure(8,figsize=(10,18))
plt.plot(dates, tot_deaths_de)
plt.xticks([])
plt.yticks(np.arange(np.min(tot_deaths_de), np.max(tot_deaths_de), 250.0))
plt.title("Total Deaths Germany")
# plt.show()
plt.savefig('Untitled Folder/totd_de.jpg')
plt.clf() 

# New Deaths
plt.figure(9,figsize=(10,10))
plt.plot(dates, new_deaths_de)
plt.xticks([])
plt.yticks(np.arange(np.min(new_deaths_de), np.max(new_deaths_de), 50.0))
plt.title("Daily New Deaths Germany")
# plt.show()
plt.savefig('Untitled Folder/newd_de.jpg')
plt.clf() 

# Reproduction Rate
plt.figure(10,figsize=(10,18))
fig = plt.figure(figsize =(10, 7)) 

plt.boxplot(r_de) 
plt.title("COVID19 Reproduction Rate Germany")
# plt.show()
plt.savefig('Untitled Folder/rr_de.jpg')
plt.clf() 

# New Tests
plt.figure(11,figsize=(10,18))
plt.plot(dates, new_tests_de)
plt.xticks([])
plt.yticks(np.arange(np.min(new_tests_de), np.max(new_tests_de), 5000))
plt.title("Daily New Tests Germany")
# plt.show()
plt.savefig('Untitled Folder/newt_de.jpg')
plt.clf() 

# New Cases vs Stringency
fig, ax1 = plt.subplots()
plt.title("New Cases vs Stringency Germany")
color = 'tab:red'
ax1.plot(dates, new_cases_de, color=color)
ax1.set_xticks([])
plt.yticks(np.arange(np.min(new_cases_de), np.max(new_cases_de), 100.0), color=color)
ax1.set_ylabel('New Cases', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(dates, stringency_de, color=color)
ax2.set_xticks([])
plt.yticks(np.arange(np.min(stringency_de), np.max(stringency_de), 10.0), color=color)
ax2.set_ylabel('Stringency', color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.set_size_inches(10.5, 22.5)
fig.tight_layout()
# plt.show()
plt.savefig('Untitled Folder/newVsstr_de.jpg')
plt.clf() 

# New Cases vs New tests
fig, ax1 = plt.subplots()
plt.title("New Cases vs New tests Germany")
color = 'tab:blue'
ax1.plot(dates, new_cases_de, color=color)
ax1.set_xticks([])
plt.yticks(np.arange(np.min(new_tests_de), np.max(new_tests_de), 10000.0), color=color)
ax1.set_ylabel('New Cases', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.plot(dates, new_tests_de, color=color)
ax2.set_xticks([])
plt.yticks(np.arange(np.min(new_tests_de), np.max(new_tests_de), 10000.0), color=color)
ax2.set_ylabel('New tests', color=color)
ax2.tick_params(axis='y', labelcolor=color)
# fig.set_size_inches(10.5, 22.5)
fig.tight_layout()
# plt.show()
plt.savefig('Untitled Folder/newVsnewt_de.jpg')
plt.clf() 
           
# India
# Total Cases
plt.figure(17,figsize=(10,22))
plt.plot(dates, tot_cases_in)
plt.xticks([])
plt.yticks(np.arange(np.min(tot_cases_in),np.max(tot_cases_in), 100000.0))
plt.title("Total Cases India")
# plt.show()
plt.savefig('Untitled Folder/tot_in.jpg')
plt.clf() 

# New Cases
plt.figure(7)
plt.plot(dates, new_cases_in)
plt.xticks([])
plt.yticks(np.arange(np.min(new_cases_in), np.max(new_cases_in), 500.0))
plt.title("Daily New Cases India")
# plt.show()
plt.savefig('Untitled Folder/daily_in.jpg')
plt.clf() 

# Total Deaths
plt.figure(8,figsize=(10,24))
plt.plot(dates, tot_deaths_in)
plt.xticks([])
plt.yticks(np.arange(np.min(tot_deaths_in), np.max(tot_deaths_in), 3000.0))
plt.title("Total Deaths India")
# plt.show()
plt.savefig('Untitled Folder/totd_in.jpg')
plt.clf() 

# New Deaths
plt.figure(9,figsize=(10,14))
plt.plot(dates, new_deaths_in)
plt.xticks([])
plt.yticks(np.arange(np.min(new_deaths_in), np.max(new_deaths_in), 150.0))
plt.title("Daily New Deaths India")
# plt.show()
plt.savefig('Untitled Folder/newd_in.jpg')
plt.clf() 

# Reproduction Rate
plt.figure(10)
fig = plt.figure(figsize =(10, 7)) 

plt.boxplot(r_in) 
plt.title("COVID19 Reproduction Rate India")
# plt.show()
plt.savefig('Untitled Folder/rr_in.jpg')
plt.clf() 

# New Tests
plt.figure(11,figsize=(10,20))
plt.plot(dates, new_tests_in)
plt.xticks([])
plt.yticks(np.arange(np.min(new_tests_in), np.max(new_tests_in), 20000))
plt.title("Daily New Tests India")
# plt.show()
plt.savefig('Untitled Folder/newt_in.jpg')
plt.clf() 

# New Cases vs Stringency
fig, ax1 = plt.subplots()
plt.title("New Cases vs Stringency India")
color = 'tab:red'
ax1.plot(dates, new_cases_in, color=color)
ax1.set_xticks([])
plt.yticks(np.arange(np.min(new_cases_in), np.max(new_cases_in), 100.0), color=color)
ax1.set_ylabel('New Cases', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(dates, stringency_in, color=color)
ax2.set_xticks([])
plt.yticks(np.arange(np.min(stringency_in), np.max(stringency_in), 10.0), color=color)
ax2.set_ylabel('Stringency', color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()

# plt.show()
plt.savefig('Untitled Folder/newVsstr_in.jpg')
plt.clf() 

# New Cases vs New tests
fig, ax1 = plt.subplots()
plt.title("New Cases vs New tests India")
color = 'tab:blue'
ax1.plot(dates, new_cases_in, color=color)
ax1.set_xticks([])
plt.yticks(np.arange(np.min(new_cases_in), np.max(new_cases_in), 50000.0), color=color)
ax1.set_ylabel('New Cases', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.plot(dates, new_tests_in, color=color)
ax2.set_xticks([])
plt.yticks(np.arange(np.min(new_tests_in), np.max(new_tests_in), 50000.0), color=color)
ax2.set_ylabel('New tests', color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()

plt.show()
# plt.savefig('Untitled Folder/newVsnewt_in.jpg')
plt.clf() 

# italy
# Total Cases
plt.figure(17,figsize=(10,18))
plt.plot(dates, tot_cases_it)
plt.xticks([])
plt.yticks(np.arange(np.min(tot_cases_it), np.max(tot_cases_it), 50000.0))
plt.title("Total Cases Italy")
# plt.show()
plt.savefig('Untitled Folder/tot_it.jpg')
plt.clf() 

# New Cases
plt.figure(7,figsize=(10,20))
plt.plot(dates, new_cases_it)
plt.xticks([])
plt.yticks(np.arange(np.min(new_cases_it), np.max(new_cases_it), 400.0))
plt.title("Daily New Cases Italy")
# plt.show()
plt.savefig('Untitled Folder/daily_it.jpg')
plt.clf() 

# Total Deaths
plt.figure(8,figsize=(10,16))
plt.plot(dates, tot_deaths_it)
plt.xticks([])
plt.yticks(np.arange(np.min(tot_deaths_it), np.max(tot_deaths_it), 1000.0))
plt.title("Total Deaths Italy")
# plt.show()
plt.savefig('Untitled Folder/totd_it.jpg')
plt.clf() 

# New Deaths
plt.figure(9)
plt.plot(dates, new_deaths_it)
plt.xticks([])
plt.yticks(np.arange(np.min(new_deaths_it), np.max(new_deaths_it), 250.0))
plt.title("Daily New Deaths Italy")
# plt.show()
plt.savefig('Untitled Folder/newd_it.jpg')
plt.clf() 

# Reproduction Rate
plt.figure(10)
fig = plt.figure(figsize =(10, 7)) 

plt.boxplot(r_it) 
plt.title("COVID19 Reproduction Rate Italy")
# plt.show()
plt.savefig('Untitled Folder/rr_it.jpg')
plt.clf() 

# New Tests
plt.figure(11,figsize=(10,14))
plt.plot(dates, new_tests_it)
plt.xticks([])
plt.yticks(np.arange(np.min(new_tests_it), np.max(new_tests_it), 5000))
plt.title("Daily New Tests Italy")
# plt.show()
plt.savefig('Untitled Folder/newt_it.jpg')
plt.clf() 

# New Cases vs Stringency
fig, ax1 = plt.subplots()
plt.title("New Cases vs Stringency Italy")
color = 'tab:red'
ax1.plot(dates, new_cases_it, color=color)
ax1.set_xticks([])
plt.yticks(np.arange(np.min(new_cases_it), np.max(new_cases_it), 100.0), color=color)
ax1.set_ylabel('New Cases', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(dates, stringency_it, color=color)
ax2.set_xticks([])
plt.yticks(np.arange(np.min(stringency_it), np.max(stringency_it), 10.0), color=color)
ax2.set_ylabel('Stringency', color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
fig.set_size_inches(10.5, 26.5)

# plt.show()
plt.savefig('Untitled Folder/newVssr.jpg')
plt.clf() 

# New Cases vs New tests
fig, ax1 = plt.subplots()
plt.title("New Cases vs New test")
color = 'tab:red'
ax1.plot(dates, new_cases_it, color=color)
ax1.set_xticks([])
plt.yticks(np.arange(np.min(new_tests_it), np.max(new_tests_it), 20000.0), color=color)
ax1.set_ylabel('New Cases', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(dates, new_tests_it, color=color)
ax2.set_xticks([])
plt.yticks(np.arange(np.min(new_tests_it), np.max(new_tests_it), 20000.0), color=color)
ax2.set_ylabel('New tests', color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
# fig.set_size_inches(10.5, 26.5)

# plt.show()
plt.savefig('Untitled Folder/newVsnewt_it.jpg')
plt.clf() 

# japan
# Total Cases
plt.figure(17,figsize=(10,14))
plt.plot(dates, tot_cases_jp)
plt.xticks([])
plt.yticks(np.arange(np.min(tot_cases_jp), np.max(tot_cases_jp), 5000.0))
plt.title("Total Cases Japan")
# plt.show()
plt.savefig('Untitled Folder/tot_jp.jpg')
plt.clf() 

# New Cases
plt.figure(7)
plt.plot(dates, new_cases_jp)
plt.xticks([])
plt.yticks(np.arange(np.min(new_cases_jp), np.max(new_cases_jp), 250.0))
plt.title("Daily New Cases Japan")
# plt.show()
plt.savefig('Untitled Folder/daily_jp.jpg')
plt.clf() 

# Total Deaths
plt.figure(8)
plt.plot(dates, tot_deaths_jp)
plt.xticks([])
plt.yticks(np.arange(np.min(tot_deaths_jp), np.max(tot_deaths_jp), 250.0))
plt.title("Total Deaths Japan")
# plt.show()
plt.savefig('Untitled Folder/totd_jp.jpg')
plt.clf() 

# New Deaths
plt.figure(9)
plt.plot(dates, new_deaths_jp)
plt.xticks([])
plt.yticks(np.arange(np.min(new_deaths_jp), np.max(new_deaths_jp), 250.0))
plt.title("Daily New Deaths Japan")
# plt.show()
plt.savefig('Untitled Folder/newd_jp.jpg')
plt.clf() 

# Reproduction Rate
plt.figure(10)
fig = plt.figure(figsize =(10, 7)) 

plt.boxplot(r_jp) 
plt.title("COVID19 Reproduction Rate Japan")
# plt.show()
plt.savefig('Untitled Folder/rr_jp.jpg')
plt.clf() 

# New Tests
plt.figure(11)
plt.plot(dates, new_tests_jp)
plt.xticks([])
plt.yticks(np.arange(np.min(new_tests_jp), np.max(new_tests_jp), 5000))
plt.title("Daily New Tests Japan")
# plt.show()
plt.savefig('Untitled Folder/newt_jp.jpg')
plt.clf() 

# New Cases vs Stringency
fig, ax1 = plt.subplots()
plt.title("New Cases vs Stringency Japan")
color = 'tab:red'
ax1.plot(dates, new_cases_jp, color=color)
ax1.set_xticks([])
plt.yticks(np.arange(np.min(new_cases_jp), np.max(new_cases_jp), 100.0), color=color)
ax1.set_ylabel('New Cases', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(dates, stringency_jp, color=color)
ax2.set_xticks([])
plt.yticks(np.arange(np.min(stringency_jp), np.max(stringency_jp), 10.0), color=color)
ax2.set_ylabel('Stringency', color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()

# plt.show()
plt.savefig('Untitled Folder/newVssr_jp.jpg')
plt.clf() 


# New Cases vs New tests
fig, ax1 = plt.subplots()
plt.title("New Cases vs New tests Japan")
color = 'tab:blue'
ax1.plot(dates, new_cases_jp, color=color)
ax1.set_xticks([])
plt.yticks(np.arange(np.min(new_tests_jp), np.max(new_tests_jp), 2000.0), color=color)
ax1.set_ylabel('New Cases', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.plot(dates, new_tests_jp, color=color)
ax2.set_xticks([])
plt.yticks(np.arange(np.min(new_tests_jp), np.max(new_tests_jp), 2000.0), color=color)
ax2.set_ylabel('New tests', color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()

# plt.show()
plt.savefig('Untitled Folder/newVsnewt_jp.jpg')
plt.clf() 


# In[ ]:




