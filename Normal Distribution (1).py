#!/usr/bin/env python
# coding: utf-8

# In[13]:


# from pydataset import data
from matplotlib import pyplot as plt
from scipy.stats import norm 
import numpy as np
from csv import reader

get_ipython().run_line_magic('matplotlib', 'notebook')

# plt.hist(data('cancer')['age'])
plt.show()

with open("ind.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
data_arm = np.array(temp)
dates = data_arm[:, 0]
new_cases_arm = data_arm[:, 26]
new_cases_arm[new_cases_arm == ''] = '0.0'
new_cases_arm = new_cases_arm.astype(np.float)

#sample standard deriation
std = np.std(new_cases_arm, ddof = 1)
mean = np.mean(new_cases_arm)

#plotting
domain = np.linspace(np.min(new_cases_arm),np.max(new_cases_arm))
                     
plt.plot(domain,norm.pdf(domain,mean,std),label = '$\mathcal{N}$ ' + f'$( \mu \\approx {round(mean)} , \sigma \\approx {round(std)} )$')
                     
plt.hist(new_cases_arm,edgecolor = 'black' ,alpha = .5,density=True)
        
                     
plt.title("India plot of covid test")
plt.xlabel("Test for covid")
plt.ylabel("Density")
plt.legend()
# plt.show()
plt.savefig('Untitled Folder/t_in.jpg')
plt.clf() 

with open("germany.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
data_in = np.array(temp)
dates = data_in[:, 0]
new_cases_arm = data_in[:, 26]
new_cases_arm[new_cases_arm == ''] = '0.0'
new_cases_arm = new_cases_arm.astype(np.float)

#sample standard deriation
std = np.std(new_cases_arm, ddof = 1)
mean = np.mean(new_cases_arm)

#plotting
domain = np.linspace(np.min(new_cases_arm),np.max(new_cases_arm))
                     
plt.plot(domain,norm.pdf(domain,mean,std),label = '$\mathcal{N}$ ' + f'$( \mu \\approx {round(mean)} , \sigma \\approx {round(std)} )$')
                     
plt.hist(new_cases_arm,edgecolor = 'black' ,alpha = .5,density=True)
        
                     
plt.title("Germany plot of covid test")
plt.xlabel("Test for covid")
plt.ylabel("Density")
plt.legend()
# plt.show()
plt.savefig('Untitled Folder/t_ge.jpg')
plt.clf() 


# In[25]:


# from pydataset import data
from matplotlib import pyplot as plt
from scipy.stats import norm 
import numpy as np
from csv import reader

get_ipython().run_line_magic('matplotlib', 'notebook')

# plt.hist(data('cancer')['age'])
# plt.show()

with open("china.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
data_ge = np.array(temp)
dates = data_ge[:, 0]
new_cases_ge = data_ge[:, 11]
new_cases_ge[new_cases_ge == ''] = '0.0'
new_cases_ge = new_cases_ge.astype(np.float)

with open("japan.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
data_it = np.array(temp)
dates = data_it[:, 0]
new_cases_it = data_arm[:, 11]
new_cases_it[new_cases_it == ''] = '0.0'
new_cases_it = new_cases_it.astype(np.float)

with open("ind.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
data_aus = np.array(temp)
dates = data_aus[:, 0]
new_cases_aus = data_aus[:, 11]
new_cases_aus[new_cases_aus == ''] = '0.0'
new_cases_aus = new_cases_aus.astype(np.float)

new_cases_arm = new_cases_aus + new_cases_it + new_cases_ge
#sample standard deriation
std = np.std(new_cases_arm, ddof = 1)
mean = np.mean(new_cases_arm)

#plotting
domain = np.linspace(np.min(new_cases_arm),np.max(new_cases_arm))
                     
plt.plot(domain,norm.pdf(domain,mean,std),label = '$\mathcal{N}$ ' + f'$( \mu \\approx {round(mean)} , \sigma \\approx {round(std)} )$')
                     
plt.hist(new_cases_arm,edgecolor = 'black' ,alpha = .5,density=True)
        
                     
plt.title("Normal plt")
plt.xlabel("New Death per milion inside asia india, Japan and china")
plt.ylabel("Density")
plt.legend()
# plt.show()
plt.savefig('Untitled Folder/Avg_asia.jpg')
plt.clf() 


# In[16]:


# from pydataset import data
from matplotlib import pyplot as plt
from scipy.stats import norm 
import numpy as np
from csv import reader

get_ipython().run_line_magic('matplotlib', 'notebook')

# plt.hist(data('cancer')['age'])
plt.show()

with open("china.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
data_arm = np.array(temp)
dates = data_arm[:, 0]
new_cases_arm = data_arm[:, 13]
new_cases_arm[new_cases_arm == ''] = '0.0'
new_cases_arm = new_cases_arm.astype(np.float)

#sample standard deriation
std = np.std(new_cases_arm, ddof = 1)
mean = np.mean(new_cases_arm)
# print(std,mean)
#plotting
domain = np.linspace(np.min(new_cases_arm),np.max(new_cases_arm))
                     
plt.plot(domain,norm.pdf(domain,mean,std),label = '$\mathcal{N}$ ' + f'$( \mu \\approx {round(mean)} , \sigma \\approx {round(std)} )$')
                     
plt.hist(new_cases_arm,edgecolor = 'black' ,alpha = .5,density=True)
        
                     
plt.title("Reproduction rate in China")
plt.xlabel("Reproduction Rate")
plt.ylabel("Density")
plt.legend()
# plt.show()
plt.savefig('Untitled Folder/r_ch.jpg')
plt.clf() 

with open("ind.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
data_in = np.array(temp)
dates = data_in[:, 0]
new_cases_arm = data_in[:, 13]
new_cases_arm[new_cases_arm == ''] = '0.0'
new_cases_arm = new_cases_arm.astype(np.float)

#sample standard deriation
std = np.std(new_cases_arm, ddof = 1)
mean = np.mean(new_cases_arm)

#plotting
domain = np.linspace(np.min(new_cases_arm),np.max(new_cases_arm))
                     
plt.plot(domain,norm.pdf(domain,mean,std),label = '$\mathcal{N}$ ' + f'$( \mu \\approx {round(mean)} , \sigma \\approx {round(std)} )$')
                     
plt.hist(new_cases_arm,edgecolor = 'black' ,alpha = .5,density=True)
        
                     
plt.title("Reproduction rate in India")
plt.xlabel("Reproduction Rate")
plt.ylabel("Density")
plt.legend()
# plt.show()
plt.savefig('Untitled Folder/r_in.jpg')
plt.clf() 


# In[6]:


# from pydataset import data
from matplotlib import pyplot as plt
from scipy.stats import norm 
import numpy as np
from csv import reader

get_ipython().run_line_magic('matplotlib', 'notebook')

# plt.hist(data('cancer')['age'])
plt.show()

with open("ind.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
data_arm = np.array(temp)
dates = data_arm[:, 0]
new_cases_arm = data_arm[:, 13]
new_cases_arm[new_cases_arm == ''] = '0.0'
new_cases_arm = new_cases_arm.astype(np.float)

#sample standard deriation
std = np.std(new_cases_arm, ddof = 1)
mean = np.mean(new_cases_arm)
# print(std,mean)
#plotting
domain = np.linspace(np.min(new_cases_arm),np.max(new_cases_arm))
                     
plt.plot(domain,norm.pdf(domain,mean,std),label = '$\mathcal{N}$ ' + f'$( \mu \\approx {round(mean)} , \sigma \\approx {round(std)} )$')
                     
plt.hist(new_cases_arm,edgecolor = 'black' ,alpha = .5,density=True)
        
                     
plt.title("Normal plt")
plt.xlabel("Value")
plt.ylabel("Reproduction Rate")
plt.legend()
# plt.show()
plt.savefig('Untitled Folder/r_in.jpg')
plt.clf() 

with open("china.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
data_in = np.array(temp)
dates = data_in[:, 0]
new_cases_arm = data_in[:, 13]
new_cases_arm[new_cases_arm == ''] = '0.0'
new_cases_arm = new_cases_arm.astype(np.float)

#sample standard deriation
std = np.std(new_cases_arm, ddof = 1)
mean = np.mean(new_cases_arm)

#plotting
domain = np.linspace(np.min(new_cases_arm),np.max(new_cases_arm))
                     
plt.plot(domain,norm.pdf(domain,mean,std),label = '$\mathcal{N}$ ' + f'$( \mu \\approx {round(mean)} , \sigma \\approx {round(std)} )$')
                     
plt.hist(new_cases_arm,edgecolor = 'black' ,alpha = .5,density=True)
        
                     
plt.title("Normal plt")
plt.xlabel("Value")
plt.ylabel("Reproduction Rate")
plt.legend()
# plt.show()
plt.savefig('Untitled Folder/r_ch.jpg')
plt.clf() 


# In[14]:


# from pydataset import data
from matplotlib import pyplot as plt
from scipy.stats import norm 
import numpy as np
from csv import reader

get_ipython().run_line_magic('matplotlib', 'notebook')

# plt.hist(data('cancer')['age'])
plt.show()

with open("travel.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
data_arm = np.array(temp)
dates = data_arm[:, 0]
tot_travel_spain = data_arm[:, 4]
tot_travel_spain = np.array(tot_travel_spain).tolist()
for i in range(len(tot_travel_spain)):
    s = tot_travel_spain[i].split(",")
    te = "".join(s)
    tot_travel_spain[i] = float(te)
new_cases_arm = np.array(tot_travel_spain)

#sample standard deriation
std = np.std(new_cases_arm, ddof = 1)
mean = np.mean(new_cases_arm)
# print(std,mean)
#plotting
domain = np.linspace(np.min(new_cases_arm),np.max(new_cases_arm))
                     
plt.plot(domain,norm.pdf(domain,mean,std),label = '$\mathcal{N}$ ' + f'$( \mu \\approx {round(mean)} , \sigma \\approx {round(std)} )$')
                     
plt.hist(new_cases_arm,edgecolor = 'black' ,alpha = .5,density=True)
        
                     
plt.title("Treval in Germany")
plt.xlabel("Tourist")
plt.ylabel("Density")
plt.legend()
# plt.show()
plt.savefig('Untitled Folder/trv_ge.jpg')
plt.clf() 


tot_travel_spain = data_arm[:, 7]
tot_travel_spain = np.array(tot_travel_spain).tolist()
for i in range(len(tot_travel_spain)):
    s = tot_travel_spain[i].split(",")
    te = "".join(s)
    tot_travel_spain[i] = float(te)
new_cases_arm = np.array(tot_travel_spain)
#sample standard deriation
std = np.std(new_cases_arm, ddof = 1)
mean = np.mean(new_cases_arm)

#plotting
domain = np.linspace(np.min(new_cases_arm),np.max(new_cases_arm))
                     
plt.plot(domain,norm.pdf(domain,mean,std),label = '$\mathcal{N}$ ' + f'$( \mu \\approx {round(mean)} , \sigma \\approx {round(std)} )$')
                     
plt.hist(new_cases_arm,edgecolor = 'black' ,alpha = .5,density=True)
        
                     
plt.title("treveller Number in Italy")
plt.xlabel("Tourist Number")
plt.ylabel("Density")
plt.legend()
# plt.show()
plt.savefig('Untitled Folder/trv_it.jpg')
plt.clf() 


# In[ ]:




