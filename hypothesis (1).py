#!/usr/bin/env python
# coding: utf-8

# In[49]:


#for armania
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import sqrt, abs, round
from scipy.stats import norm
from matplotlib import dates as mpl_dates
from csv import reader

print("Our null hypothsis is second peak in armenia is equal to first peak")
def TwoSampZ(X1, X2, sigma1, sigma2, N1, N2):
    ovr_sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
    z = (X1 - X2)/ovr_sigma
    pval = 2*(1 - norm.cdf(abs(z)))
    return pval,z

with open("armenia.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
    
data_arm = np.array(temp)


# Dates Universal
dates = data_arm[:, 0]

new_cases_arm = data_arm[:, 3]
new_cases_arm[new_cases_arm == ''] = '0.0'
new_cases_arm = new_cases_arm.astype(np.float)

n = new_cases_arm.size
new_cases_first = new_cases_arm[50:225]
new_cases_second = new_cases_arm[255:]

first = np.random.choice(new_cases_first,35)
second = np.random.choice(new_cases_second,35)

m1, m2 = first.mean(), second.mean()
sd1, sd2 = first.std(), second.std()
n1, n2 = first.shape[0], second.shape[0]
p,z = TwoSampZ(m1, m2, sd1, sd2, n1, n2)

z_score = np.round(z,8)
p_val = np.round(p,6)

print(p_val)
if (p_val<0.05):
    Hypothesis_Status = 'Reject Null Hypothesis : Significant mean peak was larger then privious one'
else:
    Hypothesis_Status = 'Do not reject Null Hypothesis : Not Significant'
    
print (Hypothesis_Status)
# print(n)


# In[23]:


#for australia

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import sqrt, abs, round
from scipy.stats import norm
from matplotlib import dates as mpl_dates
from csv import reader

print("Our null hypothsis is second peak in armenia is equal to first peak")
def TwoSampZ(X1, X2, sigma1, sigma2, N1, N2):
    ovr_sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
    z = (X1 - X2)/ovr_sigma
    pval = 2*(1 - norm.cdf(abs(z)))
    return pval,z

with open("aus.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
    
data_arm = np.array(temp)


# Dates Universal
dates = data_arm[:, 0]

new_cases_arm = data_arm[:, 3]
new_cases_arm[new_cases_arm == ''] = '0.0'
new_cases_arm = new_cases_arm.astype(np.float)

n = new_cases_arm.size
new_cases_first = new_cases_arm[40:140]
new_cases_second = new_cases_arm[140:]

first = np.random.choice(new_cases_first,60)
second = np.random.choice(new_cases_second,60)

m1, m2 = first.mean(), second.mean()
sd1, sd2 = first.std(), second.std()
n1, n2 = first.shape[0], second.shape[0]
p,z = TwoSampZ(m1, m2, sd1, sd2, n1, n2)

z_score = np.round(z,8)
p_val = np.round(p,6)

if (p_val<0.05):
    Hypothesis_Status = 'Reject Null Hypothesis : Significant mean peak was larger then privious one'
else:
    Hypothesis_Status = 'Do not reject Null Hypothesis : Not Significant'
    
print (Hypothesis_Status)
# print(n)


# In[ ]:





# In[45]:


#for india vs anstralia

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import sqrt, abs, round
from scipy.stats import norm
from matplotlib import dates as mpl_dates
from csv import reader

# print("Our null hypothsis is second peak in armenia is equal to first peak")
def TwoSampZ(X1, X2, sigma1, sigma2, N1, N2):
    ovr_sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
    z = (X1 - X2)/ovr_sigma
    pval = 2*(1 - norm.cdf(abs(z)))
    return pval,z

with open("china.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
    
data_arm = np.array(temp)


# Dates Universal
dates = data_arm[:, 0]

new_cases_arm = data_arm[:, 31]
new_cases_arm[new_cases_arm == ''] = '0.0'
new_cases_arm = new_cases_arm.astype(np.float)

with open("ind.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
    
data_in = np.array(temp)

new_cases_in = data_in[:, 31]
new_cases_in[new_cases_in == ''] = '0.0'
new_cases_in = new_cases_in.astype(np.float)

# n = new_cases_arm.size

first = np.random.choice(new_cases_arm,30)
second = np.random.choice(new_cases_in,30)

m1, m2 = first.mean(), second.mean()
sd1, sd2 = new_cases_arm.std(), new_cases_in.std()
n1, n2 = first.shape[0], second.shape[0]
p,z = TwoSampZ(m1, m2, sd1, sd2, n1, n2)

# print(m1, m2, sd1, sd2, n1, n2)
z_score = np.round(z,8)
p_val = np.round(p,6)

print("p vlaue is :" + str(p_val))
if (p_val<0.05):
    Hypothesis_Status = 'Reject Null Hypothesis : Significant mean china has hiegher strangency then india'
else:
    Hypothesis_Status = 'Do not reject Null Hypothesis : Not Significant'
    
print (Hypothesis_Status)
# print(n)


# In[ ]:





# In[46]:


#for india vs germany

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import sqrt, abs, round
from scipy.stats import norm
from matplotlib import dates as mpl_dates
from csv import reader

print("Our null hypothsis is second peak in armenia is equal to first peak")
def TwoSampZ(X1, X2, sigma1, sigma2, N1, N2):
    ovr_sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
    z = (X2 - X1)/ovr_sigma
    pval = 2*(1 - norm.cdf(abs(z)))
    return pval,z

with open("germany.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
    
data_arm = np.array(temp)


# Dates Universal
dates = data_arm[:, 0]

new_cases_arm = data_arm[:, 26]
new_cases_arm[new_cases_arm == ''] = '0.0'
new_cases_arm = new_cases_arm.astype(np.float)

with open("ind.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
    
data_in = np.array(temp)

new_cases_in = data_in[:, 26]
new_cases_in[new_cases_in == ''] = '0.0'
new_cases_in = new_cases_in.astype(np.float)

# n = new_cases_arm.size

first = np.random.choice(new_cases_arm,60)
second = np.random.choice(new_cases_in,60)

m1, m2 = first.mean(), second.mean()
sd1, sd2 = new_cases_arm.std(), new_cases_in.std()
n1, n2 = first.shape[0], second.shape[0]
p,z = TwoSampZ(m1, m2, sd1, sd2, n1, n2)

print(m1, m2, sd1, sd2, n1, n2)
z_score = np.round(z,8)
p_val = np.round(p,6)

print(p_val)
if (p_val<0.05):
    Hypothesis_Status = 'Reject Null Hypothesis : Significant mean india took more tests on daily basis then germany '
else:
    Hypothesis_Status = 'Do not reject Null Hypothesis : Not Significant'
    
print (Hypothesis_Status)
# print(n)


# In[47]:


#for asia vs non-asia

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import sqrt, abs, round
from scipy.stats import norm
from matplotlib import dates as mpl_dates
from csv import reader

print("Our null hypothsis is second peak in armenia is equal to first peak")
def TwoSampZ(X1, X2, sigma1, sigma2, N1, N2):
    ovr_sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
    z = (X2 - X1)/ovr_sigma
    pval = 2*(1 - norm.cdf(abs(z)))
    return pval,z

with open("germany.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
    
data_ge = np.array(temp)
new_cases_ge = data_ge[:, 11]
new_cases_ge[new_cases_ge == ''] = '0.0'
new_cases_ge = new_cases_ge.astype(np.float)

with open("italy.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
    
data_it = np.array(temp)
new_cases_it = data_it[:, 11]
new_cases_it[new_cases_it == ''] = '0.0'
new_cases_it = new_cases_it.astype(np.float)

with open("aus.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
    
data_aus = np.array(temp)
new_cases_aus = data_aus[:, 11]
new_cases_aus[new_cases_aus == ''] = '0.0'
new_cases_aus = new_cases_aus.astype(np.float)

new_cases_arm = new_cases_aus + new_cases_ge + new_cases_it

with open("japan.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
    
data_japan = np.array(temp)

new_cases_japan = data_japan[:, 11]
new_cases_japan[new_cases_japan == ''] = '0.0'
new_cases_japan = new_cases_japan.astype(np.float)

with open("china.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
    
data_ch = np.array(temp)

new_cases_ch = data_ch[:, 11]
new_cases_ch[new_cases_ch == ''] = '0.0'
new_cases_ch = new_cases_ch.astype(np.float)

with open("ind.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
    
data_in = np.array(temp)

new_cases_in = data_in[:, 11]
new_cases_in[new_cases_in == ''] = '0.0'
new_cases_in = new_cases_in.astype(np.float)


new_cases_asia = new_cases_in + new_cases_ch + new_cases_japan
# n = new_cases_arm.size

first = np.random.choice(new_cases_arm,60)
second = np.random.choice(new_cases_asia,60)

m1, m2 = first.mean(), second.mean()
sd1, sd2 = new_cases_arm.std(), new_cases_in.std()
n1, n2 = first.shape[0], second.shape[0]
p,z = TwoSampZ(m1, m2, sd1, sd2, n1, n2)

print(m1, m2, sd1, sd2, n1, n2)
z_score = np.round(z,8)
p_val = np.round(p,6)

print(p_val)
if (p_val<0.05):
    Hypothesis_Status = 'Reject Null Hypothesis : Significant mean asia has more death per milion '
else:
    Hypothesis_Status = 'Do not reject Null Hypothesis : Not Significant'
    
print (Hypothesis_Status)
# print(n)


# In[52]:


#for japan vs italy

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import sqrt, abs, round
from scipy.stats import norm
from matplotlib import dates as mpl_dates
from csv import reader

print("Our null hypothsis is japan has more reproduction rate then italy H1 > H0")
def TwoSampZ(X1, X2, sigma1, sigma2, N1, N2):
    ovr_sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
    z = (X1 - X2)/ovr_sigma
    pval = 2*(1 - norm.cdf(abs(z)))
    return pval,z

with open("japan.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
    
data_arm = np.array(temp)


# Dates Universal
dates = data_arm[:, 0]

new_cases_arm = data_arm[:, 13]
new_cases_arm[new_cases_arm == ''] = '0.0'
new_cases_arm = new_cases_arm.astype(np.float)

with open("italy.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
    
data_in = np.array(temp)

new_cases_in = data_in[:, 13]
new_cases_in[new_cases_in == ''] = '0.0'
new_cases_in = new_cases_in.astype(np.float)

# n = new_cases_arm.size

#taken 50 sample size
first = np.random.choice(new_cases_arm,50)
second = np.random.choice(new_cases_in,50)

m1, m2 = first.mean(), second.mean()
sd1, sd2 = new_cases_arm.std(), new_cases_in.std()
n1, n2 = first.shape[0], second.shape[0]
p,z = TwoSampZ(m1, m2, sd1, sd2, n1, n2)

print(m1, m2, sd1, sd2, n1, n2)
z_score = np.round(z,8)
p_val = np.round(p,6)

print(p_val)
if (p_val<0.05):
    Hypothesis_Status = 'Reject Null Hypothesis : Significant mean china has hiegher strangency then india'
else:
    Hypothesis_Status = 'Do not reject Null Hypothesis : Not Significant both are same'
    
print (Hypothesis_Status)
# print(n)


# In[ ]:





# In[50]:


#for india vs germany

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import sqrt, abs, round
from scipy.stats import norm
from matplotlib import dates as mpl_dates
from csv import reader

print("Our null hypothsis is second peak in armenia is equal to first peak")
def TwoSampZ(X1, X2, sigma1, sigma2, N1, N2):
    ovr_sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
    z = (X2 - X1)/ovr_sigma
    pval = 2*(1 - norm.cdf(abs(z)))
    return pval,z

with open("travel.csv", "r") as input_file:
    csv_reader = reader(input_file)
    next(input_file)
    temp = list(csv_reader)
    
data_arm = np.array(temp)


# Dates Universal
tot_travel_ge = data_arm[:, 4]
tot_travel_ge = np.array(tot_travel_ge).tolist()
for i in range(len(tot_travel_ge)):
    s = tot_travel_ge[i].split(",")
    te = "".join(s)
    tot_travel_ge[i] = float(te)
tot_travel_ge = np.array(tot_travel_ge)

tot_travel_it = data_arm[:, 7]
tot_travel_it = np.array(tot_travel_it).tolist()
for i in range(len(tot_travel_ge)):
    s = tot_travel_it[i].split(",")
    te = "".join(s)
    tot_travel_it[i] = float(te)
tot_travel_it = np.array(tot_travel_it)

# n = new_cases_arm.size

first = np.random.choice(tot_travel_it,8)
second = np.random.choice(tot_travel_ge,8)

m1, m2 = first.mean(), second.mean()
sd1, sd2 = tot_travel_it.std(), tot_travel_ge.std()
n1, n2 = first.shape[0], second.shape[0]
p,z = TwoSampZ(m1, m2, sd1, sd2, n1, n2)

# print(m1, m2, sd1, sd2, n1, n2)
z_score = np.round(z,8)
p_val = np.round(p,6)

print(p_val)
if (p_val<0.05):
    Hypothesis_Status = 'Reject Null Hypothesis : Significant mean gemany had more travel then italy '
else:
    Hypothesis_Status = 'Do not reject Null Hypothesis : Not Significant both had same'
    
print (Hypothesis_Status)
# print(n)


# In[ ]:




