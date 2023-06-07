from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_run
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read in parameter files
total_param_file = 'Toronto_Total_integration_output_2005_2019.csv'
active_param_file = 'Toronto_Active_integration_output_2005_2019.csv'
active_df = pd.read_csv(f'{active_param_file}')
total_df = pd.read_csv(f'{total_param_file}')

# transform K_s to K_a
total_df['K_a'] = total_df['K_s']/total_df['K']
active_df['K_a'] = active_df['K_s']/active_df['K']

# find time length for total population
total_df['start_date'] = pd.to_datetime(total_df['start_date'])
total_df['start DOY'] = total_df['start_date'].dt.dayofyear
total_df['end_date'] = pd.to_datetime(total_df['end_date'])
total_df['end DOY'] = total_df['end_date'].dt.dayofyear
t_length = total_df['end DOY'].max() - total_df['start DOY'].min()

#%% load LLM functions

def compute_r_and_K(rb, rs, Kb, Ks, t):
    r = rb - rs*np.cos(2*np.pi*(t)/365)
    #K = Kb - Ks*np.cos(2*np.pi*(t)/365)
    K = Kb * (1 - Ks*np.cos(2*np.pi*(t)/365))
    return (r,K)

def logistic_model(r, K, S):
    dS = r*S*(1- S/K)
    return dS

def RK4(x0, delta_t, params, t):
    # total number of steps = 1/(step size)
    n = int((1/delta_t)*len(t))

    # create a vector for S
    S = np.zeros(int(n))
  
    # extract parameter components from input    
    rb, rs, Kb, Ks = params
    
    # assign initial condition to first element of S
    S[0] = x0
        
    for i in np.arange(0,n-1):
        # compute r and K for time step t_i
        t_i = i*delta_t
        r1, K1 = compute_r_and_K(rb, rs, Kb, Ks, t_i)
    
        # compute k1
        k1 = delta_t*logistic_model(r1, K1, S[i])
        
        # compute r and K for time step t_i + (delta_t)/2
        r23, K23 = compute_r_and_K(rb, rs, Kb, Ks, t_i + (delta_t)/2)
        
        # compute k2
        k2 = delta_t*logistic_model(r23, K23, S[i] + k1/2)
        
        # compute k3
        k3 = delta_t*logistic_model(r23, K23, S[i] + k2/2)
        
        # compute r and K for time step t_i + (delta_t)/2
        r4, K4 = compute_r_and_K(rb, rs, Kb, Ks, t_i + (delta_t))

        # compute k4
        k4 = delta_t*logistic_model(r4, K4, S[i] + k3)
        
        # new computed numerical value
        comp_value = S[i] + (k1 + 2*k2 + 2*k3 + k4)/6
        
        # if new computed value exceeds the max possible
        #   carrying capacity, set it equal to the previous value
        #   This avoids the risk of exceeding carrying capacity
        #if comp_value <= Kb + Ks:
        if comp_value <= Kb + Kb*Ks:
            S[i+1] = comp_value
        else:
               S[i+1] = S[i] 
    return S

def run_LLM(t, init_params):
    
    # step size
    delta_t = 1
    
    # logistic link model simulation
    model = RK4(init_params[4], delta_t, tuple(init_params[0:4]), t)
    return model

#%% generate Sobol samples and run model

# define inputs for each population
input_total = {
    'num_vars': 5,
    'names' : ['rb', 'rs', 'Kb', 'Ka', 'x0'],
    'bounds' : [[total_df['r'].min(), total_df['r'].max()], 
                [total_df['r_s'].min(), total_df['r_s'].max()], 
                [total_df['K'].min(), total_df['K'].max()], 
                [total_df['K_a'].min(), 0.9],
                [total_df['init_cond'].min(), total_df['init_cond'].max()]]}
# note that for total population, upper bound of K_a must be restricted to 0.9 to avoid sampling numerically unstable parameter combinations

input_active = {
    'num_vars': 5,
    'names' : ['rb', 'rs', 'Kb', 'Ka', 'x0'],
    'bounds' : [[active_df['r'].min(), active_df['r'].max()], 
                [active_df['r_s'].min(), active_df['r_s'].max()], 
                [active_df['K'].min(), active_df['K'].max()], 
                [active_df['K_a'].min(), active_df['K_a'].max()],
                [active_df['init_cond'].min(), active_df['init_cond'].max()]]}

# define time array for each population
t_total = np.arange(t_length)
t_active = np.arange(300)

# get samples
sample_size = 2**10 # sobol sampling must be a power of 2
total_samples = sobol_sample.sample(input_total, sample_size, seed=1)
active_samples = sobol_sample.sample(input_active, sample_size, seed=1)

# run LLM model for each sample
Y_total = np.array([run_LLM(t_total, params) for params in total_samples])
Y_active = np.array([run_LLM(t_active, params) for params in active_samples])

# verify that simulated samples produce numerically stable outputs
def plot_simulations():
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (20,10), dpi = 300)
    ftsz = 20
    for i in np.arange(sample_size):
        plt.sca(axs[0])
        plt.plot(t_total, Y_total[i], color = 'grey')
        plt.sca(axs[1])
        plt.plot(t_active, Y_active[i], color = 'grey')
    
    plt.sca(axs[0])
    plt.plot(t_total, np.mean(Y_total, axis = 0), color = 'black', linewidth = 3, label = 'Mean')
    plt.title('Total Population', fontsize = ftsz)
    plt.ylabel('Adult Female Mosquitoes', fontsize = ftsz)

    plt.sca(axs[1])
    plt.plot(t_active, np.mean(Y_active, axis = 0), color = 'black', linewidth = 3, label = 'Mean')
    plt.title('Active Population', fontsize = ftsz)
    plt.ylabel('Average Adult Female Mosquitoes per Trap', fontsize = ftsz)

    for i in [0,1]:
        plt.sca(axs[i])
        plt.xlabel('Time (Days)', fontsize = ftsz)
        plt.xticks(fontsize = ftsz)
        plt.yticks(fontsize = ftsz)
        plt.legend(fontsize = ftsz)
    plt.show()

plot_simulations()

#%% run Sobol sensitivity
# quantities of interest: peak magnitude (max) and peak timing (peak)

mag_total = np.max(Y_total, axis = 1)
mag_active = np.max(Y_active, axis = 1)

time_total = np.argmax(Y_total, axis = 1)
time_active = np.argmax(Y_active, axis = 1)

Si_mag_total = sobol_run.analyze(input_total, mag_total, print_to_console = True)
Si_mag_active = sobol_run.analyze(input_active, mag_active, print_to_console = True)

Si_time_total = sobol_run.analyze(input_total, time_total, print_to_console = True)
Si_time_active = sobol_run.analyze(input_active, time_active, print_to_console = True)

# generate bar plots
S1_mag_total = Si_mag_total['S1']
ST_mag_total = Si_mag_total['ST']
S1_mag_active = Si_mag_active['S1']
ST_mag_active = Si_mag_active['ST']

S1_time_total = Si_time_total['S1']
ST_time_total = Si_time_total['ST']
S1_time_active = Si_time_active['S1']
ST_time_active = Si_time_active['ST']

barWidth = 0.45
bar_total = np.arange(len(S1_mag_total))
bar_active = [x + barWidth for x in bar_total]
 
# Make the plots
ftsz = 15
fig, axs = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize =(16, 8), dpi = 300)
plt.sca(axs[0])
plt.bar(bar_total, S1_mag_total, width = barWidth,
        edgecolor ='grey', label = 'Total')
plt.bar(bar_active, S1_mag_active, width = barWidth,
        edgecolor ='grey', label = 'Active')
plt.title('First Order Effects', fontsize = ftsz)
plt.xticks([r + barWidth/2 for r in range(len(S1_mag_total))],
        [r'$r_b$', r'$r_s$', r'$K_b$', r'$K_a$', r'$P(0)$'], fontsize = ftsz)
plt.ylabel('Sobol Sensitivity Index for Peak Magnitude', fontsize = ftsz)
plt.yticks(fontsize = ftsz)
plt.legend(fontsize = ftsz)
plt.ylim([0, 0.6])

plt.sca(axs[1])
plt.bar(bar_total, ST_mag_total - S1_mag_total, width = barWidth,
        edgecolor ='grey', label = 'Total')
plt.bar(bar_active, ST_mag_active - S1_mag_active, width = barWidth,
        edgecolor ='grey', label = 'Active')

plt.title('Interaction Order Effects', fontsize = ftsz)
plt.xticks([r + barWidth/2 for r in range(len(S1_mag_total))],
        [r'$r_b$', r'$r_s$', r'$K_b$', r'$K_a$', r'$P(0)$'], fontsize = ftsz)
plt.yticks(fontsize = ftsz)
plt.legend(fontsize = ftsz)
plt.show()

fig, axs = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize =(16, 8), dpi = 300)
plt.sca(axs[0])
plt.bar(bar_total, S1_time_total, width = barWidth,
        edgecolor ='grey', label = 'Total')
plt.bar(bar_active, S1_time_active, width = barWidth,
        edgecolor ='grey', label = 'Active')

plt.title('First Order Effects', fontsize = ftsz)
plt.xticks([r + barWidth/2 for r in range(len(S1_mag_total))],
        [r'$r_b$', r'$r_s$', r'$K_b$', r'$K_a$', r'$P(0)$'], fontsize = ftsz)
plt.ylabel('Sobol Sensitivity Index for Peak Timing', fontsize = ftsz)
plt.yticks(fontsize = ftsz)
plt.legend(fontsize = ftsz)
plt.ylim([0, 0.8])

plt.sca(axs[1])
plt.bar(bar_total, ST_time_total - S1_time_total, width = barWidth,
        edgecolor ='grey', label = 'Total')
plt.bar(bar_active, ST_time_active - S1_time_active, width = barWidth,
        edgecolor ='grey', label = 'Active')

plt.title('Interaction Order Effects', fontsize = ftsz)
plt.xticks([r + barWidth/2 for r in range(len(S1_mag_total))],
        [r'$r_b$', r'$r_s$', r'$K_b$', r'$K_a$', r'$P(0)$'], fontsize = ftsz)
plt.yticks(fontsize = ftsz)
plt.legend(fontsize = ftsz)
plt.show()
