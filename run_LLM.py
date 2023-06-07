import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import os
from datetime import datetime

# REQUIRED USER INPUTS:
# 1.) name of data file
# 2.) directory path where CSV should be stored
# 3.) choose location
# 4.) choose total or active mosquito population
# 5.) choose time range of when fitting season starts
# 6.) choose time range of when fitting season ends (total population only)
# 7.) choose years for fitting/simulation

# 1.) file name
data_file = 'Mosquito PBM\\Toronto_2004_to_2020_NoTempCutoffEVMort.csv'

# 2.) path for output
output_path = 'logistic-link-model'

# 3.) choose location: 'Toronto'
loc = 'Toronto'

# 4.) choose population type: 'Total' or 'Active'
pop = 'Total'

# 5.) choose range of start days for mosquito fitting season (MFS): 
    # in the format 'MM-DD' 
start1 = '05-01'    # initial start day
start2 = '06-01'   # final start day

# 6.) [Total population only] choose range of end days for MFS:
end1 = '10-01'     # initial end day
end2 = '11-01'     # final end day

# 7.) testing out years as an input as well
start_year = '2005'
end_year = '2020'

################
###Prep Data####
################

df = pd.read_csv(data_file)

location_group = df.sort_values(by='date')
location_group['date'] = pd.to_datetime(location_group['date'])

# find start date, end date, and date range of time series
ts_start = location_group['date'].min()
# convert to string
ts_start_str = datetime.strftime(ts_start, '%m/%d/%Y')
# find end date
ts_end = location_group['date'].max()
# convert end date to string
ts_end_str = datetime.strftime(ts_end, '%m/%d/%Y')
ts_range = pd.date_range(start=ts_start, end=ts_end)

if pop == 'Total':
        PBM_type = 'TotMosq'
        pop_str = 'Total_Pop'
        
date_mosq = location_group[['date', PBM_type]]
date_num_trans = pd.DataFrame({'day_index': np.arange(0, len(ts_range)), 'date':ts_range})
date_mosq_final = date_mosq.merge(date_num_trans, how = 'left', left_on = 'date', right_on = 'date')

###########################
##### compute_r_and_K #####
###########################
# summary: compute growth rate (r) and carrying capacity (K) values
# input:
    # rb: baseline growth rate (value)
    # rs: amplitude growth rate(value)
    # Kb: baseline carrying capacity (value)
    # Ks: amplitude carrying capacity (value)
    # t: time frame (value or array)
# output:
    # r: growth rate at t (value or array)
    # K: carrying capacity at t (value or array)
def compute_r_and_K(rb, rs, Kb, Ks, t):
    r = rb - rs*np.cos(2*np.pi*(t)/365)
    K = Kb - Ks*np.cos(2*np.pi*(t)/365)
    return (r,K)

##########################
##### logistic_model #####
##########################
# summary: the differential equation (dS/dt) for mosquito population
# input:
    # r: growth rate (value or array)
    # K: carrying capacity (value or array)
    # S: mosquito population (value or array)
# output: 
    # dS: rate of change of mosquito population (value or array)
def logistic_model(r, K, S):
    dS = r*S*(1- S/K)
    return dS
    
##########################
########## RK4 ###########
##########################
# summary: 4th order Runge Kutta solver for non-autonomous logistic model
# input:
    # x0: initial condition (value)
    # delta_t: time step (value)
    # params: np.array([rb, rs, Kb, Ks])
    # t: array of days (np.array)
# output:
    # S: mosquito population (array)
    
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
        if comp_value <= Kb + Ks:
            S[i+1] = comp_value
        else:
               S[i+1] = S[i] 
    return S

##########################
########## cost ##########
##########################
# summary: cost function for parameter optimization
# input:
    # init_params: initial paramter values (rb, rs, Kb, Ks)
    # mosq: mosquito PBM time series (array)
    # mosq_dates: initial and final time series points for fitting
    # ts_range: initial date of the mosquito PBM time series (date)
# output:
    # res.flatten(): 1D array of error values between model and mosquito PBM
    
def cost(init_params, mosq, mosq_dates, ts_range):
    init_params = tuple(init_params)
    mosq_start, mosq_end = mosq_dates
    
    # initial condition
    x0 = list(mosq)[0]
    
    # step size
    delta_t = 1
    
    # time frame for fitting
    t = np.arange(mosq_start, mosq_end)
    
    # logistic link model simulation
    model = RK4(x0, delta_t, init_params, t)
    
    # error between mosquito PBM time series and model
    res = model[::int(1/delta_t)] - mosq
    
    return res.flatten()

##########################
######## fit_data ########
##########################
# summary: determines when to partition time series intervals
# input:
    # init_params: initial parameters (rb, rs, Kb, Ks)
    # mosq: subset of mosquito time series (array)
    # mosq_start: initial time ponit of MFS (value)
    # mosq_end: final time point of MFS (value)
    # ts_range: initial date of the mosquito PBM time series (date)
# output:
    # resid: residuals-- difference between model fit and MPBM time series (array)
    # fitted_params: optimal rb, rs, Kb, Ks values (array)
def fit_data(init_params, mosq, mosq_start, mosq_end, ts_range):
    mosq_dates = [mosq_start, mosq_end]
    
    # bound constraints for parameters
    if pop == 'Total':
        bd = ([-0.2, -0.35, 1, 0],[0.2, 0, 1e5, 1e5]) 
    else:
        bd = ([-0.2, -0.35, 1, 0],[0.2, 0, 1e3, 1e3]) 
    
    # optimization run
    optim = least_squares(cost, init_params, bounds = bd, method = 'trf', args=(mosq, mosq_dates, ts_range))
    
    fitted_params = optim.x
    resid = optim.fun

    return (resid, fitted_params)


#####################
##### all_func ######
#####################
#summary: runs the fitting process and outputs RMSE and fitted param values
#input: times, a vector of length 3 with the format [start, end, year]
#output: [RMSE, rb, rs, Kb, Ks] for each fit

def all_func(times):
    start = times[0]
    end = times[1]
    year = str(times[2])

    start_full = datetime.strptime(year + '-' + start, '%Y-%m-%d')
    end_full = datetime.strptime(year + '-' + end, '%Y-%m-%d')
    
    sd = date_mosq_final[date_mosq_final['date'] == start_full]['day_index'].reset_index()['day_index'][0]
    ed = date_mosq_final[date_mosq_final['date'] == end_full]['day_index'].reset_index()['day_index'][0]

    # find subset of mosq PBM output
    ts_sub = np.array(date_mosq_final[(date_mosq_final['date'] >= start_full) & (date_mosq_final['date'] < end_full)][PBM_type])

    # initialize parameter values
    r_b_init = 0
    r_s_init = -0.07
    K_b_init = np.max(ts_sub)
    
    if pop == 'Total':
        K_s_init = 100
    else:
        K_s_init = 1

    ts_params = np.array([r_b_init, r_s_init, K_b_init, K_s_init])
        
    # run data fitting
    mosq_fit = fit_data(ts_params, ts_sub, sd, ed, ts_range)
    
    #pull fit params
    rb, rs, Kb, Ks = mosq_fit[1]

    RMSE = (np.sum(mosq_fit[0]**2)/len(mosq_fit[0]))**(1/2)
    return [RMSE, rb, rs, Kb, Ks]

#Get array of all possible start dates
start1_date = datetime.strptime(start1, '%m-%d')
start2_date = datetime.strptime(start2, '%m-%d')
start_range = pd.date_range(start=start1_date, end=start2_date)
start_range_format = np.array([k.strftime('%m-%d') for k in start_range])

#Get array of all possible end dates
end1_date = datetime.strptime(end1, '%m-%d')
end2_date = datetime.strptime(end2, '%m-%d')
end_range = pd.date_range(start=end1_date, end=end2_date)
end_range_format = np.array([k.strftime('%m-%d') for k in end_range])

#Get array of all possible years
years_format = np.arange(int(start_year), int(end_year) + 1, 1)

#create dataset of all possible start, end, and year combinations where each row contains a start date, end date, and year in that order
all_vals = np.array(np.meshgrid(*[start_range_format, end_range_format, years_format])).reshape(3, len(start_range_format) * len(end_range_format) * len(years_format)).T

#fit the model along the rows of all_vals
fit_out = np.apply_along_axis(all_func, 1, all_vals)

#turn fit_out into a dataframe
df = pd.DataFrame(fit_out, columns = ['RMSE', 'r', 'r_s', 'K', 'K_s'])

df.to_csv(f'{output_path}\\{loc}\\All_fits_{pop}_{loc}_{start_year}_{end_year}.csv')

df2 = pd.DataFrame(all_vals, columns = ['start_date', 'end_date', 'year'])
df_full = pd.concat([df2, df], axis = 1)
df_agg = df_full.groupby('year').agg({'RMSE':'min'}).reset_index()
df_best = df_full[df_full['RMSE'].isin(df_agg['RMSE'])].sort_values('year').reset_index(drop = True)

df_best.to_csv(f'{output_path}\\{loc}\\Best_fits_{pop}_{start_year}_{end_year}.csv')
