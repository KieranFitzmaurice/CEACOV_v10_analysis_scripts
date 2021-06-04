#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from scipy.optimize import linprog


# In[2]:


def two_dose_A_or_one_dose_B(subpop,uptake,doses_A,doses_B):
    
    """
    This function allocates two doses of vaccine A or one dose of vaccine B
    """
    
    
    
    max_uptake = subpop*uptake
    
    c = np.array([-1,-1])

    A_ub = np.array([[2,0],
                     [0,1],
                     [1,1]])

    b_ub = np.array([doses_A,doses_B,max_uptake])

    bounds = ((0,None),(0,None))

    # Try to achieve maximum coverage while randomly allocating vaccines
    if doses_B > 0 and doses_A > 0:
        
        A_eq = np.array([[1,-0.5*doses_A/doses_B]])
        b_eq = np.array([0])
        
        sol_1 = linprog(c,
                        A_ub=A_ub,
                        b_ub=b_ub,
                        A_eq=A_eq,
                        b_eq=b_eq,
                        bounds=bounds,
                        method='simplex')

        x2A_1 = sol_1.x[0]
        x1B_1 = sol_1.x[1]

        # Update vaccine supply, remaining eligibility, and inequality constraints
        doses_A = doses_A - 2*x2A_1
        doses_B = doses_B - x1B_1
        max_uptake = max_uptake - (x2A_1 + x1B_1)
        b_ub = np.array([doses_A,doses_B,max_uptake])
    else:
        x2A_1 = 0
        x1B_1 = 0

    # See if it's possible to achieve any higher coverage while 
    # allocating vaccines in a non-random fashion

    sol_2 = linprog(c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    bounds=bounds,
                    method='simplex')

    x2A_2 = sol_2.x[0]
    x1B_2 = sol_2.x[1]

    # Update vaccine supply
    doses_A = doses_A - 2*x2A_2
    doses_B = doses_B - x1B_2

    x2A = x2A_1 + x2A_2
    x1B = x1B_1 + x1B_2
        
    return(doses_A,doses_B,x2A,x1B)

def one_dose_A_or_one_dose_B(subpop,uptake,doses_A,doses_B):
    """
    This function allocates one dose of vaccine A or one dose of vaccine B
    """
    
    max_uptake = subpop*uptake
    
    c = np.array([-1,-1])

    A_ub = np.array([[1,0],
                     [0,1],
                     [1,1]])

    b_ub = np.array([doses_A,doses_B,max_uptake])

    bounds = ((0,None),(0,None))

    # Try to achieve maximum coverage while randomly allocating vaccines
    if doses_A > 0 and doses_B > 0:
        
        A_eq = np.array([[1,-1*doses_A/doses_B]])
        b_eq = np.array([0])
        
        sol_1 = linprog(c,
                        A_ub=A_ub,
                        b_ub=b_ub,
                        A_eq=A_eq,
                        b_eq=b_eq,
                        bounds=bounds,
                        method='simplex')

        x1A_1 = sol_1.x[0]
        x1B_1 = sol_1.x[1]

        # Update vaccine supply, remaining eligibility, and inequality constraints
        doses_A = doses_A - x1A_1
        doses_B = doses_B - x1B_1
        max_uptake = max_uptake - (x1A_1 + x1B_1)
        b_ub = np.array([doses_A,doses_B,max_uptake])
    else:
        x1A_1 = 0
        x1B_1 = 0

    # See if it's possible to achieve any higher coverage while 
    # allocating vaccines in a non-random fashion

    sol_2 = linprog(c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    bounds=bounds,
                    method='simplex')

    x1A_2 = sol_2.x[0]
    x1B_2 = sol_2.x[1]

    # Update vaccine supply
    doses_A = doses_A - x1A_2
    doses_B = doses_B - x1B_2

    x1A = x1A_1 + x1A_2
    x1B = x1B_1 + x1B_2
        
    return(doses_A,doses_B,x1A,x1B)

def two_dose_A_then_one_dose_B(subpop,uptake,doses_A,doses_B):
    """
    This function allocates two doses of vaccine A, 
    or one dose of B if no A remains
    """
    
    max_uptake = subpop*uptake
    
    x2A = min(max_uptake,0.5*doses_A)
    
    max_uptake = max_uptake - x2A
    doses_A = doses_A - 2*x2A
    
    x1B = min(max_uptake,doses_B)
    doses_B = doses_B - x1B
    
    return(doses_A,doses_B,x2A,x1B)

def one_dose_A_then_one_dose_B(subpop,uptake,doses_A,doses_B):
    
    """
    This function allocates one dose of vaccine A, 
    or one dose of B if no A remains
    """
    
    max_uptake = subpop*uptake
    
    x1A = min(max_uptake,doses_A)
    
    max_uptake = max_uptake - x1A
    doses_A = doses_A - x1A
    
    x1B = min(max_uptake,doses_B)
    doses_B = doses_B - x1B
    
    return(doses_A,doses_B,x1A,x1B)

def strategy_1_allocation(doses_A,doses_B,age_dist,uptake_by_age):
    """
    Corresponds to no vaccination
    """
    m = len(age_dist)
    n = 3
    
    # Rows correspond to different vaccines (2A,1A,1B)
    # Columns correspond to different age groups (0-19,20-59,60+)
    vaccine_allocation = np.zeros((m,n)) # Number of doses as % of total pop
    vaccine_coverage = np.zeros((m,n))   # Coverage within each age group
    doses_A = 0.0
    doses_B = 0.0
    
    vaccine_coverage = vaccine_allocation/age_dist
    
    return(doses_A,doses_B,vaccine_allocation,vaccine_coverage)

def strategy_2_allocation(doses_A,doses_B,age_dist,uptake_by_age):
    """
    This function allocates two doses of A or one dose of B 
    for all age groups, with older individuals given priority
    """
    m = len(age_dist)
    n = 3
    
    # Rows correspond to different vaccines (2A,1A,1B)
    # Columns correspond to different age groups (0-19,20-59,60+)
    vaccine_allocation = np.zeros((m,n)) # Number of doses as % of total pop
    vaccine_coverage = np.zeros((m,n))   # Coverage within each age group
    
    for i in np.flip(np.arange(m)):
        doses_A,doses_B,x2A,x1B = two_dose_A_or_one_dose_B(age_dist[i],uptake_by_age[i],doses_A,doses_B)
        vaccine_allocation[0,i] = x2A
        vaccine_allocation[2,i] = x1B
    
    vaccine_coverage = vaccine_allocation/age_dist
    
    return(doses_A,doses_B,vaccine_allocation,vaccine_coverage)

def strategy_3_allocation(doses_A,doses_B,age_dist,uptake_by_age):
    """
    This function allocates one dose of A or one dose of B 
    for all age groups, with older individuals given priority
    """
    m = len(age_dist)
    n = 3
    
    # Rows correspond to different vaccines (2A,1A,1B)
    # Columns correspond to different age groups (0-19,20-59,60+)
    
    vaccine_allocation = np.zeros((m,n)) # Number of doses as % of total pop
    vaccine_coverage = np.zeros((m,n))   # Coverage within each age group
    
    for i in np.flip(np.arange(m)):
        doses_A,doses_B,x1A,x1B = one_dose_A_or_one_dose_B(age_dist[i],uptake_by_age[i],doses_A,doses_B)
        vaccine_allocation[1,i] = x1A
        vaccine_allocation[2,i] = x1B
    
    vaccine_coverage = vaccine_allocation/age_dist
    
    return(doses_A,doses_B,vaccine_allocation,vaccine_coverage)
    
def strategy_4_allocation(doses_A,doses_B,age_dist,uptake_by_age):
    """
    This function allocates two doses of A to those >=60y
    (or one dose B if no A remains), followed by two doses A
    or one dose of B for those <60y
    """
    
    m = len(age_dist)
    n = 3
    
    # Rows correspond to different vaccines (2A,1A,1B)
    # Columns correspond to different age groups (0-19,20-59,60+)
    
    vaccine_allocation = np.zeros((m,n)) # Number of doses as % of total pop
    vaccine_coverage = np.zeros((m,n))   # Coverage within each age group
    
    i=2 
    doses_A,doses_B,x2A,x1B = two_dose_A_then_one_dose_B(age_dist[i],uptake_by_age[i],doses_A,doses_B)
    vaccine_allocation[0,i] = x2A
    vaccine_allocation[2,i] = x1B
    
    i=1
    doses_A,doses_B,x2A,x1B = two_dose_A_or_one_dose_B(age_dist[i],uptake_by_age[i],doses_A,doses_B)
    vaccine_allocation[0,i] = x2A
    vaccine_allocation[2,i] = x1B
    
    i=0
    doses_A,doses_B,x2A,x1B = two_dose_A_or_one_dose_B(age_dist[i],uptake_by_age[i],doses_A,doses_B)
    vaccine_allocation[0,i] = x2A
    vaccine_allocation[2,i] = x1B
    
    vaccine_coverage = vaccine_allocation/age_dist
    
    return(doses_A,doses_B,vaccine_allocation,vaccine_coverage)

def strategy_5_allocation(doses_A,doses_B,age_dist,uptake_by_age):
    """
    This function allocates two doses of A to those >=60y
    (or one dose B if no A remains), followed by one dose A
    or one dose of B for those <60y
    """
    
    m = len(age_dist)
    n = 3
    
    # Rows correspond to different vaccines (2A,1A,1B)
    # Columns correspond to different age groups (0-19,20-59,60+)
    
    vaccine_allocation = np.zeros((m,n)) # Number of doses as % of total pop
    vaccine_coverage = np.zeros((m,n))   # Coverage within each age group
    
    i=2 
    doses_A,doses_B,x2A,x1B = two_dose_A_then_one_dose_B(age_dist[i],uptake_by_age[i],doses_A,doses_B)
    vaccine_allocation[0,i] = x2A
    vaccine_allocation[2,i] = x1B
    
    i=1
    doses_A,doses_B,x1A,x1B = one_dose_A_or_one_dose_B(age_dist[i],uptake_by_age[i],doses_A,doses_B)
    vaccine_allocation[1,i] = x1A
    vaccine_allocation[2,i] = x1B
    
    i=0
    doses_A,doses_B,x1A,x1B = one_dose_A_or_one_dose_B(age_dist[i],uptake_by_age[i],doses_A,doses_B)
    vaccine_allocation[1,i] = x1A
    vaccine_allocation[2,i] = x1B
    
    vaccine_coverage = vaccine_allocation/age_dist
    
    return(doses_A,doses_B,vaccine_allocation,vaccine_coverage)

def severity_reduction(VE1,VE2,VE3,VE4,P0,P1,P2,P3,P4):
    """
    this function calculates the new severity distribution based on the 
    efficacy of a given vaccine and the original severity distribution
    of the age group
    """
    P1v = (1-VE1)*P1 + (VE2-VE1)*(P2 + P3 + P4)
    P2v = (1-VE2)*P2 + (VE3-VE2)*(P3 + P4)
    P3v = (1-VE3)*P3 + (VE4-VE3)*(P4)
    P4v = (1-VE4)*P4
    P0v = 1 - (P1v + P2v + P3v + P4v)
    
    f = np.array([P0v,P1v,P2v,P3v,P4v])
    
    return(f)

def calculate_immunity_parameters(severity_dist,VE_1vs0,VE_2vs0,vaccine_allocation):
    """
    This function calculates the outcomes of vaccination
    """
    # Severity distribution that now includes a row for those 
    # who are going to be newly immune after receiving vaccine
    # Rows: immune,asym,mild/moderate,severe,critical
    # Columns: 0-19,20-59,60+
    unvaccinated_severity_dist = np.zeros((5,3))
    unvaccinated_severity_dist[1:,:] = severity_dist
    
    # Probability that a vaccinated person got a particular vaccine, by age
    # Rows: 2A,1A,1B
    # Columns: 0-19,20-59,60+
    normalized_allocation = vaccine_allocation/np.sum(vaccine_allocation,0)
    
    # Vaccine efficacy of 2 dose regimen relative to 1 dose regimen
    # Rows: 2A,1A,1B
    # Columns: VE1, VE2, VE3, VE4
    VE_2vs1 = 1 - (1-VE_2vs0)/(1-VE_1vs0)
    
    dose_1_severity_dist = np.zeros((5,3))
    dose_2_severity_dist = np.zeros((5,3))
    
    # *** First dose ***
    
    for i in range(5):         # Immune state
        for j in range(3):     # Age 
            for v in range(3): # Vaccine 
                
                # Probability that someone age j got vaccine v
                A = normalized_allocation[v,j]
                
                # Efficacy of vaccine v
                VE1,VE2,VE3,VE4 = VE_1vs0[v,:]
                
                # Severity distribution of someone age i
                P0,P1,P2,P3,P4 = unvaccinated_severity_dist[:,j]
                
                # Probability of immune state i after vaccination
                # among those age j receiving vaccine v
                f = severity_reduction(VE1,VE2,VE3,VE4,P0,P1,P2,P3,P4)
                
                # Multiply by probability that someone age j
                # receives vaccine v, and add to new severity dist
                dose_1_severity_dist[i,j] += A*f[i]
                
    # *** Second dose *** #
    
    for i in range(5):         # Immune state
        for j in range(3):     # Age 
            for v in range(3): # Vaccine 
                
                # Probability that someone age j got vaccine v
                A = normalized_allocation[v,j]
                
                # Efficacy of vaccine v
                VE1,VE2,VE3,VE4 = VE_2vs1[v,:]
                
                # Severity distribution of someone age i
                P0,P1,P2,P3,P4 = dose_1_severity_dist[:,j]
                
                # Probability of immune state i after vaccination
                # among those age j receiving vaccine v
                f = severity_reduction(VE1,VE2,VE3,VE4,P0,P1,P2,P3,P4)
                
                # Multiply by probability that someone age j
                # receives vaccine v, and add to new severity dist
                dose_2_severity_dist[i,j] += A*f[i]
                
    # Immunity parameters
    prob_full_immunity_dose_1 = (dose_1_severity_dist[0,:]-unvaccinated_severity_dist[0,:])/(1-unvaccinated_severity_dist[0,:])
    prob_full_immunity_dose_2 = (dose_2_severity_dist[0,:]-dose_1_severity_dist[0,:])/(1-dose_1_severity_dist[0,:])
    
    partial_immunity_severity_dist_dose_1 = np.zeros((4,3))
    partial_immunity_severity_dist_dose_2 = np.zeros((4,3))
    
    for i in range(4):
        for j in range(3):
            i_mod = i+1
            partial_immunity_severity_dist_dose_1[i,j] = dose_1_severity_dist[i_mod,j]/np.sum(dose_1_severity_dist[1:,j])
            partial_immunity_severity_dist_dose_2[i,j] = dose_2_severity_dist[i_mod,j]/np.sum(dose_2_severity_dist[1:,j])
    
    # Return results
    res = [prob_full_immunity_dose_1,
           prob_full_immunity_dose_2,
           partial_immunity_severity_dist_dose_1,
           partial_immunity_severity_dist_dose_2,
           dose_1_severity_dist,
           dose_2_severity_dist]
    
    return(res)

def calculate_1st_dose_pace(vaccine_allocation,vaccinations_per_day):
    
    smallnum = 1e-12 # Small number to avoid division by zero if no doses allocated
    
    dose_1_per_day = np.zeros(3)
    
    # Loop through age groups and caculate # 1st doses administered per day
    for j in range(3):
        x2A = vaccine_allocation[0,j] # 2 dose A
        x1A = vaccine_allocation[1,j] # 1 dose A
        x1B = vaccine_allocation[2,j] # 1 dose B
        
        # Fraction of daily vaccinations that are first doses
        r_mult = (0.5*x2A + x1A + x1B)/(x2A + x1A + x1B + smallnum)
        
        dose_1_per_day[j] = r_mult*vaccinations_per_day
        
    return(dose_1_per_day)

def calculate_rollout_intervals(vaccine_allocation,dose_1_per_day,delay_to_efficacy_dose_1):
    
    smallnum = 1e-12 # Small number to avoid division by zero if no doses allocated
    
    # Total doses allocated to each age group
    doses_0_to_19,doses_20_to_59,doses_60_plus = np.sum(vaccine_allocation,0)
    
    # Number of first doses per day by age group
    pace_0_to_19,pace_20_to_59,pace_60_plus = dose_1_per_day
    
    t0 = delay_to_efficacy_dose_1
    t1 = t0 + np.round(doses_60_plus/(pace_60_plus + smallnum))
    t2 = t1 + np.round(doses_20_to_59/(pace_20_to_59 + smallnum))
    t3 = t2 + np.round(doses_0_to_19/(pace_0_to_19 + smallnum))
    
    time_intervals = np.array([t0,t1,t2,t3])
    
    return(time_intervals)

def get_run_list_row_as_dict(run_list,run):
    """
    Returns the row corresponding to a given country and strategy as a dictionary. 
    """
    run_row = run_list[(run_list['run.name']==run)].to_dict('records')[0]
    return(run_row)

def get_input_params(run_row):
    
    uptake = run_row['uptake']
    uptake_by_age = np.array([uptake,uptake,uptake])
    doses_A = run_row['supply.A']
    doses_B = run_row['supply.B']
    
    VE2A = np.zeros(4)
    VE1A = np.zeros(4)
    VE1B = np.zeros(4)
    
    for i in range(4):
        VE2A[i] = run_row[f'VE{i+1}.2A']
        VE1A[i] = run_row[f'VE{i+1}.1A']
        VE1B[i] = run_row[f'VE{i+1}.1B']
        
    VE_1vs0 = np.array([VE1A,VE1A,VE1B])
    VE_2vs0 = np.array([VE2A,VE1A,VE1B])
    
    # I hard-coded these, but if we decide to vary this I'll create an input for it
    severity_dist = np.array([[0.299342,0.179000,0.171045],
                          [0.697833,0.803755,0.763740],
                          [0.002495,0.007952,0.013583],
                          [0.00033,0.009293,0.051632]])
    
    age_dist = np.array([0.36681040,0.54292588,0.09026372])
    
    vaccinations_per_day = run_row['vaccinations.per.day']
    
    return(age_dist,severity_dist,uptake_by_age,doses_A,doses_B,VE_1vs0,VE_2vs0,vaccinations_per_day)

def tests_and_resources(infile,strat_row):
    
    infile['simulation parameters']['cohort size'] = strat_row['sim.size']
    infile['resources']['resource availabilities']['for 0 <= day# < t0']['resource 1'] = np.round(strat_row['hospital.beds.per.100k']/100000*strat_row['sim.size'])
    infile['resources']['resource availabilities']['for 0 <= day# < t0']['resource 2'] = np.round(strat_row['icu.beds.per.100k']/100000*strat_row['sim.size'])
    infile['tests']['daily test availabilities']['for t0 <= day# < t1']['test 0'] = np.round(strat_row['dose.1.per.day'][2]*strat_row['sim.size'])
    infile['tests']['daily test availabilities']['for t1 <= day# < t2']['test 1'] = np.round(strat_row['dose.1.per.day'][1]*strat_row['sim.size'])
    infile['tests']['daily test availabilities']['for t2 <= day# < t3']['test 2'] = np.round(strat_row['dose.1.per.day'][0]*strat_row['sim.size'])
    infile['tests']['test availability thresholds']['t0'] = strat_row['time.intervals'][0]
    infile['tests']['test availability thresholds']['t1'] = strat_row['time.intervals'][1]
    infile['tests']['test availability thresholds']['t2'] = strat_row['time.intervals'][2]
    infile['tests']['test availability thresholds']['t3'] = strat_row['time.intervals'][3]
    infile['tests']['test 3']['delay to test'] = strat_row['dose.2.delay']
    
    return(infile)

def immunity_parameters(infile,strat_row):
    
    for i in range(3):
        infile['initial state']['immune states dist'][f'for tn_group {i}']['recovered'] = np.round(strat_row['proportion.recovered'],8)
        infile['initial state']['immune states dist'][f'for tn_group {i}']['naive'] = 1-np.round(strat_row['proportion.recovered'],8)
        
    age = ['0-19y','20-59y','>=60y']
    
    for j in range(3):
        infile['immunity']['immunity parameters for vaccine_0']['initial prob full immunity'][f'for {age[j]}'] = strat_row['prob.full.immunity.dose.1'][j]
        infile['immunity']['immunity parameters for vaccine_1']['initial prob full immunity'][f'for {age[j]}'] = strat_row['prob.full.immunity.dose.2'][j]
        
    state = ['asymptomatic','mild/moderate','severe','critical']
    
    for i in range(4):
        for j in range(3):
            infile['immunity']['immunity parameters for vaccine_0']['partial immunity severity dist'][f'for {age[j]}'][state[i]] = strat_row['partial.immunity.severity.dist.dose.1'][i][j]
            infile['immunity']['immunity parameters for vaccine_1']['partial immunity severity dist'][f'for {age[j]}'][state[i]] = strat_row['partial.immunity.severity.dist.dose.2'][i][j]
            
    return(infile)
    
def transmission_multipliers(infile,strat_row):
    
    for i in range(18):
        infile['transmissions'][f'intervention {i}']['transmission rate multipliers']['for 0 <= day# < t0'] = strat_row['transmission.multiplier']
    
    return(infile)


# In[3]:


pwd = os.getcwd()

name = '2021-05-21'
run_list_path = os.path.join(pwd,'run_dictionary',name+'.xlsx')
outfolder = os.path.join(pwd,name) # Filepath for output

run_list = pd.read_excel(run_list_path)

template_path = os.path.join(pwd,'input_file_templates','two_dose_template.json')

# Read in template input file
with open(template_path) as f1:
    template_file = json.load(f1)
    f1.close()


# In[4]:


# Create parent folder 
if not os.path.exists(outfolder):
    os.makedirs(outfolder)
    
# Create JSON folder inside outfolder
jsonfolder = os.path.join(outfolder,'0_JSON_info')

if not os.path.exists(jsonfolder):
    os.makedirs(jsonfolder)


for run in run_list['run.name'].unique():
    
    # Create run folder
    runfolder = os.path.join(outfolder,run)
    if not os.path.exists(runfolder):
        os.makedirs(runfolder)
    
    for strat_num in [1,2,3,4,5]:
        
        # Create strategy folder 
        strategyfolder = os.path.join(runfolder,str(strat_num))
        if not os.path.exists(strategyfolder):
            os.makedirs(strategyfolder)
        
        run_row = get_run_list_row_as_dict(run_list,run)
        age_dist,severity_dist,uptake_by_age,doses_A,doses_B,VE_1vs0,VE_2vs0,vaccinations_per_day = get_input_params(run_row)
    
        # Get function used to allocate doses (specific to each strategy)
        alloc_func = locals()[f'strategy_{strat_num}_allocation']
    
        doses_A,doses_B,vaccine_allocation,vaccine_coverage = alloc_func(doses_A,doses_B,age_dist,uptake_by_age)
        
        res = calculate_immunity_parameters(severity_dist,VE_1vs0,VE_2vs0,vaccine_allocation)  

        prob_full_immunity_dose_1 = res[0]
        prob_full_immunity_dose_2 = res[1]
        partial_immunity_severity_dist_dose_1 = res[2]
        partial_immunity_severity_dist_dose_2 = res[3]
        dose_1_severity_dist = res[4]
        dose_2_severity_dist = res[5]
        
        # Ensure there's no NaN (happens when a group doesn't get vaccine)
        prob_full_immunity_dose_1 = np.nan_to_num(prob_full_immunity_dose_1)
        prob_full_immunity_dose_2 = np.nan_to_num(prob_full_immunity_dose_2)
        partial_immunity_severity_dist_dose_1 = np.nan_to_num(partial_immunity_severity_dist_dose_1)
        partial_immunity_severity_dist_dose_2 = np.nan_to_num(partial_immunity_severity_dist_dose_2)
        
        # Round to 8 decimal places, and ensure that it still sums to 1
        prob_full_immunity_dose_1 = np.round(prob_full_immunity_dose_1,8)
        prob_full_immunity_dose_2 = np.round(prob_full_immunity_dose_2,8)
        partial_immunity_severity_dist_dose_1 = np.round(partial_immunity_severity_dist_dose_1,8)
        partial_immunity_severity_dist_dose_1[0,:] = np.round(1 - np.sum(partial_immunity_severity_dist_dose_1[1:,:],0),8)
        partial_immunity_severity_dist_dose_2 = np.round(partial_immunity_severity_dist_dose_2,8)
        partial_immunity_severity_dist_dose_2[0,:] = np.round(1 - np.sum(partial_immunity_severity_dist_dose_2[1:,:],0),8)

        dose_1_per_day = calculate_1st_dose_pace(vaccine_allocation,vaccinations_per_day)
        time_intervals = calculate_rollout_intervals(vaccine_allocation,dose_1_per_day,14)
        
        strat_row = run_row.copy()
        strat_row['strategy'] = strat_num
        strat_row['doses.A.remaining'] = doses_A
        strat_row['doses.B.remaining'] = doses_B
        strat_row['vaccine.allocation'] = vaccine_allocation.tolist()
        strat_row['vaccine.coverage'] = vaccine_coverage.tolist()
        strat_row['prob.full.immunity.dose.1'] = prob_full_immunity_dose_1.tolist()
        strat_row['prob.full.immunity.dose.2'] = prob_full_immunity_dose_2.tolist()
        strat_row['partial.immunity.severity.dist.dose.1'] = partial_immunity_severity_dist_dose_1.tolist()
        strat_row['partial.immunity.severity.dist.dose.2'] = partial_immunity_severity_dist_dose_2.tolist()
        strat_row['dose.1.per.day'] = dose_1_per_day.tolist()
        strat_row['time.intervals'] = time_intervals.tolist()
        
        # Update input file with run-specific info
        infile = template_file.copy()
        infile = tests_and_resources(infile,strat_row)
        infile = immunity_parameters(infile,strat_row)
        infile = transmission_multipliers(infile,strat_row)
        
        # Save input file
        infile_name = f'{run}_strategy_{strat_num}.json'
        outJSON = os.path.join(strategyfolder,infile_name)
        with open(outJSON, 'w') as f:
            json.dump(infile, f, indent=2)
            f.close()   
            
        # Save info file
        info_name = f'{run}_strategy_{strat_num}_info.json'
        outJSON = os.path.join(jsonfolder,info_name)
        with open(outJSON, 'w') as f:
            json.dump(strat_row, f, indent=2)
            f.close()   
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




