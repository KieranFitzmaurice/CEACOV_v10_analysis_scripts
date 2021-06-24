#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

def vaccinate_subpopulation(subpop,uptake,supply,prob_A):
    """
    This function allocates two doses of vaccine A
    or two doses of vaccine B to a single age group

    subpop = size of subpopulation as % of total pop
    uptake = % accepting vaccine among those eligible
    supply = remaining supply as % of total pop
    prob_A = probability of receiving vaccine A
    """

    # Maximum number of people willing to be vaccinated
    max_uptake = subpop*uptake

    # Number of people allocated doses
    x = min(supply,max_uptake)

    # Number of people allocated vaccine A or vaccine B
    xA = x*prob_A
    xB = x*(1-prob_A)

    # Subtract vaccines allocated from remaining supply
    supply = supply - x

    return(supply,xA,xB)

def vaccinate_population(age_vaccination_dist,max_uptake,remaining_supply,prob_A):
    """
    This function allocates vaccines across a structured
    population while prioritizing older age groups
    """

    m = 2  # Number of vaccines
    n = 3  # Number of age groups

    # Rows correspond to different vaccines (A,B)
    # Cols correspond to different age groups (0-19,20-59,60+)

    vaccine_allocation = np.zeros((m,n))
    vaccine_coverage = np.zeros((m,n))

    for j in np.flip(np.arange(n)):

        # Calculate uptake among those who are still unvaccinated in age group
        number_unvaccinated = age_vaccination_dist[0,j]
        number_vaccinated = age_vaccination_dist[1,j]
        total_group_size = number_unvaccinated + number_vaccinated
        number_unvaccinated_but_accepting = max_uptake*total_group_size - number_vaccinated
        uptake_among_unvaccinated = number_unvaccinated_but_accepting/number_unvaccinated

        remaining_supply,xA,xB = vaccinate_subpopulation(number_unvaccinated,uptake_among_unvaccinated,remaining_supply,prob_A)
        vaccine_allocation[0,j] = xA
        vaccine_allocation[1,j] = xB

    vaccine_coverage = vaccine_allocation/np.sum(age_vaccination_dist,0)

    leftover_A = remaining_supply*prob_A
    leftover_B = remaining_supply*(1-prob_A)

    return(leftover_A,leftover_B,vaccine_allocation,vaccine_coverage)

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

def calculate_immunity_parameters(severity_dist,VE_1vs0,VE_2vs0,prob_A):
    """
    This function calculates the outcomes of vaccination
    """
    # Small number to avoid introducing NaNs when dividing zero by zero
    # Uses smallest number available based on machine precision (~1e-16)
    smallnum = np.finfo(float).eps

    # Severity distribution that now includes a row for those
    # who are going to be newly immune after receiving vaccine
    # Rows: immune,asym,mild/moderate,severe,critical
    # Columns: 0-19,20-59,60+
    unvaccinated_severity_dist = np.zeros((5,3))
    unvaccinated_severity_dist[1:,:] = severity_dist

    # Probability that a vaccinated person got a particular vaccine, by age
    # values: prob vaccine A, prob vaccine B
    P_vaccine = np.array([prob_A,(1-prob_A)])

    # Vaccine efficacy of 2 dose regimen relative to 1 dose regimen
    # Rows: vaccine A, vaccine B
    # Columns: VE1, VE2, VE3, VE4
    VE_2vs1 = 1 - (1-VE_2vs0)/((1-VE_1vs0)+smallnum)

    dose_1_severity_dist = np.zeros((5,3))
    dose_2_severity_dist = np.zeros((5,3))

    # *** First dose ***

    for i in range(5):         # Path
        for j in range(3):     # Age
            for v in range(2): # Vaccine

                # Efficacy of vaccine v
                VE1,VE2,VE3,VE4 = VE_1vs0[v,:]

                # Severity distribution of someone age j
                P0,P1,P2,P3,P4 = unvaccinated_severity_dist[:,j]

                # Probability of path i after vaccination
                # among those age j receiving vaccine v
                f = severity_reduction(VE1,VE2,VE3,VE4,P0,P1,P2,P3,P4)

                # Multiply by probability that someone age j
                # receives vaccine v, and add to new severity dist
                dose_1_severity_dist[i,j] += P_vaccine[v]*f[i]

    # *** Second dose *** #

    for i in range(5):         # Path
        for j in range(3):     # Age
            for v in range(2): # Vaccine

                # Efficacy of vaccine v
                VE1,VE2,VE3,VE4 = VE_2vs1[v,:]

                # Severity distribution of someone age j
                P0,P1,P2,P3,P4 = dose_1_severity_dist[:,j]

                # Probability of path i after vaccination
                # among those age j receiving vaccine v
                f = severity_reduction(VE1,VE2,VE3,VE4,P0,P1,P2,P3,P4)

                # Multiply by probability that someone age j
                # receives vaccine v, and add to new severity dist
                dose_2_severity_dist[i,j] += P_vaccine[v]*f[i]

    # Immunity parameters

    # Get probability of full immunity
    # by comparing the number predicted to be fully
    # immune before and after each dose
    prob_full_immunity_dose_1 = (dose_1_severity_dist[0,:]-unvaccinated_severity_dist[0,:])/((1-unvaccinated_severity_dist[0,:])+smallnum)
    prob_full_immunity_dose_2 = (dose_2_severity_dist[0,:]-dose_1_severity_dist[0,:])/((1-dose_1_severity_dist[0,:])+smallnum)

    partial_immunity_severity_dist_dose_1 = np.zeros((4,3))
    partial_immunity_severity_dist_dose_2 = np.zeros((4,3))

    # Renormalize partial immunity severity distribution after
    # removing those who are fully immune
    for i in range(4):
        i_mod = i+1
        for j in range(3):
            partial_immunity_severity_dist_dose_1[i,j] = dose_1_severity_dist[i_mod,j]/(np.sum(dose_1_severity_dist[1:,j])+smallnum)
            partial_immunity_severity_dist_dose_2[i,j] = dose_2_severity_dist[i_mod,j]/(np.sum(dose_2_severity_dist[1:,j])+smallnum)

    # Return results
    res = [prob_full_immunity_dose_1,
           prob_full_immunity_dose_2,
           partial_immunity_severity_dist_dose_1,
           partial_immunity_severity_dist_dose_2,
           dose_1_severity_dist,
           dose_2_severity_dist]

    return(res)

def calculate_rollout_intervals(vaccine_allocation,vaccinations_per_day,delay_to_efficacy_dose_1):

    # Small number to avoid division by zero if no doses allocated
    smallnum = np.finfo(float).eps

    # Total doses allocated to each age group
    doses_0_to_19,doses_20_to_59,doses_60_plus = np.sum(vaccine_allocation,0)

    t0 = delay_to_efficacy_dose_1
    t1 = t0 + np.round(doses_60_plus/(vaccinations_per_day + smallnum))
    t2 = t1 + np.round(doses_20_to_59/(vaccinations_per_day + smallnum))
    t3 = t2 + np.round(doses_0_to_19/(vaccinations_per_day + smallnum))

    time_intervals = np.array([t0,t1,t2,t3])

    return(time_intervals)

def get_initial_conditions(age_vaccination_dist,unvaccinated_SIR,infected_state_dist,prob_full_immunity_dose_1,prob_full_immunity_dose_2):
    """
    Determine distribution of individuals across transmission groups, age groups, and health states at model entry
    """

    tn_group_dist = np.zeros(4)
    tn_group_dist[:3] = age_vaccination_dist[0,:]
    tn_group_dist[3] = np.sum(age_vaccination_dist[1,:])
    tn_group_dist = ensure_vector_sum(tn_group_dist,8)

    age_dist_by_tn_group = np.zeros((4,3))
    age_dist_by_tn_group[0,0] = 1 # 0-19, unvaccinated
    age_dist_by_tn_group[1,1] = 1 # 20-59, unvaccinated
    age_dist_by_tn_group[2,2] = 1 # 60+, unvaccinated
    age_dist_by_tn_group[3,:] = age_vaccination_dist[1,:]/np.sum(age_vaccination_dist[1,:]) # all ages, vaccinated

    # Make sure rows sum to 1.0
    age_dist_by_tn_group = ensure_col_sum(age_dist_by_tn_group,8)

    # Those who are unvaccinated susceptible/infected are naive (all those who are unvaccinated immune are recovered)
    prob_unvaccinated_naive = unvaccinated_SIR[0] + unvaccinated_SIR[1]

    # Probability of being infected if you're unvaccinated and naive
    prob_infected_naive = unvaccinated_SIR[1]/prob_unvaccinated_naive
    prob_susceptible_naive = 1.0 - prob_infected_naive

    # Probability of being immune if you're vaccinated
    prob_immune_vaccinated = prob_full_immunity_dose_1[0] + (1-prob_full_immunity_dose_1[0])*prob_full_immunity_dose_2[0]
    prob_susceptible_vaccinated = 1.0 - prob_immune_vaccinated

    # Define disease distributions
    disease_dist_naive = np.zeros(8)
    disease_dist_recovered = np.zeros(8)
    disease_dist_vaccinated = np.zeros(8)

    # Naive disease dist
    disease_dist_naive[0] = prob_susceptible_naive                      # Susceptible
    disease_dist_naive[1:7] = prob_infected_naive*infected_state_dist   # Infected states (assume no one's naive and immune)

    # Recovered disease dist
    disease_dist_recovered[7] = 1.0                                     # Assume all recovered are immune

    # Vaccinated disease dist
    disease_dist_vaccinated[0] = prob_susceptible_vaccinated            # Susceptible
    disease_dist_vaccinated[7] = prob_immune_vaccinated                 # Immune (assume no one's vaccinated and infected)

    # Round to 8 decimal places and ensure things still sum to 1.0
    disease_dist_naive = ensure_vector_sum(disease_dist_naive,8)
    disease_dist_recovered = ensure_vector_sum(disease_dist_recovered,8)
    disease_dist_vaccinated = ensure_vector_sum(disease_dist_vaccinated,8)

    # Rows correspond to immune status: naive, recovered, vaccinated
    disease_dist = np.array([disease_dist_naive,disease_dist_recovered,disease_dist_vaccinated])

    return(tn_group_dist,age_dist_by_tn_group,prob_unvaccinated_naive,disease_dist)

def get_run_list_row_as_dict(run_list,run,strategy):
    """
    Returns the row corresponding to a given run as a dictionary.
    """
    run_row = run_list[(run_list['run.name']==run) & (run_list['strategy']==strategy)].to_dict('records')[0]
    return(run_row)

def get_vaccine_input_params(run_row):

    max_uptake = run_row['uptake']
    prob_A = run_row['prob.vaccine.A']
    initial_supply = run_row['target.coverage']
    vaccinations_per_day = run_row['vaccinations.per.day']

    VE1A = np.zeros(4)
    VE1B = np.zeros(4)
    VE2A = np.zeros(4)
    VE2B = np.zeros(4)

    for i in range(4):
        VE1A[i] = run_row[f'VE{i+1}.1A']
        VE1B[i] = run_row[f'VE{i+1}.1B']
        VE2A[i] = run_row[f'VE{i+1}.2A']
        VE2B[i] = run_row[f'VE{i+1}.2B']

    VE_1vs0 = np.array([VE1A,VE1B])
    VE_2vs0 = np.array([VE2A,VE2B])

    age_labels = ['0.to.19','20.to.59','60.plus']

    # Rows correspond to unvaccinated/vaccinated at model start
    # Columns correspond to age strata
    # Denominator is total population
    # Sum rows to get age distribution
    # Sum columns to get proportions unvaccinated/vaccinated
    age_vaccination_dist = np.zeros((2,3))

    # Age dist
    overall_age_dist = np.zeros(3)

    # Proportion of each age group that's previously been vaccinated
    prior_coverage_by_age = np.zeros(3)

    for i in range(len(age_labels)):
        overall_age_dist[i] = run_row[f'proportion.{age_labels[i]}']
        prior_coverage_by_age[i] = run_row[f'prior.coverage.{age_labels[i]}']

    age_vaccination_dist[0,:] = overall_age_dist*(1-prior_coverage_by_age)
    age_vaccination_dist[1,:] = overall_age_dist*prior_coverage_by_age

    # Number of doses still available as % of total population
    remaining_supply = initial_supply - np.sum(age_vaccination_dist[1,:])

    return(age_vaccination_dist,max_uptake,remaining_supply,prob_A,VE_1vs0,VE_2vs0,vaccinations_per_day)

def tests_and_resources(infile,strat_row):

    infile['simulation parameters']['cohort size'] = strat_row['sim.size']
    infile['resources']['resource availabilities']['for 0 <= day# < t0']['resource 1'] = np.round(strat_row['hospital.beds.per.100k']/100000*strat_row['sim.size'])
    infile['resources']['resource availabilities']['for 0 <= day# < t0']['resource 2'] = np.round(strat_row['icu.beds.per.100k']/100000*strat_row['sim.size'])
    infile['tests']['daily test availabilities']['for t0 <= day# < t1']['test 0'] = np.round(strat_row['dose.1.per.day']*strat_row['sim.size'])
    infile['tests']['daily test availabilities']['for t1 <= day# < t2']['test 1'] = np.round(strat_row['dose.1.per.day']*strat_row['sim.size'])
    infile['tests']['daily test availabilities']['for t2 <= day# < t3']['test 2'] = np.round(strat_row['dose.1.per.day']*strat_row['sim.size'])
    infile['tests']['test availability thresholds']['t0'] = strat_row['time.intervals'][0]
    infile['tests']['test availability thresholds']['t1'] = strat_row['time.intervals'][1]
    infile['tests']['test availability thresholds']['t2'] = strat_row['time.intervals'][2]
    infile['tests']['test availability thresholds']['t3'] = strat_row['time.intervals'][3]
    infile['tests']['test 3']['delay to test'] = (strat_row['dose.2.delay'] + strat_row['dosing.interval']) - strat_row['dose.1.delay']

    return(infile)

def immunity_parameters(infile,strat_row):

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

def initial_conditions(infile,strat_row):

    # Set number of people in each transmission group
    for i in range(4):
        infile['initial state']['transmission group dist'][f'tn_group {i}'] = strat_row['tn.group.dist'][i]

    age = ['0-19y','20-59y','>=60y']

    # Set age distribution of each transmission group
    for i in range(4):
        for j in range(3):
            infile['initial state']['risk category dist'][f'for tn_group {i}'][age[j]] = strat_row['age.dist.by.tn.group'][i][j]

    # Set immune status distribution for transmission groups 0, 1, 2
    for i in range(3):
        infile['initial state']['immune states dist'][f'for tn_group {i}']['naive'] = np.round(strat_row['prob.naive.if.unvaccinated'],8)
        infile['initial state']['immune states dist'][f'for tn_group {i}']['recovered'] = np.round(1 - np.round(strat_row['prob.naive.if.unvaccinated'],8),8)

    # Set immune status distribution for transmission group 3
    infile['initial state']['immune states dist']['for tn_group 3']['vaccine_1'] = 1.0

    disease_state = ['susceptible','pre-infectious incubation','asymptomatic','mild/moderate','severe','critical','recuperation','immune']

    # Set initial disease distribution
    for i in range(8):
        # For unvaccinated transmission groups
        infile['initial state']['initial disease dist']['for tn_group 0']['for naive'][disease_state[i]] = strat_row['disease.dist'][0][i]
        infile['initial state']['initial disease dist']['for tn_group 0']['for recovered'][disease_state[i]] = strat_row['disease.dist'][1][i]
        infile['initial state']['initial disease dist']['for tn_group 1']['for naive'][disease_state[i]] = strat_row['disease.dist'][0][i]
        infile['initial state']['initial disease dist']['for tn_group 1']['for recovered'][disease_state[i]] = strat_row['disease.dist'][1][i]
        infile['initial state']['initial disease dist']['for tn_group 2']['for naive'][disease_state[i]] = strat_row['disease.dist'][0][i]
        infile['initial state']['initial disease dist']['for tn_group 2']['for recovered'][disease_state[i]] = strat_row['disease.dist'][1][i]

        # For transmission group 3 (vaccinated)
        infile['initial state']['initial disease dist']['for tn_group 3']['for vaccine_1'][disease_state[i]] = strat_row['disease.dist'][2][i]

    return(infile)

def transmission_multipliers(infile,strat_row):

    # Set time intervals
    infile['transmissions']['transmission multiplier time thresholds']['t0'] = strat_row['first.surge.start.day']
    infile['transmissions']['transmission multiplier time thresholds']['t1'] = strat_row['first.surge.end.day']
    infile['transmissions']['transmission multiplier time thresholds']['t2'] = strat_row['second.surge.start.day']
    infile['transmissions']['transmission multiplier time thresholds']['t3'] = strat_row['second.surge.end.day']

    # Adjust multipliers
    for i in range(18):
        infile['transmissions'][f'intervention {i}']['transmission rate multipliers']['for 0 <= day# < t0'] = strat_row['baseline.transmission.multiplier']
        infile['transmissions'][f'intervention {i}']['transmission rate multipliers']['for t0 <= day# < t1'] = strat_row['first.surge.transmission.multiplier']
        infile['transmissions'][f'intervention {i}']['transmission rate multipliers']['for t1 <= day# < t2'] = strat_row['baseline.transmission.multiplier']
        infile['transmissions'][f'intervention {i}']['transmission rate multipliers']['for t2 <= day# < t3'] = strat_row['second.surge.transmission.multiplier']
        infile['transmissions'][f'intervention {i}']['transmission rate multipliers']['day# > t3'] = strat_row['baseline.transmission.multiplier']

    return(infile)

def contact_matrices(infile,strat_row):
    """
    Enforce homogeneous mixing by making contacts proportional to the size of each transmission group
    """

    for i in range(18):
        for j in range(4):
            for k in np.arange(j,4):

                infile['transmissions'][f'intervention {i}']['exposure matrix'][f'from tn_group {j}'][f'to tn_group {k}'] = strat_row['tn.group.dist'][k]

    return(infile)

def write_array_to_dict(row_names,col_names,values):
    """
    Save array in JSON format with labels
    """
    mydict = {}
    for i in range(len(row_names)):
        mydict[row_names[i]] = {}
        for j in range(len(col_names)):
            mydict[row_names[i]][col_names[j]] = values[i,j]

    return(mydict)

def write_vector_to_dict(labels,values):
    """
    Save vector in JSON format with labels
    """
    mydict = {}
    for i in range(len(labels)):
        mydict[labels[i]] = values[i]
    return(mydict)

def ensure_vector_sum(x,places):
    """
    Round a vector while ensuring that it's elements still sum to 1.0
    """
    n = len(x)
    x[:(n-1)] = np.round(x[:(n-1)],places)
    x[(n-1)] = np.round(1.0 - np.sum(x[:(n-1)]),places+2)
    return(x)

def ensure_row_sum(x,places):
    """
    Round a matrix while ensuring rows still sum to 1.0
    """
    m,n = x.shape

    for j in range(n):
        x[:,j] = ensure_vector_sum(x[:,j],places)

    return(x)

def ensure_col_sum(x,places):
    """
    Round a matrix while ensuring columns still sum to 1.0
    """
    m,n = x.shape

    for i in range(m):
        x[i,:] = ensure_vector_sum(x[i,:],places)

    return(x)

# Main part of script

pwd=os.getcwd()

name = 'test_run_2021-06-21_VKD'
run_list_path = os.path.join(pwd,'run_dictionary',name+'.xlsx')
outfolder = os.path.join(pwd,name) # Filepath for output

run_list = pd.read_excel(run_list_path,sheet_name = 'Main Parameters')
severity_data = pd.read_excel(run_list_path,sheet_name = 'Covid-Naive Severity Dist',index_col=0)
infected_state_data = pd.read_excel(run_list_path,sheet_name = 'Infected State Dist',index_col=0)

template_path = os.path.join(pwd,'input_file_templates','two_dose_template.json')

# Read in template input file
with open(template_path) as f1:
    template_file = json.load(f1)
    f1.close()

# Create parent folder
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

# Create JSON folder inside outfolder
jsonfolder = os.path.join(outfolder,'0_JSON_info')

if not os.path.exists(jsonfolder):
    os.makedirs(jsonfolder)

# Make sure severity distribution and infected state dist sums to 1.0
severity_dist = np.array(severity_data).T
infected_state_dist = np.array(infected_state_data).T[0]
severity_dist = ensure_row_sum(severity_dist,8)
infected_state_dist = ensure_vector_sum(infected_state_dist,8)

# Loop through runs and strategies, calculate inputs, and creating associated input files and folders
for run in run_list['run.name'].unique():

    # Create run folder
    runfolder = os.path.join(outfolder,run)
    if not os.path.exists(runfolder):
        os.makedirs(runfolder)

    for strategy in run_list[(run_list['run.name']==run)]['strategy'].unique():

        strategyfolder = os.path.join(runfolder,strategy)
        if not os.path.exists(strategyfolder):
            os.makedirs(strategyfolder)

        run_row = get_run_list_row_as_dict(run_list,run,strategy)
        age_vaccination_dist,max_uptake,remaining_supply,prob_A,VE_1vs0,VE_2vs0,vaccinations_per_day = get_vaccine_input_params(run_row)
        leftover_A,leftover_B,vaccine_allocation,additional_coverage = vaccinate_population(age_vaccination_dist,max_uptake,remaining_supply,prob_A)

        res = calculate_immunity_parameters(severity_dist,VE_1vs0,VE_2vs0,prob_A)

        prob_full_immunity_dose_1 = res[0]
        prob_full_immunity_dose_2 = res[1]
        partial_immunity_severity_dist_dose_1 = res[2]
        partial_immunity_severity_dist_dose_2 = res[3]
        dose_1_severity_dist = res[4]
        dose_2_severity_dist = res[5]

        # Round to 8 decimal places and ensure things which should sum to 1.0 still sum to 1.0
        prob_full_immunity_dose_1 = np.round(prob_full_immunity_dose_1,8)
        prob_full_immunity_dose_2 = np.round(prob_full_immunity_dose_2,8)
        partial_immunity_severity_dist_dose_1 = ensure_row_sum(partial_immunity_severity_dist_dose_1,8)
        partial_immunity_severity_dist_dose_2 = ensure_row_sum(partial_immunity_severity_dist_dose_2,8)

        time_intervals = calculate_rollout_intervals(vaccine_allocation,vaccinations_per_day,run_row['dose.1.delay'])

        # Calculate parameters needed to specify initial conditions
        unvaccinated_infected = run_row['proportion.infected.unvaccinated']
        unvaccinated_immune = run_row['proportion.immune.unvaccinated']
        unvaccinated_susceptible = 1 - unvaccinated_infected - unvaccinated_immune
        unvaccinated_SIR = np.array([unvaccinated_susceptible,unvaccinated_infected,unvaccinated_immune])

        tn_group_dist,age_dist_by_tn_group,prob_unvaccinated_naive,disease_dist = get_initial_conditions(age_vaccination_dist,unvaccinated_SIR,infected_state_dist,prob_full_immunity_dose_1,prob_full_immunity_dose_2)

        strat_row = run_row.copy()
        strat_row['doses.A.remaining'] = leftover_A
        strat_row['doses.B.remaining'] = leftover_B
        strat_row['dose.1.per.day'] = vaccinations_per_day
        strat_row['prob.naive.if.unvaccinated'] = prob_unvaccinated_naive

        vaccination_status_labels = ['unvaccinated','vaccinated']
        age_labels = ['0-19y','20-59y','>=60y']
        vaccine_labels = ['vaccine A','vaccine B']
        severity_labels = ['asymptomatic','mild/moderate','severe','critical']
        tn_group_labels = ['tn_group 0','tn_group 1','tn_group 2','tn_group 3']
        immune_status_labels = ['naive','recovered','vaccinated']
        disease_state_labels = ['susceptible','pre-infectious incubation','asymptomatic','mild/moderate','severe','critical','recuperation','immune']

        save_json = strat_row.copy()

        strat_row['covid.naive.severity.dist'] = severity_dist.tolist()
        strat_row['tn.group.dist'] = tn_group_dist.tolist()                    #
        strat_row['age.dist.by.tn.group'] = age_dist_by_tn_group.tolist()      #
        strat_row['disease.dist'] = disease_dist.tolist()                      #
        strat_row['age.vaccination.dist.at.start'] = age_vaccination_dist.tolist()
        strat_row['vaccine.allocation'] = vaccine_allocation.tolist()
        strat_row['additional.vaccine.coverage'] = additional_coverage.tolist()
        strat_row['prob.full.immunity.dose.1'] = prob_full_immunity_dose_1.tolist()
        strat_row['prob.full.immunity.dose.2'] = prob_full_immunity_dose_2.tolist()
        strat_row['partial.immunity.severity.dist.dose.1'] = partial_immunity_severity_dist_dose_1.tolist()
        strat_row['partial.immunity.severity.dist.dose.2'] = partial_immunity_severity_dist_dose_2.tolist()
        strat_row['time.intervals'] = time_intervals.tolist()

        # Update input file with run-specific info
        infile = template_file.copy()
        infile = tests_and_resources(infile,strat_row)
        infile = immunity_parameters(infile,strat_row)
        infile = initial_conditions(infile,strat_row)
        infile = transmission_multipliers(infile,strat_row)
        infile = contact_matrices(infile,strat_row)

        # Save a prettier version with labels for the user
        save_json['covid.naive.severity.dist'] = write_array_to_dict(age_labels,severity_labels,severity_dist.T)
        save_json['tn.group.dist'] = write_vector_to_dict(tn_group_labels,tn_group_dist)
        save_json['age.dist.by.tn.group'] = write_array_to_dict(tn_group_labels,age_labels,age_dist_by_tn_group)
        save_json['disease.dist'] = write_array_to_dict(immune_status_labels,disease_state_labels,disease_dist)
        save_json['age.vaccination.dist.at.start'] = write_array_to_dict(age_labels,vaccination_status_labels,age_vaccination_dist.T)
        save_json['vaccine.allocation'] = write_array_to_dict(age_labels,vaccine_labels,vaccine_allocation.T)
        save_json['additional.vaccine.coverage'] = write_array_to_dict(age_labels,vaccine_labels,additional_coverage.T)
        save_json['prob.full.immunity.dose.1'] = write_vector_to_dict(age_labels,prob_full_immunity_dose_1)
        save_json['prob.full.immunity.dose.2'] = write_vector_to_dict(age_labels,prob_full_immunity_dose_2)
        save_json['partial.immunity.severity.dist.dose.1'] = write_array_to_dict(age_labels,severity_labels,partial_immunity_severity_dist_dose_1.T)
        save_json['partial.immunity.severity.dist.dose.2'] = write_array_to_dict(age_labels,severity_labels,partial_immunity_severity_dist_dose_2.T)
        save_json['time.intervals'] = time_intervals.tolist()

        # Save input file
        infile_name = f'{run}_{strategy}_infile.json'
        outJSON = os.path.join(strategyfolder,infile_name)
        with open(outJSON, 'w') as f:
            json.dump(infile, f, indent=2)
            f.close()

        # Save info file
        info_name = f'{run}_{strategy}_info.json'
        outJSON = os.path.join(jsonfolder,info_name)
        with open(outJSON, 'w') as f:
            json.dump(save_json, f, indent=2)
            f.close()
