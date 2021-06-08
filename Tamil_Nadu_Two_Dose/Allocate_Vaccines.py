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

def vaccinate_population(age_dist,uptake,supply,prob_A):
    """
    This function allocates vaccines across a structured
    population while prioritizing older age groups
    """

    m = 2             # Number of vaccines
    n = len(age_dist) # Number of age groups

    # Rows correspond to different vaccines (A,B)
    # Cols correspond to different age groups (0-19,20-59,60+)

    vaccine_allocation = np.zeros((m,n))
    vaccine_coverage = np.zeros((m,n))

    for j in np.flip(np.arange(n)):
        supply,xA,xB = vaccinate_subpopulation(age_dist[j],uptake,supply,prob_A)
        vaccine_allocation[0,j] = xA
        vaccine_allocation[1,j] = xB

    vaccine_coverage = vaccine_allocation/age_dist

    leftover_A = supply*prob_A
    leftover_B = supply*(1-prob_A)

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

    for i in range(5):         # Immune state
        for j in range(3):     # Age
            for v in range(2): # Vaccine

                # Efficacy of vaccine v
                VE1,VE2,VE3,VE4 = VE_1vs0[v,:]

                # Severity distribution of someone age j
                P0,P1,P2,P3,P4 = unvaccinated_severity_dist[:,j]

                # Probability of immune state i after vaccination
                # among those age j receiving vaccine v
                f = severity_reduction(VE1,VE2,VE3,VE4,P0,P1,P2,P3,P4)

                # Multiply by probability that someone age j
                # receives vaccine v, and add to new severity dist
                dose_1_severity_dist[i,j] += P_vaccine[v]*f[i]

    # *** Second dose *** #

    for i in range(5):         # Immune state
        for j in range(3):     # Age
            for v in range(2): # Vaccine

                # Efficacy of vaccine v
                VE1,VE2,VE3,VE4 = VE_2vs1[v,:]

                # Severity distribution of someone age i
                P0,P1,P2,P3,P4 = dose_1_severity_dist[:,j]

                # Probability of immune state i after vaccination
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
        for j in range(3):
            i_mod = i+1
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

def get_run_list_row_as_dict(run_list,run):
    """
    Returns the row corresponding to a given run as a dictionary.
    """
    run_row = run_list[(run_list['run.name']==run)].to_dict('records')[0]
    return(run_row)

def get_vaccine_input_params(run_row):

    uptake = run_row['uptake']
    prob_A = run_row['prob.vaccine.A']
    supply = run_row['target.coverage']

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

    # I hard-coded these,
    # but if we decide to vary this I'll create an input for it

    severity_dist = np.array([[0.299342,0.179000,0.171045],
                          [0.697833,0.803755,0.763740],
                          [0.002495,0.007952,0.013583],
                          [0.00033,0.009293,0.051632]])

    # Age dist for Tamil Nadu
    age_dist = np.array([0.32271396,0.57309993,0.10418611])

    vaccinations_per_day = supply/run_row['time.to.coverage']

    return(age_dist,severity_dist,uptake,supply,prob_A,VE_1vs0,VE_2vs0,vaccinations_per_day)

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

    for i in range(3):
        infile['initial state']['immune states dist'][f'for tn_group {i}']['recovered'] = np.round(strat_row['proportion.immune'],8)
        infile['initial state']['immune states dist'][f'for tn_group {i}']['naive'] = 1-np.round(strat_row['proportion.immune'],8)

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
        infile['transmissions'][f'intervention {i}']['transmission rate multipliers']['for day# > t3'] = strat_row['baseline.transmission.multiplier']

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

# Main part of script

pwd=os.getcwd()

name = 'test_run_2021-06-08'
run_list_path = os.path.join(pwd,'run_dictionary',name+'.xlsx')
outfolder = os.path.join(pwd,name) # Filepath for output

run_list = pd.read_excel(run_list_path)

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

for run in run_list['run.name'].unique():

    # Create run folder
    runfolder = os.path.join(outfolder,run)
    if not os.path.exists(runfolder):
        os.makedirs(runfolder)

    run_row = get_run_list_row_as_dict(run_list,run)
    age_dist,severity_dist,uptake,supply,prob_A,VE_1vs0,VE_2vs0,vaccinations_per_day = get_vaccine_input_params(run_row)
    leftover_A,leftover_B,vaccine_allocation,vaccine_coverage = vaccinate_population(age_dist,uptake,supply,prob_A)

    res = calculate_immunity_parameters(severity_dist,VE_1vs0,VE_2vs0,prob_A)

    prob_full_immunity_dose_1 = res[0]
    prob_full_immunity_dose_2 = res[1]
    partial_immunity_severity_dist_dose_1 = res[2]
    partial_immunity_severity_dist_dose_2 = res[3]
    dose_1_severity_dist = res[4]
    dose_2_severity_dist = res[5]

    # Round to 8 decimal places, and ensure that it still sums to 1
    prob_full_immunity_dose_1 = np.round(prob_full_immunity_dose_1,8)
    prob_full_immunity_dose_2 = np.round(prob_full_immunity_dose_2,8)
    partial_immunity_severity_dist_dose_1 = np.round(partial_immunity_severity_dist_dose_1,8)
    partial_immunity_severity_dist_dose_1[0,:] = np.round(1 - np.sum(partial_immunity_severity_dist_dose_1[1:,:],0),8)
    partial_immunity_severity_dist_dose_2 = np.round(partial_immunity_severity_dist_dose_2,8)
    partial_immunity_severity_dist_dose_2[0,:] = np.round(1 - np.sum(partial_immunity_severity_dist_dose_2[1:,:],0),8)

    time_intervals = calculate_rollout_intervals(vaccine_allocation,vaccinations_per_day,run_row['dose.1.delay'])

    strat_row = run_row.copy()
    strat_row['doses.A.remaining'] = leftover_A
    strat_row['doses.B.remaining'] = leftover_B
    strat_row['dose.1.per.day'] = vaccinations_per_day

    age_labels = ['for 0-19y','for 20-59y','for >=60y']
    vaccine_labels = ['vaccine A','vaccine B']
    severity_labels = ['asymptomatic','mild/moderate','severe','critical']

    save_json = strat_row.copy()

    strat_row['vaccine.allocation'] = vaccine_allocation.tolist()
    strat_row['vaccine.coverage'] = vaccine_coverage.tolist()
    strat_row['prob.full.immunity.dose.1'] = prob_full_immunity_dose_1.tolist()
    strat_row['prob.full.immunity.dose.2'] = prob_full_immunity_dose_2.tolist()
    strat_row['partial.immunity.severity.dist.dose.1'] = partial_immunity_severity_dist_dose_1.tolist()
    strat_row['partial.immunity.severity.dist.dose.2'] = partial_immunity_severity_dist_dose_2.tolist()
    strat_row['time.intervals'] = time_intervals.tolist()

    # Update input file with run-specific info
    infile = template_file.copy()
    infile = tests_and_resources(infile,strat_row)
    infile = immunity_parameters(infile,strat_row)
    infile = transmission_multipliers(infile,strat_row)

    # Save a prettier version with labels for the user
    save_json['vaccine.allocation'] = write_array_to_dict(age_labels,vaccine_labels,vaccine_allocation.T)
    save_json['vaccine.coverage'] = write_array_to_dict(age_labels,vaccine_labels,vaccine_coverage.T)
    save_json['prob.full.immunity.dose.1'] = write_vector_to_dict(age_labels,prob_full_immunity_dose_1)
    save_json['prob.full.immunity.dose.2'] = write_vector_to_dict(age_labels,prob_full_immunity_dose_2)
    save_json['partial.immunity.severity.dist.dose.1'] = write_array_to_dict(age_labels,severity_labels,partial_immunity_severity_dist_dose_1.T)
    save_json['partial.immunity.severity.dist.dose.2'] = write_array_to_dict(age_labels,severity_labels,partial_immunity_severity_dist_dose_2.T)
    save_json['time.intervals'] = time_intervals.tolist()

    # Save input file
    infile_name = f'{run}_infile.json'
    outJSON = os.path.join(runfolder,infile_name)
    with open(outJSON, 'w') as f:
        json.dump(infile, f, indent=2)
        f.close()

    # Save info file
    info_name = f'{run}_info.json'
    outJSON = os.path.join(jsonfolder,info_name)
    with open(outJSON, 'w') as f:
        json.dump(save_json, f, indent=2)
        f.close()
