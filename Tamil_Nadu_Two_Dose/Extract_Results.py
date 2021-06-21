#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import json


# In[5]:


def json_name(run):
    return(f'{run}_info.json')

def get_outcomes(dfA,dfB,dfC,strat_row):

    nday = len(dfA)
    t = np.arange(nday)

    hospital_occupancy = np.array(dfA['resource untilization 1'])
    ICU_occupancy = np.array(dfA['resource untilization 2'])
    hospital_days = np.cumsum(hospital_occupancy)
    ICU_days = np.cumsum(ICU_occupancy)

    daily_new_infections_0_19 = np.array(dfA['tn_group 0 new infections'])
    daily_new_infections_20_59 = np.array(dfA['tn_group 1 new infections'])
    daily_new_infections_60 = np.array(dfA['tn_group 2 new infections'])
    daily_new_infections = daily_new_infections_0_19+daily_new_infections_20_59+daily_new_infections_60

    cumulative_new_infections_0_19 = np.cumsum(daily_new_infections_0_19)
    cumulative_new_infections_20_59 = np.cumsum(daily_new_infections_20_59)
    cumulative_new_infections_60 = np.cumsum(daily_new_infections_60)
    cumulative_new_infections = np.cumsum(daily_new_infections)

    cumulative_mortality_0_19 = np.cumsum(np.array(dfA['mortality for 0-19y']))
    cumulative_mortality_20_59 = np.cumsum(np.array(dfA['mortality for 20-59y']))
    cumulative_mortality_60 = np.cumsum(np.array(dfA['mortality for >=60y']))
    cumulative_mortality = np.array(dfA['dead'])

    cumulative_vaccinations_0_19 = np.cumsum(np.array(dfA['test 2 (+)']))
    cumulative_vaccinations_20_59 = np.cumsum(np.array(dfA['test 1 (+)']))
    cumulative_vaccinations_60 = np.cumsum(np.array(dfA['test 0 (+)']))
    cumulative_vaccinations = cumulative_vaccinations_0_19+cumulative_vaccinations_20_59+cumulative_vaccinations_60

    hospital_costs = hospital_days*strat_row['cost.per.hospital.day']
    ICU_costs = ICU_days*strat_row['cost.per.ICU.day']
    # When calculating costs, be careful to note whether you're specifying cost per dose or cost per person vaccinated
    vaccine_costs = np.ones(nday)*(strat_row['prob.vaccine.A']*strat_row['cost.per.dose.A'] + (1-strat_row['prob.vaccine.A'])*strat_row['cost.per.dose.B'])*2*strat_row['target.coverage']*strat_row['sim.size']

    total_costs = hospital_costs + ICU_costs + vaccine_costs

    YLL = cumulative_mortality_0_19*strat_row['YLL.0.to.19'] + cumulative_mortality_20_59*strat_row['YLL.20.to.59'] + cumulative_mortality_60*strat_row['YLL.60.plus']

    run_name = np.array([strat_row['run.name']]*nday)

    data = {'run':run_name,
            'day':t,
            'cumulative.vaccinations.0.to.19':cumulative_vaccinations_0_19,
            'cumulative.vaccinations.20.to.59':cumulative_vaccinations_20_59,
            'cumulative.vaccinations.60.plus':cumulative_vaccinations_60,
            'cumulative.vaccinations':cumulative_vaccinations,
            'daily.new.infections.0.to.19':daily_new_infections_0_19,
            'daily.new.infections.20.to.59':daily_new_infections_20_59,
            'daily.new.infections.60.plus':daily_new_infections_60,
            'daily.new.infections':daily_new_infections,
            'cumulative.new.infections.0.to.19':cumulative_new_infections_0_19,
            'cumulative.new.infections.20.to.59':cumulative_new_infections_20_59,
            'cumulative.new.infections.60.plus':cumulative_new_infections_60,
            'cumulative.new.infections':cumulative_new_infections,
            'cumulative.mortality.0.to.19':cumulative_mortality_0_19,
            'cumulative.mortality.20.to.59':cumulative_mortality_20_59,
            'cumulative.mortality.60.plus':cumulative_mortality_60,
            'cumulative.mortality':cumulative_mortality,
            'hospital.occupancy':hospital_occupancy,
            'ICU.occupancy':ICU_occupancy,
            'hospital.days':hospital_days,
            'ICU.days':ICU_days,
            'hospital.costs':hospital_costs,
            'ICU.costs':ICU_costs,
            'vaccine.costs':vaccine_costs,
            'total.costs':total_costs,
            'YLL':YLL}

    return(data)

pwd = os.getcwd()
parent_directory = os.path.join(pwd,'test_run_2021-06-08_500k')
runs = ['no_vaccination',
        '30p_in_6m_lumped',
        '30p_in_6m_12wk_interval']

dflist = []

for run in runs:

    info_path = os.path.join(parent_directory,'0_JSON_info',json_name(run))
    with open(info_path) as f1:
        strat_row = json.load(f1)
        f1.close()

    # Comment this out later - first runs didnt have this
    strat_row['cost.per.dose.A'] = 14.81
    strat_row['cost.per.dose.B'] = 14.81
    strat_row['cost.per.hospital.day'] = 154.31
    strat_row['cost.per.ICU.day'] = 1750.97
    strat_row['YLL.0.to.19'] = 59.00
    strat_row['YLL.20.to.59'] = 25.79
    strat_row['YLL.60.plus'] = 10.94

    pathA = os.path.join(parent_directory,run,'results',f'{run}_infile.tsv')
    pathB = os.path.join(parent_directory,run,'results',f'{run}_infile_state_data.tsv')
    pathC = os.path.join(parent_directory,run,'results',f'{run}_infile_vaccine_data.tsv')

    dfA = pd.read_csv(pathA, sep='\t')
    dfB = pd.read_csv(pathB, sep='\t')
    dfC = pd.read_csv(pathC, sep='\t')


    data = get_outcomes(dfA,dfB,dfC,strat_row)
    temp = pd.DataFrame.from_dict(data)


    dflist.append(temp.copy())

bigdf = pd.concat(dflist)

outname = os.path.join(parent_directory,'bigdataframe.csv')
bigdf.to_csv(outname)

outcome_dict = {'cumulative.vaccinations.0.to.19':'Cumulative vaccinations among those <20y, n',
        'cumulative.vaccinations.20.to.59':'Cumulative vaccinations among those 20-59y, n',
        'cumulative.vaccinations.60.plus':'Cumulative vaccinations among those ≥60y, n',
        'cumulative.vaccinations':'Cumulative vaccinations, n',
        'daily.new.infections.0.to.19':'Daily new infections among those <20y, n',
        'daily.new.infections.20.to.59':'Daily new infections among those 20-59y, n',
        'daily.new.infections.60.plus':'Daily new infections among those ≥60y, n',
        'daily.new.infections':'Daily new infections, n',
        'cumulative.new.infections.0.to.19':'Cumulative new infections among those <20y, n',
        'cumulative.new.infections.20.to.59':'Cumulative new infections among those 20-59y, n',
        'cumulative.new.infections.60.plus':'Cumulative new infections among those ≥60y, n',
        'cumulative.new.infections':'Cumulative new infections, n',
        'cumulative.mortality.0.to.19':'Cumulative mortality among those <20y, n',
        'cumulative.mortality.20.to.59':'Cumulative mortality among those 20-59y, n',
        'cumulative.mortality.60.plus':'Cumulative mortality among those ≥60y, n',
        'cumulative.mortality':'Cumulative mortality, n',
        'hospital.occupancy':'Daily hospital occupancy, n',
        'ICU.occupancy':'Daily ICU occupancy, n',
        'hospital.days':'Cumulative hospital bed-days, n',
        'ICU.days':'Cumulative ICU bed-days, n'}

outcomes = list(outcome_dict.keys())

ft=12

for run in runs:

    temp = bigdf[bigdf['run']==run]

    for outcome in outcomes:

        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(7,5))
        yorder = []

        y = np.array(temp[outcome])
        t = np.arange(len(y))
        ax.plot(t,y,linewidth=2.5,alpha=1.0)
        #yorder.append(np.max(y))

        #handles, labels = ax.get_legend_handles_labels()
        #myorder = np.argsort(-1*np.array(yorder))

        #ax.legend([handles[idx] for idx in myorder],[labels[idx] for idx in myorder],bbox_to_anchor=(0.0, 1.4),loc='upper left')
        ax.set_xlabel('Time, days',fontsize=ft)
        ax.set_xlim(0,360)
        ax.set_ylim(0,None)
        ax.set_xticks(np.arange(0,360+1,30))
        ax.set_ylabel(outcome_dict[outcome],fontsize=ft)
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        fig.tight_layout()
        outname=os.path.join(parent_directory,run,f'{outcome}.png')
        fig.savefig(outname,dpi=400,bbox_inches = 'tight',pad_inches = 0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
