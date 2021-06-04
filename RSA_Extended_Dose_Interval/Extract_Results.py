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


def json_name(run,strategy):
    return(f'{run}_strategy_{strategy}_info.json')

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
    vaccine_costs = np.ones(nday)*(strat_row['cost.per.dose.A']*strat_row['supply.A'] + strat_row['cost.per.dose.B']*strat_row['supply.B'])*strat_row['sim.size']
    
    # Don't count the cost of vaccines in the no vaccination strategy
    if strat_row['strategy'] == 1:
        vaccine_costs = np.zeros(nday)
    
    total_costs = hospital_costs + ICU_costs + vaccine_costs
    
    YLL = cumulative_mortality_0_19*strat_row['YLL.0.to.19'] + cumulative_mortality_20_59*strat_row['YLL.20.to.59'] + cumulative_mortality_60*strat_row['YLL.60.plus']
    
    run_name = np.array([strat_row['run.name']]*nday)
    strategy_name = np.array([strat_row['strategy']]*nday,dtype=int)
    
    
    data = {'run':run_name,
            'strategy':strategy_name,
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


# In[ ]:





# In[ ]:





# In[7]:


pwd = os.getcwd()
parent_directory = os.path.join(pwd,'2021-05-21')
runs = ['1x_supply_bc',
'2x_supply_bc',
'1x_supply_pace_100k',
'2x_supply_pace_100k',
'1x_supply_pace_50k',
'2x_supply_pace_50k',
'1x_supply_eff_40',
'2x_supply_eff_40',
'1x_supply_eff_60',
'2x_supply_eff_60',
'1x_supply_eff_80',
'2x_supply_eff_80',
'1x_supply_Re_11',
'2x_supply_Re_11',
'1x_supply_Re_12',
'2x_supply_Re_12',
'1x_supply_Re_18',
'2x_supply_Re_18']

strategies = ['1','2','3','4','5']

dflist = []

for run in runs:
    for strategy in strategies:
        
        info_path = os.path.join(parent_directory,'0_JSON_info',json_name(run,strategy))
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
            
        pathA = os.path.join(parent_directory,run,strategy,'results',f'{run}_strategy_{strategy}.tsv')
        pathB = os.path.join(parent_directory,run,strategy,'results',f'{run}_strategy_{strategy}_state_data.tsv')
        pathC = os.path.join(parent_directory,run,strategy,'results',f'{run}_strategy_{strategy}_vaccine_data.tsv')
        
        dfA = pd.read_csv(pathA, sep='\t')
        dfB = pd.read_csv(pathB, sep='\t')
        dfC = pd.read_csv(pathC, sep='\t')
    
        
        data = get_outcomes(dfA,dfB,dfC,strat_row)
        temp = pd.DataFrame.from_dict(data)
        
        
        dflist.append(temp.copy())
        
bigdf = pd.concat(dflist)

outname = os.path.join(parent_directory,'bigdataframe.csv')
bigdf.to_csv(outname)


# In[31]:


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

strategy_label_dict = {'1':'No vaccination',
                       '2':'2 dose Pfizer or 1 dose J&J for all ages',
                       '3':'1 dose Pfizer or 1 dose J&J for all ages',
                       '4':'2 dose Pfizer if ≥60y, 2 dose Pfizer or 1 dose J&J if <60y',
                       '5':'2 dose Pfizer if ≥60y, 1 dose Pfizer or 1 dose J&J if <60y'}

ft=12

for run in runs: 
    
    temp = bigdf[bigdf['run']==run]
    
    for outcome in outcomes:
        
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(7,5))
        yorder = []
        
        for strategy in [2,3,4,5]:
    
            t = np.arange(360)
            y = np.array(temp[temp['strategy']==strategy][outcome])/1e6*58775021
            ax.plot(t,y,label=strategy_label_dict[str(strategy)],linewidth=2.5,alpha=1.0)
            yorder.append(np.max(y))
    
        handles, labels = ax.get_legend_handles_labels()
        myorder = np.argsort(-1*np.array(yorder))

        ax.legend([handles[idx] for idx in myorder],[labels[idx] for idx in myorder],bbox_to_anchor=(0.0, 1.4),loc='upper left')
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




