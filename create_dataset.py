import pandas as pd
import numpy as np


def read_in_data(dirpath=None):
    if dirpath != None:
        measurables = pd.read_csv(f"{dirpath}/Measurables.csv")
        production = pd.read_csv(f"{dirpath}/Production.csv")
    else:
        measurables = pd.read_csv("data/Measurables.csv")
        production = pd.read_csv("data/Production.csv")
    return measurables, production

'''
Create Career NFL Production Ranking 
''' 
def create_production_score(production_df, save=True):

    nfl_production = production_df.copy(deep=True)
    nfl_production['GP%'] = nfl_production['GamesPlayed']/(1*16)
    nfl_production['GS%'] = nfl_production['GamesStarted']/(1*16)
    nfl_production['PosPlay%'] = nfl_production['PositivePlays']/nfl_production['Plays']
    nfl_production['NegPlay%'] = nfl_production['NegativePlays']/nfl_production['Plays']
    nfl_production['NeutPlay%'] = (nfl_production['Plays'] - nfl_production['PositivePlays'] - nfl_production['NegativePlays'])/nfl_production['Plays']

    alpha_1 = 0.7
    alpha_2 = 0.3
    beta_1 = 2
    beta_2 = 0.3
    beta_3 = -2
    w1=0.2
    w2=0.8
    nfl_production['Season_Score'] = w1*(alpha_1*nfl_production['GP%'] + alpha_2*nfl_production['GS%']) + \
                                w2*(beta_1*nfl_production['PosPlay%'] + beta_2*nfl_production['NeutPlay%'] + beta_3*nfl_production['NegPlay%'])
    if save:
        nfl_production.groupby("PlayerId").sum().sort_values("Season_Score", ascending=False)[['Season_Score']].to_csv("model_data/nfl_desirability.csv")

    return nfl_production