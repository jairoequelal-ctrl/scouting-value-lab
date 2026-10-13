import pandas as pd
from scouting_value_lab.pipeline import annual_frame,FEATURES
from scouting_value_lab.similarity import comparable_players

def test_future_appearances_and_prior_dates():
 p=pd.DataFrame({'player_id':[1],'name':['A'],'date_of_birth':[pd.Timestamp('2000-01-01')],'position':['Attack'],'sub_position':['Forward']})
 a=pd.DataFrame({'player_id':[1,1],'game_id':[1,2],'date':pd.to_datetime(['2022-05-01','2022-06-02']),'competition_id':['GB1','GB1'],'minutes_played':[1000,90],'goals':[2,99],'assists':[3,99],'yellow_cards':[1,0],'red_cards':[0,0]})
 v=pd.DataFrame({'player_id':[1,1,1],'date':pd.to_datetime(['2022-05-01','2022-06-01','2022-08-01']),'market_value_in_eur':[100,200,300]})
 d=annual_frame(a,v,p,2022);assert len(d)==1;assert d.iloc[0].goals==2;assert d.iloc[0].target_eur==200;assert d.iloc[0].prior_eur==100
 assert all('value' not in x and 'prior' not in x and 'target' not in x and 'current' not in x for x in FEATURES)

def test_comparables_exclude_identity_and_other_position():
 d=pd.DataFrame({'player_id':[1,2,3,4],'position':['Attack','Attack','Attack','Defender'],'age':[20,21,22,20],'minutes':[1000,1100,1200,1000],'goals_p90':[.5,.4,.3,.5],'assists_p90':[.2,.3,.1,.2],'minutes_game':[80,85,70,80]})
 r=comparable_players(d,1);assert set(r.player_id)=={2,3};assert r.profile_distance.ge(0).all()
