import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
COLS=['age','minutes','goals_p90','assists_p90','minutes_game']
def comparable_players(frame,player_id,n=8):
 if frame.player_id.duplicated().any():raise ValueError('Choose one year: player identities must be unique')
 target=frame[frame.player_id==player_id]
 if len(target)!=1:raise ValueError('Target must appear exactly once')
 row=target.iloc[0];peers=frame[(frame.player_id!=player_id)&(frame.position==row.position)].copy()
 if len(peers)<2:return peers
 scaler=StandardScaler().fit(peers[COLS]);x=scaler.transform(peers[COLS]);t=scaler.transform(target[COLS])[0]
 peers['profile_distance']=np.linalg.norm(x-t,axis=1)/np.sqrt(len(COLS))
 return peers.sort_values('profile_distance').head(n)
