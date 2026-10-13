"""Historical valuation approximation; no current-profile predictors."""
from pathlib import Path
import argparse,hashlib,json,urllib.request
import numpy as np,pandas as pd,joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
ROOT=Path(__file__).resolve().parents[2]
BASE='https://huggingface.co/datasets/ngeorgea/transfermarkt-player-scores/resolve/main/'
LEAGUES=['GB1','ES1','IT1','L1','FR1']
NUM=['age','games','minutes','goals_p90','assists_p90','yellow_p90','red_p90','minutes_game']
CAT=['league'];FEATURES=NUM+CAT

def download():
 raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
 for n in ['players','player_valuations','appearances','competitions']:
  dest=raw/(n+'.csv');tmp=raw/(n+'.tmp')
  if not dest.exists():
   req=urllib.request.Request(BASE+n+'.csv',headers={'User-Agent':'scouting-value-lab/1.0'})
   with urllib.request.urlopen(req,timeout=90) as r,tmp.open('wb') as f:
    while b:=r.read(1024*1024):f.write(b)
   tmp.replace(dest)

def annual_frame(app,values,players,year):
 cutoff=pd.Timestamp(year=year,month=6,day=1);end=cutoff+pd.Timedelta(days=90)
 past=app[(app.date<cutoff)&(app.date>=cutoff-pd.Timedelta(days=365))&app.competition_id.isin(LEAGUES)]
 g=past.groupby('player_id').agg(games=('game_id','nunique'),minutes=('minutes_played','sum'),goals=('goals','sum'),assists=('assists','sum'),yellow=('yellow_cards','sum'),red=('red_cards','sum'),last_game=('date','max'))
 last=past.sort_values(['date','game_id']).drop_duplicates('player_id',keep='last').set_index('player_id')
 g['league']=last.competition_id
 future=values[(values.date>=cutoff)&(values.date<end)].sort_values('date').drop_duplicates('player_id').set_index('player_id')
 prior=values[(values.date<cutoff)&(values.date>=cutoff-pd.Timedelta(days=365))].sort_values('date').drop_duplicates('player_id',keep='last').set_index('player_id')
 out=g.join(future[['date','market_value_in_eur']].rename(columns={'date':'target_date','market_value_in_eur':'target_eur'}),how='inner').join(prior[['date','market_value_in_eur']].rename(columns={'date':'prior_date','market_value_in_eur':'prior_eur'}),how='inner').join(players.set_index('player_id')[['name','date_of_birth','position','sub_position']])
 out['age']=(cutoff-out.date_of_birth).dt.days/365.2425
 out=out[(out.minutes>=900)&out.age.between(16,40)&(out.target_eur>0)&(out.prior_eur>0)].copy()
 for c in ['goals','assists','yellow','red']:out[c+'_p90']=90*out[c]/out.minutes
 out['minutes_game']=out.minutes/out.games;out['year']=year;out['cutoff']=cutoff
 assert (out.last_game<out.cutoff).all() and (out.prior_date<out.cutoff).all() and (out.target_date>=out.cutoff).all()
 return out.reset_index()

def build():
 raw=ROOT/'data/raw';p=pd.read_csv(raw/'players.csv',usecols=['player_id','name','date_of_birth','position','sub_position']);p.date_of_birth=pd.to_datetime(p.date_of_birth,errors='coerce')
 a=pd.read_csv(raw/'appearances.csv',usecols=['player_id','game_id','date','competition_id','minutes_played','goals','assists','yellow_cards','red_cards']);a.date=pd.to_datetime(a.date);assert not a.duplicated(['player_id','game_id']).any()
 v=pd.read_csv(raw/'player_valuations.csv',usecols=['player_id','date','market_value_in_eur']);v.date=pd.to_datetime(v.date);v=v.drop_duplicates(['player_id','date','market_value_in_eur']);assert not v.duplicated(['player_id','date']).any()
 d=pd.concat([annual_frame(a,v,p,y) for y in range(2016,2026)],ignore_index=True)
 dest=ROOT/'data/processed';dest.mkdir(parents=True,exist_ok=True);d.to_csv(dest/'player_years.csv',index=False)
 manifest={n:{'sha256':hashlib.sha256((raw/(n+'.csv')).read_bytes()).hexdigest(),'rows':sum(1 for _ in (raw/(n+'.csv')).open())-1} for n in ['players','appearances','player_valuations','competitions']}
 manifest['max_appearance_date']=str(a.date.max().date());manifest['max_valuation_date']=str(v.date.max().date());manifest['source']=BASE;manifest['max_used_target_date']=str(pd.to_datetime(d.target_date).max().date())
 (dest/'manifest.json').write_text(json.dumps(manifest,indent=2));return d

def preprocessor():
 return ColumnTransformer([('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),NUM),('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)],sparse_threshold=0)

def evaluate(y,p):
 return {'mae_eur':float(mean_absolute_error(y,p)),'rmse_eur':float(np.sqrt(mean_squared_error(y,p))),'mae_log':float(mean_absolute_error(np.log1p(y),np.log1p(p))),'median_abs_pct_error':float(np.median(np.abs(p-y)/y)*100)}

def train(d):
 train=d[d.year<=2022];val=d[d.year==2023];test=d[d.year>=2024];assert len(train) and len(val) and len(test)
 models={'ridge':make_pipeline(preprocessor(),Ridge(alpha=30)), 'hist_gradient_boosting':make_pipeline(preprocessor(),HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=15,l2_regularization=10,learning_rate=.07,random_state=42))}
 results={};target=np.log1p(train.target_eur)
 for name,m in models.items():
  m.fit(train[FEATURES],target);pred=np.maximum(0,np.expm1(m.predict(val[FEATURES])));results[name]=evaluate(val.target_eur.to_numpy(),pred)
 best=min(results,key=lambda n:results[n]['mae_log']);model=models[best]
 # Frozen pre-2023 model; validation also calibrates approximate log residual bands.
 vp=model.predict(val[FEATURES]);res=np.abs(np.log1p(val.target_eur.to_numpy())-vp);q=float(np.quantile(res,.8,method='higher'))
 lp=model.predict(test[FEATURES]);test=test.copy();test['estimate_eur']=np.maximum(0,np.expm1(lp));test['band_low_eur']=np.maximum(0,np.expm1(lp-q));test['band_high_eur']=np.expm1(lp+q)
 coverage=float(((test.target_eur>=test.band_low_eur)&(test.target_eur<=test.band_high_eur)).mean())
 med=float(train.target_eur.median());metrics={'selected_model':best,'train_years':'2016–2022','validation_years':'2023','test_years':'2024–2025','n_train':len(train),'n_validation':len(val),'n_test':len(test),'validation':results,'test_model':evaluate(test.target_eur.to_numpy(),test.estimate_eur.to_numpy()),'test_persistence':evaluate(test.target_eur.to_numpy(),test.prior_eur.to_numpy()),'test_global_median':evaluate(test.target_eur.to_numpy(),np.repeat(med,len(test))),'band_nominal':.8,'test_band_coverage':coverage,'test_band_median_width_eur':float((test.band_high_eur-test.band_low_eur).median()),'repeat_player_overlap_pct':float(test.player_id.isin(train.player_id).mean()*100)}
 unseen=test[~test.player_id.isin(train.player_id)]
 metrics['test_unseen_players_n']=len(unseen)
 metrics['test_unseen_players']=evaluate(unseen.target_eur.to_numpy(),unseen.estimate_eur.to_numpy())
 # Age+league baseline without sporting production, frozen independently.
 simple=make_pipeline(ColumnTransformer([('age',make_pipeline(SimpleImputer(),StandardScaler()),['age']),('league',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['league'])]),Ridge(alpha=30));simple.fit(train[['age','league']],target)
 metrics['test_age_league']=evaluate(test.target_eur.to_numpy(),np.maximum(0,np.expm1(simple.predict(test[['age','league']]))))
 art=ROOT/'artifacts';art.mkdir(exist_ok=True);joblib.dump({'model':model,'q':q,'features':FEATURES},art/'model.joblib');test.to_csv(art/'test_predictions.csv',index=False);(art/'metrics.json').write_text(json.dumps(metrics,indent=2))
 slices=[]
 for league,g in test.groupby('league'):slices.append({'league':league,'n':len(g),**evaluate(g.target_eur.to_numpy(),g.estimate_eur.to_numpy())})
 pd.DataFrame(slices).to_csv(art/'metrics_by_league.csv',index=False);print(json.dumps(metrics,indent=2));return metrics

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');args=parser.parse_args()
 if args.download:download()
 train(build())
if __name__=='__main__':main()
