from pathlib import Path
import json,pandas as pd,streamlit as st
from scouting_value_lab.similarity import comparable_players
R=Path(__file__).parent
st.set_page_config(page_title='Scouting Value Lab',page_icon='⚽',layout='wide')
st.title('⚽ Scouting Value Lab')
st.caption('¿Cuánto vale un futbolista? Explorador histórico de valoraciones publicadas, no precios de traspaso.')
st.warning('Las estimaciones no son precios justos. Una diferencia con el modelo no demuestra infravaloración. Datos históricos; no cotizaciones actuales.')
@st.cache_data
def load():return pd.read_csv(R/'artifacts/test_predictions.csv'),json.loads((R/'artifacts/metrics.json').read_text())
try:d,m=load()
except FileNotFoundError:st.error('Ejecuta primero: python -m scouting_value_lab.pipeline --download');st.stop()
view,validation,method=st.tabs(['Explorar jugadores','Validación temporal','Metodología'])
with view:
 year=st.selectbox('Fecha de corte: 1 de junio',sorted(d.year.unique()),index=len(d.year.unique())-1)
 q=d[d.year==year].copy();q['label']=q.name+' · '+q.league+' · ID '+q.player_id.astype(str)
 ids=q.sort_values('label').player_id.tolist();by=q.set_index('player_id');pid=st.selectbox('Jugador',ids,format_func=lambda x:by.loc[x,'label']);r=by.loc[pid]
 a,b,c=st.columns(3);a.metric('Valoración posterior publicada',f"€{r.target_eur/1e6:.2f} M");b.metric('Estimación del modelo',f"€{r.estimate_eur/1e6:.2f} M");c.metric('Última valoración previa',f"€{r.prior_eur/1e6:.2f} M")
 st.caption(f"Corte: {r.cutoff[:10]} · valoración observada: {r.target_date[:10]} · previa: {r.prior_date[:10]} · banda aproximada: €{r.band_low_eur/1e6:.2f}–{r.band_high_eur/1e6:.2f} M")
 st.dataframe(pd.DataFrame({'Indicador':['Edad al corte','Minutos previos','Goles /90','Asistencias /90'],'Valor':[r.age,r.minutes,r.goals_p90,r.assists_p90]}),hide_index=True)
 st.subheader('Comparables por perfil básico')
 st.caption('Misma posición amplia del snapshot y año. Distancia menor = perfil más parecido; no porcentaje de compatibilidad. No mide técnica, defensa, xG ni adecuación táctica.')
 peers=comparable_players(q,pid);st.dataframe(peers[['name','league','age','minutes','goals_p90','assists_p90','target_eur','profile_distance']],hide_index=True,width='stretch')
 st.download_button('Descargar comparables CSV',peers.to_csv(index=False),'comparables.csv','text/csv')
with validation:
 st.write(f"Entrenamiento: {m['train_years']} · selección/calibración: {m['validation_years']} · prueba: {m['test_years']}")
 results=pd.DataFrame({k:m[k] for k in ['test_model','test_persistence','test_age_league','test_global_median']}).T
 st.dataframe(results.rename(columns={'mae_eur':'MAE €','mae_log':'MAE log','rmse_eur':'RMSE €','median_abs_pct_error':'Mediana error %'}))
 st.write(f"Modelo elegido: {m['selected_model']}. Cobertura de la banda nominal 80% en prueba: {m['test_band_coverage']:.1%}.")
 st.info('Persistencia utiliza la valoración anterior. El modelo principal no recibe ninguna valoración como predictor; representan preguntas distintas y ambas referencias se reportan.')
 st.image(str(R/'artifacts/evaluation.png'))
with method:
 st.markdown((R/'docs/MODEL_CARD.md').read_text())
