from pathlib import Path
import json,pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2]
def main():
 m=json.loads((R/'artifacts/metrics.json').read_text());d=pd.read_csv(R/'artifacts/test_predictions.csv')
 bg='#120f28';white='#faf9ff';cyan='#72d6f1';pink='#ff438e';grey='#b7aec8'
 plt.rcParams.update({'font.family':'DejaVu Sans','text.color':white})
 f=plt.figure(figsize=(10.8,13.5),dpi=140,facecolor=bg)
 f.text(.06,.95,'SCOUTING VALUE LAB  /  JAIRO QUELAL',fontsize=12,color=cyan,weight='bold')
 f.text(.06,.89,'¿CUÁNTO VALE REALMENTE\nUN FUTBOLISTA?',fontsize=27,weight='bold',va='top')
 f.text(.06,.79,'Machine Learning sobre valoraciones históricas publicadas\nNo estima un precio justo ni el importe de un traspaso.',fontsize=12,color=grey,va='top')
 f.text(.06,.713,'LA ÚLTIMA VALORACIÓN ES DIFÍCIL DE SUPERAR',fontsize=14,color=cyan,weight='bold')
 ax=f.add_axes([.29,.43,.64,.22],facecolor=bg)
 keys=['test_global_median','test_age_league','test_model','test_persistence'];labels=['Mediana global','Edad + liga','ML deportivo','Valoración previa'];vals=[m[k]['mae_eur']/1e6 for k in keys]
 ax.barh(range(4),vals,color=[grey,grey,pink,cyan]);ax.set_yticks(range(4),labels,color=white,fontsize=11);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.25);ax.tick_params(colors=grey);ax.set_xlabel('Error absoluto medio · millones de euros · menor es mejor',color=grey,fontsize=10)
 for i,v in enumerate(vals):ax.text(v+.15,i,f'€{v:.2f} M'.replace('.',','),va='center',fontsize=12,weight='bold')
 for sp in ax.spines.values():sp.set_visible(False)
 f.text(.06,.365,'VALIDACIÓN POR TIEMPO, SIN MEZCLAR EL FUTURO',fontsize=14,color=cyan,weight='bold')
 f.text(.06,.325,f"Entrenamiento: 2016–2022 · {m['n_train']:,} registros\nSelección y bandas: 2023 · {m['n_validation']:,} registros\nPrueba final: 2024–2025 · {m['n_test']:,} registros".replace(',', '.'),fontsize=12,va='top',linespacing=1.6)
 f.text(.06,.228,'El modelo usa edad, liga y producción de los 365 días anteriores.\nLa referencia de persistencia usa la valoración anterior: aporta\ninformación que el modelo deportivo no recibe.',fontsize=11,color=grey,va='top')
 f.text(.06,.15,f"Banda aproximada nominal 80%: cobertura real {m['test_band_coverage']:.1%}.\nMinutos, goles y asistencias no explican por sí solos el mercado.\nUna diferencia modelo–valoración no demuestra infravaloración.",fontsize=10,color=grey,va='top')
 f.text(.06,.071,'Cinco grandes ligas · ≥900 min previos · jugadores con valoración previa y posterior\nCorte: 1 junio; objetivo: primera valoración en los 90 días siguientes.\nFuente: Transfermarkt vía dcaribou / mirror ngeorgea. Snapshot; no datos en vivo.',fontsize=8.5,color=grey,va='top')
 f.text(.06,.028,'Código reproducible · dashboard · comparables por perfil básico · evaluación temporal',fontsize=9,color=cyan)
 f.savefig(R/'artifacts/evaluation.png',facecolor=bg);plt.close(f)
 (R/'docs/RESULTS.md').write_text('# Resultados medidos\n\n'+'```text\n'+pd.DataFrame({k:m[k] for k in keys}).T.to_string()+'\n```'+'\n\nNo se seleccionó el modelo en prueba. Mejor modelo según MAE log de validación: '+m['selected_model']+'\n')
if __name__=='__main__':main()
