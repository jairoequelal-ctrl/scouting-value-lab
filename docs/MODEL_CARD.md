# Modelo y límites

## Pregunta
Aproximar la primera valoración de mercado publicada por Transfermarkt en los 90 días a partir del 1 de junio de cada año, con producción deportiva registrada en los 365 días anteriores. No estima traspasos reales ni valor económico intrínseco. No clasifica jugadores como infravalorados.

## Cohorte
Cinco ligas GB1, ES1, IT1, L1, FR1; ≥900 minutos de liga en el año anterior; edad 16–40; valoración positiva previa en el año anterior y posterior en el horizonte. Una observación jugador-año. Liga = última aparición doméstica antes del corte; no liga actual del perfil. Las participaciones en varias ligas se agregan. Es una cohorte seleccionada de jugadores activos con actualizaciones de valoración; no representa todo el mercado ni suplentes con pocos minutos.

## Variables
Edad calculada desde fecha de nacimiento; partidos, minutos, goles/90, asistencias/90, tarjetas/90 y minutos/partido de apariciones estrictamente anteriores al corte. Solo usa producción básica, no xG, defensa detallada, lesiones o contratos históricos. No incluye nombre, nacionalidad, club actual, internacionalidades actuales, valor máximo ni valor actual como predictores. La valoración previa solo se usa como referencia independiente, no entra en el modelo.

La posición de players.csv es una etiqueta de snapshot: únicamente filtra comparables en el explorador; NO entra en entrenamiento. Las posiciones pueden haber cambiado históricamente. Las estadísticas anteriores se reconstruyen de registros publicados en el snapshot actual; fechas efectivas de acciones/valoraciones no prueban que esos registros estuvieran disponibles sin revisiones en aquel momento. Es un backtest retrospectivo, no una certificación point-in-time.

## Evaluación
Entrenamiento 2016–2022, selección 2023, prueba 2024–2025. Se comparan Ridge y HistGradientBoosting con parámetros fijos; selección por MAE de log(1+valor) en 2023. Se conserva el modelo entrenado hasta 2022 para prueba, sin reajuste. Referencias: mediana global de entrenamiento, Ridge edad+liga y persistencia (última valoración previa). La persistencia tiene información de mercado que no recibe el modelo deportivo.

MAE y RMSE en euros se reportan junto a MAE log y mediana de error porcentual. Muchos jugadores aparecen en entrenamiento y prueba (porcentaje en metrics.json): evalúa nuevos períodos para una mezcla de jugadores conocidos y nuevos, no generalización exclusiva a identidades inéditas. El snapshot está limitado; no hay datos en vivo.

## Bandas
Percentil 80 de residuos absolutos log en validación, reutilizada para selección; se transforma a euros alrededor de la predicción. Es una banda empírica aproximada. Dependencia temporal, selección sobre esa validación y cambio de distribución impiden garantía de cobertura. Se informa cobertura medida en prueba y anchura. No se debe anunciar una confianza garantizada del 80%.

## Comparables
Mismo año y posición amplia del snapshot, excluyendo el propio jugador. StandardScaler ajustado a los candidatos sobre edad, minutos, goles/90, asistencias/90 y minutos/partido. Distancia euclídea normalizada por raíz del número de variables; menor = más semejante en esas variables. No porcentaje de compatibilidad. No mide estilo táctico, defensa, técnica, adaptación, coste de fichaje ni recomienda contratar a nadie. Para porteros/defensas las métricas de producción son especialmente incompletas.

## Resultado
El modelo deportivo mejora edad+liga y mediana global en la prueba, pero pierde claramente contra persistencia. No se puede inferir que las grandes diferencias sean oportunidades de mercado. Son residuos que pueden reflejar datos incompletos, sesgos o errores del modelo.

## Fuente y licencias
Dataset upstream: https://github.com/dcaribou/transfermarkt-datasets (CC0 según upstream). Mirror empleado: https://huggingface.co/datasets/ngeorgea/transfermarkt-player-scores . Valoraciones atribuidas a Transfermarkt. Se guarda SHA256 y fechas máximas observadas en manifest.json. Downloader /main es mutable: el manifiesto permite detectar cambios; reproducibilidad exacta requiere el mismo snapshot. No se incorpora scraping del sitio original ni fotos de jugadores.
