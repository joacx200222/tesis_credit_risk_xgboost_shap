# Avisos de terceros

La licencia MIT de [LICENSE](LICENSE) cubre solo el código de este repositorio.
Los datos y las librerías de terceros conservan sus propias licencias y condiciones.

## Datos

**Encuesta Nacional de Capacidades Financieras del Perú 2022**: Superintendencia de
Banca, Seguros y AFP (SBS) y CAF – Banco de Desarrollo de América Latina.

- La base original (`BD_Encuesta_Capacidades_Financieras_2022.dta`) **no se incluye**
  en el repositorio. Hay que descargarla de la fuente oficial y colocarla en la raíz
  para regenerar el pipeline desde cero.
- `data/processed/{train,val,test}.csv` son variables derivadas de esa encuesta
  (respuestas recodificadas, sin identificadores de los encuestados). Se incluyen solo
  para que los resultados y las pruebas sean reproducibles.
- Los derechos sobre la encuesta y sus microdatos pertenecen a la SBS y a CAF. El uso
  de estos datos está sujeto a sus condiciones, no a la licencia MIT de este repositorio.

## Librerías

El proyecto usa estas librerías como dependencias. No se redistribuye su código, se
instalan con `pip install -r requirements.txt`.

| Librería | Licencia |
| :--- | :--- |
| [NumPy](https://numpy.org) | BSD-3-Clause |
| [pandas](https://pandas.pydata.org) | BSD-3-Clause |
| [scikit-learn](https://scikit-learn.org) | BSD-3-Clause |
| [XGBoost](https://xgboost.ai) | Apache-2.0 |
| [SHAP](https://github.com/shap/shap) | MIT |
| [Optuna](https://optuna.org) | MIT |
| [Matplotlib](https://matplotlib.org) | Matplotlib License (basada en PSF) |
| [Streamlit](https://streamlit.io) | Apache-2.0 |
