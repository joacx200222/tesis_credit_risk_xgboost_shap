import json
import pandas as pd
from pathlib import Path
import sys

# `streamlit run src/app.py` solo agrega src/ al path; hace falta la raíz para importar el paquete src
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.preprocesamiento import Preprocesador
from src.entrenamiento import ModeloXGBoost
from src.explicabilidad import ExplicadorSHAP
from src.evaluacion import perdida_esperada

# Streamlit
try:
    import streamlit as st
    HAY_STREAMLIT = True
except ImportError:
    HAY_STREAMLIT = False


def perfiles_ejemplo():
    """Perfiles de ejemplo para precargar el formulario."""
    return {
        "Perfil aprobado (bajo riesgo)": {
            "edad": 38,
            "carga_familiar_adultos": 2,
            "carga_familiar_ninos": 1,
            "genero": "Mujer",
            "nivel_educacion": "Superior Técnico Completo",
            "departamento": "Lima / Callao",
            "ambito": "Urbano",
            "condicion_laboral": "Trabajador Independiente / Cuenta propia",
            "es_informal": "Informal / Sin RUC",
            "adopcion_yape_plin": "Sí",
            "tiene_billetera_activa": "Sí",
            "frecuencia_yape_plin": 4,
            "pago_celular_frecuencia": 5,
            "uso_whatsapp": 4,
            "uso_internet": 4,
            "preferencia_efectivo": 1,
            "confianza_herramientas_digitales": 5,
            "guarda_dinero_recibos": "Sí",
            "anota_recibos_vencer": "Sí",
            "puntualidad_pago_deudas": 5,
            "rango_ingreso": "De S/ 2,401 a S/ 4,000",
            "nivel_socioeconomico": "C",
            "ingresos_estables": "Sí",
            "planifica_presupuesto": "Sí",
            "anota_gastos": "Sí",
            "capacidad_afrontar_imprevisto": "Sí",
            "dificultad_emergencia_1000": 1,
            "meses_resiliencia_desempleo": 4,
            "sobra_dinero_fin_mes": 4,
            "ingreso_apenas_alcanza": 1,
            "percepcion_sobreendeudado": 1,
            "tiene_cuenta_ahorro": "Sí",
            "tiene_tarjeta_credito": "Sí",
            "tiene_prestamo_personal": "No",
            "ha_usado_prestamista": "No",
            "ha_usado_pandero_junta": "No",
            "credito_formal_negado": "No",
            "monto_solicitado": 6000.0
        },
        "Perfil revisión manual (riesgo medio)": {
            "edad": 29,
            "carga_familiar_adultos": 1,
            "carga_familiar_ninos": 2,
            "genero": "Hombre",
            "nivel_educacion": "Secundaria Completa",
            "departamento": "Arequipa",
            "ambito": "Urbano",
            "condicion_laboral": "Trabajador Independiente / Cuenta propia",
            "es_informal": "Informal / Sin RUC",
            "adopcion_yape_plin": "Sí",
            "tiene_billetera_activa": "Sí",
            "frecuencia_yape_plin": 3,
            "pago_celular_frecuencia": 3,
            "uso_whatsapp": 4,
            "uso_internet": 3,
            "preferencia_efectivo": 3,
            "confianza_herramientas_digitales": 3,
            "guarda_dinero_recibos": "Sí",
            "anota_recibos_vencer": "No",
            "puntualidad_pago_deudas": 4,
            "rango_ingreso": "De S/ 1,201 a S/ 2,400",
            "nivel_socioeconomico": "D",
            "ingresos_estables": "Sí",
            "planifica_presupuesto": "No",
            "anota_gastos": "Sí",
            "capacidad_afrontar_imprevisto": "No",
            "dificultad_emergencia_1000": 3,
            "meses_resiliencia_desempleo": 2,
            "sobra_dinero_fin_mes": 2,
            "ingreso_apenas_alcanza": 3,
            "percepcion_sobreendeudado": 1,
            "tiene_cuenta_ahorro": "Sí",
            "tiene_tarjeta_credito": "No",
            "tiene_prestamo_personal": "No",
            "ha_usado_prestamista": "No",
            "ha_usado_pandero_junta": "Sí",
            "credito_formal_negado": "No",
            "monto_solicitado": 4500.0
        },
        "Perfil rechazado (alto riesgo)": {
            "edad": 45,
            "carga_familiar_adultos": 2,
            "carga_familiar_ninos": 3,
            "genero": "Hombre",
            "nivel_educacion": "Primaria / Sin Instrucción",
            "departamento": "Piura",
            "ambito": "Rural",
            "condicion_laboral": "Campesino / Agricultor / Ganadero",
            "es_informal": "Informal / Sin RUC",
            "adopcion_yape_plin": "No",
            "tiene_billetera_activa": "No",
            "frecuencia_yape_plin": 1,
            "pago_celular_frecuencia": 1,
            "uso_whatsapp": 1,
            "uso_internet": 1,
            "preferencia_efectivo": 5,
            "confianza_herramientas_digitales": 1,
            "guarda_dinero_recibos": "No",
            "anota_recibos_vencer": "No",
            "puntualidad_pago_deudas": 2,
            "rango_ingreso": "Menor a S/ 600",
            "nivel_socioeconomico": "E",
            "ingresos_estables": "No",
            "planifica_presupuesto": "No",
            "anota_gastos": "No",
            "capacidad_afrontar_imprevisto": "No",
            "dificultad_emergencia_1000": 4,
            "meses_resiliencia_desempleo": 1,
            "sobra_dinero_fin_mes": 1,
            "ingreso_apenas_alcanza": 5,
            "percepcion_sobreendeudado": 5,
            "tiene_cuenta_ahorro": "No",
            "tiene_tarjeta_credito": "No",
            "tiene_prestamo_personal": "No",
            "ha_usado_prestamista": "Sí",
            "ha_usado_pandero_junta": "Sí",
            "credito_formal_negado": "Sí",
            "monto_solicitado": 3500.0
        },
        "Perfil en blanco": {
            "edad": 30,
            "carga_familiar_adultos": 1,
            "carga_familiar_ninos": 0,
            "genero": "Mujer",
            "nivel_educacion": "Secundaria Completa",
            "departamento": "Lima / Callao",
            "ambito": "Urbano",
            "condicion_laboral": "Trabajador Independiente / Cuenta propia",
            "es_informal": "Informal / Sin RUC",
            "adopcion_yape_plin": "No",
            "tiene_billetera_activa": "No",
            "frecuencia_yape_plin": 1,
            "pago_celular_frecuencia": 1,
            "uso_whatsapp": 1,
            "uso_internet": 1,
            "preferencia_efectivo": 3,
            "confianza_herramientas_digitales": 3,
            "guarda_dinero_recibos": "No",
            "anota_recibos_vencer": "No",
            "puntualidad_pago_deudas": 3,
            "rango_ingreso": "De S/ 1,201 a S/ 2,400",
            "nivel_socioeconomico": "C",
            "ingresos_estables": "No",
            "planifica_presupuesto": "No",
            "anota_gastos": "No",
            "capacidad_afrontar_imprevisto": "No",
            "dificultad_emergencia_1000": 2,
            "meses_resiliencia_desempleo": 2,
            "sobra_dinero_fin_mes": 3,
            "ingreso_apenas_alcanza": 3,
            "percepcion_sobreendeudado": 3,
            "tiene_cuenta_ahorro": "No",
            "tiene_tarjeta_credito": "No",
            "tiene_prestamo_personal": "No",
            "ha_usado_prestamista": "No",
            "ha_usado_pandero_junta": "No",
            "credito_formal_negado": "No",
            "monto_solicitado": 2000.0
        }
    }


def cargar_artefactos():
    preprocesador = Preprocesador.cargar(config.RUTA_PREPROCESADOR)
    modelo = ModeloXGBoost.cargar(config.RUTA_MODELO)
    explicador = ExplicadorSHAP(modelo, preprocesador)

    datos_benchmark = {}
    if config.RUTA_BENCHMARK.exists():
        with open(config.RUTA_BENCHMARK, "r", encoding="utf-8") as f:
            datos_benchmark = json.load(f)

    df_test = pd.read_csv(config.RUTA_TEST) if config.RUTA_TEST.exists() else None

    return preprocesador, modelo, explicador, datos_benchmark, df_test


def ejecutar_app():
    if not HAY_STREAMLIT:
        print("[AVISO] Streamlit no está instalado. Ejecuta: streamlit run src/app.py")
        return

    st.set_page_config(
        page_title="Scoring crediticio para microfinanzas",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    preprocesador, modelo, explicador, datos_benchmark, df_test = cargar_artefactos()
    perfiles = perfiles_ejemplo()

    st.sidebar.markdown("### Cargar Perfil de Prueba")
    perfil_elegido = st.sidebar.selectbox("Selecciona un perfil:", list(perfiles.keys()))
    perfil = perfiles[perfil_elegido]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Marco Regulatorio")
    st.sidebar.caption("Resolución SBS N.° 3780-2011 — Reglamento de la Gestión de Riesgo de Crédito")
    st.sidebar.caption("Severidad de Pérdida (LGD): 50% (Estándar normativo microfinanciero)")
    st.sidebar.caption("Modelo: XGBoost Cost-Sensitive + TreeSHAP")
    st.sidebar.caption("Base: Encuesta SBS/CAF 2022")

    st.title("Scoring crediticio para microfinanzas")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "Originación Individual",
        "Explicabilidad Global XAI",
        "Benchmarking SBS"
    ])

    with tab1:
        st.subheader("Formulario de Solicitud de Microcrédito (Cuestionario SBS/CAF 2022)")
        st.markdown("Ingresa los datos socioeconómicos, de inclusión digital y comportamiento financiero del solicitante:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### Eje 1: Sociodemográfico y Ocupación")
            edad = st.number_input("Edad (Años)", min_value=18, max_value=85, value=perfil["edad"])
            genero = st.selectbox("Género", ["Mujer", "Hombre"], index=0 if perfil["genero"] == "Mujer" else 1)
            nivel_educacion = st.selectbox("Nivel Educativo", config.OPCIONES_EDUCACION, index=config.OPCIONES_EDUCACION.index(perfil["nivel_educacion"]))
            departamento = st.selectbox("Departamento", config.OPCIONES_DEPARTAMENTO, index=config.OPCIONES_DEPARTAMENTO.index(perfil["departamento"]))
            ambito = st.selectbox("Ámbito Geográfico", ["Urbano", "Rural"], index=0 if perfil["ambito"] == "Urbano" else 1)

            c_sub1, c_sub2 = st.columns(2)
            with c_sub1:
                carga_familiar_adultos = st.number_input("Adultos en hogar", min_value=1, max_value=10, value=perfil["carga_familiar_adultos"])
            with c_sub2:
                carga_familiar_ninos = st.number_input("Hijos menores", min_value=0, max_value=10, value=perfil["carga_familiar_ninos"])

            condicion_laboral = st.selectbox("Ocupación / Condición", config.OPCIONES_CONDICION_LABORAL, index=config.OPCIONES_CONDICION_LABORAL.index(perfil["condicion_laboral"]))
            es_informal = st.selectbox("Condición Tributaria / Formalidad", ["Informal / Sin RUC", "Formalizado / Con RUC o Beneficios"], index=0 if perfil["es_informal"] == "Informal / Sin RUC" else 1)

        with col2:
            st.markdown("##### Eje 2: Inclusión Digital y Datos Alternativos")
            adopcion_yape_plin = st.selectbox("¿Usa Billeteras Móviles (Yape/Plin/BIM)?", ["Sí", "No"], index=0 if perfil["adopcion_yape_plin"] == "Sí" else 1)
            tiene_billetera_activa = st.selectbox("¿Tiene Billetera Digital Activa actualmente?", ["Sí", "No"], index=0 if perfil["tiene_billetera_activa"] == "Sí" else 1)
            frecuencia_yape_plin = st.select_slider(
                "Frecuencia de Uso de Billeteras Digitales",
                options=[1, 2, 3, 4],
                value=perfil["frecuencia_yape_plin"],
                format_func=lambda x: {1: "1. Nunca", 2: "2. Poco", 3: "3. Casi siempre", 4: "4. Siempre"}[x]
            )
            pago_celular_frecuencia = st.slider("Frecuencia de pagos/cobros con celular (1 a 5)", 1, 5, perfil["pago_celular_frecuencia"])
            uso_whatsapp = st.slider("Uso de mensajería instantánea (WhatsApp) (1 a 4)", 1, 4, perfil["uso_whatsapp"])
            uso_internet = st.slider("Uso de Internet en su vida diaria (1 a 4)", 1, 4, perfil["uso_internet"])
            preferencia_efectivo = st.slider("Aversión a lo digital: 'Prefiero efectivo a tarjetas/apps' (1 a 5)", 1, 5, perfil["preferencia_efectivo"])
            confianza_herramientas_digitales = st.slider("Confianza: 'Herramientas digitales facilitan mis finanzas' (1 a 5)", 1, 5, perfil["confianza_herramientas_digitales"])

            st.markdown("###### Puntualidad en Cuentas y Recibos Básicos")
            p_sub1, p_sub2 = st.columns(2)
            with p_sub1:
                guarda_dinero_recibos = st.selectbox("¿Reserva dinero para facturas?", ["Sí", "No"], index=0 if perfil["guarda_dinero_recibos"] == "Sí" else 1)
            with p_sub2:
                anota_recibos_vencer = st.selectbox("¿Anota fechas de vencimiento?", ["Sí", "No"], index=0 if perfil["anota_recibos_vencer"] == "Sí" else 1)

            puntualidad_pago_deudas = st.slider("Puntualidad declarada en cuentas y deudas (1: Nunca a 5: Siempre)", 1, 5, perfil["puntualidad_pago_deudas"])

        with col3:
            st.markdown("##### Eje 3: Capacidad y Comportamiento Financiero")
            rango_ingreso = st.selectbox("Nivel de Ingreso Mensual del Hogar", config.OPCIONES_RANGO_INGRESO, index=config.OPCIONES_RANGO_INGRESO.index(perfil["rango_ingreso"]))
            nivel_socioeconomico = st.selectbox("Nivel Socioeconómico (NSE)", config.OPCIONES_NSE, index=config.OPCIONES_NSE.index(perfil["nivel_socioeconomico"]))
            ingresos_estables = st.selectbox("¿Sus ingresos mensuales son regulares/estables?", ["Sí", "No"], index=0 if perfil["ingresos_estables"] == "Sí" else 1)

            pl_sub1, pl_sub2 = st.columns(2)
            with pl_sub1:
                planifica_presupuesto = st.selectbox("¿Elabora presupuesto?", ["Sí", "No"], index=0 if perfil["planifica_presupuesto"] == "Sí" else 1)
            with pl_sub2:
                anota_gastos = st.selectbox("¿Registra gastos?", ["Sí", "No"], index=0 if perfil["anota_gastos"] == "Sí" else 1)

            capacidad_afrontar_imprevisto = st.selectbox("¿Puede afrontar un gasto imprevisto hoy sin endeudarse?", ["Sí", "No"], index=0 if perfil["capacidad_afrontar_imprevisto"] == "Sí" else 1)
            dificultad_emergencia_1000 = st.select_slider(
                "Dificultad de reunir S/ 1,000 en emergencia",
                options=[1, 2, 3, 4],
                value=perfil["dificultad_emergencia_1000"],
                format_func=lambda x: {1: "1. Nada difícil", 2: "2. Algo difícil", 3: "3. Muy difícil", 4: "4. No podría"}[x]
            )
            meses_resiliencia_desempleo = st.select_slider(
                "Meses de cobertura si pierde fuente de ingresos",
                options=[1, 2, 3, 4, 5],
                value=perfil["meses_resiliencia_desempleo"],
                format_func=lambda x: {1: "< 1 semana", 2: "1 mes", 3: "3 meses", 4: "6 meses", 5: "> 6 meses"}[x]
            )
            sobra_dinero_fin_mes = st.slider("Le queda dinero de sobra a fin de mes (1 a 5)", 1, 5, perfil["sobra_dinero_fin_mes"])
            ingreso_apenas_alcanza = st.slider("Vulnerabilidad: 'El ingreso apenas alcanza' (1 a 5)", 1, 5, perfil["ingreso_apenas_alcanza"])
            percepcion_sobreendeudado = st.slider("Sobreendeudamiento: 'Tengo demasiadas deudas' (1 a 5)", 1, 5, perfil["percepcion_sobreendeudado"])

            st.markdown("###### Historial Crediticio Formal vs. Informal")
            h_sub1, h_sub2 = st.columns(2)
            with h_sub1:
                tiene_cuenta_ahorro = st.selectbox("Cuenta Ahorros", ["Sí", "No"], index=0 if perfil["tiene_cuenta_ahorro"] == "Sí" else 1)
                tiene_tarjeta_credito = st.selectbox("Tarjeta Crédito", ["Sí", "No"], index=0 if perfil["tiene_tarjeta_credito"] == "Sí" else 1)
                tiene_prestamo_personal = st.selectbox("Préstamo Personal", ["Sí", "No"], index=0 if perfil["tiene_prestamo_personal"] == "Sí" else 1)
            with h_sub2:
                ha_usado_prestamista = st.selectbox("Prestamista Informal", ["Sí", "No"], index=0 if perfil["ha_usado_prestamista"] == "Sí" else 1)
                ha_usado_pandero_junta = st.selectbox("Juntas / Panderos", ["Sí", "No"], index=0 if perfil["ha_usado_pandero_junta"] == "Sí" else 1)
                credito_formal_negado = st.selectbox("Crédito Negado Previo", ["Sí", "No"], index=0 if perfil["credito_formal_negado"] == "Sí" else 1)

            monto_solicitado = st.number_input("Monto Solicitado (S/)", min_value=500.0, max_value=50000.0, value=float(perfil["monto_solicitado"]), step=500.0)

        solicitud = {
            "edad": edad,
            "carga_familiar_adultos": carga_familiar_adultos,
            "carga_familiar_ninos": carga_familiar_ninos,
            "genero": genero,
            "nivel_educacion": nivel_educacion,
            "departamento": departamento,
            "ambito": ambito,
            "condicion_laboral": condicion_laboral,
            "es_informal": es_informal,
            "adopcion_yape_plin": adopcion_yape_plin,
            "tiene_billetera_activa": tiene_billetera_activa,
            "frecuencia_yape_plin": frecuencia_yape_plin,
            "pago_celular_frecuencia": pago_celular_frecuencia,
            "uso_whatsapp": uso_whatsapp,
            "uso_internet": uso_internet,
            "preferencia_efectivo": preferencia_efectivo,
            "confianza_herramientas_digitales": confianza_herramientas_digitales,
            "guarda_dinero_recibos": guarda_dinero_recibos,
            "anota_recibos_vencer": anota_recibos_vencer,
            "puntualidad_pago_deudas": puntualidad_pago_deudas,
            "rango_ingreso": rango_ingreso,
            "nivel_socioeconomico": nivel_socioeconomico,
            "ingresos_estables": ingresos_estables,
            "planifica_presupuesto": planifica_presupuesto,
            "anota_gastos": anota_gastos,
            "capacidad_afrontar_imprevisto": capacidad_afrontar_imprevisto,
            "dificultad_emergencia_1000": dificultad_emergencia_1000,
            "meses_resiliencia_desempleo": meses_resiliencia_desempleo,
            "sobra_dinero_fin_mes": sobra_dinero_fin_mes,
            "ingreso_apenas_alcanza": ingreso_apenas_alcanza,
            "percepcion_sobreendeudado": percepcion_sobreendeudado,
            "tiene_cuenta_ahorro": tiene_cuenta_ahorro,
            "tiene_tarjeta_credito": tiene_tarjeta_credito,
            "tiene_prestamo_personal": tiene_prestamo_personal,
            "ha_usado_prestamista": ha_usado_prestamista,
            "ha_usado_pandero_junta": ha_usado_pandero_junta,
            "credito_formal_negado": credito_formal_negado,
            "monto_solicitado": monto_solicitado
        }

        st.markdown("---")
        if st.button("Evaluar Solicitud con XGBoost + TreeSHAP", type="primary", width="stretch"):
            resultado = explicador.evaluar_solicitud(solicitud)

            st.markdown("### Resultado de la Evaluación Crediticia")

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric("Score Crediticio (0 - 1000)", f"{resultado['score']} pts")
            with kpi2:
                st.metric("Probabilidad de Mora (PD)", f"{resultado['pd_pct']}%")
            with kpi3:
                el = perdida_esperada(resultado['pd'], monto_solicitado, config.LGD)
                st.metric("Pérdida Esperada (EL)", f"S/ {el:,.2f}", help="Calculado con LGD normativo SBS del 50%")
            with kpi4:
                st.markdown("**Dictamen del Modelo:**")
                st.markdown(
                    f"<div style='background-color:{resultado['color']}; color:white; font-size:18px; font-weight:bold; text-align:center; padding:10px; border-radius:8px;'>"
                    f"{resultado['decision']}</div>",
                    unsafe_allow_html=True
                )

            st.caption(resultado['politica'])
            st.markdown("---")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown("##### Índices Derivados de Comportamiento")
                for nombre, valor in resultado["indices"].items():
                    st.write(f"- **{nombre}:** `{valor}`")

                st.markdown("##### Razones de Acción Adversa (Ley N.° 32294)")
                st.caption("Factores prioritarios que incrementaron el riesgo en la decisión algorítmica:")
                for i, texto in enumerate(resultado["motivos_denegacion"], 1):
                    st.markdown(f"**{i}.** {texto}")

                st.markdown("##### Dictamen para el Comité de Créditos")
                dictamen = resultado["dictamen"]
                st.write(f"**Resumen:** {dictamen['resumen']}")

                st.markdown("**Fortalezas Detectadas:**")
                for f in dictamen["puntos_fuertes"]:
                    st.markdown(f"- {f}")

                st.markdown("**Factores de Riesgo:**")
                for a in dictamen["alertas_riesgo"]:
                    st.markdown(f"- {a}")

                st.markdown("**Recomendaciones:**")
                for rec in dictamen["recomendaciones"]:
                    st.markdown(f"- {rec}")

            with res_col2:
                st.markdown("##### Explicabilidad Local TreeSHAP (Waterfall Plot)")
                st.caption(f"Valor Base E[f(x)] = {resultado['valor_base']:.4f} log-odds")

                fig = explicador.grafico_waterfall(solicitud, max_variables=9)
                st.pyplot(fig, width="stretch")

    with tab2:
        st.subheader("Explicabilidad Global del Portafolio (TreeSHAP Beeswarm)")
        st.markdown("Ranking de las variables con mayor contribución promedio sobre la base empírica SBS/CAF 2022:")

        ruta_beeswarm = config.DIR_MODELOS / "shap_beeswarm.png"
        if ruta_beeswarm.exists():
            st.image(str(ruta_beeswarm), caption="SHAP Summary Plot (Beeswarm) - Base Oficial SBS/CAF 2022", width="stretch")

        if df_test is not None:
            st.markdown("##### Tabla de Importancia Relativa Global")
            df_imp = explicador.importancia_global(df_test, top=12)
            st.dataframe(
                df_imp[["nombre", "shap_medio_abs", "importancia_pct"]].rename(
                    columns={"nombre": "Variable", "shap_medio_abs": "Impacto Medio |SHAP|", "importancia_pct": "Importancia Relativa (%)"}
                ),
                width="stretch"
            )

    with tab3:
        st.subheader("Validación y Comparativa de Modelos")
        st.markdown("Validación experimental sobre el conjunto de prueba independiente de la encuesta SBS/CAF 2022:")

        if datos_benchmark and "resumen_comparativo" in datos_benchmark:
            df_bench = pd.DataFrame(datos_benchmark["resumen_comparativo"])
            st.dataframe(df_bench, width="stretch")

            if "comparacion_base_vs_optimizado" in datos_benchmark:
                st.markdown("##### Hiperparámetros base vs optimizados con Optuna (test):")
                st.dataframe(pd.DataFrame(datos_benchmark["comparacion_base_vs_optimizado"]), width="stretch")

            m = {k: v["metricas"] for k, v in datos_benchmark["modelos"].items()}
            xgb_m, rf_m, lr_m = m["XGBoost_Propuesto"], m["Random_Forest"], m["Regresion_Logistica"]
            supera = "supera" if xgb_m["ks_pct"] >= 30 else "no alcanza"
            optuna_txt = (
                "los tres modelos se ajustaron con Optuna (TPE) y validación cruzada estratificada de 5 folds "
                "solo sobre train; el set de prueba no participó en la búsqueda."
                if datos_benchmark.get("hiperparametros_optimizados")
                else "se usaron los hiperparámetros base (no se encontró models/best_params.json)."
            )

            st.markdown("##### Cumplimiento de Estándares SBS:")
            st.markdown(
                f"""
- **KS (por deciles de riesgo):** XGBoost llega a **{xgb_m['ks_pct']:.2f}%** y {supera} el piso de referencia de **30%**; Random Forest {rf_m['ks_pct']:.2f}% y Regresión Logística {lr_m['ks_pct']:.2f}%.
- **AUC-ROC:** XGBoost **{xgb_m['auc_roc']:.4f}**, Random Forest {rf_m['auc_roc']:.4f} y Regresión Logística {lr_m['auc_roc']:.4f}.
- **Hiperparámetros:** {optuna_txt}
- **`scale_pos_weight` ≈ {modelo.scale_pos_weight:.2f}:** negativos/positivos del set de train; compensa el desbalance de mora y evita el sesgo hacia aprobar.
"""
            )


if __name__ == "__main__":
    ejecutar_app()
