"""
Descripción: Prototipo web interactivo en Streamlit (CRISP-DM Fase 6: Despliegue)
             para evaluación de microcréditos con la base empírica oficial SBS/CAF 2022,
             explicabilidad local (TreeSHAP Waterfall) y global (Beeswarm),
             análisis What-If y razones de acción adversa según la Res. SBS 3780-2021.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Importar módulos del sistema
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.data_preprocessing import CreditDataPreprocessor
from src.model_training import CreditRiskXGBoostWrapper
from src.explainability import (
    CreditRiskExplainer,
    FEATURE_FRIENDLY_NAMES,
    ADVERSE_ACTION_REASON_MAP,
    get_default_explainer
)
from src.model_evaluation import (
    calculate_expected_loss,
    convert_pd_to_credit_score,
    get_credit_decision
)

# Streamlit
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def create_sample_profiles():
    """Perfiles preestablecidos basados en el cuestionario oficial SBS/CAF 2022."""
    return {
        "🟢 Perfil Aprobado (Bajo Riesgo)": {
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
        "🟡 Perfil Revisión Manual (Riesgo Medio)": {
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
            "percepcion_sobreendeudado": 2,
            "tiene_cuenta_ahorro": "Sí",
            "tiene_tarjeta_credito": "No",
            "tiene_prestamo_personal": "No",
            "ha_usado_prestamista": "No",
            "ha_usado_pandero_junta": "Sí",
            "credito_formal_negado": "No",
            "monto_solicitado": 4500.0
        },
        "🔴 Perfil Rechazado (Alto Riesgo)": {
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
        "⚪ Perfil en Blanco (Personalizado)": {
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


def load_system_artifacts():
    """Carga los artefactos del modelo y datasets procesados."""
    preprocessor = CreditDataPreprocessor.load(config.PREPROCESSOR_PATH)
    
    # Intentar cargar desde model.joblib o json
    model = None
    alias_path = config.MODELS_DIR / "model.joblib"
    if alias_path.exists():
        try:
            model = joblib.load(alias_path)
        except Exception:
            model = None

    if model is None:
        model = CreditRiskXGBoostWrapper.load(config.XGBOOST_MODEL_JSON)

    explainer = CreditRiskExplainer(model, preprocessor)
    
    benchmark_data = {}
    if config.BENCHMARK_RESULTS_PATH.exists():
        with open(config.BENCHMARK_RESULTS_PATH, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

    df_test = pd.read_csv(config.PROCESSED_TEST_PATH) if config.PROCESSED_TEST_PATH.exists() else None

    return preprocessor, model, explainer, benchmark_data, df_test


def run_streamlit_app():
    """Función principal de la aplicación Streamlit."""
    if not STREAMLIT_AVAILABLE:
        print("[AVISO] Streamlit no está instalado. Ejecuta: streamlit run src/app.py")
        return

    st.set_page_config(
        page_title="Scoring crediticio para microfinanzas",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Cargar artefactos del modelo
    preprocessor, model, explainer, benchmark_data, df_test = load_system_artifacts()
    sample_profiles = create_sample_profiles()

    # SIDEBAR: Presets y Parámetros Regulatorios
    st.sidebar.markdown("### Cargar Perfil de Prueba")
    selected_preset = st.sidebar.selectbox("Selecciona un perfil:", list(sample_profiles.keys()))
    preset = sample_profiles[selected_preset]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Marco Regulatorio")
    st.sidebar.caption("Resolución SBS N.° 3780-2021 — Reglamento de Gestión del Riesgo de Crédito")
    st.sidebar.caption("Severidad de Pérdida (LGD): 50% (Estándar normativo microfinanciero)")
    st.sidebar.caption("Modelo: XGBoost Cost-Sensitive + TreeSHAP")
    st.sidebar.caption("Base: Encuesta SBS/CAF 2022")

    # HEADER PRINCIPAL
    st.title("Scoring crediticio para microfinanzas")
    st.markdown("---")

    # TABS DEL SISTEMA
    tab1, tab2, tab3, tab4 = st.tabs([
        "Originación Individual",
        "Explicabilidad Global XAI",
        "Benchmarking SBS",
        "Metodología SBS y ODS"
    ])

    # TAB 1: ORIGINACIÓN INDIVIDUAL
    with tab1:
        st.subheader("Formulario de Solicitud de Microcrédito (Cuestionario SBS/CAF 2022)")
        st.markdown("Ingresa los datos socioeconómicos, de inclusión digital y comportamiento financiero del solicitante:")

        col1, col2, col3 = st.columns(3)

        # Columna 1: Eje 1 (Sociodemográfico y Ocupación)
        with col1:
            st.markdown("##### Eje 1: Sociodemográfico y Ocupación")
            edad = st.number_input("Edad (Años)", min_value=18, max_value=85, value=preset["edad"])
            genero = st.selectbox("Género", ["Mujer", "Hombre"], index=0 if preset["genero"] == "Mujer" else 1)
            nivel_educacion = st.selectbox("Nivel Educativo", config.OPCIONES_EDUCACION, index=config.OPCIONES_EDUCACION.index(preset["nivel_educacion"]))
            departamento = st.selectbox("Departamento", config.OPCIONES_DEPARTAMENTO, index=config.OPCIONES_DEPARTAMENTO.index(preset["departamento"]))
            ambito = st.selectbox("Ámbito Geográfico", ["Urbano", "Rural"], index=0 if preset["ambito"] == "Urbano" else 1)
            
            c_sub1, c_sub2 = st.columns(2)
            with c_sub1:
                carga_familiar_adultos = st.number_input("Adultos en hogar", min_value=1, max_value=10, value=preset["carga_familiar_adultos"])
            with c_sub2:
                carga_familiar_ninos = st.number_input("Hijos menores", min_value=0, max_value=10, value=preset["carga_familiar_ninos"])
            
            condicion_laboral = st.selectbox("Ocupación / Condición", config.OPCIONES_CONDICION_LABORAL, index=config.OPCIONES_CONDICION_LABORAL.index(preset["condicion_laboral"]))
            es_informal = st.selectbox("Condición Tributaria / Formalidad", ["Informal / Sin RUC", "Formalizado / Con RUC o Beneficios"], index=0 if preset["es_informal"] == "Informal / Sin RUC" else 1)

        # Columna 2: Eje 2 (Inclusión Digital y Datos Alternativos - Core Tesis)
        with col2:
            st.markdown("##### Eje 2: Inclusión Digital y Datos Alternativos")
            adopcion_yape_plin = st.selectbox("¿Usa Billeteras Móviles (Yape/Plin/BIM)?", ["Sí", "No"], index=0 if preset["adopcion_yape_plin"] == "Sí" else 1)
            tiene_billetera_activa = st.selectbox("¿Tiene Billetera Digital Activa actualmente?", ["Sí", "No"], index=0 if preset["tiene_billetera_activa"] == "Sí" else 1)
            frecuencia_yape_plin = st.select_slider(
                "Frecuencia de Uso de Billeteras Digitales",
                options=[1, 2, 3, 4],
                value=preset["frecuencia_yape_plin"],
                format_func=lambda x: {1: "1. Nunca", 2: "2. Poco", 3: "3. Casi siempre", 4: "4. Siempre"}[x]
            )
            pago_celular_frecuencia = st.slider("Frecuencia de pagos/cobros con celular (1 a 5)", 1, 5, preset["pago_celular_frecuencia"])
            uso_whatsapp = st.slider("Uso de mensajería instantánea (WhatsApp) (1 a 4)", 1, 4, preset["uso_whatsapp"])
            uso_internet = st.slider("Uso de Internet en su vida diaria (1 a 4)", 1, 4, preset["uso_internet"])
            preferencia_efectivo = st.slider("Aversión a lo digital: 'Prefiero efectivo a tarjetas/apps' (1 a 5)", 1, 5, preset["preferencia_efectivo"])
            confianza_herramientas_digitales = st.slider("Confianza: 'Herramientas digitales facilitan mis finanzas' (1 a 5)", 1, 5, preset["confianza_herramientas_digitales"])
            
            st.markdown("###### Puntualidad en Cuentas y Recibos Básicos")
            p_sub1, p_sub2 = st.columns(2)
            with p_sub1:
                guarda_dinero_recibos = st.selectbox("¿Reserva dinero para facturas?", ["Sí", "No"], index=0 if preset["guarda_dinero_recibos"] == "Sí" else 1)
            with p_sub2:
                anota_recibos_vencer = st.selectbox("¿Anota fechas de vencimiento?", ["Sí", "No"], index=0 if preset["anota_recibos_vencer"] == "Sí" else 1)
            
            puntualidad_pago_deudas = st.slider("Puntualidad declarada en cuentas y deudas (1: Nunca a 5: Siempre)", 1, 5, preset["puntualidad_pago_deudas"])

        # Columna 3: Eje 3 (Capacidad, Resiliencia y Crédito Previo)
        with col3:
            st.markdown("##### Eje 3: Capacidad y Comportamiento Financiero")
            rango_ingreso = st.selectbox("Nivel de Ingreso Mensual del Hogar", config.OPCIONES_RANGO_INGRESO, index=config.OPCIONES_RANGO_INGRESO.index(preset["rango_ingreso"]))
            nivel_socioeconomico = st.selectbox("Nivel Socioeconómico (NSE)", config.OPCIONES_NSE, index=config.OPCIONES_NSE.index(preset["nivel_socioeconomico"]))
            ingresos_estables = st.selectbox("¿Sus ingresos mensuales son regulares/estables?", ["Sí", "No"], index=0 if preset["ingresos_estables"] == "Sí" else 1)
            
            pl_sub1, pl_sub2 = st.columns(2)
            with pl_sub1:
                planifica_presupuesto = st.selectbox("¿Elabora presupuesto?", ["Sí", "No"], index=0 if preset["planifica_presupuesto"] == "Sí" else 1)
            with pl_sub2:
                anota_gastos = st.selectbox("¿Registra gastos?", ["Sí", "No"], index=0 if preset["anota_gastos"] == "Sí" else 1)
            
            capacidad_afrontar_imprevisto = st.selectbox("¿Puede afrontar un gasto imprevisto hoy sin endeudarse?", ["Sí", "No"], index=0 if preset["capacidad_afrontar_imprevisto"] == "Sí" else 1)
            dificultad_emergencia_1000 = st.select_slider(
                "Dificultad de reunir S/ 1,000 en emergencia",
                options=[1, 2, 3, 4],
                value=preset["dificultad_emergencia_1000"],
                format_func=lambda x: {1: "1. Nada difícil", 2: "2. Algo difícil", 3: "3. Muy difícil", 4: "4. No podría"}[x]
            )
            meses_resiliencia_desempleo = st.select_slider(
                "Meses de cobertura si pierde fuente de ingresos",
                options=[1, 2, 3, 4, 5],
                value=preset["meses_resiliencia_desempleo"],
                format_func=lambda x: {1: "< 1 semana", 2: "1 mes", 3: "3 meses", 4: "6 meses", 5: "> 6 meses"}[x]
            )
            sobra_dinero_fin_mes = st.slider("Le queda dinero de sobra a fin de mes (1 a 5)", 1, 5, preset["sobra_dinero_fin_mes"])
            ingreso_apenas_alcanza = st.slider("Vulnerabilidad: 'El ingreso apenas alcanza' (1 a 5)", 1, 5, preset["ingreso_apenas_alcanza"])
            percepcion_sobreendeudado = st.slider("Sobreendeudamiento: 'Tengo demasiadas deudas' (1 a 5)", 1, 5, preset["percepcion_sobreendeudado"])
            
            st.markdown("###### Historial Crediticio Formal vs. Informal")
            h_sub1, h_sub2 = st.columns(2)
            with h_sub1:
                tiene_cuenta_ahorro = st.selectbox("Cuenta Ahorros", ["Sí", "No"], index=0 if preset["tiene_cuenta_ahorro"] == "Sí" else 1)
                tiene_tarjeta_credito = st.selectbox("Tarjeta Crédito", ["Sí", "No"], index=0 if preset["tiene_tarjeta_credito"] == "Sí" else 1)
                tiene_prestamo_personal = st.selectbox("Préstamo Personal", ["Sí", "No"], index=0 if preset["tiene_prestamo_personal"] == "Sí" else 1)
            with h_sub2:
                ha_usado_prestamista = st.selectbox("Prestamista Informal", ["Sí", "No"], index=0 if preset["ha_usado_prestamista"] == "Sí" else 1)
                ha_usado_pandero_junta = st.selectbox("Juntas / Panderos", ["Sí", "No"], index=0 if preset["ha_usado_pandero_junta"] == "Sí" else 1)
                credito_formal_negado = st.selectbox("Crédito Negado Previo", ["Sí", "No"], index=0 if preset["credito_formal_negado"] == "Sí" else 1)

            monto_solicitado = st.number_input("Monto Solicitado (S/)", min_value=500.0, max_value=50000.0, value=float(preset["monto_solicitado"]), step=500.0)

        # Consolidar diccionario de entrada
        applicant_dict = {
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
        if st.button("Evaluar Solicitud con XGBoost + TreeSHAP", type="primary", use_container_width=True):
            eval_result = explainer.explain_single_applicant(applicant_dict)
            
            st.markdown("### Resultado de la Evaluación Crediticia")
            
            # Tarjetas de KPI Principales
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric("Score Crediticio (0 - 1000)", f"{eval_result['score_crediticio']} pts")
            with kpi2:
                st.metric("Probabilidad de Mora (PD)", f"{eval_result['pd_pct']}%")
            with kpi3:
                el_custom = calculate_expected_loss(eval_result['pd_value'], monto_solicitado, config.DEFAULT_LGD)
                st.metric("Pérdida Esperada (EL)", f"S/ {el_custom:,.2f}", help="Calculado con LGD normativo SBS del 50%")
            with kpi4:
                st.markdown(f"**Dictamen del Modelo:**")
                st.markdown(
                    f"<div style='background-color:{eval_result['badge_color']}; color:white; font-size:18px; font-weight:bold; text-align:center; padding:10px; border-radius:8px;'>"
                    f"{eval_result['decision']}</div>",
                    unsafe_allow_html=True
                )

            st.caption(f"{eval_result['politica_descripcion']}")
            st.markdown("---")

            # Panel de Explicabilidad y Dictamen Regulatorio
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown("##### Índices Derivados de Comportamiento")
                for r_name, r_val in eval_result["ratios_clave"].items():
                    st.write(f"- **{r_name}:** `{r_val}`")

                st.markdown("##### Razones de Acción Adversa (Res. SBS N.° 3780-2021)")
                st.caption("Factores prioritarios que incrementaron el riesgo en la decisión algorítmica:")
                for i, r_text in enumerate(eval_result["razones_accion_adversa_sbs"], 1):
                    st.markdown(f"**{i}.** {r_text}")

                st.markdown("##### Dictamen para el Comité de Créditos")
                narr = eval_result["narrativa_comite"]
                st.write(f"**Resumen:** {narr['resumen_ejecutivo']}")
                
                st.markdown("**Fortalezas Detectadas:**")
                for f in narr["puntos_fuertes"]:
                    st.markdown(f"- {f}")
                
                st.markdown("**Factores de Riesgo:**")
                for a in narr["alertas_riesgo"]:
                    st.markdown(f"- {a}")

                st.markdown("**Recomendaciones:**")
                for rec in narr["recomendaciones_analista"]:
                    st.markdown(f"- {rec}")

            with res_col2:
                st.markdown("##### Explicabilidad Local TreeSHAP (Waterfall Plot)")
                st.caption(f"Valor Base E[f(x)] = {eval_result['expected_value']:.4f} log-odds | En tiempo real (< 0.5 s)")
                
                # Renderizar Waterfall Plot
                fig_waterfall = explainer.generate_waterfall_plot_figure(applicant_dict, max_display=9)
                st.pyplot(fig_waterfall, use_container_width=True)

    # TAB 2: EXPLICABILIDAD GLOBAL XAI
    with tab2:
        st.subheader("Explicabilidad Global del Portafolio (TreeSHAP Beeswarm)")
        st.markdown("Ranking de las variables con mayor contribución promedio sobre la base empírica SBS/CAF 2022:")

        beeswarm_img_path = config.MODELS_DIR / "shap_beeswarm.png"
        if beeswarm_img_path.exists():
            st.image(str(beeswarm_img_path), caption="SHAP Summary Plot (Beeswarm) - Base Oficial SBS/CAF 2022", use_container_width=True)

        if df_test is not None:
            st.markdown("##### Tabla de Importancia Relativa Global")
            df_global_imp = explainer.get_global_feature_importance(df_test, top_n=12)
            st.dataframe(
                df_global_imp[["feature_name", "mean_abs_shap", "importance_pct"]].rename(
                    columns={"feature_name": "Variable", "mean_abs_shap": "Impacto Medio |SHAP|", "importance_pct": "Importancia Relativa (%)"}
                ),
                use_container_width=True
            )

    # TAB 3: BENCHMARKING DE MODELOS SBS
    with tab3:
        st.subheader("Validación y Comparativa de Modelos")
        st.markdown("Validación experimental sobre el conjunto de prueba independiente de la encuesta SBS/CAF 2022:")

        if benchmark_data and "resumen_comparativo" in benchmark_data:
            df_bench = pd.DataFrame(benchmark_data["resumen_comparativo"])
            st.dataframe(df_bench, use_container_width=True)

            st.markdown("##### Cumplimiento de Estándares SBS:")
            st.markdown(
                r"""
                - **Estadístico Kolmogorov-Smirnov (KS):** El modelo XGBoost alcanza un **KS > 33%**, superando el piso mínimo regulatorio para microfinanzas ($KS \ge 30\%$).
                - **Capacidad de Discriminación (AUC-ROC):** Logra un **AUC de 0.7134**, superior al scorecard lineal tradicional en datos alternativos.
                - **Ponderación Sensible al Costo (`scale_pos_weight = 2.99`):** Compensa el desbalance real de morosidad empírica de la SBS (25.06%), evitando el sesgo pro-solvente.
                """
            )

    # TAB 4: MARCO TEÓRICO, SBS Y ODS
    with tab4:
        st.subheader("Marco Normativo SBS y Contribución a los ODS")
        st.markdown(
            """
            ### Resolución SBS N.° 3780-2021
            - **Uso de Datos Alternativos:** Permite el uso de fuentes de información no tradicionales (billeteras digitales, pagos de servicios) para evaluar clientes sin historial en centrales de riesgo.
            - **Auditabilidad y Deber de Notificación:** Exige que los modelos internos de riesgo provean explicaciones claras sobre los motivos de denegación o reducción de línea a los usuarios (Razones de Acción Adversa).

            ### Alineamiento con los Objetivos de Desarrollo Sostenible (ODS)
            - **ODS 8 (Trabajo Decente y Crecimiento Económico):** Facilita el acceso al crédito productivo para microempresarios informales y trabajadores por cuenta propia.
            - **ODS 10 (Reducción de las Desigualdades):** Rompe la barrera del "thin-file" permitiendo a personas no bancarizadas ser evaluadas a través de su comportamiento en Yape, Plin y pagos de recibos.
            """
        )


if __name__ == "__main__":
    run_streamlit_app()
