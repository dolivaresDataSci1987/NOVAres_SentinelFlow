# SentinelFlow

Plataforma end-to-end de detección de fraude para transacciones de alto volumen
Python · Machine Learning · MLflow · FastAPI · Streamlit · Docker

![Estado](https://img.shields.io/badge/status-en%20desarrollo-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-informational)
![MLflow](https://img.shields.io/badge/mlflow-experiment%20tracking-orange)
![FastAPI](https://img.shields.io/badge/api-FastAPI-009688)
![Licencia](https://img.shields.io/badge/license-MIT-green)

-------------------------------------------------------------------------------

DESCRIPCIÓN

SentinelFlow es una plataforma modular orientada a la detección de fraude en entornos transaccionales de alto volumen. La solución integra generación y preparación de datos, construcción de variables, entrenamiento de modelos, serving mediante API y monitorización operativa, siguiendo una arquitectura reproducible, mantenible y preparada para evolucionar hacia escenarios de despliegue más amplios.

El sistema está diseñado para asignar un fraud score a cada transacción y traducirlo en una decisión operativa accionable, combinando analítica predictiva, trazabilidad experimental y una estructura técnica clara para el ciclo completo del modelo.

-------------------------------------------------------------------------------

OBJETIVO

Construir una solución integral para:

- procesar transacciones y entidades relacionadas
- identificar patrones compatibles con fraude
- generar variables predictivas históricas, contextuales y comportamentales
- entrenar y comparar modelos de clasificación
- registrar experimentos y versiones de modelo
- exponer inferencia mediante una API REST
- monitorizar estabilidad, drift y comportamiento del score
- facilitar una base técnica clara para futuras extensiones de despliegue

-------------------------------------------------------------------------------

CONTEXTO DE NEGOCIO

Las operaciones transaccionales de alto volumen presentan un equilibrio complejo entre control de riesgo y continuidad operativa. A medida que crece el número de transacciones, también lo hacen la superficie de exposición al fraude, la necesidad de priorización y el coste asociado a revisiones manuales o bloqueos erróneos.

En este contexto, una solución de scoring debe responder simultáneamente a varios retos:

- detectar operaciones potencialmente fraudulentas con suficiente anticipación
- minimizar falsos positivos que afecten la experiencia del cliente
- mantener trazabilidad sobre modelos, datos y resultados
- integrar una lógica de decisión interpretable para operación y riesgo
- permitir seguimiento continuo del comportamiento del sistema en el tiempo

SentinelFlow aborda este escenario mediante una arquitectura que conecta datos, modelado, serving y monitorización en un mismo flujo.

-------------------------------------------------------------------------------

ALCANCE FUNCIONAL

Cada nueva transacción recibe un fraud score entre 0 y 1. A partir de ese score, el sistema asigna una decisión operativa:

- APPROVE -> operación de bajo riesgo
- REVIEW -> operación que requiere revisión manual
- BLOCK -> operación con alta probabilidad de fraude

Además del score y la decisión, la capa de inferencia puede devolver factores explicativos simplificados asociados al resultado.

-------------------------------------------------------------------------------

ARQUITECTURA FUNCIONAL

Datos sintéticos / eventos crudos
            ↓
         Capa raw
            ↓
 Limpieza y validación de datos
            ↓
        Capa silver
            ↓
 Feature engineering + dataset analítico
            ↓
         Capa gold
            ↓
 Entrenamiento y evaluación de modelos
            ↓
 MLflow tracking + registro de experimentos
            ↓
      Modelo final seleccionado
            ↓
         API con FastAPI
            ↓
 Score + banda de riesgo + decisión
            ↓
 Dashboard demo + monitorización

-------------------------------------------------------------------------------

COMPONENTES PRINCIPALES

1. Ingesta y preparación de datos
La solución parte de un conjunto de entidades y eventos transaccionales estructurados en capas de datos locales:

- raw: datos originales o generados
- silver: datos limpios, normalizados y validados
- gold: dataset analítico preparado para modelado

2. Feature engineering
Se construyen variables orientadas a capturar comportamiento transaccional, señales históricas, desviaciones respecto al patrón esperado y atributos de contexto útiles para fraude.

3. Modelado
Se entrenan modelos de clasificación con foco en fraude como problema desbalanceado, comparando desempeño técnico y utilidad operativa.

4. Tracking y versionado
El ciclo de entrenamiento se registra con MLflow, incluyendo parámetros, métricas, artefactos y versiones de modelo.

5. Serving
El modelo seleccionado se expone mediante FastAPI, permitiendo consumo por otros sistemas a través de endpoints REST.

6. Monitorización
Se incluyen chequeos de drift, estabilidad del score, calidad de datos y métricas operativas para simular buenas prácticas de seguimiento post-entrenamiento.

-------------------------------------------------------------------------------

DATASET SINTÉTICO

La plataforma utiliza un dataset sintético realista diseñado específicamente para representar un entorno de fraude transaccional de alto volumen. La lógica de generación combina comportamiento normal con patrones sospechosos frecuentes en escenarios reales.

TABLAS PRINCIPALES

transactions
Tabla central del sistema.

Columnas esperadas:

- transaction_id
- timestamp
- customer_id
- merchant_id
- device_id
- payment_method_id
- channel
- transaction_amount
- currency
- merchant_category
- country
- city
- is_international
- hour_of_day
- day_of_week
- is_weekend
- auth_status
- label_fraud

customers
Perfil e histórico resumido del cliente.

- customer_id
- customer_tenure_days
- customer_segment
- age_band
- home_country
- home_city
- avg_monthly_txn_count
- avg_monthly_amount
- historical_chargeback_count
- historical_fraud_flag_count
- usual_channel
- usual_device_count

merchants
Información del comercio asociado a la transacción.

- merchant_id
- merchant_name
- merchant_category
- merchant_country
- merchant_risk_level
- avg_ticket_size
- chargeback_rate
- merchant_age_days
- is_high_risk_merchant

devices
Señales asociadas al dispositivo.

- device_id
- device_type
- os_family
- browser_family
- device_trust_score
- shared_device_flag
- num_customers_seen_last_30d

payment_methods
Información básica del método de pago.

- payment_method_id
- payment_type
- issuer_country
- card_present_flag
- prepaid_flag
- virtual_card_flag
- account_age_days

-------------------------------------------------------------------------------

PATRONES DE FRAUDE SIMULADOS

El dataset puede incorporar, entre otros, los siguientes escenarios:

Velocity fraud
Secuencias con muchas transacciones en poco tiempo asociadas al mismo cliente, dispositivo o método de pago.

Card testing
Series de importes pequeños, repetitivos y de baja cuantía, compatibles con validación fraudulenta de medios de pago.

Anomalía geográfica
Operaciones generadas desde ubicaciones incompatibles con el histórico del cliente.

Cambio brusco de comportamiento
Importes, horarios, canales o categorías muy alejados del patrón habitual.

Device anomaly
Dispositivos asociados a múltiples clientes o con actividad atípica en un intervalo corto.

Merchant concentration risk
Concentración anómala del fraude en determinados comercios o categorías.

Account takeover pattern
Cambios súbitos de dispositivo, ubicación y comportamiento transaccional compatibles con toma de control de cuenta.

-------------------------------------------------------------------------------

FEATURE ENGINEERING

La capa de variables está orientada a capturar señales útiles para scoring de fraude.

Variables transaccionales directas
- importe
- log del importe
- canal
- categoría del comercio
- hora de la operación
- indicador de fin de semana
- indicador de operación internacional

Variables históricas del cliente
- número de transacciones en distintas ventanas temporales
- importe acumulado en 24h y 7d
- ticket medio histórico
- desviación respecto a su comportamiento habitual
- número de dispositivos utilizados recientemente

Variables del merchant
- tasa histórica de fraude
- ticket medio del comercio
- categoría de riesgo
- intensidad relativa del canal o categoría

Variables del dispositivo
- número de clientes asociados
- nivel de confianza
- actividad reciente
- indicadores de compartición o reutilización

Variables geográficas
- operación fuera del país habitual
- cambio brusco de ciudad o país
- distancia aproximada respecto al patrón esperado

-------------------------------------------------------------------------------

MODELADO

La estrategia de modelado se plantea de forma progresiva.

Baseline
- Logistic Regression

Modelos principales
- Random Forest
- Gradient Boosting
- XGBoost o LightGBM

Consideraciones de evaluación
La detección de fraude es un problema típicamente desbalanceado, por lo que la evaluación no se centra en accuracy.

Métricas prioritarias
- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Recall en percentiles altos
- Precision en top-k
- fraude capturado
- coste evitado estimado
- coste operativo por falsos positivos

-------------------------------------------------------------------------------

CAPA DE DECISIÓN

El modelo produce un score continuo y posteriormente se aplica una lógica de negocio configurable:

- score bajo -> APPROVE
- score intermedio -> REVIEW
- score alto -> BLOCK

Esta separación permite mantener desacopladas:

- la capa predictiva
- la política de decisión
- la calibración de thresholds según objetivos operativos

-------------------------------------------------------------------------------

MLFLOW

El proyecto utiliza MLflow para:

- seguimiento de experimentos
- registro de hiperparámetros
- almacenamiento de métricas
- versionado de modelos
- comparación entre ejecuciones
- selección del mejor modelo final

Este enfoque permite estructurar el entrenamiento con trazabilidad y facilitar iteraciones posteriores.

-------------------------------------------------------------------------------

API REST

El modelo final se expone a través de FastAPI.

Endpoints propuestos

POST /predict
Recibe una transacción y devuelve:

- fraud_score
- risk_band
- decision
- top_risk_factors

GET /health
Verificación de salud del servicio.

GET /model-info
Devuelve información del modelo cargado:

- nombre
- versión
- fecha de entrenamiento
- métricas principales

Ejemplo de respuesta

{
  "transaction_id": "TXN_00012345",
  "fraud_score": 0.91,
  "risk_band": "high",
  "decision": "BLOCK",
  "top_risk_factors": [
    "high_velocity_1h",
    "new_device_for_customer",
    "international_transaction",
    "merchant_high_risk"
  ]
}

-------------------------------------------------------------------------------

MONITORIZACIÓN

La plataforma contempla una capa de monitorización para simular buenas prácticas de seguimiento:

- drift de variables de entrada
- drift del score
- evolución temporal del volumen de transacciones
- cambios en la tasa estimada de fraude
- estabilidad de métricas del modelo
- chequeos de calidad de datos

La monitorización refuerza la continuidad del sistema más allá de la fase de entrenamiento.

-------------------------------------------------------------------------------

ESTRUCTURA DEL REPOSITORIO

sentinelflow/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── .gitignore
├── LICENSE
│
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   ├── api_config.yaml
│   └── monitoring_config.yaml
│
├── data/
│   ├── raw/
│   ├── silver/
│   ├── gold/
│   └── sample/
│
├── notebooks/
│   ├── 01_negocio_y_objetivos.ipynb
│   ├── 02_generacion_datos_sinteticos.ipynb
│   ├── 03_eda_fraude.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_entrenamiento_mlflow.ipynb
│   ├── 06_thresholds_negocio.ipynb
│   └── 07_monitorizacion.ipynb
│
├── src/
│   ├── data/
│   │   ├── generate_transactions.py
│   │   ├── clean_transactions.py
│   │   └── build_gold_dataset.py
│   │
│   ├── features/
│   │   ├── transaction_features.py
│   │   ├── customer_features.py
│   │   ├── merchant_features.py
│   │   └── risk_aggregates.py
│   │
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   ├── register_model.py
│   │   └── thresholds.py
│   │
│   ├── inference/
│   │   ├── preprocess.py
│   │   ├── predict.py
│   │   └── explain.py
│   │
│   ├── monitoring/
│   │   ├── drift.py
│   │   ├── performance.py
│   │   └── data_quality.py
│   │
│   └── utils/
│       ├── io.py
│       ├── logging.py
│       └── schemas.py
│
├── api/
│   ├── main.py
│   ├── models.py
│   └── routers/
│       ├── predict.py
│       ├── health.py
│       └── model_info.py
│
├── dashboard/
│   └── streamlit_app.py
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_api.py
│   └── test_model.py
│
└── assets/
    ├── architecture.png
    ├── api_demo.png
    └── dashboard_demo.png

-------------------------------------------------------------------------------

FLUJO DE TRABAJO

1. Generación e ingesta
Se generan las entidades base y las transacciones con patrones normales y fraudulentos.

2. Limpieza y validación
Se revisan tipos de datos, duplicados, valores faltantes y consistencia general.

3. Construcción del dataset analítico
Se integran tablas, se construyen variables históricas y se consolida la capa gold.

4. Entrenamiento y tracking
Se entrenan distintos modelos, se registran experimentos en MLflow y se comparan resultados.

5. Selección de thresholds
Se calibran puntos de corte según objetivos de negocio y operación.

6. Serving
Se publica el mejor modelo a través de una API REST con FastAPI.

7. Monitorización
Se analizan drift, estabilidad del score y métricas operativas.

-------------------------------------------------------------------------------

ROADMAP DE IMPLEMENTACIÓN

Fase 1 — Diseño funcional
- definir el caso de uso
- modelar entidades y relaciones
- establecer métricas de éxito

Fase 2 — Dataset sintético
- generar customers, merchants, devices y payment methods
- generar transacciones legítimas
- inyectar patrones de fraude

Fase 3 — EDA
- validar distribuciones
- analizar fraude por canal, merchant y franja horaria
- comprobar desbalance

Fase 4 — Feature engineering
- crear variables históricas
- consolidar dataset final
- documentar el significado de las features

Fase 5 — Modelado y MLflow
- entrenar baseline
- entrenar modelos principales
- comparar métricas
- registrar el mejor modelo

Fase 6 — API FastAPI
- construir endpoint de predicción
- validar inputs
- devolver score y decisión

Fase 7 — Monitorización
- crear chequeos de drift
- evaluar estabilidad del score
- preparar dashboard demo

Fase 8 — Endurecimiento técnico
- incorporar tests adicionales
- mejorar validaciones
- documentar ejecución y resultados
- preparar la solución para extensiones posteriores

-------------------------------------------------------------------------------

CÓMO EJECUTAR EL PROYECTO

1. Clonar el repositorio

git clone <URL_DEL_REPOSITORIO>
cd sentinelflow

2. Crear entorno virtual

python -m venv .venv

3. Activar entorno

Windows
.venv\Scripts\activate

Linux / macOS
source .venv/bin/activate

4. Instalar dependencias

pip install -r requirements.txt

5. Ejecutar el pipeline principal

python src/data/generate_transactions.py
python src/data/build_gold_dataset.py
python src/training/train_model.py

6. Lanzar MLflow

mlflow ui

7. Levantar la API

uvicorn api.main:app --reload

8. Lanzar dashboard

streamlit run dashboard/streamlit_app.py

-------------------------------------------------------------------------------

STACK TECNOLÓGICO

- Lenguaje: Python
- Procesamiento de datos: pandas, numpy
- Modelado: scikit-learn, XGBoost / LightGBM
- Tracking y lifecycle: MLflow
- API: FastAPI
- Visualización y monitorización: Streamlit, matplotlib, plotly
- Serialización: joblib / pickle
- Calidad y pruebas: pytest
- Contenerización: Docker
- Control de versiones: Git

-------------------------------------------------------------------------------

EXTENSIONES PREVISTAS

- incorporación de procesamiento distribuido real
- orquestación con Airflow
- despliegue en servicios cloud
- integración de CI/CD
- explicabilidad avanzada con SHAP
- scoring casi en tiempo real
- autenticación y control de acceso para la API
- integración con una feature store

-------------------------------------------------------------------------------

DISCLAIMER

Este repositorio utiliza datos sintéticos y no contiene información real de clientes, transacciones ni entidades financieras. Los nombres, patrones y escenarios incluidos tienen únicamente fines de desarrollo, validación técnica y simulación controlada.

-------------------------------------------------------------------------------

AUTOR

Desarrollado por David Olivares.
