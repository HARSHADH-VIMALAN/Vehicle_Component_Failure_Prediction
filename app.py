# ==========================================================
# AI-Based Predictive Maintenance System
# Roots Industries Internship Project
# Vehicle Component Failure Prediction
# ==========================================================

# -----------------------------
# IMPORT LIBRARIES
# -----------------------------

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from datetime import datetime

import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(

    page_title="AI Predictive Maintenance",

    page_icon="🚗",

    layout="wide",

    initial_sidebar_state="expanded"

)

st.markdown("""
<style>

/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

/* Main Background */
.stApp{
    background:linear-gradient(135deg,#eef5ff,#dbeafe,#f8fbff);
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0f172a,#1e3a8a);
}

section[data-testid="stSidebar"] *{
    color:white;
}

/* Metric Cards */
[data-testid="metric-container"]{
    background:white;
    border-radius:18px;
    padding:18px;
    border-left:6px solid #2563eb;
    box-shadow:0 8px 18px rgba(0,0,0,0.08);
}

/* Buttons */
.stButton>button{
    width:100%;
    border-radius:12px;
    height:50px;
    background:#2563eb;
    color:white;
    border:none;
    font-weight:bold;
    font-size:16px;
}

.stButton>button:hover{
    background:#1d4ed8;
    transform:scale(1.02);
    transition:0.3s;
}

/* Dataframes */
[data-testid="stDataFrame"]{
    border-radius:15px;
    overflow:hidden;
}

/* Progress */
.stProgress>div>div>div{
    background:#16a34a;
}

</style>
""",unsafe_allow_html=True)

# -----------------------------
# SESSION STATE
# -----------------------------

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "model" not in st.session_state:
    st.session_state.model = None

if "history" not in st.session_state:
    st.session_state.history = []

if "accuracy" not in st.session_state:
    st.session_state.accuracy = None

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background:#F4F8FB;
}

.block-container{
    padding-top:1rem;
    padding-bottom:1rem;
}

h1,h2,h3{
    color:#0F172A;
    font-weight:700;
}

[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0F172A,#1E3A8A);
}

[data-testid="stSidebar"] *{
    color:white;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:14px;
    padding:18px;
    box-shadow:0 4px 15px rgba(0,0,0,0.10);
}

.stButton>button{
    width:100%;
    border-radius:10px;
    height:48px;
    border:none;
    background:#2563EB;
    color:white;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1D4ED8;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🚗 AI Predictive Maintenance")

st.sidebar.markdown(
"""
Vehicle Component Failure Prediction

**Roots Industries Internship Project**
"""
)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    st.session_state.dataset = pd.read_csv(uploaded_file)

    st.sidebar.success("Dataset Loaded")

st.sidebar.markdown("---")

menu = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Dashboard",

        "📂 Dataset",

        "📊 Data Analysis",

        "🤖 Model Training",

        "📈 Model Comparison",

        "🔍 Prediction",

        "📜 Prediction History",

        "🛠 Maintenance",

        "📄 Report",

        "🤖 AI Expert Opinion"

    ]

)

st.sidebar.markdown("---")

st.sidebar.info(

f"""

📅 Date

{datetime.now().strftime("%d-%m-%Y")}

⏰ Time

{datetime.now().strftime("%H:%M:%S")}

"""

)

# ==========================================================
# GLOBAL DATASET
# ==========================================================

df = st.session_state.dataset
model = st.session_state.model

# ==========================================================
# DASHBOARD
# ==========================================================

if menu == "🏠 Dashboard":

    st.title("🏠 Dashboard")

    st.info("""
    ### 📌 Internship Project

    **Company :** ROOTS INDUSTRIES INDIA LTD.

    **Domain :** Predictive Maintenance using Artificial Intelligence

    **Technology :** Python | Streamlit | Machine Learning | Data Analytics

    **Objective :**
    Predict component failures before breakdown occurs to improve reliability,
    reduce downtime and support preventive maintenance.
    """)

    st.markdown("""
    # 🏭 ROOTS INDUSTRIES

    ## AI Powered Predictive Maintenance System

    ### Industrial Equipment Health Monitoring Dashboard
    """)
    

    if df is None:

        st.warning("📂 Please upload a dataset from the sidebar to continue.")

        st.stop()

    # ------------------------------------------------------
    # KPI CALCULATIONS
    # ------------------------------------------------------

    total_components = len(df)

    total_failures = int(df["failures"].sum())

    healthy_components = total_components - total_failures

    failure_rate = round(
        (total_failures / total_components) * 100,
        2
    )

    avg_temperature = round(df["temperature"].mean(), 2)

    avg_vibration = round(df["vibration"].mean(), 2)

    # ------------------------------------------------------
    # KPI CARDS
    # ------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Components",
        df.shape[0]
    )

    c2.metric(
        "Failures",
        int(df["failures"].sum())
    )

    c3.metric(
        "Healthy",
        int(len(df) - df["failures"].sum())
    )

    c4.metric(
        "Failure Rate",
        f"{df['failures'].mean() * 100:.1f}%"
    )

    # ------------------------------------------------------
    # PIE CHART
    # ------------------------------------------------------

    left, right = st.columns([1, 1])

    with left:

        st.subheader("Component Health Distribution")

        health = pd.DataFrame({

            "Status": ["Healthy", "Failure"],

            "Count": [

                healthy_components,

                total_failures

            ]

        })

        fig = px.pie(

            health,

            names="Status",

            values="Count",

            hole=0.55,

            color="Status",

            color_discrete_map={

                "Healthy": "#16A34A",

                "Failure": "#DC2626"

            }

        )

        fig.update_layout(

            height=430

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    # ------------------------------------------------------
    # BAR CHART
    # ------------------------------------------------------

    with right:

        st.subheader("Average Operating Parameters")

        avg_df = pd.DataFrame({

            "Parameter": [

                "Temperature",

                "Vibration",

                "Duty Cycle",

                "Running Hours"

            ],

            "Average": [

                df["temperature"].mean(),

                df["vibration"].mean(),

                df["duty_cycle"].mean(),

                df["running_hours"].mean()

            ]

        })

        fig = px.bar(

            avg_df,

            x="Parameter",

            y="Average",

            color="Average",

            text_auto=".2f",

            color_continuous_scale="Blues"

        )

        fig.update_layout(

            height=430,

            coloraxis_showscale=False

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.markdown("---")

    # ------------------------------------------------------
    # DATASET PREVIEW
    # ------------------------------------------------------

    st.subheader("Recent Dataset Records")

    st.dataframe(

        df.head(10),

        use_container_width=True

    )

# ==========================================================
# DATA ANALYSIS MODULE
# ==========================================================

if menu == "📊 Data Analysis":

    st.title("📊 Industrial Data Analysis")

    if df is None:

        st.warning("📂 Please upload a dataset.")

        st.stop()

    st.subheader("Dataset Preview")

    st.dataframe(df.head(15), use_container_width=True)

    st.markdown("---")

    st.subheader("Dataset Information")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", df.shape[0])

    c2.metric("Columns", df.shape[1])

    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("---")

    st.subheader("Column Data Types")

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Datatype": df.dtypes.astype(str)
    })

    st.dataframe(dtype_df, use_container_width=True)

    st.markdown("---")

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(include="all"),
        use_container_width=True
    )

    st.markdown("---")

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_columns) == 0:

        st.error("No numerical columns found.")

        st.stop()

    selected_feature = st.selectbox(
        "Select Feature",
        numeric_columns
    )

    st.markdown("---")

    st.subheader(f"Distribution of {selected_feature}")

    fig = px.histogram(

        df,

        x=selected_feature,

        nbins=35,

        marginal="box",

        color_discrete_sequence=["royalblue"]

    )

    fig.update_layout(height=500)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader(f"Box Plot of {selected_feature}")

    fig = px.box(

        df,

        y=selected_feature,

        color_discrete_sequence=["crimson"]

    )

    fig.update_layout(height=500)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Correlation Heatmap")

    corr = df[numeric_columns].corr()

    fig = px.imshow(

        corr,

        text_auto=".2f",

        color_continuous_scale="RdBu_r",

        aspect="auto"

    )

    fig.update_layout(
        height=700
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Scatter Plot")

    col1, col2 = st.columns(2)

    with col1:

        x_axis = st.selectbox(

            "X Axis",

            numeric_columns,

            key="scatter_x"

        )

    with col2:

        y_axis = st.selectbox(

            "Y Axis",

            numeric_columns,

            index=1,

            key="scatter_y"

        )

    fig = px.scatter(

        df,

        x=x_axis,

        y=y_axis,

        color="failures" if "failures" in df.columns else None,

        height=550

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if "failures" in df.columns:

        st.markdown("---")

        st.subheader("Failure Distribution")

        failure_df = pd.DataFrame(

            df["failures"].value_counts()

        ).reset_index()

        failure_df.columns = [

            "Status",

            "Count"

        ]

        failure_df["Status"] = failure_df["Status"].replace({

            0: "Healthy",

            1: "Failure"

        })

        fig = px.pie(

            failure_df,

            names="Status",

            values="Count",

            hole=0.55,

            color="Status",

            color_discrete_map={

                "Healthy": "#16A34A",

                "Failure": "#DC2626"

            }

        )

        fig.update_layout(height=500)

        st.plotly_chart(

            fig,

            use_container_width=True

        )
    
# ==========================================================
# MODEL TRAINING CENTER
# ==========================================================

if menu == "🤖 Model Training":

    st.title("🤖 AI Model Training Center")

    if df is None:

        st.warning("📂 Please upload a dataset.")

        st.stop()

    if "failures" not in df.columns:

        st.error("Dataset must contain a 'failures' column.")

        st.stop()

    st.subheader("Dataset Ready for Training")

    st.write(f"**Rows:** {df.shape[0]}")
    st.write(f"**Columns:** {df.shape[1]}")

    # --------------------------------------------
    # Feature Selection
    # --------------------------------------------

    feature_columns = [
        "component_type",
        "equipment_age_years",
        "ambient_temperature",
        "humidity",
        "temperature",
        "vibration",
        "voltage",
        "current",
        "pressure",
        "rpm",
        "duty_cycle",
        "running_hours",
        "machine_load",
        "maintenance_history"
    ]

    X = df[feature_columns].copy()

    # Encode categorical column
    le = LabelEncoder()
    X["component_type"] = le.fit_transform(X["component_type"])

    y = df["failures"]

    st.session_state.label_encoder = le
    st.session_state.feature_columns = feature_columns

    # Encode categorical columns

    object_cols = X.select_dtypes(include=["object"]).columns

    encoders = {}

    for col in object_cols:

        le = LabelEncoder()

        X[col] = le.fit_transform(X[col])

        encoders[col] = le

    # Save encoders

    st.session_state.encoders = encoders

    # Train/Test Split

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    if st.button("🚀 Train AI Model"):

        progress = st.progress(0)

        status = st.empty()

        status.info("Training Random Forest...")

        model = RandomForestClassifier(

            n_estimators=300,

            random_state=42

        )

        progress.progress(20)

        model.fit(X_train, y_train)

        progress.progress(60)

        prediction = model.predict(X_test)

        progress.progress(80)

        accuracy = accuracy_score(y_test, prediction)

        precision = precision_score(y_test, prediction)

        recall = recall_score(y_test, prediction)

        f1 = f1_score(y_test, prediction)

        progress.progress(100)

        status.success("Training Completed Successfully")

        st.session_state.model = model

        st.session_state.feature_columns = list(X.columns)

        st.session_state.accuracy = accuracy

        joblib.dump(model, "vehicle_failure_model.pkl")
    
        st.markdown("---")

        st.subheader("Model Performance")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(

            "Accuracy",

            f"{accuracy*100:.2f}%"

        )

        c2.metric(

            "Precision",

            f"{precision*100:.2f}%"

        )

        c3.metric(

            "Recall",

            f"{recall*100:.2f}%"

        )

        c4.metric(

            "F1 Score",

            f"{f1*100:.2f}%"

        )
    
        st.markdown("---")

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(

            y_test,

            prediction

        )

        fig = px.imshow(

            cm,

            text_auto=True,

            color_continuous_scale="Blues"

        )

        fig.update_layout(

            height=450

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )
    
        st.markdown("---")

        st.subheader("Feature Importance")

        importance = pd.DataFrame({

            "Feature": X.columns,

            "Importance": model.feature_importances_

        })

        importance = importance.sort_values(

            by="Importance",

            ascending=False

        )

        fig = px.bar(

            importance,

            x="Importance",

            y="Feature",

            orientation="h",

            color="Importance",

            color_continuous_scale="Viridis"

        )

        fig.update_layout(

            height=550,

            yaxis=dict(categoryorder="total ascending")

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        st.success("✅ Model saved as vehicle_failure_model.pkl")

# ==========================================================
# MODEL COMPARISON CENTER
# ==========================================================

if menu == "📈 Model Comparison":

    st.title("📈 Machine Learning Model Comparison")

    if df is None:

        st.warning("📂 Please upload a dataset.")

        st.stop()

    if "failures" not in df.columns:

        st.error("Dataset must contain a 'failures' column.")

        st.stop()

    # ==========================
    # Features used for training
    # ==========================

    feature_columns = [
        "component_type",
        "equipment_age_years",
        "ambient_temperature",
        "humidity",
        "temperature",
        "vibration",
        "voltage",
        "current",
        "pressure",
        "rpm",
        "duty_cycle",
        "running_hours",
        "machine_load",
        "maintenance_history"
    ]

    # Keep only these columns
    X = df[feature_columns].copy()

    # Encode component type
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()

    X["component_type"] = le.fit_transform(X["component_type"])

    # Target column
    y = df["failures"]

    # Encode categorical columns

    object_cols = X.select_dtypes(include=["object"]).columns

    for col in object_cols:

        le = LabelEncoder()

        X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    if st.button("🚀 Compare All Models"):

        models = {

            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                random_state=42
            ),

            "Decision Tree": DecisionTreeClassifier(
                random_state=42
            ),

            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42
            ),

            "Logistic Regression": LogisticRegression(
                max_iter=1000
            ),

            "Support Vector Machine": SVC(
                probability=True
            )

        }

        progress = st.progress(0)

        results = []

        trained_models = {}

        for i, (name, model) in enumerate(models.items()):

            model.fit(

                X_train,

                y_train

            )

            pred = model.predict(

                X_test

            )

            acc = accuracy_score(

                y_test,

                pred

            )

            pre = precision_score(

                y_test,

                pred

            )

            rec = recall_score(

                y_test,

                pred

            )

            f1 = f1_score(

                y_test,

                pred

            )

            results.append({

                "Model": name,

                "Accuracy": round(acc * 100, 2),

                "Precision": round(pre * 100, 2),

                "Recall": round(rec * 100, 2),

                "F1 Score": round(f1 * 100, 2)

            })

            trained_models[name] = model

            progress.progress((i + 1) / len(models))
        
        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(

            by="Accuracy",

            ascending=False

        )

        results_df.index = np.arange(

            1,

            len(results_df) + 1

        )

        st.success("✅ Model Comparison Completed")

        st.subheader("🏆 Model Leaderboard")

        st.dataframe(

            results_df,

            use_container_width=True

        )

        fig = px.bar(

            results_df,

            x="Model",

            y="Accuracy",

            color="Accuracy",

            text="Accuracy",

            color_continuous_scale="Viridis"

        )

        fig.update_traces(

            textposition="outside"

        )

        fig.update_layout(

            height=550,

            coloraxis_showscale=False

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        best_model_name = results_df.iloc[0]["Model"]

        best_model = trained_models[best_model_name]

        st.session_state.model = best_model

        st.session_state.feature_columns = list(X.columns)

        joblib.dump(

            best_model,

            "best_vehicle_model.pkl"

        )

        st.success(

            f"🏆 Best Model : {best_model_name}"

        )

        st.info(

            "Best model automatically saved as best_vehicle_model.pkl"

        )

# ==========================================================
# AI PREDICTION CENTER
# ==========================================================

if menu == "🔍 Prediction":

    st.title("🔍 Vehicle Component Failure Prediction")

    if st.session_state.model is None:

        st.warning("⚠ Please train a model first.")

        st.stop()

    model = st.session_state.model

    st.markdown("---")

    st.subheader("Enter Component Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:

        component_type = st.selectbox(

            "Component Type",

            ["Horn", "Pump", "Motor"]

        )

        temperature = st.number_input(

            "Temperature (°C)",

            0.0,

            150.0,

            65.0

        )

        vibration = st.number_input(

            "Vibration (mm/s)",

            0.0,

            5.0,

            0.35

        )

    with col2:

        voltage = st.number_input(

            "Voltage (V)",

            150.0,

            300.0,

            230.0

        )

        current = st.number_input(

            "Current (A)",

            0.0,

            50.0,

            8.0

        )

        pressure = st.number_input(

            "Pressure (PSI)",

            0.0,

            200.0,

            85.0

        )

    with col3:

        rpm = st.number_input(

            "RPM",

            0,

            5000,

            1800

        )

        duty_cycle = st.number_input(

            "Duty Cycle (%)",

            0,

            100,

            70

        )

        running_hours = st.number_input(

            "Running Hours",

            0,

            5000,

            1200

        )

        machine_load = st.slider(

            "Machine Load (%)",

            0,

            100,

            55

        )

        st.markdown("---")

    if st.button("🚀 Predict Failure"):

        component_map = {

            "Horn":0,

            "Pump":1,

            "Motor":2

        }

        component_map = {
            "Horn": "Horn",
            "Pump": "Pump",
            "Motor": "Motor"
        }

        input_df = pd.DataFrame({
            "component_type": [component_map[component_type]],
            "equipment_age_years": [5],
            "ambient_temperature": [30],
            "humidity": [60],
            "temperature": [temperature],
            "vibration": [vibration],
            "voltage": [voltage],
            "current": [current],
            "pressure": [pressure],
            "rpm": [rpm],
            "duty_cycle": [duty_cycle],
            "running_hours": [running_hours],
            "machine_load": [machine_load],
            "maintenance_history": [3]
        })

        input_df["component_type"] = st.session_state.label_encoder.transform(
            input_df["component_type"]
        )

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0]

        confidence = max(probability) * 100

        failure_probability = probability[1] * 100

        st.markdown("---")

        c1,c2,c3 = st.columns(3)

        with c1:

            if prediction == 1:

                st.error("❌ FAILURE PREDICTED")

            else:

                st.success("✅ COMPONENT HEALTHY")

        with c2:

            st.metric(

                "Failure Probability",

                f"{failure_probability:.2f}%"

            )

        with c3:

            st.metric(

                "Model Confidence",

                f"{confidence:.2f}%"

            )

        st.markdown("---")

        st.subheader("Industrial Risk Meter")

        fig = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=failure_probability,

                title={"text":"Risk Score"},

                gauge={

                    "axis":{"range":[0,100]},

                    "bar":{"color":"darkred"},

                    "steps":[

                        {"range":[0,30],"color":"green"},

                        {"range":[30,60],"color":"yellow"},

                        {"range":[60,80],"color":"orange"},

                        {"range":[80,100],"color":"red"}

                    ]

                }

            )

        )

        fig.update_layout(

            height=400

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )    

        st.markdown("---")

        st.subheader("Parameter Health Check")

        safe_limits = {

            "Temperature (°C)":(45,75,temperature),

            "Vibration":(0.2,0.6,vibration),

            "Voltage":(210,240,voltage),

            "Current":(4,12,current),

            "Pressure":(60,110,pressure),

            "RPM":(1000,3000,rpm),

            "Duty Cycle":(40,90,duty_cycle),

            "Running Hours":(0,3000,running_hours),

            "Machine Load":(20,90,machine_load)

        }

        result=[]

        for name,(low,high,val) in safe_limits.items():

            status="🟢 Safe"

            if val<low or val>high:

                status="🔴 Critical"

            result.append([name,val,f"{low}-{high}",status])

        safe_df=pd.DataFrame(

            result,

            columns=[

                "Parameter",

                "Current",

                "Safe Range",

                "Status"

            ]

        )

        st.dataframe(

            safe_df,

            use_container_width=True

        )

        st.markdown("---")

        st.subheader("Maintenance Recommendation")

        recommendations=[]

        if temperature>75:

            recommendations.append("✔ Reduce operating temperature.")

        if vibration>0.6:

            recommendations.append("✔ Inspect bearings and alignment.")

        if voltage<210 or voltage>240:

            recommendations.append("✔ Check electrical supply.")

        if current>12:

            recommendations.append("✔ Inspect motor winding.")

        if pressure>110:

            recommendations.append("✔ Check pressure valve.")

        if running_hours>3000:

            recommendations.append("✔ Schedule preventive maintenance.")

        if machine_load>90:

            recommendations.append("✔ Reduce machine load.")

        if len(recommendations)==0:

            st.success("✅ Component operating within recommended limits.")

        else:

            for rec in recommendations:

                st.warning(rec)

        # ------------------------------------
        # SAVE PREDICTION HISTORY
        # ------------------------------------

        history_entry = {

            "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),

            "Component": component_type,

            "Temperature": temperature,

            "Vibration": vibration,

            "Voltage": voltage,

            "Current": current,

            "Pressure": pressure,

            "RPM": rpm,

            "Duty Cycle": duty_cycle,

            "Running Hours": running_hours,

            "Machine Load": machine_load,

            "Prediction": "Failure" if prediction==1 else "Healthy",

            "Failure Probability (%)": round(failure_probability,2),

            "Confidence (%)": round(confidence,2)

        }

        st.session_state.history.append(history_entry)

# ==========================================================
# PREDICTION HISTORY
# ==========================================================

if menu == "📜 Prediction History":

    st.title("📜 Prediction History")

    if len(st.session_state.history) == 0:

        st.info("No predictions available.")

        st.stop()

    history_df = pd.DataFrame(st.session_state.history)

    total_predictions = len(history_df)

    total_failures = len(
        history_df[
            history_df["Prediction"]=="Failure"
        ]
    )

    healthy = total_predictions-total_failures

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Total Predictions",
        total_predictions
    )

    c2.metric(
        "Healthy",
        healthy
    )

    c3.metric(
        "Failures",
        total_failures
    )

    st.markdown("---")

    search = st.text_input(
        "Search Component"
    )

    if search:

        history_df = history_df[
            history_df["Component"].str.contains(
                search,
                case=False
            )
        ]

    st.markdown("---")

    st.subheader("Prediction Records")

    st.dataframe(

        history_df,

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("Failure Probability Trend")

    fig = px.line(

        history_df,

        x="Time",

        y="Failure Probability (%)",

        markers=True

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("Prediction Distribution")

    pie = history_df["Prediction"].value_counts().reset_index()

    pie.columns = [

        "Prediction",

        "Count"

    ]

    fig = px.pie(

        pie,

        names="Prediction",

        values="Count",

        hole=0.55,

        color="Prediction",

        color_discrete_map={

            "Healthy":"green",

            "Failure":"red"

        }

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    csv = history_df.to_csv(index=False).encode()

    st.download_button(

        "⬇ Download History",

        csv,

        "prediction_history.csv",

        "text/csv"

    )

# ==========================================================
# MAINTENANCE COMMAND CENTER
# ==========================================================

if menu == "🛠 Maintenance":

    st.title("🛠 Maintenance Command Center")

    if len(st.session_state.history) == 0:

        st.warning("No prediction history available.")

        st.stop()

    latest = st.session_state.history[-1]

    st.subheader("Latest Component Status")

    probability = latest["Failure Probability (%)"]

    health_score = round(100 - probability,2)

    c1,c2,c3 = st.columns(3)

    c1.metric(

        "Health Score",

        f"{health_score}%"

    )

    c2.metric(

        "Failure Risk",

        f"{probability}%"

    )

    confidence = latest.get("Confidence (%)", 0)

    c3.metric(

        "Model Confidence",

        f"{confidence}%"

    )

    st.markdown("---")

    if probability < 30:

        priority = "🟢 LOW"

    elif probability < 60:

        priority = "🟡 MEDIUM"

    elif probability < 80:

        priority = "🟠 HIGH"

    else:

        priority = "🔴 CRITICAL"

    st.subheader("Maintenance Priority")

    st.info(priority)

    running_hours = latest["Running Hours"]

    estimated_life = max(0,4000-running_hours)

    st.markdown("---")

    st.subheader("Estimated Remaining Useful Life")

    st.metric(

        "Remaining Hours",

        estimated_life

    )

    service_due = max(0,3000-running_hours)

    st.subheader("Next Preventive Service")

    if service_due == 0:

        st.error(

            "Service Required Immediately"

        )

    else:

        st.success(

            f"Service after {service_due} hours"

        )

    st.markdown("---")

    st.subheader("Parameter Health")

    parameters = {

        "Temperature": latest["Temperature"]/100,

        "Vibration": latest["Vibration"],

        "Voltage": latest["Voltage"]/250,

        "Current": latest["Current"]/20,

        "Pressure": latest["Pressure"]/150,

        "RPM": latest["RPM"]/4000,

        "Machine Load": latest["Machine Load"]/100

    }

    for name,value in parameters.items():

        st.write(name)

        st.progress(

            min(float(value),1.0)

        )

    st.markdown("---")

    st.subheader("Recommended Maintenance Checklist")

    checklist = [

        "Inspect Bearings",

        "Check Lubrication",

        "Inspect Electrical Connections",

        "Check Temperature Sensors",

        "Check Voltage Stability",

        "Inspect Cooling System",

        "Perform Visual Inspection"

    ]

    for item in checklist:

        st.checkbox(item)

# ==========================================================
# DATASET EXPLORER
# ==========================================================

if menu == "📂 Dataset":

    st.title("📂 Dataset Explorer")

    if df is None:

        st.warning("Please upload a dataset.")

        st.stop()

    st.subheader("Dataset Preview")

    st.dataframe(

        df,

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("Dataset Summary")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(

        "Rows",

        df.shape[0]

    )

    c2.metric(

        "Columns",

        df.shape[1]

    )

    c3.metric(

        "Missing Values",

        int(df.isnull().sum().sum())

    )

    c4.metric(

        "Duplicate Rows",

        int(df.duplicated().sum())

    )

    st.markdown("---")

    st.subheader("Missing Values")

    missing = pd.DataFrame(

        df.isnull().sum(),

        columns=["Missing"]

    )

    st.dataframe(

        missing,

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("Column Explorer")

    selected_column = st.selectbox(

        "Choose Column",

        df.columns

    )

    st.write(df[selected_column])

    st.markdown("---")

    st.subheader("Statistical Summary")

    st.dataframe(

        df.describe(

            include="all"

        ),

        use_container_width=True

    )

    st.markdown("---")

    csv = df.to_csv(

        index=False

    ).encode()

    st.download_button(

        "⬇ Download Dataset",

        csv,

        "dataset.csv",

        "text/csv"

    )

# ==========================================================
# PDF REPORT
# ==========================================================

if menu == "📄 Report":

    st.title("📄 AI Maintenance Report")

    if len(st.session_state.history) == 0:

        st.warning("No prediction history available.")

        st.stop()

    latest = st.session_state.history[-1]

    st.subheader("Latest Prediction")

    st.dataframe(

        pd.DataFrame([latest]),

        use_container_width=True
    )

    def create_pdf():

        doc = SimpleDocTemplate(

            "Vehicle_Maintenance_Report.pdf"

        )

        styles = getSampleStyleSheet()

        story=[]

        story.append(

            Paragraph(

                "<b>ROOTS INDUSTRIES INDIA LTD.</b>",

                styles["Title"]

            )

        )

        story.append(

            Paragraph(

                "AI-Based Predictive Maintenance Report",

                styles["Heading2"]

            )

        )

        story.append(

            Paragraph(

                "<br/>",

                styles["Normal"]

            )

        )

        for key,value in latest.items():

            story.append(

                Paragraph(

                    f"<b>{key}</b> : {value}",

                    styles["BodyText"]

                )

            )

        story.append(

            Paragraph(

                "<br/><br/>",

                styles["Normal"]

            )

        )

        story.append(

            Paragraph(

                "Generated using AI-Based Predictive Maintenance System.",

                styles["Italic"]

            )

        )

        doc.build(

            story

        )

    if st.button(

        "📄 Generate PDF Report"

    ):

        create_pdf()

        with open(

            "Vehicle_Maintenance_Report.pdf",

            "rb"

        ) as file:

            st.download_button(

                "⬇ Download PDF",

                file,

                "Vehicle_Maintenance_Report.pdf",

                mime="application/pdf"

            )

        st.success(

            "PDF Report Generated Successfully."

        )

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;
font-size:16px;
color:gray;'>

🏭 ROOTS INDUSTRIES INDIA LTD.<br>

AI Powered Vehicle Component Failure Prediction System<br>

Developed using Python • Streamlit • Machine Learning

</div>
""",
unsafe_allow_html=True
)

# ==========================================================
# 🤖 AI EXPERT OPINION
# ==========================================================

if menu == "🤖 AI Expert Opinion":

    from groq import Groq

    # ------------------------------------------------------
    # Enter your Groq API Key here
    # ------------------------------------------------------
    GROQ_API_KEY = "YOUR_API_KEY"

    client = Groq(api_key=GROQ_API_KEY)

    st.title("🤖 AI Expert Opinion")

    st.write(
        "Generate an AI-powered maintenance report for the selected component."
    )

    st.divider()

    component_type = st.selectbox(
        "Component Type",
        ["Horn", "Pump", "Motor"]
    )

    col1, col2 = st.columns(2)

    with col1:

        temperature = st.number_input(
            "Temperature (°C)",
            value=72.0
        )

        voltage = st.number_input(
            "Voltage (V)",
            value=230.0
        )

        pressure = st.number_input(
            "Pressure (PSI)",
            value=95.0
        )

        duty_cycle = st.number_input(
            "Duty Cycle (%)",
            value=70
        )

        machine_load = st.number_input(
            "Machine Load (%)",
            value=60
        )

    with col2:

        vibration = st.number_input(
            "Vibration",
            value=0.40
        )

        current = st.number_input(
            "Current (A)",
            value=8.0
        )

        rpm = st.number_input(
            "RPM",
            value=2200
        )

        running_hours = st.number_input(
            "Running Hours",
            value=1800
        )

    st.divider()

    if st.button("🧠 Generate AI Expert Opinion", use_container_width=True):

        prompt = f"""
You are a Senior Predictive Maintenance Engineer at Roots Industries.

Analyze the following vehicle component.

Component Type: {component_type}

Temperature: {temperature} °C
Vibration: {vibration}
Voltage: {voltage} V
Current: {current} A
Pressure: {pressure} PSI
RPM: {rpm}
Duty Cycle: {duty_cycle} %
Running Hours: {running_hours}
Machine Load: {machine_load} %

Prepare a professional maintenance report.

Include:

1. Overall Health
2. Risk Level
3. Root Cause Analysis
4. Components to Inspect
5. Maintenance Recommendation
6. Preventive Actions
7. Estimated Remaining Useful Life
8. Business Impact
9. Final Recommendation

Use headings and bullet points.
"""

        with st.spinner("🤖 AI is generating the report..."):

            try:

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )

                st.success("✅ AI Report Generated Successfully")

                st.markdown(response.choices[0].message.content)

            except Exception as e:

                st.error(f"❌ {e}")





