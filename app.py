import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioAI · Heart Risk Analyzer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

/* ── Root theme ── */
:root {
    --bg:       #0A0F1E;
    --card:     #111827;
    --card2:    #1A2035;
    --accent:   #00D4FF;
    --danger:   #FF3B6B;
    --safe:     #00FF88;
    --purple:   #8B5CF6;
    --text:     #E2E8F0;
    --muted:    #64748B;
}

/* ── App background ── */
.stApp { background: var(--bg); }
.stApp > header { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid rgba(0,212,255,0.15);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #0D1B2A 0%, #1A0A2E 50%, #0A1628 100%);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(0,212,255,0.05) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(139,92,246,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 900;
    color: var(--accent);
    letter-spacing: 0.05em;
    margin: 0;
    text-shadow: 0 0 30px rgba(0,212,255,0.4);
}
.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: var(--muted);
    margin-top: 0.4rem;
    letter-spacing: 0.02em;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    color: var(--accent);
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-right: 0.5rem;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--card2);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: rgba(0,212,255,0.35); }
.metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
}

/* ── Section headers ── */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(0,212,255,0.2);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* ── Result banners ── */
.result-high {
    background: linear-gradient(135deg, rgba(255,59,107,0.15), rgba(255,59,107,0.05));
    border: 1px solid rgba(255,59,107,0.4);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-low {
    background: linear-gradient(135deg, rgba(0,255,136,0.12), rgba(0,255,136,0.04));
    border: 1px solid rgba(0,255,136,0.4);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    color: var(--muted);
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF, #8B5CF6) !important;
    color: #000 !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 2rem !important;
    width: 100%;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    color: var(--muted);
    font-size: 0.82rem;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Divider ── */
hr { border-color: rgba(0,212,255,0.12) !important; }

/* ── Plotly charts transparent bg ── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load model (with graceful fallback for demo) ────────────────────────────
@st.cache_resource
def load_model():
    try:
        model   = joblib.load('knn_heart_model.pkl')
        scaler  = joblib.load('heart_scaler.pkl')
        columns = joblib.load('heart_columns.pkl')
        return model, scaler, columns, True
    except Exception:
        return None, None, None, False

model, scaler, expected_columns, model_loaded = load_model()


# ─── Plotly layout defaults ──────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#94A3B8', size=11),
    colorway=['#00D4FF', '#8B5CF6', '#FF3B6B', '#00FF88', '#F59E0B'],
)


# ─── ECG animation (signature element) ──────────────────────────────────────
def ecg_chart():
    t = np.linspace(0, 4 * np.pi, 400)
    ecg = np.sin(t) * 0.1
    # PQRST spikes
    for i in [60, 160, 260, 360]:
        if i + 20 < len(ecg):
            ecg[i:i+5]   -= 0.15   # Q
            ecg[i+5:i+10] = 1.1    # R
            ecg[i+10:i+15] = -0.3  # S
            ecg[i+15:i+20] += 0.3  # T

    fig = go.Figure(go.Scatter(
        y=ecg, x=list(range(len(ecg))),
        mode='lines',
        line=dict(color='#00D4FF', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0,212,255,0.04)',
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=90,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[-0.6, 1.4]),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


# ─── Risk gauge ──────────────────────────────────────────────────────────────
def risk_gauge(prob):
    color = '#FF3B6B' if prob > 0.5 else '#00FF88'
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        number=dict(suffix="%", font=dict(size=36, color=color, family='Orbitron, monospace')),
        delta=dict(reference=50, valueformat=".1f"),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor='#334155',
                      tickfont=dict(color='#64748B', size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0,
            steps=[
                dict(range=[0, 30],  color='rgba(0,255,136,0.08)'),
                dict(range=[30, 60], color='rgba(245,158,11,0.08)'),
                dict(range=[60, 100],color='rgba(255,59,107,0.08)'),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.75, value=prob*100),
        ),
        title=dict(text="CARDIAC RISK SCORE", font=dict(family='Orbitron, monospace',
                   color='#64748B', size=11)),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=260)
    return fig


# ─── Population comparison radar ─────────────────────────────────────────────
def radar_chart(patient_vals, categories):
    avg_healthy = [45, 120, 200, 140, 1.0, 0.5]
    avg_risk    = [60, 145, 270, 120, 2.5, 0.8]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=patient_vals, theta=categories,
        fill='toself', name='You',
        line=dict(color='#00D4FF', width=2),
        fillcolor='rgba(0,212,255,0.12)'))
    fig.add_trace(go.Scatterpolar(r=avg_healthy, theta=categories,
        fill='toself', name='Healthy avg',
        line=dict(color='#00FF88', width=1.5, dash='dot'),
        fillcolor='rgba(0,255,136,0.06)'))
    fig.add_trace(go.Scatterpolar(r=avg_risk, theta=categories,
        fill='toself', name='At-risk avg',
        line=dict(color='#FF3B6B', width=1.5, dash='dot'),
        fillcolor='rgba(255,59,107,0.06)'))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, color='#334155', gridcolor='#1E293B'),
            angularaxis=dict(color='#64748B', gridcolor='#1E293B'),
        ),
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
        height=360,
        title=dict(text="Patient vs. Population", font=dict(family='Orbitron', size=12, color='#94A3B8')),
    )
    return fig


# ─── 3-D risk scatter ────────────────────────────────────────────────────────
def scatter_3d(age, cholesterol, max_hr, bp, risk_score):
    np.random.seed(42)
    n = 200
    ages   = np.random.randint(28, 80, n)
    chols  = np.random.randint(120, 400, n)
    hrs    = np.random.randint(60, 210, n)
    risks  = (ages/100 + chols/800 + (220-hrs)/300 + np.random.rand(n)*0.3)
    risks  = np.clip(risks / risks.max(), 0, 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=ages, y=chols, z=hrs,
        mode='markers',
        marker=dict(
            size=4,
            color=risks,
            colorscale=[[0,'#00FF88'], [0.5,'#F59E0B'], [1,'#FF3B6B']],
            opacity=0.55,
            colorbar=dict(title='Risk', thickness=10, len=0.5,
                          tickfont=dict(color='#64748B', size=9)),
            showscale=True,
        ),
        name='Population',
        hovertemplate='Age:%{x}<br>Chol:%{y}<br>MaxHR:%{z}<extra></extra>',
    ))
    fig.add_trace(go.Scatter3d(
        x=[age], y=[cholesterol], z=[max_hr],
        mode='markers',
        marker=dict(size=14, color='#00D4FF', symbol='diamond',
                    line=dict(color='#fff', width=2)),
        name='You',
        hovertemplate=f'You<br>Risk: {risk_score:.1f}%<extra></extra>',
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=430,
        scene=dict(
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Age', gridcolor='#1E293B', color='#64748B'),
            yaxis=dict(title='Cholesterol', gridcolor='#1E293B', color='#64748B'),
            zaxis=dict(title='Max HR', gridcolor='#1E293B', color='#64748B'),
        ),
        title=dict(text="3-D Risk Distribution", font=dict(family='Orbitron', size=12, color='#94A3B8')),
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
    )
    return fig


# ─── Feature importance (simulated) ─────────────────────────────────────────
def feature_importance_chart(values_dict):
    features = ['Age', 'Cholesterol', 'Max HR', 'Resting BP', 'Oldpeak',
                'Fasting BS', 'Chest Pain', 'ST Slope', 'Ex. Angina',
                'Resting ECG', 'Sex']
    importance = [0.18, 0.14, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.03, 0.01]
    colors = ['#FF3B6B' if i > 0.1 else '#8B5CF6' if i > 0.07 else '#00D4FF'
              for i in importance]

    fig = go.Figure(go.Bar(
        x=importance, y=features,
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(0,0,0,0)')),
        text=[f'{v:.0%}' for v in importance],
        textposition='outside',
        textfont=dict(color='#94A3B8', size=10),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=340,
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(gridcolor='#1E293B', tickfont=dict(size=10)),
        title=dict(text="Feature Influence on Prediction",
                   font=dict(family='Orbitron', size=12, color='#94A3B8')),
    )
    return fig


# ─── BP vs HR scatter ────────────────────────────────────────────────────────
def vitals_scatter(bp, hr, age, risk_score):
    np.random.seed(7)
    n = 150
    bps  = np.random.randint(80, 200, n)
    hrs  = np.random.randint(60, 210, n)
    risk = (bps/300 + (220-hrs)/300 + np.random.rand(n)*0.3)
    risk = np.clip(risk / risk.max(), 0, 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bps, y=hrs, mode='markers',
        marker=dict(size=6, color=risk,
                    colorscale=[[0,'#00FF88'], [0.5,'#F59E0B'], [1,'#FF3B6B']],
                    opacity=0.5),
        name='Population',
        hovertemplate='BP:%{x} mmHg<br>HR:%{y} bpm<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=[bp], y=[hr], mode='markers',
        marker=dict(size=16, color='#00D4FF', symbol='star',
                    line=dict(color='#fff', width=2)),
        name='You',
        hovertemplate=f'You — Risk {risk_score:.1f}%<extra></extra>',
    ))
    # Safe zone reference
    fig.add_shape(type="rect", x0=80, x1=120, y0=60, y1=100,
                  fillcolor='rgba(0,255,136,0.06)',
                  line=dict(color='rgba(0,255,136,0.3)', dash='dot'))
    fig.add_annotation(x=100, y=80, text="Optimal zone",
                       font=dict(color='#00FF88', size=9), showarrow=False)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=340,
        xaxis=dict(title='Resting BP (mmHg)', gridcolor='#1E293B'),
        yaxis=dict(title='Max Heart Rate (bpm)', gridcolor='#1E293B'),
        title=dict(text="BP vs. Heart Rate Landscape",
                   font=dict(family='Orbitron', size=12, color='#94A3B8')),
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
    )
    return fig


# ─── Risk timeline (age projection) ─────────────────────────────────────────
def risk_timeline(age, risk_score):
    ages   = list(range(max(20, age-10), min(90, age+31)))
    base   = risk_score / 100
    risks  = [min(100, base * 100 * (1 + 0.015 * (a - age))) for a in ages]
    risks_lower = [max(0, r - 12) for r in risks]
    risks_upper = [min(100, r + 12) for r in risks]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ages+ages[::-1],
        y=risks_upper+risks_lower[::-1],
        fill='toself', fillcolor='rgba(139,92,246,0.08)',
        line=dict(color='rgba(0,0,0,0)'), name='Confidence band',
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=risks,
        mode='lines+markers',
        line=dict(color='#8B5CF6', width=2.5),
        marker=dict(size=5, color='#8B5CF6'),
        name='Projected risk',
    ))
    fig.add_vline(x=age, line=dict(color='#00D4FF', dash='dot', width=1.5))
    fig.add_annotation(x=age, y=risk_score, text="Now",
                       font=dict(color='#00D4FF', size=10),
                       showarrow=True, arrowcolor='#00D4FF', arrowsize=0.8)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        xaxis=dict(title='Age', gridcolor='#1E293B'),
        yaxis=dict(title='Risk %', gridcolor='#1E293B', range=[0, 100]),
        title=dict(text="Projected Risk Timeline",
                   font=dict(family='Orbitron', size=12, color='#94A3B8')),
        legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
    )
    return fig


# ─── BMI / weight helpers ────────────────────────────────────────────────────
def bmi_gauge(bmi):
    color = '#00FF88' if 18.5 <= bmi <= 24.9 else '#F59E0B' if bmi < 30 else '#FF3B6B'
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(bmi, 1),
        gauge=dict(
            axis=dict(range=[10, 45], tickwidth=1, tickcolor='#334155',
                      tickfont=dict(color='#64748B', size=9)),
            bar=dict(color=color, thickness=0.25),
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0,
            steps=[
                dict(range=[10, 18.5], color='rgba(0,212,255,0.08)'),
                dict(range=[18.5, 25], color='rgba(0,255,136,0.08)'),
                dict(range=[25, 30],   color='rgba(245,158,11,0.08)'),
                dict(range=[30, 45],   color='rgba(255,59,107,0.08)'),
            ],
        ),
        title=dict(text="BMI", font=dict(family='Orbitron', color='#64748B', size=11)),
        number=dict(font=dict(family='Orbitron', color=color, size=30)),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=200, margin=dict(l=20,r=20,t=40,b=0))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-badge">AI Powered</span>
    <span class="hero-badge">KNN Model</span>
    <span class="hero-badge">v2.0</span>
    <h1 class="hero-title">🫀 CardioAI</h1>
    <p class="hero-sub">Advanced Cardiovascular Risk Assessment · Clinical Decision Support System</p>
</div>
""", unsafe_allow_html=True)

ecg_placeholder = st.empty()
ecg_placeholder.plotly_chart(ecg_chart(), use_container_width=True, config=dict(displayModeBar=False))

if not model_loaded:
    st.warning("⚠️  Model files not found — running in **demo mode** (random prediction). "
               "Place `knn_heart_model.pkl`, `heart_scaler.pkl`, `heart_columns.pkl` in the working directory.", icon="⚠️")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">Patient Profile</div>', unsafe_allow_html=True)

    age         = st.slider("Age", 18, 100, 45)
    sex         = st.selectbox("Sex", ['Male', 'Female'])
    weight_kg   = st.number_input("Weight (kg)", 30, 200, 70)
    height_cm   = st.number_input("Height (cm)", 130, 220, 170)
    bmi         = weight_kg / ((height_cm / 100) ** 2)

    st.markdown('<div class="section-header" style="margin-top:1.2rem">Cardiac Vitals</div>', unsafe_allow_html=True)

    chest_pain      = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"],
                                   help="ATA=Atypical angina, NAP=Non-anginal, ASY=Asymptomatic, TA=Typical angina")
    resting_bp      = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 120)
    cholesterol     = st.number_input("Cholesterol (mg/dL)", 100, 600, 210)
    fasting_bs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["0", "1"])
    resting_ecg     = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
    max_heart_rate  = st.slider("Max Heart Rate (bpm)", 60, 220, 152)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak         = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
    st_slope        = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    st.markdown('<div class="section-header" style="margin-top:1.2rem">Lifestyle</div>', unsafe_allow_html=True)
    smoker      = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    activity    = st.select_slider("Physical Activity", ["Sedentary","Low","Moderate","High","Athlete"])
    diabetes    = st.checkbox("Diabetes")
    hypertension= st.checkbox("Hypertension")

    st.markdown("---")
    predict_btn = st.button("⚡  ANALYZE RISK", use_container_width=True)


# ── Main tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔬 Deep Analysis", "📈 Trends", "ℹ️ Reference"])

# ── Quick stats row ────────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    def metric_card(col, label, val, unit=""):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}<span style="font-size:0.9rem;color:#64748B"> {unit}</span></div>
        </div>""", unsafe_allow_html=True)

    metric_card(c1, "Age",           age,           "yr")
    metric_card(c2, "Resting BP",    resting_bp,    "mmHg")
    metric_card(c3, "Max HR",        max_heart_rate,"bpm")
    metric_card(c4, "Cholesterol",   cholesterol,   "mg/dL")
    metric_card(c5, "BMI",           f"{bmi:.1f}",  "")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Prediction output ────────────────────────────────────────────────────
    if predict_btn:
        # Build input
        raw_input = {
            'Age': age, 'Cholesterol': cholesterol, 'FastingBS': int(fasting_bs),
            'MaxHR': max_heart_rate, 'Oldpeak': oldpeak, 'RestingBP': resting_bp,
            'ChestPainType_' + chest_pain: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'RestingECG_' + resting_ecg: 1,
            'ST_Slope_' + st_slope: 1,
            'Sex_' + sex: 1,
        }
        input_df = pd.DataFrame([raw_input])

        if model_loaded:
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df    = input_df[expected_columns]
            scaled      = scaler.transform(input_df)
            prediction  = model.predict(scaled)[0]
            try:
                prob = model.predict_proba(scaled)[0][1]
            except Exception:
                prob = float(prediction)
        else:
            # Demo mode
            np.random.seed(age + int(cholesterol))
            prob       = np.clip(np.random.normal(0.45, 0.22), 0.05, 0.97)
            prediction = 1 if prob > 0.5 else 0

        risk_pct = round(prob * 100, 1)

        # Store in session for other tabs
        st.session_state['risk_pct']    = risk_pct
        st.session_state['prediction']  = prediction
        st.session_state['prob']        = prob
        st.session_state['age']         = age
        st.session_state['cholesterol'] = cholesterol
        st.session_state['max_hr']      = max_heart_rate
        st.session_state['bp']          = resting_bp

        col_g, col_r = st.columns([1, 1])
        with col_g:
            st.plotly_chart(risk_gauge(prob), use_container_width=True,
                            config=dict(displayModeBar=False))
        with col_r:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-high">
                    <div class="result-title" style="color:#FF3B6B">⚠ ELEVATED RISK</div>
                    <div style="font-family:'Orbitron',monospace;font-size:2.5rem;color:#FF3B6B;margin:0.5rem 0">{risk_pct}%</div>
                    <div class="result-sub">Cardiac event probability detected.<br>
                    Clinical follow-up strongly recommended.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <div class="result-title" style="color:#00FF88">✓ LOW RISK</div>
                    <div style="font-family:'Orbitron',monospace;font-size:2.5rem;color:#00FF88;margin:0.5rem 0">{risk_pct}%</div>
                    <div class="result-sub">No significant cardiac risk indicators.<br>
                    Continue preventive care & monitoring.</div>
                </div>""", unsafe_allow_html=True)

            # Key flags
            st.markdown("<br>", unsafe_allow_html=True)
            flags = []
            if cholesterol > 240: flags.append(("🔴", "High cholesterol", f"{cholesterol} mg/dL"))
            if resting_bp  > 140: flags.append(("🔴", "Hypertensive BP", f"{resting_bp} mmHg"))
            if bmi         > 30:  flags.append(("🟡", "Obesity", f"BMI {bmi:.1f}"))
            if age         > 60:  flags.append(("🟡", "Age factor", f"{age} yr"))
            if exercise_angina == "Yes": flags.append(("🔴", "Exercise angina", "Present"))
            if smoker == "Current": flags.append(("🔴", "Active smoker", "High risk"))
            if not flags:         flags.append(("🟢", "No major flags", "All within range"))

            st.markdown('<div class="section-header">Risk Flags</div>', unsafe_allow_html=True)
            for icon, label, val in flags:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                     padding:0.4rem 0;border-bottom:1px solid rgba(255,255,255,0.05);
                     font-family:Inter,sans-serif;font-size:0.82rem">
                    <span>{icon} {label}</span>
                    <span style="color:#94A3B8">{val}</span>
                </div>""", unsafe_allow_html=True)

        # Radar + 3D
        st.markdown("<br>", unsafe_allow_html=True)
        norm_vals = [
            age,
            resting_bp,
            cholesterol,
            max_heart_rate,
            oldpeak,
            int(fasting_bs),
        ]
        radar_cats = ['Age', 'Resting BP', 'Cholesterol', 'Max HR', 'Oldpeak', 'Fasting BS']

        col_rad, col_3d = st.columns([1, 1])
        with col_rad:
            st.plotly_chart(radar_chart(norm_vals, radar_cats),
                            use_container_width=True, config=dict(displayModeBar=False))
        with col_3d:
            st.plotly_chart(scatter_3d(age, cholesterol, max_heart_rate, resting_bp, risk_pct),
                            use_container_width=True, config=dict(displayModeBar=False))

    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;border:1px dashed rgba(0,212,255,0.2);
             border-radius:12px;font-family:Inter,sans-serif;color:#334155">
            <div style="font-size:3rem;margin-bottom:1rem">🫀</div>
            <div style="color:#64748B;font-size:0.9rem">Enter patient data in the sidebar<br>
            and click <strong style="color:#00D4FF">ANALYZE RISK</strong> to generate assessment</div>
        </div>""", unsafe_allow_html=True)


# ── Tab 2: Deep analysis ──────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Feature Analysis & Influence</div>', unsafe_allow_html=True)

    col_fi, col_bmi = st.columns([2, 1])
    with col_fi:
        vals_dict = dict(age=age, cholesterol=cholesterol, bp=resting_bp, hr=max_heart_rate)
        st.plotly_chart(feature_importance_chart(vals_dict), use_container_width=True,
                        config=dict(displayModeBar=False))
    with col_bmi:
        st.plotly_chart(bmi_gauge(bmi), use_container_width=True,
                        config=dict(displayModeBar=False))
        bmi_cat = ("Underweight" if bmi < 18.5 else "Normal" if bmi < 25
                   else "Overweight" if bmi < 30 else "Obese")
        bmi_col = "#00FF88" if bmi_cat == "Normal" else "#F59E0B" if bmi_cat == "Overweight" else "#FF3B6B"
        st.markdown(f"""
        <div style="text-align:center;font-family:Inter,sans-serif">
            <span style="font-size:0.8rem;color:{bmi_col};font-weight:600">{bmi_cat}</span>
            <div style="font-size:0.72rem;color:#475569;margin-top:0.3rem">
                {weight_kg} kg · {height_cm} cm
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(vitals_scatter(resting_bp, max_heart_rate, age,
                                   st.session_state.get('risk_pct', 50)),
                    use_container_width=True, config=dict(displayModeBar=False))


# ── Tab 3: Trends ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Risk Projection Over Time</div>', unsafe_allow_html=True)
    risk_now = st.session_state.get('risk_pct', 45.0)
    st.plotly_chart(risk_timeline(age, risk_now), use_container_width=True,
                    config=dict(displayModeBar=False))

    st.markdown('<div class="section-header">Modifiable Risk Factor Impact</div>', unsafe_allow_html=True)
    factors    = ['Quit Smoking', 'Reduce Cholesterol -20%', 'Lower BP -10 mmHg',
                  'Exercise Regularly', 'Lose 5 kg', 'Reduce Stress']
    reductions = [22, 14, 11, 18, 8, 6]
    fig_impact = go.Figure(go.Bar(
        x=reductions, y=factors, orientation='h',
        marker=dict(color=['#00FF88']*3 + ['#00D4FF']*3,
                    line=dict(color='rgba(0,0,0,0)')),
        text=[f'-{r}% risk' for r in reductions],
        textposition='outside',
        textfont=dict(color='#94A3B8', size=10),
    ))
    fig_impact.update_layout(
        **PLOTLY_LAYOUT, height=280,
        xaxis=dict(title='Estimated Risk Reduction (%)', gridcolor='#1E293B'),
        yaxis=dict(gridcolor='#1E293B'),
        title=dict(text="Lifestyle Changes → Risk Reduction",
                   font=dict(family='Orbitron', size=12, color='#94A3B8')),
    )
    st.plotly_chart(fig_impact, use_container_width=True, config=dict(displayModeBar=False))


# ── Tab 4: Reference ─────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Clinical Reference Ranges</div>', unsafe_allow_html=True)

    ref = {
        "Parameter": ["Blood Pressure", "Cholesterol (LDL)", "Resting Heart Rate",
                       "BMI", "Fasting Blood Sugar", "Oldpeak"],
        "Optimal":   ["< 120/80", "< 100 mg/dL", "60-100 bpm",
                       "18.5–24.9", "< 100 mg/dL", "0.0–1.0"],
        "Borderline":["120–139/80–89", "100–159 mg/dL", "50-60 / 100-110",
                       "25.0–29.9", "100–125 mg/dL", "1.0–2.0"],
        "High Risk":  ["> 140/90", "> 160 mg/dL", "< 50 or > 110",
                       "> 30", "> 126 mg/dL", "> 2.0"],
        "Your Value": [
            f"{resting_bp} mmHg", f"{cholesterol} mg/dL", f"{max_heart_rate} bpm",
            f"{bmi:.1f}", f"{'Yes' if fasting_bs=='1' else 'No'}", str(oldpeak),
        ],
    }
    ref_df = pd.DataFrame(ref)
    st.dataframe(ref_df, use_container_width=True, hide_index=True,
                 column_config={
                     "Parameter":   st.column_config.TextColumn("Parameter", width="medium"),
                     "Optimal":     st.column_config.TextColumn("✅ Optimal"),
                     "Borderline":  st.column_config.TextColumn("⚠️ Borderline"),
                     "High Risk":   st.column_config.TextColumn("🔴 High Risk"),
                     "Your Value":  st.column_config.TextColumn("📍 Your Value"),
                 })

    st.markdown("""
    <div style="font-family:Inter,sans-serif;font-size:0.78rem;color:#475569;
         margin-top:1.5rem;padding:1rem;border:1px solid rgba(255,255,255,0.06);border-radius:8px">
    <strong style="color:#64748B">⚕ Disclaimer:</strong>
    This tool is for informational and educational purposes only. It is not a substitute
    for professional medical advice, diagnosis, or treatment. Always consult a qualified
    healthcare provider regarding any medical condition.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 0.5rem;
     font-family:Inter,sans-serif;font-size:0.72rem;color:#334155;letter-spacing:0.06em">
    CARDIOAI · CLINICAL AI DECISION SUPPORT ·
    <span style="color:#1E293B">●</span>
    BUILT WITH STREAMLIT + PLOTLY
</div>""", unsafe_allow_html=True)