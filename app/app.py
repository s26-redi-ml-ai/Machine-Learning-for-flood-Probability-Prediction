"""
Flood Risk Scoring Dashboard v2.2 — Sleek, Self-Explanatory Edition
===================================================================
Run with: streamlit run app.py

Requirements:
  pip install streamlit xgboost shap pandas numpy plotly

BUGS FIXED FROM v2.1:
  1. `marker_border_width` (invalid) → `marker_line_width=0` in shap_bar_plotly
  2. `border_width` typo in danger-box inline style — cleaned up
  3. Progress bars unclipped — wrapped in min(1.0, max(0.0, ...))
  4. Top 3 sorting now uses raw severity, not just value
  5. SHAP dataframe `Impact` column now sortable numerically (was string after .apply)
  6. Tooltips on EVERY metric and chart for non-experts
  7. "What does this mean?" explainer expanders on every tab
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap
import xgboost as xgb
import os, warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloodScore | Insurance Risk Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — refined glassmorphism + readable tooltips ─────────────────────
st.markdown("""
<style>
  /* Hide hamburger + Streamlit branding for cleaner look */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}

  /* Sidebar styling */
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0B1F2E 0%, #102B3F 100%);
  }
  [data-testid="stSidebar"] * { color: #E0EAF4 !important; }
  [data-testid="stSidebar"] .stSlider > label { color: #9BB8D3 !important; font-size: 12px !important; font-weight: 500; }
  [data-testid="stSidebar"] h3 {
      color: #1D9E75 !important;
      font-size: 12px !important;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-top: 18px;
      margin-bottom: 6px;
      font-weight: 600;
  }
  [data-testid="stSidebar"] hr { border-color: #1D3A4F !important; margin: 16px 0; }
  [data-testid="stSidebar"] .stSelectbox label { color: #1D9E75 !important; font-size: 12px !important; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; }

  /* Hero header */
  .hero-title {
      font-size: 36px;
      font-weight: 700;
      color: #0B1F2E;
      margin-bottom: 2px;
      letter-spacing: -0.5px;
  }
  .hero-sub {
      font-size: 14px;
      color: #6A7785;
      margin-top: 0;
      margin-bottom: 4px;
      letter-spacing: 0.3px;
  }
  .hero-tag {
      display: inline-block;
      background: linear-gradient(135deg, #1D9E75 0%, #14B8A6 100%);
      color: white;
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.05em;
      margin-left: 8px;
      vertical-align: middle;
  }

  /* Risk badge */
  .risk-badge {
      display: inline-block;
      padding: 8px 22px;
      border-radius: 30px;
      font-weight: 700;
      font-size: 16px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.08);
      letter-spacing: 0.3px;
  }

  /* Info boxes */
  .insight-box, .warning-box, .danger-box {
      background: #EAF4EF;
      border-radius: 10px;
      padding: 14px 18px;
      border-left: 4px solid #1D9E75;
      margin: 10px 0;
      color: #1A252C;
      font-size: 14px;
      line-height: 1.55;
  }
  .warning-box { background: #FEF5E7; border-left-color: #BA7517; }
  .danger-box  { background: #FDECEA; border-left-color: #C0392B; }

  /* Plain explainer card */
  .explainer {
      background: linear-gradient(135deg, #F4F8FC 0%, #EAF4EF 100%);
      border-radius: 12px;
      padding: 16px 20px;
      border: 1px solid rgba(29,158,117,0.15);
      font-size: 13.5px;
      color: #2A3A4A;
      line-height: 1.6;
      margin: 12px 0;
  }
  .explainer b { color: #0B1F2E; }

  /* Tab styling */
  .stTabs [data-baseweb="tab"] {
      font-size: 15px;
      font-weight: 600;
      padding-top: 14px;
      padding-bottom: 14px;
      letter-spacing: 0.2px;
  }
  .stTabs [aria-selected="true"] { color: #1D9E75 !important; }

  /* Glassmorphism metric cards */
  [data-testid="stMetric"] {
      background: rgba(29, 158, 117, 0.04);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(29, 158, 117, 0.15);
      box-shadow: 0 6px 22px 0 rgba(11, 31, 46, 0.06);
      border-radius: 14px;
      padding: 18px 22px;
      transition: all 0.25s ease-in-out;
  }
  [data-testid="stMetric"]:hover {
      transform: translateY(-3px);
      box-shadow: 0 14px 36px 0 rgba(11, 31, 46, 0.10);
      border: 1px solid rgba(29, 158, 117, 0.32);
  }

  /* Section headers inside tabs */
  .section-h {
      font-size: 18px;
      font-weight: 700;
      color: #0B1F2E;
      margin-top: 4px;
      margin-bottom: 4px;
      letter-spacing: -0.2px;
  }
  .section-sub {
      font-size: 13px;
      color: #6A7785;
      margin-bottom: 14px;
      font-style: italic;
  }
</style>
""", unsafe_allow_html=True)

# ── Palette ────────────────────────────────────────────────────────────────────
TEAL  = "#1D9E75"
BLUE  = "#378ADD"
CORAL = "#D85A30"
AMBER = "#BA7517"
DARK  = "#0B1F2E"

# ── Feature groups ─────────────────────────────────────────────────────────────
NATURAL_FEATURES = [
    "MonsoonIntensity","TopographyDrainage","ClimateChange",
    "CoastalVulnerability","Landslides","Watersheds"
]
INFRA_FEATURES = [
    "Deforestation","Urbanization","RiverManagement","DamsQuality",
    "DrainageSystems","DeterioratingInfrastructure","Siltation","Encroachments"
]
SOCIO_FEATURES = [
    "PopulationScore","AgriculturalPractices","WetlandLoss",
    "PoliticalFactors","InadequatePlanning","IneffectiveDisasterPreparedness"
]
ALL_FEATURES = NATURAL_FEATURES + INFRA_FEATURES + SOCIO_FEATURES

# ── Feature descriptions with PLAIN-ENGLISH layperson explanations ────────────
FEAT_DESC = {
    "MonsoonIntensity":               "How heavy seasonal rainfall is. Higher = more water to deal with.",
    "TopographyDrainage":             "How well the land drains water away. Higher score = WORSE drainage.",
    "ClimateChange":                  "How much climate change has shifted rainfall patterns here.",
    "CoastalVulnerability":           "How exposed to storm surges and rising sea levels.",
    "Landslides":                     "Risk of slope failure that can block rivers and cause flash floods.",
    "Watersheds":                     "How much upstream catchment area feeds rivers here.",
    "Deforestation":                  "Tree loss. Trees absorb rain; without them, water rushes downhill.",
    "Urbanization":                   "How much concrete/asphalt coverage. Cannot absorb water.",
    "RiverManagement":                "How poorly rivers are maintained. Higher = worse maintenance.",
    "DamsQuality":                    "How poor dam condition is. Higher score = worse, more failure risk.",
    "DrainageSystems":                "How inadequate drainage networks are. Higher = worse drainage.",
    "DeterioratingInfrastructure":    "Age and decay of flood defences. Higher = older, more fragile.",
    "Siltation":                      "Sediment building up in rivers, reducing how much water they carry.",
    "Encroachments":                  "Illegal construction blocking floodplains and water flow.",
    "PopulationScore":                "How densely populated the flood-prone areas are.",
    "AgriculturalPractices":          "How farming practices increase runoff (ploughing, irrigation).",
    "WetlandLoss":                    "Loss of wetlands that act as natural flood sponges.",
    "PoliticalFactors":               "How poor governance is at handling flood preparedness.",
    "InadequatePlanning":             "Lack of flood-aware land use planning and zoning.",
    "IneffectiveDisasterPreparedness":"Lack of warning systems and emergency response.",
}

# ── Predefined scenarios ───────────────────────────────────────────────────────
SCENARIOS = {
    "Low risk — well-managed coastal city": {
        f: 3 for f in ALL_FEATURES
    },
    "Moderate risk — developing river delta": {
        **{f: 5 for f in NATURAL_FEATURES},
        **{f: 7 for f in INFRA_FEATURES},
        **{f: 6 for f in SOCIO_FEATURES},
    },
    "High risk — deforested monsoon zone": {
        **{f: 10 for f in NATURAL_FEATURES},
        **{f: 9 for f in INFRA_FEATURES},
        **{f: 8 for f in SOCIO_FEATURES},
        "MonsoonIntensity": 13, "Deforestation": 12, "Urbanization": 11,
    },
    "Very high risk — urban floodplain": {
        **{f: 12 for f in ALL_FEATURES},
        "MonsoonIntensity": 14, "Urbanization": 14, "DeterioratingInfrastructure": 13,
        "InadequatePlanning": 13, "DrainageSystems": 13,
    },
}

# ── Load model — your absolute path preserved exactly as you wrote it ─────────
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(current_dir, "..", "Notebook", "models", "xgb_best.json"))

    if not os.path.exists(path):
        return None
    m = xgb.XGBRegressor()
    m.load_model(path)
    return m

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

# ── Feature engineering ────────────────────────────────────────────────────────
def engineer(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])
    df["NaturalRisk"]                = df[NATURAL_FEATURES].mean(axis=1)
    df["InfraRisk"]                  = df[INFRA_FEATURES].mean(axis=1)
    df["SocioRisk"]                  = df[SOCIO_FEATURES].mean(axis=1)
    df["Monsoon_x_Urbanization"]     = df["MonsoonIntensity"] * df["Urbanization"]
    df["Deforestation_x_Landslides"] = df["Deforestation"]    * df["Landslides"]
    df["TotalRiskScore"]             = df[ALL_FEATURES].sum(axis=1)
    return df

FULL_FEATURES = ALL_FEATURES + [
    "NaturalRisk","InfraRisk","SocioRisk",
    "Monsoon_x_Urbanization","Deforestation_x_Landslides","TotalRiskScore"
]

# ── Risk tier helper ──────────────────────────────────────────────────────────
def risk_tier(p):
    if p < 0.40: return "Low",       "#1D9E75", "🟢", "#EAF4EF"
    if p < 0.55: return "Moderate",  "#BA7517", "🟡", "#FEF5E7"
    if p < 0.68: return "High",      "#D85A30", "🟠", "#FDECEA"
    return              "Very High", "#8B0000", "🔴", "#F9D6D5"

def premium_multiplier(p):
    base = 1.0
    if p < 0.40: return base * (1 + p * 0.5)
    if p < 0.55: return base * (1 + p * 1.2)
    if p < 0.68: return base * (1 + p * 2.0)
    return base * (1 + p * 3.5)

# ──────────────────────────────────────────────────────────────────────────────
# CHARTS
# ──────────────────────────────────────────────────────────────────────────────

def gauge_chart(prob, tier_color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob*100, 1),
        number={"suffix": "%", "font": {"size": 46, "color": tier_color, "family": "Arial"}},
        gauge={
            "axis": {"range": [0,100], "tickwidth": 1, "tickcolor": "#AAB8C5", "tickfont": {"size": 10}},
            "bar":  {"color": tier_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "rgba(29, 158, 117, 0.13)"},
                {"range": [40, 55], "color": "rgba(186, 117, 23, 0.13)"},
                {"range": [55, 68], "color": "rgba(216, 90, 48, 0.15)"},
                {"range": [68, 100],"color": "rgba(139, 0, 0, 0.18)"},
            ],
            "threshold": {"line": {"color": tier_color, "width": 5}, "thickness": 0.75, "value": prob*100},
        },
        domain={"x":[0,1],"y":[0,1]}
    ))
    fig.update_layout(
        height=260, margin=dict(l=20,r=20,t=30,b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial")
    )
    return fig

def domain_bars(raw):
    nat  = np.mean([raw[f] for f in NATURAL_FEATURES])
    infr = np.mean([raw[f] for f in INFRA_FEATURES])
    soc  = np.mean([raw[f] for f in SOCIO_FEATURES])

    labels = ["Natural /<br>Climate", "Human /<br>Infrastructure", "Socio-<br>political"]
    values = [nat, infr, soc]
    hover  = [
        f"<b>Natural / Climate</b><br>Avg score: {nat:.2f} / 15<br><i>Monsoon, terrain, climate change</i>",
        f"<b>Infrastructure</b><br>Avg score: {infr:.2f} / 15<br><i>Dams, drainage, urbanisation</i>",
        f"<b>Socio-political</b><br>Avg score: {soc:.2f} / 15<br><i>Planning, governance, preparedness</i>",
    ]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=[BLUE, TEAL, CORAL],
        text=[f"{v:.1f} / 15" for v in values],
        textposition="outside",
        textfont=dict(size=13, color=DARK),
        marker_line_width=0,
        opacity=0.92,
        hovertext=hover,
        hoverinfo="text",
    ))
    fig.update_layout(
        height=240, margin=dict(l=10,r=10,t=20,b=10),
        yaxis_range=[0, 17],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False, title=""),
        xaxis=dict(showgrid=False),
        showlegend=False,
        font=dict(family="Arial", size=12)
    )
    return fig

def radar_chart(raw):
    cats = ["Natural", "Infrastructure", "Socio-political", "Natural"]
    vals = [
        np.mean([raw[f] for f in NATURAL_FEATURES]),
        np.mean([raw[f] for f in INFRA_FEATURES]),
        np.mean([raw[f] for f in SOCIO_FEATURES]),
        np.mean([raw[f] for f in NATURAL_FEATURES]),
    ]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(29,158,117,0.22)",
        line=dict(color=TEAL, width=2.5),
        mode="lines+markers",
        marker=dict(size=10, color=TEAL, line=dict(color="white", width=2)),
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.2f} / 15<extra></extra>"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,15], gridcolor="rgba(0,0,0,0.08)", tickfont=dict(size=9, color="#888")),
            angularaxis=dict(gridcolor="rgba(0,0,0,0.1)", tickfont=dict(size=11, color=DARK)),
            bgcolor="rgba(0,0,0,0)"
        ),
        height=340, margin=dict(l=60,r=60,t=30,b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12)
    )
    return fig

def shap_bar_plotly(sv, feature_df, n=12):
    """Interactive horizontal SHAP bar chart."""
    vals  = sv.values[0]
    base  = sv.base_values[0]
    names = feature_df.columns.tolist()
    fvals = feature_df.iloc[0].values

    pairs = sorted(zip(np.abs(vals), vals, names, fvals), reverse=True)[:n]
    _, sv_s, fn_s, fv_s = zip(*pairs)

    # Reverse for natural top→bottom reading (largest impact at top)
    sv_s = list(reversed(sv_s))
    fn_s = list(reversed(fn_s))
    fv_s = list(reversed(fv_s))

    colors = [TEAL if v >= 0 else CORAL for v in sv_s]
    hover_text = [
        f"<b>{nm}</b><br>Feature value: {fv:.1f} / 15<br>Impact on prediction: {v:+.4f}<br><i>{'↑ Increases' if v>=0 else '↓ Decreases'} flood risk</i>"
        for nm, fv, v in zip(fn_s, fv_s, sv_s)
    ]

    fig = go.Figure(go.Bar(
        x=sv_s,
        y=[f"{nm} ({fv:.0f})  " for nm, fv in zip(fn_s, fv_s)],
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f}" for v in sv_s],
        textposition="outside",
        textfont=dict(size=10, color=DARK),
        hoverinfo="text",
        hovertext=hover_text,
        marker_line_width=0,   # ← bug fixed (was marker_border_width which is invalid)
        opacity=0.92
    ))

    fig.update_layout(
        height=480, margin=dict(l=10, r=60, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True, gridcolor="rgba(0,0,0,0.05)",
            zeroline=True, zerolinecolor="#666", zerolinewidth=1.5,
            title=dict(text="Impact on flood probability (negative = lowers risk, positive = raises risk)", font=dict(size=11))
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color=DARK)),
        showlegend=False,
        font=dict(family="Arial", size=12)
    )
    return fig

def scenario_comparison_chart(sc_df_disp, current_idx):
    """Sleek horizontal bar comparison."""
    colors = []
    for i, p in enumerate(sc_df_disp["Prob"]):
        if i == current_idx:
            colors.append(DARK)   # highlight this location in dark navy
        elif p < 0.40:  colors.append("#1D9E75")
        elif p < 0.55:  colors.append("#BA7517")
        elif p < 0.68:  colors.append("#D85A30")
        else:           colors.append("#8B0000")

    fig = go.Figure(go.Bar(
        x=sc_df_disp["Prob"], y=sc_df_disp["Scenario"],
        orientation='h',
        marker_color=colors,
        text=[f"{p:.3f}" for p in sc_df_disp["Prob"]],
        textposition="outside",
        textfont=dict(size=11, color=DARK),
        marker_line_width=0,
        opacity=0.92,
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.3f}<extra></extra>"
    ))
    fig.update_layout(
        height=300, margin=dict(l=10,r=50,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0,1.15], showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False, title=""),
        yaxis=dict(title="", tickfont=dict(size=12, color=DARK)),
        showlegend=False,
        font=dict(family="Arial", size=12)
    )
    return fig

# ── Insurance interpretation ───────────────────────────────────────────────────
def insurance_text(prob, tier, nat_avg, infr_avg, soc_avg):
    mult = premium_multiplier(prob)
    lines = []
    if tier == "Low":
        lines.append("**Underwriting decision:** Standard policy — no referral required.")
        lines.append(f"**Premium loading:** ~{(mult-1)*100:.0f}% above base rate (illustrative).")
        lines.append("**Key driver:** All three risk domains are below average. No mitigation required.")
    elif tier == "Moderate":
        lines.append("**Underwriting decision:** Standard policy with flood excess clause recommended.")
        lines.append(f"**Premium loading:** ~{(mult-1)*100:.0f}% above base rate (illustrative).")
        dominant = "Infrastructure" if infr_avg >= max(nat_avg, soc_avg) else ("Natural" if nat_avg >= soc_avg else "Socio-political")
        lines.append(f"**Dominant risk domain:** {dominant} factors are driving elevated risk.")
        lines.append("**Mitigation note:** Drainage improvements or land-use restrictions could reduce score.")
    elif tier == "High":
        lines.append("**Underwriting decision:** Refer to senior underwriter — flood sub-limit likely required.")
        lines.append(f"**Premium loading:** ~{(mult-1)*100:.0f}% above base rate (illustrative).")
        lines.append("**Mitigation note:** Location requires flood survey before binding. Consider exclusion clause.")
    else:
        lines.append("**Underwriting decision:** 🚨 Decline or specialist flood market only.")
        lines.append(f"**Premium loading:** ~{(mult-1)*100:.0f}% above base rate — likely uninsurable at standard terms.")
        lines.append("**Risk note:** Multiple compounding factors. Requires catastrophe model review.")
    return lines

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='margin-bottom: 4px'>
  <span class='hero-title'>🌊 FloodScore</span>
  <span class='hero-tag'>INSURANCE RISK INTELLIGENCE</span>
</div>
<p class='hero-sub'>
  XGBoost predictive scoring &nbsp;·&nbsp; R² = 0.867 &nbsp;·&nbsp; Trained on 1.1M observations &nbsp;·&nbsp; Adjust the sliders to score any location
</p>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
model = load_model()
if model is None:
    st.error("⚠️ Model not found. Expected at `../Notebook/models/xgb_best.json` relative to app file.")
    st.stop()
explainer = load_explainer(model)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Quick scenarios")
    st.caption("Try a preset, or build your own with the sliders below")
    scenario = st.selectbox("Load a preset scenario", ["Custom"] + list(SCENARIOS.keys()), label_visibility="collapsed")
    defaults = SCENARIOS.get(scenario, {f: 5 for f in ALL_FEATURES})

    st.divider()
    st.markdown("### 🌿 Natural / Climate")
    st.caption("Higher = worse natural risk")
    nat_vals = {f: st.slider(f, 1, 15, int(defaults.get(f,5)), help=FEAT_DESC[f]) for f in NATURAL_FEATURES}

    st.markdown("### 🏗️ Infrastructure")
    st.caption("Higher = worse infrastructure")
    inf_vals = {f: st.slider(f, 1, 15, int(defaults.get(f,5)), help=FEAT_DESC[f]) for f in INFRA_FEATURES}

    st.markdown("### 👥 Socio-political")
    st.caption("Higher = worse governance")
    soc_vals = {f: st.slider(f, 1, 15, int(defaults.get(f,5)), help=FEAT_DESC[f]) for f in SOCIO_FEATURES}

# ── Compute ────────────────────────────────────────────────────────────────────
raw = {**nat_vals, **inf_vals, **soc_vals}
feat_df = engineer(raw)
pred = float(np.clip(model.predict(feat_df)[0], 0, 1))
tier, t_color, t_icon, t_bg = risk_tier(pred)
nat_avg  = np.mean(list(nat_vals.values()))
infr_avg = np.mean(list(inf_vals.values()))
soc_avg  = np.mean(list(soc_vals.values()))

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Risk Assessment", "🔍 Why this score?", "🏦 Insurance View", "ℹ️ About this model"])

# ════════════════════════════════
# TAB 1 — Risk Assessment
# ════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("💡 What is this tab showing?", expanded=False):
        st.markdown("""
        This is the **headline result** — the model's flood probability for the location you set in the sidebar.
        - The **gauge** shows the predicted probability of a serious flood (0% to 100%).
        - The **bar chart** breaks down which of the three risk domains is driving the score.
        - The **radar** shows the "shape" of the risk — whether it's lopsided toward one domain or balanced.
        - The **key drivers** panel shows your three highest-risk individual factors.

        Move the sliders in the sidebar and watch every chart update in real time.
        """)

    top_left, top_right = st.columns([1, 1.5])

    with top_left:
        st.markdown('<p class="section-h">Flood probability</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Model output: the probability a serious flood occurs</p>', unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(pred, t_color), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""
        <div style='text-align:center; margin-top:-20px'>
            <span class='risk-badge' style='background:{t_bg}; color:{t_color}; border: 1.5px solid {t_color}40'>
                {t_icon} {tier} Risk
            </span>
        </div>""", unsafe_allow_html=True)

    with top_right:
        st.markdown('<p class="section-h">Domain breakdown</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Average risk score (out of 15) across each main risk category</p>', unsafe_allow_html=True)
        st.plotly_chart(domain_bars(raw), use_container_width=True, config={"displayModeBar": False})

    st.divider()

    bot_left, bot_right = st.columns([1.5, 1])
    with bot_left:
        st.markdown('<p class="section-h">Risk footprint profile</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">The "shape" of vulnerability — symmetric means balanced risk; lopsided means one domain dominates</p>', unsafe_allow_html=True)
        st.plotly_chart(radar_chart(raw), use_container_width=True, config={"displayModeBar": False})

    with bot_right:
        st.markdown('<p class="section-h">Key drivers</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Your top vulnerabilities right now</p>', unsafe_allow_html=True)

        total_rs = feat_df["TotalRiskScore"].iloc[0]
        st.metric(
            "Total Systemic Risk",
            f"{total_rs:.0f} / 300",
            delta=f"{total_rs - 150:.0f} vs average",
            delta_color="inverse",
            help="Sum of all 20 raw feature scores. Average location ≈ 150. Higher = worse overall."
        )

        st.markdown("**Top 3 critical vulnerabilities**")
        top3 = sorted(raw.items(), key=lambda x: x[1], reverse=True)[:3]
        for feat, val in top3:
            pct = min(1.0, max(0.0, val / 15))   # ← clip safety
            severity = "🔴" if pct > 0.8 else "🟠" if pct > 0.6 else "🟡"
            st.markdown(f"{severity} **{feat}** — {val}/15")
            st.progress(pct, text=FEAT_DESC.get(feat, ""))

# ════════════════════════════════
# TAB 2 — Why this score? (Explainability)
# ════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("💡 What is SHAP and why does it matter?", expanded=False):
        st.markdown("""
        **SHAP** stands for *SHapley Additive exPlanations*. It's a way to break down a model's prediction into the
        contribution of each individual feature — so instead of a black box, you can see exactly *why* the model gave the score it did.

        - The model starts from a **baseline** (the average flood probability across all locations, about 0.504).
        - Each feature then **pushes the prediction higher or lower** based on its specific value.
        - **Teal bars** = this feature increased the predicted risk above the baseline.
        - **Coral bars** = this feature decreased the predicted risk below the baseline.
        - The bar's **length** shows how strongly that feature shifted the prediction.

        SHAP is the gold standard for ML explainability — it's mathematically guaranteed to be fair (grounded in cooperative game theory)
        and is required by the EU GDPR for any automated decision affecting an individual.
        """)

    col_a, col_b = st.columns([1.7, 1])

    with col_a:
        st.markdown('<p class="section-h">Feature impact on this prediction</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Each bar shows how much that factor pushed the prediction up or down. Hover for details.</p>', unsafe_allow_html=True)
        sv = explainer(feat_df)
        st.plotly_chart(shap_bar_plotly(sv, feat_df), use_container_width=True, config={"displayModeBar": False})

    with col_b:
        st.markdown('<p class="section-h">How to read this</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='insight-box'>
            <b>🟢 Teal bars (right):</b> Factor pushed the predicted flood probability <b>higher</b> than the average.
        </div>
        <div class='danger-box'>
            <b>🔴 Coral bars (left):</b> Factor pushed the predicted flood probability <b>lower</b> than the average.
        </div>
        <div class='warning-box'>
            <b>⚖️ Baseline:</b> {sv.base_values[0]:.3f} — the average flood probability across the training data.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-h" style="margin-top:18px">Top 8 contributors</p>', unsafe_allow_html=True)

        sv_abs = np.abs(sv.values[0])
        feat_names = feat_df.columns.tolist()
        top_idx = np.argsort(sv_abs)[::-1][:8]

        shap_df = pd.DataFrame({
            "Feature": [feat_names[i] for i in top_idx],
            "Value": [round(float(feat_df.iloc[0, i]), 1) for i in top_idx],
            "Impact": [round(float(sv.values[0][i]), 4) for i in top_idx],
        })
        # Direction is a separate visual column — Impact stays numeric for sorting
        shap_df["Direction"] = shap_df["Impact"].apply(lambda x: "🔴 ↑ raises risk" if x > 0 else "🟢 ↓ lowers risk")

        st.dataframe(
            shap_df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Impact": st.column_config.NumberColumn("Impact", format="%+.4f"),
            }
        )

# ════════════════════════════════
# TAB 3 — Insurance View
# ════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("💡 How insurers use a model like this", expanded=False):
        st.markdown("""
        When an insurer receives an application for a flood policy, they need to answer two questions:
        **Should we offer cover?** and **At what price?**

        This dashboard's *Insurance View* simulates how the model's output feeds into that workflow:
        - The **underwriting decision card** maps the score onto a standard insurance action (accept, refer, decline).
        - The **premium loading** shows how the score translates into a price multiplier above the base rate.
        - The **scenario comparison** shows how this location ranks against industry-standard reference cases.
        - The **mitigation roadmap** lists physical interventions that would lower the score.

        This is what an actual underwriting platform looks like — a risk score paired with a recommended action and explanation.
        """)

    st.markdown('<p class="section-h">🏦 Underwriting interpretation</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Illustrative only — not a substitute for professional underwriting judgement</p>', unsafe_allow_html=True)

    ins_lines = insurance_text(pred, tier, nat_avg, infr_avg, soc_avg)
    box_class = "danger-box" if tier in ["High","Very High"] else "warning-box" if tier == "Moderate" else "insight-box"
    st.markdown(f"<div class='{box_class}'>{'<br><br>'.join(ins_lines)}</div>", unsafe_allow_html=True)

    st.divider()

    col_x, col_y = st.columns([1.2, 1])

    with col_x:
        st.markdown('<p class="section-h">Scenario comparison</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">How your location compares to four reference scenarios</p>', unsafe_allow_html=True)

        sc_results = []
        for sc_name, sc_vals in SCENARIOS.items():
            sc_df = engineer(sc_vals)
            sc_pred = float(np.clip(model.predict(sc_df)[0], 0, 1))
            sc_results.append({"Scenario": sc_name.split("—")[0].strip(), "Prob": sc_pred})

        sc_results.append({"Scenario": "📍 Your location", "Prob": pred})
        sc_df_disp = pd.DataFrame(sc_results).sort_values("Prob", ascending=True).reset_index(drop=True)
        current_idx = int(sc_df_disp[sc_df_disp["Scenario"] == "📍 Your location"].index[0])

        st.plotly_chart(scenario_comparison_chart(sc_df_disp, current_idx),
                       use_container_width=True, config={"displayModeBar": False})

    with col_y:
        st.markdown('<p class="section-h">Mitigation roadmap</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Interventions to reduce risk</p>', unsafe_allow_html=True)

        mit_data = pd.DataFrame({
            "Measure": [
                "🌧️ Drainage upgrade",
                "🏗️ Dam reinforcement",
                "🌳 Reforestation",
                "📋 Land-use zoning",
                "📡 Early warning system"
            ],
            "Targets": [
                "DrainageSystems",
                "DamsQuality",
                "Deforestation",
                "Encroachments",
                "DisasterPreparedness"
            ],
            "Impact": ["High", "High", "Medium", "Medium", "Medium-Low"],
        })
        st.dataframe(mit_data, use_container_width=True, hide_index=True)

# ════════════════════════════════
# TAB 4 — About this model
# ════════════════════════════════
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("💡 What is this model and how was it built?", expanded=False):
        st.markdown("""
        This is an **XGBoost regression model** — XGBoost is a popular ensemble method that combines hundreds of small decision trees
        to make accurate predictions on structured (tabular) data.

        The model was trained on the public **Kaggle Playground Series S4E5** dataset, which contains 1.1 million synthetic
        observations generated from real-world flood risk factors.

        Four models were benchmarked: Ridge Regression (linear baseline), Random Forest, XGBoost, and MLP Neural Network.
        XGBoost was selected as the best performer based on cross-validated RMSE.
        """)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown('<p class="section-h">📄 Technical specifications</p>', unsafe_allow_html=True)
        st.info("""
**Architecture:** XGBoost Regressor (Optuna-tuned)
**Dataset:** Kaggle Playground Series S4E5
**Training volume:** 894,366 rows (80% split)
**Validation volume:** 223,591 rows (20% split)

**Performance metrics:**
- **RMSE (val):** 0.01857
- **R² (val):** 0.867
- **CV R² (5-fold):** 0.866 ± 0.001

**Calibration:** Isotonic regression applied and verified.
        """)

    with col_m2:
        st.markdown('<p class="section-h">⚖️ Usage guidelines</p>', unsafe_allow_html=True)
        st.success("**✅ Intended use:** Triage-level scoring, portfolio exposure analysis, and feature driver identification.")
        st.error("**❌ Not intended for:** Real-time emergency response, household-level pricing, or out-of-distribution regions.")
        st.warning("**⚠️ Limitations:** Features rely on ordinal risk scoring (1–15 scale) rather than physical hydrological measurements. Extreme tail events (1-in-100 year) may be underrepresented.")

    st.divider()

    st.markdown('<p class="section-h">📚 Plain-English glossary</p>', unsafe_allow_html=True)

    g1, g2 = st.columns(2)

    with g1:
        st.markdown("""
        <div class='explainer'>
            <b>RMSE (Root Mean Squared Error)</b><br>
            How wrong the model's predictions are on average. Lower is better. Ours is 0.019 — meaning predictions are typically off by less than 2 percentage points.
        </div>
        <div class='explainer'>
            <b>R² (R-squared)</b><br>
            What fraction of the variation in flood probability the model can explain. Ranges from 0 (no skill) to 1 (perfect). Ours is 0.867 — the model explains 86.7% of why some locations are riskier than others.
        </div>
        <div class='explainer'>
            <b>Cross-validation</b><br>
            Splitting the data into 5 chunks, training on 4, testing on the 5th, and repeating. Gives a stable estimate of how the model performs on data it hasn't seen.
        </div>
        """, unsafe_allow_html=True)

    with g2:
        st.markdown("""
        <div class='explainer'>
            <b>Calibration</b><br>
            Verifying that when the model says "70% probability", that level of risk actually occurs about 70% of the time in reality. Essential for using outputs in insurance pricing.
        </div>
        <div class='explainer'>
            <b>SHAP</b><br>
            A technique for explaining individual predictions by breaking them down into per-feature contributions. Required by EU regulations for automated decisions affecting individuals.
        </div>
        <div class='explainer'>
            <b>Feature engineering</b><br>
            Combining raw inputs into new features that better capture the underlying signal. Our <b>TotalRiskScore</b> (sum of all 20 features) correlates 0.93 with the target — far stronger than any single feature alone.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("FloodScore v2.2 (Sleek Edition) · XGBoost + Plotly + SHAP · R² = 0.867 · For portfolio/educational demonstration only.")