"""
Flood Risk Scoring Dashboard
=============================
Run with: streamlit run app.py in terminal

Requirements:
  pip install streamlit xgboost shap matplotlib pandas numpy

Place xgb_best.json in models/ folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import xgboost as xgb
import json, os, warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flood Risk Scoring",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colour palette (matches project) ─────────────────────────────────────────
TEAL  = "#1D9E75"
BLUE  = "#378ADD"
CORAL = "#D85A30"
AMBER = "#BA7517"

# ── Feature groups ────────────────────────────────────────────────────────────
NATURAL_FEATURES = [
    "MonsoonIntensity", "TopographyDrainage", "ClimateChange",
    "CoastalVulnerability", "Landslides", "Watersheds"
]
INFRA_FEATURES = [
    "Deforestation", "Urbanization", "RiverManagement", "DamsQuality",
    "DrainageSystems", "DeterioratingInfrastructure", "Siltation", "Encroachments"
]
SOCIO_FEATURES = [
    "PopulationScore", "AgriculturalPractices", "WetlandLoss",
    "PoliticalFactors", "InadequatePlanning", "IneffectiveDisasterPreparedness"
]
ALL_FEATURES = NATURAL_FEATURES + INFRA_FEATURES + SOCIO_FEATURES

# ── Feature descriptions (shown in tooltips) ──────────────────────────────────
FEAT_DESC = {
    "MonsoonIntensity":                 "Volume of rain during monsoon season",
    "TopographyDrainage":               "Drainage capacity based on terrain shape",
    "ClimateChange":                    "Degree of climate change impact in the region",
    "CoastalVulnerability":             "Exposure to storm surges and sea-level rise",
    "Landslides":                       "Risk of slope failure destabilising waterways",
    "Watersheds":                       "Number and health of upstream catchment areas",
    "Deforestation":                    "Extent of tree cover loss reducing absorption",
    "Urbanization":                     "Proportion of impermeable surface area",
    "RiverManagement":                  "Quality of dredging and bank maintenance",
    "DamsQuality":                      "Structural integrity of upstream dams",
    "DrainageSystems":                  "Capacity and maintenance of drainage networks",
    "DeterioratingInfrastructure":      "Age and degradation of flood defences",
    "Siltation":                        "Sediment buildup reducing river capacity",
    "Encroachments":                    "Illegal construction in flood-prone areas",
    "PopulationScore":                  "Population density in at-risk zones",
    "AgriculturalPractices":            "Land-use practices affecting water runoff",
    "WetlandLoss":                      "Loss of natural flood buffer ecosystems",
    "PoliticalFactors":                 "Governance quality affecting flood response",
    "InadequatePlanning":               "Absence of land-use and flood-risk planning",
    "IneffectiveDisasterPreparedness":  "Lack of early warning and response systems",
}

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    current_dir=os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir,"..","Notebook","models", "xgb_best.json"))
    if not os.path.exists(model_path):
        return None
    m = xgb.XGBRegressor()
    m.load_model(model_path)
    return m

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

# ── Feature engineering (mirrors notebook) ────────────────────────────────────
def engineer_features(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])
    df["NaturalRisk"]  = df[NATURAL_FEATURES].mean(axis=1)
    df["InfraRisk"]    = df[INFRA_FEATURES].mean(axis=1)
    df["SocioRisk"]    = df[SOCIO_FEATURES].mean(axis=1)
    df["Monsoon_x_Urbanization"]     = df["MonsoonIntensity"] * df["Urbanization"]
    df["Deforestation_x_Landslides"] = df["Deforestation"]    * df["Landslides"]
    df["TotalRiskScore"] = df[ALL_FEATURES].sum(axis=1)
    return df

# ── Risk tier helper ──────────────────────────────────────────────────────────
def risk_tier(score: float):
    if score < 0.40:  return "Low",      TEAL,  "🟢"
    if score < 0.55:  return "Moderate", AMBER, "🟡"
    if score < 0.70:  return "High",     CORAL, "🟠"
    return                   "Very High", "#8B0000", "🔴"

# ── SHAP waterfall plot ───────────────────────────────────────────────────────
def plot_waterfall(shap_vals, feature_df, n=12):
    feat_names = feature_df.columns.tolist()
    sv = shap_vals.values[0]
    base = shap_vals.base_values[0]

    pairs = sorted(zip(np.abs(sv), sv, feat_names, feature_df.iloc[0].values),
                   reverse=True)[:n]
    _, sv_top, fn_top, fv_top = zip(*pairs)
    sv_top  = list(reversed(sv_top))
    fn_top  = list(reversed(fn_top))
    fv_top  = list(reversed(fv_top))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = [TEAL if v >= 0 else CORAL for v in sv_top]
    bars = ax.barh(range(n), sv_top, color=colours, alpha=0.85, height=0.6)

    for i, (v, fname, fval) in enumerate(zip(sv_top, fn_top, fv_top)):
        label = f"{fname} = {fval:.0f}"
        ax.text(-0.001 if v < 0 else 0.001, i, label,
                va="center", ha="right" if v < 0 else "left",
                fontsize=8.5, color="#222")

    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_yticks([])
    ax.set_xlabel("SHAP value (impact on flood probability)", fontsize=9)
    ax.set_title(f"What drove this prediction (baseline = {base:.3f})", fontsize=10)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)

    pos_patch = mpatches.Patch(color=TEAL,  label="Increases risk ↑")
    neg_patch = mpatches.Patch(color=CORAL, label="Decreases risk ↓")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8, loc="lower right")
    plt.tight_layout()
    return fig

# ── Domain radar chart ────────────────────────────────────────────────────────
def plot_domain_radar(raw: dict):
    nat  = np.mean([raw[f] for f in NATURAL_FEATURES])
    infr = np.mean([raw[f] for f in INFRA_FEATURES])
    soc  = np.mean([raw[f] for f in SOCIO_FEATURES])
    labels = ["Natural /\nClimate", "Human /\nInfrastructure", "Socio-\npolitical"]
    vals   = [nat, infr, soc]
    max_v  = 10

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    vals_c = vals + vals[:1]

    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True))
    ax.fill(angles, vals_c, color=TEAL, alpha=0.25)
    ax.plot(angles, vals_c, color=TEAL, linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max_v)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2","4","6","8","10"], fontsize=7, color="#999")
    ax.set_title("Domain risk profile", fontsize=10, pad=14)
    ax.spines["polar"].set_visible(False)
    ax.grid(color="#ccc", linewidth=0.5)
    plt.tight_layout()
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# UI
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<h1 style='color:{TEAL}; margin-bottom:0'>🌊 Flood Risk Scoring</h1>
<p style='color:#666; margin-top:4px; font-size:15px'>
  XGBoost-based flood probability scoring for insurance underwriting · Adjust the sliders to score a location
</p>
""", unsafe_allow_html=True)
st.divider()

model = load_model()
if model is None:
    st.error("⚠️  Model file not found. Place `xgb_best.json` in a `models/` folder next to this script and restart.")
    st.stop()
explainer = load_explainer(model)

# ── Sidebar — feature inputs ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 🌿 Natural / Climate")
    nat_vals = {f: st.slider(f, 1, 15, 5, help=FEAT_DESC[f]) for f in NATURAL_FEATURES}

    st.divider()
    st.markdown(f"### 🏗️ Human / Infrastructure")
    inf_vals = {f: st.slider(f, 1, 15, 5, help=FEAT_DESC[f]) for f in INFRA_FEATURES}

    st.divider()
    st.markdown(f"### 👥 Socio-political")
    soc_vals = {f: st.slider(f, 1, 15, 5, help=FEAT_DESC[f]) for f in SOCIO_FEATURES}

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        score_btn = st.button("▶ Score location", type="primary", use_container_width=True)
    with col2:
        reset_btn = st.button("↺ Reset", use_container_width=True)

# ── Combine inputs ────────────────────────────────────────────────────────────
raw_input = {**nat_vals, **inf_vals, **soc_vals}
feature_df = engineer_features(raw_input)

# ── Predict ───────────────────────────────────────────────────────────────────
pred = float(model.predict(feature_df)[0])
pred = np.clip(pred, 0, 1)
tier, tier_color, tier_icon = risk_tier(pred)

# ── Top row — score cards ─────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Flood Probability", f"{pred:.3f}", help="XGBoost prediction (0 = no risk, 1 = certain flood)")
with c2:
    st.metric("Risk Tier", f"{tier_icon} {tier}")
with c3:
    total_rs = feature_df["TotalRiskScore"].iloc[0]
    st.metric("Total Risk Score", f"{total_rs:.0f} / 300", help="Sum of all 20 raw feature values")
with c4:
    pct = int(pred * 100)
    st.metric("Percentile approx.", f"~{pct}th", help="Approximate position in the 0–1 probability scale")

st.divider()

# ── Main content: waterfall + radar ───────────────────────────────────────────
col_left, col_right = st.columns([2.2, 1])

with col_left:
    st.markdown("#### Feature impact breakdown (SHAP)")
    st.caption("Red bars = feature increases flood risk · Teal bars = feature decreases flood risk")
    shap_vals = explainer(feature_df)
    fig_wf = plot_waterfall(shap_vals, feature_df)
    st.pyplot(fig_wf, use_container_width=True)

with col_right:
    st.markdown("#### Domain risk profile")
    fig_radar = plot_domain_radar(raw_input)
    st.pyplot(fig_radar, use_container_width=True)

    st.markdown("#### Domain averages")
    nat_avg  = np.mean(list(nat_vals.values()))
    inf_avg  = np.mean(list(inf_vals.values()))
    soc_avg  = np.mean(list(soc_vals.values()))
    st.progress(nat_avg  / 15, text=f"Natural risk: {nat_avg:.1f}/15")
    st.progress(inf_avg  / 15, text=f"Infra risk:   {inf_avg:.1f}/15")
    st.progress(soc_avg  / 15, text=f"Socio risk:   {soc_avg:.1f}/15")

st.divider()

# ── Raw feature table (expandable) ────────────────────────────────────────────
with st.expander("📋 View all engineered features"):
    display_df = feature_df.T.rename(columns={0: "Value"}).round(3)
    st.dataframe(display_df, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption("Model: XGBoost (Optuna-tuned) · Dataset: Kaggle Playground Series S4E5 · R² = 0.867 · RMSE = 0.01857")