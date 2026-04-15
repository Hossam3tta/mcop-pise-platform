from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mcop.ai_optimizer import analyze_metrics
from mcop.data_collector import get_current_metrics
from mcop.decision_engine import decide_action
from pise.predictor import predict


st.set_page_config(
    page_title="SimCloud Dynamics | MCOP + PISE",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #111827 0%, #0b1020 45%, #070b14 100%);
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .main * {
        color: #e5e7eb;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    pre, code {
        background: transparent !important;
        border: none !important;
    }

    .hero-card {
        background:
            radial-gradient(circle at top right, rgba(59,130,246,0.18), transparent 30%),
            linear-gradient(135deg, #0c1425 0%, #0b1220 45%, #09101c 100%);
        border: 1px solid rgba(96, 165, 250, 0.14);
        border-radius: 26px;
        padding: 22px 28px;
        box-shadow: 0 20px 45px rgba(0,0,0,0.35);
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 20px;
        flex-wrap: wrap;
    }

    .hero-left {
        display: flex;
        flex-direction: column;
    }

    .hero-right {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        justify-content: flex-start;
        padding-top: 40px;
    }

    .hero-title-small {
        font-size: 17px;
        color: #94a3b8;
        font-weight: 700;
    }

    .hero-title-main {
        font-size: 48px;
        font-weight: 900;
        line-height: 1.04;
        margin-top: 8px;
    }

    .hero-title-sub {
        font-size: 20px;
        color: #94a3b8;
        margin-top: 6px;
    }

    .last-updated-label {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }

    .last-updated-value {
        font-size: 20px;
        font-weight: 600;
        color: #cbd5e1;
        line-height: 1.2;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 16px;
        border-radius: 20px;
        font-size: 15px;
        font-weight: 700;
        min-width: 140px;
        justify-content: center;
        border: 1px solid transparent;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
    }

    .pill-green {
        background: rgba(34,197,94,0.12);
        color: #72db72;
        border-color: rgba(34,197,94,0.24);
    }

    .pill-blue {
        background: rgba(96,165,250,0.12);
        color: #73a9ff;
        border-color: rgba(96,165,250,0.24);
    }

    .pill-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        display: inline-block;
        flex-shrink: 0;
    }

    .pill-dot-green {
        background: #69d86b;
        box-shadow: 0 0 10px rgba(105,216,107,0.45);
    }

    .pill-dot-blue {
        background: #73a9ff;
        box-shadow: 0 0 10px rgba(115,169,255,0.45);
    }

    .section-title {
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: #94a3b8;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .section-title::before {
        content: "";
        display: inline-block;
        width: 5px;
        height: 20px;
        border-radius: 999px;
        background: #60a5fa;
        margin-right: 10px;
        vertical-align: middle;
    }

    .section-shell {
        background: linear-gradient(180deg, rgba(11,18,32,0.98) 0%, rgba(8,14,26,0.98) 100%);
        border: 1px solid rgba(96, 165, 250, 0.10);
        border-radius: 24px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 14px 34px rgba(0,0,0,0.24);
        margin-top: 18px;
    }

    .metric-card {
        background: linear-gradient(180deg, #0c1425 0%, #0b1220 100%);
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 22px;
        padding: 20px 20px 18px 20px;
        min-height: 150px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.20);
    }

    .metric-title {
        color: #94a3b8;
        font-size: 14px;
        letter-spacing: 0.02em;
        margin-bottom: 14px;
    }

    .metric-value {
        font-size: 40px;
        font-weight: 800;
        line-height: 1.0;
        margin-bottom: 10px;
    }

    .metric-sub {
        color: #94a3b8;
        font-size: 14px;
    }

    label[data-testid="stWidgetLabel"] p {
        color: #60a5fa !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="select"] > div {
        background: #0b1220 !important;
        border: 1px solid rgba(148,163,184,0.14) !important;
        border-radius: 16px !important;
        min-height: 52px !important;
    }

    div[data-baseweb="select"] * {
        color: #60a5fa !important;
        fill: #60a5fa !important;
    }

    .stButton > button {
        border-radius: 16px !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.16);
        border: none !important;
        width: 100%;
    }

    button[kind="secondary"] {
        background: #2563eb !important;
        color: white !important;
        min-height: 36px !important;
    }

    button[kind="primary"] {
        background: #22c55e !important;
        color: #02160a !important;
        font-weight: bold !important;
        min-height: 52px !important;
        border-radius: 12px !important;
        border: none !important;
    }

    .recommend-card {
        background: linear-gradient(180deg, #09101c 0%, #080d18 100%);
        border: 1px solid rgba(96, 165, 250, 0.12);
        border-radius: 20px;
        padding: 18px;
        margin-top: 12px;
    }

    .recommend-title {
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .risk-badge {
        float: right;
        padding: 8px 12px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 13px;
    }

    .risk-high {
        background: rgba(239,68,68,0.14);
        color: #f87171;
        border: 1px solid rgba(239,68,68,0.24);
    }

    .risk-medium {
        background: rgba(245,158,11,0.14);
        color: #fbbf24;
        border: 1px solid rgba(245,158,11,0.24);
    }

    .risk-low {
        background: rgba(34,197,94,0.14);
        color: #4ade80;
        border: 1px solid rgba(34,197,94,0.24);
    }

    .small-note {
        color: #94a3b8;
        font-size: 13px;
    }

    .info-box {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(96, 165, 250, 0.16);
        border-radius: 18px;
        padding: 18px;
        margin-top: 14px;
    }

    .footer-note {
        color: #94a3b8;
        text-align: center;
        margin-top: 8px;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

providers = ["AWS", "Azure", "GCP", "OCI"]
regions = ["us-east-1", "eastus", "us-central1", "us-ashburn-1", "us-west-2", "westeurope"]

REPORTS_DIR = Path("data/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_nodes() -> pd.DataFrame:
    rows = []
    for i in range(6):
        rows.append(
            {
                "Node ID": f"{providers[i % len(providers)].lower()}-node-0{i+1}",
                "Provider": providers[i % len(providers)],
                "Region": regions[i],
                "CPU Usage": round(random.uniform(15, 90), 1),
                "Memory": round(random.uniform(20, 90), 1),
                "Latency": round(random.uniform(25, 120), 1),
                "Cost/hr": round(random.uniform(1.20, 3.60), 2),
            }
        )
    return pd.DataFrame(rows)


def build_forecast(hours: int, workload: str) -> pd.DataFrame:
    base = {
        "Steady State": 50,
        "Burst Load": 72,
        "Spiky Traffic": 64,
    }.get(workload, 50)

    load = []
    stress = []
    points = list(range(hours))

    for h in points:
        wave = math.sin((h / max(hours, 1)) * 2 * math.pi)
        surge = 8 if workload == "Burst Load" and max(1, hours // 3) <= h <= max(2, (hours // 3) + 2) else 0
        spikes = 6 if workload == "Spiky Traffic" and h % max(2, hours // 6) == 0 else 0

        load_val = base + wave * 5 + surge + spikes
        stress_val = (load_val * 0.55) + (8 if workload == "Burst Load" else 0)

        load.append(round(load_val, 1))
        stress.append(round(stress_val, 1))

    return pd.DataFrame(
        {
            "Hour": [f"{h}h" for h in points],
            "Forecast Load": load,
            "Stress Index": stress,
        }
    )


def colored_value(
    value: float | int,
    *,
    good_low: bool = True,
    medium_cutoff: float = 70,
    high_cutoff: float = 85,
    prefix: str = "",
    suffix: str = "",
) -> str:
    if good_low:
        if value >= high_cutoff:
            color = "#f87171"
        elif value >= medium_cutoff:
            color = "#fbbf24"
        else:
            color = "#4ade80"
    else:
        if value <= high_cutoff:
            color = "#4ade80"
        elif value <= medium_cutoff:
            color = "#fbbf24"
        else:
            color = "#f87171"

    return f"<div class='metric-value' style='color:{color};'>{prefix}{value}{suffix}</div>"


def make_dark_line_chart(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    color: str,
    fill_color: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Hour"],
            y=df[y_col],
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=7, color=color),
            fill="tozeroy",
            fillcolor=fill_color,
            hovertemplate="%{x}<br>%{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e5e7eb")),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        margin=dict(l=20, r=20, t=55, b=20),
        height=360,
        font=dict(color="#cbd5e1"),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.10)",
            tickfont=dict(color="#94a3b8"),
            title="",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.10)",
            tickfont=dict(color="#94a3b8"),
            title="",
            range=[0, 100],
        ),
    )
    return fig


def make_nodes_table(df: pd.DataFrame) -> go.Figure:
    provider_colors = {
        "AWS": "#fbbf24",
        "Azure": "#60a5fa",
        "GCP": "#60a5fa",
        "OCI": "#f87171",
    }

    cpu_colors = []
    mem_colors = []
    latency_colors = []
    cost_colors = []
    provider_font_colors = []

    for _, row in df.iterrows():
        provider_font_colors.append(provider_colors.get(row["Provider"], "#e5e7eb"))

        cpu = row["CPU Usage"]
        mem = row["Memory"]
        latency = row["Latency"]
        cost = row["Cost/hr"]

        cpu_colors.append("#f87171" if cpu >= 85 else "#fbbf24" if cpu >= 70 else "#4ade80")
        mem_colors.append("#f87171" if mem >= 85 else "#fbbf24" if mem >= 70 else "#4ade80")
        latency_colors.append("#f87171" if latency >= 100 else "#fbbf24" if latency >= 70 else "#4ade80")
        cost_colors.append("#f87171" if cost >= 3.0 else "#fbbf24" if cost >= 2.0 else "#4ade80")

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[140, 90, 140, 110, 100, 110, 90],
                header=dict(
                    values=[
                        "NODE ID",
                        "PROVIDER",
                        "REGION",
                        "CPU USAGE",
                        "MEMORY",
                        "LATENCY",
                        "COST/HR",
                    ],
                    fill_color="#0f172a",
                    line_color="#0f172a",
                    font=dict(color="#60a5fa", size=13, family="Arial"),
                    align="left",
                    height=36,
                ),
                cells=dict(
                    values=[
                        df["Node ID"],
                        df["Provider"],
                        df["Region"],
                        [f"{v:.1f}%" for v in df["CPU Usage"]],
                        [f"{v:.1f}%" for v in df["Memory"]],
                        [f"{v:.1f} ms" for v in df["Latency"]],
                        [f"${v:.2f}" for v in df["Cost/hr"]],
                    ],
                    fill_color="#0b1220",
                    line_color="#0b1220",
                    align="left",
                    height=34,
                    font=dict(
                        color=[
                            ["#93c5fd"] * len(df),
                            provider_font_colors,
                            ["#cbd5e1"] * len(df),
                            cpu_colors,
                            mem_colors,
                            latency_colors,
                            cost_colors,
                        ],
                        size=13,
                        family="Arial",
                    ),
                ),
            )
        ]
    )

    fig.update_layout(
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        margin=dict(l=0, r=0, t=0, b=0),
        height=310,
    )

    return fig


def run_full_analysis_pipeline(workload: str, hours: int) -> dict:
    metrics = get_current_metrics()
    analysis = analyze_metrics(metrics)
    prediction = predict(metrics)
    decision = decide_action(metrics, analysis, prediction)

    nodes_df = build_nodes()
    forecast_df = build_forecast(hours, workload)

    health_score = round(
        100
        - (
            (metrics["cpu"] * 0.35)
            + (metrics["memory"] * 0.25)
            + (min(metrics["network_latency"], 100) * 0.20)
            + (metrics["cost_per_hour"] * 4.0)
        )
        / 1.6,
        1,
    )
    health_score = max(20.0, min(97.0, health_score))

    issues_detected = int((nodes_df["CPU Usage"] > 80).sum() + (nodes_df["Memory"] > 85).sum())
    monthly_savings = max(0, round((4.5 - metrics["cost_per_hour"]) * 120, 0))
    current_cost = round(nodes_df["Cost/hr"].sum(), 2)

    optimized_cost = current_cost
    if decision["action"] == "scale_down":
        optimized_cost = round(current_cost * 0.82, 2)
    elif decision["action"] == "maintain":
        optimized_cost = round(current_cost * 0.97, 2)
    elif decision["action"] == "optimize_network":
        optimized_cost = round(current_cost * 0.95, 2)

    peak_load = round(forecast_df["Forecast Load"].max(), 1)
    avg_stress = round(forecast_df["Stress Index"].mean(), 1)
    thermal_stress = round(avg_stress * 1.15, 1)
    breach_windows = int((forecast_df["Forecast Load"] > 85).sum())

    if breach_windows >= 4 or decision["priority"] == "high":
        risk_level = "HIGH"
    elif breach_windows >= 1 or decision["priority"] == "medium":
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "run_time": datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"),
        "workload": workload,
        "hours": hours,
        "metrics": metrics,
        "analysis": analysis,
        "prediction": prediction,
        "decision": decision,
        "nodes_df": nodes_df,
        "forecast_df": forecast_df,
        "health_score": health_score,
        "issues_detected": issues_detected,
        "monthly_savings": monthly_savings,
        "current_cost": current_cost,
        "optimized_cost": optimized_cost,
        "peak_load": peak_load,
        "avg_stress": avg_stress,
        "thermal_stress": thermal_stress,
        "breach_windows": breach_windows,
        "risk_level": risk_level,
    }


def save_report_to_disk(result: dict) -> tuple[str, str]:
    payload = {
        "run_time": result["run_time"],
        "workload": result["workload"],
        "hours": result["hours"],
        "metrics": result["metrics"],
        "analysis": result["analysis"],
        "prediction": result["prediction"],
        "decision": result["decision"],
        "health_score": result["health_score"],
        "issues_detected": result["issues_detected"],
        "monthly_savings": result["monthly_savings"],
        "current_cost": result["current_cost"],
        "optimized_cost": result["optimized_cost"],
        "peak_load": result["peak_load"],
        "avg_stress": result["avg_stress"],
        "thermal_stress": result["thermal_stress"],
        "breach_windows": result["breach_windows"],
        "risk_level": result["risk_level"],
    }

    latest_file = REPORTS_DIR / "latest_report.json"
    latest_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    timestamp_safe = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_file = REPORTS_DIR / f"report_{timestamp_safe}.json"
    versioned_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return str(latest_file), str(versioned_file)


st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-left">
            <div class="hero-title-small">SimCloud Dynamics</div>
            <div class="hero-title-main">MCOP + PISE</div>
            <div class="hero-title-sub">Platform v1.0</div>
        </div>
        <div class="hero-right">
            <div class="last-updated-label">Last updated</div>
            <div class="last-updated-value">{datetime.now().strftime("%I:%M:%S %p")}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

status_col1, status_col2, _ = st.columns([0.9, 0.9, 2.2])

with status_col1:
    st.markdown(
        """
        <div class="pill pill-green">
            <span class="pill-dot pill-dot-green"></span>
            <span>MCOP Active</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with status_col2:
    st.markdown(
        """
        <div class="pill pill-blue">
            <span class="pill-dot pill-dot-blue"></span>
            <span>PISE Active</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-title" style="position: relative; top: 160px;">
        MCOP - AI ORCHESTRATED MULTI-CLOUD OPTIMIZATION
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<div style='height:0px;'></div>", unsafe_allow_html=True)

control_1, control_2, control_3 = st.columns([1.15, 1.0, 0.85], vertical_alignment="bottom")

with control_1:
    workload = st.selectbox(
        "Workload",
        ["Steady State", "Burst Load", "Spiky Traffic"],
        index=0,
    )

with control_2:
    duration_label = st.selectbox(
        "Simulation",
        ["24 Hours", "12 Hours", "6 Hours"],
        index=0,
    )

with control_3:
    st.markdown("<div style='height:34px;'></div>", unsafe_allow_html=True)

    refresh_demo = st.button(
        "↻ Refresh Demo Data",
        use_container_width=True,
        type="secondary",
    )

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    run_analysis = st.button(
        "▶ Run Full Analysis",
        use_container_width=True,
        type="primary",
    )

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    save_report = st.button(
        "💾 Save Report",
        use_container_width=True,
        type="secondary",
    )

duration_map = {"24 Hours": 24, "12 Hours": 12, "6 Hours": 6}
hours = duration_map[duration_label]

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = run_full_analysis_pipeline(workload, hours)

if refresh_demo:
    st.session_state.analysis_result = run_full_analysis_pipeline(workload, hours)

if run_analysis:
    with st.spinner("Running full analysis..."):
        time.sleep(1.2)
        st.session_state.analysis_result = run_full_analysis_pipeline(workload, hours)
    st.success("Analysis completed successfully 🚀")

if save_report:
    latest_file, versioned_file = save_report_to_disk(st.session_state.analysis_result)
    st.success(f"Report saved successfully: {latest_file}")
    st.info(f"Versioned copy created: {versioned_file}")

result = st.session_state.analysis_result

metrics = result["metrics"]
analysis = result["analysis"]
prediction = result["prediction"]
decision = result["decision"]
nodes_df = result["nodes_df"]
forecast_df = result["forecast_df"]
health_score = result["health_score"]
issues_detected = result["issues_detected"]
monthly_savings = result["monthly_savings"]
current_cost = result["current_cost"]
optimized_cost = result["optimized_cost"]
peak_load = result["peak_load"]
avg_stress = result["avg_stress"]
thermal_stress = result["thermal_stress"]
breach_windows = result["breach_windows"]
risk_level = result["risk_level"]

metric_cols_top = st.columns(4)
top_cards = [
    ("Infrastructure Health", colored_value(health_score, good_low=False, medium_cutoff=55, high_cutoff=40), "Overall score / 100"),
    ("Active Nodes", colored_value(len(nodes_df), good_low=False, medium_cutoff=4, high_cutoff=2), "Across all providers"),
    ("Issues Detected", colored_value(issues_detected, good_low=True, medium_cutoff=1, high_cutoff=3), "Requiring attention"),
    ("Est. Monthly Savings", colored_value(int(monthly_savings), good_low=False, medium_cutoff=50, high_cutoff=10, prefix="$"), "After optimization"),
]

for col, (title, value_html, sub) in zip(metric_cols_top, top_cards):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                {value_html}
                <div class="metric-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

metric_cols_bottom = st.columns(2)
bottom_cards = [
    ("Current Cost / Hour", colored_value(current_cost, good_low=True, medium_cutoff=8, high_cutoff=11, prefix="$", suffix="/hr"), "Current blended operating rate"),
    ("Optimized Cost / Hour", colored_value(optimized_cost, good_low=True, medium_cutoff=8, high_cutoff=11, prefix="$", suffix="/hr"), "Projected after optimization"),
]

for col, (title, value_html, sub) in zip(metric_cols_bottom, bottom_cards):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                {value_html}
                <div class="metric-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="section-shell">
        <div class="section-title">Multi-Cloud Node Status</div>
    """,
    unsafe_allow_html=True,
)

nodes_table_fig = make_nodes_table(nodes_df)
st.plotly_chart(nodes_table_fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

highest_mem_node = nodes_df.sort_values("Memory", ascending=False).iloc[0]
priority_class = "risk-high" if highest_mem_node["Memory"] >= 85 else "risk-medium" if highest_mem_node["Memory"] >= 70 else "risk-low"
priority_label = "HIGH" if highest_mem_node["Memory"] >= 85 else "MEDIUM" if highest_mem_node["Memory"] >= 70 else "LOW"

st.markdown(
    f"""
    <div class="section-shell">
        <div class="section-title">Optimization Recommendations</div>
        <div class="recommend-card">
            <div class="recommend-title">
                {highest_mem_node["Node ID"]} ({highest_mem_node["Provider"]} / {highest_mem_node["Region"]})
                <span class="risk-badge {priority_class}">{priority_label}</span>
            </div>
            <div style="font-size:18px; color:#fbbf24; margin-bottom:10px;">⚠ Memory pressure ({highest_mem_node["Memory"]}%)</div>
            <div style="font-size:18px; color:#4ade80;">→ Upgrade instance tier or rebalance workload placement</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-shell">
        <div class="section-title">PISE — Physics-Informed Simulation Engine</div>
    </div>
    """,
    unsafe_allow_html=True,
)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_load = make_dark_line_chart(
        forecast_df,
        "Forecast Load",
        "Load Forecast — State-Space Model",
        "#60a5fa",
        "rgba(96,165,250,0.16)",
    )
    st.plotly_chart(fig_load, use_container_width=True)

with chart_col2:
    fig_stress = make_dark_line_chart(
        forecast_df,
        "Stress Index",
        "Infrastructure Stress Index — Physics-Informed Projection",
        "#f87171",
        "rgba(248,113,113,0.14)",
    )
    st.plotly_chart(fig_stress, use_container_width=True)

if risk_level == "HIGH":
    pise_note = "PISE Recommendation: Projected overload windows detected. Apply proactive scaling."
elif risk_level == "MEDIUM":
    pise_note = "PISE Recommendation: Moderate stress expected. Monitor and prepare burst capacity."
else:
    pise_note = "PISE Recommendation: Infrastructure stable. No immediate action required."

sim_left, sim_right = st.columns(2)

with sim_left:
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-title">Simulation Results</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px 28px; font-size:18px;">
                <div><span class="small-note">Workload Type</span><br><strong>{workload.upper()}</strong></div>
                <div><span class="small-note">Duration</span><br><strong>{hours} hours</strong></div>
                <div><span class="small-note">Peak Load Forecast</span><br><strong style="color:#fbbf24;">{peak_load}%</strong></div>
                <div><span class="small-note">Avg Stress Index</span><br><strong style="color:#f87171;">{avg_stress}%</strong></div>
                <div><span class="small-note">Thermal Stress Analog</span><br><strong>{thermal_stress}%</strong></div>
                <div><span class="small-note">Predicted CPU</span><br><strong>{prediction["predicted_cpu"]}%</strong></div>
                <div><span class="small-note">Breach Risk Windows</span><br><strong style="color:#f87171;">{breach_windows}</strong></div>
                <div><span class="small-note">Risk Level</span><br><strong>{risk_level}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sim_right:
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-title">Operational Decision Summary</div>
            <div style="font-size:18px; margin-bottom:10px;"><span class="small-note">MCOP Action</span><br><strong>{decision["action"].replace("_", " ").title()}</strong></div>
            <div style="font-size:18px; margin-bottom:10px;"><span class="small-note">Reason</span><br>{decision["reason"]}</div>
            <div style="font-size:18px; margin-bottom:16px;"><span class="small-note">Optimization Target</span><br>{analysis["optimization_target"].title()}</div>
            <div class="info-box">
                <div style="font-size:20px; font-weight:800;">💡 {pise_note}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='footer-note'>MCOP = AI-Orchestrated Multi-Cloud Optimization | PISE = Physics-Informed Simulation Engine</div>",
    unsafe_allow_html=True,
)