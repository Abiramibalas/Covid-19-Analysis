from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

np.random.seed(42)

# ── 1. GENERATE REALISTIC INDIA COVID DATA ──────────────────────────────────

states = [
    "Maharashtra", "Kerala", "Karnataka", "Tamil Nadu", "Delhi",
    "Uttar Pradesh", "West Bengal", "Andhra Pradesh", "Rajasthan", "Gujarat",
    "Madhya Pradesh", "Odisha", "Telangana", "Bihar", "Punjab"
]

state_population = {
    "Maharashtra": 12400, "Kerala": 3500, "Karnataka": 6700, "Tamil Nadu": 7700,
    "Delhi": 2000, "Uttar Pradesh": 23500, "West Bengal": 9800,
    "Andhra Pradesh": 5300, "Rajasthan": 8000, "Gujarat": 6800,
    "Madhya Pradesh": 8500, "Odisha": 4600, "Telangana": 3900,
    "Bihar": 12800, "Punjab": 3000
}

# Total cases per state (thousands)
state_cases = {
    "Maharashtra": 7890, "Kerala": 6720, "Karnataka": 4050, "Tamil Nadu": 3440,
    "Delhi": 2010, "Uttar Pradesh": 2100, "West Bengal": 2050,
    "Andhra Pradesh": 2330, "Rajasthan": 1280, "Gujarat": 1250,
    "Madhya Pradesh": 1050, "Odisha": 1340, "Telangana": 900,
    "Bihar": 760, "Punjab": 760
}
state_deaths = {s: int(state_cases[s] * np.random.uniform(0.008, 0.022)) for s in states}
state_recovered = {s: int(state_cases[s] * np.random.uniform(0.91, 0.97)) for s in states}
state_vaccinated = {s: int(state_population[s] * np.random.uniform(0.60, 0.92)) for s in states}

df_state = pd.DataFrame({
    "State": states,
    "Total_Cases": [state_cases[s] * 1000 for s in states],
    "Deaths": [state_deaths[s] * 1000 for s in states],
    "Recovered": [state_recovered[s] * 1000 for s in states],
    "Vaccinated": [state_vaccinated[s] * 10000 for s in states],
    "Population": [state_population[s] * 10000 for s in states],
})
df_state["Active"] = df_state["Total_Cases"] - df_state["Deaths"] - df_state["Recovered"]
df_state["Active"] = df_state["Active"].clip(lower=0)
df_state["CFR"] = (df_state["Deaths"] / df_state["Total_Cases"] * 100).round(2)
df_state["Vax_Pct"] = (df_state["Vaccinated"] / df_state["Population"] * 100).round(1)

# Time series — daily new cases (Mar 2020 – Dec 2022)
dates = pd.date_range("2020-03-01", "2022-12-31", freq="D")
n = len(dates)

def wave(peak_day, width, height):
    x = np.arange(n)
    return height * np.exp(-((x - peak_day) ** 2) / (2 * width ** 2))

daily = (
    wave(60,  30,  8000)   +   # Wave 1
    wave(220, 25,  95000)  +   # Wave 2
    wave(290, 40,  420000) +   # Wave 3 (Delta)
    wave(640, 20,  320000) +   # Wave 4 (Omicron)
    np.random.normal(0, 3000, n)
).clip(min=0).astype(int)

df_ts = pd.DataFrame({"Date": dates, "Daily_Cases": daily})
df_ts["7day_avg"] = df_ts["Daily_Cases"].rolling(7, center=True).mean().bfill().ffill()
df_ts["Cumulative"] = df_ts["Daily_Cases"].cumsum()
df_ts["Month"] = df_ts["Date"].dt.to_period("M").astype(str)

# Monthly vaccination rollout
vax_dates = pd.date_range("2021-01-16", "2022-12-31", freq="D")
vax_n = len(vax_dates)

def wave_v(peak_day, width, height, n):
    x = np.arange(n)
    return height * np.exp(-((x - peak_day) ** 2) / (2 * width ** 2))

vax_daily = (
    wave_v(60,  40, 800000, vax_n)  +
    wave_v(200, 50, 4500000, vax_n) +
    wave_v(350, 60, 6000000, vax_n) +
    np.random.normal(0, 150000, vax_n)
).clip(min=100000).astype(int)
df_vax = pd.DataFrame({"Date": vax_dates, "Doses_Per_Day": vax_daily})
df_vax["Cumulative_Doses"] = df_vax["Doses_Per_Day"].cumsum()

# ── 2. SUMMARY METRICS ──────────────────────────────────────────────────────

total_cases    = int(df_state["Total_Cases"].sum())
total_deaths   = int(df_state["Deaths"].sum())
total_recovered= int(df_state["Recovered"].sum())
total_active   = int(df_state["Active"].sum())
total_vax      = int(df_vax["Cumulative_Doses"].iloc[-1])
recovery_rate  = round(total_recovered / total_cases * 100, 1)
cfr            = round(total_deaths / total_cases * 100, 2)

# ── 3. COLOUR PALETTE ───────────────────────────────────────────────────────

C_BG      = "#0f1117"
C_CARD    = "#1a1d27"
C_BORDER  = "#2a2d3a"
C_TEXT    = "#e8eaf0"
C_MUTED   = "#8b90a8"
C_BLUE    = "#4f9cf9"
C_GREEN   = "#2dd4a0"
C_RED     = "#f26c6c"
C_AMBER   = "#f6ad55"
C_PURPLE  = "#a78bfa"
C_WAVE    = "#60a5fa"

FONT = "Inter, system-ui, sans-serif"

# ── 4. CHART HELPERS ────────────────────────────────────────────────────────

def base_layout(title="", height=380):
    return dict(
        title=dict(text=title, font=dict(color=C_TEXT, size=14, family=FONT), x=0.01, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_MUTED, family=FONT, size=11),
        height=height,
        margin=dict(l=10, r=10, t=44, b=10),
        xaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, tickcolor=C_BORDER, zerolinecolor=C_BORDER),
        yaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, tickcolor=C_BORDER, zerolinecolor=C_BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hovermode="x unified",
    )

def to_html(fig, div_id):
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                       config={"displayModeBar": False, "responsive": True})

# ── CHART 1: Daily cases + 7-day average ─────────────────────────────────────
fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=df_ts["Date"], y=df_ts["Daily_Cases"],
    name="Daily cases", marker_color=C_BLUE, opacity=0.35,
    hovertemplate="%{y:,.0f}",
))
fig1.add_trace(go.Scatter(
    x=df_ts["Date"], y=df_ts["7day_avg"],
    name="7-day avg", line=dict(color=C_AMBER, width=2),
    hovertemplate="%{y:,.0f}",
))

wave_labels = [
    ("2020-06-15", "Wave 1"),
    ("2021-05-05", "Wave 2 (Delta)"),
    ("2021-10-15", "Wave 3"),
    ("2022-01-15", "Wave 4 (Omicron)"),
]
for d, lbl in wave_labels:
    fig1.add_vline(x=d, line_dash="dot", line_color=C_BORDER, line_width=1)
    fig1.add_annotation(x=d, y=1, yref="paper", text=lbl, showarrow=False,
                        font=dict(size=9, color=C_MUTED), yanchor="top", xanchor="left", xshift=4)

fig1.update_layout(**base_layout("Daily new cases — India (Mar 2020 – Dec 2022)", 340))
c1 = to_html(fig1, "fig1")

# ── CHART 2: Top 10 states bar ───────────────────────────────────────────────
top10 = df_state.nlargest(10, "Total_Cases").sort_values("Total_Cases")
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    y=top10["State"], x=top10["Total_Cases"],
    orientation="h", name="Cases",
    marker=dict(color=top10["Total_Cases"], colorscale=[[0,"#1e3a5f"],[0.5,C_BLUE],[1,"#bfdbfe"]]),
    text=[f"{v/1e6:.2f}M" for v in top10["Total_Cases"]],
    textposition="outside", textfont=dict(color=C_TEXT, size=11),
    hovertemplate="%{y}: %{x:,.0f} cases<extra></extra>",
))
fig2.update_layout(**base_layout("Top 10 states by total cases", 360))
fig2.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
c2 = to_html(fig2, "fig2")

# ── CHART 3: CFR scatter ─────────────────────────────────────────────────────
fig3 = px.scatter(
    df_state, x="Total_Cases", y="CFR", size="Deaths",
    color="CFR", color_continuous_scale=[[0, C_GREEN],[0.5, C_AMBER],[1, C_RED]],
    hover_name="State",
    hover_data={"Total_Cases": ":,.0f", "CFR": ":.2f", "Deaths": ":,.0f"},
    size_max=40,
)
fig3.update_traces(marker=dict(line=dict(width=0)))
fig3.update_layout(**base_layout("Case fatality rate vs total cases (bubble = deaths)", 360))
fig3.update_layout(coloraxis_showscale=False, xaxis_title="Total cases", yaxis_title="CFR %")
c3 = to_html(fig3, "fig3")

# ── CHART 4: Vaccination progress ───────────────────────────────────────────
df_vax_m = df_vax.set_index("Date").resample("W")["Doses_Per_Day"].sum().reset_index()
df_vax_m.columns = ["Date", "Weekly_Doses"]
df_vax_m["Cumulative"] = df_vax_m["Weekly_Doses"].cumsum()

fig4 = make_subplots(specs=[[{"secondary_y": True}]])
fig4.add_trace(go.Bar(
    x=df_vax_m["Date"], y=df_vax_m["Weekly_Doses"],
    name="Weekly doses", marker_color=C_GREEN, opacity=0.6,
    hovertemplate="%{y:,.0f}",
), secondary_y=False)
fig4.add_trace(go.Scatter(
    x=df_vax_m["Date"], y=df_vax_m["Cumulative"],
    name="Cumulative doses", line=dict(color=C_PURPLE, width=2.5),
    hovertemplate="%{y:,.0f}",
), secondary_y=True)
fig4.update_layout(**base_layout("Vaccination rollout — weekly doses & cumulative", 340))
fig4.update_yaxes(gridcolor=C_BORDER, linecolor=C_BORDER, tickcolor=C_BORDER, secondary_y=False)
fig4.update_yaxes(gridcolor="rgba(0,0,0,0)", showgrid=False, secondary_y=True)
c4 = to_html(fig4, "fig4")

# ── CHART 5: Recovery vs Active vs Deaths donut ───────────────────────────────
fig5 = go.Figure(go.Pie(
    labels=["Recovered", "Active", "Deaths"],
    values=[total_recovered, total_active, total_deaths],
    hole=0.6,
    marker=dict(colors=[C_GREEN, C_AMBER, C_RED], line=dict(color=C_BG, width=3)),
    textinfo="label+percent",
    textfont=dict(color=C_TEXT, size=12),
    hovertemplate="%{label}: %{value:,.0f} (%{percent})<extra></extra>",
))
fig5.add_annotation(
    text=f"<b>{total_cases/1e6:.1f}M</b><br><span style='font-size:11px'>total cases</span>",
    x=0.5, y=0.5, showarrow=False,
    font=dict(color=C_TEXT, size=16, family=FONT),
    align="center",
)
fig5.update_layout(**base_layout("Case outcome breakdown", 340))
fig5.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.08))
c5 = to_html(fig5, "fig5")

# ── CHART 6: State vaccination coverage bar ───────────────────────────────────
df_vax_state = df_state.sort_values("Vax_Pct", ascending=True)
colors_bar = [C_GREEN if v >= 75 else C_AMBER if v >= 55 else C_RED for v in df_vax_state["Vax_Pct"]]
fig6 = go.Figure(go.Bar(
    y=df_vax_state["State"], x=df_vax_state["Vax_Pct"],
    orientation="h",
    marker_color=colors_bar,
    text=[f"{v}%" for v in df_vax_state["Vax_Pct"]],
    textposition="outside", textfont=dict(color=C_TEXT, size=11),
    hovertemplate="%{y}: %{x:.1f}% vaccinated<extra></extra>",
))
fig6.add_vline(x=75, line_dash="dash", line_color=C_GREEN, line_width=1.5,
               annotation_text="75% target", annotation_font=dict(color=C_GREEN, size=10),
               annotation_position="top right")
fig6.update_layout(**base_layout("Vaccination coverage by state (%)", 420))
fig6.update_layout(xaxis_title="% of population vaccinated", showlegend=False, xaxis_range=[0, 110])
c6 = to_html(fig6, "fig6")

# ── 5. ASSEMBLE HTML DASHBOARD ──────────────────────────────────────────────

def metric_card(label, value, sub, color):
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{color}">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>"""

kpis = "".join([
    metric_card("Total cases",    f"{total_cases/1e6:.2f}M",     "confirmed since Mar 2020",  C_BLUE),
    metric_card("Total deaths",   f"{total_deaths/1e5:.1f}L",    f"CFR: {cfr}%",              C_RED),
    metric_card("Recovered",      f"{total_recovered/1e6:.2f}M", f"recovery rate {recovery_rate}%", C_GREEN),
    metric_card("Active cases",   f"{total_active/1e3:.0f}K",    "at peak wave 4",            C_AMBER),
    metric_card("Doses given",    f"{total_vax/1e9:.2f}B",       "cumulative doses",          C_PURPLE),
    metric_card("States tracked", "15",                          "across India",              C_MUTED),
])

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>COVID-19 India Public Health Tracker</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: {C_BG};
    color: {C_TEXT};
    font-family: {FONT};
    font-size: 13px;
    min-height: 100vh;
  }}
  .dashboard {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px 20px 40px;
  }}
  .header {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 24px;
    flex-wrap: wrap;
    gap: 12px;
  }}
  .header-left h1 {{
    font-size: 22px;
    font-weight: 700;
    color: {C_TEXT};
    margin-bottom: 4px;
  }}
  .header-left p {{
    font-size: 13px;
    color: {C_MUTED};
  }}
  .badge {{
    background: #1e2535;
    border: 1px solid {C_BORDER};
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    color: {C_MUTED};
  }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
  }}
  .kpi-card {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 16px 18px;
  }}
  .kpi-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: {C_MUTED};
    margin-bottom: 6px;
  }}
  .kpi-value {{
    font-size: 26px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
  }}
  .kpi-sub {{
    font-size: 11px;
    color: {C_MUTED};
  }}
  .charts-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
  }}
  .chart-card {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 4px 8px 8px;
    overflow: hidden;
  }}
  .chart-card.full {{ grid-column: 1 / -1; }}
  .footer {{
    margin-top: 24px;
    text-align: center;
    font-size: 11px;
    color: {C_MUTED};
  }}
  @media (max-width: 700px) {{
    .charts-grid {{ grid-template-columns: 1fr; }}
    .chart-card.full {{ grid-column: 1; }}
  }}
</style>
</head>
<body>
<div class="dashboard">

  <div class="header">
    <div class="header-left">
      <h1>COVID-19 India — Public Health Tracker</h1>
      <p>Simulated data · Mar 2020 – Dec 2022 · 15 states · Built with Python & Plotly</p>
    </div>
    <div class="badge">Data Analyst Portfolio Project</div>
  </div>

  <div class="kpi-grid">
    {kpis}
  </div>

  <div class="charts-grid">
    <div class="chart-card full">{c1}</div>
    <div class="chart-card">{c2}</div>
    <div class="chart-card">{c3}</div>
    <div class="chart-card full">{c4}</div>
    <div class="chart-card">{c5}</div>
    <div class="chart-card">{c6}</div>
  </div>

  <div class="footer">
    Built with Python · Plotly · Pandas &nbsp;|&nbsp; Portfolio project — COVID-19 Public Health Tracker
  </div>

</div>
</body>
</html>"""

output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
out = output_dir / "covid_dashboard.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Dashboard saved → {out}")
print(f"Total cases: {total_cases:,}")
print(f"Total deaths: {total_deaths:,}")
print(f"Recovery rate: {recovery_rate}%")
print(f"CFR: {cfr}%")