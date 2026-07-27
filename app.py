import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Canada Student Source Dashboard",
    page_icon="🍁",
    layout="wide",
)

WB_API_BASE = "https://api.worldbank.org/v2"

INDICATORS = {
    "total_population":    "SP.POP.TOTL",
    "gdp_per_capita":      "NY.GDP.PCAP.CD",
    "tertiary_enrollment": "SE.TER.ENRR",
    "youth_unemployment":  "SL.UEM.1524.ZS",
    "working_age_pct":     "SP.POP.1564.TO.ZS",
    "internet_users":      "IT.NET.USER.ZS",
}

INDICATOR_LABELS = {
    "total_population":    "Total Population",
    "gdp_per_capita":      "GDP per Capita (USD)",
    "tertiary_enrollment": "Tertiary Enrollment Rate (%)",
    "youth_unemployment":  "Youth Unemployment Rate (%)",
    "working_age_pct":     "Working Age Population (15-64, %)",
    "internet_users":      "Internet Users (% of population)",
    "youth_population":    "Estimated Youth Population (15-29)",
    "composite_score":     "Composite Score (0-100)",
}

REGIONS = [
    "All Regions",
    "East Asia & Pacific",
    "Europe & Central Asia",
    "Latin America & Caribbean",
    "Middle East & North Africa",
    "South Asia",
    "Sub-Saharan Africa",
]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_indicator(code: str, year: int) -> pd.DataFrame:
    url = f"{WB_API_BASE}/country/all/indicator/{code}"
    params = {"date": str(year), "format": "json", "per_page": 300}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()
        if len(payload) < 2 or not payload[1]:
            return pd.DataFrame()
        rows = []
        for item in payload[1]:
            if item.get("value") is not None and item.get("countryiso3code"):
                rows.append({
                    "iso3":    item["countryiso3code"],
                    "country": item["country"]["value"],
                    "region":  item.get("region", {}).get("value", ""),
                    code:      item["value"],
                })
        return pd.DataFrame(rows)
    except Exception as exc:
        st.warning(f"Could not fetch indicator {code}: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(year: int) -> pd.DataFrame:
    frames = []
    for key, code in INDICATORS.items():
        df = fetch_indicator(code, year)
        if not df.empty:
            df = df.rename(columns={code: key})
            frames.append((key, df[["iso3", "country", "region", key]]))

    if not frames:
        return pd.DataFrame()

    merged = frames[0][1]
    for key, df in frames[1:]:
        merged = pd.merge(merged, df[["iso3", key]], on="iso3", how="outer")

    merged = merged[merged["iso3"].str.len() == 3]
    merged = merged[~merged["region"].str.contains("Aggregates", na=True)]
    merged = merged[merged["region"].notna() & (merged["region"] != "")]

    merged["youth_population"] = (
        merged["total_population"] * merged["working_age_pct"] / 100 * 0.30
    )

    return merged.reset_index(drop=True)


def minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    norm = (series - mn) / (mx - mn)
    return (1 - norm) if invert else norm


def gdp_sweet_spot(series: pd.Series) -> pd.Series:
    low, high, taper = 4_000, 22_000, 100_000
    result = pd.Series(np.nan, index=series.index)
    for i, v in series.items():
        if pd.isna(v):
            continue
        if v < low:
            result[i] = v / low
        elif v <= high:
            result[i] = 1.0
        else:
            result[i] = max(0.0, 1.0 - (v - high) / taper)
    return result


def score_data(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    out = df.copy()

    out["s_youth"]    = minmax(out["youth_population"])
    out["s_gdp"]      = gdp_sweet_spot(out["gdp_per_capita"])
    out["s_tertiary"] = minmax(out["tertiary_enrollment"])
    out["s_unemploy"] = minmax(out["youth_unemployment"])
    out["s_internet"] = minmax(out["internet_users"])

    out["composite_score"] = (
        weights["youth"]    * out["s_youth"].fillna(0) +
        weights["gdp"]      * out["s_gdp"].fillna(0) +
        weights["tertiary"] * out["s_tertiary"].fillna(0) +
        weights["unemploy"] * out["s_unemploy"].fillna(0) +
        weights["internet"] * out["s_internet"].fillna(0)
    )

    out["composite_score"] = minmax(out["composite_score"]) * 100
    out["rank"] = (
        out["composite_score"].rank(ascending=False, method="min").astype("Int64")
    )

    return out.sort_values("composite_score", ascending=False).reset_index(drop=True)


def main():
    st.title("🍁 Canada International Student Source Dashboard")
    st.markdown(
        "Rank and explore countries by their potential as sources of international "
        "students to Canada, using live demographic and economic data from the "
        "[World Bank Open Data API](https://api.worldbank.org/v2)."
    )

    with st.sidebar:
        st.header("Settings")

        year = st.selectbox(
            "Reference Year",
            options=list(range(2023, 2018, -1)),
            index=0,
        )

        region_filter = st.selectbox("Region Filter", REGIONS)
        top_n = st.slider("Top N countries to highlight", 10, 50, 20)

        st.divider()
        st.subheader("Scoring Weights")
        st.caption("Drag sliders to adjust how much each factor contributes. Weights are auto-normalised.")

        w_youth  = st.slider("Youth Population Size",                    0, 10, 3)
        w_gdp    = st.slider("GDP per Capita (middle-income sweet spot)", 0, 10, 3)
        w_tert   = st.slider("Tertiary Enrollment Rate",                  0, 10, 2)
        w_unemp  = st.slider("Youth Unemployment (push factor)",          0, 10, 1)
        w_inet   = st.slider("Internet Connectivity",                     0, 10, 1)

        total_w = w_youth + w_gdp + w_tert + w_unemp + w_inet
        if total_w == 0:
            st.error("At least one weight must be greater than zero.")
            st.stop()

        weights = {
            "youth":    w_youth  / total_w,
            "gdp":      w_gdp    / total_w,
            "tertiary": w_tert   / total_w,
            "unemploy": w_unemp  / total_w,
            "internet": w_inet   / total_w,
        }

        st.divider()
        st.caption("Data: World Bank Open Data · App built with Streamlit + Plotly")

    with st.spinner("Fetching live data from the World Bank API..."):
        raw = load_data(year)

    if raw.empty:
        st.error("No data could be loaded. Please check your internet connection and try again.")
        st.stop()

    if region_filter != "All Regions":
        raw = raw[raw["region"].str.contains(region_filter, na=False)]
        if raw.empty:
            st.warning(f"No data found for region: {region_filter}")
            st.stop()

    scored = score_data(raw, weights)
    top    = scored.dropna(subset=["composite_score"]).head(top_n)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Countries Analysed", f"{scored['composite_score'].notna().sum():,}")
    k2.metric("Top Country",        top.iloc[0]["country"] if not top.empty else "-")
    k3.metric("Top Score",          f"{top.iloc[0]['composite_score']:.1f}/100" if not top.empty else "-")
    k4.metric("Median Score",       f"{scored['composite_score'].median():.1f}/100")
    k5.metric("Reference Year",     str(year))

    st.divider()

    tab_map, tab_rank, tab_explore, tab_data = st.tabs(
        ["World Map", "Rankings", "Explorer", "Data"]
    )

    with tab_map:
        st.subheader("Student Source Potential - Global View")
        map_df = scored.dropna(subset=["composite_score"])
        fig_map = px.choropleth(
            map_df,
            locations="iso3",
            color="composite_score",
            hover_name="country",
            hover_data={
                "iso3": False,
                "composite_score": ":.1f",
                "rank": True,
                "gdp_per_capita": ":,.0f",
                "tertiary_enrollment": ":.1f",
                "youth_unemployment": ":.1f",
            },
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            labels={
                "composite_score": "Score",
                "gdp_per_capita": "GDP per Capita (USD)",
                "tertiary_enrollment": "Tertiary Enroll. (%)",
                "youth_unemployment": "Youth Unemp. (%)",
            },
        )
        fig_map.update_layout(
            geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
            coloraxis_colorbar=dict(title="Score (0-100)"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=500,
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with tab_rank:
        col_a, col_b = st.columns([1, 1], gap="large")

        with col_a:
            st.subheader(f"Top {top_n} Countries")
            fig_bar = px.bar(
                top.iloc[::-1],
                x="composite_score",
                y="country",
                orientation="h",
                color="composite_score",
                color_continuous_scale="RdYlGn",
                range_color=[0, 100],
                hover_data={"region": True, "gdp_per_capita": ":,.0f"},
                labels={"composite_score": "Score", "country": ""},
            )
            fig_bar.update_layout(
                height=max(420, top_n * 22),
                coloraxis_showscale=False,
                margin=dict(l=0, r=10, t=10, b=0),
                yaxis=dict(tickfont=dict(size=11)),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            st.subheader("Score Factor Breakdown - Top 15")
            top15 = top.head(15).copy()
            factor_map = {
                "s_youth":    "Youth Population",
                "s_gdp":      "GDP Sweet Spot",
                "s_tertiary": "Tertiary Enrollment",
                "s_unemploy": "Youth Unemployment",
                "s_internet": "Internet Access",
            }
            weight_map = {
                "s_youth": "youth",
                "s_gdp": "gdp",
                "s_tertiary": "tertiary",
                "s_unemploy": "unemploy",
                "s_internet": "internet",
            }
            for col, label in factor_map.items():
                top15[col + "_w"] = top15[col].fillna(0) * weights.get(weight_map[col], 0)

            breakdown = top15[["country"] + [c + "_w" for c in factor_map]].melt(
                id_vars="country", var_name="factor", value_name="value"
            )
            breakdown["factor"] = breakdown["factor"].str.replace("_w", "").map(
                {k: v for k, v in factor_map.items()}
            )
            fig_stack = px.bar(
                breakdown, x="value", y="country", color="factor",
                orientation="h", barmode="stack",
                labels={"value": "Weighted Score", "country": "", "factor": "Factor"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_stack.update_layout(
                height=450,
                margin=dict(l=0, r=10, t=10, b=0),
                yaxis=dict(categoryorder="total ascending", tickfont=dict(size=11)),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_stack, use_container_width=True)

    with tab_explore:
        st.subheader("Indicator Explorer")
        exp_cols = [k for k in INDICATOR_LABELS.keys() if k in scored.columns]
        col1, col2 = st.columns(2)
        x_key = col1.selectbox("X Axis", exp_cols, index=1, format_func=lambda k: INDICATOR_LABELS[k])
        y_key = col2.selectbox("Y Axis", exp_cols, index=2, format_func=lambda k: INDICATOR_LABELS[k])

        plot_df = scored.dropna(subset=["composite_score"])
        valid   = plot_df.dropna(subset=[x_key, y_key])

        fig_scatter = px.scatter(
            valid, x=x_key, y=y_key,
            size="composite_score", color="region",
            hover_name="country",
            hover_data={"composite_score": ":.1f", "rank": True, "region": False},
            labels={
                x_key: INDICATOR_LABELS.get(x_key, x_key),
                y_key: INDICATOR_LABELS.get(y_key, y_key),
                "composite_score": "Score",
            },
            size_max=45, opacity=0.8,
        )
        fig_scatter.update_layout(height=520, legend=dict(title="Region"))
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.divider()
        st.subheader("Country Profile")
        country_list = scored["country"].dropna().sort_values().tolist()
        selected = st.selectbox("Select a country to profile", country_list)

        row = scored[scored["country"] == selected]
        if not row.empty:
            r = row.iloc[0]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Composite Score",  f"{r['composite_score']:.1f}/100"  if pd.notna(r["composite_score"])     else "N/A")
            m2.metric("Global Rank",      f"#{int(r['rank'])}"                if pd.notna(r["rank"])                else "N/A")
            m3.metric("GDP per Capita",   f"${r['gdp_per_capita']:,.0f}"      if pd.notna(r["gdp_per_capita"])      else "N/A")
            m4.metric("Tertiary Enroll.", f"{r['tertiary_enrollment']:.1f}%"  if pd.notna(r["tertiary_enrollment"]) else "N/A")
            m5.metric("Youth Unemploy.",  f"{r['youth_unemployment']:.1f}%"   if pd.notna(r["youth_unemployment"])  else "N/A")

            labels   = ["Youth Population", "GDP Sweet Spot", "Tertiary Enrollment", "Youth Unemployment", "Internet Access"]
            raw_vals = [
                float(r.get("s_youth",    0) or 0),
                float(r.get("s_gdp",      0) or 0),
                float(r.get("s_tertiary", 0) or 0),
                float(r.get("s_unemploy", 0) or 0),
                float(r.get("s_internet", 0) or 0),
            ]
            vals = raw_vals + [raw_vals[0]]
            cats = labels   + [labels[0]]

            fig_radar = go.Figure(go.Scatterpolar(
                r=vals, theta=cats, fill="toself",
                line_color="#D62728",
                fillcolor="rgba(214, 39, 40, 0.15)",
                name=selected,
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                title=f"{selected} - Factor Profile",
                height=420,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab_data:
        st.subheader("Full Scored Dataset")
        display = scored[[
            "rank", "country", "region", "composite_score",
            "gdp_per_capita", "total_population",
            "tertiary_enrollment", "youth_unemployment", "internet_users",
        ]].rename(columns={
            "rank":                "Rank",
            "country":             "Country",
            "region":              "Region",
            "composite_score":     "Score (0-100)",
            "gdp_per_capita":      "GDP per Capita (USD)",
            "total_population":    "Total Population",
            "tertiary_enrollment": "Tertiary Enrollment (%)",
            "youth_unemployment":  "Youth Unemployment (%)",
            "internet_users":      "Internet Users (%)",
        }).dropna(subset=["Score (0-100)"]).sort_values("Rank")

        st.dataframe(
            display.style.format({
                "Score (0-100)":           "{:.1f}",
                "GDP per Capita (USD)":    "${:,.0f}",
                "Total Population":        "{:,.0f}",
                "Tertiary Enrollment (%)": "{:.1f}",
                "Youth Unemployment (%)":  "{:.1f}",
                "Internet Users (%)":      "{:.1f}",
            }, na_rep="N/A").background_gradient(subset=["Score (0-100)"], cmap="RdYlGn"),
            use_container_width=True,
            height=520,
        )

        csv = display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download as CSV", data=csv,
            file_name=f"canada_student_sources_{year}.csv",
            mime="text/csv",
        )

    st.divider()
    st.caption(
        f"Data: World Bank Open Data API · Reference year: {year} · "
        "Composite score is a weighted index of normalised indicators. "
        "Intended for exploratory analysis only."
    )


if __name__ == "__main__":
    main()


