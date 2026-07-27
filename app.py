import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import wbgapi as wb

st.set_page_config(
    page_title="Canada Student Source Dashboard",
    page_icon="🍁",
    layout="wide",
)

# ─── Constants ────────────────────────────────────────────────────────────────

INDICATORS = {
    "total_population":    "SP.POP.TOTL",
    "gdp_per_capita":      "NY.GDP.PCAP.CD",
    "tertiary_enrollment": "SE.TER.ENRR",
    "youth_unemployment":  "SL.UEM.1524.ZS",
    "working_age_pct":     "SP.POP.1564.TO.ZS",
    "internet_users":      "IT.NET.USER.ZS",
    # Program-fit extension indicators
    "rnd_expenditure":     "GB.XPD.RSDV.GD.ZS",   # R&D expenditure (% GDP)
    "health_exp_pc":       "SH.XPD.CHEX.PC.CD",   # Health expenditure per capita
    "physicians_per_1k":   "SH.MED.PHYS.ZS",      # Physicians per 1,000 people
    "mobile_subs":         "IT.CEL.SETS.P2",      # Mobile subscriptions per 100
    "trade_pct_gdp":       "NE.TRD.GNFS.ZS",      # Trade (% of GDP)
    "urban_pct":           "SP.URB.TOTL.IN.ZS",   # Urban population (%)
    "edu_exp_pct_gdp":     "SE.XPD.TOTL.GD.ZS",   # Education expenditure (% GDP)
    "secondary_enroll":    "SE.SEC.ENRR",         # Secondary enrollment (%)
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
    "rnd_expenditure":     "R&D Expenditure (% of GDP)",
    "health_exp_pc":       "Health Expenditure per Capita (USD)",
    "physicians_per_1k":   "Physicians per 1,000 People",
    "mobile_subs":         "Mobile Subscriptions per 100 People",
    "trade_pct_gdp":       "Trade (% of GDP)",
    "urban_pct":           "Urban Population (%)",
    "edu_exp_pct_gdp":     "Education Expenditure (% of GDP)",
    "secondary_enroll":    "Secondary Enrollment Rate (%)",
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

# Each program maps score-column -> weight. Unlisted scores default to 0.
# "invert" scores (physicians_per_1k) reward LOW supply as a proxy for
# unmet domestic demand driving students abroad.
PROGRAM_PROFILES = {
    "General (Balanced)": {
        "s_youth": 3, "s_gdp": 3, "s_tertiary": 2, "s_unemploy": 1, "s_internet": 1,
    },
    "Engineering & Computer Science": {
        "s_tertiary": 3, "s_internet": 3, "s_rnd": 3, "s_mobile": 2,
        "s_youth": 2, "s_gdp": 1,
    },
    "Business & MBA": {
        "s_gdp": 3, "s_trade": 3, "s_urban": 2, "s_tertiary": 2, "s_youth": 1,
    },
    "Health Sciences": {
        "s_physicians_inv": 3, "s_health": 2, "s_youth": 2, "s_tertiary": 2, "s_gdp": 1,
    },
    "Arts & Humanities": {
        "s_urban": 2, "s_secondary": 2, "s_internet": 1, "s_gdp": 1, "s_youth": 1,
    },
    "Law": {
        "s_gdp": 2, "s_tertiary": 2, "s_urban": 2, "s_trade": 1, "s_youth": 1,
    },
    "Education": {
        "s_edu": 3, "s_secondary": 3, "s_tertiary": 2, "s_youth": 1,
    },
    "Science (Pure / Applied)": {
        "s_rnd": 3, "s_tertiary": 3, "s_secondary": 2, "s_internet": 1, "s_youth": 1,
    },
}

ALL_SCORE_KEYS = [
    "s_youth", "s_gdp", "s_tertiary", "s_unemploy", "s_internet",
    "s_rnd", "s_health", "s_physicians_inv", "s_mobile", "s_trade",
    "s_urban", "s_edu", "s_secondary",
]

SCORE_LABELS = {
    "s_youth": "Youth Population",
    "s_gdp": "GDP Sweet Spot",
    "s_tertiary": "Tertiary Enrollment",
    "s_unemploy": "Youth Unemployment",
    "s_internet": "Internet Access",
    "s_rnd": "R&D Investment",
    "s_health": "Health Expenditure",
    "s_physicians_inv": "Unmet Health Capacity",
    "s_mobile": "Mobile Connectivity",
    "s_trade": "Trade Openness",
    "s_urban": "Urbanisation",
    "s_edu": "Education Investment",
    "s_secondary": "Secondary Enrollment",
}

# ─── Data Fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(year: int) -> pd.DataFrame:
    indicator_codes = list(INDICATORS.values())
    code_to_key = {v: k for k, v in INDICATORS.items()}

    try:
        raw = wb.data.DataFrame(
            indicator_codes, time=year, labels=True, numericTimeKeys=True,
        )
    except Exception as e:
        st.error(f"Failed to fetch data from World Bank API: {e}")
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    raw = raw.reset_index()
    raw = raw.rename(columns=code_to_key)

    eco_col = next((c for c in raw.columns if c.lower() == "economy"), None)
    if eco_col is None:
        st.error("Unexpected data format from World Bank API.")
        return pd.DataFrame()

    keep = [eco_col] + [k for k in INDICATORS if k in raw.columns]
    df = raw[keep].copy()
    df = df.rename(columns={eco_col: "iso3"})

    try:
        meta = wb.economy.DataFrame(labels=True).reset_index()
        meta = meta[["id", "name", "region"]].rename(
            columns={"id": "iso3", "name": "country", "region": "region_code"}
        )
        try:
            reg_meta = wb.region.info()
            reg_map = {r["code"]: r["name"] for r in reg_meta.items}
        except Exception:
            reg_map = {}
        meta["region"] = meta["region_code"].map(reg_map).fillna(meta["region_code"])
        meta = meta[["iso3", "country", "region"]]
    except Exception:
        meta = pd.DataFrame(columns=["iso3", "country", "region"])

    df = pd.merge(df, meta, on="iso3", how="left")

    df = df[df["iso3"].str.len() == 3]
    df = df[df["region"].notna() & (df["region"] != "")]
    df = df[~df["region"].str.contains("Aggregates", na=True)]

    if "total_population" in df.columns and "working_age_pct" in df.columns:
        df["youth_population"] = (
            df["total_population"] * df["working_age_pct"] / 100 * 0.30
        )

    return df.reset_index(drop=True)


def load_york_csv(uploaded_file) -> pd.DataFrame:
    """Load and normalise a York applicant CSV. Expected columns:
    country, applicants, enrolled (enrolled optional)."""
    try:
        ydf = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded CSV: {e}")
        return pd.DataFrame()

    ydf.columns = [c.strip().lower() for c in ydf.columns]
    if "country" not in ydf.columns or "applicants" not in ydf.columns:
        st.error("CSV must contain at least 'country' and 'applicants' columns.")
        return pd.DataFrame()

    ydf["country"] = ydf["country"].astype(str).str.strip()
    ydf["applicants"] = pd.to_numeric(ydf["applicants"], errors="coerce")
    if "enrolled" in ydf.columns:
        ydf["enrolled"] = pd.to_numeric(ydf["enrolled"], errors="coerce")
        ydf["acceptance_rate"] = (ydf["enrolled"] / ydf["applicants"] * 100).round(1)

    return ydf


# ─── Scoring ──────────────────────────────────────────────────────────────────

def minmax(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


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


def build_sub_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["s_youth"]          = minmax(out["youth_population"])    if "youth_population"  in out.columns else np.nan
    out["s_gdp"]            = gdp_sweet_spot(out["gdp_per_capita"]) if "gdp_per_capita"  in out.columns else np.nan
    out["s_tertiary"]       = minmax(out["tertiary_enrollment"]) if "tertiary_enrollment" in out.columns else np.nan
    out["s_unemploy"]       = minmax(out["youth_unemployment"])  if "youth_unemployment"  in out.columns else np.nan
    out["s_internet"]       = minmax(out["internet_users"])      if "internet_users"      in out.columns else np.nan
    out["s_rnd"]            = minmax(out["rnd_expenditure"])     if "rnd_expenditure"     in out.columns else np.nan
    out["s_health"]         = minmax(out["health_exp_pc"])       if "health_exp_pc"       in out.columns else np.nan
    out["s_physicians_inv"] = minmax(1 / out["physicians_per_1k"].replace(0, np.nan)) if "physicians_per_1k" in out.columns else np.nan
    out["s_mobile"]         = minmax(out["mobile_subs"])         if "mobile_subs"         in out.columns else np.nan
    out["s_trade"]          = minmax(out["trade_pct_gdp"])       if "trade_pct_gdp"       in out.columns else np.nan
    out["s_urban"]          = minmax(out["urban_pct"])           if "urban_pct"           in out.columns else np.nan
    out["s_edu"]            = minmax(out["edu_exp_pct_gdp"])     if "edu_exp_pct_gdp"     in out.columns else np.nan
    out["s_secondary"]      = minmax(out["secondary_enroll"])    if "secondary_enroll"    in out.columns else np.nan
    return out


def score_with_weights(df: pd.DataFrame, weight_dict: dict) -> pd.DataFrame:
    """Compute a composite score (0-100) given a dict of score-key -> weight."""
    out = df.copy()
    total_w = sum(weight_dict.values())
    if total_w == 0:
        out["composite_score"] = np.nan
        out["rank"] = pd.NA
        return out

    accum = pd.Series(0.0, index=out.index)
    for key, w in weight_dict.items():
        if key in out.columns and w:
            accum += (w / total_w) * out[key].fillna(0)

    out["composite_score"] = minmax(accum) * 100
    out["rank"] = out["composite_score"].rank(ascending=False, method="min").astype("Int64")
    return out.sort_values("composite_score", ascending=False).reset_index(drop=True)


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    st.title("🍁 Canada International Student Source Dashboard")
    st.markdown(
        "Rank and explore countries by their potential as sources of international "
        "students to Canada, using live demographic and economic data from the "
        "[World Bank Open Data API](https://data.worldbank.org)."
    )

    with st.sidebar:
        st.header("Settings")

        year = st.selectbox(
            "Reference Year", options=list(range(2022, 2017, -1)), index=0,
            help="More recent years may have incomplete data for some indicators.",
        )
        region_filter = st.selectbox("Region Filter", REGIONS)
        top_n = st.slider("Top N countries to highlight", 10, 50, 20)

        st.divider()
        st.subheader("Program Focus")
        program = st.selectbox(
            "Select a program to auto-tune scoring weights",
            list(PROGRAM_PROFILES.keys()),
        )
        st.caption(
            "Weights below are pre-set for the selected program using public "
            "indicators that correlate with typical applicant profiles for that "
            "field of study. Adjust freely — this is a starting framework, not a "
            "validated model."
        )

        preset = PROGRAM_PROFILES[program]
        weight_dict = {}
        with st.expander("Adjust weights", expanded=False):
            for key in ALL_SCORE_KEYS:
                default = preset.get(key, 0)
                weight_dict[key] = st.slider(SCORE_LABELS[key], 0, 10, default, key=f"w_{key}")

        st.divider()
        st.subheader("York Applicant Data (optional)")
        st.caption(
            "Upload a CSV with columns: country, applicants, enrolled (optional). "
            "Once uploaded, applicant concentration and underpenetrated-market "
            "views will appear."
        )
        york_file = st.file_uploader("Upload York applicant CSV", type=["csv"])

        st.divider()
        st.caption("Data: World Bank Open Data · Built with Streamlit + Plotly")

    with st.spinner("Fetching live data from the World Bank API..."):
        raw = load_data(year)

    if raw is None or raw.empty:
        st.error(
            "No data could be loaded. Try selecting an earlier year such as "
            "2021 or 2020 from the sidebar."
        )
        st.stop()

    if region_filter != "All Regions":
        raw = raw[raw["region"].str.contains(region_filter, na=False)]
        if raw.empty:
            st.warning(f"No data found for region: {region_filter}")
            st.stop()

    sub_scored = build_sub_scores(raw)
    scored = score_with_weights(sub_scored, weight_dict)

    # ── Merge York data if provided ─────────────────────────────────────────
    york_df = pd.DataFrame()
    if york_file is not None:
        york_df = load_york_csv(york_file)
        if not york_df.empty:
            scored = pd.merge(
                scored, york_df, left_on="country", right_on="country", how="left"
            )
            unmatched = york_df[~york_df["country"].isin(scored["country"])]
            if not unmatched.empty:
                st.sidebar.warning(
                    f"{len(unmatched)} countries in your CSV did not match World "
                    f"Bank country names (e.g. {', '.join(unmatched['country'].head(3))}). "
                    "Check spelling against standard country names."
                )

    top = scored.dropna(subset=["composite_score"]).head(top_n)
    if top.empty:
        st.warning("Not enough scored data to display. Try a different year or weights.")
        st.stop()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Countries Analysed", f"{scored['composite_score'].notna().sum():,}")
    k2.metric("Top Country", top.iloc[0]["country"])
    k3.metric("Top Score", f"{top.iloc[0]['composite_score']:.1f}/100")
    k4.metric("Median Score", f"{scored['composite_score'].median():.1f}/100")
    k5.metric("Program Focus", program)

    st.divider()

    tabs = ["World Map", "Rankings", "Program Comparison", "Explorer", "Data"]
    if not york_df.empty:
        tabs.append("York Applicant Insights")
    tab_objs = st.tabs(tabs)
    tab_map, tab_rank, tab_compare, tab_explore, tab_data = tab_objs[:5]
    tab_york = tab_objs[5] if not york_df.empty else None

    # ── World Map ─────────────────────────────────────────────────────────────
    with tab_map:
        st.subheader(f"Student Source Potential — {program}")
        map_df = scored.dropna(subset=["composite_score"])
        fig_map = px.choropleth(
            map_df, locations="iso3", color="composite_score", hover_name="country",
            hover_data={"iso3": False, "composite_score": ":.1f", "rank": True},
            color_continuous_scale="RdYlGn", range_color=[0, 100],
            labels={"composite_score": "Score"},
        )
        fig_map.update_layout(
            geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
            margin=dict(l=0, r=0, t=10, b=0), height=500,
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # ── Rankings ──────────────────────────────────────────────────────────────
    with tab_rank:
        col_a, col_b = st.columns([1, 1], gap="large")
        with col_a:
            st.subheader(f"Top {top_n} Countries — {program}")
            fig_bar = px.bar(
                top.iloc[::-1], x="composite_score", y="country", orientation="h",
                color="composite_score", color_continuous_scale="RdYlGn", range_color=[0, 100],
                labels={"composite_score": "Score", "country": ""},
            )
            fig_bar.update_layout(
                height=max(420, top_n * 22), coloraxis_showscale=False,
                margin=dict(l=0, r=10, t=10, b=0),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            st.subheader("Score Factor Breakdown — Top 15")
            top15 = top.head(15).copy()
            active_keys = [k for k, w in weight_dict.items() if w > 0]
            total_w = sum(weight_dict.values())
            for k in active_keys:
                top15[k + "_w"] = top15[k].fillna(0) * (weight_dict[k] / total_w)

            breakdown = top15[["country"] + [k + "_w" for k in active_keys]].melt(
                id_vars="country", var_name="factor", value_name="value"
            )
            breakdown["factor"] = breakdown["factor"].str.replace("_w", "").map(SCORE_LABELS)
            fig_stack = px.bar(
                breakdown, x="value", y="country", color="factor", orientation="h",
                barmode="stack", labels={"value": "Weighted Score", "country": "", "factor": "Factor"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_stack.update_layout(
                height=450, margin=dict(l=0, r=10, t=10, b=0),
                yaxis=dict(categoryorder="total ascending"),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_stack, use_container_width=True)

    # ── Program Comparison ───────────────────────────────────────────────────
    with tab_compare:
        st.subheader("Compare Rankings Across Programs")
        chosen_programs = st.multiselect(
            "Select programs to compare",
            list(PROGRAM_PROFILES.keys()),
            default=["Engineering & Computer Science", "Business & MBA", "Health Sciences"],
        )
        compare_n = st.slider("Countries per program", 5, 20, 10, key="compare_n")

        if chosen_programs:
            frames = []
            for prog in chosen_programs:
                prog_scored = score_with_weights(sub_scored, PROGRAM_PROFILES[prog])
                prog_top = prog_scored.dropna(subset=["composite_score"]).head(compare_n)
                prog_top = prog_top[["country", "composite_score", "rank"]].copy()
                prog_top["program"] = prog
                frames.append(prog_top)

            compare_df = pd.concat(frames, ignore_index=True)
            fig_cmp = px.bar(
                compare_df, x="composite_score", y="country", color="program",
                orientation="h", barmode="group",
                labels={"composite_score": "Score", "country": "", "program": "Program"},
            )
            fig_cmp.update_layout(
                height=max(500, compare_n * 30), margin=dict(l=0, r=10, t=10, b=0),
                legend=dict(orientation="h", y=-0.1),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            st.caption(
                "Each program uses a different weighting of public World Bank "
                "indicators (see sidebar Program Focus for details). This is a "
                "directional framework — refine weights based on institutional "
                "knowledge of what drives applicants to each program."
            )
        else:
            st.info("Select at least one program above to compare.")

    # ── Explorer ──────────────────────────────────────────────────────────────
    with tab_explore:
        st.subheader("Indicator Explorer")
        exp_cols = [k for k in INDICATOR_LABELS if k in scored.columns]
        col1, col2 = st.columns(2)
        x_key = col1.selectbox("X Axis", exp_cols, index=min(1, len(exp_cols) - 1),
                               format_func=lambda k: INDICATOR_LABELS[k])
        y_key = col2.selectbox("Y Axis", exp_cols, index=min(2, len(exp_cols) - 1),
                               format_func=lambda k: INDICATOR_LABELS[k])

        valid = scored.dropna(subset=["composite_score", x_key, y_key])
        fig_scatter = px.scatter(
            valid, x=x_key, y=y_key, size="composite_score", color="region",
            hover_name="country",
            hover_data={"composite_score": ":.1f", "rank": True, "region": False},
            labels={x_key: INDICATOR_LABELS.get(x_key, x_key),
                    y_key: INDICATOR_LABELS.get(y_key, y_key), "composite_score": "Score"},
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
            m1, m2, m3 = st.columns(3)
            m1.metric("Composite Score", f"{r['composite_score']:.1f}/100" if pd.notna(r.get("composite_score")) else "N/A")
            m2.metric("Global Rank", f"#{int(r['rank'])}" if pd.notna(r.get("rank")) else "N/A")
            m3.metric("GDP per Capita", f"${r['gdp_per_capita']:,.0f}" if pd.notna(r.get("gdp_per_capita")) else "N/A")

            active_keys = [k for k, w in weight_dict.items() if w > 0]
            labels = [SCORE_LABELS[k] for k in active_keys]
            raw_vals = [float(r.get(k, 0) or 0) for k in active_keys]
            if raw_vals:
                vals = raw_vals + [raw_vals[0]]
                cats = labels + [labels[0]]
                fig_radar = go.Figure(go.Scatterpolar(
                    r=vals, theta=cats, fill="toself",
                    line_color="#D62728", fillcolor="rgba(214, 39, 40, 0.15)", name=selected,
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False, title=f"{selected} — Factor Profile ({program})", height=420,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

    # ── Data Table ────────────────────────────────────────────────────────────
    with tab_data:
        st.subheader("Full Scored Dataset")
        base_cols = ["rank", "country", "region", "composite_score",
                     "gdp_per_capita", "total_population", "tertiary_enrollment",
                     "youth_unemployment", "internet_users"]
        if not york_df.empty:
            base_cols += [c for c in ["applicants", "enrolled", "acceptance_rate"] if c in scored.columns]
        cols_available = [c for c in base_cols if c in scored.columns]

        display = scored[cols_available].dropna(subset=["composite_score"]).sort_values(
            "rank" if "rank" in cols_available else "composite_score"
        )
        st.dataframe(display, use_container_width=True, height=520)

        csv = display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download as CSV", data=csv,
            file_name=f"canada_student_sources_{year}_{program.replace(' ', '_')}.csv",
            mime="text/csv",
        )

    # ── York Applicant Insights (only if uploaded) ──────────────────────────
    if tab_york is not None:
        with tab_york:
            st.subheader("York Applicant Data vs. World Bank Potential Score")
            merged = scored.dropna(subset=["applicants"]).copy()

            if merged.empty:
                st.warning(
                    "No York records matched World Bank country names. Check that "
                    "country names in your CSV match standard names (e.g. "
                    "'United States' not 'USA')."
                )
            else:
                c1, c2 = st.columns(2)
                c1.metric("Total Applicants (matched)", f"{merged['applicants'].sum():,.0f}")
                if "enrolled" in merged.columns:
                    c2.metric("Total Enrolled (matched)", f"{merged['enrolled'].sum():,.0f}")

                st.markdown("#### Underpenetrated Markets")
                st.caption(
                    "Countries with a HIGH potential score but LOW current applicant "
                    "volume — candidates for targeted recruitment."
                )
                merged["applicant_rank"] = merged["applicants"].rank(ascending=False)
                merged["potential_rank"] = merged["composite_score"].rank(ascending=False)
                merged["gap"] = merged["applicant_rank"] - merged["potential_rank"]
                underpenetrated = merged.sort_values("gap", ascending=False).head(15)

                fig_gap = px.bar(
                    underpenetrated.sort_values("gap"),
                    x="gap", y="country", orientation="h",
                    color="composite_score", color_continuous_scale="RdYlGn",
                    labels={"gap": "Untapped Potential (rank gap)", "country": ""},
                )
                fig_gap.update_layout(height=450, margin=dict(l=0, r=10, t=10, b=0))
                st.plotly_chart(fig_gap, use_container_width=True)

                st.markdown("#### Applicants vs. Composite Score")
                fig_scatter2 = px.scatter(
                    merged, x="composite_score", y="applicants",
                    size="applicants", color="region", hover_name="country",
                    labels={"composite_score": "Potential Score", "applicants": "Current Applicants"},
                    size_max=40,
                )
                fig_scatter2.update_layout(height=480)
                st.plotly_chart(fig_scatter2, use_container_width=True)

    st.divider()
    st.caption(
        f"Data: World Bank Open Data API · Reference year: {year} · Program focus: {program} · "
        "Composite score is a weighted index of normalised public indicators. "
        "Program weightings are a directional starting framework, not a validated model."
    )


if __name__ == "__main__":
    main()

