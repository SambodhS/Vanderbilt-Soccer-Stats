# streamlit run ~/Downloads/Vanderbilt\ Soccer/Streamlit.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import yaml
from pymongo import MongoClient
import numpy as np
from streamlit import empty

st.set_page_config(layout="wide", page_icon="vanderbilt_logo.svg", page_title="Vanderbilt Analytics")
# st.sidebar.image("vanderbilt_logo.svg", width=75)
MONGO_URI = st.secrets["MONGODB_URI"]

def load_data():
    client = MongoClient(MONGO_URI)
    db = client["data"]
    coll = db["SEC"]
    data = list(coll.find())
    for d in data:
        d.pop("_id", None)
    df = pd.DataFrame(data)
    df['player_name'] = df['player_name'].str.replace('Player stats ', '', regex=False).str.strip()
    return df

@st.cache_data
def load_config():
    # Path to config.yaml in the same folder as the app
    config_path = Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        st.error(f"⚠️ config.yaml not found at {config_path}")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f)

def calculate_percentile(match_df, season_df):
    key_cols = ("team", "player_name")
    grp = match_df.groupby(list(key_cols)).mean().reset_index()
    stat_cols = [c for c in grp.columns if c not in key_cols]
    out = grp[list(key_cols)].copy()

    n_players = len(grp)
    for c in stat_cols:
        ref = np.sort(season_df[c].dropna().values)
        if ref.size == 0:
            out[c] = np.nan
            continue

        vals = grp[c].values
        pct = np.full(n_players, np.nan, dtype=float)
        valid = ~pd.isna(vals)
        if valid.any():
            counts = np.searchsorted(ref, vals[valid], side="right")  # counts <= value
            pct[valid] = counts / ref.size * 100.0
        out[c] = np.round(pct, 2)

    return out

def make_positional_subsets(df: pd.DataFrame):
    return {
        "Wingers": df[df["position"].str.contains("LW|RW", na=False)],
        "Forwards": df[df["position"].str.contains("CF|LWF|RWF", na=False)],
        "Attacking Midfielders": df[df["position"].str.contains("AMF", na=False)],
        "Central Midfielders": df[df["position"].str.contains("CMF", na=False)],
        "Defensive Midfielders": df[df["position"].str.contains("DMF", na=False)],
        "Outside Backs": df[df["position"].str.contains("LB|RB|LWB|RWB", na=False)],
        "Center Backs": df[df["position"].str.contains("CB|LCB|LCB3|RCB|RCB3", na=False)],
        "Goalkeepers": df[df["position"].str.contains("GK", na=False)],
    }

def plot_radar(players, position, match_df, match_df2, season_df, config, selected_metric):
    cfg = config.get("columns_config", {}).get(position, {})
    # Build cols and labels in matching order
    col_map = cfg.get("column_names", {})  # expected: {col_name: label, ...}
    cols = list(col_map.keys())
    labels = [col_map[c] for c in cols]

    # Pair them so we can filter together
    col_label_pairs = [(c, l) for c, l in zip(cols, labels)]

    # Only keep columns that exist in season_df (we need season as percentile baseline)
    available_pairs = [(c, l) for c, l in col_label_pairs if c in season_df.columns]
    if not available_pairs:
        fig = go.Figure()
        fig.update_layout(
            title="No metric columns available for this position",
            margin=dict(t=40, b=40, l=40, r=40)
        )
        return fig

    # Unzip back
    cols_filtered, labels_filtered = zip(*available_pairs)  # tuples

    fig = go.Figure()

    if selected_metric == "Percentile":
        # --- determine selected year and ref_df as before (your existing logic) ---
        selected_year = None
        if "year" in season_df.columns and not season_df["year"].dropna().empty:
            try:
                selected_year = int(pd.to_numeric(season_df["year"].dropna().unique()[0]))
            except Exception:
                selected_year = None

        ref_year = 2024 if selected_year == 2025 else selected_year

        ref_df = None
        if ref_year is not None:
            try:
                ref_df_candidate = data[data["year"] == ref_year].copy()
                if not ref_df_candidate.empty:
                    ref_df = ref_df_candidate
            except Exception:
                ref_df = None

        if ref_df is None:
            ref_df = season_df

        # --- NEW: only use the columns that actually exist in the reference dataframe ---
        ref_cols = [c for c in cols_filtered if c in ref_df.columns]

        # If ref_cols empty and ref_df was the "prior year", try falling back to season_df columns
        if not ref_cols and ref_df is not season_df:
            ref_cols = [c for c in cols_filtered if c in season_df.columns]
            if ref_cols:
                # if we fell back to season_df, also set ref_df to season_df so percentile uses season_df
                ref_df = season_df

        # If still empty, do not attempt to compute SEC avg (avoid KeyError). Create empty pct_season_df
        if not ref_cols:
            pct_season_df = pd.DataFrame(columns=["player_name"] + list(cols_filtered))
            # optionally add a light message in the figure title later (handled below)
            sec_trace_added = False
        else:
            # compute percentiles using only the ref_cols that are present in ref_df
            pct_season_df = calculate_percentile(ref_df[ref_cols + ["team", "player_name"]], ref_df)
            sec_trace_added = True

        # Add SEC Avg trace only if we computed it
        if sec_trace_added:
            # determine labels corresponding to ref_cols
            labels_for_ref = [l for c, l in zip(cols_filtered, labels_filtered) if c in ref_cols]
            sec_avg = pct_season_df[ref_cols].mean().round(2).values
            fig.add_trace(go.Scatterpolar(
                r=sec_avg,
                theta=labels_for_ref,
                fill="toself",
                name="SEC Avg (mean percentile)" if season_select == "2025" else "2024 SEC Avg (mean percentile)",
                opacity=0.7
            ))

            baseline_50 = [50.0] * len(ref_cols)
            fig.add_trace(go.Scatterpolar(
                r=baseline_50,
                theta=labels_for_ref,
                fill="toself",
                name="50th percentile",
                hoverinfo="none",
                opacity=0.15,
                showlegend=True
            ))
        else:
            # no SEC baseline available for this season/ref - optionally annotate
            fig.update_layout(title="SEC Avg not available for chosen season/reference (missing KPI columns)")

        # --- Player percentiles for match 1 — use only columns available in match_df (subset of filtered if missing)
        match1_cols = [c for c in cols_filtered if c in match_df.columns]
        if match1_cols:
            player_pct_df = calculate_percentile(match_df[match1_cols + ["team", "player_name"]], season_df)
        else:
            player_pct_df = pd.DataFrame(columns=["player_name"] + list(cols_filtered))

        # Player percentiles for match 2 (if provided)
        player_pct_df2 = None
        if match_df2 is not None and not match_df2.empty:
            match2_cols = [c for c in cols_filtered if c in match_df2.columns]
            if match2_cols:
                player_pct_df2 = calculate_percentile(match_df2[match2_cols + ["team", "player_name"]], season_df)
            else:
                player_pct_df2 = pd.DataFrame(columns=["player_name"] + list(cols_filtered))

    # helper to truncate long legend names
    def _truncate_label(text, max_len=25):
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    # Add traces for match 1 (use match1_cols; if a column missing it won't be plotted)
    for p in players:
        row = player_pct_df[player_pct_df["player_name"] == p] if not player_pct_df.empty else pd.DataFrame()
        if not row.empty:
            # ensure ordering of columns matches labels used — use match1_cols and corresponding labels
            plot_cols = [c for c in cols_filtered if c in match_df.columns]
            plot_labels = [l for c, l in zip(cols_filtered, labels_filtered) if c in match_df.columns]

            fig.add_trace(go.Scatterpolar(
                r=row[plot_cols].values.flatten(),
                theta=plot_labels,
                fill="toself",
                name=_truncate_label(f"{p} (All)" if match_select == "All" else f"{p} (Match 1)"),
                opacity=0.7
            ))

    # Add traces for match 2 (if available)
    if match_df2 is not None and player_pct_df2 is not None:
        for p in players:
            row2 = player_pct_df2[player_pct_df2["player_name"] == p] if not player_pct_df2.empty else pd.DataFrame()
            if not row2.empty:
                plot_cols2 = [c for c in cols_filtered if c in match_df2.columns]
                plot_labels2 = [l for c, l in zip(cols_filtered, labels_filtered) if c in match_df2.columns]

                fig.add_trace(go.Scatterpolar(
                    r=row2[plot_cols2].values.flatten(),
                    theta=plot_labels2,
                    fill="toself",
                    name=_truncate_label(f"{p} (Match 2)"),
                    opacity=0.5
                ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 100], showticklabels=False),
            angularaxis=dict(showticklabels=True)
        ),
        margin=dict(t=80, b=80, l=80, r=80)
    )
    return fig



#### MAIN CODE BELOW ####

data = load_data()
data.columns = data.columns.str.strip()
data.columns = data.columns.str.lower()
data.columns = data.columns.str.replace(" ", "_")
config = load_config()

st.sidebar.header("Filters")
season_select = st.sidebar.selectbox("Select Season", options=sorted(data["year"].unique()))
selected_metric = st.sidebar.selectbox("Select Metric", ["Percentile"])
match_select = st.sidebar.selectbox("Select Match", options=["All"] + sorted(data[(data["year"] == season_select) & data["match"].str.contains("Vanderbilt Commodores", na=False)]["match"].unique()))
compare = st.sidebar.checkbox("Compare two matches?")

season_df = data[data["year"] == season_select].copy()
if match_select != "All":
    match_df = season_df[season_df["match"] == match_select].copy()
else:
    match_df = season_df

if compare:
    second_match = [m for m in sorted(data[(data["year"] == season_select) & data["match"].str.contains("Vanderbilt Commodores", na=False)]["match"].unique()) if m != match_select]
    match_select_2 = st.sidebar.selectbox("Select Second Match", options=second_match, index=0, key="match2")
    match_df2 = season_df[season_df["match"] == match_select_2].copy()
else:
    match_df2 = None

for new_col, (num, den) in config.get("metrics", {}).items():
    if num in season_df.columns and den in season_df.columns:
        season_df[new_col] = (season_df[num] / season_df[den]) * 100
        match_df[new_col]  = (match_df[num] / match_df[den]) * 100
season_df = season_df.fillna(0)
match_df = match_df.fillna(0)

season_pos_subsets = make_positional_subsets(season_df)
match_pos_subsets = make_positional_subsets(match_df)
if compare:
    match_pos_subsets2 = make_positional_subsets(match_df2)


st.title("Vanderbilt Player Performance Charts")
positions = [("Forwards", "Wingers"), ("Attacking Midfielders", "Central Midfielders"), ("Defensive Midfielders", "Outside Backs"),
    ("Center Backs", "Goalkeepers")]

for row in positions:
    cols = st.columns(2)
    for col, pos in zip(cols, row):
        with col:
            st.header(pos)
            season_pos_df = season_pos_subsets[pos]
            match_pos_df = match_pos_subsets[pos]
            if compare:
                match_pos_df2 = match_pos_subsets2[pos]
            else:
                match_pos_df2 = None

            vandy_players = sorted(match_pos_df[match_pos_df["team"] == "Vanderbilt Commodores"]["player_name"].unique())
            opts = vandy_players
            key = f"multiselect_{pos.replace(' ', '_')}"
            sel = st.multiselect(f"Select {pos}", opts, default=None, key=key, placeholder="Choose a player")

            players = [p for p in sel]
            fig = plot_radar(players, pos, match_pos_df, match_pos_df2, season_pos_df, config, selected_metric)

            chart_key = f"radar_{season_select}_{match_select}_{selected_metric}_{pos.replace(' ','_')}"
            st.plotly_chart(fig, use_container_width=True, key=chart_key)

            if players:
                st.subheader("Raw stats")

                stats1_raw = match_pos_df[match_pos_df["player_name"].isin(players)].copy()
                if stats1_raw.empty:
                    st.info("No rows for selected player(s) in this match / position.")
                else:
                    stats1 = stats1_raw.drop(["competition", "home_team", "away_team", "year", "team"], axis=1, errors="ignore").copy()
                    stats1 = stats1.round(2)

                    if "match" in stats1.columns:
                        stats1.insert(1, "Match", stats1["match"])
                        stats1 = stats1.drop(columns=["match"])
                    else:
                        stats1.insert(1, "Match", match_select)

                    cols1 = list(stats1.columns)
                    if "player_name" in cols1:
                        cols1.insert(0, cols1.pop(cols1.index("player_name")))
                        stats1 = stats1[cols1]

                    pos_cfg_map = config.get("columns_config", {}).get(pos, {}).get("column_names", {})
                    pos_cfg_cols = list(pos_cfg_map.keys())

                    available_kpis = [c for c in pos_cfg_cols if c in stats1.columns]

                    if available_kpis:
                        display_cols = ["player_name", "Match"] + available_kpis
                        display_df = stats1[display_cols].copy()
                        rename_map = {c: pos_cfg_map[c] for c in available_kpis}
                        display_df = display_df.rename(columns=rename_map)
                        st.write(f"**Match 1**")
                        st.write(display_df)

                if match_pos_df2 is not None and not match_pos_df2.empty:
                    found_players = [p for p in players if p in match_pos_df2["player_name"].values]
                else:
                    found_players = []

                if compare:
                    if found_players:
                        stats2_raw = match_pos_df2[match_pos_df2["player_name"].isin(players)].copy()
                        if stats2_raw.empty:
                            st.info("No rows for selected player(s) in the second match / position.")
                        else:
                            stats2 = stats2_raw.drop(["competition", "home_team", "away_team", "year", "team"], axis=1, errors="ignore").copy()
                            stats2 = stats2.round(2)

                            if "match" in stats2.columns:
                                stats2.insert(1, "Match", stats2["match"])
                                stats2 = stats2.drop(columns=["match"])
                            else:
                                stats2.insert(1, "Match", match_select_2)

                            cols2 = list(stats2.columns)
                            if "player_name" in cols2:
                                cols2.insert(0, cols2.pop(cols2.index("player_name")))
                                stats2 = stats2[cols2]

                            available_kpis_2 = [c for c in pos_cfg_cols if c in stats2.columns]
                            if available_kpis_2:
                                display_cols2 = ["player_name", "Match"] + available_kpis_2
                                display_df2 = stats2[display_cols2].copy()
                                rename_map2 = {c: pos_cfg_map[c] for c in available_kpis_2}
                                display_df2 = display_df2.rename(columns=rename_map2)
                                st.write(f"**Match 2**")
                                st.write(display_df2)
                            else:
                                st.info("No configured KPI columns for this position exist in the dataframe for Match 2 — showing full raw stats instead.")
                                st.write(f"**Match 2**")
                                st.write(stats2)
                    else:
                        st.info("The player did not play in the second match")

            else:
                st.info("Select one or more players to see their raw stats here.")
