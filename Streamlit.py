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

@st.cache_data
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
        # Season percentiles (reference)
        pct_season_df = calculate_percentile(season_df[list(cols_filtered) + ["team", "player_name"]], season_df)

        sec_avg = pct_season_df[list(cols_filtered)].mean().round(2).values
        fig.add_trace(go.Scatterpolar(
            r=sec_avg,
            theta=list(labels_filtered),
            fill="toself",
            name="SEC Avg (mean percentile)",
            opacity=0.7
        ))

        baseline_50 = [50.0] * len(cols_filtered)
        fig.add_trace(go.Scatterpolar(
            r=baseline_50,
            theta=list(labels_filtered),
            fill="toself",
            name="50th percentile",
            hoverinfo="none",
            opacity=0.15,
            showlegend=True
        ))

        # Player percentiles for match 1 — use only columns available in match_df (subset of filtered if missing)
        match1_cols = [c for c in cols_filtered if c in match_df.columns]
        if match1_cols:
            player_pct_df = calculate_percentile(match_df[match1_cols + ["team", "player_name"]], season_df)
        else:
            # create an empty frame with player_name column to avoid breaking loops below
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

                # First match stats
                stats1 = match_pos_df[match_pos_df["player_name"].isin(players)].drop(
                    ["match", "competition", "home_team", "away_team", "year", "team"], axis=1
                )
                cols1 = list(stats1.columns)
                cols1.insert(0, cols1.pop(cols1.index("player_name")))
                stats1 = stats1[cols1]
                stats1[cols1] = stats1[cols1].round(2)
                stats1.insert(1, "Match", match_select)  # Add match name column

                st.write(f"**Match 1: {match_select}**")
                st.write(stats1)

                # Second match stats (if compare mode)
                if match_pos_df2 is not None and not match_pos_df2.empty:
                    found_players = [p for p in players if p in match_pos_df2["player_name"].values]
                else:
                    found_players = []  # no players found if match_pos_df2 doesn't exist

                if compare:
                    if found_players:
                        stats2 = match_pos_df2[match_pos_df2["player_name"].isin(players)].drop(["match", "competition", "home_team", "away_team", "year", "team"], axis=1)
                        cols2 = list(stats2.columns)
                        cols2.insert(0, cols2.pop(cols2.index("player_name")))
                        stats2 = stats2[cols2]
                        stats2[cols2] = stats2[cols2].round(2)
                        stats2.insert(1, "Match", match_select_2)  # Add match name column
                        st.write(f"**Match 2: {match_select_2}**")
                        st.write(stats2)
                    else:
                        st.info("The player did not play in the second match")

            else:
                st.info("Select one or more players to see their raw stats here.")
