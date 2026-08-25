from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="ScoutVision | Recruitment intelligence",
    page_icon=":material/sports_soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
)


POSITION_METRICS = {
    "Defender": [
        "Tackles per 90",
        "Tackle Success Rate (%)",
        "Interceptions per 90",
        "Clearances per 90",
        "Blocks per 90",
        "Accurate Passes per 90",
        "Pass Success (%)",
        "Accurate Long Balls per 90",
        "Successful Long Balls (%)",
    ],
    "Midfielder": [
        "Tackles per 90",
        "Interceptions per 90",
        "Possessions Won Midfield per 90",
        "Accurate Passes per 90",
        "Pass Success (%)",
        "Accurate Long Balls per 90",
        "Chances Created per 90",
        "Expected Assists per 90",
        "Actual Assists per 90",
        "Successful Dribbles per 90",
    ],
    "Forward": [
        "Goals per 90",
        "Expected Goals per 90",
        "Shots per 90",
        "Shots on Target per 90",
        "Shot Accuracy (%)",
        "Shot Conversion Rate (%)",
        "Expected Assists per 90",
        "Actual Assists per 90",
        "Chances Created per 90",
        "Successful Dribbles per 90",
    ],
}

ROLE_SCORES = {
    "Defender": ["Defensive Defender Score", "Ball Playing Defender Score"],
    "Midfielder": ["Ball Winning Midfielder Score", "Creative Midfielder Score"],
    "Forward": ["Goal Scorer Score", "Creative Forward Score"],
}

NAVIGATION = {
    "Command center": ":material/space_dashboard:",
    "Player finder": ":material/person_search:",
    "Compare": ":material/compare_arrows:",
    "Recruitment lab": ":material/biotech:",
    "Shortlist": ":material/bookmark:",
}


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    data_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "processed"
        / "premier_league_2023_24_scouting_final.csv"
    )
    data = pd.read_csv(data_path)
    numeric_columns = [column for column in data.columns if column not in {"Player", "Team", "Country", "Pos", "Position_Group"}]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return data


try:
    df = load_data()
except FileNotFoundError:
    st.error("The processed scouting dataset could not be found in `data/processed`.")
    st.stop()
except Exception as exc:
    st.error(f"The scouting dataset could not be loaded: {exc}")
    st.stop()


if "shortlist" not in st.session_state:
    st.session_state.shortlist = []


def available(columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def as_number(value, decimals: int = 1, suffix: str = "") -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):,.{decimals}f}{suffix}"


def as_int(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{int(float(value)):,}"


def get_player(name: str) -> pd.Series | None:
    match = df.loc[df["Player"] == name]
    return None if match.empty else match.iloc[0]


def metrics_for(position: str) -> list[str]:
    return available(POSITION_METRICS.get(position, []))


def percentile_table(name: str) -> pd.DataFrame:
    player = get_player(name)
    if player is None:
        return pd.DataFrame(columns=["Metric", "Value", "Percentile"])
    records = []
    for metric in metrics_for(str(player.get("Position_Group", ""))):
        percentile_column = f"{metric} Percentile"
        if percentile_column in df.columns and pd.notna(player.get(percentile_column)):
            records.append(
                {
                    "Metric": metric,
                    "Value": player.get(metric),
                    "Percentile": float(player.get(percentile_column)),
                }
            )
    return pd.DataFrame(records)


def similarity_pool(name: str) -> pd.DataFrame:
    target = get_player(name)
    if target is None:
        return pd.DataFrame()
    position = str(target.get("Position_Group", ""))
    metrics = metrics_for(position)
    if len(metrics) < 3:
        return pd.DataFrame()

    candidates = df.loc[df["Position_Group"] == position].copy()
    coverage = candidates[metrics].notna().sum(axis=1)
    candidates = candidates.loc[coverage >= int(np.ceil(len(metrics) * 0.7))].copy()
    if name not in candidates["Player"].values:
        return pd.DataFrame()

    values = candidates[metrics].apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median(numeric_only=True)).fillna(0)
    scaled = StandardScaler().fit_transform(values)
    target_index = candidates["Player"].tolist().index(name)
    candidates["Similarity"] = cosine_similarity(scaled[target_index].reshape(1, -1), scaled)[0] * 100
    return candidates


def similar_players(
    name: str,
    max_age: int | None = None,
    min_minutes: int = 0,
    min_similarity: int = 0,
    exclude_club: bool = False,
    limit: int = 10,
) -> pd.DataFrame:
    target = get_player(name)
    pool = similarity_pool(name)
    if target is None or pool.empty:
        return pd.DataFrame()
    result = pool.loc[pool["Player"] != name].copy()
    if exclude_club and "Team" in result:
        result = result.loc[result["Team"] != target.get("Team")]
    if max_age is not None and "Age" in result:
        result = result.loc[result["Age"] <= max_age]
    if "Minutes" in result:
        result = result.loc[result["Minutes"] >= min_minutes]
    result = result.loc[result["Similarity"] >= min_similarity]
    columns = available(["Player", "Team", "Position_Group", "Age", "Minutes", "Scouting_Score", "Similarity"])
    return result.sort_values("Similarity", ascending=False)[columns].head(limit).reset_index(drop=True)


def shortlist_add(name: str) -> None:
    if name not in st.session_state.shortlist:
        st.session_state.shortlist.append(name)


def shortlist_remove(name: str) -> None:
    if name in st.session_state.shortlist:
        st.session_state.shortlist.remove(name)


def default_player(players: list[str], preferred: str) -> int:
    return players.index(preferred) if preferred in players else 0


def chart_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=30, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE7E2"),
        hoverlabel=dict(bgcolor="#13231D", font_color="#FFFFFF"),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)", zeroline=False)
    return fig


def radar_chart(first_name: str, second_name: str | None = None) -> go.Figure | None:
    first = get_player(first_name)
    if first is None:
        return None
    metrics = metrics_for(str(first.get("Position_Group", "")))
    pairs = [(metric, f"{metric} Percentile") for metric in metrics if f"{metric} Percentile" in df.columns]
    if len(pairs) < 3:
        return None

    labels = [
        metric.replace(" per 90", "").replace("Expected Goals", "xG").replace("Expected Assists", "xA")
        for metric, _ in pairs
    ]
    values = [float(first.get(column)) if pd.notna(first.get(column)) else 50 for _, column in pairs]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            name=first_name,
            line_color="#39D98A",
            fillcolor="rgba(57,217,138,.22)",
        )
    )
    if second_name:
        second = get_player(second_name)
        if second is not None and second.get("Position_Group") == first.get("Position_Group"):
            second_values = [float(second.get(column)) if pd.notna(second.get(column)) else 50 for _, column in pairs]
            fig.add_trace(
                go.Scatterpolar(
                    r=second_values,
                    theta=labels,
                    fill="toself",
                    name=second_name,
                    line_color="#60A5FA",
                    fillcolor="rgba(96,165,250,.16)",
                )
            )
    fig.update_layout(
        height=480,
        margin=dict(l=45, r=45, t=35, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCE7E2"),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], tickvals=[20, 40, 60, 80, 100], gridcolor="rgba(148,163,184,.18)"),
            angularaxis=dict(gridcolor="rgba(148,163,184,.14)"),
        ),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08),
    )
    return fig


def player_header(player: pd.Series, key: str) -> None:
    top = st.container(horizontal=True, vertical_alignment="center")
    with top:
        with st.container():
            st.subheader(str(player["Player"]))
            st.caption(
                f"{player.get('Team', 'Unknown club')} · {player.get('Position_Group', 'Unknown role')} · "
                f"{player.get('Country', '—')}"
            )
        if player["Player"] in st.session_state.shortlist:
            if st.button("Shortlisted", icon=":material/bookmark_added:", key=f"remove_{key}", type="tertiary"):
                shortlist_remove(str(player["Player"]))
                st.rerun()
        elif st.button("Add to shortlist", icon=":material/bookmark_add:", key=f"add_{key}", type="primary"):
            shortlist_add(str(player["Player"]))
            st.rerun()


def player_snapshot(player: pd.Series, prefix: str) -> None:
    metrics = st.container(horizontal=True)
    metrics.metric("Scout score", as_number(player.get("Scouting_Score")), border=True)
    metrics.metric("Age", as_int(player.get("Age")), border=True)
    metrics.metric("Minutes", as_int(player.get("Minutes")), border=True)
    metrics.metric("Matches", as_int(player.get("Matches")), border=True)
    role_columns = [column for column in ROLE_SCORES.get(str(player.get("Position_Group", "")), []) if column in player.index]
    if role_columns:
        roles = st.container(horizontal=True)
        for role in role_columns:
            roles.metric(role.replace(" Score", ""), as_number(player.get(role)), border=True)


def player_detail(name: str, key: str) -> None:
    player = get_player(name)
    if player is None:
        return
    player_header(player, key)
    player_snapshot(player, key)
    overview, profile, report = st.tabs(["Overview", "Percentile profile", "Scout note"])
    with overview:
        columns = available(
            [
                "Goals per 90",
                "Expected Goals per 90",
                "Actual Assists per 90",
                "Expected Assists per 90",
                "Chances Created per 90",
                "Tackles per 90",
                "Interceptions per 90",
                "Pass Success (%)",
                "FotMob Rating",
            ]
        )
        rows = pd.DataFrame({"Metric": columns, "Value": [player.get(column) for column in columns]}).dropna()
        st.dataframe(rows, hide_index=True, width="stretch")
    with profile:
        radar = radar_chart(name)
        if radar:
            st.plotly_chart(radar, width="stretch", config={"displayModeBar": False})
        else:
            st.info("There is not enough percentile data to draw this profile.")
    with report:
        percentiles = percentile_table(name).sort_values("Percentile", ascending=False)
        if percentiles.empty:
            st.info("There is not enough percentile data to generate a report.")
        else:
            strengths = percentiles.head(3)
            risks = percentiles.loc[percentiles["Percentile"] < 45].tail(3).sort_values("Percentile")
            st.markdown(
                f"**Profile:** {player.get('Position_Group', 'Player')} with a scout score of "
                f"**{as_number(player.get('Scouting_Score'))}** across the available position-relative metrics."
            )
            st.write("**Standout qualities**")
            for row in strengths.itertuples():
                st.markdown(f"- {row.Metric} — {row.Percentile:.0f}th percentile")
            st.write("**Watch points**")
            if risks.empty:
                st.markdown("- No metric falls below the 45th percentile in the available profile.")
            else:
                for row in risks.itertuples():
                    st.markdown(f"- {row.Metric} — {row.Percentile:.0f}th percentile")
            st.caption("Use this statistical note alongside video, tactical, medical and financial assessment.")


def player_table(data: pd.DataFrame, key: str):
    display = data[available(["Player", "Team", "Position_Group", "Age", "Minutes", "Scouting_Score", "Similarity"])]
    config = {
        "Player": st.column_config.TextColumn("Player", pinned=True),
        "Age": st.column_config.NumberColumn("Age", format="%d"),
        "Minutes": st.column_config.NumberColumn("Minutes", format="%d"),
        "Scouting_Score": st.column_config.ProgressColumn("Scout score", min_value=0, max_value=100, format="%.1f"),
        "Similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=100, format="%.1f%%"),
    }
    return st.dataframe(
        display,
        key=key,
        hide_index=True,
        width="stretch",
        height=min(520, 42 + len(display) * 35),
        column_config=config,
        on_select="rerun",
        selection_mode="single-row",
    )


with st.sidebar:
    st.title("ScoutVision")
    st.caption("Recruitment intelligence · Premier League 2023/24")
    st.success("Analytics engine online", icon=":material/check_circle:")
    st.divider()
    page = st.radio(
        "Workspace",
        list(NAVIGATION),
        format_func=lambda item: f"{NAVIGATION[item]}  {item}",
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("DATASET")
    st.metric("Players", f"{df['Player'].nunique():,}")
    summary = st.container(horizontal=True)
    summary.metric("Clubs", f"{df['Team'].nunique():,}")
    summary.metric("Shortlist", len(st.session_state.shortlist))
    st.caption("Scores are position-relative and intended to support, not replace, live scouting.")


st.title(page)
st.caption("Turn performance data into faster, more confident recruitment decisions.")


if page == "Command center":
    minimum_minutes = st.segmented_control(
        "Sample threshold",
        options=[0, 450, 900, 1800],
        default=900,
        format_func=lambda value: "All players" if value == 0 else f"{value:,}+ minutes",
    )
    qualified = df.loc[df["Minutes"].fillna(0) >= (minimum_minutes or 0)].copy()
    stats = st.container(horizontal=True)
    stats.metric("Qualified players", f"{len(qualified):,}", border=True)
    stats.metric("Clubs covered", f"{qualified['Team'].nunique():,}", border=True)
    stats.metric("Median age", as_number(qualified["Age"].median()), border=True)
    stats.metric("Median scout score", as_number(qualified["Scouting_Score"].median()), border=True)

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Top performers")
            leaders = qualified.dropna(subset=["Scouting_Score"]).nlargest(12, "Scouting_Score").sort_values("Scouting_Score")
            fig = px.bar(
                leaders,
                x="Scouting_Score",
                y="Player",
                color="Position_Group",
                orientation="h",
                hover_data=available(["Team", "Age", "Minutes"]),
                labels={"Scouting_Score": "Scout score", "Player": ""},
            )
            st.plotly_chart(chart_layout(fig), width="stretch", config={"displayModeBar": False})
    with right:
        with st.container(border=True):
            st.subheader("Age versus performance")
            fig = px.scatter(
                qualified.dropna(subset=["Age", "Scouting_Score"]),
                x="Age",
                y="Scouting_Score",
                color="Position_Group",
                size="Minutes",
                size_max=18,
                hover_name="Player",
                hover_data=available(["Team", "Minutes"]),
                labels={"Scouting_Score": "Scout score"},
            )
            st.plotly_chart(chart_layout(fig), width="stretch", config={"displayModeBar": False})

    st.subheader("Role leaders")
    role_tabs = st.tabs(list(ROLE_SCORES))
    for tab, position in zip(role_tabs, ROLE_SCORES):
        with tab:
            position_data = qualified.loc[qualified["Position_Group"] == position]
            role_cols = available(ROLE_SCORES[position])
            if role_cols:
                role_choice = st.segmented_control(
                    "Role",
                    role_cols,
                    default=role_cols[0],
                    format_func=lambda value: value.replace(" Score", ""),
                    key=f"role_{position}",
                )
                leaders = position_data.dropna(subset=[role_choice]).nlargest(10, role_choice)
                event = player_table(leaders, f"leaderboard_{position}")
                if event.selection.rows:
                    with st.expander("Selected player", expanded=True):
                        player_detail(str(leaders.iloc[event.selection.rows[0]]["Player"]), f"leader_{position}")


elif page == "Player finder":
    st.write("Build a target list, select a row, and inspect the player without leaving the page.")
    with st.form("finder_filters", border=True):
        row = st.container(horizontal=True)
        positions = row.multiselect("Positions", sorted(df["Position_Group"].dropna().unique()), placeholder="All positions")
        clubs = row.multiselect("Clubs", sorted(df["Team"].dropna().unique()), placeholder="All clubs")
        age_range = row.slider("Age", int(df["Age"].min()), int(df["Age"].max()), (18, 30))
        minutes = row.number_input("Minimum minutes", 0, int(df["Minutes"].max()), 900, 100)
        score = row.slider("Minimum scout score", 0, 100, 50)
        submitted = st.form_submit_button("Apply filters", icon=":material/filter_alt:", type="primary")

    filtered = df.copy()
    if positions:
        filtered = filtered.loc[filtered["Position_Group"].isin(positions)]
    if clubs:
        filtered = filtered.loc[filtered["Team"].isin(clubs)]
    filtered = filtered.loc[
        filtered["Age"].between(*age_range)
        & (filtered["Minutes"] >= minutes)
        & (filtered["Scouting_Score"] >= score)
    ].sort_values("Scouting_Score", ascending=False)

    st.subheader(f"{len(filtered):,} matching players")
    if filtered.empty:
        st.info("No players match these filters. Widen the age range or lower the minutes and score thresholds.")
    else:
        event = player_table(filtered, "finder_results")
        selected_index = event.selection.rows[0] if event.selection.rows else 0
        with st.expander("Player dossier", expanded=bool(event.selection.rows)):
            player_detail(str(filtered.iloc[selected_index]["Player"]), "finder")


elif page == "Compare":
    all_players = sorted(df["Player"].dropna().unique().tolist())
    selectors = st.container(horizontal=True)
    player_a = selectors.selectbox("First player", all_players, index=default_player(all_players, "Rodri"))
    row_a = get_player(player_a)
    peer_names = sorted(
        df.loc[(df["Position_Group"] == row_a.get("Position_Group")) & (df["Player"] != player_a), "Player"].dropna().unique().tolist()
    )
    player_b = selectors.selectbox("Positional peer", peer_names, index=0)
    row_b = get_player(player_b)

    first, second = st.columns(2, gap="large")
    with first:
        with st.container(border=True):
            player_header(row_a, "compare_a")
            player_snapshot(row_a, "compare_a")
    with second:
        with st.container(border=True):
            player_header(row_b, "compare_b")
            player_snapshot(row_b, "compare_b")

    radar = radar_chart(player_a, player_b)
    if radar:
        with st.container(border=True):
            st.subheader("Position-relative profile")
            st.plotly_chart(radar, width="stretch", config={"displayModeBar": False})

    metrics = metrics_for(str(row_a.get("Position_Group", "")))
    comparison = pd.DataFrame(
        {
            "Metric": metrics,
            player_a: [row_a.get(metric) for metric in metrics],
            player_b: [row_b.get(metric) for metric in metrics],
        }
    )
    comparison["Leader"] = comparison.apply(
        lambda row: player_a if row[player_a] > row[player_b] else player_b if row[player_b] > row[player_a] else "Level",
        axis=1,
    )
    st.subheader("Metric breakdown")
    st.dataframe(comparison, hide_index=True, width="stretch")


elif page == "Recruitment lab":
    st.write("Use a known player as the profile template, then tighten the recruitment constraints.")
    outfield = df.loc[df["Position_Group"].isin(POSITION_METRICS), "Player"].dropna().sort_values().unique().tolist()
    with st.form("recruitment_constraints", border=True):
        controls = st.container(horizontal=True)
        reference = controls.selectbox("Reference player", outfield, index=default_player(outfield, "Erling Haaland"))
        max_age = controls.slider("Maximum age", 18, 40, 27)
        min_minutes = controls.number_input("Minimum minutes", 0, int(df["Minutes"].max()), 900, 100)
        min_similarity = controls.slider("Minimum similarity", 0, 100, 55)
        result_count = controls.select_slider("Results", options=[5, 10, 15, 20], value=10)
        search = st.form_submit_button("Find alternatives", icon=":material/search:", type="primary")

    reference_player = get_player(reference)
    st.caption(
        f"Matching the {reference_player.get('Position_Group', 'player')} profile of {reference} · "
        f"{reference_player.get('Team', '')}"
    )
    alternatives = similar_players(reference, max_age, min_minutes, min_similarity, True, result_count)
    if alternatives.empty:
        st.info("No alternatives match every constraint. Lower the similarity or minutes threshold, or raise the age limit.")
    else:
        event = player_table(alternatives, "alternative_results")
        selected_index = event.selection.rows[0] if event.selection.rows else 0
        chosen = str(alternatives.iloc[selected_index]["Player"])
        detail, comparison_tab = st.tabs(["Candidate dossier", "Compare with reference"])
        with detail:
            player_detail(chosen, "alternative")
        with comparison_tab:
            radar = radar_chart(reference, chosen)
            if radar:
                st.plotly_chart(radar, width="stretch", config={"displayModeBar": False})


elif page == "Shortlist":
    if not st.session_state.shortlist:
        st.info("Your shortlist is empty. Add candidates from the player finder, comparison, or recruitment lab.", icon=":material/bookmark:")
    else:
        shortlist = (
            df.loc[df["Player"].isin(st.session_state.shortlist)]
            .sort_values("Scouting_Score", ascending=False)
            .reset_index(drop=True)
        )
        st.write(f"{len(shortlist)} saved candidate{'s' if len(shortlist) != 1 else ''}")
        event = player_table(shortlist, "shortlist_table")
        selected_index = event.selection.rows[0] if event.selection.rows else 0
        selected_name = str(shortlist.iloc[selected_index]["Player"])
        actions = st.container(horizontal=True)
        if actions.button("Remove selected", icon=":material/bookmark_remove:"):
            shortlist_remove(selected_name)
            st.rerun()
        csv = shortlist[available(["Player", "Team", "Position_Group", "Age", "Minutes", "Scouting_Score"])].to_csv(index=False)
        actions.download_button(
            "Export shortlist",
            csv,
            "scoutvision_shortlist.csv",
            "text/csv",
            icon=":material/download:",
        )
        with st.expander("Selected player", expanded=bool(event.selection.rows)):
            player_detail(selected_name, "shortlist")


st.divider()
st.caption("ScoutVision · Statistical recruitment intelligence · Premier League 2023/24")
