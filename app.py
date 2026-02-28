import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Helper Functions
# ---------------------------------

def expand_position(pos):
    mapping = {
        "ST": "Striker",
        "CF": "Centre Forward",
        "LW": "Left Winger",
        "RW": "Right Winger",
        "LM": "Left Midfielder",
        "RM": "Right Midfielder",
        "CAM": "Attacking Midfielder",
        "CM": "Central Midfielder",
        "CDM": "Defensive Midfielder",
        "CB": "Centre Back",
        "LB": "Left Back",
        "RB": "Right Back",
        "LWB": "Left Wing Back",
        "RWB": "Right Wing Back",
        "GK": "Goalkeeper"
    }

    if pd.isna(pos):
        return pos

    pos = str(pos).replace("/", ",")
    positions = [p.strip() for p in pos.split(",")]
    expanded = [mapping.get(p, p) for p in positions]

    return ", ".join(expanded)


def position_coordinates_image(pos):
    mapping = {
        "GK": (8, 50),
        "CB": (22, 50),
        "LB": (22, 21),
        "RB": (22, 76.5),
        "LWB": (40, 32),
        "RWB": (40, 68),
        "CDM": (36, 50),
        "CM": (48, 50),
        "LM": (48, 21),
        "RM": (48, 76.5),
        "CAM": (61, 50),
        "LW": (75, 27),
        "RW": (75, 71),
        "CF": (70, 50),
        "ST": (61, 50)
    }
    return mapping.get(pos, (50, 50))


def get_player_skills(player):
    pace = (player["acceleration"] + player["sprint_speed"]) / 2
    shooting = (player["finishing"] + player["shot_power"] + player["long_shots"]) / 3
    passing = (player["short_passing"] + player["long_passing"] + player["vision"]) / 3
    dribbling = (player["dribbling"] + player["ball_control"] + player["agility"]) / 3
    defending = (player["standing_tackle"] + player["sliding_tackle"] + player["interceptions"]) / 3
    physical = (player["strength"] + player["stamina"] + player["balance"]) / 3

    return {
        "Pace": round(pace, 1),
        "Shooting": round(shooting, 1),
        "Passing": round(passing, 1),
        "Dribbling": round(dribbling, 1),
        "Defending": round(defending, 1),
        "Physical": round(physical, 1)
    }


# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="AI Football Scouting Dashboard",
    page_icon="⚽",
    layout="wide"
)

st.markdown("<h2 style='margin-top:0;'>⚽ AI Football Scouting & Talent Analysis</h2>",
            unsafe_allow_html=True)

st.markdown("""
<style>
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 0.8rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}
div[data-testid="stMetric"] {
    padding: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Load Dataset
# ---------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

@st.cache_resource
def load_knn_model():
    knn = joblib.load("similar_player_knn.joblib")
    scaler = joblib.load("similarity_scaler.joblib")
    features = joblib.load("similarity_features.joblib")
    return knn, scaler, features

knn_model, knn_scaler, knn_features = load_knn_model()


@st.cache_resource
def load_ml_model():
    model = joblib.load("future_overall_rf_model.joblib")
    features = joblib.load("model_features.joblib")
    return model, features

ml_model, ml_features = load_ml_model()

def build_player_features(player, feature_list):
    data = {}

    for feature in feature_list:
        if feature == "growth_gap":
            data[feature] = player["potential"] - player["overall_rating"]
        elif feature == "age_penalty":
            data[feature] = max(0, player["age"] - 27)
        else:
            data[feature] = player[feature]

    return pd.DataFrame([data])

@st.cache_resource
def load_archetype_model():
    kmeans = joblib.load("player_archetype_kmeans.joblib")
    scaler = joblib.load("archetype_scaler.joblib")
    features = joblib.load("archetype_features.joblib")
    return kmeans, scaler, features

archetype_model, archetype_scaler, archetype_features = load_archetype_model()

# -------- SAFE POSITION COLUMN HANDLING --------
possible_position_cols = [
    "Position",
    "Pos",
    "player_positions",
    "positions",
    "Position(s)"
]

position_col = None
for col in possible_position_cols:
    if col in df.columns:
        position_col = col
        break

if position_col:
    df["Position_Full"] = df[position_col].apply(expand_position)
else:
    df["Position_Full"] = "Unknown"
    
def build_similarity_features(player):
    return pd.DataFrame([{
        "Pace": (player["acceleration"] + player["sprint_speed"]) / 2,
        "Shooting": (player["finishing"] + player["shot_power"] + player["long_shots"]) / 3,
        "Passing": (player["short_passing"] + player["long_passing"] + player["vision"]) / 3,
        "Dribbling": (player["dribbling"] + player["ball_control"] + player["agility"]) / 3,
        "Defending": (player["standing_tackle"] + player["sliding_tackle"] + player["interceptions"]) / 3,
        "Physical": (player["strength"] + player["stamina"] + player["balance"]) / 3
    }])

# =====================================
# SIDEBAR (GLOBAL — ONLY ONCE)
# =====================================
st.sidebar.header("🔍 Player Selection")

selected_player_name = st.sidebar.selectbox(
    "Select Player",
    sorted(df["name"].unique())
)

selected_player = df[df["name"] == selected_player_name].iloc[0]
# =====================================
# SIDEBAR: SIMILAR PLAYERS (KNN)
# =====================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔁 Similar Players")

try:
    # Build feature vector for selected player
    sim_input = build_similarity_features(selected_player)
    sim_scaled = knn_scaler.transform(sim_input[knn_features])

    # Find nearest neighbors
    distances, indices = knn_model.kneighbors(sim_scaled)

    # Exclude the selected player itself
    similar_df = df.iloc[indices[0][1:]][["name", "overall_rating"]]

    # Display top 5 similar players
    for _, row in similar_df.iterrows():
        st.sidebar.markdown(
            f"• **{row['name']}** (OVR {int(row['overall_rating'])})"
        )

except Exception as e:
    st.sidebar.info("Similar players unavailable")



def get_prediction_confidence(model, X_row):
    """
    Estimate confidence using prediction variance across trees.
    Lower variance = higher confidence.
    """
    tree_preds = np.array([tree.predict(X_row)[0] for tree in model.estimators_])
    std_dev = np.std(tree_preds)

    if std_dev < 1.5:
        return "High"
    elif std_dev < 3.0:
        return "Medium"
    else:
        return "Low"
    
    
def build_archetype_features(player):
    return pd.DataFrame([{
        "Pace": (player["acceleration"] + player["sprint_speed"]) / 2,
        "Shooting": (player["finishing"] + player["shot_power"] + player["long_shots"]) / 3,
        "Passing": (player["short_passing"] + player["long_passing"] + player["vision"]) / 3,
        "Dribbling": (player["dribbling"] + player["ball_control"] + player["agility"]) / 3,
        "Defending": (player["standing_tackle"] + player["sliding_tackle"] + player["interceptions"]) / 3,
        "Physical": (player["strength"] + player["stamina"] + player["balance"]) / 3
    }])      

# ---------------------------------
# Tabs
# ---------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🧑 Player Profile",
    "🟩 Pitch View",
    "🆚 Compare Players",
    "🏆 Team Builder"
])
# =====================================================
# TAB 1: PLAYER PROFILE
# =====================================================
with tab1:
    player = selected_player
    # TOP ROW: PLAYER INFO (COMPACT)
    # ==============================
    c1, c2, c3, c4, c5 = st.columns(5)

    overall = int(player["overall_rating"])
    potential = int(player["potential"])
    growth = potential - overall

    # ✅ FIX: use expanded full position
    primary_pos_full = player["Position_Full"].split(",")[0].strip()

    c1.metric("Age", int(player["age"]))
    c5.metric("Position", primary_pos_full)
    c2.metric("Overall", f"{overall}/100")
    c3.metric("Potential", f"{potential}/100")
    c4.metric("Growth", f"+{growth}")
    # ==============================
    # DERIVED SKILLS
    # ==============================
    pace = (player["acceleration"] + player["sprint_speed"]) / 2
    shooting = (player["finishing"] + player["shot_power"] + player["long_shots"]) / 3
    passing = (player["short_passing"] + player["long_passing"] + player["vision"]) / 3
    dribbling = (player["dribbling"] + player["ball_control"] + player["agility"]) / 3
    defending = (player["standing_tackle"] + player["sliding_tackle"] + player["interceptions"]) / 3
    physical = (player["strength"] + player["stamina"] + player["balance"]) / 3

    skills = {
        "Pace": pace,
        "Shoot": shooting,
        "Pass": passing,
        "Dribble": dribbling,
        "Defend": defending,
        "Physical": physical
    }

    # ==============================
    # MIDDLE ROW: CHARTS (VERY SMALL)
    # ==============================
    col_left, col_mid, col_right = st.columns([1.9, 1.3, 1.6])

    # ---- SKILL BAR CHART (VERY SMALL)
    with col_left:
        fig, ax = plt.subplots(figsize=(3.2, 2.2))
        ax.barh(list(skills.keys()), list(skills.values()))
        ax.set_xlim(0, 100)
        ax.tick_params(labelsize=7)
        ax.set_title("Skills", fontsize=8)
        st.pyplot(fig)

    # ---- PLAYING STYLE PIE (VERY SMALL)
    with col_mid:
        attack = skills["Shoot"] + skills["Pace"]
        midfield = skills["Pass"] + skills["Dribble"]
        defense = skills["Defend"] + skills["Physical"]

        fig2, ax2 = plt.subplots(figsize=(2.6, 2.2))
        ax2.pie(
            [attack, midfield, defense],
            labels=["Att", "Mid", "Def"],
            autopct="%1.0f%%",
            startangle=90,
            textprops={"fontsize": 7}
        )
        ax2.set_title("Style", fontsize=8)
        st.pyplot(fig2)


    # ==============================
    # RIGHT SIDE: AI ROLE + INSIGHTS
    # ==============================
    with col_right:
        # ---- AI ROLE RECOMMENDATION
        if shooting > 70 and pace > 70:
            role = "Attacking Forward"
        elif passing > 70 and dribbling > 70:
            role = "Playmaking Midfielder"
        elif defending > 70 and physical > 70:
            role = "Defensive Specialist"
        else:
            role = "Balanced Utility Player"

        st.markdown("**🧠 AI Role**")
        st.success(role)
# ==============================
# PLAYER ARCHETYPE (UNSUPERVISED ML)
# ==============================
        archetype_input = build_archetype_features(player)
        archetype_scaled = archetype_scaler.transform(archetype_input[archetype_features])
        cluster_id = archetype_model.predict(archetype_scaled)[0]

        archetype_map = {
            0: "Goal Poacher",
            1: "Playmaker",
            2: "Defensive Anchor",
            3: "Pace Winger",
            4: "Box-to-Box Midfielder"
        }

        player_archetype = archetype_map.get(cluster_id, "Balanced Profile")

        st.markdown("**🧬 Player Archetype**")
        st.info(player_archetype)

        # ---- PLAYER INSIGHTS (SHORT)
        strongest = max(skills, key=skills.get)
        weakest = min(skills, key=skills.get)

        st.markdown(
            f"""
            <div style="font-size:20px; line-height:1.4;">
            <b>📌 Insights</b><br>
            • <b>Strongest:</b> {strongest}<br>
            • <b>Weakest:</b> {weakest}<br>
            • <b>Versatility:</b> High<br>
            • <b>Growth Level:</b> {'High' if growth >= 8 else 'Moderate'}
            </div>
            """,
            unsafe_allow_html=True
        )
        # ==============================
        # ML FUTURE OVERALL PREDICTION
        # ==============================
    st.markdown("---")
    st.markdown("### 🔮 ML Prediction")

    player_features = build_player_features(player, ml_features)
    predicted_peak = ml_model.predict(player_features)[0]

    predicted_peak = int(round(predicted_peak))
    expected_growth = predicted_peak - overall
    
    confidence = get_prediction_confidence(ml_model, player_features)

    m1, m2 ,m3= st.columns(3)
    m1.metric("Predicted Peak Overall", predicted_peak)
    m2.metric("Expected Growth", f"+{expected_growth}")
    m3.metric("Model Confidence", confidence)
# =====================================================
# OTHER TABS (WILL BE IMPLEMENTED NEXT)
# =====================================================
with tab2:
    player = selected_player
    st.subheader("🟩 Player Position on Football Pitch")

    # Load pitch image
    pitch_img = plt.imread("football_pitch.png")

    # Get primary position
    primary_pos = player["positions"].split(",")[0].strip()

    # Get coordinates (0–100 scale)
    x, y = position_coordinates_image(primary_pos)

    # Create figure
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Show pitch image
    ax.imshow(pitch_img)
    ax.axis("off")

    # Convert 0–100 coords to image pixels
    img_h, img_w, _ = pitch_img.shape
    plot_x = (x / 100) * img_w
    plot_y = (y / 100) * img_h

    # Plot player
    ax.scatter(
    plot_x,
    plot_y,
    s=100,                 # reduced for small pitch
    c="red",
    edgecolors="black",
    linewidths=0.8,
    zorder=5
    )
    
    ax.text(
    plot_x,
    plot_y - 23,
    player["name"],
    ha="center",
    va="bottom",
    color = "#FFEE00",
    fontsize=5,
    weight="bold",
    zorder=6
    )

    st.pyplot(fig)

    st.caption(f"📍 Position: {primary_pos} ({expand_position(primary_pos)})")

with tab3:
    pass
    st.subheader("🆚 Player vs Player Comparison")

    # ==============================
    # PLAYER SELECTION
    # ==============================
    sel1, sel2 = st.columns(2)

    with sel1:
        player_a_name = st.selectbox(
            "Player A",
            sorted(df["name"].unique()),
            key="cmp_a"
        )

    with sel2:
        player_b_name = st.selectbox(
            "Player B",
            sorted(df["name"].unique()),
            index=1,
            key="cmp_b"
        )
    
    player_a = df[df["name"] == player_a_name].iloc[0]
    player_b = df[df["name"] == player_b_name].iloc[0]

    # ==============================
    # KEY METRICS (COMPACT)
    # ==============================
    m1, m2, m3,m4 = st.columns(4)

    a_overall = int(player_a["overall_rating"])
    b_overall = int(player_b["overall_rating"])

    a_potential = int(player_a["potential"])
    b_potential = int(player_b["potential"])

    a_growth = a_potential - a_overall
    b_growth = b_potential - b_overall

    # Primary positions
    pos_a = player_a["positions"].split(",")[0].strip()
    pos_b = player_b["positions"].split(",")[0].strip()

    m1.metric("Overall (A | B)", f"{a_overall}| {b_overall}")
    m2.metric("Potential (A | B)", f"{a_potential}| {b_potential}")
    m3.metric("Growth (A | B)", f"+{a_growth} | +{b_growth}")
    m4.metric(
        "Position (A | B)",
        f"{pos_a} | {pos_b}",
        help=f"{expand_position(pos_a)} | {expand_position(pos_b)}"
    )
    st.markdown("---")
    
    
    # ==============================
    # DERIVED SKILLS
    # ==============================
    skills_a = get_player_skills(player_a)
    skills_b = get_player_skills(player_b)

    # ==============================
    # CHART ROW (3 CHARTS, COMPACT)
    # ==============================
    c1, c2, c3 = st.columns([2, 1.4, 1.4])

    # ---- SKILL COMPARISON (BAR)
    with c1:
        labels = list(skills_a.keys())
        values_a = list(skills_a.values())
        values_b = list(skills_b.values())
        y = range(len(labels))

        fig, ax = plt.subplots(figsize=(3.2, 2.2))
        ax.barh([i + 0.18 for i in y], values_a, height=0.35, label="A")
        ax.barh([i - 0.18 for i in y], values_b, height=0.35, label="B")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_title("Skills", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        plt.tight_layout(pad=0.6)

        st.pyplot(fig)

    # ---- PLAYER A STYLE (PIE)
    with c2:
        a_attack = skills_a["Shooting"] + skills_a["Pace"]
        a_mid = skills_a["Passing"] + skills_a["Dribbling"]
        a_def = skills_a["Defending"] + skills_a["Physical"]

        fig_a, ax_a = plt.subplots(figsize=(2.2, 2.2))
        ax_a.pie(
            [a_attack, a_mid, a_def],
            labels=["Att", "Mid", "Def"],
            autopct="%1.0f%%",
            startangle=90,
            radius=0.8,
            textprops={"fontsize": 7}
        )
        ax_a.set_title("Player A", fontsize=9)
        plt.tight_layout(pad=0.6)

        st.pyplot(fig_a)

    # ---- PLAYER B STYLE (PIE)
    with c3:
        b_attack = skills_b["Shooting"] + skills_b["Pace"]
        b_mid = skills_b["Passing"] + skills_b["Dribbling"]
        b_def = skills_b["Defending"] + skills_b["Physical"]

        fig_b, ax_b = plt.subplots(figsize=(2.2, 2.2))
        ax_b.pie(
            [b_attack, b_mid, b_def],
            labels=["Att", "Mid", "Def"],
            autopct="%1.0f%%",
            startangle=90,
            radius=0.8,
            textprops={"fontsize": 7}
        )
        ax_b.set_title("Player B", fontsize=9)
        plt.tight_layout(pad=0.6)

        st.pyplot(fig_b)

    # ==============================
    # FINAL AI VERDICT (TEXT ONLY)
    # ==============================
    better_now = player_a_name if a_overall > b_overall else player_b_name
    better_future = player_a_name if a_potential > b_potential else player_b_name

    st.markdown(
        f"""
        <div style="font-size:19px; line-height:1.4; margin-top:6px;">
        🧠 <b>AI Verdict</b><br>
        • Better current performer: <b>{better_now}</b><br>
        • Higher future potential: <b>{better_future}</b><br>
        • Skill profiles suggest different tactical suitability
        </div>
        """,
        unsafe_allow_html=True
    )   
    
    
# =====================================================
# TAB 4 — TEAM BUILDER
# =====================================================
    
with tab4:
    st.subheader("🏆 Team Builder – Squad Optimization")

    # ---------------------------------
    # FIXED UNIQUE ROLES (ONE PER POSITION)
    # ---------------------------------
    TEAM_ROLES = [
        "GK",
        "LB", "CB", "RB",
        "CDM", "CM",
        "LM", "RM",
        "LW", "RW",
        "ST"
        ]

    POSITION_SKILLS = {
    "GK": ["gk_diving", "gk_reflexes"],

    "CB": ["standing_tackle", "interceptions", "strength"],
    "LB": ["pace", "standing_tackle"],
    "RB": ["pace", "standing_tackle"],

    "CDM": ["interceptions", "short_passing"],
    "CM": ["short_passing", "dribbling"],

    "LM": ["pace", "dribbling", "short_passing", "standing_tackle"],
    "RM": ["pace", "dribbling", "short_passing", "standing_tackle"],

    "LW": ["pace", "dribbling"],
    "RW": ["pace", "dribbling"],

    "ST": ["finishing", "shot_power"]
    }

    # =====================================================
    # SESSION STATE — INITIAL BEST XI
    # =====================================================
    if "team" not in st.session_state:
        st.session_state.team = {}
        used_players = set()

        for role in TEAM_ROLES:
            eligible = df[df["positions"].str.contains(role, na=False)]
            eligible = eligible.sort_values("overall_rating", ascending=False)

            for _, row in eligible.iterrows():
                if row["name"] not in used_players:
                    st.session_state.team[role] = row
                    used_players.add(row["name"])
                    break

    team = st.session_state.team

    # =====================================================
    # PITCH (SAME POSITION, SMALLER SIZE)
    # =====================================================
    fig, ax = plt.subplots(figsize=(6.2,4.0))  # reduced size only

    ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color="black")
    ax.axhline(50, color="gray", linewidth=0.4)
    ax.plot([0, 16, 16, 0], [21, 21, 79, 79], color="black")
    ax.plot([100, 84, 84, 100], [21, 21, 79, 79], color="black")
    ax.add_patch(plt.Circle((50, 50), 9.15, fill=False))

    for role, row in team.items():
        x, y = position_coordinates_image(role)
        ax.scatter(x, y, s=100, color="green", zorder=5)
        ax.text(
            x,
            y + 2.5,
            row["name"],
            ha="center",
            va="bottom",
            fontsize=6
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    st.pyplot(fig)

    # =====================================================
    # BELOW PITCH: CURRENT XI  |  REPLACEMENTS
    # =====================================================
    col_left, col_right = st.columns([1.4, 1.6])

    # ---------------- LEFT: CURRENT XI ------------------
    with col_left:
        st.markdown("### 👥Team details")

        current_xi_df = pd.DataFrame([
            {
                "Role": role,
                "Player": row["name"],
                "Overall": row["overall_rating"]
            }
            for role, row in team.items()
        ])

        st.dataframe(
            current_xi_df,
            use_container_width=True,
            height=380
        )

    # ---------------- RIGHT: REPLACEMENTS ----------------
    with col_right:
        st.markdown("### 🔁 Replacement Suggestions")

        replace_role = st.selectbox(
            "Select position to replace",
            TEAM_ROLES
        )

        current_player = team[replace_role]
        current_rating = current_player["overall_rating"]

        candidates = df[
            df["positions"].str.contains(replace_role, na=False) &
            (~df["name"].isin([p["name"] for p in team.values()]))
        ].copy()

        def fit_score(row):
            skills = POSITION_SKILLS.get(replace_role, [])
            vals = [row[s] for s in skills if s in row]
            return sum(vals) / len(vals) if vals else 0

        candidates["FitScore"] = candidates.apply(fit_score, axis=1)
        candidates["Upgrade"] = candidates["overall_rating"] - current_rating

        top5 = candidates.sort_values(
            ["Upgrade", "FitScore"],
            ascending=False
        ).head(5)

        for _, row in top5.iterrows():
            c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 1.5])

            c1.markdown(f"**{row['name']}**")
            c2.markdown(f"{row['overall_rating']}")
            c3.markdown(f"+{row['Upgrade']}")

            if c4.button("Replace", key=f"replace_{replace_role}_{row['name']}"):
                st.session_state.team[replace_role] = row
                st.rerun()