import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from database import (
    authenticate_user,
    create_user,
    add_expense,
    get_user_expenses,
    get_user,
    init_database,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Expense Intelligence",
    page_icon="₹",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

TRANSACTIONS_FILE = BASE_DIR / "processed_transactions.csv"
LEAKS_FILE = BASE_DIR / "leak_summary.csv"
CLUSTERS_FILE = BASE_DIR / "user_clusters.csv"

try:
    init_database()
except Exception:
    pass


# ============================================================
# CLEAN, PRESENTABLE UI
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- BASE ---------- */
    .stApp {
        background: #F7F3EF;
    }
    

    .block-container {
        max-width: 1400px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #342823 !important;
        letter-spacing: -0.01em;
    }

    p, label, [data-testid="stMarkdownContainer"] {
        color: #4E413B;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: #292831;
        border-right: 1px solid #3B3943;
    }

    section[data-testid="stSidebar"] * {
        color: #F8F3EF !important;
    }

    section[data-testid="stSidebar"] .stCaption {
        color: #C8BFB9 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #F8F3EF !important;
    }

    .sidebar-brand {
        padding: 0.25rem 0 0.75rem 0;
    }

    .sidebar-brand-title {
        font-size: 1.22rem;
        font-weight: 800;
        color: #FFFFFF !important;
        line-height: 1.2;
    }

    .sidebar-brand-subtitle {
        margin-top: 0.3rem;
        font-size: 0.8rem;
        color: #C8BFB9 !important;
    }

    .sidebar-user-card {
        background: #34323A;
        border: 1px solid #47444E;
        border-radius: 14px;
        padding: 13px;
        margin: 0.8rem 0 1rem 0;
    }

    .sidebar-user-name {
        color: #FFFFFF !important;
        font-size: 0.95rem;
        font-weight: 700;
    }

    .sidebar-user-email {
        color: #CFC6C0 !important;
        font-size: 0.78rem;
        margin-top: 4px;
        word-break: break-word;
    }

    /* ---------- HERO ---------- */
    .hero {
        background: linear-gradient(135deg, #5D4037 0%, #8D6E63 100%);
        border-radius: 22px;
        padding: 28px 32px;
        margin-bottom: 26px;
        box-shadow: 0 12px 30px rgba(76, 52, 45, 0.14);
    }

    .hero-title {
        color: #FFFFFF !important;
        font-size: 2.15rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.15;
    }

    .hero-subtitle {
        color: #F5EDEA !important;
        font-size: 1rem;
        margin-top: 7px;
    }

    /* ---------- SECTION TITLES ---------- */
    .section-title {
        color: #4E342E !important;
        font-size: 1.42rem;
        font-weight: 800;
        margin: 1.35rem 0 0.85rem 0;
    }

    .page-intro {
        color: #6D625C !important;
        margin-top: -0.4rem;
        margin-bottom: 1rem;
    }

    /* ---------- METRIC CARDS ---------- */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5DDD7;
        border-radius: 17px;
        padding: 18px 19px;
        min-height: 112px;
        box-shadow: 0 6px 18px rgba(67, 50, 42, 0.06);
    }

    .metric-label {
        color: #766961 !important;
        font-size: 0.82rem;
        font-weight: 700;
    }

    .metric-value {
        color: #342823 !important;
        font-size: 1.82rem;
        line-height: 1.2;
        font-weight: 800;
        margin-top: 7px;
    }

    /* ---------- GENERAL CARDS ---------- */
    .white-card {
        background: #FFFFFF;
        border: 1px solid #E5DDD7;
        border-radius: 17px;
        padding: 20px;
        box-shadow: 0 6px 18px rgba(67, 50, 42, 0.06);
    }

    .profile-card {
        background: #FFFFFF;
        border: 1px solid #E5DDD7;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 6px 18px rgba(67, 50, 42, 0.06);
        min-height: 130px;
    }

    .profile-label {
        color: #7A6D66 !important;
        font-size: 0.82rem;
        font-weight: 700;
    }

    .profile-name {
        color: #4E342E !important;
        font-size: 1.5rem;
        font-weight: 800;
        margin-top: 6px;
    }

    /* ---------- RISK ---------- */
    .risk-high {
        background: #FFF1F0;
        border: 1px solid #F0C9C6;
        border-left: 6px solid #C62828;
        border-radius: 14px;
        padding: 17px 20px;
    }

    .risk-medium {
        background: #FFF8E8;
        border: 1px solid #F1DFB7;
        border-left: 6px solid #C98A00;
        border-radius: 14px;
        padding: 17px 20px;
    }

    .risk-low {
        background: #EEF8F0;
        border: 1px solid #CDE4D1;
        border-left: 6px solid #2E7D32;
        border-radius: 14px;
        padding: 17px 20px;
    }

    .info-card {
        background: #EEF4FA;
        border: 1px solid #CFDFEE;
        border-radius: 14px;
        padding: 14px 17px;
        color: #314A62 !important;
    }

    /* ---------- INPUTS ---------- */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    [data-testid="stTimeInput"] input {
        background: #FFFFFF !important;
        color: #342823 !important;
        -webkit-text-fill-color: #342823 !important;
        border: 1px solid #CFC5BE !important;
        border-radius: 10px !important;
        caret-color: #4E342E !important;
    }

    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stNumberInput"] input::placeholder {
        color: #8E837C !important;
        -webkit-text-fill-color: #8E837C !important;
        opacity: 1 !important;
    }

    [data-testid="stWidgetLabel"] p {
        color: #40332D !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border-color: #CFC5BE !important;
        color: #342823 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] span {
        color: #342823 !important;
    }

    /* ---------- BUTTONS ---------- */
    [data-testid="stButton"] button,
    [data-testid="stFormSubmitButton"] button {
        background: #6D4C41 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        min-height: 42px !important;
        font-weight: 750 !important;
        box-shadow: none !important;
    }

    [data-testid="stButton"] button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background: #4E342E !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    [data-testid="stButton"] button p,
    [data-testid="stButton"] button span,
    [data-testid="stFormSubmitButton"] button p,
    [data-testid="stFormSubmitButton"] button span {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    /* ---------- LOGIN ---------- */
    .login-shell {
        max-width: 760px;
        margin: 3.2rem auto 0 auto;
        background: #FFFFFF;
        border: 1px solid #E5DDD7;
        border-radius: 24px;
        padding: 30px 34px 34px 34px;
        box-shadow: 0 14px 38px rgba(67, 50, 42, 0.10);
    }

    .login-logo {
        width: 68px;
        height: 68px;
        margin: 0 auto 14px auto;
        border-radius: 20px;
        background: #5D4037;
        color: #FFFFFF !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 8px 20px rgba(76, 52, 45, 0.16);
    }

    .login-title {
        text-align: center;
        color: #342823 !important;
        font-size: 2rem;
        font-weight: 850;
        margin: 0;
    }

    .login-subtitle {
        text-align: center;
        color: #746961 !important;
        font-size: 0.98rem;
        line-height: 1.5;
        max-width: 560px;
        margin: 8px auto 24px auto;
    }

    /* ---------- TABS ---------- */
    [data-testid="stTabs"] button {
        color: #74665F !important;
        font-weight: 700 !important;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #4E342E !important;
    }

    /* ---------- DATAFRAMES / TABLES ---------- */
    [data-testid="stDataFrame"] {
        border: 1px solid #E5DDD7;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Streamlit metric visibility */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E3D8D1 !important;
    border-radius: 16px !important;
    padding: 16px !important;
}

[data-testid="stMetricLabel"] {
    color: #6D5A51 !important;
}

[data-testid="stMetricValue"] {
    color: #342823 !important;
    font-weight: 800 !important;
}

[data-testid="stMetricDelta"] {
    color: #6D5A51 !important;
}
/* Make normal chart/title text readable */
.stSubheader, .stMarkdown, .stCaption {
    color: #4E413B !important;
}

    /* ---------- ALERTS ---------- */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    /* ---------- REMOVE DEFAULT CHROME ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA
# ============================================================

@st.cache_data
def load_demo_data():
    transactions = pd.read_csv(TRANSACTIONS_FILE)

    leaks = (
        pd.read_csv(LEAKS_FILE)
        if LEAKS_FILE.exists()
        else pd.DataFrame()
    )

    clusters = (
        pd.read_csv(CLUSTERS_FILE)
        if CLUSTERS_FILE.exists()
        else pd.DataFrame()
    )

    return transactions, leaks, clusters


try:
    df, leaks, clusters = load_demo_data()
except Exception as exc:
    st.error("Unable to load project data.")
    st.code(str(exc))
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "logged_in": False,
    "selected_user": None,
    "page": "Dashboard",
    "user_name": None,
    "user_email": None,
    "monthly_income": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# TRANSACTION PREPARATION
# ============================================================

def prepare_transactions(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                "date_time",
                "amount",
                "description",
                "category",
                "monthly_income",
                "hour",
                "day_of_week",
                "is_weekend",
                "month",
                "is_late_night",
                "time_since_last_txn",
            ]
        )

    data = data.copy()

    data["date_time"] = pd.to_datetime(
        data["date_time"],
        errors="coerce",
    )

    data["amount"] = pd.to_numeric(
        data["amount"],
        errors="coerce",
    )

    data["monthly_income"] = pd.to_numeric(
        data["monthly_income"],
        errors="coerce",
    ).fillna(0.0)

    data = data.dropna(
        subset=["date_time", "amount"]
    )

    data = data.sort_values(
        "date_time"
    ).reset_index(drop=True)

    data["hour"] = data["date_time"].dt.hour
    data["day_of_week"] = data["date_time"].dt.dayofweek
    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["month"] = (
        data["date_time"]
        .dt.to_period("M")
        .astype(str)
    )

    data["is_late_night"] = (
        (data["hour"] >= 22)
        | (data["hour"] < 6)
    ).astype(int)

    gap = (
        data["date_time"]
        .diff()
        .dt.total_seconds()
        / 3600.0
    )

    median_gap = (
        float(gap.dropna().median())
        if gap.notna().any()
        else 0.0
    )

    data["time_since_last_txn"] = (
        gap.fillna(median_gap)
        .clip(lower=0)
    )

    return data


def is_sqlite_user(user_id: int) -> bool:
    try:
        return get_user(int(user_id)) is not None
    except Exception:
        return False


def get_user_transactions(user_id: int) -> pd.DataFrame:
    if is_sqlite_user(user_id):
        data = get_user_expenses(int(user_id))

        if data is None or data.empty:
            return prepare_transactions(
                pd.DataFrame()
            )

        return prepare_transactions(data)

    legacy = df[
        df["user_id"] == user_id
    ].copy()

    return prepare_transactions(legacy)


# ============================================================
# USER SUMMARY
# ============================================================

def build_dynamic_summary(
    user_id: int,
    transactions: pd.DataFrame,
) -> dict:

    if transactions.empty:
        income = float(
            st.session_state.get(
                "monthly_income"
            )
            or 0
        )

        return {
            "user_id": user_id,
            "monthly_income": income,
            "monthly_spend": 0.0,
            "spending_income_ratio": 0.0,
            "transaction_count": 0,
            "average_transaction": 0.0,
            "late_night_ratio": 0.0,
            "average_time_between_transactions": 0.0,
        }

    income = float(
        transactions["monthly_income"].iloc[0]
    )

    months = max(
        1,
        transactions["month"].nunique(),
    )

    total_spend = float(
        transactions["amount"].sum()
    )

    monthly_spend = (
        total_spend / months
    )

    return {
        "user_id": user_id,
        "monthly_income": income,
        "monthly_spend": monthly_spend,
        "spending_income_ratio": (
            monthly_spend / income
            if income > 0
            else 0.0
        ),
        "transaction_count": len(transactions),
        "average_transaction": float(
            transactions["amount"].mean()
        ),
        "late_night_ratio": float(
            transactions["is_late_night"].mean()
        ),
        "average_time_between_transactions": float(
            transactions["time_since_last_txn"].mean()
        ),
    }


# ============================================================
# RISK MODEL
# ============================================================

RISK_FEATURES = [
    "previous_spending_income_ratio",
    "previous_monthly_spend",
    "previous_average_transaction",
    "spending_growth",
    "previous_late_night_ratio",
    "previous_weekend_ratio",
    "previous_average_time_between_transactions",
    "previous_transaction_count",
]


@st.cache_resource
def train_risk_model(data: pd.DataFrame):
    monthly = prepare_transactions(data).copy()

    if monthly.empty:
        return None

    monthly_user = (
        monthly.groupby(
            ["user_id", "month"]
        )
        .agg(
            monthly_spend=("amount", "sum"),
            monthly_income=(
                "monthly_income",
                "first",
            ),
            average_transaction=(
                "amount",
                "mean",
            ),
            transaction_count=(
                "amount",
                "count",
            ),
            late_night_ratio=(
                "is_late_night",
                "mean",
            ),
            weekend_ratio=(
                "is_weekend",
                "mean",
            ),
            average_time_between_transactions=(
                "time_since_last_txn",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            ["user_id", "month"]
        )
    )

    monthly_user[
        "spending_income_ratio"
    ] = (
        monthly_user["monthly_spend"]
        / monthly_user["monthly_income"]
        .replace(0, np.nan)
    )

    mappings = [
        (
            "monthly_spend",
            "previous_monthly_spend",
        ),
        (
            "spending_income_ratio",
            "previous_spending_income_ratio",
        ),
        (
            "average_transaction",
            "previous_average_transaction",
        ),
        (
            "transaction_count",
            "previous_transaction_count",
        ),
        (
            "late_night_ratio",
            "previous_late_night_ratio",
        ),
        (
            "weekend_ratio",
            "previous_weekend_ratio",
        ),
        (
            "average_time_between_transactions",
            "previous_average_time_between_transactions",
        ),
    ]

    for source, target in mappings:
        monthly_user[target] = (
            monthly_user
            .groupby("user_id")[source]
            .shift(1)
        )

    monthly_user["spending_growth"] = (
        monthly_user["monthly_spend"]
        / monthly_user["previous_monthly_spend"]
        .replace(0, np.nan)
        - 1
    )

    monthly_user[
        "future_spending_ratio"
    ] = (
        monthly_user
        .groupby("user_id")[
            "spending_income_ratio"
        ]
        .shift(-1)
    )

    monthly_user[
        "future_overspending"
    ] = (
        monthly_user[
            "future_spending_ratio"
        ] > 0.80
    ).astype(int)

    model_data = monthly_user.dropna(
        subset=RISK_FEATURES
    )

    if (
        len(model_data) < 100
        or model_data["future_overspending"].nunique()
        < 2
    ):
        return None

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=3,
    )

    model.fit(
        model_data[RISK_FEATURES],
        model_data["future_overspending"],
    )

    return model


risk_model = train_risk_model(df)


def calculate_user_risk(
    user_id: int,
    transactions: pd.DataFrame,
) -> float:

    transactions = prepare_transactions(
        transactions
    )

    if transactions.empty:
        return 0.0

    monthly = (
        transactions.groupby("month")
        .agg(
            monthly_spend=("amount", "sum"),
            monthly_income=(
                "monthly_income",
                "first",
            ),
            average_transaction=(
                "amount",
                "mean",
            ),
            transaction_count=(
                "amount",
                "count",
            ),
            late_night_ratio=(
                "is_late_night",
                "mean",
            ),
            weekend_ratio=(
                "is_weekend",
                "mean",
            ),
            average_time_between_transactions=(
                "time_since_last_txn",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("month")
    )

    monthly[
        "spending_income_ratio"
    ] = (
        monthly["monthly_spend"]
        / monthly["monthly_income"]
        .replace(0, np.nan)
    ).fillna(0)

    # Use ML when there are at least two months.
    if (
        len(monthly) >= 2
        and risk_model is not None
    ):
        current = monthly.iloc[-1]
        previous = monthly.iloc[-2]

        growth = (
            current["monthly_spend"]
            / previous["monthly_spend"]
            - 1
            if previous["monthly_spend"] > 0
            else 0.0
        )

        features = pd.DataFrame(
            [{
                "previous_spending_income_ratio":
                    previous[
                        "spending_income_ratio"
                    ],
                "previous_monthly_spend":
                    previous[
                        "monthly_spend"
                    ],
                "previous_average_transaction":
                    previous[
                        "average_transaction"
                    ],
                "spending_growth": growth,
                "previous_late_night_ratio":
                    previous[
                        "late_night_ratio"
                    ],
                "previous_weekend_ratio":
                    previous[
                        "weekend_ratio"
                    ],
                "previous_average_time_between_transactions":
                    previous[
                        "average_time_between_transactions"
                    ],
                "previous_transaction_count":
                    previous[
                        "transaction_count"
                    ],
            }]
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(0)

        try:
            probability = (
                risk_model
                .predict_proba(
                    features[RISK_FEATURES]
                )[0][1]
            )

            return float(
                probability * 100
            )

        except Exception:
            pass

    # Transparent fallback for short histories.
    ratio = float(
        monthly.iloc[-1][
            "spending_income_ratio"
        ]
    )

    late = float(
        monthly.iloc[-1][
            "late_night_ratio"
        ]
    )

    score = min(
        55.0,
        max(0.0, ratio * 70.0),
    )

    score += min(
        20.0,
        max(0.0, late * 40.0),
    )

    return float(
        min(100.0, score)
    )


def get_risk_details(user_id: int, transactions: pd.DataFrame) -> dict:
    tx = prepare_transactions(transactions)
    months = int(tx["month"].nunique()) if not tx.empty else 0
    if months >= 2 and risk_model is not None:
        return {
            "score": calculate_user_risk(user_id, tx),
            "method": "ML Model Estimate",
            "history": months,
        }
    return {
        "score": calculate_user_risk(user_id, tx),
        "method": "Short-History Fallback",
        "history": months,
    }


# ============================================================
# PROFILE / LEAKS
# ============================================================

def get_user_profile(
    user_id: int,
    transactions: pd.DataFrame,
) -> str:

    if transactions.empty:
        return "New User"

    if not is_sqlite_user(user_id):
        result = clusters[
            clusters["user_id"] == user_id
        ]

        if (
            not result.empty
            and "user_type" in result.columns
        ):
            return str(
                result.iloc[0]["user_type"]
            )

    count = len(transactions)

    if count < 3:
        return "Early-Stage User"

    late = float(
        transactions["is_late_night"].mean()
    )

    average = float(
        transactions["amount"].mean()
    )

    if late >= 0.25:
        return "Late-Night Spender"

    if average >= 700:
        return "High-Value Spender"

    if count >= 100:
        return "Frequent Spender"

    return "Balanced Spender"


def detect_dynamic_leaks(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    columns = [
        "month",
        "category",
        "leak_type",
        "leak_amount",
        "leak_percentage",
        "severity",
        "evidence",
    ]

    transactions = prepare_transactions(
        transactions
    )

    if len(transactions) < 3:
        return pd.DataFrame(
            columns=columns
        )

    total_spend = float(
        transactions["amount"].sum()
    )

    if total_spend <= 0:
        return pd.DataFrame(
            columns=columns
        )

    rows = []

    # Category concentration.
    if len(transactions) >= 5:
        category_total = (
            transactions
            .groupby("category")["amount"]
            .sum()
        )

        for category, amount in category_total.items():
            pct = float(
                amount / total_spend
            )

            if pct >= 0.55:
                severity = (
                    "high"
                    if pct >= 0.70
                    else "medium"
                )

                rows.append(
                    {
                        "month":
                            transactions[
                                "month"
                            ].iloc[-1],
                        "category":
                            category,
                        "leak_type":
                            "category_concentration",
                        "leak_amount":
                            float(amount),
                        "leak_percentage":
                            pct * 100,
                        "severity":
                            severity,
                        "evidence":
                            (
                                f"{pct * 100:.1f}% "
                                f"of recorded spending "
                                f"is concentrated in "
                                f"{category}."
                            ),
                    }
                )

    # Category spikes across months.
    if transactions["month"].nunique() >= 2:
        category_month = (
            transactions
            .groupby(
                ["month", "category"]
            )["amount"]
            .sum()
        )

        months = sorted(
            transactions["month"].unique()
        )

        categories = (
            transactions[
                "category"
            ]
            .dropna()
            .unique()
        )

        for category in categories:
            values = [
                float(
                    category_month.get(
                        (month, category),
                        0.0,
                    )
                )
                for month in months
            ]

            for idx in range(1, len(values)):
                previous = values[idx - 1]
                current = values[idx]

                if previous <= 0:
                    continue

                growth = (
                    current / previous - 1
                )

                if (
                    growth >= 0.50
                    and current >= 500
                ):
                    severity = (
                        "high"
                        if growth >= 1.0
                        else "medium"
                    )

                    rows.append(
                        {
                            "month":
                                months[idx],
                            "category":
                                category,
                            "leak_type":
                                "category_spike",
                            "leak_amount":
                                current,
                            "leak_percentage":
                                growth * 100,
                            "severity":
                                severity,
                            "evidence":
                                (
                                    f"{category} "
                                    f"spending increased "
                                    f"{growth * 100:.1f}% "
                                    f"versus the previous "
                                    f"month."
                                ),
                        }
                    )

    # Micro-spending.
    micro = transactions[
        transactions["amount"] <= 150
    ]

    if len(micro) >= 5:
        amount = float(
            micro["amount"].sum()
        )

        pct = amount / total_spend

        if pct >= 0.05:
            severity = (
                "high"
                if pct >= 0.15
                else "medium"
            )

            rows.append(
                {
                    "month":
                        transactions[
                            "month"
                        ].iloc[-1],
                    "category":
                        "multiple",
                    "leak_type":
                        "micro_spending",
                    "leak_amount":
                        amount,
                    "leak_percentage":
                        pct * 100,
                    "severity":
                        severity,
                    "evidence":
                        (
                            f"{len(micro)} small "
                            f"transactions contributed "
                            f"₹{amount:,.0f}."
                        ),
                }
            )

    # Late-night behavior.
    late = transactions[
        transactions["is_late_night"] == 1
    ]

    if len(late) >= 3:
        amount = float(
            late["amount"].sum()
        )

        pct = amount / total_spend

        if pct >= 0.05:
            severity = (
                "high"
                if pct >= 0.15
                else "medium"
            )

            rows.append(
                {
                    "month":
                        transactions[
                            "month"
                        ].iloc[-1],
                    "category":
                        "multiple",
                    "leak_type":
                        "late_night",
                    "leak_amount":
                        amount,
                    "leak_percentage":
                        pct * 100,
                    "severity":
                        severity,
                    "evidence":
                        (
                            f"{len(late)} "
                            f"late-night transactions "
                            f"represented "
                            f"{pct * 100:.1f}% of "
                            f"recorded spending."
                        ),
                }
            )

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def get_user_leaks(
    user_id: int,
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    if is_sqlite_user(user_id):
        return detect_dynamic_leaks(
            transactions
        )

    if leaks.empty:
        return pd.DataFrame()

    return leaks[
        leaks["user_id"] == user_id
    ].copy()


# ============================================================
# LOGIN / REGISTER
# ============================================================

def login_page():
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown('<div class="login-logo">₹</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-title">Expense Intelligence</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="login-subtitle">Track spending, discover behavioral patterns, and understand your financial risk.</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            tab_login, tab_register = st.tabs(["Log in", "Create account"])

            with tab_login:
                st.subheader("Welcome back")
                email = st.text_input(
                    "Email address",
                    placeholder="you@example.com",
                    key="login_email",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    key="login_password",
                )

                if st.button("Log in", key="login_button", width="stretch"):
                    clean_email = email.strip().lower()
                    if not clean_email or not password:
                        st.warning("Please enter both email and password.")
                    else:
                        try:
                            user = authenticate_user(clean_email, password)
                        except Exception as exc:
                            st.error("Unable to access the login database.")
                            st.code(str(exc))
                            user = None

                        if user:
                            user_id, name, user_email, monthly_income = user
                            st.session_state.logged_in = True
                            st.session_state.selected_user = int(user_id)
                            st.session_state.user_name = name
                            st.session_state.user_email = user_email
                            st.session_state.monthly_income = float(monthly_income)
                            st.session_state.page = "Dashboard"
                            st.rerun()
                        else:
                            st.error("Invalid email or password.")

            with tab_register:
                st.subheader("Create your account")
                st.caption("Your expenses will be stored in your personal account.")
                name = st.text_input("Full name", placeholder="Enter your name", key="register_name")
                email = st.text_input("Email address", placeholder="you@example.com", key="register_email")
                password = st.text_input("Password", type="password", placeholder="Create a password", key="register_password")
                confirm_password = st.text_input("Confirm password", type="password", placeholder="Re-enter your password", key="register_confirm_password")
                monthly_income = st.number_input("Monthly income (₹)", min_value=1.0, value=30000.0, step=1000.0, key="register_income")

                if st.button("Create account", key="register_button", width="stretch"):
                    clean_name = name.strip()
                    clean_email = email.strip().lower()
                    if not clean_name or not clean_email or not password:
                        st.warning("Please complete all required fields.")
                    elif password != confirm_password:
                        st.error("Passwords do not match.")
                    elif monthly_income <= 0:
                        st.error("Monthly income must be greater than ₹0.")
                    else:
                        try:
                            user_id = create_user(clean_name, clean_email, password, monthly_income)
                            if user_id:
                                st.success("Account created successfully. You can now log in.")
                            else:
                                st.error("An account with this email already exists.")
                        except Exception as exc:
                            st.error("The account could not be created.")
                            st.code(str(exc))


# ============================================================
# SIDEBAR
# ============================================================

def sidebar():
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">
                Expense Intelligence
            </div>
            <div class="sidebar-brand-subtitle">
                Personal financial analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.divider()

    st.sidebar.markdown(
        f"""
        <div class="sidebar-user-card">
            <div class="sidebar-user-name">
                {st.session_state.get("user_name") or "User"}
            </div>
            <div class="sidebar-user-email">
                {st.session_state.get("user_email") or ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pages = [
        "Dashboard",
        "Add Expense",
        "Spending Analysis",
        "Spending Leaks",
        "Future Risk",
        "Spending Profile",
        "Smart Insights",
        "What-If Simulator",
    ]

    # Keep the navigation widget separate from the page variable.
    # This prevents the double-click navigation problem.
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = st.session_state.page

    # Keep radio selection synchronized with the current page.
    if st.session_state.nav_page != st.session_state.page:
        st.session_state.nav_page = st.session_state.page

    def change_page():
        st.session_state.page = st.session_state.nav_page

    st.sidebar.radio(
        "Navigate",
        pages,
        key="nav_page",
        on_change=change_page,
    )

    st.sidebar.divider()

    if st.sidebar.button(
        "Log out",
        width="stretch",
    ):
        for key, value in DEFAULTS.items():
            st.session_state[key] = value

        st.session_state.pop("nav_page", None)

        st.rerun()

# ============================================================
# HEADER
# ============================================================

def dashboard_header(
    user_id: int,
    transactions: pd.DataFrame,
):

    profile = get_user_profile(
        user_id,
        transactions,
    )

    display_name = (
        st.session_state.get("user_name")
        or f"User {user_id}"
    )

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">
                Expense Intelligence
            </div>
            <div class="hero-subtitle">
                {display_name} &nbsp;·&nbsp; {profile}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DASHBOARD
# ============================================================

def dashboard_page(user_id: int):

    transactions = get_user_transactions(
        user_id
    )

    summary = build_dynamic_summary(
        user_id,
        transactions,
    )

    risk = calculate_user_risk(
        user_id,
        transactions,
    )

    user_leaks = get_user_leaks(
        user_id,
        transactions,
    )

    st.markdown(
        "<div class='section-title'>Financial overview</div>",
        unsafe_allow_html=True,
    )

    columns = st.columns(4)

    metrics = [
        (
            "Monthly income",
            f"₹{summary['monthly_income']:,.0f}",
        ),
        (
            "Monthly spending",
            f"₹{summary['monthly_spend']:,.0f}",
        ),
        (
            "Spending / income",
            f"{summary['spending_income_ratio'] * 100:.1f}%",
        ),
        (
            "Future risk",
            f"{risk:.1f}%",
        ),
    ]

    for col, (label, value) in zip(
        columns,
        metrics,
    ):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        {label}
                    </div>
                    <div class="metric-value">
                        {value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div class='section-title'>Risk status</div>",
        unsafe_allow_html=True,
    )

    if risk >= 70:
        st.markdown(
            f"""
            <div class="risk-high">
                <h3>Elevated future risk</h3>
                <p>
                    Current estimated probability:
                    <b>{risk:.1f}%</b>.
                    Recent spending signals indicate
                    stronger overspending risk.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif risk >= 40:
        st.markdown(
            f"""
            <div class="risk-medium">
                <h3>Moderate future risk</h3>
                <p>
                    Current estimated probability:
                    <b>{risk:.1f}%</b>.
                    Monitor recent spending changes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            f"""
            <div class="risk-low">
                <h3>Lower future risk</h3>
                <p>
                    Current estimated probability:
                    <b>{risk:.1f}%</b>.
                    Current signals look relatively controlled.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='section-title'>Spending trends</div>",
        unsafe_allow_html=True,
    )

    if transactions.empty:
        st.markdown(
            """
            <div class="info-card">
                No expenses have been recorded yet.
                Open <b>Add Expense</b> to enter your first transaction.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        left, right = st.columns(2)

        with left:

            category = (
                transactions
                .groupby("category")["amount"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            st.caption(
                "Total spending by category"
            )

            st.bar_chart(
                category
            )

        with right:

            monthly = (
                transactions
                .groupby("month")["amount"]
                .sum()
                .sort_index()
            )

            st.caption(
                "Spending by month"
            )

            st.line_chart(
                monthly
            )

    st.markdown(
        "<div class='section-title'>Your spending profile</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
            f"""
            <div class="profile-card">
                <div class="profile-label">
                    Behavioral profile
                </div>
                <div class="profile-name">
                    {get_user_profile(user_id, transactions)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            f"""
            <div class="profile-card">
                <div class="profile-label">
                    Behavioral alerts
                </div>
                <div class="profile-name">
                    {len(user_leaks)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ADD EXPENSE
# ============================================================

def add_expense_page(user_id: int):

    st.markdown(
        "<div class='section-title'>Add expense</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='page-intro'>Record a new transaction in your personal expense history.</div>",
        unsafe_allow_html=True,
    )

    with st.form(
        "expense_form",
        clear_on_submit=True,
    ):

        left, right = st.columns(2)

        with left:

            amount = st.number_input(
                "Amount (₹)",
                min_value=1.0,
                value=500.0,
                step=50.0,
            )

            category = st.selectbox(
                "Category",
                [
                    "food",
                    "transport",
                    "shopping",
                    "entertainment",
                    "health",
                    "utilities",
                ],
            )

            description = st.text_input(
                "Description",
                placeholder="Lunch, cab, online purchase...",
            )

        with right:

            expense_date = st.date_input(
                "Date"
            )

            expense_time = st.time_input(
                "Time"
            )

            current_income = float(
                st.session_state.get(
                    "monthly_income"
                )
                or 30000
            )

            income = st.number_input(
                "Monthly income (₹)",
                min_value=1.0,
                value=current_income,
                step=1000.0,
            )

        submitted = st.form_submit_button(
            "Save expense",
            width="stretch",
        )

    if submitted:

        timestamp = (
            f"{expense_date} "
            f"{expense_time}"
        )

        try:

            add_expense(
                user_id=user_id,
                date_time=timestamp,
                amount=float(amount),
                description=description.strip(),
                category=category,
                monthly_income=float(income),
            )

            st.session_state.monthly_income = (
                float(income)
            )

            st.cache_data.clear()

            st.success(
                f"₹{amount:,.2f} expense saved successfully."
            )

            st.rerun()

        except Exception as exc:

            st.error(
                "The expense could not be saved."
            )

            st.code(
                str(exc)
            )

    transactions = get_user_transactions(
        user_id
    )

    if not transactions.empty:

        st.markdown(
            "<div class='section-title'>Recent expenses</div>",
            unsafe_allow_html=True,
        )

        recent = (
            transactions[
                [
                    "date_time",
                    "amount",
                    "category",
                    "description",
                ]
            ]
            .sort_values(
                "date_time",
                ascending=False,
            )
            .head(10)
            .copy()
        )

        st.dataframe(
            recent,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# SPENDING ANALYSIS
# ============================================================

def spending_page(user_id: int):
    st.markdown("<div class='section-title'>Spending analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-intro'>Review your spending trend, category mix, and recent transactions.</div>", unsafe_allow_html=True)
    transactions = get_user_transactions(user_id)
    if transactions.empty:
        st.info("No spending data yet. Add your first expense to begin.")
        return

    summary = build_dynamic_summary(user_id, transactions)
    monthly = transactions.groupby("month")["amount"].sum().sort_index()
    category = transactions.groupby("category")["amount"].sum().sort_values(ascending=False)
    weekend_pct = float(transactions["is_weekend"].mean() * 100)
    late_pct = float(transactions["is_late_night"].mean() * 100)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total spending", f"₹{transactions['amount'].sum():,.0f}")
    c2.metric("Monthly average", f"₹{summary['monthly_spend']:,.0f}")
    c3.metric("Transactions", f"{len(transactions):,}")
    c4.metric("Average transaction", f"₹{transactions['amount'].mean():,.0f}")
    c5.metric("Largest transaction", f"₹{transactions['amount'].max():,.0f}")

    left, right = st.columns(2)
    with left:
        st.markdown("<div class='white-card'><b>Monthly spending</b></div>", unsafe_allow_html=True)
        st.line_chart(monthly)
    with right:
        st.markdown("<div class='white-card'><b>Category spending</b></div>", unsafe_allow_html=True)
        st.bar_chart(category)

    st.markdown("<div class='section-title'>Behavior snapshot</div>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    b1.metric("Late-night activity", f"{late_pct:.1f}%")
    b2.metric("Weekend activity", f"{weekend_pct:.1f}%")
    b3.metric("Spending / income", f"{summary['spending_income_ratio'] * 100:.1f}%")

    st.markdown("<div class='section-title'>Transaction history</div>", unsafe_allow_html=True)
    display = transactions[["date_time", "category", "amount", "description", "is_late_night", "is_weekend"]].sort_values("date_time", ascending=False).head(50).copy()
    display["category"] = display["category"].str.title()
    display["amount"] = display["amount"].map(lambda x: f"₹{x:,.2f}")
    display["is_late_night"] = display["is_late_night"].map({1: "Yes", 0: "No"})
    display["is_weekend"] = display["is_weekend"].map({1: "Yes", 0: "No"})
    display.columns = ["Date & Time", "Category", "Amount", "Description", "Late Night", "Weekend"]
    st.dataframe(display, width="stretch", hide_index=True)


# ============================================================
# LEAKS
# ============================================================

def leaks_page(user_id: int):
    st.markdown("<div class='section-title'>Spending leak detection</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-intro'>Behavioral signals highlight recurring patterns that may deserve attention.</div>",
        unsafe_allow_html=True,
    )
    transactions = get_user_transactions(user_id)
    user_leaks = get_user_leaks(user_id, transactions)

    if len(transactions) < 3:
        st.info(f"You currently have {len(transactions)} transaction(s). Add at least 3 transactions before meaningful leak detection is shown.")
        return

    if user_leaks.empty:
        st.success("No significant behavioral spending leaks detected in the available history.")
        return

    high = int(user_leaks["severity"].eq("high").sum())
    medium = int(user_leaks["severity"].eq("medium").sum())
    low = int(user_leaks["severity"].eq("low").sum())
    st.markdown("<div class='section-title'>Severity overview</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detected patterns", len(user_leaks))
    c2.metric("High", high)
    c3.metric("Medium", medium)
    c4.metric("Low", low)

    severity_filter = st.selectbox("Filter by severity", ["All", "high", "medium", "low"], key="leak_severity")
    display = user_leaks.copy()
    if severity_filter != "All":
        display = display[display["severity"] == severity_filter]

    if not display.empty:
        top = display.sort_values("leak_amount", ascending=False).head(4)
        for _, leak in top.iterrows():
            cls = {"high": "risk-high", "medium": "risk-medium", "low": "risk-low"}.get(str(leak["severity"]), "info-card")
            title = str(leak["leak_type"]).replace("_", " ").title()
            st.markdown(
                f"<div class='{cls}'><b>{title}</b><br>Category: {str(leak['category']).title()}<br>Amount involved: ₹{float(leak['leak_amount']):,.2f}<br>Impact: {float(leak['leak_percentage']):.1f}%<br><span>{str(leak['evidence'])}</span></div>",
                unsafe_allow_html=True,
            )

    table = display.sort_values("leak_amount", ascending=False).copy()
    table["category"] = table["category"].astype(str).str.title()
    table["leak_type"] = table["leak_type"].str.replace("_", " ").str.title()
    table["leak_amount"] = table["leak_amount"].map(lambda x: f"₹{x:,.2f}")
    table["leak_percentage"] = table["leak_percentage"].map(lambda x: f"{x:.1f}%")
    table.columns = ["Month", "Category", "Pattern", "Amount", "Impact", "Severity", "Evidence"]
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption("Note: amounts from different signals may overlap; the total is not necessarily unique spending.")


# ============================================================
# FUTURE RISK
# ============================================================

def risk_page(user_id: int):
    st.markdown("<div class='section-title'>Future risk prediction</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-intro'>Estimate the likelihood of future overspending from your recorded behavior.</div>",
        unsafe_allow_html=True,
    )
    transactions = get_user_transactions(user_id)
    details = get_risk_details(user_id, transactions)
    summary = build_dynamic_summary(user_id, transactions)
    risk = details["score"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated risk", f"{risk:.1f}%")
    c2.metric("History available", f"{details['history']} month(s)")
    c3.metric("Estimation method", details["method"])

    if details["method"] == "ML Model Estimate":
        st.success("This score is produced by the trained Random Forest model because at least two months of history are available.")
    else:
        st.info("This is a transparent short-history fallback. More monthly history is needed before the ML model can be used for this user.")

    if risk >= 70:
        st.error("High risk: recent spending shows stronger warning signals.")
    elif risk >= 40:
        st.warning("Moderate risk: monitor recent spending behavior.")
    else:
        st.success("Lower risk: current behavioral signals are relatively stable.")

    st.markdown("<div class='section-title'>Risk factors</div>", unsafe_allow_html=True)
    factors = pd.DataFrame({
        "Factor": ["Spending / income ratio", "Average transaction", "Transaction count", "Late-night activity"],
        "Value": [
            f"{summary['spending_income_ratio'] * 100:.1f}%",
            f"₹{summary['average_transaction']:,.2f}",
            f"{summary['transaction_count']:,}",
            f"{summary['late_night_ratio'] * 100:.1f}%",
        ],
    })
    st.dataframe(factors, width="stretch", hide_index=True)
    st.markdown(
        "<div class='info-card'>The risk score is an analytical estimate based on recorded spending behavior. It is not financial advice.</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# PROFILE
# ============================================================

def profile_page(user_id: int):

    st.markdown(
        "<div class='section-title'>Spending profile</div>",
        unsafe_allow_html=True,
    )

    transactions = get_user_transactions(
        user_id
    )

    summary = build_dynamic_summary(
        user_id,
        transactions,
    )

    profile = get_user_profile(
        user_id,
        transactions,
    )

    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-label">
                Behavioral classification
            </div>
            <div class="profile-name">
                {profile}
            </div>
            <p>
                Based on transaction frequency, transaction size,
                timing behavior, and spending relative to income.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='section-title'>Behavior metrics</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Average transaction",
        f"₹{summary['average_transaction']:,.0f}",
    )

    c2.metric(
        "Transactions",
        f"{summary['transaction_count']:,}",
    )

    c3.metric(
        "Late-night activity",
        f"{summary['late_night_ratio'] * 100:.1f}%",
    )

    c4.metric(
        "Average transaction gap",
        f"{summary['average_time_between_transactions']:.1f} hrs",
    )

    if len(transactions) < 3:

        st.info(
            "Add a few more transactions to strengthen behavioral classification."
        )


# ============================================================
# SMART INSIGHTS
# ============================================================

def insights_page(user_id: int):
    st.markdown("<div class='section-title'>Smart insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-intro'>A readable summary of your current spending behavior and the strongest signals detected.</div>", unsafe_allow_html=True)
    transactions = get_user_transactions(user_id)
    summary = build_dynamic_summary(user_id, transactions)
    profile = get_user_profile(user_id, transactions)
    user_leaks = get_user_leaks(user_id, transactions)

    c1, c2, c3 = st.columns(3)
    c1.metric("Spending profile", profile)
    c2.metric("Monthly spending", f"₹{summary['monthly_spend']:,.0f}")
    c3.metric("Spending / income", f"{summary['spending_income_ratio'] * 100:.1f}%")

    if transactions.empty:
        st.info("Add expenses to receive personalized insights.")
        return

    ratio = summary["spending_income_ratio"]
    if ratio >= 0.80:
        st.error("Your estimated monthly spending is using a high share of your income. Focus first on the largest recurring categories.")
    elif ratio >= 0.60:
        st.warning("Your spending is taking a substantial share of income. Review discretionary categories regularly.")
    else:
        st.success("Your current spending-to-income ratio is relatively controlled.")

    st.markdown("<div class='section-title'>Detected signals</div>", unsafe_allow_html=True)
    if user_leaks.empty:
        if len(transactions) < 3:
            st.info("Add more transactions to unlock stronger behavioral insights.")
        else:
            st.success("No major behavioral spending patterns were detected.")
    else:
        for _, leak in user_leaks.sort_values("leak_amount", ascending=False).head(5).iterrows():
            st.markdown(
                f"**{str(leak['leak_type']).replace('_', ' ').title()}**  \nCategory: {str(leak['category']).title()}  \nAmount involved: ₹{float(leak['leak_amount']):,.2f}  \nSeverity: **{str(leak['severity']).title()}**  \nEvidence: {str(leak['evidence'])}",
            )

    st.markdown("<div class='section-title'>Recommendations</div>", unsafe_allow_html=True)
    recs = []
    if ratio >= 0.80:
        recs.append("Reduce discretionary spending and set a clear monthly limit.")
    elif ratio >= 0.60:
        recs.append("Monitor discretionary categories closely before the end of each month.")
    else:
        recs.append("Maintain your current spending discipline and continue monitoring your ratio.")
    types = set(user_leaks["leak_type"]) if not user_leaks.empty else set()
    if "micro_spending" in types:
        recs.append("Review frequent low-value purchases because they can accumulate over time.")
    if "late_night" in types:
        recs.append("Consider a personal late-night spending limit.")
    if "category_spike" in types:
        recs.append("Review the category with the strongest month-over-month increase.")
    if "category_concentration" in types:
        recs.append("Consider a separate budget for your most concentrated spending category.")
    for item in recs[:4]:
        st.info(item)


# ============================================================
# WHAT-IF
# ============================================================

def what_if_page(user_id: int):

    st.markdown(
        "<div class='section-title'>What-if spending simulator</div>",
        unsafe_allow_html=True,
    )

    transactions = get_user_transactions(
        user_id
    )

    summary = build_dynamic_summary(
        user_id,
        transactions,
    )

    income = summary["monthly_income"]
    spending = summary["monthly_spend"]

    if income <= 0:

        st.info(
            "Add your income information to use the simulator."
        )

        return

    st.markdown(
        "<div class='page-intro'>Explore how reducing spending changes your monthly financial buffer.</div>",
        unsafe_allow_html=True,
    )

    reduction = st.slider(
        "Reduce monthly spending by",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        format="%d%%",
    )

    simulated_spending = (
        spending
        * (1 - reduction / 100)
    )

    current_buffer = (
        income - spending
    )

    simulated_buffer = (
        income - simulated_spending
    )

    improvement = (
        simulated_buffer
        - current_buffer
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Current spending",
        f"₹{spending:,.0f}",
    )

    c2.metric(
        "Simulated spending",
        f"₹{simulated_spending:,.0f}",
    )

    c3.metric(
        "Additional monthly buffer",
        f"₹{improvement:,.0f}",
    )

    comparison = pd.DataFrame(
        {
            "Scenario": [
                "Current",
                "Simulated",
            ],
            "Monthly spending": [
                spending,
                simulated_spending,
            ],
        }
    )

    st.subheader(
        "Spending comparison"
    )

    st.bar_chart(
        comparison.set_index(
            "Scenario"
        )
    )

    new_ratio = (
        simulated_spending / income
        if income
        else 0.0
    )

    st.subheader(
        "Simulated spending ratio"
    )

    st.progress(
        min(
            float(new_ratio),
            1.0,
        )
    )

    st.write(
        "Simulated spending-to-income ratio: "
        f"**{new_ratio * 100:.1f}%**"
    )

    if new_ratio < 0.60:

        st.success(
            "This scenario keeps spending below 60% of income."
        )

    elif new_ratio < 0.80:

        st.warning(
            "This scenario leaves a moderate financial buffer."
        )

    else:

        st.error(
            "This scenario still leaves a relatively high spending-to-income ratio."
        )


# ============================================================
# MAIN
# ============================================================

if not st.session_state.logged_in:

    login_page()
    st.stop()


user_id = int(
    st.session_state.selected_user
)

transactions = get_user_transactions(
    user_id
)

sidebar()

dashboard_header(
    user_id,
    transactions,
)

if st.session_state.page == "Dashboard":

    dashboard_page(
        user_id
    )

elif st.session_state.page == "Add Expense":

    add_expense_page(
        user_id
    )

elif st.session_state.page == "Spending Analysis":

    spending_page(
        user_id
    )

elif st.session_state.page == "Spending Leaks":

    leaks_page(
        user_id
    )

elif st.session_state.page == "Future Risk":

    risk_page(
        user_id
    )

elif st.session_state.page == "Spending Profile":

    profile_page(
        user_id
    )

elif st.session_state.page == "Smart Insights":

    insights_page(
        user_id
    )

elif st.session_state.page == "What-If Simulator":

    what_if_page(
        user_id
    )
