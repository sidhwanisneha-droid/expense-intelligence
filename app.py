import streamlit as st
import pandas as pd

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Expense Intelligence",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to bottom right, #f8f5f2, #efe7df);
}

/* Remove top spacing */
.block-container {
    padding-top: 2rem;
}

/* Main Heading */
.main-title {
    text-align: center;
    font-size: 56px;
    font-weight: 700;
    color: #4E342E;
    margin-bottom: 10px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    font-size: 22px;
    color: #795548;
    margin-bottom: 50px;
}

/* Cards */
.option-card {
    background-color: white;
    padding: 35px;
    border-radius: 22px;
    text-align: center;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    border: 2px solid transparent;
    transition: 0.3s;
}

/* Hover */
.option-card:hover {
    border: 2px solid #8D6E63;
    transform: scale(1.02);
}

/* Card Title */
.card-title {
    font-size: 28px;
    font-weight: bold;
    color: #4E342E;
    margin-bottom: 10px;
}

/* Card Text */
.card-text {
    color: #6D4C41;
    font-size: 18px;
}

/* Labels */
label {
    color: #4E342E !important;
    font-weight: 600 !important;
}

/* Input Fields */
.stTextInput input,
.stDateInput input {
    background-color: white !important;
    color: #3E2723 !important;
    border: 2px solid #D7CCC8 !important;
    border-radius: 14px !important;
    padding: 14px !important;
    font-size: 16px !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background-color: #6D4C41;
    color: white;
    border-radius: 14px;
    border: none;
    padding: 14px;
    font-size: 17px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #5D4037;
    color: white;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ---------------- #
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------------- HOME PAGE ---------------- #
def home_page():

    st.markdown(
        "<div class='main-title'>💰 Welcome Back 👋</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sub-title'>Track spending. Detect leaks. Build smarter financial habits.</div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # -------- LOGIN CARD -------- #
    with col1:

        st.markdown("""
        <div class='option-card'>
            <div class='card-title'>🔐 Already a User</div>
            <div class='card-text'>
                Login to your workspace and continue tracking your financial behavior.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        if st.button("Login"):
            st.session_state.page = "login"
            st.rerun()

    # -------- SIGNUP CARD -------- #
    with col2:

        st.markdown("""
        <div class='option-card'>
            <div class='card-title'>✨ Create Account</div>
            <div class='card-text'>
                Start your financial journey and discover hidden spending patterns.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        if st.button("Sign Up"):
            st.session_state.page = "signup"
            st.rerun()

# ---------------- LOGIN PAGE ---------------- #
def login_page():

    st.markdown(
        "<div class='main-title'>🔐 Login</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sub-title'>Enter your credentials to continue</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1,1.5,1])

    with col2:

        st.text_input(
            "📧 Email",
            placeholder="Enter your email"
        )

        st.text_input(
            "🔒 Password",
            type="password",
            placeholder="Enter your password"
        )

        st.write("")

        if st.button("Login to Dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()

        if st.button("⬅ Back"):
            st.session_state.page = "home"
            st.rerun()

# ---------------- SIGNUP PAGE ---------------- #
def signup_page():

    st.markdown(
        "<div class='main-title'>✨ Create Account</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sub-title'>Start your financial journey today</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1,1.8,1])

    with col2:

        st.text_input(
            "👤 Full Name",
            placeholder="Enter your full name"
        )

        st.text_input(
            "📱 Phone Number",
            placeholder="Enter phone number"
        )

        st.date_input("🎂 Date of Birth")

        st.text_input(
            "📧 Email",
            placeholder="Enter your email"
        )

        st.text_input(
            "🔒 Password",
            type="password",
            placeholder="Create password"
        )

        st.text_input(
            "🔐 Confirm Password",
            type="password",
            placeholder="Confirm password"
        )

        st.write("")

        if st.button("Create Account"):
            st.success("Account created successfully!")

        if st.button("⬅ Back"):
            st.session_state.page = "home"
            st.rerun()

# ---------------- DASHBOARD ---------------- #
def dashboard():

    # -------- LOAD DATA -------- #
    df = pd.read_csv("processed_transactions.csv")

    # -------- TOP BAR -------- #
    col1, col2 = st.columns([6,1])

    with col1:
        st.markdown(
            "<h1 style='color:#4E342E;'>📊 Expense Dashboard</h1>",
            unsafe_allow_html=True
        )

    with col2:
        if st.button("Logout"):
            st.session_state.page = "home"
            st.rerun()

    st.markdown("---")

    # -------- USER SELECT -------- #
    user_id = st.selectbox(
        "👤 Select User",
        df["user_id"].unique()
    )

    user_data = df[df["user_id"] == user_id].copy()

    # -------- SUMMARY -------- #
    total_spend = round(user_data["amount"].sum(), 2)

    avg_transaction = round(user_data["amount"].mean(), 2)

    monthly_spend = round(total_spend / 12, 2)

    # -------- SUMMARY CARDS -------- #
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='option-card'>
            <div class='card-title'>💰 Total Spend</div>
            <div class='card-text'>₹ {total_spend}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='option-card'>
            <div class='card-title'>📈 Avg Transaction</div>
            <div class='card-text'>₹ {avg_transaction}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='option-card'>
            <div class='card-title'>📅 Monthly Spend</div>
            <div class='card-text'>₹ {monthly_spend}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------- CHART SECTION -------- #
    st.markdown(
        "<h2 style='color:#4E342E;'>📊 Spending Analytics</h2>",
        unsafe_allow_html=True
    )

    # Category spending
    category_spend = user_data.groupby("category")["amount"].sum()

    col1, col2 = st.columns(2)

    # -------- BAR CHART -------- #
    with col1:

        st.markdown("""
        <div class='option-card'>
            <div class='card-title'>📦 Category Spending</div>
        </div>
        """, unsafe_allow_html=True)

        st.bar_chart(category_spend)

    # -------- LINE CHART -------- #
    with col2:

        st.markdown("""
        <div class='option-card'>
            <div class='card-title'>📈 Spending Trend</div>
        </div>
        """, unsafe_allow_html=True)

        user_data["date_time"] = pd.to_datetime(
            user_data["date_time"]
        )

        trend = user_data.groupby(
            user_data["date_time"].dt.date
        )["amount"].sum()

        st.line_chart(trend)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------- TRANSACTION HISTORY -------- #
    st.markdown(
        "<h2 style='color:#4E342E;'>🧾 Recent Transactions</h2>",
        unsafe_allow_html=True
    )

    recent = user_data.sort_values(
        by="date_time",
        ascending=False
    )

    recent = recent[
        ["date_time", "category", "amount"]
    ].head(10)

    recent.columns = [
        "Date & Time",
        "Category",
        "Amount (₹)"
    ]

    st.dataframe(
        recent,
        use_container_width=True
    )

# ---------------- MAIN FLOW ---------------- #
if st.session_state.page == "home":
    home_page()

elif st.session_state.page == "login":
    login_page()

elif st.session_state.page == "signup":
    signup_page()

elif st.session_state.page == "dashboard":
    dashboard()