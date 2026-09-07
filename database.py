import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).parent / "expense_intelligence.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            monthly_income REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            expense_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date_time TEXT NOT NULL,
            amount REAL NOT NULL,
            description TEXT,
            category TEXT NOT NULL,
            monthly_income REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    conn.commit()
    conn.close()


def create_user(name, email, password, monthly_income):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO users
            (name, email, password, monthly_income)
            VALUES (?, ?, ?, ?)
        """, (name, email, password, monthly_income))

        conn.commit()
        return cursor.lastrowid

    except sqlite3.IntegrityError:
        return None

    finally:
        conn.close()


def authenticate_user(email, password):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT user_id, name, email, monthly_income
        FROM users
        WHERE email = ?
          AND password = ?
    """, (email, password))

    user = cursor.fetchone()
    conn.close()
    return user


def add_expense(user_id, date_time, amount, description, category, monthly_income):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO expenses
        (
            user_id,
            date_time,
            amount,
            description,
            category,
            monthly_income
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        date_time,
        amount,
        description,
        category,
        monthly_income
    ))

    conn.commit()
    conn.close()


def get_user_expenses(user_id):
    conn = get_connection()

    query = """
        SELECT
            expense_id,
            user_id,
            date_time,
            amount,
            description,
            category,
            monthly_income
        FROM expenses
        WHERE user_id = ?
        ORDER BY date_time DESC
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(user_id,)
    )

    conn.close()
    return df


def get_user(user_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            user_id,
            name,
            email,
            monthly_income
        FROM users
        WHERE user_id = ?
    """, (user_id,))

    user = cursor.fetchone()
    conn.close()
    return user


if __name__ == "__main__":
    init_database()

    print("==============================================")
    print("      EXPENSE INTELLIGENCE DATABASE")
    print("==============================================")
    print("SQLite database initialized")
    print(f"Database: {DB_PATH}")
    print("Users table ready")
