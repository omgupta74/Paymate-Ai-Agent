"""
PayMate — A Fintech AI Agent built with Groq (free, open-source LLM) + Streamlit

Built for the Razorpay AI Builder Internship.

What it does (an AI agent that reasons about which tool to use):
- calculate: basic math
- calculate_gst: compute GST-inclusive/exclusive amounts at any rate
- convert_currency: live currency conversion (free API, no key needed)
- categorize_expense: classify a spend description into a category
- add_expense / list_expenses / remove_expense: persistent expense tracker (SQLite)
- generate_invoice: format a simple invoice with GST applied

Run:
    pip install -r requirements.txt
    set GROQ_API_KEY=your_key   (Windows)  /  export GROQ_API_KEY=your_key (Mac/Linux)
    streamlit run app.py
"""

import os
import ast
import json
import sqlite3
import operator
import hashlib
import requests
import streamlit as st
from datetime import datetime
from groq import Groq

DB_PATH = "paymate.db"
MODEL_NAME = "openai/gpt-oss-120b"
IDEMPOTENCY_WINDOW_SECONDS = 300  # 5 minutes — duplicate identical requests within this window reuse the existing link

# ---------- Database setup ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_name TEXT NOT NULL,
            items TEXT NOT NULL,
            subtotal REAL NOT NULL,
            gst_amount REAL NOT NULL,
            total REAL NOT NULL,
            payment_link_id TEXT,
            payment_url TEXT,
            status TEXT NOT NULL DEFAULT 'created',
            created_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS idempotency_cache (
            idempotency_key TEXT PRIMARY KEY,
            payment_link_id TEXT NOT NULL,
            payment_url TEXT NOT NULL,
            response_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    conn.commit()
    conn.close()

def calculate(expression: str) -> str:
    """Safely evaluate a basic math expression without eval()."""
    allowed_ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return allowed_ops[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as e:
        return f"Error evaluating expression: {e}"


def calculate_gst(amount: float, rate: float = 18, mode: str = "exclusive") -> str:
    """Calculate GST. mode='exclusive' means amount doesn't include GST yet.
    mode='inclusive' means amount already includes GST and we back it out."""
    try:
        if mode == "inclusive":
            base = amount / (1 + rate / 100)
            gst_amount = amount - base
            return (f"On an inclusive amount of Rs.{amount:.2f} at {rate}% GST: "
                    f"base price = Rs.{base:.2f}, GST = Rs.{gst_amount:.2f}")
        else:
            gst_amount = amount * (rate / 100)
            total = amount + gst_amount
            return (f"On Rs.{amount:.2f} at {rate}% GST: "
                    f"GST amount = Rs.{gst_amount:.2f}, total payable = Rs.{total:.2f}")
    except Exception as e:
        return f"Error calculating GST: {e}"


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency using Frankfurter (free, no API key needed)."""
    try:
        resp = requests.get(
            "https://api.frankfurter.app/latest",
            params={"amount": amount, "from": from_currency.upper(), "to": to_currency.upper()},
            timeout=10,
        ).json()
        rate_result = resp["rates"].get(to_currency.upper())
        if rate_result is None:
            return f"Could not convert {from_currency} to {to_currency}."
        return f"{amount} {from_currency.upper()} = {rate_result:.2f} {to_currency.upper()}"
    except Exception as e:
        return f"Error converting currency: {e}"


CATEGORY_KEYWORDS = {
    "Food": ["swiggy", "zomato", "restaurant", "food", "cafe", "coffee", "lunch", "dinner"],
    "Travel": ["uber", "ola", "flight", "train", "taxi", "fuel", "petrol", "travel"],
    "Shopping": ["amazon", "flipkart", "myntra", "shopping", "mall", "clothes"],
    "Bills & Utilities": ["electricity", "recharge", "bill", "wifi", "internet", "rent"],
    "Entertainment": ["movie", "netflix", "spotify", "subscription", "game"],
}


def categorize_expense(description: str) -> str:
    """Deterministic keyword-based categorization (fast, no LLM call needed)."""
    desc_lower = description.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return category
    return "Others"


def add_expense(description: str, amount: float) -> str:
    category = categorize_expense(description)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO expenses (description, amount, category, created_at) VALUES (?, ?, ?, ?)",
        (description, amount, category, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return f"Added expense: '{description}' — Rs.{amount:.2f} (auto-categorized as {category})"


def list_expenses() -> str:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT description, amount, category FROM expenses ORDER BY id").fetchall()
    conn.close()
    if not rows:
        return "No expenses recorded yet."
    total = sum(r[1] for r in rows)
    lines = [f"{i+1}. {r[0]} — Rs.{r[1]:.2f} [{r[2]}]" for i, r in enumerate(rows)]
    lines.append(f"\nTotal spent: Rs.{total:.2f}")
    return "\n".join(lines)


def remove_expense(description: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT id FROM expenses WHERE description LIKE ? LIMIT 1", (f"%{description}%",))
    row = cur.fetchone()
    if row:
        conn.execute("DELETE FROM expenses WHERE id = ?", (row[0],))
        conn.commit()
        conn.close()
        return f"Removed expense matching '{description}'."
    conn.close()
    return f"No expense found matching '{description}'."


def generate_invoice(client_name: str, items: str, gst_rate: float = 18,
                      customer_contact: str = "", customer_email: str = "") -> str:
    """items format: 'item1:price1,item2:price2'
    Also creates a matching Razorpay payment link (if keys are set) and logs
    the invoice for reconciliation, and sends it via WhatsApp if a contact
    number and Twilio keys are available."""
    try:
        line_items = []
        subtotal = 0.0
        for pair in items.split(","):
            name, price = pair.split(":")
            price = float(price.strip())
            line_items.append((name.strip(), price))
            subtotal += price

        gst_amount = subtotal * (gst_rate / 100)
        total = subtotal + gst_amount

        lines = [f"INVOICE for {client_name}", "-" * 30]
        for name, price in line_items:
            lines.append(f"{name:<20} Rs.{price:.2f}")
        lines.append("-" * 30)
        lines.append(f"{'Subtotal':<20} Rs.{subtotal:.2f}")
        lines.append(f"{'GST (' + str(gst_rate) + '%)':<20} Rs.{gst_amount:.2f}")
        lines.append(f"{'TOTAL':<20} Rs.{total:.2f}")
        invoice_text = "\n".join(lines)

        payment_link_id = None
        payment_url = None
        whatsapp_note = ""

        auth = _razorpay_auth()
        if auth:
            link_result = create_payment_link(
                total, f"Invoice for {client_name}", client_name,
                customer_contact, customer_email
            )
            if "Payment link created successfully" in link_result:
                for line in link_result.split("\n"):
                    if line.startswith("Link ID:"):
                        payment_link_id = line.replace("Link ID:", "").strip()
                    if line.startswith("URL:"):
                        payment_url = line.replace("URL:", "").strip()
                invoice_text += f"\n\nPayment link: {payment_url}"

                if customer_contact and _twilio_auth():
                    wa_result = send_whatsapp_message(
                        customer_contact,
                        f"Hi {client_name}, here is your invoice of Rs.{total:.2f}. "
                        f"Pay here: {payment_url}"
                    )
                    whatsapp_note = f"\n{wa_result}"
            else:
                invoice_text += f"\n\n(Payment link could not be created: {link_result})"

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO invoices (client_name, items, subtotal, gst_amount, total, "
            "payment_link_id, payment_url, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (client_name, items, subtotal, gst_amount, total, payment_link_id,
             payment_url, "created", datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

        return invoice_text + whatsapp_note
    except Exception as e:
        return f"Error generating invoice: {e}"


def list_invoices() -> str:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, client_name, total, status, payment_link_id FROM invoices ORDER BY id"
    ).fetchall()
    conn.close()
    if not rows:
        return "No invoices recorded yet."
    lines = []
    for inv_id, client, total, status, link_id in rows:
        lines.append(f"#{inv_id} — {client}: Rs.{total:.2f} [{status}]")
    return "\n".join(lines)


def sync_invoice_statuses() -> str:
    """Check Razorpay for the live status of every invoice's payment link
    and update the local database — this is the reconciliation step."""
    auth = _razorpay_auth()
    if not auth:
        return "Razorpay API keys are not set, cannot sync statuses."

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, payment_link_id, client_name FROM invoices WHERE payment_link_id IS NOT NULL"
    ).fetchall()

    updated = []
    for inv_id, link_id, client in rows:
        try:
            resp = requests.get(
                f"https://api.razorpay.com/v1/payment_links/{link_id}",
                auth=auth, timeout=15,
            )
            data = resp.json()
            new_status = data.get("status", "unknown")
            conn.execute("UPDATE invoices SET status = ? WHERE id = ?", (new_status, inv_id))
            updated.append(f"#{inv_id} {client}: {new_status}")
        except Exception:
            continue

    conn.commit()
    conn.close()

    if not updated:
        return "No invoices with payment links to sync."
    return "Reconciliation complete:\n" + "\n".join(updated)


# ---------- Razorpay Payment Links API (Test Mode) ----------

def _razorpay_auth():
    key_id = os.environ.get("RAZORPAY_KEY_ID")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        return None
    return (key_id, key_secret)


def create_payment_link(amount: float, description: str, customer_name: str = "",
                         customer_contact: str = "", customer_email: str = "") -> str:
    """Create a real Razorpay payment link (Test Mode) via the Payment Links API.

    Idempotency protection: an AI agent can accidentally call this twice for
    the same request (e.g. a retried tool call, or the model re-attempting
    after a slow response). Without protection, that means creating TWO
    separate payment links for the same charge — a customer could be asked
    to pay twice, or two live links exist for one invoice. This function
    builds a deterministic key from the request details and checks a cache
    before hitting Razorpay's API, so a duplicate request returns the
    ORIGINAL link instead of creating a new one.
    """
    auth = _razorpay_auth()
    if not auth:
        return ("Razorpay API keys are not set. Please set RAZORPAY_KEY_ID and "
                "RAZORPAY_KEY_SECRET environment variables (Test Mode keys from "
                "the Razorpay Dashboard).")

    idempotency_key = hashlib.sha256(
        f"{round(amount, 2)}|{description.strip().lower()}|{customer_contact.strip()}"
        .encode("utf-8")
    ).hexdigest()

    conn = sqlite3.connect(DB_PATH)
    cached = conn.execute(
        "SELECT response_text, created_at FROM idempotency_cache WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()

    if cached:
        cached_response, cached_at = cached
        cached_time = datetime.fromisoformat(cached_at)
        if (datetime.now() - cached_time).total_seconds() < IDEMPOTENCY_WINDOW_SECONDS:
            conn.close()
            return cached_response + "\n(Note: identical request detected within the last few minutes — returned the existing link instead of creating a duplicate.)"
    conn.close()

    try:
        payload = {
            "amount": int(round(amount * 100)),  # amount in paise
            "currency": "INR",
            "description": description,
            "customer": {
                "name": customer_name,
                "contact": customer_contact,
                "email": customer_email,
            },
            "notify": {"sms": bool(customer_contact), "email": bool(customer_email)},
            "reminder_enable": True,
        }
        resp = requests.post(
            "https://api.razorpay.com/v1/payment_links",
            json=payload,
            auth=auth,
            timeout=15,
        )
        data = resp.json()
        if resp.status_code >= 400:
            error_msg = data.get("error", {}).get("description", "Unknown error")
            return f"Razorpay error: {error_msg}"

        short_url = data.get("short_url")
        link_id = data.get("id")
        response_text = (f"Payment link created successfully!\n"
                          f"Link ID: {link_id}\n"
                          f"Amount: Rs.{amount:.2f}\n"
                          f"URL: {short_url}\n"
                          f"(This is a Test Mode link — no real money is charged)")

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO idempotency_cache "
            "(idempotency_key, payment_link_id, payment_url, response_text, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (idempotency_key, link_id, short_url, response_text, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

        return response_text
    except Exception as e:
        return f"Error creating payment link: {e}"


def check_payment_status(payment_link_id: str) -> str:
    """Check the status of a Razorpay payment link (Test Mode)."""
    auth = _razorpay_auth()
    if not auth:
        return "Razorpay API keys are not set."
    try:
        resp = requests.get(
            f"https://api.razorpay.com/v1/payment_links/{payment_link_id}",
            auth=auth,
            timeout=15,
        )
        data = resp.json()
        if resp.status_code >= 400:
            error_msg = data.get("error", {}).get("description", "Unknown error")
            return f"Razorpay error: {error_msg}"

        status = data.get("status")
        amount = data.get("amount", 0) / 100
        return f"Payment link {payment_link_id}: status = {status}, amount = Rs.{amount:.2f}"
    except Exception as e:
        return f"Error checking payment status: {e}"


# ---------- Twilio WhatsApp (Sandbox) ----------

def _twilio_auth():
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    if not sid or not token:
        return None
    return (sid, token)


def send_whatsapp_message(to_number: str, message: str) -> str:
    """Send a WhatsApp message via Twilio's free Sandbox.
    Note: the recipient must have joined the Twilio Sandbox first by sending
    the sandbox join code to the Twilio WhatsApp number."""
    auth = _twilio_auth()
    if not auth:
        return "Twilio keys not set — WhatsApp delivery skipped (payment link still created above)."

    sid = auth[0]
    from_number = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    to_number_clean = to_number.strip()
    if not to_number_clean.startswith("+"):
        to_number_clean = "+91" + to_number_clean.lstrip("0")

    try:
        resp = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
            data={
                "From": from_number,
                "To": f"whatsapp:{to_number_clean}",
                "Body": message,
            },
            auth=auth,
            timeout=15,
        )
        data = resp.json()
        if resp.status_code >= 300:
            error_msg = data.get("message", "Unknown Twilio error")
            return f"WhatsApp delivery failed: {error_msg} (recipient may need to join the Sandbox first)"
        return f"WhatsApp message sent to {to_number_clean}."
    except Exception as e:
        return f"Error sending WhatsApp message: {e}"


# ---------- Tool schema for the LLM ----------
TOOLS = [
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate a basic math expression",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "calculate_gst",
        "description": "Calculate GST on an amount, either exclusive (add GST) or inclusive (extract GST already included)",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"},
            "rate": {"type": "number", "description": "GST rate percent, default 18"},
            "mode": {"type": "string", "enum": ["exclusive", "inclusive"]},
        }, "required": ["amount"]}}},
    {"type": "function", "function": {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another using live rates",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"},
            "from_currency": {"type": "string", "description": "3-letter currency code, e.g. INR"},
            "to_currency": {"type": "string", "description": "3-letter currency code, e.g. USD"},
        }, "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {
        "name": "add_expense",
        "description": "Add a new expense; it will be auto-categorized and saved permanently",
        "parameters": {"type": "object", "properties": {
            "description": {"type": "string"},
            "amount": {"type": "number"},
        }, "required": ["description", "amount"]}}},
    {"type": "function", "function": {
        "name": "list_expenses",
        "description": "List all recorded expenses with total spend",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "remove_expense",
        "description": "Remove an expense by matching part of its description",
        "parameters": {"type": "object", "properties": {
            "description": {"type": "string"}}, "required": ["description"]}}},
    {"type": "function", "function": {
        "name": "generate_invoice",
        "description": "Generate a formatted invoice with GST for a client. Automatically creates a matching Razorpay payment link and, if a customer contact and Twilio are configured, sends it via WhatsApp. Logs the invoice for reconciliation tracking.",
        "parameters": {"type": "object", "properties": {
            "client_name": {"type": "string"},
            "items": {"type": "string", "description": "e.g. 'Design work:5000,Hosting:1200'"},
            "gst_rate": {"type": "number", "description": "default 18"},
            "customer_contact": {"type": "string", "description": "10-digit phone number, optional, enables WhatsApp delivery"},
            "customer_email": {"type": "string", "description": "optional"},
        }, "required": ["client_name", "items"]}}},
    {"type": "function", "function": {
        "name": "list_invoices",
        "description": "List all invoices with their client, total amount, and current payment status",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "sync_invoice_statuses",
        "description": "Check Razorpay for the live payment status of every invoice and update records — use this when the user asks 'who has paid' or 'reconcile my invoices' or 'check pending payments'",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "create_payment_link",
        "description": "Create a real Razorpay payment link (Test Mode) that a customer can use to pay, and send it via WhatsApp if a contact number and Twilio are configured. Use this when the user wants to collect payment from someone, e.g. 'send a payment link of 500 to Neha'.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number", "description": "Amount in Rupees"},
            "description": {"type": "string", "description": "What the payment is for"},
            "customer_name": {"type": "string"},
            "customer_contact": {"type": "string", "description": "10-digit phone number, optional"},
            "customer_email": {"type": "string", "description": "optional"},
        }, "required": ["amount", "description"]}}},
    {"type": "function", "function": {
        "name": "check_payment_status",
        "description": "Check the status of a previously created Razorpay payment link using its link ID",
        "parameters": {"type": "object", "properties": {
            "payment_link_id": {"type": "string"},
        }, "required": ["payment_link_id"]}}},
]

FUNCTION_MAP = {
    "calculate": lambda a: calculate(a["expression"]),
    "calculate_gst": lambda a: calculate_gst(a["amount"], a.get("rate", 18), a.get("mode", "exclusive")),
    "convert_currency": lambda a: convert_currency(a["amount"], a["from_currency"], a["to_currency"]),
    "add_expense": lambda a: add_expense(a["description"], a["amount"]),
    "list_expenses": lambda a: list_expenses(),
    "remove_expense": lambda a: remove_expense(a["description"]),
    "generate_invoice": lambda a: generate_invoice(
        a["client_name"], a["items"], a.get("gst_rate", 18),
        a.get("customer_contact", ""), a.get("customer_email", "")),
    "list_invoices": lambda a: list_invoices(),
    "sync_invoice_statuses": lambda a: sync_invoice_statuses(),
    "create_payment_link": lambda a: create_payment_link(
        a["amount"], a["description"], a.get("customer_name", ""),
        a.get("customer_contact", ""), a.get("customer_email", "")),
    "check_payment_status": lambda a: check_payment_status(a["payment_link_id"]),
}

SYSTEM_PROMPT = (
    "You are PayMate, a fintech assistant agent for freelancers and small businesses. "
    "You can track expenses, calculate GST, convert currencies, generate invoices that "
    "automatically create real Razorpay payment links (Test Mode) and send them via "
    "WhatsApp, and reconcile which invoices have been paid. Use the available tools "
    "whenever a request needs a calculation, GST, currency conversion, expense/invoice "
    "management, payment collection, or checking who has paid. Always mention currency "
    "amounts in Rupees (Rs.) unless the user specifies otherwise. Be concise and clear."
)


# ---------- Streamlit UI ----------
st.set_page_config(page_title="PayMate — AI Fintech Agent", page_icon="💸", layout="wide")

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .rzp-header-row {
        display: flex;
        align-items: center;
        gap: 18px;
        margin-bottom: 4px;
    }
    .rzp-logo-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 60px; height: 60px;
        background: linear-gradient(135deg, #3395FF, #0a3d91);
        border-radius: 16px;
        font-size: 1.9rem;
        box-shadow: 0 4px 14px rgba(51,149,255,0.35);
        flex-shrink: 0;
    }
    div[data-testid="stMarkdownContainer"] p.rzp-hero-title {
        font-size: 2.6rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -0.5px;
        color: #f2f5f8 !important;
        text-align: left;
        line-height: 1.2 !important;
    }
    div[data-testid="stMarkdownContainer"] p.rzp-hero-sub {
        color: #8ea3bb !important;
        font-size: 0.95rem !important;
        font-weight: 400 !important;
        margin-top: 8px !important;
        margin-bottom: 4px !important;
        text-align: left;
    }
    .rzp-pill {
        display: inline-block;
        background: rgba(51,149,255,0.15);
        color: #7fbcff;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 8px;
        border: 1px solid rgba(51,149,255,0.35);
    }

    div[data-testid="stMetric"] {
        background: #161f2e;
        border: 1px solid #263449;
        border-radius: 14px;
        padding: 18px 20px;
        border-top: 3px solid #3395FF;
    }
    div[data-testid="stMetricLabel"] { font-weight: 600; color: #cdd9e6; }
    div[data-testid="stMetricValue"] { color: #7fbcff; font-weight: 800; }

    .badge-paid {
        background: rgba(34,197,94,0.15); color: #4ade80; padding: 3px 12px;
        border-radius: 999px; font-size: 0.75rem; font-weight: 700;
        border: 1px solid rgba(34,197,94,0.3);
    }
    .badge-pending {
        background: rgba(251,191,36,0.15); color: #fbbf24; padding: 3px 12px;
        border-radius: 999px; font-size: 0.75rem; font-weight: 700;
        border: 1px solid rgba(251,191,36,0.3);
    }

    .stButton button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #3395FF;
        color: #7fbcff;
        background: rgba(51,149,255,0.1);
        transition: all 0.15s ease;
    }
    .stButton button:hover {
        background: #3395FF;
        color: #ffffff;
        border-color: #3395FF;
        transform: translateY(-1px);
    }

    section[data-testid="stSidebar"] .stButton button {
        background: #3395FF; color: #ffffff !important; border: none;
    }
    section[data-testid="stSidebar"] .stButton button:hover { background: #1f7ae0; }

    div[data-testid="stChatMessage"] {
        background: #161f2e;
        border: 1px solid #263449;
        border-radius: 14px;
        padding: 6px;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<p class="rzp-hero-title"><i class="fa-solid fa-wallet" style="color:#3395FF; margin-right:14px;"></i>PayMate — AI Fintech Assistant</p>
<p class="rzp-hero-sub">Built with Groq (open-source LLM) · Tool-calling AI agent ·
Real Razorpay Payment Links (Test Mode) · Persistent expense tracking</p>
""", unsafe_allow_html=True)
st.write("")

init_db()

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY environment variable not set. Set it in your terminal before running Streamlit.")
    st.stop()

client = Groq(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


def _load_dashboard_data():
    conn = sqlite3.connect(DB_PATH)
    exp_rows = conn.execute("SELECT description, amount, category FROM expenses ORDER BY id DESC").fetchall()
    inv_rows = conn.execute("SELECT client_name, total, status, payment_url FROM invoices ORDER BY id DESC").fetchall()
    conn.close()
    return exp_rows, inv_rows


exp_rows, inv_rows = _load_dashboard_data()
total_spent = sum(r[1] for r in exp_rows)
total_invoiced = sum(r[1] for r in inv_rows)
total_paid = sum(r[1] for r in inv_rows if r[2] == "paid")
total_pending = total_invoiced - total_paid

# ---------- Top dashboard metrics ----------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Invoiced", f"Rs.{total_invoiced:,.0f}")
m2.metric("Collected", f"Rs.{total_paid:,.0f}")
m3.metric("Pending", f"Rs.{total_pending:,.0f}")
m4.metric("Expenses", f"Rs.{total_spent:,.0f}")

st.write("")

# ---------- Quick action buttons ----------
st.markdown("**Quick Actions**")
qa1, qa2, qa3, qa4 = st.columns(4)
if qa1.button("New Payment Link", use_container_width=True):
    st.session_state.pending_prompt = "Create a payment link for "
if qa2.button("New Invoice", use_container_width=True):
    st.session_state.pending_prompt = "Generate an invoice for "
if qa3.button("Check Payments", use_container_width=True):
    st.session_state.pending_prompt = "Who has paid?"
if qa4.button("Show Expenses", use_container_width=True):
    st.session_state.pending_prompt = "Show my expenses"

st.write("")
st.divider()

# Sidebar
with st.sidebar:
    st.markdown(
        '<h3><i class="fa-solid fa-wallet" style="color:#3395FF; margin-right:8px;"></i>PayMate</h3>',
        unsafe_allow_html=True,
    )
    if st.button("Start New Conversation", use_container_width=True):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.rerun()

    st.divider()
    st.markdown(
        '<h4><i class="fa-solid fa-file-invoice" style="margin-right:8px;"></i>Recent Invoices</h4>',
        unsafe_allow_html=True,
    )
    if inv_rows:
        for inv_client, inv_total, inv_status, inv_url in inv_rows[:8]:
            badge_class = "badge-paid" if inv_status == "paid" else "badge-pending"
            st.markdown(
                f"**{inv_client}** — Rs.{inv_total:,.2f}  \n"
                f'<span class="{badge_class}">{inv_status.upper()}</span>',
                unsafe_allow_html=True,
            )
            st.write("")
    else:
        st.caption("No invoices yet.")

    st.divider()
    st.markdown(
        '<h4><i class="fa-solid fa-receipt" style="margin-right:8px;"></i>Recent Expenses</h4>',
        unsafe_allow_html=True,
    )
    if exp_rows:
        for desc, amt, cat in exp_rows[:8]:
            st.write(f"**{desc}** — Rs.{amt:.2f}  \n_{cat}_")
    else:
        st.caption("Try: 'Add expense: Swiggy order 450'")

    st.divider()
    st.markdown(
        '<h4><i class="fa-solid fa-plug" style="margin-right:8px;"></i>Integrations</h4>',
        unsafe_allow_html=True,
    )
    if os.environ.get("RAZORPAY_KEY_ID") and os.environ.get("RAZORPAY_KEY_SECRET"):
        st.success("Razorpay Test Mode ✓")
    else:
        st.warning("Razorpay keys not set")

    if os.environ.get("TWILIO_ACCOUNT_SID") and os.environ.get("TWILIO_AUTH_TOKEN"):
        st.success("Twilio WhatsApp ✓")
    else:
        st.info("Twilio not set (optional)")

# Display chat history (skip system message)
for msg in st.session_state.messages:
    if msg["role"] in ("user", "assistant") and msg.get("content"):
        avatar = "🧑‍💼" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])

col_input, col_mic = st.columns([12, 1])

with col_mic:
    with st.popover("🎤", use_container_width=True):
        st.caption("Record a voice command")
        audio_value = st.audio_input("Record", label_visibility="collapsed")
        if audio_value is not None:
            audio_bytes = audio_value.getvalue()
            audio_hash = hash(audio_bytes)
            if st.session_state.get("last_audio_hash") != audio_hash:
                st.session_state.last_audio_hash = audio_hash
                with st.spinner("Transcribing..."):
                    try:
                        transcription = client.audio.transcriptions.create(
                            file=("voice_command.wav", audio_bytes),
                            model="whisper-large-v3",
                        )
                        st.session_state.pending_prompt = transcription.text
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not transcribe audio: {e}")

with col_input:
    typed_input = st.chat_input("Ask PayMate anything — GST, currency, expenses, invoices...")
user_input = typed_input or st.session_state.pending_prompt
if st.session_state.pending_prompt and not typed_input:
    st.session_state.pending_prompt = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.write(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                final_reply = None
                max_iterations = 5

                for _ in range(max_iterations):
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=st.session_state.messages,
                        tools=TOOLS,
                        tool_choice="auto",
                    )
                    response_message = response.choices[0].message
                    tool_calls = response_message.tool_calls

                    if not tool_calls:
                        final_reply = response_message.content
                        st.session_state.messages.append({
                            "role": "assistant", "content": final_reply
                        })
                        break

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    })

                    for tool_call in tool_calls:
                        fn_name = tool_call.function.name
                        fn_args = json.loads(tool_call.function.arguments)
                        result = FUNCTION_MAP[fn_name](fn_args)
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": fn_name,
                            "content": result,
                        })

                if final_reply is None:
                    final_reply = "I completed several steps but ran out of room to summarize — check the results above."

                st.write(final_reply)
                st.rerun()

            except Exception as e:
                st.error(f"Something went wrong: {e}")
