"""
PayMate Webhook Receiver — Real-Time Payment Reconciliation

This is a small, standalone Flask server that listens for Razorpay webhook
events. When a payment link is paid, Razorpay sends an event here INSTANTLY
(instead of PayMate having to poll and ask). Every incoming webhook's
signature is verified using HMAC-SHA256 against your webhook secret before
being trusted — this prevents anyone from forging a fake "payment succeeded"
event.

This runs SEPARATELY from the main Streamlit app (app.py), but writes to the
same SQLite database, so updates made here are immediately visible in PayMate.

Setup:
    pip install flask

    # Get a webhook secret by creating a webhook in the Razorpay Dashboard:
    # Settings > Webhooks > Add New Webhook
    # Set the Secret to any string you choose, and use that same value below.
    set RAZORPAY_WEBHOOK_SECRET=your_chosen_secret

    python webhook_server.py

Then expose this local server to the internet (Razorpay needs a public URL
to send webhooks to). The easiest free way is ngrok:
    1. Download ngrok from https://ngrok.com/download
    2. Run: ngrok http 5000
    3. Copy the https:// URL it gives you (e.g. https://abcd1234.ngrok-free.app)
    4. In the Razorpay Dashboard webhook settings, set the Webhook URL to:
       https://abcd1234.ngrok-free.app/razorpay-webhook
    5. Subscribe to the "payment_link.paid" event
    6. Save — Razorpay will now notify this server instantly when a test
       payment link is paid.
"""

import os
import hmac
import hashlib
import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)

DB_PATH = "paymate.db"
WEBHOOK_SECRET = os.environ.get("RAZORPAY_WEBHOOK_SECRET")


def verify_signature(payload_bytes: bytes, received_signature: str) -> bool:
    """Verify the webhook actually came from Razorpay using HMAC-SHA256.

    Without this check, anyone who discovers this URL could send a fake
    "payment succeeded" event and trick the system into marking an unpaid
    invoice as paid. This is the single most important security control
    on any webhook endpoint.
    """
    if not WEBHOOK_SECRET or not received_signature:
        return False
    expected_signature = hmac.new(
        WEBHOOK_SECRET.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    # Using compare_digest instead of == prevents timing attacks that could
    # let an attacker guess the correct signature one character at a time.
    return hmac.compare_digest(expected_signature, received_signature)


@app.route("/razorpay-webhook", methods=["POST"])
def razorpay_webhook():
    raw_payload = request.get_data()
    signature = request.headers.get("X-Razorpay-Signature", "")

    if not verify_signature(raw_payload, signature):
        print("⚠️  Rejected webhook: invalid or missing signature")
        return jsonify({"status": "invalid signature"}), 400

    event = request.get_json(silent=True) or {}
    event_type = event.get("event", "")
    print(f"✅ Verified webhook received: {event_type}")

    if event_type == "payment_link.paid":
        try:
            link_id = event["payload"]["payment_link"]["entity"]["id"]
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                "UPDATE invoices SET status = 'paid' WHERE payment_link_id = ?",
                (link_id,),
            )
            conn.commit()
            conn.close()
            print(f"💰 Invoice with payment link {link_id} marked as PAID instantly")
        except Exception as e:
            print(f"Error updating database: {e}")

    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET"])
def health_check():
    return "PayMate webhook receiver is running. POST events to /razorpay-webhook", 200


if __name__ == "__main__":
    if not WEBHOOK_SECRET:
        print("WARNING: RAZORPAY_WEBHOOK_SECRET is not set. All webhooks will be rejected.")
    print("Starting PayMate webhook receiver on http://localhost:5000")
    app.run(port=5000)
