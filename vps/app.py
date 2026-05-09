"""Simple Flask app for API key validation and Pin Payments webhook handling."""

import os
import uuid
import sqlite3
import secrets
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
DB_PATH = Path("/var/www/portal/keys.db")
MODELS_DIR = Path("/var/www/models")

# Pin Payments configuration
PIN_API_KEY = os.environ.get("PIN_API_KEY", "your_pin_api_key_here")
PIN_CURRENCY = "AUD"
PIN_AMOUNT = 2900  # $29.00 in cents

def get_db():
    """Get SQLite database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the API keys database."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS keys (
            id INTEGER PRIMARY KEY,
            api_key TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()


@app.before_first_request
def setup():
    init_db()


@app.route("/api/validate", methods=["GET"])
def validate_key():
    """Validate API key for nginx auth_request."""
    key = request.headers.get("X-API-Key") or request.args.get("key")
    if not key:
        return "", 403

    conn = get_db()
    row = conn.execute(
        "SELECT is_active, expires_at FROM keys WHERE api_key = ?",
        (key,)
    ).fetch_one()
    conn.close()

    if not row or not row["is_active"]:
        return "", 403

    if row["expires_at"] and row["expires_at"] < sqlite3.datetime.now():
        return "", 403

    # Key is valid - pass it through for logging
    resp = jsonify({})
    resp.headers["X-API-Key"] = key
    return resp, 200


@app.route("/api/webhook/pin", methods=["POST"])
def pin_webhook():
    """Handle Pin Payments webhook."""
    import json
    import requests
    from datetime import datetime, timedelta

    data = request.get_json()

    # Verify webhook signature (simplified - add proper verification in production)
    # Pin Payments sends a signature header for verification

    if data.get("event_type") != "charge_succeeded":
        return jsonify({"status": "ignored"}), 200

    # Extract customer email from charge data
    charge = data.get("data", {})
    email = charge.get("email") or charge.get("card", {}).get("email")
    if not email:
        return jsonify({"error": "No email"}), 400

    # Generate API key
    api_key = secrets.token_urlsafe(32)

    # Store in database (1 year expiry)
    expires = datetime.now() + timedelta(days=365)

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO keys (api_key, email, expires_at) VALUES (?, ?, ?)",
            (api_key, email, expires)
        )
        conn.commit()
    finally:
        conn.close()

    # Send email to customer with their API key
    send_key_email(email, api_key, expires)

    return jsonify({
        "api_key": api_key,
        "expires_at": expires.isoformat(),
        "message": f"API key sent to {email}"
    }), 201


def send_key_email(email, api_key, expires):
    """Send API key to customer via email."""
    # In production, use a proper email service (SendGrid, AWS SES, etc.)
    # For now, just print (admin can manually email)
    print(f"\n=== NEW API KEY GENERATED ===")
    print(f"To: {email}")
    print(f"API Key: {api_key}")
    print(f"Expires: {expires.strftime('%Y-%m-%d')}")
    print(f"Download command: python vps_downloader.py --key={api_key}")
    print("=== END ===\n")


@app.route("/api/generate-test-key", methods=["GET"])
def generate_test_key():
    """Generate a test key (protected, for testing only)."""
    if not app.debug:
        return "", 403

    test_key = secrets.token_urlsafe(32)
    conn = get_db()
    conn.execute(
        "INSERT INTO keys (api_key, email) VALUES (?, ?)",
        (test_key, "test@example.com")
    )
    conn.commit()
    conn.close()

    return jsonify({"api_key": test_key}), 201


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
