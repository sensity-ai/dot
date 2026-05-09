#!/usr/bin/env python3
"""Generic subscription portal with BTCPay and PayID support.

Run:
    python vps_portal/server.py --host 127.0.0.1 --port 7861
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import tempfile
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


PORTAL_DIR = Path(__file__).resolve().parent
ROOT = PORTAL_DIR.parent
INDEX_FILE = PORTAL_DIR / "index.html"
KEYS_FILE = PORTAL_DIR / "keys.json"
SUBSCRIPTIONS_FILE = PORTAL_DIR / "subscriptions.json"
AUDIT_FILE = PORTAL_DIR / "audit_log.jsonl"
ALLOWED_DOWNLOADS = {
    "bundle.tar.gz": ROOT / "dot_models_bundle.tar.gz",
    "saved_models/simswap/checkpoints/512/550000_net_G.pth": ROOT
    / "saved_models"
    / "simswap"
    / "checkpoints"
    / "512"
    / "550000_net_G.pth",
    "saved_models/simswap/arcface_model/arcface_checkpoint.tar": ROOT
    / "saved_models"
    / "simswap"
    / "arcface_model"
    / "arcface_checkpoint.tar",
    "saved_models/simswap/parsing_model/checkpoint/79999_iter.pth": ROOT
    / "saved_models"
    / "simswap"
    / "parsing_model"
    / "checkpoint"
    / "79999_iter.pth",
    "saved_models/onnx/inswapper_128_fp16.onnx": ROOT
    / "saved_models"
    / "onnx"
    / "inswapper_128_fp16.onnx",
}

DEFAULT_CURRENCY = os.environ.get("PLAN_CURRENCY", "AUD")
PLANS = {
    "monthly": {
        "name": "Monthly",
        "price": os.environ.get("PLAN_MONTHLY_PRICE", "29.00"),
        "currency": DEFAULT_CURRENCY,
        "days": 30,
    },
    "yearly": {
        "name": "Yearly",
        "price": os.environ.get("PLAN_YEARLY_PRICE", "290.00"),
        "currency": DEFAULT_CURRENCY,
        "days": 365,
    },
}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return now_utc().isoformat()


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=2, sort_keys=True, default=json_default)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def append_audit(event: str, payload: dict[str, Any]) -> None:
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": iso_now(), "event": event, **payload}
    with AUDIT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=json_default) + "\n")


def load_keys() -> dict[str, Any]:
    return read_json(KEYS_FILE, {})


def save_keys(keys: dict[str, Any]) -> None:
    write_json(KEYS_FILE, keys)


def load_subscriptions() -> dict[str, Any]:
    return read_json(SUBSCRIPTIONS_FILE, {"customers": {}, "invoices": {}})


def save_subscriptions(data: dict[str, Any]) -> None:
    data.setdefault("customers", {})
    data.setdefault("invoices", {})
    write_json(SUBSCRIPTIONS_FILE, data)


def clean_email(email: str) -> str:
    return email.strip().lower()


def generate_key(prefix: str = "DOT") -> str:
    part = lambda: secrets.token_hex(4).upper()
    return f"{prefix}-{part()}-{part()}-{part()}"


def invoice_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(8).upper()}"


def is_expired(expires_at: str | None) -> bool:
    expiry = parse_time(expires_at)
    return bool(expiry and expiry <= now_utc())


def verify_key_record(record: dict[str, Any] | None) -> bool:
    if not record or not record.get("active", True):
        return False
    return not is_expired(record.get("expires_at"))


def verify_key(key: str) -> bool:
    return verify_key_record(load_keys().get(key))


def get_plan(plan_id: str) -> dict[str, Any]:
    if plan_id not in PLANS:
        raise ValueError(f"Unknown plan: {plan_id}")
    return PLANS[plan_id]


def ensure_customer(data: dict[str, Any], email: str) -> dict[str, Any]:
    customers = data.setdefault("customers", {})
    customer = customers.setdefault(
        email,
        {
            "email": email,
            "status": "pending",
            "created_at": iso_now(),
            "expires_at": None,
            "key": None,
            "renewal_cancelled": False,
        },
    )
    customer.setdefault("email", email)
    customer.setdefault("created_at", iso_now())
    customer.setdefault("renewal_cancelled", False)
    return customer


def extend_expiry(current_expiry: str | None, days: int) -> str:
    current = parse_time(current_expiry)
    base = current if current and current > now_utc() else now_utc()
    return (base + timedelta(days=days)).isoformat()


def issue_or_extend_key(email: str, plan_id: str, source: str) -> dict[str, Any]:
    data = load_subscriptions()
    keys = load_keys()
    plan = get_plan(plan_id)
    customer = ensure_customer(data, email)

    expires_at = extend_expiry(customer.get("expires_at"), int(plan["days"]))
    key = customer.get("key")
    if not key or key not in keys:
        key = generate_key()

    keys[key] = {
        **keys.get(key, {}),
        "active": True,
        "created": keys.get(key, {}).get("created", iso_now()),
        "email": email,
        "expires_at": expires_at,
        "plan": plan_id,
    }

    customer.update(
        {
            "status": "active",
            "expires_at": expires_at,
            "key": key,
            "plan": plan_id,
            "renewal_cancelled": False,
            "updated_at": iso_now(),
        }
    )

    save_keys(keys)
    save_subscriptions(data)
    append_audit("key_issued", {"email": email, "plan": plan_id, "source": source})
    return {"key": key, "expires_at": expires_at}


def public_subscription(customer: dict[str, Any] | None) -> dict[str, Any]:
    if not customer:
        return {"found": False, "active": False}
    active = customer.get("status") == "active" and not is_expired(
        customer.get("expires_at")
    )
    return {
        "found": True,
        "active": active,
        "status": "expired" if customer.get("status") == "active" and not active else customer.get("status"),
        "email": customer.get("email"),
        "plan": customer.get("plan"),
        "expires_at": customer.get("expires_at"),
        "key": customer.get("key") if active else None,
        "renewal_cancelled": customer.get("renewal_cancelled", False),
    }


def require_json(body: bytes) -> dict[str, Any]:
    if not body:
        return {}
    try:
        value = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON body") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON body must be an object")
    return value


def env_required(*names: str) -> list[str]:
    return [name for name in names if not os.environ.get(name)]


def create_btcpay_invoice(email: str, plan_id: str, local_invoice_id: str) -> dict[str, Any]:
    missing = env_required(
        "BTCPAY_BASE_URL", "BTCPAY_STORE_ID", "BTCPAY_API_KEY", "PORTAL_BASE_URL"
    )
    if missing:
        raise RuntimeError(f"BTCPay is not configured: missing {', '.join(missing)}")

    plan = get_plan(plan_id)
    base_url = os.environ["BTCPAY_BASE_URL"].rstrip("/")
    store_id = os.environ["BTCPAY_STORE_ID"]
    portal_base = os.environ["PORTAL_BASE_URL"].rstrip("/")
    url = f"{base_url}/api/v1/stores/{store_id}/invoices"
    payload = {
        "amount": str(plan["price"]),
        "currency": plan["currency"],
        "metadata": {
            "orderId": local_invoice_id,
            "buyerEmail": email,
            "posData": {"plan": plan_id, "email": email},
        },
        "checkout": {
            "redirectURL": f"{portal_base}/?invoice={local_invoice_id}",
        },
    }
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"token {os.environ['BTCPAY_API_KEY']}",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"BTCPay invoice error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"BTCPay connection error: {exc.reason}") from exc


def verify_btcpay_signature(body: bytes, signature: str | None) -> bool:
    secret = os.environ.get("BTCPAY_WEBHOOK_SECRET")
    if not secret or not signature:
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    expected_header = f"sha256={expected}"
    return hmac.compare_digest(signature, expected_header) or hmac.compare_digest(
        signature, expected
    )


class PortalHandler(BaseHTTPRequestHandler):
    server_version = "SubscriptionPortal/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[portal] {self.address_string()} {fmt % args}")

    def read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length) if length else b""

    def send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_index(self) -> None:
        body = INDEX_FILE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            self.send_index()
            return

        if parsed.path == "/api/verify":
            key = params.get("key", [""])[0].strip()
            record = load_keys().get(key)
            if verify_key_record(record):
                self.send_json(
                    {
                        "valid": True,
                        "key": key,
                        "email": record.get("email"),
                        "expires_at": record.get("expires_at"),
                    }
                )
            else:
                self.send_json({"valid": False, "error": "invalid or expired key"}, 403)
            return

        if parsed.path == "/api/download":
            self.handle_download(params)
            return

        if parsed.path == "/api/status":
            data = load_subscriptions()
            keys = load_keys()
            customers = data.get("customers", {})
            invoices = data.get("invoices", {})
            active_keys = sum(1 for record in keys.values() if verify_key_record(record))
            active_subs = sum(
                1
                for customer in customers.values()
                if public_subscription(customer).get("active")
            )
            self.send_json(
                {
                    "active_keys": active_keys,
                    "active_subscriptions": active_subs,
                    "customers": len(customers),
                    "invoices": len(invoices),
                    "plans": PLANS,
                }
            )
            return

        if parsed.path == "/api/subscription/status":
            email = clean_email(params.get("email", [""])[0])
            key = params.get("key", [""])[0].strip()
            data = load_subscriptions()
            customer = None
            if email:
                customer = data.get("customers", {}).get(email)
            elif key:
                key_record = load_keys().get(key)
                if key_record:
                    customer = data.get("customers", {}).get(key_record.get("email"))
            self.send_json(public_subscription(customer))
            return

        self.send_json({"error": "not found"}, 404)

    def handle_download(self, params: dict[str, list[str]]) -> None:
        key = params.get("key", [""])[0].strip()
        requested = params.get("file", [""])[0].strip()
        if not verify_key(key):
            self.send_json({"error": "invalid or expired key"}, 403)
            return
        if requested not in ALLOWED_DOWNLOADS:
            self.send_json({"error": "file not available"}, 404)
            return

        path = ALLOWED_DOWNLOADS[requested]
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(ROOT.resolve())
        except (FileNotFoundError, ValueError):
            self.send_json({"error": "file not found"}, 404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(resolved.stat().st_size))
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="{resolved.name}"',
        )
        self.end_headers()
        try:
            with resolved.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            append_audit("download_aborted", {"file": requested})

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            return
        if parsed.path.startswith("/api/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        body = self.read_body()
        parsed = urlparse(self.path)

        try:
            if parsed.path == "/api/checkout/create":
                self.handle_checkout_create(body)
                return
            if parsed.path == "/api/payid/submit":
                self.handle_payid_submit(body)
                return
            if parsed.path == "/api/admin/payid/approve":
                self.handle_payid_approve(body)
                return
            if parsed.path == "/api/subscription/cancel":
                self.handle_cancel(body)
                return
            if parsed.path == "/api/webhooks/btcpay":
                self.handle_btcpay_webhook(body)
                return
            self.send_json({"error": "not found"}, 404)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, 400)
        except RuntimeError as exc:
            self.send_json({"error": str(exc)}, 502)
        except Exception as exc:
            append_audit("server_error", {"path": parsed.path, "error": str(exc)})
            self.send_json({"error": "internal server error"}, 500)

    def handle_checkout_create(self, body: bytes) -> None:
        payload = require_json(body)
        email = clean_email(str(payload.get("email", "")))
        plan_id = str(payload.get("plan", "monthly")).strip().lower()
        provider = str(payload.get("provider", "btcpay")).strip().lower()
        if not email or "@" not in email:
            raise ValueError("A valid email is required")
        plan = get_plan(plan_id)
        if provider not in {"btcpay", "payid"}:
            raise ValueError("Provider must be btcpay or payid")

        data = load_subscriptions()
        ensure_customer(data, email)
        local_id = invoice_id("INV")
        invoice = {
            "id": local_id,
            "email": email,
            "plan": plan_id,
            "provider": provider,
            "amount": plan["price"],
            "currency": plan["currency"],
            "status": "pending",
            "created_at": iso_now(),
        }

        response: dict[str, Any] = {
            "invoice_id": local_id,
            "provider": provider,
            "status": "pending",
            "amount": plan["price"],
            "currency": plan["currency"],
        }

        if provider == "btcpay":
            btcpay_invoice = create_btcpay_invoice(email, plan_id, local_id)
            invoice.update(
                {
                    "external_id": btcpay_invoice.get("id"),
                    "checkout_url": btcpay_invoice.get("checkoutLink"),
                    "btcpay_status": btcpay_invoice.get("status"),
                }
            )
            response.update(
                {
                    "checkout_url": btcpay_invoice.get("checkoutLink"),
                    "external_id": btcpay_invoice.get("id"),
                }
            )
        else:
            reference = f"{os.environ.get('PAYID_REFERENCE_PREFIX', 'SUB')}-{local_id}"
            invoice["payid_reference"] = reference
            response["payid"] = {
                "name": os.environ.get("PAYID_NAME", "Configured account name"),
                "address": os.environ.get("PAYID_ADDRESS", ""),
                "reference": reference,
            }

        data.setdefault("invoices", {})[local_id] = invoice
        save_subscriptions(data)
        append_audit("checkout_created", {"email": email, "plan": plan_id, "provider": provider, "invoice_id": local_id})
        self.send_json(response)

    def handle_payid_submit(self, body: bytes) -> None:
        payload = require_json(body)
        invoice_id_value = str(payload.get("invoice_id", "")).strip()
        reference = str(payload.get("reference", "")).strip()
        payer_name = str(payload.get("payer_name", "")).strip()
        if not invoice_id_value or not reference:
            raise ValueError("invoice_id and reference are required")

        data = load_subscriptions()
        invoice = data.get("invoices", {}).get(invoice_id_value)
        if not invoice or invoice.get("provider") != "payid":
            raise ValueError("PayID invoice not found")
        invoice.update(
            {
                "status": "payment_submitted",
                "submitted_reference": reference,
                "payer_name": payer_name,
                "submitted_at": iso_now(),
            }
        )
        save_subscriptions(data)
        append_audit("payid_submitted", {"invoice_id": invoice_id_value, "email": invoice.get("email")})
        self.send_json({"ok": True, "status": "payment_submitted"})

    def handle_payid_approve(self, body: bytes) -> None:
        token = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        expected = os.environ.get("PORTAL_ADMIN_TOKEN")
        if not expected or not hmac.compare_digest(token, expected):
            self.send_json({"error": "admin token required"}, 403)
            return

        payload = require_json(body)
        invoice_id_value = str(payload.get("invoice_id", "")).strip()
        if not invoice_id_value:
            raise ValueError("invoice_id is required")

        data = load_subscriptions()
        invoice = data.get("invoices", {}).get(invoice_id_value)
        if not invoice or invoice.get("provider") != "payid":
            raise ValueError("PayID invoice not found")

        invoice.update({"status": "paid", "approved_at": iso_now()})
        save_subscriptions(data)
        key_info = issue_or_extend_key(invoice["email"], invoice["plan"], "payid")
        append_audit("payid_approved", {"invoice_id": invoice_id_value, "email": invoice.get("email")})
        self.send_json({"ok": True, "status": "active", **key_info})

    def handle_cancel(self, body: bytes) -> None:
        payload = require_json(body)
        email = clean_email(str(payload.get("email", "")))
        key = str(payload.get("key", "")).strip()
        data = load_subscriptions()
        keys = load_keys()

        if not email and key in keys:
            email = keys[key].get("email", "")
        if not email:
            raise ValueError("email or key is required")

        customer = data.get("customers", {}).get(email)
        if not customer:
            raise ValueError("subscription not found")
        customer["renewal_cancelled"] = True
        customer["status"] = "cancelled"
        customer["updated_at"] = iso_now()
        if customer.get("key") in keys:
            keys[customer["key"]]["active"] = False
        save_subscriptions(data)
        save_keys(keys)
        append_audit("subscription_cancelled", {"email": email})
        self.send_json({"ok": True, "status": "cancelled"})

    def handle_btcpay_webhook(self, body: bytes) -> None:
        signature = self.headers.get("BTCPay-Sig") or self.headers.get("X-BTCPay-Sig")
        if not verify_btcpay_signature(body, signature):
            self.send_json({"error": "invalid webhook signature"}, 401)
            return

        payload = require_json(body)
        event_type = str(payload.get("type", ""))
        external_id = str(payload.get("invoiceId") or payload.get("id") or "")
        data = load_subscriptions()
        invoice = None
        for item in data.get("invoices", {}).values():
            if item.get("external_id") == external_id:
                invoice = item
                break
        if not invoice:
            append_audit("btcpay_webhook_unmatched", {"external_id": external_id, "type": event_type})
            self.send_json({"ok": True, "matched": False})
            return

        invoice["btcpay_event"] = event_type
        invoice["webhook_at"] = iso_now()
        invoice["webhook_payload"] = payload
        paid_events = {"InvoiceSettled", "InvoiceProcessing", "InvoicePaymentSettled"}
        invalid_events = {"InvoiceExpired", "InvoiceInvalid"}
        response: dict[str, Any] = {"ok": True, "matched": True, "status": invoice.get("status")}

        if event_type in paid_events:
            invoice["status"] = "paid"
            save_subscriptions(data)
            key_info = issue_or_extend_key(invoice["email"], invoice["plan"], "btcpay")
            response.update({"status": "active", **key_info})
        elif event_type in invalid_events:
            invoice["status"] = "expired" if event_type == "InvoiceExpired" else "invalid"
            save_subscriptions(data)
            response["status"] = invoice["status"]
        else:
            save_subscriptions(data)

        append_audit("btcpay_webhook", {"external_id": external_id, "type": event_type, "invoice_id": invoice.get("id")})
        self.send_json(response)


def create_manual_key(email: str, plan: str) -> str:
    email = clean_email(email or f"manual-{secrets.token_hex(4)}@local")
    key_info = issue_or_extend_key(email, plan, "manual")
    return key_info["key"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic subscription portal")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--create-key", action="store_true")
    parser.add_argument("--email", default="")
    parser.add_argument("--plan", choices=sorted(PLANS), default="monthly")
    args = parser.parse_args()

    if args.create_key:
        print(create_manual_key(args.email, args.plan))
        return

    if not INDEX_FILE.exists():
        raise SystemExit(f"Missing {INDEX_FILE}")

    KEYS_FILE.touch(exist_ok=True)
    if not KEYS_FILE.read_text().strip():
        save_keys({})
    if not SUBSCRIPTIONS_FILE.exists():
        save_subscriptions({"customers": {}, "invoices": {}})

    print(f"[portal] Serving on http://{args.host}:{args.port}")
    print("[portal] Payment methods: btcpay, payid")
    server = ThreadingHTTPServer((args.host, args.port), PortalHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[portal] Stopped.")


if __name__ == "__main__":
    main()
