"""
B2B Logistics Manager Dashboard
Predicts CO2 emissions and transport cost from pre-trained ML models,
allows route selection, and emails the selected route details to a driver.

⚠️  PROTOTYPE NOTE
Experimental prototype. Results are indicative, not validated for real-world use.
"""

import logging
import smtplib
import ssl
import warnings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
_log = logging.getLogger(__name__)

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="B2B Logistics Dashboard",
    page_icon="🚚",
    layout="centered",
)

# ── Load models (cached so they are only unpickled once per session) ──────────
MODEL_DIR = Path(__file__).parent


@st.cache_resource
def load_models():
    """Load and return the pre-trained CO2 and cost models."""
    errors = []
    co2_model = cost_model = None

    co2_path = MODEL_DIR / "co2_model.pkl"
    cost_path = MODEL_DIR / "cost_model.pkl"

    if not co2_path.exists():
        errors.append(f"Model file not found: {co2_path}")
    else:
        try:
            co2_model = joblib.load(co2_path)
        except Exception as exc:
            errors.append(f"Failed to load co2_model.pkl: {exc}")

    if not cost_path.exists():
        errors.append(f"Model file not found: {cost_path}")
    else:
        try:
            cost_model = joblib.load(cost_path)
        except Exception as exc:
            errors.append(f"Failed to load cost_model.pkl: {exc}")

    return co2_model, cost_model, errors


co2_model, cost_model, _load_errors = load_models()

# Show any model-loading problems at the top of the page
if _load_errors:
    for _err in _load_errors:
        st.error(f"❌ {_err}")
    st.stop()

# ── Route definitions ─────────────────────────────────────────────────────────
# Distance multipliers are heuristic prototype assumptions only — the training
# data does not encode named routes.  Predictions for different routes differ
# because of the distance multiplier, not because of real route characteristics.
#
# traffic_level is an INTERNAL model input only — it is not shown to the user.
# The trained LinearRegression models require this feature (coef ≠ 0), so a
# fixed per-route default is passed silently.  These values are purely synthetic
# defaults inherited from the training data; they carry no real-world meaning.
#   Route 1 – Highway   : traffic_level = 1  (internal default)
#   Route 2 – Expressway: traffic_level = 2  (internal default)
#   Route 3 – Shortcuts : traffic_level = 3  (internal default)

ROUTES = {
    "Route 1 – Highway": {
        "description": "Direct highway route using the base distance.",
        "distance_multiplier": 1.00,
        "distance_note": "Base distance (no adjustment)",
        "_internal_traffic_level": 1,   # model input only — not displayed to users
    },
    "Route 2 – Expressway": {
        "description": (
            "Urban expressway adding ~15 % to the base distance "
            "due to interchanges and controlled-access ramps."
        ),
        "distance_multiplier": 1.15,
        "distance_note": "Distance × 1.15 (expressway detour)",
        "_internal_traffic_level": 2,   # model input only — not displayed to users
    },
    "Route 3 – Shortcuts": {
        "description": (
            "Local shortcuts reducing the base distance by ~10 %."
        ),
        "distance_multiplier": 0.90,
        "distance_note": "Distance × 0.90 (shortcut saving)",
        "_internal_traffic_level": 3,   # model input only — not displayed to users
    },
}

# ── Email helper ──────────────────────────────────────────────────────────────

def send_route_email(
    sender_email: str,
    app_password: str,
    driver_email: str,
    route_name: str,
    route_info: dict,
    weight_kg: float,
    weight_tons: float,
    distance_km: float,
    effective_distance_km: float,
    co2_kg: float,
    cost_inr: float,
) -> None:
    """Send route-assignment email to the driver via Gmail SMTP."""
    subject = f"🚚 Route Assignment: {route_name}"

    body_html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #1f2328;">
        <h2 style="color: #2563eb;">Route Assignment — B2B Logistics</h2>
        <p>Hello Driver,</p>
        <p>Your shipment details and assigned route are below.
           Please review before departure.</p>
        <p style="color:#b45309; font-size:0.9em;">
            ⚠️ <strong>Experimental prototype.</strong>
            Results are indicative, not validated for real-world use.
        </p>

        <table style="border-collapse:collapse; width:100%; max-width:560px;">
            <tr style="background:#f0f4ff;">
                <td style="padding:8px 12px; font-weight:bold;">Assigned Route</td>
                <td style="padding:8px 12px;">{route_name}</td>
            </tr>
            <tr>
                <td style="padding:8px 12px; font-weight:bold;">Description</td>
                <td style="padding:8px 12px;">{route_info['description']}</td>
            </tr>
            <tr style="background:#f0f4ff;">
                <td style="padding:8px 12px; font-weight:bold;">Payload Weight</td>
                <td style="padding:8px 12px;">
                    {weight_kg:,.0f} kg ({weight_tons:.3f} metric tons)
                </td>
            </tr>
            <tr>
                <td style="padding:8px 12px; font-weight:bold;">Base Distance</td>
                <td style="padding:8px 12px;">{distance_km:.1f} km</td>
            </tr>
            <tr style="background:#f0f4ff;">
                <td style="padding:8px 12px; font-weight:bold;">Effective Distance</td>
                <td style="padding:8px 12px;">
                    {effective_distance_km:.1f} km
                    ({route_info['distance_note']})
                </td>
            </tr>
            <tr>
                <td style="padding:8px 12px; font-weight:bold;">
                    Predicted CO₂ Emissions
                </td>
                <td style="padding:8px 12px;">{co2_kg:.2f} kg (prototype)</td>
            </tr>
            <tr style="background:#f0f4ff;">
                <td style="padding:8px 12px; font-weight:bold;">
                    Predicted Transport Cost
                </td>
                <td style="padding:8px 12px;">₹{cost_inr:,.2f} (prototype)</td>
            </tr>
        </table>

        <p style="margin-top:20px;">Safe travels,<br>
        <strong>B2B Logistics Operations</strong></p>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = driver_email
    msg.attach(MIMEText(body_html, "html"))

    ssl_ctx = ssl.create_default_context()   # verifies Gmail's certificate

    _log.debug("SMTP: opening connection to smtp.gmail.com:587")
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
        _log.debug("SMTP: sending EHLO")
        server.ehlo()

        _log.debug("SMTP: issuing STARTTLS")
        server.starttls(context=ssl_ctx)      # explicit context — required on Python 3.12

        _log.debug("SMTP: re-sending EHLO after TLS upgrade")
        server.ehlo()

        # Log every AUTH mechanism Gmail advertised post-TLS.
        # If AUTH LOGIN / AUTH PLAIN are missing, the account is blocking
        # programmatic authentication and an App Password will not work.
        esmtp_features = server.esmtp_features  # dict populated by the last ehlo()
        _log.debug("SMTP: server ESMTP features after TLS: %s", list(esmtp_features.keys()))
        auth_mechanisms = esmtp_features.get("auth", "")
        _log.debug("SMTP: advertised AUTH mechanisms: '%s'", auth_mechanisms)
        if not auth_mechanisms:
            _log.error(
                "SMTP: server did not advertise any AUTH mechanism after STARTTLS. "
                "This account may have programmatic login blocked. "
                "Verify that a Gmail App Password exists and 2-Step Verification is ON."
            )

        _log.debug("SMTP: logging in (credentials not logged)")
        try:
            server.login(sender_email, app_password)
        except smtplib.SMTPAuthenticationError as auth_exc:
            _log.error(
                "SMTP: login failed — code=%s msg=%s",
                auth_exc.smtp_code,
                auth_exc.smtp_error,
            )
            raise
        except smtplib.SMTPException as smtp_exc:
            # Catches hangs that resolve as a generic SMTPException rather than
            # a clean SMTPAuthenticationError (e.g. timeout inside AUTH exchange)
            _log.error("SMTP: login raised %s: %s", type(smtp_exc).__name__, smtp_exc)
            raise

        _log.debug("SMTP: login succeeded")

        _log.debug("SMTP: sending message")
        server.sendmail(sender_email, driver_email, msg.as_string())
        _log.debug("SMTP: sendmail returned — message accepted by Gmail")


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🚚 B2B Logistics Manager Dashboard")
st.markdown(
    "Enter shipment details to get prototype CO₂ and cost estimates, "
    "then assign a route and notify the driver."
)
st.warning(
    "⚠️ **Experimental prototype.** "
    "Results are indicative, not validated for real-world use.",
    icon="⚠️",
)

st.divider()

# ── Section 1 : Shipment Inputs ───────────────────────────────────────────────
st.subheader("📦 Shipment Details")

col1, col2 = st.columns(2)
with col1:
    weight_kg = st.number_input(
        "Payload Weight (kg)",
        min_value=100.0,
        max_value=50_000.0,
        value=5_000.0,
        step=100.0,
        help="Total payload weight of the shipment in kilograms.",
    )
with col2:
    distance_km = st.number_input(
        "Distance (km)",
        min_value=1.0,
        max_value=5_000.0,
        value=200.0,
        step=10.0,
        help="Total one-way distance of the shipment in kilometres.",
    )

# Convert kg → metric tons for the model
weight_tons = weight_kg / 1_000.0

st.caption(f"Payload converted to metric tons: **{weight_tons:.3f} t**")

# ── Section 2 : Route Selection ───────────────────────────────────────────────
st.subheader("🗺️ Route Selection")

st.info(
    "**Route assumptions (heuristic, prototype only):** "
    "Highway = base distance × 1.00. "
    "Expressway = base × 1.15 (interchange detour). "
    "Shortcuts = base × 0.90 (shorter path). "
    "These multipliers are synthetic assumptions — the training data does not "
    "encode real route characteristics.",
    icon="ℹ️",
)

route_name = st.selectbox(
    "Choose a route",
    options=list(ROUTES.keys()),
    help="Each route applies a different distance multiplier to the base distance.",
)

selected_route = ROUTES[route_name]
effective_distance_km = distance_km * selected_route["distance_multiplier"]

# traffic_level is required by both trained models (coef ≠ 0) but is not
# exposed to the user.  The per-route internal default is used silently.
traffic_level = selected_route["_internal_traffic_level"]

with st.expander("Route details", expanded=True):
    rc1, rc2 = st.columns(2)
    rc1.metric("Base Distance", f"{distance_km:.1f} km")
    rc2.metric("Effective Distance", f"{effective_distance_km:.1f} km")
    st.caption(
        f"{selected_route['description']}  \n"
        f"_Distance note: {selected_route['distance_note']}_"
    )

# ── Section 3 : Predictions ───────────────────────────────────────────────────
st.subheader("📊 Predictions")

predict_btn = st.button("🔍 Predict Emissions & Cost", type="primary", use_container_width=True)

# Run the model ONLY when the button was clicked in this rerun.
# Changing an input widget triggers a rerun but predict_btn will be False,
# so last_prediction is never overwritten until the user explicitly clicks.
if predict_btn:
    # Build feature DataFrame matching the trained column names and order:
    # distance_km | payload_tons | traffic_level
    X = pd.DataFrame(
        [[effective_distance_km, weight_tons, traffic_level]],
        columns=["distance_km", "payload_tons", "traffic_level"],
    )

    try:
        co2_kg = float(co2_model.predict(X)[0])
        cost_inr = float(cost_model.predict(X)[0])

        # Clamp to physically plausible floor
        co2_kg = max(co2_kg, 0.0)
        cost_inr = max(cost_inr, 0.0)

        # Persist prediction in session state so it survives future reruns.
        # All inputs are stored so the email staleness check can compare them
        # against whatever the UI shows at send-time.
        st.session_state["last_prediction"] = {
            "co2_kg": co2_kg,
            "cost_inr": cost_inr,
            "weight_kg": weight_kg,
            "weight_tons": weight_tons,
            "distance_km": distance_km,
            "effective_distance_km": effective_distance_km,
            "route_name": route_name,
        }

    except Exception as exc:
        st.error(f"❌ Prediction failed: {exc}")
        st.stop()

# Display section is separate from the calculation block above.
# It reads whatever is in session state (frozen at the last button click)
# and is unaffected by subsequent input changes.
if "last_prediction" in st.session_state:
    p = st.session_state["last_prediction"]

    pred_col1, pred_col2 = st.columns(2)
    pred_col1.metric(
        label="🌿 CO₂ Emissions (prototype)",
        value=f"{p['co2_kg']:.2f} kg",
        help="Estimated carbon dioxide emissions for this shipment.",
    )
    pred_col2.metric(
        label="💰 Transport Cost (prototype)",
        value=f"₹{p['cost_inr']:,.2f}",
        help="Estimated total transport cost in Indian Rupees.",
    )

    st.caption(
        f"Inputs used — effective distance: **{p['effective_distance_km']:.1f} km**, "
        f"payload: **{p['weight_tons']:.3f} t**, "
        f"route: **{p['route_name']}**"
    )
else:
    st.info("ℹ️ Enter the parameters above and click **Predict Emissions & Cost** to calculate results.")

st.divider()

# ── Section 4 : Email Configuration ──────────────────────────────────────────
st.subheader("📧 Send Route to Driver")

# ── Pending-clear flag ────────────────────────────────────────────────────────
# Streamlit forbids writing to a widget's session-state key after the widget
# has been instantiated in the current rerun.  The safe pattern is to check
# for a plain (non-widget) flag BEFORE the widgets are created, derive the
# initial values from it, then clear the flag.  The widgets are then rendered
# with the correct (empty) values without any post-instantiation writes.
_clear_creds = st.session_state.pop("_clear_credentials", False)

with st.expander("Email Settings", expanded=False):
    st.markdown(
        "> **Gmail App Password required.** "
        "[Generate one here](https://myaccount.google.com/apppasswords) — "
        "2-Step Verification must be enabled on your Gmail account.  \n\n"
        "> 🔒 Your credentials are used to send this email and are not saved "
        "to files or persistent storage. They may remain temporarily in memory "
        "while your browser session is active."
    )
    sender_email = st.text_input(
        "Manager Gmail Address",
        placeholder="manager@gmail.com",
        value="" if _clear_creds else st.session_state.get("sender_email", ""),
        key="sender_email",
    )
    app_password = st.text_input(
        "Gmail App Password",
        type="password",
        placeholder="xxxx xxxx xxxx xxxx",
        value="" if _clear_creds else st.session_state.get("app_password", ""),
        key="app_password",
    )
    driver_email = st.text_input(
        "Driver Email Address", placeholder="driver@example.com"
    )

# Show success banner persisted from the previous send.
if "email_success" in st.session_state:
    st.success(st.session_state.pop("email_success"))

send_btn = st.button(
    "📤 Send Route to Driver", use_container_width=True, type="secondary"
)

if send_btn:
    # Require a prediction to exist before allowing email
    if "last_prediction" not in st.session_state:
        st.error("❌ Run a prediction first before sending the email.")
    else:
        p = st.session_state["last_prediction"]

        # ── Staleness guard ───────────────────────────────────────────────────
        # Check that the current UI selections match what was predicted.
        # If the user changed route, distance, or weight after the last
        # prediction, the stored results are stale and must not be sent.
        stale_reasons = []
        if p["route_name"] != route_name:
            stale_reasons.append(
                f"route changed from **{p['route_name']}** to **{route_name}**"
            )
        if abs(p["distance_km"] - distance_km) > 0.01:
            stale_reasons.append(
                f"distance changed from **{p['distance_km']:.1f} km** to **{distance_km:.1f} km**"
            )
        if abs(p["weight_kg"] - weight_kg) > 0.01:
            stale_reasons.append(
                f"weight changed from **{p['weight_kg']:,.0f} kg** to **{weight_kg:,.0f} kg**"
            )

        if stale_reasons:
            st.warning(
                "⚠️ The current inputs no longer match the last prediction. "
                "Please click **Predict Emissions & Cost** again before sending the email.  \n"
                "Changed: " + "; ".join(stale_reasons) + "."
            )
        else:
            missing = []
            if not sender_email:
                missing.append("Manager Gmail Address")
            if not app_password:
                missing.append("Gmail App Password")
            if not driver_email:
                missing.append("Driver Email Address")

            if missing:
                st.error(f"❌ Please fill in: {', '.join(missing)}")
            else:
                with st.spinner("Sending email…"):
                    try:
                        send_route_email(
                            sender_email=sender_email.strip(),
                            app_password=app_password.strip(),
                            driver_email=driver_email.strip(),
                            route_name=p["route_name"],
                            route_info=ROUTES[p["route_name"]],
                            weight_kg=p["weight_kg"],
                            weight_tons=p["weight_tons"],
                            distance_km=p["distance_km"],
                            effective_distance_km=p["effective_distance_km"],
                            co2_kg=p["co2_kg"],
                            cost_inr=p["cost_inr"],
                        )
                        # Set the pending-clear flag and save the success
                        # message, then rerun.  On the next pass the flag is
                        # read BEFORE the widgets are instantiated (above), so
                        # no StreamlitWidgetAlreadyInstantiatedError is raised.
                        st.session_state["email_success"] = (
                            f"✅ Route details sent to **{driver_email.strip()}** successfully!"
                        )
                        st.session_state["_clear_credentials"] = True
                        st.rerun()
                    except smtplib.SMTPAuthenticationError as auth_exc:
                        # smtp_code is e.g. 535; smtp_error is the server's text (safe to show)
                        code = auth_exc.smtp_code
                        detail = auth_exc.smtp_error.decode(errors="replace") if isinstance(
                            auth_exc.smtp_error, bytes
                        ) else str(auth_exc.smtp_error)
                        st.error(
                            f"❌ Gmail authentication failed (SMTP {code}).  \n"
                            f"Server said: `{detail}`  \n\n"
                            "**Common causes:**  \n"
                            "- You used your regular Gmail password instead of an **App Password**.  \n"
                            "- 2-Step Verification is not enabled on the Gmail account.  \n"
                            "- The App Password was revoked or typed incorrectly.  \n"
                            "- The Gmail account has \"Less secure app access\" blocked and no App "
                            "Password was created."
                        )
                    except smtplib.SMTPConnectError as exc:
                        st.error(
                            f"❌ Could not connect to smtp.gmail.com:587.  "
                            f"Check your internet connection or firewall.  Details: {exc}"
                        )
                    except smtplib.SMTPServerDisconnected as exc:
                        st.error(
                            f"❌ Server disconnected unexpectedly.  "
                            f"This usually means the STARTTLS handshake was refused.  "
                            f"Details: {exc}"
                        )
                    except smtplib.SMTPException as exc:
                        st.error(f"❌ SMTP error: {exc}")
                    except OSError as exc:
                        st.error(
                            f"❌ Network error (timeout or DNS failure): {exc}"
                        )
                    except Exception as exc:
                        st.error(f"❌ Unexpected error: {type(exc).__name__}: {exc}")

st.divider()
st.caption(
    "B2B Logistics Operations Dashboard · "
    "Powered by Streamlit & scikit-learn · "
    "⚠️ Experimental prototype — results are indicative only"
)
