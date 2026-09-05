"""
Deterministic, procedural case generator for FraudCourt.

Design goal: same seed -> byte-identical case, every time. This is what
makes the held-out batch reproducible for reviewers (no hand-picked demo
cases, no hidden randomness). Nothing in this module calls an LLM or any
external service - it must stay fully inspectable and non-gameable.
"""
from __future__ import annotations
import random
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional

FRAUD_TYPES = [
    "stolen_card",
    "friendly_fraud",       # card + account + customer are all real; only *intent* is disputed
    "account_takeover",
    "merchant_collusion",
    "refund_abuse",
]

CATEGORIES = ["electronics", "travel", "digital_goods", "food_delivery"]

DISPUTE_REASON_CODES = [
    "10.4 - Other Fraud, Card-Absent Environment",
    "13.1 - Merchandise/Services Not Received",
    "13.7 - Cancelled Merchandise/Services",
    "4853 - Cardholder Dispute",
    "12.6 - Duplicate Processing",
]

MERCHANT_NAMES = [
    "Kiraana Mart Online", "SkyFare Travels", "Byte Bazaar Electronics",
    "QuickPlate Delivery", "Nimbus Digital Store", "Sundara Fashion Hub",
]

CUSTOMER_FIRST = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Isha", "Karan", "Meera", "Arjun", "Divya"]
CUSTOMER_LAST = ["Sharma", "Iyer", "Patel", "Nair", "Reddy", "Gupta", "Menon", "Chauhan", "Kapoor", "Bose"]

# Each fraud_type has a signature set of evidence flags that, if revealed,
# should push a rational agent toward "fraud". Legitimate cases carry the
# same tool surface but return benign explanations instead.
EVIDENCE_TOOLS = [
    "check_device_fingerprint",
    "query_transaction_history",
    "verify_cardholder_identity",
    "check_velocity",
    "check_merchant_risk_category",
    "check_dispute_history",
]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _case_hash(seed: int) -> str:
    return hashlib.sha256(str(seed).encode()).hexdigest()[:10]


@dataclass
class Case:
    id: str
    seed: int
    fraud_type: str
    category: str
    dispute_reason_code: str
    amount: float
    currency: str
    customer_name: str
    merchant_name: str
    is_repeat_customer: bool
    visible: dict = field(default_factory=dict)
    hidden_ground_truth_label: str = "legitimate"  # "fraud" | "legitimate"
    hidden_evidence: dict = field(default_factory=dict)  # tool_name -> {"result": str, "signal": "fraud"|"benign"|"neutral"}

    def public_dict(self) -> dict:
        """What the agent is allowed to see at reset() time — no hidden fields."""
        return {
            "id": self.id,
            "category": self.category,
            "dispute_reason_code": self.dispute_reason_code,
            "amount": self.amount,
            "currency": self.currency,
            "customer_name": self.customer_name,
            "merchant_name": self.merchant_name,
            "is_repeat_customer": self.is_repeat_customer,
            "visible": self.visible,
            "available_tools": EVIDENCE_TOOLS + ["convene_debate_panel"],
            "terminal_actions": ["flag_fraud", "allow_transaction", "escalate_to_review"],
            "confidence_levels": ["HIGH", "MED", "LOW"],
        }


def _build_evidence(rng: random.Random, fraud_type: str, is_fraud: bool) -> dict:
    """
    Build the hidden evidence dict. Roughly half the tools point toward the
    true label, the rest are neutral/red-herring - this is what forces
    multi-step investigation instead of a single lucky tool call.
    """
    evidence = {}

    templates = {
        "check_device_fingerprint": {
            "fraud": "Device fingerprint does not match any of the last 12 months of prior sessions; new device, new IP block, no prior pairing.",
            "benign": "Device fingerprint matches a device seen on this account 6 times in the last 90 days.",
        },
        "query_transaction_history": {
            "fraud": "Three similar high-value transactions attempted on unrelated accounts from the same device in the last 48 hours.",
            "benign": "Transaction amount and merchant category are consistent with this customer's last 8 months of spending pattern.",
        },
        "verify_cardholder_identity": {
            "fraud": "OTP was approved from a session with a different registered mobile number than the one on file.",
            "benign": "OTP approved from the customer's registered mobile number, matching KYC records.",
        },
        "check_velocity": {
            "fraud": "5 transactions attempted within 90 seconds across different merchants, well above the account's historical velocity.",
            "benign": "Transaction velocity is within the customer's normal range (1-2 transactions/day).",
        },
        "check_merchant_risk_category": {
            "fraud": "Merchant category has a 3x higher-than-average chargeback rate over the trailing quarter.",
            "benign": "Merchant category has a below-average chargeback rate and is a well-established, verified merchant.",
        },
        "check_dispute_history": {
            "fraud": "Customer has filed 4 'item not received' disputes in the last year, all against different merchants.",
            "benign": "Customer has zero prior disputes in 3 years on file; this is a first-time complaint.",
        },
    }

    # friendly_fraud is the hard case on purpose: surface signals mostly look
    # legitimate (real device, real customer, real OTP) - only the dispute
    # pattern and the merchant's delivery-confirmation evidence hint at
    # disputed intent. This is exactly the case a rules engine cannot catch.
    if fraud_type == "friendly_fraud" and is_fraud:
        signal_bias = {
            "check_device_fingerprint": "benign",
            "query_transaction_history": "benign",
            "verify_cardholder_identity": "benign",
            "check_velocity": "benign",
            "check_merchant_risk_category": "neutral",
            "check_dispute_history": "fraud",  # the one real tell: repeat-dispute pattern
        }
    else:
        # generic bias: about half the tools point the right way, seeded per-case
        tools = EVIDENCE_TOOLS[:]
        rng.shuffle(tools)
        n_signal = rng.randint(3, 5) if is_fraud else rng.randint(0, 2)
        signal_tools = set(tools[:n_signal]) if is_fraud else set(tools[:n_signal])
        signal_bias = {}
        for t in EVIDENCE_TOOLS:
            if is_fraud:
                signal_bias[t] = "fraud" if t in signal_tools else "benign"
            else:
                signal_bias[t] = "benign" if t not in signal_tools else "fraud"
        # legitimate cases should almost never have >2 fraud-pointing tools
        if not is_fraud:
            fraud_count = sum(1 for v in signal_bias.values() if v == "fraud")
            if fraud_count > 2:
                # demote extras back to benign
                extras = [t for t, v in signal_bias.items() if v == "fraud"][2:]
                for t in extras:
                    signal_bias[t] = "benign"

    for tool in EVIDENCE_TOOLS:
        signal = signal_bias.get(tool, "neutral")
        result = templates[tool].get(signal, templates[tool]["benign"])
        evidence[tool] = {"result": result, "signal": signal}

    return evidence


def generate_case(seed: int, force_fraud_type: Optional[str] = None, force_label: Optional[str] = None) -> Case:
    """Deterministic: same seed (and same force_* args) -> identical Case."""
    rng = _rng(seed)

    fraud_type = force_fraud_type or rng.choice(FRAUD_TYPES)
    category = rng.choice(CATEGORIES)
    reason_code = rng.choice(DISPUTE_REASON_CODES)
    amount = round(rng.uniform(499, 84999), 2)
    customer_name = f"{rng.choice(CUSTOMER_FIRST)} {rng.choice(CUSTOMER_LAST)}"
    merchant_name = rng.choice(MERCHANT_NAMES)
    is_repeat = rng.random() < 0.6

    if force_label is not None:
        is_fraud = force_label == "fraud"
    else:
        is_fraud = rng.random() < 0.45  # slightly under 50% so "allow" isn't a free strategy

    evidence = _build_evidence(rng, fraud_type, is_fraud)

    case_id = f"case_{_case_hash(seed)}_{fraud_type}"

    visible = {
        "channel": rng.choice(["UPI", "card_not_present", "netbanking", "wallet"]),
        "transaction_time": f"2026-{rng.randint(1,9):02d}-{rng.randint(1,28):02d}T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:00Z",
        "customer_tenure_months": rng.randint(1, 60),
    }

    return Case(
        id=case_id,
        seed=seed,
        fraud_type=fraud_type,
        category=category,
        dispute_reason_code=reason_code,
        amount=amount,
        currency="INR",
        customer_name=customer_name,
        merchant_name=merchant_name,
        is_repeat_customer=is_repeat,
        visible=visible,
        hidden_ground_truth_label="fraud" if is_fraud else "legitimate",
        hidden_evidence=evidence,
    )


# --- Golden Trap Library -----------------------------------------------
# A small, hand-picked, seed-pinned suite designed to include cases that
# would fool a naive single-shot classifier (especially friendly-fraud
# cases where surface signals look clean). Run in CI on every commit.
GOLDEN_TRAP_SEEDS = [
    (101, "friendly_fraud", "fraud"),
    (102, "friendly_fraud", "legitimate"),
    (103, "stolen_card", "fraud"),
    (104, "stolen_card", "legitimate"),
    (105, "account_takeover", "fraud"),
    (106, "account_takeover", "legitimate"),
    (107, "merchant_collusion", "fraud"),
    (108, "merchant_collusion", "legitimate"),
    (109, "refund_abuse", "fraud"),
    (110, "refund_abuse", "legitimate"),
    (111, "friendly_fraud", "fraud"),
    (112, "friendly_fraud", "legitimate"),
    (113, "stolen_card", "fraud"),
    (114, "account_takeover", "fraud"),
    (115, "refund_abuse", "legitimate"),
]


def golden_trap_cases() -> list[Case]:
    return [generate_case(seed, force_fraud_type=ft, force_label=label) for seed, ft, label in GOLDEN_TRAP_SEEDS]
