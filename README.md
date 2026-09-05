# FraudCourt

**Calibrated fraud & chargeback adjudication — built for the "AI Risk Manager" track.**

A working environment where an AI agent investigates a suspicious transaction or chargeback
dispute, convenes an adversarial **Court Panel** (Prosecutor vs. Defender) when the evidence
conflicts, and must declare a **calibrated confidence level (HIGH / MED / LOW)** before every
terminal decision. The reward function punishes confident wrongness far more than uncertain
wrongness — this is a direct, measurable answer to the track's "honest metrics including
false-positive cost" bar.

> Domain-adapted from an insurance-claims calibration architecture (ClaimCourt). The mechanic —
> investigate → debate → declare a calibrated confidence, scored on a 3×2 matrix — is unchanged.
> What changed is the subject matter: payment fraud and chargeback disputes instead of insurance
> claims.

---

## 1. Overview

### The problem

Payments risk teams triage fraud/chargeback alerts using classifiers that output a bare score
(e.g. "0.83 fraud probability") with no guarantee that 0.83-scored cases are actually right 83%
of the time. That mismatch between *stated confidence* and *actual correctness* is the exact gap
between "our model has good accuracy" and "our model is safe to auto-act on." An overconfident
wrong flag on a legitimate customer burns trust and analyst time; a model that hedges everything
is useless. Starting in 2026, this stopped being a purely internal cost: RBI's proposed UPI fraud
compensation pilot puts a fixed liability share on the bank/PSP for a mishandled case — a
confidently-wrong decision now has a real, budgeted rupee cost attached to it, not just a vague
"trust" cost.

### The solution

FraudCourt doesn't just classify — it makes an agent **earn** its confidence. Every case is
decomposed into a multi-step investigation (partial observability — evidence is revealed
progressively, not handed over in one shot), an adversarial debate when the evidence is mixed,
and a mandatory confidence declaration scored against a matrix that makes overconfident
wrongness the single worst outcome. On a reproducible held-out batch, the system reports
precision, recall, **and** a calibration score — plus a rupee figure for the cost of
overconfidence, so "false-positive cost" is a number, not an adjective.

### Who it's for

A payments risk/fraud-ops team (in the hackathon framing, the actual end user is a risk
analyst) that needs to know not just "is this fraud" but "how much should I trust this call
before I act on it."

---

## 2. What's implemented (all real, none of it faked)

| Component | What it does | AI used? |
|---|---|---|
| **Case Generator** (`server/case_generator.py`) | Deterministic, seeded, procedural case generation: 5 fraud types × 4 categories × 5 dispute-reason codes. Same seed → byte-identical case, always. | No — deliberately deterministic/inspectable |
| **Investigative Tool Set** (`server/tools.py`) | 6 tools that progressively reveal hidden evidence (device fingerprint, transaction history, identity verification, velocity, merchant risk category, dispute history) | No |
| **Agent policy** (`server/agent.py`) | Chooses which tool to call next and, eventually, the terminal decision + confidence | Yes, when `GEMINI_API_KEY` is set — deterministic heuristic fallback otherwise |
| **Court Panel** (`server/court_panel.py`) | Prosecutor argues fraud, Defender argues legitimacy, Judge weighs both — from the *same* evidence base, no leakage | Yes / heuristic fallback |
| **Calibration Grader** (`server/calibration_grader.py`) | The 3×2 reward matrix. Fully deterministic — the scoring must be trustworthy and non-gameable | No, by design |
| **Anti-Gaming Detector** (`server/anti_gaming.py`) | Rolling-window watchdog: LOW-confidence rate > 70% over 10+ decisions triggers a progressive reward penalty, closing the "just always hedge" exploit | No |
| **Human-in-the-Loop Audit** (`server/audit_memory.py` + `/cases/{id}/audit_feedback`) | Every escalation + every HIGH-confidence fraud flag lands in an auditor queue. The override itself is never AI-mediated. | No (the decision); yes (the correction summary) |
| **Knowledge Base** (Tier A) | Structured key-match precedent lookup over an `evidence_signature` — explicitly *not* a vector store, so it's fully explainable | No (retrieval); yes (summarizing free-text reasons) |
| **Held-out batch eval** (`server/batch.py`, `scripts/run_batch.py`) | Precision, recall, calibration score, false-positive cost by tier, cost of overconfidence — written to committed JSON, no hand-edits | N/A |
| **Golden Trap Library** | 15 hand-authored, seed-pinned cases (including friendly-fraud traps a naive classifier would miss), run in CI | N/A |
| **Ablation harness** | Same held-out seeds run through a deliberately naive single-shot baseline vs. the full pipeline — an *empirical* answer to "why does this need this much AI" | N/A |
| **"Break It" button** | Live UI button that injects a real failure (simulated LLM timeout) into the running episode and shows the system recover in front of you | N/A |
| **Failures ledger** | Auto-generated postmortem (what broke / what we did / outcome) for every real or injected incident | No |

### Why FraudCourt over a plain classifier — the headline case

The hardest, most realistic case in this environment is **friendly fraud**: the card is real,
the account is real, the customer is real — only their *intent* is disputed. There's no
anomalous device fingerprint or velocity spike to catch; it's a credibility judgment call. A
rules engine structurally cannot help here. This is exactly the shape of problem the Court
Panel is built for, and it's the environment's default "showcase" task.

---

## 3. Architecture

```
Browser (frontend/index.html — single-file cockpit, no build step)
        │  fetch()
        ▼
FastAPI app (app/main.py)
        │
        ├── server/engine.py         orchestrates reset/step/terminal, one code path
        │       ├── server/case_generator.py   (deterministic)
        │       ├── server/tools.py             (deterministic)
        │       ├── server/agent.py             (LLM or heuristic)
        │       ├── server/court_panel.py       (LLM or heuristic)
        │       ├── server/calibration_grader.py (deterministic)
        │       └── server/anti_gaming.py       (deterministic)
        │
        ├── server/audit_memory.py    SQLite: corrections, knowledge base, failures ledger
        ├── server/batch.py           held-out batch + ablation runner
        └── server/llm_client.py      Gemini REST wrapper, raises LLMFailure on any problem
```

**Plain-language flow:** User → Dashboard → FastAPI → Case Generator (deterministic) → Agent
(LLM or heuristic) → Court Panel (LLM or heuristic, when evidence is mixed) → Calibration Grader
(deterministic) → SQLite (audit trail) → Dashboard → User.

**Where AI is used, and where it deliberately isn't:** the agent's tool-selection and terminal
decision, and the Court Panel's three roles, use an LLM when configured. The case generator, the
calibration grader, and the anti-gaming detector are plain, deterministic Python on purpose — the
scoring has to be trustworthy and non-gameable, so it's a lookup table, not a model call. This
split is documented, not incidental.

---

## 4. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI + uvicorn | Matches the OpenEnv scaffold pattern; fast to run, self-documenting via `/docs` |
| LLM | Google Gemini (REST, no SDK dependency) | Free tier is generous; swap `GEMINI_MODEL` for any Gemini model, or point `llm_client.py` at any OpenAI-compatible endpoint |
| Multi-agent orchestration | Plain Python + Pydantic-validated structured calls | The Court Panel is a fixed 3-step debate, not an open-ended agent graph — a framework (LangGraph/CrewAI/AutoGen) would be unnecessary weight |
| Storage | SQLite (stdlib) | Zero setup, durable audit trail; sufficient for this scope |
| Knowledge base | Structured key-match (Tier A) | Zero dependencies, fully explainable — no black-box vector search. A ChromaDB + `sentence-transformers` Tier-B upgrade is a natural next step (see §10) |
| Frontend | Single-file HTML/CSS/JS | Zero build step, demos reliably, served directly by FastAPI |
| Tests | pytest + FastAPI's `TestClient` | 35 tests covering determinism, the grader matrix, anti-gaming, the audit/KB loop, and full API flows |

---

## 5. Repository structure

```
fraudcourt/
├── app/
│   ├── main.py              FastAPI app: all endpoints
│   └── schemas.py           Pydantic request models
├── server/
│   ├── case_generator.py    deterministic procedural case generation + Golden Trap Library
│   ├── tools.py              investigative tool handlers
│   ├── agent.py               agent policy: LLM or heuristic fallback
│   ├── court_panel.py       Prosecutor / Defender / Judge
│   ├── naive_agent.py       single-shot baseline, used only for the ablation
│   ├── calibration_grader.py 3x2 matrix, deterministic
│   ├── anti_gaming.py       rolling-window LOW-rate watchdog
│   ├── audit_memory.py      SQLite: corrections, knowledge base, failures ledger
│   ├── engine.py             reset/step/terminal orchestration (shared by API + CLI)
│   ├── batch.py               held-out batch + ablation runner
│   ├── episode_store.py     in-memory active-episode store
│   ├── llm_client.py        Gemini REST wrapper
│   └── db.py                   SQLite connection/schema
├── frontend/
│   └── index.html            single-file cockpit dashboard
├── scripts/
│   └── run_batch.py          CLI: regenerate reports/*.json
├── tests/                     35 tests: determinism, grader, anti-gaming, audit/KB, full API flows
├── reports/                    committed batch reports (regenerate with scripts/run_batch.py)
├── .github/workflows/ci.yml   runs tests + verifies determinism + diffs reports against committed numbers
├── openenv.yaml                OpenEnv-style manifest
├── Dockerfile / docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 6. Installation & running locally

**Prerequisites:** Python 3.11+ (tested on 3.12).

```bash
git clone <this-repo-url>
cd fraudcourt

pip install -r requirements.txt

cp .env.example .env
# Optional: add GEMINI_API_KEY to .env for real LLM reasoning.
# The app runs fully without it, in heuristic "Demo Mode".

uvicorn app.main:app --reload --port 8000
```

Then open **http://localhost:8000** — the dashboard is served directly by the backend, no
separate frontend process or build step needed.

### Running with Docker

```bash
docker compose up --build
# then open http://localhost:8000
```

(Docker build was not executed in the environment this repository was authored in — no Docker
daemon was available there — but the Dockerfile is a standard single-stage Python image build and
was reviewed line-by-line for correctness against the tested `requirements.txt`/entrypoint.)

### Running the test suite

```bash
python -m pytest tests/ -v
```

35 tests, all passing as shipped (see `reports/` and CI for the exact numbers).

### Regenerating the held-out reports

```bash
python scripts/run_batch.py --n 50 --seed-start 1000        # random held-out batch
python scripts/run_batch.py --golden-trap                    # 15-case Golden Trap suite
python scripts/run_batch.py --ablation --n 40 --seed-start 2000   # naive vs. full pipeline
```

Each writes to `reports/*.json`. CI re-runs the Golden Trap suite on every push and fails the
build if the committed numbers don't match a fresh run (see `.github/workflows/ci.yml`) — this is
how "no hand-edits" is enforced, not just claimed.

---

## 7. Environment variables

| Variable | Required? | What it does | Where to get it |
|---|---|---|---|
| `GEMINI_API_KEY` | No | Enables real LLM reasoning for the agent and Court Panel. Without it, the app runs in deterministic **Demo Mode** (clearly labeled in the UI header) | Free tier: https://aistudio.google.com/apikey |
| `GEMINI_MODEL` | No | Which Gemini model to call | Defaults to `gemini-2.0-flash-lite`; check https://ai.google.dev/gemini-api/docs/models for current IDs |
| `LIABILITY_SHARE` | No | Fraction of transaction amount attributed as "cost of overconfidence" on wrongly-HIGH-confidence decisions. Defaults to `0.20`, mirroring RBI's proposed 20% bank-liability share — labeled as illustrative in the UI | — |
| `FRAUDCOURT_DB_PATH` | No | SQLite file location | Defaults to `./data/fraudcourt.db` |

**If `GEMINI_API_KEY` is missing:** every AI-driven component (agent decisions, Court Panel)
falls back to a deterministic, evidence-strength heuristic policy. Nothing crashes, nothing is
faked — the UI badge honestly says "Heuristic / demo mode," and the failures ledger does **not**
log this as an incident (a missing key isn't a failure; a real network error or malformed LLM
response is, and those *are* logged and recoverable — see the "Break It" button).

---

## 8. Demo flow (3 minutes)

1. **Open the problem** (15 sec): "Fraud classifiers report accuracy, not whether their
   confidence is honest. A model that's right 70% of the time can still be dangerously
   overconfident on its wrong calls — and since this year, that mistake has a real, budgeted
   rupee cost under RBI's proposed liability-sharing rules."
2. **Dashboard tab** — click **Run held-out batch**. Point at precision/recall, and specifically
   the **calibration score** and **cost of overconfidence** numbers — these are the two things a
   plain classifier's dashboard doesn't have.
3. **Live Case Runner tab** — select **friendly_fraud_showcase**, click **Run episode**. Narrate
   the investigation trace as it appears: real device, real customer, real OTP — then the one
   tool that actually matters (`check_dispute_history`) surfaces the pattern. Evidence is mixed →
   **Court Panel fires** (the wow moment): Prosecutor and Defender argue from the same evidence,
   Judge renders a verdict. Point at the 3×2 matrix with the scored cell highlighted.
4. **Press "Break it"** — inject a live failure into the running episode and show the system
   recover instead of crashing or silently guessing. Flip to the **Failures Ledger** tab to show
   the auto-generated postmortem.
5. **Audit Queue tab** — show a pending escalated case, submit an auditor override with a reason.
   Flip to **Knowledge Base** to show the correction now indexed by evidence signature.
6. **Close on the ablation** — run **naive vs. full pipeline** on the Dashboard tab. This is the
   empirical answer to "why does this need all this machinery": the same held-out cases, scored
   both ways, with a real calibration-score delta and a real rupee delta.

---

## 9. Evaluation criteria mapping

| Criterion | What we do | Where to look |
|---|---|---|
| **Problem taste** | Calibration/false-positive-cost gap in fraud triage, narrowed to the friendly-fraud case a rules engine structurally cannot catch, tied to RBI's real proposed 20% bank-liability share | `friendly_fraud_showcase` task; `LIABILITY_SHARE` cost calculator on the dashboard |
| **Build quality** | Runs end-to-end from a fresh clone; 35 passing tests; CI re-verifies determinism and diffs committed reports against a fresh run on every push; one-command Docker Compose | `tests/`, `.github/workflows/ci.yml`, `docker-compose.yml` |
| **AI judgment** | Explicit AI/non-AI split (see §2 table); an empirical ablation (naive single-shot vs. full pipeline) instead of an architectural assertion; LLM calls are validated and rejected if malformed rather than trusted blindly | `server/naive_agent.py`, `/run_ablation`, `agent._validate_llm_action` |
| **Failure recovery** | Deterministic anti-gaming detector; malformed LLM output rejected → heuristic fallback for that turn; live judge-triggered "Break It" button with an auto-generated postmortem; real human-override path with a logged reason | `server/anti_gaming.py`, `/debug/inject_failure`, `server/audit_memory.py` |

---

## 10. Known limitations (genuine, not hedging)

- **Episodes are in-memory, not persisted.** An active (not-yet-terminal) episode does not survive
  a server restart. Terminal artifacts — corrections, failures, batch reports — are durable in
  SQLite. Documented, not hidden.
- **Heuristic fallback is intentionally simple.** It's an evidence-count policy, not a learned
  model — it can still miss a single strong signal on hard friendly-fraud cases (this is visible
  and honest in the shipped `reports/golden_trap_summary.json` recall number). Configuring
  `GEMINI_API_KEY` gives materially better reasoning quality, especially on the friendly-fraud
  case, since an LLM can weigh evidence *quality* over raw count.
- **Knowledge Base is Tier A (structured key-match), not semantic retrieval.** This is a
  deliberate choice (explainability over sophistication), not a shortcut we ran out of time to
  fix — see the code comment in `server/audit_memory.py`. A ChromaDB + `sentence-transformers`
  Tier-B upgrade is a natural, low-risk next step if semantic matching over free-text reasons
  becomes valuable.
- **No fine-tuning / GRPO training run.** This environment is built to be *evaluated* zero-shot
  or few-shot against any LLM backend (or the deterministic heuristic), not to ship a trained
  checkpoint. Reporting an honest zero-shot/heuristic calibration number was chosen over claiming
  a training lift that wasn't actually reproduced in the time available.
- **No auth.** This is a judge-facing/internal eval tool for the hackathon, not a
  multi-tenant merchant-facing product — explicitly out of scope, and it is **strictly
  defense-only**: it adjudicates and scores, it never auto-executes a financial action.
- **Docker build was reviewed but not executed** in the authoring environment (no Docker daemon
  available there). The image is a standard single-stage Python 3.12-slim build against the same
  `requirements.txt` that was actually installed and tested.

---

## 11. Future scope

- Tier-B semantic knowledge base (ChromaDB + local `sentence-transformers` embeddings — both
  free, no hosted service required)
- Persist active episodes to SQLite instead of in-memory, for restart-safety
- A real fine-tuning pass (GRPO or otherwise) against the heuristic-vs-LLM ablation baseline
  established here
- Multi-provider backend-swap test (Gemini vs. a local Ollama model) to further evidence the
  "interchangeable component, not vendor-locked glue code" AI-judgment claim

---

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError` on startup | Dependencies not installed | `pip install -r requirements.txt` |
| Dashboard header says "Heuristic / demo mode" | No `GEMINI_API_KEY` set | Expected — the app is fully functional in this mode. Add the key to `.env` for real LLM reasoning |
| `sqlite3.OperationalError: no such table` | Database not initialized | Should not happen via `uvicorn` (lifespan hook calls `init_db()` on startup); if calling `server/*` modules directly in a script, call `server.db.init_db()` first |
| Port 8000 already in use | Another process on that port | `uvicorn app.main:app --port 8001` |
| Frontend loads but API calls fail (CORS) | Only relevant if serving frontend from a different origin than the API | CORS is already wide-open (`allow_origins=["*"]`) for hackathon-demo simplicity; tighten for production |

---

## License

MIT — see `LICENSE`.
