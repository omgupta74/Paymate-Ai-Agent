# TaskMate — An AI Agent Built with Groq (Free, Open-Source LLM)

TaskMate is a personal assistant **AI agent** that doesn't just chat — it decides
which real-world tool to use based on what you ask, then acts and responds.

## What problem does it solve?

Freelancers and small businesses waste real time on the invoice-to-cash cycle:
writing an invoice, generating a separate payment link, sending it manually,
then repeatedly checking their bank to see who's actually paid. PayMate
automates the whole chain through plain English:

- Solve math problems, calculate GST (inclusive/exclusive, any rate)
- Convert currencies with live exchange rates
- Auto-categorize and permanently save expenses
- **Generate an invoice → automatically create a matching Razorpay payment link
  → send it via WhatsApp → track when the client actually pays**, all from one
  sentence
- **Reconcile invoices**: ask "who has paid?" and PayMate checks Razorpay live
  and tells you which clients are still pending

This demonstrates real "AI agent" behavior: the model reasons about intent,
chains multiple tools together (invoice → payment link → WhatsApp → tracking),
and replies in natural language — without the user managing any of the
underlying steps themselves.

## Why this design

Razorpay has publicly launched an MCP (Model Context Protocol) Server so AI
agents can create payment links, process refunds, and manage transactions
directly — their own stated example is "Send a payment link of ₹500 to Neha
on WhatsApp." Razorpay has also publicly emphasized voice and conversational
interfaces as the next frontier for commerce in India. PayMate is built
around that exact philosophy: a single natural-language request completes an
entire real-world payment collection workflow.

## Tech Stack

- **Groq API** — free, extremely fast inference using an open-source LLM
- **Python + Streamlit** — agent logic, tool orchestration, and web UI
- **Razorpay Payment Links API (Test Mode)** — real payment link creation and status checks
- **Twilio WhatsApp Sandbox** — free WhatsApp delivery of payment links and invoices
- **SQLite** — persistent expense and invoice storage, powering reconciliation
- **Open-Meteo / Frankfurter APIs** — free, no-key weather and currency data
- **Tool/function calling** — the LLM chains multiple tools per request

## How it works

1. User types a request (e.g. "What's the weather in Bengaluru and add 'buy groceries' to my list")
2. The LLM analyzes the request and decides which tool(s) to call
3. Python executes the actual tool function (weather API call, to-do update, etc.)
4. The tool's result is sent back to the LLM
5. The LLM crafts a natural, conversational final reply

## Setup & Run

```bash
git clone <your-repo-url>
cd taskmate-agent
pip install -r requirements.txt

# Get a free Groq key at https://console.groq.com/keys
set GROQ_API_KEY=your_key_here          (Windows)
export GROQ_API_KEY=your_key_here       (Mac/Linux)

# Get free Test Mode keys at https://dashboard.razorpay.com (Settings > API Keys)
set RAZORPAY_KEY_ID=your_test_key_id
set RAZORPAY_KEY_SECRET=your_test_key_secret

# Optional: for WhatsApp delivery, get free Sandbox credentials at https://twilio.com
set TWILIO_ACCOUNT_SID=your_twilio_sid
set TWILIO_AUTH_TOKEN=your_twilio_token

streamlit run app.py
```

Note: Twilio's WhatsApp Sandbox requires the recipient to first send the
sandbox join code to the Twilio WhatsApp number before they can receive
messages — this is a one-time step per recipient in test mode.

## Example interactions

```
You: Calculate GST on 5000 at 18%
PayMate: On Rs.5000 at 18% GST: GST amount = Rs.900, total payable = Rs.5900

You: Generate an invoice for Acme Corp with Design work:5000, Hosting:1200,
     send it to 9876543210
PayMate: INVOICE for Acme Corp
         ------------------------------
         Design work          Rs.5000.00
         Hosting              Rs.1200.00
         ------------------------------
         Subtotal             Rs.6200.00
         GST (18%)            Rs.1116.00
         TOTAL                Rs.7316.00

         Payment link: https://rzp.io/i/XXXXXXX
         WhatsApp message sent to +919876543210.

You: Who has paid?
PayMate: Reconciliation complete:
         #1 Acme Corp: paid
         #2 Beta LLC: created

You: Send a payment link of 500 to Neha for consulting
PayMate: Payment link created successfully!
Link ID: plink_XXXXXXXXXX
Amount: Rs.500.00
URL: https://rzp.io/i/XXXXXXX
(This is a Test Mode link — no real money is charged)
```

## Build Challenges & How They Were Solved

- **Choosing a free, fast LLM provider**: Used Groq instead of paid APIs since
  it offers a generous free tier and very low latency for tool-calling use cases.
- **Model deprecation mid-build**: The originally chosen model was retired by
  Groq; fixed by querying the `/models` endpoint live and switching to
  `openai/gpt-oss-120b`.
- **Tool-call message format errors**: Streamlit's session state needed plain
  dictionaries, not SDK response objects, and Groq's API rejected extra fields
  from a naive `model_dump()` — fixed by constructing clean message dicts with
  only the fields the API expects.
- **Safe math evaluation**: Avoided Python's `eval()` for security; implemented
  a safe AST-based expression evaluator instead.
- **Real payment integration without production risk**: Used Razorpay's Test
  Mode Payment Links API so the agent can demonstrate real payment-link
  creation end-to-end with zero financial risk.
- **Chaining multiple tools reliably**: Invoice generation now internally
  calls the payment-link and WhatsApp tools and logs the result, so one user
  request can trigger a full multi-step workflow instead of one isolated action.
- **WhatsApp Sandbox opt-in constraint**: Twilio's free Sandbox requires each
  recipient to join once before receiving messages, and requires pre-approved
  templates for business-initiated messages outside an active conversation
  window — handled gracefully by reporting delivery failures without breaking
  the rest of the invoice flow.
- **Single-round tool calling breaking on complex requests**: The agent
  initially only handled one round of tool calls per message, causing errors
  when a request needed multiple chained actions. Fixed by looping the
  tool-calling logic until the model returns a final answer with no further
  tool calls, with a safety cap on iterations.

## Engineering Decisions & Production Tradeoffs

This project makes deliberate choices appropriate for a fast prototype, with a
clear view of what would change in production:

- **Polling vs. Webhooks**: "Who has paid?" currently works by actively asking
  Razorpay for each payment link's status. A production system would instead
  use **Razorpay webhooks**, where Razorpay pushes a `payment_link.paid` event
  to a server endpoint the instant a payment completes — eliminating the need
  to ask at all, and enabling instant reconciliation instead of on-demand checks.
  I used polling here because this prototype doesn't expose a public webhook
  URL; webhooks would be the correct production architecture.
- **Webhook security**: If webhooks were added, every incoming webhook would
  need its **signature verified** using Razorpay's webhook secret before being
  trusted — without this, anyone could forge a fake "payment succeeded" event.
  This matters enormously at a payments company, where trusting an unverified
  event could mean releasing goods or services for a payment that never happened.
- **SQLite vs. a production database**: SQLite is used here because this is a
  single-user prototype with no concurrent access. A real multi-user version
  would move to PostgreSQL to handle concurrent writes safely.
- **Cost and scale**: Each Groq API call is effectively free at this scale, but
  a production version handling real traffic would need response caching for
  repeated queries (like currency rates) and rate-limiting per user to control
  cost and prevent abuse.
- **WhatsApp delivery**: Twilio's Sandbox requires pre-approved message
  templates for business-initiated messages outside an active conversation
  window (WhatsApp Business Platform policy) — a production deployment would
  use a verified WhatsApp Business number with approved templates rather than
  the developer Sandbox.

## Future Improvements

- Replace polling-based reconciliation with **Razorpay webhooks** (with
  signature verification) for real-time, event-driven payment status updates
- Add **Razorpay Refunds API** support so PayMate can process refunds
  conversationally, completing the full payment lifecycle — not just collection
- Move from SQLite to PostgreSQL for multi-user, concurrent-safe storage
- Add response caching and per-user rate-limiting before handling real traffic
- Deploy on Streamlit Community Cloud for a live, clickable public demo
- Move from Twilio Sandbox to a verified WhatsApp Business number for production use
