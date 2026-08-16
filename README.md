# Animal Friends Dashboard

A Streamlit app for analyzing animal adoption data with interactive visualizations.

## Installation

### Prerequisites
- Python 3.7 or higher

### Setup

1. **Clone the repository:**
```
   git clone https://github.com/shriyapeddakama/AnimalFriends_Dashboard.git
   cd AnimalFriends_Dashboard
```
2. **Install dependencies:**
```
   pip install -r requirements.txt
```
3. **Run the app:**
```
   streamlit run streamlit_app.py
```
The app will open automatically in your browser at `http://localhost:8501`

## AI Insights (optional)

Each dashboard tab has a **🤖 AI Insights & Recommendations** panel. It always
computes rule-based observations for free. If a [Groq](https://groq.com) API key
is provided, it also turns those numbers into a natural-language summary plus
recommended actions. Groq has a free tier — no credit card required.

Only aggregate statistics (counts, shares, trends) are sent to the API. Adopter
names, emails, phones, and row-level data never leave your machine.

### Add a key

1. Get a free key at [console.groq.com/keys](https://console.groq.com/keys).
2. Copy the template and paste your key:
```
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
   Then edit `.streamlit/secrets.toml` and set `GROQ_API_KEY = "gsk_your_key_here"`.
3. Restart the app. The sidebar will show *"Groq API key detected."*

`.streamlit/secrets.toml` is gitignored, so your key is never committed. You can
also paste a key directly into the sidebar field for a quick test.
