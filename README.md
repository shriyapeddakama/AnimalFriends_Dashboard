# Animal Friends Dashboard

An interactive dashboard for Animal Friends' adoption, foster, intake and
outcome data. Upload the Excel reports from the shelter database and the charts
build themselves.

---

## For Animal Friends staff

**You do not need to install anything.** Open the dashboard link, sign in with
the email address you were invited with, and upload your reports.

> **App link:** _(paste the share.streamlit.io URL here once deployed)_

### What to upload

Export each of these from the shelter database as an Excel (`.xlsx`) file.
Only the first one is required — every extra file opens up more of the
dashboard.

| Report | Needed? | Tabs it unlocks |
|---|---|---|
| **Adopter support** | **Required** | Overview, Trends, Geography, Animal Profiles, Staff/Outcome, Repeat Adopters |
| Foster activity | Optional | Foster |
| Animal intake | Optional | Intake & Outcome, Seasonality, Surrenders |
| Animal outcome | Optional | Intake & Outcome, Seasonality, Flow Diagnostics, Foster pipeline |

The adopter support report only covers animals that **were adopted**. The intake
and outcome reports cover *every* animal that came through the shelter, which is
what makes surrender trends, live release rate, and intake-to-outcome flow
possible.

### Two normal things that look like problems

**"Yes, get this app back up!"** — the dashboard goes to sleep after 12 hours
with nobody using it. Click the button and wait about 30 seconds. Anyone can
wake it, not just the person who set it up.

**Your files are gone when you come back.** This is deliberate. Nothing you
upload is stored on any server — the charts are built in memory and discarded
when you close the tab, so adopter names, emails and phone numbers never sit
anywhere outside your own computer and the shelter database. The cost is that
you re-upload each visit. Within a visit, your files stay put while you click
between tabs.

### Getting access

Access is granted to a specific email address — any address works, including a
personal Gmail; it does not have to be an Animal Friends one. Ask the
dashboard's maintainer to add you and you will get an emailed invite link.

If your address is a Google account, you sign in with one click. Otherwise
Streamlit emails you a single-use link each time you sign in.

---

## For developers

### Run it locally

Requires Python 3.9+.

```bash
git clone https://github.com/shriyapeddakama/AnimalFriends_Dashboard.git
cd AnimalFriends_Dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

### Project layout

| File | What it does |
|---|---|
| `streamlit_app.py` | Uploads, cleaning, tab layout, and the adopter-report sections |
| `intake_outcome.py` | Intake/outcome, seasonality, surrender and flow sections |
| `geo_enrich.py` | ZIP → county/distance enrichment and the catchment-area charts |
| `viz_theme.py` | Shared colour palette, validated for colour-vision accessibility |
| `ai_insights.py` | Rule-based observations plus the optional AI narrative |

No data files are committed, and `.gitignore` blocks `*.xlsx` so a real export
cannot be added by accident.

### Deploying

The app runs as a **private** app on
[Streamlit Community Cloud](https://share.streamlit.io), deployed from `main`.

- **To ship a change:** commit and push to `main`. Cloud redeploys
  automatically, usually within a minute. Editing `requirements.txt` also
  rebuilds the environment, which takes a few minutes.
- **To manage access:** in the Streamlit dashboard, **Share → enter email →
  Invite**. Any address works, including personal Gmail — the allowlist is
  per-address, not per-domain, and there is no way to grant a whole domain.
  Remove someone with the **×** beside their name.

  Two things to keep in mind. GitHub collaborators on this repo get access
  automatically, separate from that list. And because access follows the inbox
  rather than a role, someone invited at a personal address keeps access until
  they are explicitly removed — worth reviewing the viewer list periodically,
  since the data includes adopter names, emails and phone numbers.
- **To change the API key:** Streamlit dashboard → **App settings → Secrets**.
  Never commit it.

### AI Insights (optional)

Every tab has a **🤖 AI Insights & Recommendations** panel. The **Key
observations** bullets are always computed locally in pandas and need no API key
or network. If a [Groq](https://groq.com) key is present, those numbers are also
turned into a written summary and recommended actions. Groq has a free tier.

Only aggregate statistics (counts, shares, trends) are sent to the API. Adopter
names, emails, phones, and row-level data never leave the machine.

The narrative model is **discovered at runtime** from what your key can actually
call, ranked by `MODEL_PREFERENCES` in `ai_insights.py`. Groq retires model ids
periodically; because the app asks rather than assumes, a retirement does not
break the panel, and a sidebar dropdown lets you pick a different model.

**Add a key locally:**

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then set `GROQ_API_KEY = "gsk_your_key_here"`. `.streamlit/secrets.toml` is
gitignored. On Streamlit Cloud, paste the same line into **App settings →
Secrets** instead. You can also paste a key into the sidebar for a quick test.

### Known constraints

- **Memory.** Private Community Cloud apps get 1 GB. Reading four workbooks at
  once with `openpyxl` is the realistic ceiling; if the app reports going over
  its resource limits, load fewer reports or set `max_entries=1` on the
  `load_excel` cache.
- **Single points of failure.** The deployment currently depends on one personal
  GitHub account and one personal Groq key. Moving both to Animal Friends-owned
  accounts is the right long-term step.
