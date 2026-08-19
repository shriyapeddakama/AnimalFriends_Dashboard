"""AI insights & recommendations for the Animal Friends dashboard.

Hybrid design:
  * Deterministic, rule-based observations are ALWAYS computed and shown.
  * When a Groq API key is available, an LLM turns those aggregate numbers
    into a natural-language summary plus concrete recommendations.
  * With no key (or if the call fails), the rule-based bullets stand on their own.

Groq offers a free API tier (https://console.groq.com/keys). Only aggregate
statistics (counts, shares, trends) are ever sent to the API — adopter names,
emails, phones, and any row-level data never leave the machine.
"""

import calendar
import os

import pandas as pd
import streamlit as st

# Groq retires model ids on a rolling basis — a hardcoded id eventually returns
# "model_not_found" and the whole insights panel stops working. So the model is
# DISCOVERED at runtime from what the key can actually see (`models.list()`), and
# this list only expresses a *preference order* among whatever is available.
# Anything not named here still works; it just ranks below these. Bigger models
# give richer narratives; the instant/small ones are faster and cheaper.
MODEL_PREFERENCES = [
    'llama-3.3-70b-versatile',
    'openai/gpt-oss-120b',
    'moonshotai/kimi-k2-instruct',
    'meta-llama/llama-4-maverick-17b-128e-instruct',
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'qwen/qwen3-32b',
    'openai/gpt-oss-20b',
    'llama-3.1-8b-instant',
]

# Used only when the model list cannot be fetched at all (offline, proxy, etc.).
FALLBACK_MODEL = 'llama-3.1-8b-instant'

# Substrings identifying models that are not chat-completion models, so they
# never get picked as a narrative model.
_NON_CHAT_HINTS = ('whisper', 'tts', 'embed', 'guard', 'playai', 'moderation')

SYSTEM_PROMPT = (
    'You are a senior data analyst and operations advisor for Animal Friends, a '
    'Pittsburgh-based nonprofit animal shelter and rescue serving the Greater '
    'Pittsburgh region and Western Pennsylvania. The organization runs on '
    'adoptions, foster homes, volunteers, and donations, with limited staff and '
    'budget. Your job is to help staff increase adoptions, allocate scarce '
    'resources (staff time, event budget, foster capacity, marketing spend) more '
    'effectively, and grow the organization\'s reach and impact.\n\n'
    'You are given aggregate statistics from ONE section of an internal '
    'dashboard, plus an analyst focus for that section. Respond in concise plain '
    'markdown with two parts:\n'
    '- "**What stands out**": 2-4 bullets naming the most decision-relevant '
    'patterns. Interpret each — say why it matters for a shelter — rather than '
    'restating the number.\n'
    '- "**Recommended actions**": 2-4 concrete recommendations the team could '
    'execute this quarter. Every recommendation MUST name (a) WHAT to do, '
    '(b) WHERE/WHO it targets using a specific value from the data — a named zip, '
    'city, species, breed, age group, month, weekday, or staff pattern — and '
    '(c) the intended OUTCOME (more adoptions, better geographic coverage, less '
    'staff strain, more foster capacity, higher repeat/donor conversion).\n\n'
    'Quality bar — recommendations must fit Animal Friends\' actual reality:\n'
    '- Animal Friends SERVES the Pittsburgh region. Heavy local concentration is '
    'normal and healthy. NEVER recommend abandoning core areas or vaguely '
    '"expanding to new markets / beyond Pittsburgh." Instead deepen engagement '
    'where demand already exists and close concrete, nearby gaps.\n'
    '- Use only levers a shelter controls: adoption events (timing + location), '
    'mobile/pop-up adoption units, fee-waived or promotional events, foster and '
    'volunteer recruitment drives, targeted social and local media, transport, '
    'staffing schedules, and partnerships (local vets, pet stores, groomers, '
    'apartment complexes, community centers, libraries, universities).\n'
    '- Anchor timing to the data (schedule around peak months/weekdays), '
    'geography to specific zips/cities, and species/age/breed patterns to '
    'targeted marketing or specialized foster and medical support.\n'
    '- If a figure more likely reflects a data-entry or reporting gap than '
    'reality, say so and name the fix.\n\n'
    'Ground every statement strictly in the numbers provided — never invent '
    'figures or facts. No preamble, no hedging, no generic filler. Be specific '
    'enough that a staffer knows exactly what to do on Monday.'
)

# Section-specific analyst focus appended to each brief so recommendations are
# tailored to the levers that matter for that view.
SECTION_FOCUS = {
    'overview': (
        'Give a high-level read on adoption volume, repeat-adopter loyalty, and '
        'species mix. Focus recommendations on converting one-time adopters into '
        'repeat adopters, fosters, volunteers, or donors, and where realistic '
        'growth in total adoptions could come from.'
    ),
    'trends': (
        'Focus on seasonality and momentum. Recommend when to concentrate '
        'adoption events, staffing, and marketing pushes around the busiest and '
        'slowest months and the strongest weekdays. If volume is declining, name '
        'concrete ways to reverse it; if a month is slow, propose a promotion to '
        'lift it.'
    ),
    'geography': (
        'The shelter serves the Pittsburgh region — do NOT suggest leaving it. '
        'Recommend where to place pop-up/mobile adoption events and local '
        'partnerships based on the leading zips and cities, and how to reach '
        'underserved but nearby zip codes that show low volume despite being in '
        'the service area.'
    ),
    'profiles': (
        'Focus on which animals get adopted quickly versus which likely linger '
        '(e.g. older animals, specific species or breeds). Recommend targeted '
        'marketing, foster support, fee promotions, or adopter-expectation '
        'setting for the harder-to-place groups named in the data.'
    ),
    'staff': (
        'Focus on workload distribution and outcomes. If a few staff handle most '
        'adoptions, recommend cross-training and schedule balancing to reduce '
        'reliance and burnout, and process improvements suggested by the most '
        'common outcome subtypes.'
    ),
    'repeat': (
        'Focus on adopter retention and lifetime value. Use the typical gap '
        'between repeat adoptions to time re-engagement campaigns, and recommend '
        'loyalty/referral programs and ways to turn repeat adopters into fosters, '
        'volunteers, or recurring donors.'
    ),
    'foster': (
        'Focus on foster capacity and where it is strained. Recommend foster '
        'recruitment targeted at the categories carrying the most animals or '
        'hours, ways to balance load across foster homes, and retention of '
        'existing fosters.'
    ),
    'intake_outcome': (
        'This section covers EVERY animal that came through the shelter, not just '
        'the adopted ones. Focus on the balance between intake and outcome volume '
        '(is the in-care population growing?), the live release rate, and the '
        'adoption share of outcomes. Recommend capacity, transfer-partner, and '
        'placement actions. If intake persistently exceeds outcome, treat that as '
        'a capacity warning and say what to do about it.'
    ),
    'seasonality': (
        'Focus on the calendar. The numbers are period-normalized (average per '
        'occurrence of that month) and indexed so 100 = each series\' own annual '
        'average, so they are directly comparable. Identify the months where '
        'intake runs hot while adoption runs cold — those are the capacity crunch '
        'months — and recommend when to schedule adoption events, foster '
        'recruitment drives, and seasonal staffing.'
    ),
    'surrender': (
        'Owner surrenders are the intake stream a shelter can most realistically '
        'PREVENT. Focus on the surrender share of total intake, whether it is '
        'rising, its seasonality, and the specific stated reasons. Map each leading '
        'reason to a concrete prevention programme a shelter can run: housing/'
        'landlord issues to pet-deposit assistance and landlord outreach, cost to a '
        'pet food bank or low-cost vet and spay/neuter clinic, behaviour to free '
        'training classes or a behaviour helpline, allergies/time to rehoming '
        'counselling. Name the reason and the programme together.'
    ),
    'flow': (
        'This traces which intake channel each outcome came from. Focus on where '
        'the adoption rate is actually produced and where it leaks. A channel with '
        'low adoption is not automatically failing — strays returned to their '
        'owners are a success, not a lost adoption — so interpret each channel by '
        'its job before recommending anything. Recommend actions on the channels '
        'that genuinely underperform: transfer partner selection, marketing for '
        'animals from a specific channel, or intake diversion.'
    ),
}


# --------------------------------------------------------------------------- #
# API key handling                                                            #
# --------------------------------------------------------------------------- #
def _get_api_key():
    """Resolve a Groq key from Streamlit secrets, env, or the sidebar."""
    try:
        if 'GROQ_API_KEY' in st.secrets:
            return st.secrets['GROQ_API_KEY']
    except Exception:
        pass
    env_key = os.environ.get('GROQ_API_KEY')
    if env_key:
        return env_key
    return st.session_state.get('groq_api_key') or None


@st.cache_data(show_spinner=False, ttl=3600)
def _list_models(_api_key):
    """Chat models this key can currently use. `_api_key` is underscored so
    Streamlit does not hash (or persist) the secret in the cache key."""
    from groq import Groq

    listing = Groq(api_key=_api_key).models.list()
    ids = [m.id for m in listing.data]
    return sorted(i for i in ids if not any(h in i.lower() for h in _NON_CHAT_HINTS))


def resolve_model(api_key):
    """Pick a usable model: the user's explicit choice, else the highest-ranked
    preference the key can actually see, else anything chat-capable.

    Returns ``(model_id, available_ids)``; `available_ids` is empty when the
    listing call failed, in which case the caller is flying blind on a fallback.
    """
    chosen = st.session_state.get('groq_model')
    try:
        available = _list_models(api_key)
    except Exception:
        return chosen or FALLBACK_MODEL, []

    if chosen and chosen in available:
        return chosen, available
    for pref in MODEL_PREFERENCES:
        if pref in available:
            return pref, available
    return (available[0] if available else FALLBACK_MODEL), available


def sidebar_ai_settings():
    """Render AI configuration in the sidebar and report key status."""
    st.sidebar.markdown('---')
    st.sidebar.subheader('🤖 AI insights')
    key = _get_api_key()
    if key:
        model, available = resolve_model(key)
        if available:
            st.sidebar.success(f'Groq key detected — {len(available)} models available.')
            st.sidebar.selectbox(
                'Narrative model',
                available,
                index=available.index(model),
                key='groq_model',
                help='Discovered from your key. The default is the best-ranked '
                     'model Groq currently serves; smaller ones are faster.',
            )
        else:
            st.sidebar.warning(
                f'Groq key detected, but the model list could not be fetched. '
                f'Falling back to `{model}` — if that id has been retired, the '
                'narrative will fail while the computed observations still work.'
            )
    else:
        st.sidebar.caption(
            'Add a free Groq API key to turn the computed stats into natural-language '
            'summaries and recommendations. Without a key you still get rule-based insights. '
            'Get one at console.groq.com/keys.'
        )
        entered = st.sidebar.text_input(
            'Groq API key (optional)',
            type='password',
            key='groq_api_key_input',
            placeholder='gsk_...',
        )
        if entered:
            st.session_state['groq_api_key'] = entered.strip()
    st.sidebar.caption('Only aggregate numbers are sent to the API — no names, emails, or rows.')


# --------------------------------------------------------------------------- #
# Formatting helpers                                                          #
# --------------------------------------------------------------------------- #
def _pct(n, d):
    return f'{(100.0 * n / d):.0f}%' if d else '0%'


def _clean_months(df):
    """Monthly adoption counts with the 'NaT' bucket removed, chronological."""
    monthly = df.groupby('Adoption_Month').size()
    monthly = monthly[monthly.index != 'NaT']
    return monthly.sort_index()


# --------------------------------------------------------------------------- #
# Per-tab observation builders (deterministic, aggregate-only)                #
# --------------------------------------------------------------------------- #
def _obs_overview(df):
    obs = []
    total = len(df)
    adopters = df['Adopter ID'].nunique()
    counts = df['Adopter ID'].value_counts()
    repeat = int((counts > 1).sum())
    obs.append(f'{total:,} adoptions across {adopters:,} unique adopters.')
    obs.append(f'{repeat:,} adopters ({_pct(repeat, adopters)}) adopted more than once.')

    avg_age = df['Age_at_Adoption_yrs'].mean()
    if pd.notna(avg_age):
        obs.append(f'Average age at adoption is {avg_age:.1f} years.')

    top_sp = df['Species'].value_counts().head(3)
    if not top_sp.empty:
        parts = [f'{s} ({_pct(c, total)})' for s, c in top_sp.items()]
        obs.append('Top species: ' + ', '.join(parts) + '.')

    top_city = df['City'].value_counts().head(3)
    if not top_city.empty:
        lead, lead_n = top_city.index[0], top_city.iloc[0]
        obs.append(f'Most adoptions come from {lead} ({_pct(lead_n, total)} of all adoptions).')

    # Contact reachability — the base available for re-engagement / donor outreach.
    if 'Has_Email' in df.columns and 'Has_Phone' in df.columns:
        obs.append(
            f'{df["Has_Email"].mean() * 100:.0f}% of records have an email and '
            f'{df["Has_Phone"].mean() * 100:.0f}% a phone — the reachable base for '
            f're-engagement and donor campaigns.'
        )
    if 'Has_Microchip' in df.columns:
        obs.append(f'{df["Has_Microchip"].mean() * 100:.0f}% of adopted animals have a microchip on record.')
    return obs


def _obs_trends(df):
    obs = []
    dated = df.dropna(subset=['Adoption_Year'])
    yearly = dated.groupby('Adoption_Year').size().sort_index()
    # Boundary years in an export are usually partial (e.g. data starting in June
    # or ending mid-year). Count months of coverage per year so we can compare
    # only COMPLETE calendar years and avoid false "volume collapsed" alarms.
    months_per_year = dated.groupby('Adoption_Year')['Adoption_Month'].nunique()
    complete = yearly[months_per_year.reindex(yearly.index).fillna(0) >= 12]

    if len(complete) >= 2:
        first_y, last_y = int(complete.index[0]), int(complete.index[-1])
        first_v, last_v = int(complete.iloc[0]), int(complete.iloc[-1])
        direction = 'up' if last_v > first_v else 'down' if last_v < first_v else 'flat'
        obs.append(
            f'Across complete calendar years, volume moved {direction}: {first_v:,} in {first_y} '
            f'to {last_v:,} in {last_y} (partial start/end years excluded from this comparison).'
        )
        if len(complete) >= 2 and int(complete.iloc[-2]):
            change = (last_v - int(complete.iloc[-2])) / int(complete.iloc[-2]) * 100
            obs.append(
                f'The most recent complete year ({last_y}) changed {change:+.0f}% versus the prior '
                f'complete year.'
            )
    elif len(yearly) >= 2:
        first_y, last_y = int(yearly.index[0]), int(yearly.index[-1])
        obs.append(
            f'Yearly volume ran {int(yearly.iloc[0]):,} in {first_y} to {int(yearly.iloc[-1]):,} '
            f'in {last_y}, but boundary years may be partial — interpret the trend with care.'
        )

    # Flag any partial boundary year explicitly so the model does not read a
    # half-year total as a full-year decline.
    boundary_years = list(dict.fromkeys([yearly.index[0], yearly.index[-1]])) if len(yearly) else []
    for yr in boundary_years:
        m = int(months_per_year.get(yr, 0))
        if m < 12:
            obs.append(
                f'Note: {int(yr)} is a PARTIAL year ({m} of 12 months present) — its total is not '
                f'comparable to a full year and should not be read as growth or decline.'
            )

    monthly = _clean_months(df)
    if not monthly.empty:
        peak_m, peak_v = monthly.idxmax(), int(monthly.max())
        low_m, low_v = monthly.idxmin(), int(monthly.min())
        obs.append(f'Busiest month on record: {peak_m} ({peak_v:,} adoptions); slowest: {low_m} ({low_v:,}).')

    # Recent momentum: last 3 months vs the prior 3 (needs 6+ months of history).
    if len(monthly) >= 6:
        last3 = monthly.iloc[-3:].mean()
        prev3 = monthly.iloc[-6:-3].mean()
        if prev3:
            mom = (last3 - prev3) / prev3 * 100
            obs.append(
                f'Recent momentum: the last 3 months averaged {last3:.0f} adoptions/month, '
                f'{mom:+.0f}% versus the prior 3 months.'
            )

    # Calendar seasonality across all years (which time of year to plan around).
    dt = df['Date Of Adoption'].dropna()
    if not dt.empty:
        by_cal_month = dt.dt.month.value_counts()
        strong_m, weak_m = int(by_cal_month.idxmax()), int(by_cal_month.idxmin())
        obs.append(
            f'Seasonally, {calendar.month_name[strong_m]} is the strongest month of the year '
            f'and {calendar.month_name[weak_m]} the weakest (totaled across all years).'
        )
        by_q = dt.dt.quarter.value_counts()
        obs.append(f'Q{int(by_q.idxmax())} is the busiest quarter; Q{int(by_q.idxmin())} the slowest.')

    dow = df['Adoption_DOW'].value_counts()
    if not dow.empty:
        weakest_dow = dow.index[-1]
        obs.append(
            f'{dow.index[0]} is the strongest weekday ({int(dow.iloc[0]):,} adoptions); '
            f'{weakest_dow} is the weakest ({int(dow.iloc[-1]):,}).'
        )
    return obs


def _obs_geography(df):
    obs = []
    total = len(df)
    cities = df['City'].value_counts()
    if not cities.empty:
        top3_share = cities.head(3).sum()
        obs.append(
            f'The top 3 cities account for {_pct(top3_share, total)} of adoptions; '
            f'{cities.index[0]} leads with {int(cities.iloc[0]):,}.'
        )

    # Keep only well-formed 5-digit zips so counts and rankings are meaningful.
    zips = df['Zip_Clean'].value_counts()
    zips = zips[zips.index.str.match(r'^\d{5}$')]
    if not zips.empty:
        top5 = zips.head(5)
        obs.append('Top zip codes: ' + ', '.join(f'{z} ({int(c):,})' for z, c in top5.items()) + '.')
        obs.append(f'Adoptions are spread across {zips.size:,} distinct zip codes.')

        # Concentration: how few zips make up ~80% of volume.
        cum = zips.cumsum()
        n80 = int((cum <= 0.8 * zips.sum()).sum()) + 1
        obs.append(f'Just {n80} zip codes account for ~80% of all adoptions — demand is highly concentrated.')

        # Emerging mid-volume zips (ranks 6-12): proven interest, room to grow.
        emerging = zips.iloc[5:12]
        if not emerging.empty:
            obs.append(
                'Mid-volume zips with room to grow (ranks 6-12), good pop-up/outreach targets: '
                + ', '.join(f'{z} ({int(c):,})' for z, c in emerging.items()) + '.'
            )

        # Thin coverage: zips with only 1-2 adoptions on record.
        thin = int((zips <= 2).sum())
        obs.append(f'{thin:,} zip codes have 2 or fewer adoptions on record — thin, underserved coverage.')
    return obs


def _obs_profiles(df):
    obs = []
    total = len(df)
    sp = df['Species'].value_counts()
    if not sp.empty:
        parts = [f'{s} ({_pct(c, total)})' for s, c in sp.head(3).items()]
        obs.append('Species mix: ' + ', '.join(parts) + '.')

    buckets = df['Age_Bucket'].value_counts()
    buckets = buckets[buckets.index != 'Unknown']
    if not buckets.empty:
        obs.append(f'Most adopted age group is "{buckets.index[0]}" ({_pct(buckets.iloc[0], total)}).')

    breeds = df['Primary Breed'].value_counts()
    if not breeds.empty:
        obs.append(f'Most common breed overall is {breeds.index[0]} ({int(breeds.iloc[0]):,} adoptions).')

    # Senior animals are typically the slowest to place — size the challenge.
    if 'Age_Bucket' in df.columns:
        senior = int((df['Age_Bucket'] == 'Senior (7yr+)').sum())
        obs.append(f'Senior animals (7yr+) make up {_pct(senior, total)} of adoptions — usually the hardest to place.')

    # Typical adoption age for the leading species (targets marketing/foster support).
    for sp in df['Species'].value_counts().head(2).index:
        avg_sp_age = df.loc[df['Species'] == sp, 'Age_at_Adoption_yrs'].mean()
        if pd.notna(avg_sp_age):
            obs.append(f'Average {sp} is adopted at {avg_sp_age:.1f} years old.')
    return obs


def _obs_staff(df):
    obs = []
    total = len(df)
    staff = df['By (User)'].value_counts()
    if not staff.empty:
        obs.append(
            f'{staff.size:,} staff members processed adoptions; the top processor handled '
            f'{int(staff.iloc[0]):,} ({_pct(staff.iloc[0], total)}).'
        )
        top5 = staff.head(5).sum()
        obs.append(f'The 5 most active staff account for {_pct(top5, total)} of all adoptions.')

    outcomes = df['Outcome Subtype'].value_counts()
    if not outcomes.empty:
        parts = [f'"{o}" ({_pct(c, total)})' for o, c in outcomes.head(3).items()]
        obs.append('Top outcome subtypes: ' + ', '.join(parts) + '.')
    return obs


def _obs_repeat(df):
    obs = []
    counts = df['Adopter ID'].value_counts()
    adopters = counts.size
    repeat = int((counts > 1).sum())
    obs.append(f'{repeat:,} of {adopters:,} adopters ({_pct(repeat, adopters)}) are repeat adopters.')
    if not counts.empty:
        obs.append(f'The most active adopter has {int(counts.max()):,} adoptions.')

    # Repeat adopters punch above their headcount — size their volume share.
    repeat_vol = int(counts[counts > 1].sum())
    obs.append(f'Repeat adopters drive {_pct(repeat_vol, int(counts.sum()))} of total adoption volume.')
    if repeat:
        obs.append(f'Repeat adopters take home {counts[counts > 1].mean():.1f} animals each on average.')

    repeat_ids = counts[counts > 1].index
    rdf = df[df['Adopter ID'].isin(repeat_ids)].sort_values(['Adopter ID', 'Date Of Adoption'])
    gaps = rdf.groupby('Adopter ID')['Date Of Adoption'].diff().dt.days.dropna()
    if not gaps.empty:
        obs.append(
            f'Median gap between repeat adoptions is {int(gaps.median()):,} days '
            f'(average {int(gaps.mean()):,} days) — the window to time re-engagement.'
        )
    return obs


# Foster count columns keyed by category (mirrors streamlit_app.foster_section).
_FOSTER_COUNT_COLS = {
    'Dogs': 'Foster Dogs (total unique count)',
    'Cats': 'Foster Cats (total unique count)',
    'Birds': 'Foster Birds (total unique count)',
    'Rabbits': 'Foster Rabbits (total unique count)',
    'Barnyards': 'Foster Barnyards (total unique count)',
    'Small Mammals': 'Foster Small Mammals (total unique count)',
    'Exotic/Others': 'Foster Exotic/Others (total unique count)',
}
_FOSTER_HOURS_COLS = {
    'Dogs': 'Total Foster Hours for Dogs',
    'Cats': 'Total Foster Hours for Cats',
    'Birds': 'Total Foster Hours for Birds',
    'Rabbits': 'Total Foster Hours for Rabbits',
    'Barnyards': 'Total Foster Hours for Barnyards',
    'Small Mammals': 'Total Foster Hours for Small Mammals',
    'Exotic/Others': 'Total Foster Hours for Exotic/Others',
}


def _obs_foster(df_foster):
    obs = []
    if df_foster is None or df_foster.empty:
        return ['No foster dataset is loaded.']

    per_cat_counts = {}
    per_cat_hours = {}
    for cat, col in _FOSTER_COUNT_COLS.items():
        if col in df_foster.columns:
            per_cat_counts[cat] = float(pd.to_numeric(df_foster[col], errors='coerce').fillna(0).sum())
    for cat, col in _FOSTER_HOURS_COLS.items():
        if col in df_foster.columns:
            per_cat_hours[cat] = float(pd.to_numeric(df_foster[col], errors='coerce').fillna(0).sum())

    if 'Foster Person Name' in df_foster.columns:
        obs.append(f'{df_foster["Foster Person Name"].nunique():,} distinct foster homes are on record.')

    if per_cat_counts:
        total_animals = sum(per_cat_counts.values())
        top_cat = max(per_cat_counts, key=per_cat_counts.get)
        obs.append(
            f'{int(total_animals):,} fostered animals across categories; {top_cat} is the largest '
            f'({_pct(per_cat_counts[top_cat], total_animals)}).'
        )
    if per_cat_hours:
        total_hours = sum(per_cat_hours.values())
        top_hours_cat = max(per_cat_hours, key=per_cat_hours.get)
        obs.append(
            f'{int(total_hours):,} total foster hours logged; {top_hours_cat} accounts for the most '
            f'({_pct(per_cat_hours[top_hours_cat], total_hours)}).'
        )
    return obs


# --------------------------------------------------------------------------- #
# Shelter-wide (intake / outcome) observation builders                        #
# --------------------------------------------------------------------------- #
_MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
_NON_LIVE = {'Euthanasia', 'Died in Custody', 'Lost-Stolen'}


def _year_series(df, year_labels):
    return df.groupby('YearLabel').size().reindex(year_labels).fillna(0)


def _obs_intake_outcome(context):
    din, dout = context.get('intake'), context.get('outcome')
    years = context.get('year_labels') or []
    obs = ['Periods are rolling 12-month buckets anchored to the export start date, '
           'not calendar years — so no bucket is a partial year.']

    if din is not None:
        counts = _year_series(din, years)
        obs.append(f'{len(din):,} intake records across {len(years)} rolling years.')
        if len(counts) >= 2 and counts.iloc[0]:
            obs.append(
                f'Intake volume ran {int(counts.iloc[0]):,} in {years[0]} to '
                f'{int(counts.iloc[-1]):,} in {years[-1]} '
                f'({(counts.iloc[-1] - counts.iloc[0]) / counts.iloc[0] * 100:+.0f}%).'
            )
        mix = din['Species_Group'].value_counts(normalize=True) * 100
        obs.append('Intake species mix: ' + ', '.join(f'{k} ({v:.0f}%)' for k, v in mix.items()) + '.')
        if 'Intake Type' in din.columns:
            ch = din['Intake Type'].value_counts(normalize=True).head(4) * 100
            obs.append('Intake channels: ' + ', '.join(f'{k} ({v:.0f}%)' for k, v in ch.items()) + '.')

    if dout is not None and 'Outcome Type' in dout.columns:
        obs.append(f'{len(dout):,} outcome records.')
        live = (~dout['Outcome Type'].isin(_NON_LIVE)).mean() * 100
        obs.append(f'Overall live release rate is {live:.1f}% across the whole window.')
        oc = dout['Outcome Type'].value_counts(normalize=True).head(4) * 100
        obs.append('Outcome mix: ' + ', '.join(f'{k} ({v:.0f}%)' for k, v in oc.items()) + '.')

        by_year = dout.groupby('YearLabel').apply(
            lambda d: (~d['Outcome Type'].isin(_NON_LIVE)).mean() * 100
        ).reindex(years).dropna()
        if len(by_year) >= 2:
            obs.append(
                f'Live release rate moved from {by_year.iloc[0]:.1f}% ({by_year.index[0]}) '
                f'to {by_year.iloc[-1]:.1f}% ({by_year.index[-1]}).'
            )
        adopt_year = dout.groupby('YearLabel').apply(
            lambda d: (d['Outcome Type'] == 'Adoption').mean() * 100
        ).reindex(years).dropna()
        if len(adopt_year) >= 2:
            obs.append(
                f'Adoption share of outcomes moved from {adopt_year.iloc[0]:.1f}% to '
                f'{adopt_year.iloc[-1]:.1f}%; lowest year was {adopt_year.idxmin()} '
                f'({adopt_year.min():.1f}%).'
            )

    if din is not None and dout is not None:
        gap = len(din) - len(dout)
        obs.append(
            f'Across the window, intake exceeded outcome by {gap:,} animals '
            f'({gap / len(din) * 100:+.1f}% of intake) — the net change in the in-care population.'
        )
    return obs


def _obs_seasonality(context):
    din, dout = context.get('intake'), context.get('outcome')
    obs = ['Monthly figures are period-normalized (each day weighted by 1/days-in-month), '
           'so partial months at the edges of the export do not distort the pattern.']

    def _norm_month(df, date_col):
        dates = df[date_col]
        days = pd.date_range(dates.min(), dates.max(), freq='D')
        w = pd.DataFrame({'d': days})
        occ = (1 / w['d'].dt.days_in_month).groupby(w['d'].dt.month).sum().reindex(range(1, 13), fill_value=0)
        monthly = df.groupby('Month').size().reindex(range(1, 13)).fillna(0)
        return monthly / occ.replace(0, pd.NA)

    if din is not None:
        s = _norm_month(din, 'Intake Date').astype(float)
        obs.append(
            f'Intake peaks in {_MONTH_ABBR[int(s.idxmax()) - 1]} ({s.max():.0f}/month on average) '
            f'and bottoms in {_MONTH_ABBR[int(s.idxmin()) - 1]} ({s.min():.0f}/month).'
        )
        idx = s / s.mean() * 100
        hot = [_MONTH_ABBR[m - 1] for m in idx[idx >= 115].index]
        if hot:
            obs.append(f'Months running 15%+ above the annual intake average: {", ".join(hot)}.')
        if 'Intake Type' in din.columns:
            surr = din[din['Intake Type'] == 'Owner Surrender']
            if not surr.empty:
                ss = _norm_month(surr, 'Intake Date').astype(float)
                obs.append(f'Owner surrenders peak in {_MONTH_ABBR[int(ss.idxmax()) - 1]}.')

    if dout is not None and 'Outcome Type' in dout.columns:
        adopt = dout[dout['Outcome Type'] == 'Adoption']
        if not adopt.empty:
            a = _norm_month(adopt, 'Outcome Date').astype(float)
            obs.append(
                f'Adoptions peak in {_MONTH_ABBR[int(a.idxmax()) - 1]} ({a.max():.0f}/month) '
                f'and bottom in {_MONTH_ABBR[int(a.idxmin()) - 1]} ({a.min():.0f}/month).'
            )
            if din is not None:
                s = _norm_month(din, 'Intake Date').astype(float)
                si, ai = s / s.mean() * 100, a / a.mean() * 100
                crunch = [_MONTH_ABBR[m - 1] for m in range(1, 13)
                          if si.get(m, 0) > 100 and ai.get(m, 0) < 100]
                if crunch:
                    obs.append(
                        'Capacity crunch months (intake above its own average while adoption '
                        f'is below its own): {", ".join(crunch)}.'
                    )
    return obs


def _obs_surrender(context):
    din = context.get('intake')
    years = context.get('year_labels') or []
    if din is None or 'Intake Type' not in din.columns:
        return ['No intake data with an Intake Type column is loaded.']

    surr = din[din['Intake Type'] == 'Owner Surrender']
    if surr.empty:
        return ['No owner-surrender records in the intake file.']

    obs = [
        f'{len(surr):,} owner surrenders, {_pct(len(surr), len(din))} of all intake.',
        'Species split of surrenders: '
        + ', '.join(f'{k} ({v:.0f}%)' for k, v in
                    (surr['Species_Group'].value_counts(normalize=True) * 100).items()) + '.',
    ]

    share = (surr.groupby('YearLabel').size() / din.groupby('YearLabel').size() * 100)
    share = share.reindex(years).dropna()
    if len(share) >= 2:
        obs.append(
            f'Surrender share of intake moved from {share.iloc[0]:.1f}% ({share.index[0]}) to '
            f'{share.iloc[-1]:.1f}% ({share.index[-1]}); peak was {share.idxmax()} at {share.max():.1f}%.'
        )
    counts = surr.groupby('YearLabel').size().reindex(years).fillna(0)
    if len(counts) >= 2:
        obs.append(f'Surrender headcount by year: '
                   + ', '.join(f'{y} {int(v):,}' for y, v in counts.items()) + '.')

    if 'Intake Sub-type' in surr.columns:
        for sp in ('Cat', 'Dog'):
            sub = surr[surr['Species_Group'] == sp]
            if sub.empty:
                continue
            vc = sub['Intake Sub-type'].value_counts().head(5)
            obs.append(
                f'Top {sp.lower()} surrender reasons (n={len(sub):,}): '
                + ', '.join(f'{r} ({c / len(sub) * 100:.0f}%)' for r, c in vc.items()) + '.'
            )
    return obs


def _obs_flow(context):
    dout = context.get('outcome')
    years = context.get('year_labels') or []
    if dout is None or not {'Intake Type', 'Outcome Type'}.issubset(dout.columns):
        return ['No outcome data with both Intake Type and Outcome Type is loaded.']

    obs = []
    ct = pd.crosstab(dout['Intake Type'], dout['Outcome Type'])
    totals = ct.sum(axis=1)
    if 'Adoption' in ct.columns:
        adopt_rate = (ct['Adoption'] / totals * 100).sort_values(ascending=False)
        obs.append(
            'Adoption rate by intake channel: '
            + ', '.join(f'{ch} {v:.0f}% (n={int(totals[ch]):,})' for ch, v in adopt_rate.items()) + '.'
        )
        contribution = (ct['Adoption'] / len(dout) * 100).sort_values(ascending=False)
        obs.append(
            'Share of ALL outcomes that is an adoption from each channel: '
            + ', '.join(f'{ch} {v:.1f} pts' for ch, v in contribution.head(4).items()) + '.'
        )

    if 'Return to Owner' in ct.columns:
        rto = (ct['Return to Owner'] / totals * 100).sort_values(ascending=False)
        obs.append(
            'Return-to-owner rate by channel: '
            + ', '.join(f'{ch} {v:.0f}%' for ch, v in rto.head(3).items())
            + '. A high rate on stray intake is a SUCCESS (owners reclaiming pets), not a lost adoption.'
        )

    for ch in ('Stray In', 'Owner Surrender'):
        sub = dout[dout['Intake Type'] == ch]
        if sub.empty:
            continue
        by_year = sub.groupby('YearLabel').apply(
            lambda d: (d['Outcome Type'] == 'Adoption').mean() * 100
        ).reindex(years).dropna()
        if len(by_year) >= 2:
            obs.append(
                f'"{ch}" adoption rate by year: '
                + ', '.join(f'{y} {v:.0f}%' for y, v in by_year.items()) + '.'
            )
    return obs


_BUILDERS = {
    'overview': _obs_overview,
    'trends': _obs_trends,
    'geography': _obs_geography,
    'profiles': _obs_profiles,
    'staff': _obs_staff,
    'repeat': _obs_repeat,
}

# Builders that read the intake/outcome exports rather than the adopter report.
_CONTEXT_BUILDERS = {
    'intake_outcome': _obs_intake_outcome,
    'seasonality': _obs_seasonality,
    'surrender': _obs_surrender,
    'flow': _obs_flow,
}


def _build_observations(tab_key, df, df_foster, context=None):
    try:
        if tab_key in _CONTEXT_BUILDERS:
            return _CONTEXT_BUILDERS[tab_key](context or {})
        if tab_key == 'foster':
            return _obs_foster(df_foster)
        return _BUILDERS[tab_key](df)
    except Exception as exc:  # never let insight-building break a tab
        return [f'Could not compute observations for this section ({exc}).']


# --------------------------------------------------------------------------- #
# Groq narrative (cached)                                                      #
# --------------------------------------------------------------------------- #
# Both reported failures were one bug: a completion budget of 1024 that the
# longer briefs (Trends and Surrenders build the most observation bullets) ran
# straight past. It shows up two different ways depending on the model:
#
#   * a reasoning model spends the budget thinking BEFORE it writes, so the
#     answer comes back empty  -> "the model declined to summarize"
#   * an ordinary model spends it WHILE writing, so the answer stops mid-word
#     -> a narrative that trails off
#
# Both report finish_reason='length'. Headroom fixes both, and costs nothing
# when unused: a narrative is only a few hundred tokens, so a model that
# finishes normally never touches the rest of the allowance.
NARRATIVE_MAX_TOKENS = 4096

# One larger retry for the rare model verbose enough to still be mid-sentence at
# 4096, so a truncated narrative is never the first thing the user is shown.
NARRATIVE_RETRY_TOKENS = 8192

# Families that reason before answering. They also accept `reasoning_effort`,
# which keeps the thinking (and the latency) short for what is only a
# summarization task.
_REASONING_FAMILIES = ('gpt-oss', 'qwen3', 'deepseek-r1', 'magistral')


def _is_reasoning_model(model: str) -> bool:
    lowered = model.lower()
    return any(family in lowered for family in _REASONING_FAMILIES)


@st.cache_data(show_spinner=False)
def _ai_narrative(brief, model, _api_key, budget=NARRATIVE_MAX_TOKENS):
    """Send the aggregate brief to Groq, returning ``(text, finish_reason)``.

    `budget` is part of the cache key, so the larger retry is stored separately
    from the first attempt rather than overwriting it. `_api_key` is underscored
    so Streamlit does not hash (or persist) the secret in the cache key.
    """
    from groq import Groq

    client = Groq(api_key=_api_key)
    kwargs = dict(
        model=model,
        max_completion_tokens=budget,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': brief},
        ],
    )
    if _is_reasoning_model(model):
        kwargs['reasoning_effort'] = 'low'

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        # A model that does not take `reasoning_effort` rejects the whole
        # request. Drop the hint and ask again rather than reporting a failure
        # the user can do nothing about.
        if 'reasoning_effort' not in kwargs or not _is_param_error(exc):
            raise
        kwargs.pop('reasoning_effort')
        response = client.chat.completions.create(**kwargs)

    choice = response.choices[0]
    return (choice.message.content or '').strip(), getattr(choice, 'finish_reason', None)


def _is_model_error(exc) -> bool:
    """True when the failure is 'this model id no longer exists'."""
    text = str(exc).lower()
    return 'model_not_found' in text or 'does not exist' in text


def _is_param_error(exc) -> bool:
    """True when the request was rejected over an unsupported parameter."""
    text = str(exc).lower()
    return 'reasoning_effort' in text or 'unsupported' in text or 'unrecognized' in text


def _candidate_models(api_key, tried):
    """Models worth trying, best first, skipping any already attempted.

    Ranked against what the key can actually see, so a retry never burns a
    request on an id this account was never served. Only when the listing itself
    is unreachable do we fall back to guessing from the preference list.
    """
    try:
        available = _list_models(api_key)
    except Exception:
        available = []
    ranked = [m for m in MODEL_PREFERENCES if m in available]
    ranked += [m for m in available if m not in ranked]
    if not ranked:
        ranked = MODEL_PREFERENCES + [FALLBACK_MODEL]
    return [m for m in dict.fromkeys(ranked) if m not in tried]


def _narrate(brief, api_key):
    """Ask Groq for a narrative, working down the candidate models.

    Two failure modes are both recoverable and both handled here:

    * **A retired model id** (`model_not_found`) means the cached listing is
      stale, so the cache is dropped and the next candidate is tried.
    * **An empty answer** means the model produced nothing usable — in practice
      a reasoning model that spent its whole budget thinking. Showing the user a
      blank panel is useless when another model would have answered, so this
      counts as a failure and moves on too.

    A *truncated* answer (text present, ``finish_reason == 'length'``) is not a
    failure — it is a real narrative that stops mid-sentence. That earns one
    retry at a larger budget before being handed back, flagged, so the caller
    can tell the reader it was cut short instead of presenting half a
    recommendation as the whole thing.

    The sidebar's model selection is never mutated here — it is a widget key,
    and rewriting it mid-run would reset the control under the user.

    Returns ``(text, model_used, truncated)``. Raises only when every candidate
    failed.
    """
    first, _ = resolve_model(api_key)
    tried = set()
    empties = []
    listing_refreshed = False

    candidates = [first] + _candidate_models(api_key, {first})
    while candidates:
        candidate = candidates.pop(0)
        if candidate in tried:
            continue
        tried.add(candidate)

        try:
            text, finish = _ai_narrative(brief, candidate, api_key)
        except Exception as exc:
            if not _is_model_error(exc):
                raise
            if not listing_refreshed:
                # The listing is stale — refresh it once and re-plan from what
                # the key can actually see now.
                _list_models.clear()
                listing_refreshed = True
                candidates = _candidate_models(api_key, tried)
            continue

        if text and finish == 'length':
            # Real prose, but it ran out mid-sentence. Give this model one more
            # go with room to finish before settling for the truncated version.
            try:
                longer, longer_finish = _ai_narrative(
                    brief, candidate, api_key, NARRATIVE_RETRY_TOKENS)
                if longer:
                    return longer, candidate, longer_finish == 'length'
            except Exception:
                pass  # keep the shorter answer we already have
            return text, candidate, True

        if text:
            return text, candidate, False

        # Empty answer. `length` here means the budget went entirely on
        # reasoning. The empty stays cached deliberately — a repeat click skips
        # straight past this model to the one that did answer, rather than
        # paying for it again, and clearing the whole cache would discard every
        # other tab's good narrative along with it.
        empties.append(f'{candidate} (finish_reason={finish})')

    if empties:
        raise RuntimeError(
            'no model returned any text — tried ' + ', '.join(empties)
        )
    raise RuntimeError('no usable model was available for this key')


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def render_insights(tab_key, tab_title, df, df_foster=None, context=None):
    """Render an AI-insights panel for one tab. Safe to call once per tab.

    `context` carries the prepared intake/outcome frames for the shelter-wide
    tabs, which do not read the adopter-support dataframe at all.
    """
    with st.expander('🤖 AI Insights & Recommendations', expanded=False):
        gen_key = f'insights_generated_{tab_key}'
        if st.button('Generate insights', key=f'gen_btn_{tab_key}'):
            st.session_state[gen_key] = True

        if not st.session_state.get(gen_key):
            st.caption('Click **Generate insights** to analyze this section.')
            return

        observations = _build_observations(tab_key, df, df_foster, context)

        st.markdown('**Key observations** (computed)')
        st.markdown('\n'.join(f'- {o}' for o in observations))

        key = _get_api_key()
        if not key:
            st.info('Add a free Groq API key in the sidebar to get an AI summary and recommendations.')
            return

        focus = SECTION_FOCUS.get(tab_key, '')
        brief = f'Section: {tab_title}\n'
        if focus:
            brief += f'\nAnalyst focus for this section: {focus}\n'
        brief += '\nAggregate statistics:\n' + '\n'.join(f'- {o}' for o in observations)
        with st.spinner('Asking the AI for insights…'):
            try:
                narrative, model_used, truncated = _narrate(brief, key)
            except Exception as exc:
                st.warning(f'AI narrative unavailable ({exc}). The computed observations above still apply.')
                return

        if narrative:
            st.markdown('---')
            st.markdown(narrative)
            if truncated:
                st.warning(
                    'This summary was cut off before it finished — the last point '
                    'is incomplete. Pick a different model in the sidebar, or read '
                    'the computed observations above, which are always complete.',
                    icon='✂️',
                )
            st.caption(f'Generated by `{model_used}` via Groq from the aggregate numbers above.')
        else:
            # Unreachable in normal operation — _narrate raises rather than
            # returning empty — but kept so an empty answer can never render as
            # a silently blank panel.
            st.warning(
                f'`{model_used}` returned an empty summary for this section. '
                'The computed observations above still apply; try a different '
                'model in the sidebar.'
            )
