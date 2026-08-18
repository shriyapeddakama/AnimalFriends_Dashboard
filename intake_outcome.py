"""Intake / outcome analytics for the Animal Friends dashboard.

Ported from the Part 1-2 macro-overview & surrender analysis notebooks. These
sections read the raw ``animal-intake`` and ``animal-outcome`` exports, which are
separate files from the adopter-support report the original dashboard tabs use.

Two methodology choices carried over from the notebooks:

* **Rolling 12-month year buckets, not calendar years.** The exports are rolling
  5-year pulls, so the first and last calendar years are partial and would read
  as a false collapse in volume. Buckets are anchored to the earliest date across
  whichever files are loaded, and the *number* of buckets is derived from the
  actual date span — this works unchanged on a 3-month or a 10-year export.
* **Period-normalized months.** For "which month of the year is busiest",
  each calendar day is weighted by 1/(days in its month), so a partial month at
  the edge of the export counts as a fraction of a month rather than a whole one.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from viz_theme import (
    ACCENT,
    INTAKE_COLORS,
    INTAKE_ORDER,
    NON_LIVE_OUTCOMES,
    OTHER,
    OUTCOME_COLORS,
    OUTCOME_ORDER,
    SPECIES_COLORS,
    SPECIES_GROUPS,
    SPECIES_SEGMENTS,
    apply_layout,
)

MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Outcome sub-types that indicate an animal passed through foster care. There is
# no dedicated foster outcome type in the source system, so this is a directional
# proxy, not a census of the foster network.
FOSTER_SUBTYPES = ['Foster to Adopt', 'Foster Fail', 'Died in Foster']


# --------------------------------------------------------------------------- #
# Shared preparation                                                          #
# --------------------------------------------------------------------------- #
def _species_group(s):
    if s in ('Cat', 'Dog'):
        return s
    return 'Other'


def _bucket_span(dates, start):
    """Number of rolling 12-month buckets covered by `dates`, from `start`."""
    if dates.empty:
        return 1
    return max(int(np.ceil((dates.max() - start).days / 365.25)), 1)


@st.cache_data(show_spinner=False)
def prepare_records(df: pd.DataFrame, date_col: str, start, n_years: int) -> pd.DataFrame:
    """Parse dates, group species, and attach rolling year buckets."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['Species_Group'] = df['Species'].apply(_species_group) if 'Species' in df.columns else 'Other'
    df['YearBucket'] = df[date_col].apply(
        lambda d: min(int((d - start).days // 365.25) + 1, n_years)
    )
    df['YearLabel'] = 'Year ' + df['YearBucket'].astype(str)
    df['Month'] = df[date_col].dt.month
    return df


def report_window(din: pd.DataFrame, dout: pd.DataFrame):
    """Anchor date and bucket count shared by both files, so years line up.

    Returns ``(start, n_years, year_labels)`` or ``(None, 0, [])`` when neither
    file carries a usable date.
    """
    stamps = []
    for df, col in ((din, 'Intake Date'), (dout, 'Outcome Date')):
        if df is not None and col in df.columns:
            s = pd.to_datetime(df[col], errors='coerce').dropna()
            if not s.empty:
                stamps.append(s)
    if not stamps:
        return None, 0, []
    combined = pd.concat(stamps)
    start = combined.min()
    n_years = _bucket_span(combined, start)
    return start, n_years, [f'Year {i}' for i in range(1, n_years + 1)]


def _month_occurrences(dates: pd.Series) -> pd.Series:
    """How many times each calendar month occurs in the export, fractionally.

    A month that is only half covered by the export counts as 0.5, so the
    "average per occurrence" figures below are not skewed by partial edges.
    """
    all_days = pd.date_range(dates.min(), dates.max(), freq='D')
    weights = pd.DataFrame({'date': all_days})
    weights['month'] = weights['date'].dt.month
    weights['w'] = 1 / weights['date'].dt.days_in_month
    return weights.groupby('month')['w'].sum().reindex(range(1, 13), fill_value=0)


def _seg_filter(df, seg):
    return df if seg == 'Overall' else df[df['Species_Group'] == seg]


def _segment_table(df, year_labels, value_fn):
    """Overall/Cat/Dog x year table built by applying `value_fn` per segment."""
    return pd.DataFrame({
        seg: value_fn(_seg_filter(df, seg)).reindex(year_labels)
        for seg in SPECIES_SEGMENTS
    })


def _segment_line(table, title, y_title, pct=False, height=430):
    """Overall/Cat/Dog trend lines. Every point is directly labelled — the
    palette's aqua/yellow/magenta steps sit under 3:1 contrast, so identity
    never rests on colour alone."""
    table = table.copy()
    table.index = pd.Index(table.index, name='Year')
    tidy = table.reset_index().melt(id_vars='Year', var_name='Segment', value_name='Value')
    fig = px.line(
        tidy, x='Year', y='Value', color='Segment', markers=True,
        title=title, color_discrete_map=SPECIES_COLORS,
        category_orders={'Segment': SPECIES_SEGMENTS},
        text=tidy['Value'].map(lambda v: '' if pd.isna(v) else (f'{v:.1f}%' if pct else f'{v:,.0f}')),
    )
    fig.update_traces(line_width=2, marker_size=9, textposition='top center',
                      textfont_size=10, cliponaxis=False)
    fig.update_layout(yaxis_title=y_title, xaxis_title='', hovermode='x unified')
    if pct:
        fig.update_yaxes(ticksuffix='%')
    return apply_layout(fig, height=height)


def _stacked_share(ct, order, colors, title, y_title, height=460, pct=True):
    """Stacked bar with in-segment labels; segments under 4% are left unlabelled
    rather than squeezed."""
    frame = ct.reindex(columns=[c for c in order if c in ct.columns]).fillna(0)
    if pct:
        frame = frame.div(frame.sum(axis=1).replace(0, np.nan), axis=0) * 100
    frame.index = pd.Index(frame.index, name='Year')
    tidy = frame.reset_index().melt(id_vars='Year', var_name='Category', value_name='Value')
    tidy['Label'] = tidy['Value'].map(
        lambda v: (f'{v:.1f}%' if pct else f'{v:,.0f}') if v >= (4 if pct else frame.values.max() * 0.04) else ''
    )
    fig = px.bar(
        tidy, x='Year', y='Value', color='Category', text='Label',
        title=title, color_discrete_map=colors,
        category_orders={'Category': [c for c in order if c in frame.columns]},
    )
    fig.update_traces(textposition='inside', insidetextanchor='middle',
                      textfont=dict(size=10, color='white'),
                      marker_line_color='white', marker_line_width=2)
    fig.update_layout(barmode='stack', yaxis_title=y_title, xaxis_title='')
    if pct:
        fig.update_yaxes(ticksuffix='%', range=[0, 100])
    return apply_layout(fig, height=height)


def _table_expander(label, frame):
    with st.expander(label):
        st.dataframe(frame, use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab: Intake & Outcome                                                       #
# --------------------------------------------------------------------------- #
def intake_outcome_section(din, dout, year_labels, start, n_years):
    st.subheader('Intake & Outcome Macro View')
    st.caption(
        f'Rolling 12-month periods anchored to {start.date()} — **not** calendar years. '
        f'The export spans {n_years} such periods; partial calendar years at either end '
        'would otherwise read as a collapse in volume.'
    )

    # --- KPI row ----------------------------------------------------------- #
    cols = st.columns(4)
    cols[0].metric('Intake records', f'{len(din):,}' if din is not None else '—')
    cols[1].metric('Outcome records', f'{len(dout):,}' if dout is not None else '—')

    if dout is not None and 'Outcome Type' in dout.columns:
        live = (~dout['Outcome Type'].isin(NON_LIVE_OUTCOMES)).mean() * 100
        adopt = (dout['Outcome Type'] == 'Adoption').mean() * 100
        cols[2].metric('Live release rate', f'{live:.1f}%',
                       help='Outcomes that are not euthanasia, died in custody, or lost/stolen.')
        cols[3].metric('Adoption share of outcomes', f'{adopt:.1f}%')
    elif din is not None and 'Intake Type' in din.columns:
        surr = (din['Intake Type'] == 'Owner Surrender').mean() * 100
        cols[2].metric('Owner surrender share', f'{surr:.1f}%')

    # --- Annual volume trends ---------------------------------------------- #
    if din is not None:
        q1 = _segment_table(din, year_labels,
                            lambda d: d.groupby('YearLabel').size()).fillna(0)
        st.plotly_chart(_segment_line(q1, 'Annual Intake Volume', 'Animals taken in'),
                        use_container_width=True)
        _table_expander('Intake counts by year', q1.astype(int))

    if dout is not None:
        q2 = _segment_table(dout, year_labels,
                            lambda d: d.groupby('YearLabel').size()).fillna(0)
        st.plotly_chart(_segment_line(q2, 'Annual Outcome Volume', 'Animals leaving'),
                        use_container_width=True)
        _table_expander('Outcome counts by year', q2.astype(int))

    # --- Intake composition ------------------------------------------------ #
    if din is not None:
        ct = pd.crosstab(din['YearLabel'], din['Species_Group']).reindex(year_labels).fillna(0)
        ct.index.name = 'Year'
        left, right = st.columns(2)
        left.plotly_chart(
            _stacked_share(ct, SPECIES_GROUPS, SPECIES_COLORS,
                           'Intake Mix by Species', 'Share of intake'),
            use_container_width=True)
        right.plotly_chart(
            _stacked_share(ct, SPECIES_GROUPS, SPECIES_COLORS,
                           'Intake Volume by Species', 'Animals', pct=False),
            use_container_width=True)
        _table_expander('Intake composition by year', ct.astype(int))

    # --- Where animals come from ------------------------------------------- #
    if din is not None and 'Intake Type' in din.columns:
        ch = pd.crosstab(din['YearLabel'], din['Intake Type']).reindex(year_labels).fillna(0)
        ch.index.name = 'Year'
        extra = [c for c in ch.columns if c not in INTAKE_ORDER]
        st.plotly_chart(
            _stacked_share(ch, INTAKE_ORDER + extra, INTAKE_COLORS,
                           'Where Animals Come From — Intake Channel Mix by Year',
                           'Share of intake'),
            use_container_width=True)
        _table_expander('Intake channel counts by year', ch.astype(int))

    # --- Where animals go -------------------------------------------------- #
    if dout is not None and 'Outcome Type' in dout.columns:
        oc = pd.crosstab(dout['YearLabel'], dout['Outcome Type']).reindex(year_labels).fillna(0)
        oc.index.name = 'Year'
        extra = [c for c in oc.columns if c not in OUTCOME_ORDER]
        st.plotly_chart(
            _stacked_share(oc, OUTCOME_ORDER + extra, OUTCOME_COLORS,
                           'Where Animals Go — Outcome Mix by Year', 'Share of outcomes'),
            use_container_width=True)
        _table_expander('Outcome counts by year', oc.astype(int))

        # Adoption share and live release rate — the two headline quality metrics.
        adopt_tbl = _segment_table(
            dout, year_labels,
            lambda d: (d[d['Outcome Type'] == 'Adoption'].groupby('YearLabel').size()
                       / d.groupby('YearLabel').size() * 100).round(1),
        )
        st.plotly_chart(
            _segment_line(adopt_tbl, 'Adoption Share of Outcomes', '% of outcomes', pct=True),
            use_container_width=True)

        live_tbl = _segment_table(
            dout, year_labels,
            lambda d: (d[~d['Outcome Type'].isin(NON_LIVE_OUTCOMES)].groupby('YearLabel').size()
                       / d.groupby('YearLabel').size() * 100).round(1),
        )
        st.plotly_chart(
            _segment_line(live_tbl, 'Live Release Rate', '% released alive', pct=True),
            use_container_width=True)
        st.caption(
            'Live release excludes euthanasia, died in custody, and lost/stolen. '
            'It is the metric most funders and Best Friends-style benchmarks ask for.'
        )
        _table_expander('Adoption share & live release rate by year',
                        adopt_tbl.join(live_tbl, lsuffix=' — adoption %', rsuffix=' — live release %'))

    # --- Intake vs outcome balance ----------------------------------------- #
    if din is not None and dout is not None:
        balance = pd.DataFrame({
            'Intake': din.groupby('YearLabel').size().reindex(year_labels).fillna(0),
            'Outcome': dout.groupby('YearLabel').size().reindex(year_labels).fillna(0),
        })
        balance['Net (intake − outcome)'] = balance['Intake'] - balance['Outcome']
        fig = go.Figure()
        fig.add_bar(x=balance.index, y=balance['Intake'], name='Intake',
                    marker_color=SPECIES_COLORS['Dog'], marker_line_color='white',
                    marker_line_width=2,
                    text=balance['Intake'].map('{:,.0f}'.format), textposition='outside')
        fig.add_bar(x=balance.index, y=balance['Outcome'], name='Outcome',
                    marker_color=SPECIES_COLORS['Cat'], marker_line_color='white',
                    marker_line_width=2,
                    text=balance['Outcome'].map('{:,.0f}'.format), textposition='outside')
        fig.update_layout(barmode='group', title='Intake vs Outcome Volume by Year',
                          yaxis_title='Animals')
        st.plotly_chart(apply_layout(fig, height=440), use_container_width=True)
        st.caption(
            'A year where intake runs persistently above outcome means the in-care '
            'population is growing — the leading indicator for kennel and foster capacity strain.'
        )
        _table_expander('Intake vs outcome balance', balance.astype(int))


# --------------------------------------------------------------------------- #
# Tab: Seasonality                                                            #
# --------------------------------------------------------------------------- #
def _monthly_normalized(df, date_col, occurrences, by_species=True):
    """Average count per occurrence of each calendar month."""
    if by_species:
        out = {}
        for seg in ('Cat', 'Dog'):
            d = df[df['Species_Group'] == seg]
            monthly = d.groupby('Month').size().reindex(range(1, 13)).fillna(0)
            out[seg] = (monthly / occurrences.replace(0, np.nan)).round(1)
        return pd.DataFrame(out)
    monthly = df.groupby('Month').size().reindex(range(1, 13)).fillna(0)
    return (monthly / occurrences.replace(0, np.nan)).round(1)


def _monthly_chart(frame, title, y_title):
    frame = frame.copy()
    frame.index = pd.Index([MONTH_ABBR[m - 1] for m in frame.index], name='Month')
    tidy = frame.reset_index().melt(id_vars='Month', var_name='Species', value_name='Value')
    fig = px.line(
        tidy, x='Month', y='Value', color='Species', markers=True, title=title,
        color_discrete_map=SPECIES_COLORS, category_orders={'Month': MONTH_ABBR},
        text=tidy['Value'].map(lambda v: '' if pd.isna(v) else f'{v:.0f}'),
    )
    fig.update_traces(line_width=2, marker_size=9, textposition='top center',
                      textfont_size=10, cliponaxis=False)
    fig.update_layout(yaxis_title=y_title, xaxis_title='', hovermode='x unified')
    return apply_layout(fig, height=430)


def seasonality_section(din, dout, year_labels):
    st.subheader('Seasonality — When the Pressure Lands')
    st.caption(
        'Months are **period-normalized**: each calendar day is weighted by '
        '1/(days in its month), so a half-covered month at the edge of the export '
        'counts as half a month rather than a whole one. Values are the average '
        'count per occurrence of that month.'
    )

    index_frames = {}

    if din is not None:
        occ = _month_occurrences(din['Intake Date'])
        intake_monthly = _monthly_normalized(din, 'Intake Date', occ)
        st.plotly_chart(
            _monthly_chart(intake_monthly, 'Monthly Intake — Cat vs Dog',
                           'Avg. intakes per occurrence of that month'),
            use_container_width=True)
        index_frames['Intake'] = _monthly_normalized(din, 'Intake Date', occ, by_species=False)

        if 'Intake Type' in din.columns:
            surr = din[din['Intake Type'] == 'Owner Surrender']
            if not surr.empty:
                index_frames['Owner surrender'] = _monthly_normalized(
                    surr, 'Intake Date', occ, by_species=False)

    if dout is not None and 'Outcome Type' in dout.columns:
        occ_o = _month_occurrences(dout['Outcome Date'])
        adopt = dout[dout['Outcome Type'] == 'Adoption']
        if not adopt.empty:
            adopt_monthly = _monthly_normalized(adopt, 'Outcome Date', occ_o)
            st.plotly_chart(
                _monthly_chart(adopt_monthly, 'Monthly Adoptions — Cat vs Dog',
                               'Avg. adoptions per occurrence of that month'),
                use_container_width=True)
            index_frames['Adoption'] = _monthly_normalized(adopt, 'Outcome Date', occ_o,
                                                           by_species=False)

    # --- Seasonal index ---------------------------------------------------- #
    # Each series rescaled so its own annual average = 100. This is the chart
    # that answers "does supply peak before or after demand?" — the raw counts
    # can't, because intake and adoption run at different absolute volumes and a
    # dual axis would be a lie.
    if index_frames:
        idx = pd.DataFrame({
            name: (s / s.mean() * 100).round(1) for name, s in index_frames.items()
        })
        idx.index = pd.Index([MONTH_ABBR[m - 1] for m in idx.index], name='Month')
        tidy = idx.reset_index().melt(id_vars='Month', var_name='Series', value_name='Index')
        series_colors = {'Intake': SPECIES_COLORS['Dog'], 'Adoption': '#1baf7a',
                         'Owner surrender': SPECIES_COLORS['Cat']}
        fig = px.line(tidy, x='Month', y='Index', color='Series', markers=True,
                      title='Seasonal Index — Intake vs Adoption vs Surrender (100 = own annual average)',
                      color_discrete_map=series_colors,
                      category_orders={'Month': MONTH_ABBR})
        fig.add_hline(y=100, line_dash='dash', line_color='#b7b5ae',
                      annotation_text='annual average', annotation_position='top left')
        fig.update_traces(line_width=2, marker_size=9)
        fig.update_layout(yaxis_title='Index (100 = annual average)', xaxis_title='',
                          hovermode='x unified')
        st.plotly_chart(apply_layout(fig, height=450), use_container_width=True)
        st.caption(
            'Indexing each series to its own average puts three different-sized flows on '
            'one honest axis. A month where the intake line runs above 100 while adoption '
            'runs below is a month where the shelter fills up faster than it empties.'
        )
        _table_expander('Seasonal index values', idx)

    # --- Month x Year heatmap ---------------------------------------------- #
    for label, df, date_col in (('Intake', din, 'Intake Date'), ('Outcome', dout, 'Outcome Date')):
        if df is None:
            continue
        heat = pd.crosstab(df['YearLabel'], df['Month']).reindex(year_labels).fillna(0)
        heat = heat.reindex(columns=range(1, 13), fill_value=0)
        heat.columns = MONTH_ABBR
        fig = px.imshow(
            heat, text_auto=True, aspect='auto', color_continuous_scale='Blues',
            title=f'{label} Volume — Month by Rolling Year',
            labels=dict(x='', y='', color=f'{label}s'),
        )
        fig.update_traces(textfont_size=10)
        fig.update_xaxes(side='top')
        st.plotly_chart(apply_layout(fig, height=90 + 46 * len(heat)), use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab: Owner surrenders                                                       #
# --------------------------------------------------------------------------- #
def surrender_section(din, year_labels):
    st.subheader('Owner Surrender Analysis')

    if 'Intake Type' not in din.columns:
        st.warning('The intake file has no "Intake Type" column — surrender analysis needs it.')
        return

    surr = din[din['Intake Type'] == 'Owner Surrender']
    if surr.empty:
        st.info('No owner-surrender records found in the intake file.')
        return

    st.caption(
        'Population = `Intake Type == "Owner Surrender"`. `Intake Sub-type` is used only to '
        'break down *why* within that population — never to reclassify or exclude a record '
        'from the surrender count.'
    )

    # --- KPI row ----------------------------------------------------------- #
    share = len(surr) / len(din) * 100
    cat_share = (surr['Species_Group'] == 'Cat').mean() * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Owner surrenders', f'{len(surr):,}')
    c2.metric('Share of all intake', f'{share:.1f}%')
    c3.metric('Cats', f'{cat_share:.0f}% of surrenders')
    c4.metric('Dogs', f'{(surr["Species_Group"] == "Dog").mean() * 100:.0f}% of surrenders')

    # --- Headcount and share trend (two charts, never a dual axis) ---------- #
    count_tbl = _segment_table(surr, year_labels,
                               lambda d: d.groupby('YearLabel').size()).fillna(0)
    share_tbl = _segment_table(
        din, year_labels,
        lambda d: (d[d['Intake Type'] == 'Owner Surrender'].groupby('YearLabel').size()
                   / d.groupby('YearLabel').size() * 100).round(1),
    )
    left, right = st.columns(2)
    left.plotly_chart(_segment_line(count_tbl, 'Owner Surrenders — Headcount', 'Animals'),
                      use_container_width=True)
    right.plotly_chart(_segment_line(share_tbl, 'Owner Surrenders — Share of Intake',
                                     '% of that year\'s intake', pct=True),
                       use_container_width=True)
    st.caption(
        'Headcount and share are shown as two charts on their own axes rather than one '
        'dual-axis chart — a rising headcount with a flat share means the shelter got '
        'busier overall, not that surrenders got worse.'
    )
    _table_expander('Surrender headcount and share by year',
                    count_tbl.astype(int).join(share_tbl, lsuffix=' (n)', rsuffix=' (%)'))

    # --- Seasonality ------------------------------------------------------- #
    occ = _month_occurrences(din['Intake Date'])
    st.plotly_chart(
        _monthly_chart(_monthly_normalized(surr, 'Intake Date', occ),
                       'Monthly Owner Surrenders — Cat vs Dog',
                       'Avg. surrenders per occurrence of that month'),
        use_container_width=True)

    # --- Why people surrender ---------------------------------------------- #
    if 'Intake Sub-type' not in surr.columns:
        st.info('No "Intake Sub-type" column — surrender reasons unavailable.')
        return

    top_n = st.slider('Reasons to show', 5, 20, 10, key='surrender_top_n')
    reason_frames = []
    for sp in ('Cat', 'Dog'):
        sub = surr[surr['Species_Group'] == sp]
        if sub.empty:
            continue
        vc = sub['Intake Sub-type'].value_counts(dropna=True).head(top_n)
        reason_frames.append(pd.DataFrame({
            'Reason': vc.index, 'Animals': vc.values,
            'Share': (vc.values / len(sub) * 100).round(1),
            'Species': sp, 'n': len(sub),
        }))

    if reason_frames:
        reasons = pd.concat(reason_frames, ignore_index=True)
        reasons['Label'] = reasons.apply(
            lambda r: f'{r["Animals"]:,} ({r["Share"]:.1f}%)', axis=1)
        fig = px.bar(
            reasons.sort_values('Animals'), x='Animals', y='Reason', orientation='h',
            facet_col='Species', color='Species', text='Label',
            color_discrete_map=SPECIES_COLORS,
            title=f'Top {top_n} Owner-Surrender Reasons — Cat vs Dog',
        )
        fig.update_traces(textposition='outside', textfont_size=10, cliponaxis=False,
                          marker_line_color='white', marker_line_width=1)
        fig.update_yaxes(matches=None, showticklabels=True, title='')
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
        fig.update_xaxes(matches=None)
        st.plotly_chart(apply_layout(fig, height=120 + 34 * top_n), use_container_width=True)
        st.caption(
            'Percentages are of that species\' surrender total, so cat and dog reasons are '
            'comparable even though the volumes differ.'
        )
        _table_expander('Surrender reasons', reasons.drop(columns='Label'))

        # --- Has the reason mix shifted? ----------------------------------- #
        top_reasons = (surr['Intake Sub-type'].value_counts().head(6).index.tolist())
        mix = surr[surr['Intake Sub-type'].isin(top_reasons)]
        ct = pd.crosstab(mix['YearLabel'], mix['Intake Sub-type']).reindex(year_labels).fillna(0)
        ct.index.name = 'Year'
        ct = ct.reindex(columns=top_reasons, fill_value=0)
        share_ct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100
        tidy = share_ct.reset_index().melt(id_vars='Year', var_name='Reason', value_name='Share')
        fig = px.line(tidy, x='Year', y='Share', color='Reason', markers=True,
                      title='Surrender Reason Mix Over Time (top 6 reasons)',
                      color_discrete_sequence=['#2a78d6', '#eb6834', '#1baf7a',
                                               '#eda100', '#e87ba4', '#008300'])
        fig.update_traces(line_width=2, marker_size=8)
        fig.update_layout(yaxis_title='% of that year\'s surrenders', xaxis_title='',
                          hovermode='x unified')
        fig.update_yaxes(ticksuffix='%')
        st.plotly_chart(apply_layout(fig, height=450), use_container_width=True)
        st.caption(
            'A reason that climbs year over year is a programme gap the shelter can act on — '
            'housing-driven surrenders point at landlord/pet-deposit assistance, behaviour at '
            'free training clinics, cost at a pet food bank or low-cost vet clinic.'
        )
        _table_expander('Reason mix by year (%)', share_ct.round(1))


# --------------------------------------------------------------------------- #
# Tab: Flow diagnostics                                                       #
# --------------------------------------------------------------------------- #
def flow_section(dout, year_labels):
    st.subheader('Flow Diagnostics — Intake Channel to Outcome')

    required = {'Intake Type', 'Outcome Type'}
    if not required.issubset(dout.columns):
        st.warning('The outcome file needs both "Intake Type" and "Outcome Type" columns for this view.')
        return

    st.caption(
        'Each outcome record still carries the channel the animal arrived through, so the '
        'two can be crossed. This is where a change in the headline adoption rate gets '
        'traced back to a specific channel rather than guessed at.'
    )

    c1, c2 = st.columns(2)
    species_opts = ['All'] + [s for s in SPECIES_GROUPS if s in dout['Species_Group'].unique()]
    species = c1.selectbox('Species', species_opts, key='flow_species')
    year_opts = ['All years'] + year_labels
    year = c2.selectbox('Period', year_opts, index=0, key='flow_year')

    scoped = dout if species == 'All' else dout[dout['Species_Group'] == species]
    if year != 'All years':
        scoped = scoped[scoped['YearLabel'] == year]

    if scoped.empty:
        st.info('No records for that combination.')
        return

    # --- Sankey ------------------------------------------------------------ #
    ct = pd.crosstab(scoped['Intake Type'], scoped['Outcome Type'])
    links = ct.reset_index().melt(id_vars='Intake Type', var_name='Outcome Type',
                                  value_name='count')
    links = links[links['count'] > 0]

    sources = [c for c in INTAKE_ORDER if c in ct.index] + \
              [c for c in ct.index if c not in INTAKE_ORDER]
    targets = [c for c in OUTCOME_ORDER if c in ct.columns] + \
              [c for c in ct.columns if c not in OUTCOME_ORDER]
    labels = sources + targets
    node_colors = ([INTAKE_COLORS.get(c, OTHER) for c in sources]
                   + [OUTCOME_COLORS.get(c, OTHER) for c in targets])

    def _rgba(hex_color, alpha):
        h = hex_color.lstrip('#')
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return f'rgba({r},{g},{b},{alpha})'

    fig = go.Figure(go.Sankey(
        node=dict(label=[f'{l} ({int(v):,})' for l, v in
                         zip(labels,
                             [ct.loc[s].sum() for s in sources] + [ct[t].sum() for t in targets])],
                  color=node_colors, pad=18, thickness=18,
                  line=dict(color='white', width=1)),
        link=dict(
            source=links['Intake Type'].map(labels.index).tolist(),
            target=links['Outcome Type'].map(labels.index).tolist(),
            value=links['count'].tolist(),
            # Colour each ribbon by the outcome it lands in: the question this
            # chart answers is "which channel feeds adoption", so the outcome is
            # the entity worth tracking across the diagram.
            color=[_rgba(OUTCOME_COLORS.get(t, OTHER), 0.45) for t in links['Outcome Type']],
            hovertemplate='%{source.label} → %{target.label}<br>%{value:,} animals<extra></extra>',
        ),
    ))
    fig.update_layout(
        title=f'Intake Channel → Outcome  ({species}, {year}; n={len(scoped):,})',
        font=dict(size=12, color='#2f2e2b'), height=520,
        margin=dict(l=10, r=10, t=70, b=10), template='plotly_white',
    )
    st.plotly_chart(fig, use_container_width=True)
    _table_expander('Intake × outcome counts', ct)

    # --- Outcome mix per channel ------------------------------------------- #
    order_in = [c for c in INTAKE_ORDER if c in ct.index] + \
               [c for c in ct.index if c not in INTAKE_ORDER]
    share = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100
    share = share.reindex(order_in)
    share.index.name = 'Intake channel'
    cols_order = [c for c in OUTCOME_ORDER if c in share.columns] + \
                 [c for c in share.columns if c not in OUTCOME_ORDER]
    tidy = share.reset_index().melt(id_vars='Intake channel', var_name='Outcome',
                                    value_name='Share')
    tidy['Label'] = tidy['Share'].map(lambda v: f'{v:.0f}%' if v >= 6 else '')
    fig = px.bar(tidy, y='Intake channel', x='Share', color='Outcome', orientation='h',
                 text='Label', color_discrete_map=OUTCOME_COLORS,
                 category_orders={'Outcome': cols_order, 'Intake channel': order_in},
                 title='Outcome Mix by Intake Channel')
    fig.update_traces(textposition='inside', insidetextanchor='middle',
                      textfont=dict(size=10, color='white'),
                      marker_line_color='white', marker_line_width=2)
    fig.update_layout(barmode='stack', xaxis_title='% of that channel\'s outcomes',
                      yaxis_title='')
    fig.update_xaxes(ticksuffix='%', range=[0, 100])
    st.plotly_chart(apply_layout(fig, height=110 + 52 * len(order_in)), use_container_width=True)
    st.caption(
        'Read across a row: a channel where the adoption band is thin and the '
        'return-to-owner band is wide is not a failing channel — it is doing a different job.'
    )
    _table_expander('Outcome mix by channel (%)', share.round(1))

    # --- Channel contribution to the adoption rate ------------------------- #
    scoped_all_years = dout if species == 'All' else dout[dout['Species_Group'] == species]
    contrib = {}
    for yl in year_labels:
        sub = scoped_all_years[scoped_all_years['YearLabel'] == yl]
        if sub.empty:
            continue
        ct_y = pd.crosstab(sub['Intake Type'], sub['Outcome Type'])
        total = len(sub)
        adopt_col = ct_y['Adoption'] if 'Adoption' in ct_y.columns else pd.Series(dtype=float)
        contrib[yl] = (adopt_col / total * 100)
    if contrib:
        contrib_df = pd.DataFrame(contrib).T.fillna(0)
        contrib_df.index.name = 'Year'
        order = [c for c in INTAKE_ORDER if c in contrib_df.columns] + \
                [c for c in contrib_df.columns if c not in INTAKE_ORDER]
        st.plotly_chart(
            _stacked_share(contrib_df, order, INTAKE_COLORS,
                           f'What Drives the Adoption Rate — Contribution by Intake Channel ({species})',
                           'Percentage points of the adoption rate', pct=False),
            use_container_width=True)
        st.caption(
            'Bar heights sum to that year\'s overall adoption rate. A drop in the total '
            'is attributable to whichever band shrank — that is the channel to investigate, '
            'and often the drop is a *shift* (e.g. more strays reclaimed by their owners) '
            'rather than a failure to place animals.'
        )
        _table_expander('Adoption-rate contribution by channel (percentage points)',
                        contrib_df.round(2))

    # --- Single-channel deep dive ------------------------------------------ #
    channel_opts = order_in
    channel = st.selectbox('Trace one channel across the years', channel_opts,
                           index=channel_opts.index('Stray In') if 'Stray In' in channel_opts else 0,
                           key='flow_channel')
    chan_df = scoped_all_years[scoped_all_years['Intake Type'] == channel]
    if not chan_df.empty:
        ct_c = pd.crosstab(chan_df['YearLabel'], chan_df['Outcome Type'])
        ct_c = ct_c.reindex(year_labels).fillna(0)
        ct_c.index.name = 'Year'
        cols_c = [c for c in OUTCOME_ORDER if c in ct_c.columns] + \
                 [c for c in ct_c.columns if c not in OUTCOME_ORDER]
        st.plotly_chart(
            _stacked_share(ct_c, cols_c, OUTCOME_COLORS,
                           f'"{channel}" Animals — Outcome Mix by Year ({species})',
                           'Share of that channel\'s outcomes'),
            use_container_width=True)
        _table_expander(f'{channel} outcomes by year', ct_c.astype(int))


# --------------------------------------------------------------------------- #
# Foster proxy (rendered inside the Foster tab)                                #
# --------------------------------------------------------------------------- #
def foster_proxy_section(dout, year_labels):
    st.markdown('---')
    st.subheader('Foster Pipeline (from outcome records)')
    st.warning(
        'Directional proxy, not a census. The source system has no dedicated foster '
        'outcome type, so this counts only the outcome sub-types that prove an animal '
        f'passed through foster care — {", ".join(FOSTER_SUBTYPES)}. Animals fostered '
        'and then returned to the shelter for a different final outcome are invisible here.',
        icon='⚠️',
    )

    if 'Outcome Sub-type' not in dout.columns:
        st.info('The outcome file has no "Outcome Sub-type" column.')
        return

    foster = dout[dout['Outcome Sub-type'].isin(FOSTER_SUBTYPES)]
    if foster.empty:
        st.info('No foster-related outcome sub-types found in this export.')
        return

    subtype_by_year = pd.crosstab(foster['YearLabel'], foster['Outcome Sub-type'])
    subtype_by_year = subtype_by_year.reindex(year_labels).fillna(0)
    subtype_by_year = subtype_by_year.reindex(columns=FOSTER_SUBTYPES, fill_value=0)
    subtype_by_year.index.name = 'Year'

    totals = subtype_by_year.sum(axis=1)
    nonzero = totals[totals > 0]
    growth = (nonzero.iloc[-1] / nonzero.iloc[0]) if len(nonzero) >= 2 and nonzero.iloc[0] else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric('Foster-related outcomes', f'{len(foster):,}')
    c2.metric('Share of all outcomes', f'{len(foster) / len(dout) * 100:.1f}%')
    c3.metric('Growth across the window',
              f'{growth:.1f}×' if pd.notna(growth) else '—',
              help='First to last rolling year with any foster-related volume.')

    subtype_colors = {'Foster to Adopt': '#2a78d6', 'Foster Fail': '#eb6834',
                      'Died in Foster': OTHER}
    st.plotly_chart(
        _stacked_share(subtype_by_year, FOSTER_SUBTYPES, subtype_colors,
                       'Foster Sub-type Volume by Year', 'Records', pct=False),
        use_container_width=True)
    st.caption(
        '"Foster to Adopt" and "Foster Fail" both end with a foster household keeping the '
        'animal, but they are not the same label renamed: if "Foster Fail" volume held '
        'steady as "Foster to Adopt" appeared, the two are concurrent tracks. Confirm the '
        'operational definitions with shelter staff before presenting either as settled.'
    )
    _table_expander('Foster sub-type counts by year', subtype_by_year.astype(int))

    left, right = st.columns(2)
    mix = foster['Outcome Sub-type'].value_counts()
    fig = px.pie(values=mix.values, names=mix.index, title='Foster Sub-type Mix', hole=0.45,
                 color=mix.index, color_discrete_map=subtype_colors)
    fig.update_traces(textinfo='label+percent', textposition='outside',
                      marker_line_color='white', marker_line_width=2)
    left.plotly_chart(apply_layout(fig, height=420), use_container_width=True)

    sp = foster['Species_Group'].value_counts().reindex(SPECIES_GROUPS).fillna(0)
    fig2 = px.bar(x=sp.index, y=sp.values, color=sp.index, title='Foster-Related Records by Species',
                  color_discrete_map=SPECIES_COLORS, text=sp.values.astype(int))
    fig2.update_traces(textposition='outside', cliponaxis=False,
                       marker_line_color='white', marker_line_width=2)
    fig2.update_layout(xaxis_title='', yaxis_title='Records', showlegend=False)
    right.plotly_chart(apply_layout(fig2, height=420), use_container_width=True)
    st.caption(f'Sample is small (n={len(foster):,}) — read the species split as directional.')
