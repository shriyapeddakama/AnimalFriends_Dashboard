"""ZIP-level geographic enrichment for the Animal Friends dashboard.

Ported from the Part 4 adoption & foster geography notebook. Every adopter ZIP is
resolved offline to a county and an approximate centroid via the ``zipcodes``
package, then a haversine distance to the shelter is attached. No geocoding API
is called — nothing about an adopter leaves the machine.

**Precision:** this is ZIP-centroid accuracy, which is right for county-level and
regional-distance questions and wrong for anything block-level. Distances are
"how far is your ZIP from ours", not driving distance.
"""

from math import atan2, cos, radians, sin, sqrt

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from viz_theme import ACCENT, SPECIES_COLORS, apply_layout

# Animal Friends, 562 Camp Horne Rd, Pittsburgh, PA 15202.
SHELTER_NAME = 'Animal Friends — 562 Camp Horne Rd, Pittsburgh PA'
SHELTER_ZIP = '15202'

# Adopters beyond this are almost always relocations or long-distance transfers.
# They are counted in every statistic but clipped out of the distance histogram,
# where a handful of 2,000-mile records would flatten the local detail.
CLIP_MI = 100
REGION_MI = 60


def geocoding_available() -> bool:
    try:
        import zipcodes  # noqa: F401
        return True
    except ImportError:
        return False


@st.cache_data(show_spinner=False)
def geocode_zips(zips: tuple) -> pd.DataFrame:
    """Resolve a tuple of 5-digit ZIPs to county / lat / lon / city / state."""
    import zipcodes

    rows = []
    for z in zips:
        if not isinstance(z, str) or len(z) != 5 or not z.isdigit():
            continue
        matches = zipcodes.matching(z)
        if not matches:
            continue
        m = matches[0]
        try:
            lat, lon = float(m['lat']), float(m['long'])
        except (TypeError, ValueError, KeyError):
            lat = lon = np.nan
        rows.append({
            'Zip_Clean': z,
            'County': m.get('county'),
            'Geo_State': m.get('state'),
            'Geo_City': m.get('city'),
            'lat': lat,
            'lon': lon,
        })
    return pd.DataFrame(rows, columns=['Zip_Clean', 'County', 'Geo_State', 'Geo_City', 'lat', 'lon'])


def _haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


@st.cache_data(show_spinner=False)
def enrich_with_geography(df: pd.DataFrame) -> pd.DataFrame:
    """Attach County / lat / lon / distance-from-shelter to an adoption frame."""
    import zipcodes

    lookup = geocode_zips(tuple(sorted(df['Zip_Clean'].dropna().astype(str).unique())))
    shelter = zipcodes.matching(SHELTER_ZIP)[0]
    s_lat, s_lon = float(shelter['lat']), float(shelter['long'])

    lookup['Distance_mi'] = lookup.apply(
        lambda r: _haversine_miles(r['lat'], r['lon'], s_lat, s_lon)
        if pd.notna(r['lat']) else np.nan,
        axis=1,
    ).round(1)

    return df.merge(lookup, on='Zip_Clean', how='left')


def shelter_coords():
    import zipcodes
    m = zipcodes.matching(SHELTER_ZIP)[0]
    return float(m['lat']), float(m['long'])


# --------------------------------------------------------------------------- #
# Section renderer                                                            #
# --------------------------------------------------------------------------- #
def catchment_section(df: pd.DataFrame):
    """County, distance, and catchment-area analysis for the adoption dataset."""
    st.markdown('---')
    st.subheader('Catchment Area — Counties & Distance from the Shelter')

    if not geocoding_available():
        st.info(
            'Install the `zipcodes` package to enable county and distance analysis: '
            '`pip install zipcodes` (it is offline — no geocoding API is called).'
        )
        return

    with st.spinner('Resolving ZIP codes to counties…'):
        geo = enrich_with_geography(df)

    resolved = geo.dropna(subset=['County'])
    if resolved.empty:
        st.warning('No ZIP codes in this dataset could be resolved to a county.')
        return

    coverage = len(resolved) / len(geo)
    dist = resolved['Distance_mi'].dropna()

    st.caption(
        f'{len(resolved):,} of {len(geo):,} adoption records ({coverage:.0%}) carry a ZIP that '
        f'resolves to a US county. Everything below is computed on that base. '
        f'Distances are straight-line from ZIP centroid to {SHELTER_NAME} — regional accuracy, '
        'not street-level.'
    )

    # --- KPI row ----------------------------------------------------------- #
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Counties reached', f'{resolved["County"].nunique():,}')
    c2.metric('Median adopter distance', f'{dist.median():.1f} mi' if not dist.empty else '—')
    c3.metric('Within 10 miles', f'{(dist <= 10).mean() * 100:.0f}%' if not dist.empty else '—')
    c4.metric('Within 25 miles', f'{(dist <= 25).mean() * 100:.0f}%' if not dist.empty else '—')

    # --- Counties ---------------------------------------------------------- #
    county_counts = resolved['County'].value_counts()
    top_counties = county_counts.head(12).sort_values()
    frame = pd.DataFrame({
        'County': top_counties.index,
        'Adoptions': top_counties.values,
        'Share': (top_counties.values / len(resolved) * 100),
    })
    frame['Label'] = frame.apply(lambda r: f'{r["Adoptions"]:,.0f}  ({r["Share"]:.0f}%)', axis=1)
    fig = px.bar(frame, x='Adoptions', y='County', orientation='h', text='Label',
                 title=f'Adoption Destinations by County (top 12 of {county_counts.size})',
                 color_discrete_sequence=[SPECIES_COLORS['Dog']])
    fig.update_traces(textposition='outside', textfont_size=10, cliponaxis=False,
                      marker_line_color='white', marker_line_width=1)
    fig.update_layout(yaxis_title='', xaxis_title='Adoptions')
    fig.update_xaxes(range=[0, frame['Adoptions'].max() * 1.22])
    st.plotly_chart(apply_layout(fig, height=460), use_container_width=True)

    lead = county_counts.index[0]
    st.caption(
        f'{lead} accounts for {county_counts.iloc[0] / len(resolved):.0%} of geocoded adoptions. '
        'Heavy concentration in the home county is normal and healthy for a regional shelter — '
        'the actionable question is which *adjacent* counties are thin relative to their population.'
    )

    # --- Distance distribution --------------------------------------------- #
    if not dist.empty:
        clipped = dist[dist <= CLIP_MI]
        beyond = int((dist > CLIP_MI).sum())
        fig = px.histogram(clipped, nbins=30, title='How Far Adopters Travel',
                           color_discrete_sequence=[SPECIES_COLORS['Dog']])
        fig.add_vline(x=float(dist.median()), line_dash='dash', line_color=ACCENT,
                      annotation_text=f'median {dist.median():.1f} mi',
                      annotation_position='top right')
        fig.update_traces(marker_line_color='white', marker_line_width=1)
        fig.update_layout(xaxis_title=f'Distance from shelter (miles, axis capped at {CLIP_MI})',
                          yaxis_title='Adoptions', showlegend=False)
        st.plotly_chart(apply_layout(fig, height=420), use_container_width=True)
        st.caption(
            f'{beyond:,} of {len(dist):,} geocoded adoptions ({beyond / len(dist):.1%}) are beyond '
            f'{CLIP_MI} miles and sit off the right of this axis — they are counted in the median '
            'and every percentage above, just not plotted, since a handful of cross-country '
            'relocations would otherwise compress all the local detail into one bar.'
        )

    # --- Distance by species ------------------------------------------------ #
    species_dist = resolved.dropna(subset=['Distance_mi'])
    top_species = species_dist['Species'].value_counts()
    keep = top_species[top_species >= 30].head(4).index.tolist()
    if keep and not species_dist.empty:
        sub = species_dist[species_dist['Species'].isin(keep)]
        fig = px.box(sub, x='Species', y='Distance_mi', color='Species',
                     points=False, title='Adopter Distance by Species',
                     color_discrete_map=SPECIES_COLORS,
                     color_discrete_sequence=['#2a78d6', '#eb6834', '#1baf7a', '#eda100'])
        fig.update_layout(xaxis_title='', yaxis_title='Distance from shelter (mi)',
                          showlegend=False)
        fig.update_yaxes(range=[0, min(CLIP_MI, float(sub['Distance_mi'].quantile(0.98)) * 1.1)])
        st.plotly_chart(apply_layout(fig, height=420), use_container_width=True)

        summary = sub.groupby('Species')['Distance_mi'].agg(
            median='median',
            within_10mi=lambda s: round((s <= 10).mean() * 100, 1),
            n='count',
        ).sort_values('n', ascending=False)
        st.caption('Species with fewer than 30 geocoded adoptions are excluded — too few to read.')
        with st.expander('Distance by species'):
            st.dataframe(summary, use_container_width=True)

    # --- County x species --------------------------------------------------- #
    if keep:
        ct = pd.crosstab(resolved['County'], resolved['Species'])
        ct = ct.reindex(columns=[s for s in keep if s in ct.columns], fill_value=0)
        ct['__total'] = ct.sum(axis=1)
        ct = ct.sort_values('__total', ascending=True).tail(8).drop(columns='__total')
        tidy = ct.reset_index().melt(id_vars='County', var_name='Species', value_name='Adoptions')
        fig = px.bar(tidy, x='Adoptions', y='County', color='Species', orientation='h',
                     title='County Mix by Species (top 8 counties)',
                     color_discrete_map=SPECIES_COLORS,
                     color_discrete_sequence=['#2a78d6', '#eb6834', '#1baf7a', '#eda100'],
                     category_orders={'Species': keep})
        fig.update_traces(marker_line_color='white', marker_line_width=2)
        fig.update_layout(barmode='stack', yaxis_title='', xaxis_title='Adoptions')
        st.plotly_chart(apply_layout(fig, height=440), use_container_width=True)
        with st.expander('County × species counts'):
            st.dataframe(ct, use_container_width=True)

    # --- Catchment over time ------------------------------------------------ #
    yearly = resolved.dropna(subset=['Distance_mi', 'Adoption_Year'])
    if yearly['Adoption_Year'].nunique() >= 2:
        med = yearly.groupby('Adoption_Year')['Distance_mi'].median().round(1)
        lead_share = (resolved.dropna(subset=['Adoption_Year'])
                      .groupby('Adoption_Year')['County']
                      .apply(lambda s: (s == lead).mean() * 100).round(1))

        left, right = st.columns(2)
        fig = px.line(x=med.index.astype(int), y=med.values, markers=True,
                      title='Median Adopter Distance by Year',
                      color_discrete_sequence=[SPECIES_COLORS['Dog']],
                      text=[f'{v:.1f}' for v in med.values])
        fig.update_traces(line_width=2, marker_size=9, textposition='top center',
                          textfont_size=10, cliponaxis=False)
        fig.update_layout(xaxis_title='', yaxis_title='Median distance (mi)')
        fig.update_xaxes(dtick=1)
        left.plotly_chart(apply_layout(fig, height=400), use_container_width=True)

        fig2 = px.line(x=lead_share.index.astype(int), y=lead_share.values, markers=True,
                       title=f'{lead} Share of Adoptions by Year',
                       color_discrete_sequence=[SPECIES_COLORS['Cat']],
                       text=[f'{v:.0f}%' for v in lead_share.values])
        fig2.update_traces(line_width=2, marker_size=9, textposition='top center',
                           textfont_size=10, cliponaxis=False)
        fig2.update_layout(xaxis_title='', yaxis_title=f'{lead} share')
        fig2.update_yaxes(ticksuffix='%')
        fig2.update_xaxes(dtick=1)
        right.plotly_chart(apply_layout(fig2, height=400), use_container_width=True)
        st.caption(
            'A widening median distance with a falling home-county share means the shelter is '
            'drawing from farther out — good for reach, but it also means longer round trips for '
            'meet-and-greets and returns, which is worth planning transport around.'
        )

    # --- Bubble map --------------------------------------------------------- #
    mapped = resolved.dropna(subset=['lat', 'lon'])
    if not mapped.empty:
        agg = (mapped.groupby(['Zip_Clean', 'Geo_City', 'County', 'lat', 'lon'])
               .size().reset_index(name='Adoptions'))
        s_lat, s_lon = shelter_coords()
        region = agg[
            (agg['lat'].between(s_lat - 1.5, s_lat + 1.5))
            & (agg['lon'].between(s_lon - 1.8, s_lon + 1.8))
        ]
        plot_df = region if len(region) >= 5 else agg
        # px.scatter_map / MapLibre, not the deprecated *_mapbox family.
        fig = px.scatter_map(
            plot_df, lat='lat', lon='lon', size='Adoptions', color='Adoptions',
            hover_name='Geo_City',
            hover_data={'Zip_Clean': True, 'County': True, 'Adoptions': True,
                        'lat': False, 'lon': False},
            color_continuous_scale='Blues', size_max=34, zoom=8,
            center={'lat': s_lat, 'lon': s_lon},
            map_style='carto-positron',
            title='Adopter ZIP Codes by Volume (bubble size = adoptions)',
        )
        fig.add_scattermap(
            lat=[s_lat], lon=[s_lon], mode='markers+text',
            marker=dict(size=16, color=ACCENT), text=['Shelter'],
            textposition='top center', name='Shelter', hoverinfo='text',
        )
        fig.update_layout(margin=dict(l=0, r=0, t=60, b=0), height=560,
                          showlegend=False, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            'Every ZIP with at least one adoption, sized by volume — unlike the choropleth '
            'above, this is not restricted to a fixed list of Pittsburgh ZIPs, so genuinely '
            'new or thin areas show up instead of being silently dropped.'
        )
        with st.expander('Adopter ZIPs with county and distance'):
            table = (mapped.groupby(['Zip_Clean', 'Geo_City', 'County'])
                     .agg(Adoptions=('Zip_Clean', 'size'),
                          Distance_mi=('Distance_mi', 'first'))
                     .reset_index().sort_values('Adoptions', ascending=False))
            st.dataframe(table, use_container_width=True)
