
import json
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

sns.set_theme(style='whitegrid', palette='muted')

st.set_page_config(
    page_title='Animal Friends Dashboard',
    page_icon='🐾',
    layout='wide',
)


# Canonical city aliases: variants/typos that should be folded into one city.
# Keys are compared after case/whitespace normalization (strip + collapse + title-case).
# Note: "East Pittsburgh" is intentionally NOT included — it is a distinct municipality.
CITY_ALIASES = {
    'Pgh': 'Pittsburgh',
    'Pgh.': 'Pittsburgh',
    'Pittsburgh Pa': 'Pittsburgh',
    'Pittsburgh, Pa': 'Pittsburgh',
    'Pittsburgh.': 'Pittsburgh',
    'Pittsburgh Metro Area': 'Pittsburgh',
    '15210 Pittsburgh': 'Pittsburgh',
    'Pittsburg': 'Pittsburgh',
    'Pittsbugh': 'Pittsburgh',
    'Pittburgh': 'Pittsburgh',
}


@st.cache_data
def load_excel(file) -> pd.DataFrame:
    return pd.read_excel(file)


@st.cache_data
def load_geojson(url: str) -> dict:
    with urlopen(url) as response:
        geojson = json.load(response)
    for feature in geojson['features']:
        geo_zip = str(feature['properties'].get('ZCTA5CE10')).strip().zfill(5)
        feature['properties']['ZCTA5CE10'] = geo_zip
        feature['id'] = geo_zip
    return geojson


def clean_adoption_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df['Date Of Adoption'] = pd.to_datetime(df['Date Of Adoption'], errors='coerce')
    df['Est. Birthdate'] = pd.to_datetime(df['Est. Birthdate'], errors='coerce')
    df['Age_at_Adoption_yrs'] = ((df['Date Of Adoption'] - df['Est. Birthdate']).dt.days / 365.25).round(1)

    def age_bucket(yrs):
        if pd.isna(yrs):
            return 'Unknown'
        if yrs < 0.5:
            return 'Baby (<6mo)'
        if yrs < 1:
            return 'Junior (6-12mo)'
        if yrs < 3:
            return 'Young Adult (1-3yr)'
        if yrs < 7:
            return 'Adult (3-7yr)'
        return 'Senior (7yr+)'

    df['Age_Bucket'] = df['Age_at_Adoption_yrs'].apply(age_bucket)
    df['Adoption_Month'] = df['Date Of Adoption'].dt.to_period('M').astype(str)
    df['Adoption_Quarter'] = df['Date Of Adoption'].dt.to_period('Q').astype(str)
    df['Adoption_DOW'] = df['Date Of Adoption'].dt.day_name()
    df['Adoption_Year'] = df['Date Of Adoption'].dt.year
    adopter_counts = df['Adopter ID'].value_counts()
    df['Is_Repeat_Adopter'] = df['Adopter ID'].map(adopter_counts) > 1
    df['Has_Microchip'] = df['Microchip Number'].notna() & (df['Microchip Number'].astype(str).str.strip() != '')
    df['Has_Email'] = df['Primary Email'].notna() & (df['Primary Email'].astype(str).str.strip() != '')
    df['Has_Phone'] = df['Primary Phone'].notna() & (df['Primary Phone'].astype(str).str.strip() != '')
    df['Contact_Score'] = df['Has_Email'].astype(int) + df['Has_Phone'].astype(int)
    df['Zip_Clean'] = df['Zip'].astype(str).str.strip().str[:5].str.zfill(5)
    # Normalize city names: strip surrounding/duplicate whitespace and unify casing
    # so "PITTSBURGH", "pittsburgh", and "Pittsburgh " all collapse to "Pittsburgh".
    df['City'] = (
        df['City']
        .astype('string')
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
        .str.title()
    )
    # Fold known variants/typos (e.g. "Pgh", "Pittsburgh, Pa", "Pittsburg") into
    # their canonical city name. Values not in the map are left unchanged.
    df['City'] = df['City'].replace(CITY_ALIASES)
    return df


def safe_sorted_unique(series: pd.Series):
    values = series.dropna().astype(str).unique()
    return sorted(values)


def clean_foster_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def build_sidebar_helpers():
    st.sidebar.header('Upload files')
    adoption_file = st.sidebar.file_uploader('Adoption dataset', type=['xlsx'], key='adoption')
    foster_file = st.sidebar.file_uploader('Foster dataset', type=['xlsx'], key='foster')
    if st.sidebar.button('Use sample files if available'):
        adoption_file = 'adopter-support-06_04_2021-06_03_2026-report-2026-06-04-020609.xlsx'
        foster_file = 'foster-activity-06_04_2021-06_03_2026-report-2026-06-04-030628.xlsx'
    st.sidebar.markdown('---')
    st.sidebar.write('No sidebar filters are enabled. All sections render on the full dataset.')
    return adoption_file, foster_file


def overview_section(df: pd.DataFrame):
    st.header('Data Overview')
    st.write('Track the dataset at a glance and explore the full adoption dataset.')

    total_records = len(df)
    total_adopters = df['Adopter ID'].nunique()
    repeat_adopters = df['Is_Repeat_Adopter'].sum()
    avg_age = df['Age_at_Adoption_yrs'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total records', f'{total_records:,}')
    c2.metric('Unique adopters', f'{total_adopters:,}')
    c3.metric('Repeat adopters', f'{repeat_adopters:,}')
    c4.metric('Average age at adoption', f'{avg_age:.1f} yrs' if not np.isnan(avg_age) else 'N/A')

    top_species = df['Species'].value_counts().head(5).reset_index()
    top_species.columns = ['Species', 'Count']
    top_cities = df['City'].value_counts().head(5).reset_index()
    top_cities.columns = ['City', 'Count']

    col1, col2 = st.columns(2)
    col1.plotly_chart(
        px.bar(top_species, x='Species', y='Count', title='Top species', text='Count'),
        use_container_width=True,
    )
    col2.plotly_chart(
        px.bar(top_cities, x='City', y='Count', title='Top cities', text='Count').update_layout(xaxis_tickangle=-45),
        use_container_width=True,
    )

    with st.expander('View adoption rows'):
        st.dataframe(
            df[
                [
                    'Date Of Adoption',
                    'Adopter Name',
                    'Adopter ID',
                    'Species',
                    'Primary Breed',
                    'City',
                    'Zip_Clean',
                    'Age_Bucket',
                    'Outcome Subtype',
                ]
            ].sort_values(by='Date Of Adoption', ascending=False).reset_index(drop=True),
            use_container_width=True,
        )


def adoption_trends(df: pd.DataFrame):
    st.subheader('Adoption Trends')
    species_year = df.groupby(['Adoption_Year', 'Species']).size().reset_index(name='Count')
    fig = px.line(
        species_year,
        x='Adoption_Year',
        y='Count',
        color='Species',
        markers=True,
        title='Yearly Adoption Trend by Species',
    )
    fig.update_layout(xaxis=dict(dtick=1), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    monthly = df.groupby('Adoption_Month').size().reset_index(name='Adoptions').sort_values('Adoption_Month')
    fig2 = px.area(
        monthly,
        x='Adoption_Month',
        y='Adoptions',
        title='Monthly Adoption Volume',
        markers=True,
    )
    fig2.update_xaxes(tickangle=45, rangeslider_visible=True)
    st.plotly_chart(fig2, use_container_width=True)

    species_monthly = df.groupby(['Adoption_Month', 'Species']).size().reset_index(name='Count')
    fig3 = px.line(
        species_monthly,
        x='Adoption_Month',
        y='Count',
        color='Species',
        markers=True,
        title='Monthly Adoption Trend by Species',
    )
    fig3.update_xaxes(tickangle=45, rangeslider_visible=True)
    st.plotly_chart(fig3, use_container_width=True)

    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts = df['Adoption_DOW'].value_counts().reindex(dow_order).reset_index()
    dow_counts.columns = ['Day', 'Adoptions']
    fig4 = px.bar(
        dow_counts,
        x='Day',
        y='Adoptions',
        title='Adoptions by Day of Week',
        color='Adoptions',
        color_continuous_scale='Tealgrn',
    )
    st.plotly_chart(fig4, use_container_width=True)


def geographic_section(df: pd.DataFrame, show_map: bool):
    st.subheader('Geographic Adoption Insights')

    city_counts = df['City'].value_counts().head(20).reset_index()
    city_counts.columns = ['City', 'Adoptions']
    fig = px.bar(
        city_counts,
        x='City',
        y='Adoptions',
        title='Top 20 Cities by Adoption Count',
        text='Adoptions',
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    zip_counts = df['Zip_Clean'].value_counts().head(20).reset_index()
    zip_counts.columns = ['Zip', 'Adoptions']
    fig2 = px.bar(
        zip_counts,
        x='Zip',
        y='Adoptions',
        title='Top 20 Zip Codes by Adoption Count',
        color='Adoptions',
        color_continuous_scale='Greens',
    )
    # Zip codes are all-numeric strings; force a categorical axis so Plotly shows
    # the full zip (e.g. "15237") instead of SI-abbreviating ticks to "15.2k".
    fig2.update_xaxes(type='category')
    st.plotly_chart(fig2, use_container_width=True)

    if show_map:
        years = sorted(df['Adoption_Year'].dropna().unique())
        if years:
            selected_year = st.selectbox('Choropleth year', years, index=len(years) - 1)
            species_options = ['All'] + safe_sorted_unique(df['Species'])
            selected_species = st.selectbox('Choropleth species', species_options)

            pittsburgh_zips = {
                '15237', '15202', '15108', '15229', '15101', '15143',
                '15212', '15090', '15116', '16066', '15044', '15205',
                '15216', '15227', '15209', '15236', '15136', '15214',
                '15217', '15206'
            }
            map_df = df[df['Adoption_Year'] == selected_year]
            if selected_species != 'All':
                map_df = map_df[map_df['Species'] == selected_species]

            totals = (
                map_df[map_df['Zip_Clean'].isin(pittsburgh_zips)]
                .groupby('Zip_Clean')
                .size()
                .reset_index(name='Adoptions')
            )

            if totals.empty:
                st.warning('No adoption records for the selected year/species combination.')
            else:
                try:
                    geojson = load_geojson(
                        'https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/pa_pennsylvania_zip_codes_geo.min.json'
                    )
                    fig3 = px.choropleth_mapbox(
                        totals,
                        geojson=geojson,
                        locations='Zip_Clean',
                        color='Adoptions',
                        featureidkey='properties.ZCTA5CE10',
                        center={'lat': 40.48, 'lon': -80.02},
                        mapbox_style='carto-positron',
                        zoom=9,
                        title='Pittsburgh Area Adoption Volume by Zip Code',
                        color_continuous_scale='YlGnBu',
                    )
                    fig3.update_layout(margin={'r': 0, 't': 50, 'l': 0, 'b': 0}, height=550)
                    st.plotly_chart(fig3, use_container_width=True)
                except Exception as exc:
                    st.error(f'GeoJSON map load failed: {exc}')


def profile_analysis(df: pd.DataFrame):
    st.subheader('Animal Profile Analysis')

    species_counts = df['Species'].value_counts().reset_index()
    species_counts.columns = ['Species', 'Count']
    col1, col2 = st.columns([2, 1])
    col1.plotly_chart(
        px.bar(species_counts, x='Species', y='Count', title='Adoptions by Species', color='Species'),
        use_container_width=True,
    )
    col2.plotly_chart(
        px.pie(species_counts, values='Count', names='Species', title='Species share'),
        use_container_width=True,
    )

    age_order = ['Baby (<6mo)', 'Junior (6-12mo)', 'Young Adult (1-3yr)', 'Adult (3-7yr)', 'Senior (7yr+)', 'Unknown']
    age_species = df.groupby(['Species', 'Age_Bucket']).size().reset_index(name='Count')
    st.plotly_chart(
        px.bar(
            age_species,
            x='Age_Bucket',
            y='Count',
            color='Species',
            barmode='group',
            category_orders={'Age_Bucket': age_order},
            title='Age at Adoption by Species',
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        px.histogram(
            df,
            x='Age_at_Adoption_yrs',
            nbins=18,
            color='Species',
            title='Age distribution at adoption',
            marginal='box',
        ),
        use_container_width=True,
    )

    chosen_species = st.selectbox('Choose species for breed breakdown', ['All'] + sorted(df['Species'].dropna().unique()))
    breed_df = df if chosen_species == 'All' else df[df['Species'] == chosen_species]
    breed_counts = breed_df['Primary Breed'].value_counts().head(15).reset_index()
    breed_counts.columns = ['Breed', 'Count']
    st.plotly_chart(
        px.bar(breed_counts, x='Breed', y='Count', title=f'Top Breeds for {chosen_species}', color='Count'),
        use_container_width=True,
    )


def staff_outcome_section(df: pd.DataFrame):
    st.subheader('Staff and Outcome Performance')
    staff_counts = df['By (User)'].value_counts().reset_index()
    staff_counts.columns = ['Staff', 'Adoptions']
    st.plotly_chart(
        px.bar(staff_counts.head(20), x='Adoptions', y='Staff', orientation='h', title='Top staff by adoptions', color='Adoptions'),
        use_container_width=True,
    )

    staff_species = df.groupby(['By (User)', 'Species']).size().reset_index(name='Count')
    st.plotly_chart(
        px.bar(
            staff_species,
            x='By (User)',
            y='Count',
            color='Species',
            barmode='stack',
            title='Species mix per staff member',
        ).update_layout(xaxis_tickangle=-45),
        use_container_width=True,
    )

    outcome_counts = df['Outcome Subtype'].value_counts().reset_index()
    outcome_counts.columns = ['Outcome Subtype', 'Count']
    st.plotly_chart(
        px.bar(outcome_counts, x='Outcome Subtype', y='Count', title='Outcome subtype distribution', color='Outcome Subtype'),
        use_container_width=True,
    )


def repeat_adopter_section(df: pd.DataFrame):
    st.subheader('Repeat Adopter Analysis')
    repeat_summary = df['Is_Repeat_Adopter'].value_counts()
    repeat_summary.index = ['Repeat' if i else 'First-time' for i in repeat_summary.index]
    st.plotly_chart(
        px.pie(values=repeat_summary.values, names=repeat_summary.index, title='First-time vs repeat adopters'),
        use_container_width=True,
    )

    adopter_counts = df['Adopter ID'].value_counts()
    row1, row2 = st.columns(2)
    row1.metric('Repeat adopters', f'{(adopter_counts > 1).sum():,}')
    row2.metric('Max adoptions by one adopter', f'{adopter_counts.max():,}')

    top_repeats = adopter_counts[adopter_counts > 1].reset_index()
    top_repeats.columns = ['Adopter ID', 'Total Adoptions']
    top_repeats = top_repeats.merge(df[['Adopter ID', 'Adopter Name']].drop_duplicates(), on='Adopter ID', how='left')
    st.write('Top repeat adopters')
    st.dataframe(top_repeats.sort_values('Total Adoptions', ascending=False).head(10), use_container_width=True)

    repeat_ids = adopter_counts[adopter_counts > 1].index
    repeat_df = df[df['Adopter ID'].isin(repeat_ids)].sort_values(['Adopter ID', 'Date Of Adoption'])
    repeat_df['Days_Since_Last'] = repeat_df.groupby('Adopter ID')['Date Of Adoption'].diff().dt.days
    repeat_df = repeat_df.dropna(subset=['Days_Since_Last'])
    if len(repeat_df) > 0:
        st.plotly_chart(
            px.histogram(repeat_df, x='Days_Since_Last', nbins=30, title='Days between repeat adoptions'),
            use_container_width=True,
        )
        st.write('Median days between repeat adoptions:', int(repeat_df['Days_Since_Last'].median()))
        st.write('Average days between repeat adoptions:', int(repeat_df['Days_Since_Last'].mean()))


def foster_section(df_foster: pd.DataFrame):
    st.subheader('Foster Activity Overview')
    categories = {
        'Dogs': ('Foster Dogs (total unique count)', 'Total Foster Hours for Dogs', 'Foster Reason(s) for Dogs'),
        'Cats': ('Foster Cats (total unique count)', 'Total Foster Hours for Cats', 'Foster Reason(s) for Cats'),
        'Birds': ('Foster Birds (total unique count)', 'Total Foster Hours for Birds', 'Foster Reason(s) for Birds'),
        'Rabbits': ('Foster Rabbits (total unique count)', 'Total Foster Hours for Rabbits', 'Foster Reason(s) for Rabbits'),
        'Barnyards': ('Foster Barnyards (total unique count)', 'Total Foster Hours for Barnyards', 'Foster Reason(s) for Barnyards'),
        'Small Mammals': ('Foster Small Mammals (total unique count)', 'Total Foster Hours for Small Mammals', 'Foster Reason(s) for Small Mammals'),
        'Exotic/Others': ('Foster Exotic/Others (total unique count)', 'Total Foster Hours for Exotic/Others', 'Foster Reason(s) for Exotic/Others'),
    }
    tab = st.selectbox('Choose foster category', list(categories.keys()))
    count_col, hours_col, reason_col = categories[tab]
    if count_col not in df_foster.columns:
        st.warning(f'Missing column: {count_col}')
        return
    top_fosters = df_foster[df_foster[count_col] > 0].sort_values(by=count_col, ascending=False).head(10)
    if top_fosters.empty:
        st.write('No foster activity found for this category.')
        return
    st.plotly_chart(
        px.bar(
            top_fosters,
            x=count_col,
            y='Foster Person Name',
            orientation='h',
            title=f'Top Foster Homes for {tab}',
            text=count_col,
            color=count_col,
        ),
        use_container_width=True,
    )
    st.dataframe(top_fosters[['Foster Person Name', 'Foster Person Email', count_col, hours_col, reason_col]].head(10), use_container_width=True)


def main():
    st.title('Animal Friends Analysis Dashboard')
    st.write('Upload the adoption and foster Excel reports to explore the dashboard.')

    adoption_file, foster_file = build_sidebar_helpers()

    if isinstance(adoption_file, str):
        adoption_data = load_excel(adoption_file)
    elif adoption_file is not None:
        adoption_data = load_excel(adoption_file)
    else:
        adoption_data = None

    foster_data = None
    if foster_file is not None and not isinstance(foster_file, str):
        foster_data = load_excel(foster_file)
    elif foster_file is not None and isinstance(foster_file, str):
        foster_data = load_excel(foster_file)

    if adoption_data is None:
        st.warning('Please upload the adoption dataset to enable the dashboard.')
        return

    df = clean_adoption_data(adoption_data)

    tabs = st.tabs([
        'Overview',
        'Trends',
        'Geography',
        'Animal Profiles',
        'Staff / Outcome',
        'Repeat Adopters',
        'Foster',
    ])

    with tabs[0]:
        overview_section(df)
    with tabs[1]:
        adoption_trends(df)
    with tabs[2]:
        geographic_section(df, True)
    with tabs[3]:
        profile_analysis(df)
    with tabs[4]:
        staff_outcome_section(df)
    with tabs[5]:
        repeat_adopter_section(df)
    with tabs[6]:
        if foster_data is not None:
            foster_section(clean_foster_data(foster_data))
        else:
            st.info('Upload the foster dataset to view foster analytics.')


if __name__ == '__main__':
    main()
