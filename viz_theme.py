"""Shared chart palette for the Animal Friends dashboard.

Hues are taken from a validated categorical palette rather than Plotly's default
cycle. Every ordering below was checked with a colour-vision validator: adjacent
pairs clear ΔE >= 8 under protan/deutan/tritan simulation and ΔE >= 15 under
normal vision on a light surface, so a stacked bar stays readable for colour-
blind viewers and in greyscale print.

Three of the hues (aqua, yellow, magenta) fall below 3:1 contrast against a white
surface. Charts that use them ship direct value labels or an expandable data
table alongside, so identity is never carried by colour alone.

Colour follows the *entity* (a species, an intake channel, an outcome type), not
its rank in the current chart — filtering a series out never repaints the rest.
"""

# --------------------------------------------------------------------------- #
# Species                                                                      #
# --------------------------------------------------------------------------- #
CAT = '#eb6834'      # orange
DOG = '#2a78d6'      # blue
OTHER = '#8a8f98'    # neutral grey — deliberately recessive residual bucket
OVERALL = '#52514e'  # ink grey — a total, not a peer series

SPECIES_COLORS = {
    'Overall': OVERALL,
    'Cat': CAT,
    'Dog': DOG,
    'Other': OTHER,
}

SPECIES_SEGMENTS = ['Overall', 'Cat', 'Dog']
SPECIES_GROUPS = ['Cat', 'Dog', 'Other']

# --------------------------------------------------------------------------- #
# Intake channels — stack/legend order matches this dict's order               #
# --------------------------------------------------------------------------- #
INTAKE_ORDER = [
    'Owner Surrender',
    'Stray In',
    'Transfer In',
    'Service In',
    'Adoption Return',
    'Born in Care',
]
INTAKE_COLORS = {
    'Owner Surrender': '#2a78d6',
    'Stray In': '#eb6834',
    'Transfer In': '#1baf7a',
    'Service In': '#eda100',
    'Adoption Return': '#e87ba4',
    'Born in Care': '#008300',
    'Other': OTHER,
}

# --------------------------------------------------------------------------- #
# Outcome types                                                                #
# --------------------------------------------------------------------------- #
OUTCOME_ORDER = [
    'Adoption',
    'Return to Owner',
    'Transfer Out',
    'Service Out',
    'Euthanasia',
    'Died in Custody',
    'Lost-Stolen',
]
OUTCOME_COLORS = {
    'Adoption': '#1baf7a',
    'Return to Owner': '#2a78d6',
    'Transfer Out': '#eda100',
    'Service Out': OTHER,
    'Euthanasia': '#e34948',
    'Died in Custody': '#4a3aa7',
    'Lost-Stolen': '#e87ba4',
    'Other': OTHER,
}

# Outcomes that do NOT count as a live release (methodology carried over from
# the Part 1-2 macro analysis).
NON_LIVE_OUTCOMES = {'Euthanasia', 'Died in Custody', 'Lost-Stolen'}

# Sequential ramp for magnitude (single hue, light -> dark).
SEQUENTIAL = 'Blues'

# Accent used for annotations/reference lines.
ACCENT = '#e34948'
INK = '#52514e'


def apply_layout(fig, height=None, legend_title=None):
    """Recede the chrome: light grid, no vertical grid clutter, ink-coloured text.

    Applied to every chart in the new sections so they read as one system.
    """
    fig.update_layout(
        template='plotly_white',
        font=dict(color='#2f2e2b', size=12),
        title_font=dict(size=15, color='#0b0b0b'),
        legend=dict(title_text=legend_title or '', orientation='h',
                    yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=10, r=10, t=70, b=10),
        hoverlabel=dict(font_size=12),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(showgrid=False, linecolor='#d8d7d2')
    fig.update_yaxes(gridcolor='#eceae5', zerolinecolor='#d8d7d2')
    return fig
