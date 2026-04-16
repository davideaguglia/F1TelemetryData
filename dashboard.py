import dash
from dash import Dash, html, dcc, Input, Output, callback
import fastf1
import numpy as np

# ── FastF1 cache ────────────────────────────────────────────────────────────
fastf1.Cache.enable_cache('./cache')

# ── App init ────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap'
    ],
    suppress_callback_exceptions=True,
)
server = app.server

# ── Helpers ──────────────────────────────────────────────────────────────────
YEARS = [2025, 2024, 2023]

def get_race_options(year):
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        names = schedule['OfficialEventName'].dropna().unique().tolist()
        return [{'label': n, 'value': n} for n in names]
    except Exception:
        return []

def get_default_race(year):
    opts = get_race_options(year)
    return opts[0]['value'] if opts else None

SESSION_TYPES = [
    {'label': 'Race',            'value': 'R'},
    {'label': 'Qualifying',      'value': 'Q'},
    {'label': 'Practice 1',      'value': 'FP1'},
    {'label': 'Practice 2',      'value': 'FP2'},
    {'label': 'Practice 3',      'value': 'FP3'},
    {'label': 'Sprint',          'value': 'S'},
    {'label': 'Sprint Qualifying','value': 'SQ'},
]

# Pre-load 2024 race options (default year)
_default_race_options = get_race_options(2024)
_default_race = _default_race_options[0]['value'] if _default_race_options else None

# ── Layout ───────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # Shared state across pages
    dcc.Store(id='session-store', data={
        'year': 2024,
        'race': _default_race,
        'session_type': 'R',
    }),
    dcc.Store(id='year-schedule-store', data={'year': 2024, 'options': _default_race_options}),

    # ── Sidebar ───────────────────────────────────────────────────────────
    html.Div([

        # Logo / branding
        html.Div([
            html.Img(src='/assets/logo.png', style={'height': '34px'}),
            html.Div([
                html.Span('F1 TELEMETRY', style={'display': 'block'}),
                html.Span('DASHBOARD', style={'display': 'block', 'color': 'var(--f1-red)'}),
            ], className='sidebar-logo-text'),
        ], className='sidebar-logo'),

        # Navigation links
        html.Div([
            html.Div('PAGES', className='sidebar-section-label'),
            dcc.Link(
                html.Div([html.Span('▣', className='nav-icon'), 'Race Overview'],
                         className='sidebar-nav-link', id='nav-overview'),
                href='/',
            ),
            dcc.Link(
                html.Div([html.Span('◈', className='nav-icon'), 'Driver Telemetry'],
                         className='sidebar-nav-link', id='nav-telemetry'),
                href='/telemetry',
            ),
            dcc.Link(
                html.Div([html.Span('◉', className='nav-icon'), 'Race Pace'],
                         className='sidebar-nav-link', id='nav-race-pace'),
                href='/race-pace',
            ),
        ], className='sidebar-section'),

        # Session controls
        html.Div([
            html.Div('SESSION', className='sidebar-controls-title'),

            html.Label('YEAR', className='control-label'),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': str(y), 'value': y} for y in YEARS],
                value=2024,
                clearable=False,
                style={'marginBottom': '10px'},
            ),

            html.Label('RACE', className='control-label'),
            dcc.Dropdown(
                id='race-dropdown',
                options=_default_race_options,
                value=_default_race,
                clearable=False,
                style={'marginBottom': '10px'},
            ),

            html.Label('SESSION', className='control-label'),
            dcc.Dropdown(
                id='session-type-dropdown',
                options=SESSION_TYPES,
                value='R',
                clearable=False,
            ),
        ], className='sidebar-controls'),

    ], className='sidebar'),

    # ── Main content ──────────────────────────────────────────────────────
    html.Div([
        dash.page_container
    ], className='main-content'),

], className='app-container')


# ── Callbacks ────────────────────────────────────────────────────────────────

@callback(
    Output('race-dropdown', 'options'),
    Output('race-dropdown', 'value'),
    Output('year-schedule-store', 'data'),
    Input('year-dropdown', 'value'),
)
def update_race_options(year):
    options = get_race_options(year)
    default = options[0]['value'] if options else None
    return options, default, {'year': year, 'options': options}


@callback(
    Output('session-store', 'data'),
    Input('year-dropdown', 'value'),
    Input('race-dropdown', 'value'),
    Input('session-type-dropdown', 'value'),
)
def update_session_store(year, race, session_type):
    return {'year': year, 'race': race, 'session_type': session_type}


if __name__ == '__main__':
    app.run(debug=True)
