import dash
from dash import html, dcc, callback, Output, Input
from dash.exceptions import PreventUpdate
import fastf1
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

dash.register_page(__name__, path='/telemetry', name='Driver Telemetry', title='F1 — Driver Telemetry')

# ── Plotly dark template ──────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor='#141414',
    plot_bgcolor='#141414',
    font=dict(color='#ffffff', family='Rajdhani'),
    xaxis=dict(gridcolor='#2a2a2a', zerolinecolor='#2a2a2a', tickfont=dict(color='#a0a0a0')),
    yaxis=dict(gridcolor='#2a2a2a', zerolinecolor='#2a2a2a', tickfont=dict(color='#a0a0a0')),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#a0a0a0')),
    margin=dict(l=56, r=24, t=48, b=48),
)

# Palette for up to 20 drivers
DRIVER_COLORS = [
    '#e10600', '#00d2be', '#ffd700', '#ff8700', '#0090ff',
    '#b0b0b0', '#9b0000', '#ff80cc', '#00e0d0', '#ffaa00',
    '#39b54a', '#c8102e', '#6692ff', '#ff4dc4', '#ff9f36',
    '#2b4562', '#dc0000', '#ffffff', '#aaaaaa', '#ff6600',
]

def empty_figure(msg='Select drivers to display telemetry'):
    fig = go.Figure()
    base = {k: v for k, v in DARK_LAYOUT.items() if k not in ('xaxis', 'yaxis')}
    fig.update_layout(
        **base,
        height=300,
        annotations=[dict(
            text=msg, x=0.5, y=0.5, xref='paper', yref='paper',
            showarrow=False, font=dict(size=13, color='#666666', family='Orbitron'),
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, **DARK_LAYOUT['xaxis']),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, **DARK_LAYOUT['yaxis']),
    )
    return fig

# ── Layout ────────────────────────────────────────────────────────────────────
layout = html.Div([
    html.Div([
        html.H1('Driver Telemetry', className='page-title'),
        html.Div(id='telemetry-session-badge'),
    ], className='page-header'),

    # Driver selector
    html.Div([
        html.Div([
            html.Label('SELECT DRIVERS', className='control-label'),
            dcc.Dropdown(id='telemetry-driver-select', multi=True, placeholder='Choose drivers…'),
        ], style={'flex': '1'}),
        html.Div([
            html.Label('LAP', className='control-label'),
            dcc.Dropdown(
                id='telemetry-lap-select',
                options=[{'label': 'Fastest Lap', 'value': 'fastest'}],
                value='fastest',
                clearable=False,
            ),
        ], style={'width': '200px'}),
    ], className='card section', style={'display': 'flex', 'gap': '20px', 'alignItems': 'flex-end'}),

    # Speed comparison chart
    dcc.Loading(
        html.Div([
            html.Div('SPEED COMPARISON', className='card-title'),
            dcc.Graph(id='telemetry-speed-chart', figure=empty_figure(),
                      config={'displayModeBar': 'hover'}),
        ], className='card section'),
        type='circle', color='#e10600',
    ),

    # Detailed telemetry panel (per-driver)
    html.Div([
        html.Div('DETAILED TELEMETRY — click a driver on the speed chart', className='card-title'),
        html.Div('Hover over the speed chart to select a driver', id='telemetry-detail-hint',
                 style={'color': '#666', 'fontFamily': 'Orbitron', 'fontSize': '11px',
                        'letterSpacing': '1px', 'marginBottom': '12px'}),
        dcc.Loading(
            dcc.Graph(id='telemetry-detail-chart', figure=empty_figure('Click a driver on the speed chart'),
                      config={'displayModeBar': 'hover'}),
            type='circle', color='#e10600',
        ),
    ], className='card section'),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output('telemetry-session-badge',  'children'),
    Output('telemetry-driver-select',  'options'),
    Output('telemetry-driver-select',  'value'),
    Input('session-store', 'data'),
)
def populate_drivers(store):
    if not store or not store.get('race'):
        raise PreventUpdate

    year         = store['year']
    race_name    = store['race']
    session_type = store.get('session_type', 'R')

    type_label = {'R': 'RACE', 'Q': 'QUALIFYING', 'FP1': 'FP1', 'FP2': 'FP2',
                  'FP3': 'FP3', 'S': 'SPRINT', 'SQ': 'SPRINT QUALI'}.get(session_type, session_type)
    badge = html.Div([
        html.Div(className='session-dot'),
        f'{year} · {race_name} · {type_label}',
    ], className='session-badge')

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        event_row = schedule[schedule['OfficialEventName'] == race_name]
        if event_row.empty:
            return badge, [], []
        location = event_row.iloc[0]['Location']

        session = fastf1.get_session(year, location, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        drivers = sorted(session.laps['Driver'].dropna().unique().tolist())
        options = [{'label': d, 'value': d} for d in drivers]
        default = drivers[:2] if len(drivers) >= 2 else drivers
    except Exception:
        return badge, [], []

    return badge, options, default


@callback(
    Output('telemetry-speed-chart', 'figure'),
    Input('session-store',          'data'),
    Input('telemetry-driver-select','value'),
    Input('telemetry-lap-select',   'value'),
)
def update_speed_chart(store, drivers, lap_mode):
    if not store or not drivers:
        raise PreventUpdate

    year         = store['year']
    race_name    = store['race']
    session_type = store.get('session_type', 'R')

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        location = schedule[schedule['OfficialEventName'] == race_name].iloc[0]['Location']
        session  = fastf1.get_session(year, location, session_type)
        session.load()
    except Exception as e:
        return empty_figure(f'Could not load session: {e}')

    fig = go.Figure()
    for i, driver in enumerate(drivers):
        try:
            if lap_mode == 'fastest':
                lap = session.laps.pick_drivers(driver).pick_fastest()
            else:
                lap = session.laps.pick_drivers(driver).pick_fastest()  # fallback
            tel = lap.get_car_data().add_distance()

            color = DRIVER_COLORS[i % len(DRIVER_COLORS)]
            fig.add_trace(go.Scatter(
                x=tel['Distance'],
                y=tel['Speed'],
                mode='lines',
                name=driver,
                line=dict(color=color, width=2),
                hovertemplate=f'<b>{driver}</b><br>Distance: %{{x:.0f}} m<br>Speed: %{{y:.0f}} km/h<extra></extra>',
            ))
        except Exception as e:
            print(f'Could not load telemetry for {driver}: {e}')

    fig.update_layout(
        **DARK_LAYOUT,
        height=380,
        title=dict(text='Speed vs Distance — Fastest Lap',
                   font=dict(family='Orbitron', size=13, color='#ffffff'),
                   x=0.5, xanchor='center'),
        xaxis=dict(title='Distance (m)', **DARK_LAYOUT['xaxis']),
        yaxis=dict(title='Speed (km/h)', **DARK_LAYOUT['yaxis']),
        hovermode='x unified',
    )
    return fig


@callback(
    Output('telemetry-detail-chart', 'figure'),
    Output('telemetry-detail-hint',  'children'),
    Input('telemetry-speed-chart',   'hoverData'),
    Input('session-store',           'data'),
    Input('telemetry-lap-select',    'value'),
)
def update_detail_telemetry(hover_data, store, lap_mode):
    if hover_data is None or not store:
        raise PreventUpdate

    # Extract hovered driver from the curve name
    try:
        driver = hover_data['points'][0]['data']['name']
    except (KeyError, IndexError, TypeError):
        raise PreventUpdate

    year         = store['year']
    race_name    = store['race']
    session_type = store.get('session_type', 'R')

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        location = schedule[schedule['OfficialEventName'] == race_name].iloc[0]['Location']
        session  = fastf1.get_session(year, location, session_type)
        session.load()
    except Exception as e:
        return empty_figure(f'Session error: {e}'), driver

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=['Speed (km/h)', 'Throttle / Brake (%)', 'Gear', 'RPM'],
    )

    try:
        lap = session.laps.pick_drivers(driver).pick_fastest()
        tel = lap.get_car_data().add_distance()
        tel['Brake'] = tel['Brake'].astype(float) * 100

        # Row 1: Speed
        fig.add_trace(go.Scatter(
            x=tel['Distance'], y=tel['Speed'], mode='lines',
            line=dict(color='#00d2be', width=2),
            name='Speed', hovertemplate='%{y:.0f} km/h<extra></extra>',
        ), row=1, col=1)

        # Row 2: Throttle + Brake
        fig.add_trace(go.Scatter(
            x=tel['Distance'], y=tel['Throttle'], mode='lines',
            line=dict(color='#39b54a', width=1.5),
            name='Throttle', hovertemplate='Throttle: %{y:.0f}%<extra></extra>',
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=tel['Distance'], y=tel['Brake'], mode='lines',
            line=dict(color='#e10600', width=1.5),
            name='Brake', hovertemplate='Brake: %{y:.0f}%<extra></extra>',
        ), row=2, col=1)

        # Row 3: Gear
        fig.add_trace(go.Scatter(
            x=tel['Distance'], y=tel['nGear'], mode='lines',
            line=dict(color='#ffd700', width=1.5),
            name='Gear', hovertemplate='Gear: %{y}<extra></extra>',
        ), row=3, col=1)

        # Row 4: RPM
        fig.add_trace(go.Scatter(
            x=tel['Distance'], y=tel['RPM'], mode='lines',
            line=dict(color='#ff8700', width=1.5),
            name='RPM', hovertemplate='%{y:.0f} RPM<extra></extra>',
        ), row=4, col=1)

    except Exception as e:
        return empty_figure(f'Telemetry unavailable for {driver}: {e}'), driver

    fig.update_layout(
        paper_bgcolor='#141414',
        plot_bgcolor='#141414',
        font=dict(color='#ffffff', family='Rajdhani'),
        height=700,
        title=dict(
            text=f'{driver} — Fastest Lap Telemetry',
            font=dict(family='Orbitron', size=14, color='#ffffff'),
            x=0.5, xanchor='center',
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#a0a0a0')),
        margin=dict(l=60, r=24, t=60, b=48),
    )

    # Update all subplot axes to dark theme
    for i in range(1, 5):
        fig.update_xaxes(gridcolor='#2a2a2a', zerolinecolor='#2a2a2a',
                         tickfont=dict(color='#a0a0a0'), row=i, col=1)
        fig.update_yaxes(gridcolor='#2a2a2a', zerolinecolor='#2a2a2a',
                         tickfont=dict(color='#a0a0a0'), row=i, col=1)

    fig.update_xaxes(title_text='Distance (m)', row=4, col=1,
                     title_font=dict(color='#a0a0a0'))

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font.color  = '#a0a0a0'
        ann.font.family = 'Orbitron'
        ann.font.size   = 10

    hint = f'Showing: {driver}'
    return fig, hint
