import dash
from dash import html, dcc, callback, Output, Input
from dash.exceptions import PreventUpdate
import fastf1
import numpy as np
import pandas as pd
import plotly.graph_objects as go

dash.register_page(__name__, path='/', name='Race Overview', title='F1 — Race Overview')

# ── Plotly dark template ──────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor='#141414',
    plot_bgcolor='#141414',
    font=dict(color='#ffffff', family='Rajdhani'),
    xaxis=dict(gridcolor='#2a2a2a', zerolinecolor='#2a2a2a', tickfont=dict(color='#a0a0a0')),
    yaxis=dict(gridcolor='#2a2a2a', zerolinecolor='#2a2a2a', tickfont=dict(color='#a0a0a0')),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#a0a0a0')),
    margin=dict(l=48, r=24, t=48, b=48),
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def rotate(xy, *, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)

def empty_figure(msg='Select a session to load data'):
    fig = go.Figure()
    fig.update_layout(
        **DARK_LAYOUT,
        annotations=[dict(
            text=msg, x=0.5, y=0.5, xref='paper', yref='paper',
            showarrow=False, font=dict(size=14, color='#666666', family='Orbitron'),
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, **DARK_LAYOUT['xaxis']),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, **DARK_LAYOUT['yaxis']),
    )
    return fig

def format_laptime(td):
    if pd.isnull(td):
        return '—'
    total = td.total_seconds()
    mins = int(total // 60)
    secs = total % 60
    return f'{mins}:{secs:06.3f}'

def get_pos_badge(pos):
    try:
        pos = int(pos)
    except (ValueError, TypeError):
        return html.Span('—')
    cls = {1: 'pos-1', 2: 'pos-2', 3: 'pos-3'}.get(pos, 'pos-default')
    return html.Span(str(pos), className=f'pos-badge {cls}')

# ── Layout ────────────────────────────────────────────────────────────────────
layout = html.Div([
    html.Div([
        html.H1('Race Overview', className='page-title'),
        html.Div(id='overview-session-badge'),
    ], className='page-header'),

    # Stats cards
    dcc.Loading(
        html.Div(id='overview-stats', className='stats-row'),
        type='circle', color='#e10600',
    ),

    # Circuit map + Results table
    dcc.Loading(
        html.Div([
            # Circuit map (left, wider)
            html.Div([
                html.Div('CIRCUIT MAP', className='card-title'),
                dcc.Graph(id='overview-circuit', figure=empty_figure('Loading circuit…'),
                          config={'displayModeBar': False}),
            ], className='card'),

            # Results table (right)
            html.Div([
                html.Div('RACE RESULTS', className='card-title'),
                html.Div(id='overview-results-table'),
            ], className='card', style={'overflowY': 'auto', 'maxHeight': '620px'}),

        ], className='grid-2-wide section'),
        type='circle', color='#e10600',
    ),

    # Position vs Best Lap Time
    dcc.Loading(
        html.Div([
            html.Div('POSITION vs BEST LAP TIME', className='card-title'),
            dcc.Graph(id='overview-pos-lap', figure=empty_figure(),
                      config={'displayModeBar': False}),
        ], className='card section'),
        type='circle', color='#e10600',
    ),
])


# ── Callback ──────────────────────────────────────────────────────────────────
@callback(
    Output('overview-session-badge', 'children'),
    Output('overview-stats',         'children'),
    Output('overview-circuit',       'figure'),
    Output('overview-results-table', 'children'),
    Output('overview-pos-lap',       'figure'),
    Input('session-store', 'data'),
)
def update_overview(store):
    if not store or not store.get('race'):
        raise PreventUpdate

    year         = store['year']
    race_name    = store['race']
    session_type = store.get('session_type', 'R')

    # ── Load session ──────────────────────────────────────────────────────
    try:
        event_schedule = fastf1.get_event_schedule(year, include_testing=False)
        event_row = event_schedule[event_schedule['OfficialEventName'] == race_name]
        if event_row.empty:
            raise ValueError(f'Race not found: {race_name}')
        location = event_row.iloc[0]['Location']

        session = fastf1.get_session(year, location, session_type)
        session.load()
        df = session.laps
    except Exception as e:
        err_msg = f'Could not load session: {e}'
        badge = html.Div(err_msg, style={'color': '#e10600', 'fontFamily': 'Orbitron', 'fontSize': '11px'})
        return badge, [], empty_figure(err_msg), html.P(err_msg), empty_figure(err_msg)

    # ── Session badge ─────────────────────────────────────────────────────
    type_label = {'R': 'RACE', 'Q': 'QUALIFYING', 'FP1': 'FP1', 'FP2': 'FP2',
                  'FP3': 'FP3', 'S': 'SPRINT', 'SQ': 'SPRINT QUALI'}.get(session_type, session_type)
    badge = html.Div([
        html.Div(className='session-dot'),
        f'{year} · {race_name} · {type_label}',
    ], className='session-badge')

    # ── Stats cards ───────────────────────────────────────────────────────
    stats_children = []
    try:
        # Winner / Pole
        if session_type == 'R':
            winner_row = df[df['Position'] == 1.0].sort_values('LapNumber', ascending=False).iloc[0]
            winner_drv = winner_row['Driver']
            stats_children.append(html.Div([
                html.Div('RACE WINNER', className='stat-label'),
                html.Div(winner_drv, className='stat-value'),
                html.Div('1st Place', className='stat-sub'),
            ], className='stat-card'))
        else:
            stats_children.append(html.Div([
                html.Div('SESSION TYPE', className='stat-label'),
                html.Div(type_label, className='stat-value'),
                html.Div(str(year), className='stat-sub'),
            ], className='stat-card'))

        # Fastest lap
        try:
            fl = df.groupby('Driver')['LapTime'].min().idxmin()
            fl_time = df.groupby('Driver')['LapTime'].min().min()
            stats_children.append(html.Div([
                html.Div('FASTEST LAP', className='stat-label'),
                html.Div(fl, className='stat-value'),
                html.Div(format_laptime(fl_time), className='stat-sub'),
            ], className='stat-card'))
        except Exception:
            stats_children.append(html.Div([
                html.Div('FASTEST LAP', className='stat-label'),
                html.Div('—', className='stat-value'),
            ], className='stat-card'))

        # Number of drivers / laps
        n_drivers = df['Driver'].nunique()
        n_laps    = int(df['LapNumber'].max()) if not df.empty else '—'
        stats_children.append(html.Div([
            html.Div('DRIVERS / LAPS', className='stat-label'),
            html.Div(f'{n_drivers} / {n_laps}', className='stat-value'),
            html.Div('Classified', className='stat-sub'),
        ], className='stat-card'))
    except Exception:
        pass

    # ── Circuit map (speed heatmap) ───────────────────────────────────────
    circuit_fig = empty_figure('No circuit data')
    try:
        circuit_info = session.get_circuit_info()
        fastest_lap  = df.pick_fastest()
        tel = fastest_lap.get_telemetry()

        track_angle = circuit_info.rotation / 180 * np.pi

        # Position data for track outline
        pos_data = fastest_lap.get_pos_data()
        track = pos_data.loc[:, ('X', 'Y')].to_numpy()
        if not np.allclose(track[0], track[-1]):
            track = np.vstack([track, track[0]])
        rotated_outline = rotate(track, angle=track_angle)

        # Telemetry for speed heatmap
        tel_xy = tel.loc[:, ('X', 'Y', 'Speed')].dropna()
        tel_xy_rot = rotate(tel_xy[['X', 'Y']].to_numpy(), angle=track_angle)
        speed = tel_xy['Speed'].to_numpy()

        x_min = rotated_outline[:, 0].min() - 600
        x_max = rotated_outline[:, 0].max() + 600
        y_min = rotated_outline[:, 1].min() - 600
        y_max = rotated_outline[:, 1].max() + 600

        circuit_fig = go.Figure()

        # Thick track outline
        circuit_fig.add_trace(go.Scatter(
            x=rotated_outline[:, 0],
            y=rotated_outline[:, 1],
            mode='lines',
            line=dict(color='#2a2a2a', width=18),
            hoverinfo='skip',
            showlegend=False,
        ))

        # Speed heatmap overlay
        circuit_fig.add_trace(go.Scatter(
            x=tel_xy_rot[:, 0],
            y=tel_xy_rot[:, 1],
            mode='markers',
            marker=dict(
                color=speed,
                colorscale=[
                    [0.0,  '#e10600'],
                    [0.35, '#ff8c00'],
                    [0.65, '#ffe400'],
                    [1.0,  '#00d2be'],
                ],
                size=4,
                colorbar=dict(
                    title=dict(text='Speed (km/h)', font=dict(color='#a0a0a0', family='Rajdhani', size=11)),
                    tickfont=dict(color='#a0a0a0', family='Rajdhani'),
                    thickness=12,
                    len=0.7,
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='#2a2a2a',
                    outlinecolor='#2a2a2a',
                ),
            ),
            text=[f'{s:.0f} km/h' for s in speed],
            hovertemplate='%{text}<extra></extra>',
            showlegend=False,
        ))

        circuit_fig.update_layout(
            **DARK_LAYOUT,
            height=500,
            title=dict(
                text=f'{location} · {year}',
                font=dict(family='Orbitron', size=14, color='#ffffff'),
                x=0.5, xanchor='center',
            ),
            xaxis=dict(range=[x_min, x_max], scaleanchor='y',
                       showgrid=False, zeroline=False, showticklabels=False,
                       fixedrange=True),
            yaxis=dict(range=[y_min, y_max],
                       showgrid=False, zeroline=False, showticklabels=False,
                       fixedrange=True),
        )
    except Exception as e:
        circuit_fig = empty_figure(f'Circuit unavailable: {e}')

    # ── Results table ─────────────────────────────────────────────────────
    results_table = html.Div('No data', style={'color': '#666', 'padding': '20px'})
    try:
        summary = (
            df.groupby('Driver')
            .agg(
                Position=('Position', lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan),
                BestLap=('LapTime', 'min'),
            )
            .reset_index()
            .sort_values('Position')
        )

        # Gap to leader
        leader_time = summary.iloc[0]['BestLap'].total_seconds() if not summary.empty else None

        rows = []
        for _, row in summary.iterrows():
            drv   = row['Driver']
            pos   = row['Position']
            lap_t = row['BestLap']
            gap   = ''
            if leader_time and not pd.isnull(lap_t):
                delta = lap_t.total_seconds() - leader_time
                gap   = f'+{delta:.3f}s' if delta > 0 else 'LEADER'

            rows.append(html.Tr([
                html.Td(get_pos_badge(pos)),
                html.Td(html.Span(drv, className='driver-abbr')),
                html.Td(format_laptime(lap_t)),
                html.Td(gap, style={'color': '#a0a0a0', 'fontSize': '13px'}),
            ]))

        results_table = html.Table(
            [html.Thead(html.Tr([
                html.Th('POS'), html.Th('DRV'), html.Th('BEST LAP'), html.Th('GAP'),
            ])),
             html.Tbody(rows)],
            className='results-table',
        )
    except Exception:
        pass

    # ── Position vs Best Lap Time ─────────────────────────────────────────
    pos_lap_fig = empty_figure('No data')
    try:
        summary2 = (
            df.groupby('Driver')
            .agg(
                Position=('Position', lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan),
                LapTime=('LapTime', 'min'),
            )
            .reset_index()
        )
        summary2['LapTime_s'] = summary2['LapTime'].dt.total_seconds()
        summary2 = summary2.dropna(subset=['Position', 'LapTime_s'])

        pos_lap_fig = go.Figure()
        pos_lap_fig.add_trace(go.Scatter(
            x=summary2['Position'],
            y=summary2['LapTime_s'],
            mode='markers+text',
            text=summary2['Driver'],
            textposition='top center',
            textfont=dict(family='Orbitron', size=10, color='#a0a0a0'),
            marker=dict(
                size=12,
                color='#e10600',
                line=dict(color='#ffffff', width=1.5),
            ),
            hovertemplate='<b>%{text}</b><br>Position: %{x}<br>Best Lap: %{y:.3f}s<extra></extra>',
        ))
        pos_lap_fig.update_layout(
            **DARK_LAYOUT,
            height=320,
            title=dict(text='Final Position vs Best Lap Time',
                       font=dict(family='Orbitron', size=13, color='#ffffff'),
                       x=0.5, xanchor='center'),
            xaxis=dict(title='Final Position', **DARK_LAYOUT['xaxis']),
            yaxis=dict(title='Best Lap Time (s)', **DARK_LAYOUT['yaxis']),
        )
    except Exception:
        pass

    return badge, stats_children, circuit_fig, results_table, pos_lap_fig
