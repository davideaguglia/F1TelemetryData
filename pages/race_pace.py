import dash
from dash import html, dcc, callback, Output, Input
from dash.exceptions import PreventUpdate
import fastf1
import numpy as np
import pandas as pd
import plotly.graph_objects as go

dash.register_page(__name__, path='/race-pace', name='Race Pace', title='F1 — Race Pace & Strategy')

# ── Constants ─────────────────────────────────────────────────────────────────
COMPOUND_COLORS = {
    'SOFT':    '#ff3333',
    'MEDIUM':  '#fff200',
    'HARD':    '#e8e8e8',
    'INTER':   '#39b54a',
    'WET':     '#0067ff',
    'UNKNOWN': '#888888',
    'TEST_UNKNOWN': '#888888',
}

# Up to 20 drivers — spread of distinguishable colors on dark background
DRIVER_PALETTE = [
    '#e10600', '#00d2be', '#ffd700', '#ff8700', '#0090ff',
    '#b0b0b0', '#ff80cc', '#00e0d0', '#ffaa00', '#39b54a',
    '#c8102e', '#6692ff', '#ff4dc4', '#ff9f36', '#2b4562',
    '#dc0000', '#aaaaaa', '#ff6600', '#9b0000', '#4dc4ff',
]

DARK_LAYOUT = dict(
    paper_bgcolor='#141414',
    plot_bgcolor='#141414',
    font=dict(color='#ffffff', family='Rajdhani'),
    xaxis=dict(gridcolor='#2a2a2a', zerolinecolor='#2a2a2a', tickfont=dict(color='#a0a0a0')),
    yaxis=dict(gridcolor='#2a2a2a', zerolinecolor='#2a2a2a', tickfont=dict(color='#a0a0a0')),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#a0a0a0'), orientation='v'),
    margin=dict(l=56, r=120, t=48, b=48),
)

def empty_figure(msg='No data available', height=360):
    fig = go.Figure()
    fig.update_layout(
        **DARK_LAYOUT,
        height=height,
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
        html.H1('Race Pace & Strategy', className='page-title'),
        html.Div(id='pace-session-badge'),
    ], className='page-header'),

    # Lap Time Progression
    dcc.Loading(
        html.Div([
            html.Div('LAP TIME PROGRESSION', className='card-title'),
            dcc.Graph(id='pace-laptimes', figure=empty_figure('Load a race session'),
                      config={'displayModeBar': 'hover'}),
        ], className='card section'),
        type='circle', color='#e10600',
    ),

    # Position Evolution
    dcc.Loading(
        html.Div([
            html.Div('POSITION EVOLUTION', className='card-title'),
            dcc.Graph(id='pace-positions', figure=empty_figure('Load a race session'),
                      config={'displayModeBar': 'hover'}),
        ], className='card section'),
        type='circle', color='#e10600',
    ),

    # Two-column: Tire Strategy + Gap to Leader
    dcc.Loading(
        html.Div([
            html.Div([
                html.Div('TIRE STRATEGY', className='card-title'),
                dcc.Graph(id='pace-strategy', figure=empty_figure('Load a race session'),
                          config={'displayModeBar': False}),
            ], className='card'),

            html.Div([
                html.Div('GAP TO LEADER', className='card-title'),
                dcc.Graph(id='pace-gap', figure=empty_figure('Load a race session'),
                          config={'displayModeBar': 'hover'}),
            ], className='card'),
        ], className='grid-2 section'),
        type='circle', color='#e10600',
    ),
])


# ── Callback ──────────────────────────────────────────────────────────────────
@callback(
    Output('pace-session-badge', 'children'),
    Output('pace-laptimes',      'figure'),
    Output('pace-positions',     'figure'),
    Output('pace-strategy',      'figure'),
    Output('pace-gap',           'figure'),
    Input('session-store', 'data'),
)
def update_race_pace(store):
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

    # ── Load session ──────────────────────────────────────────────────────
    try:
        schedule  = fastf1.get_event_schedule(year, include_testing=False)
        event_row = schedule[schedule['OfficialEventName'] == race_name]
        if event_row.empty:
            raise ValueError(f'Race not found: {race_name}')
        location = event_row.iloc[0]['Location']
        session  = fastf1.get_session(year, location, session_type)
        session.load()
        df = session.laps
    except Exception as e:
        err = f'Could not load: {e}'
        return badge, empty_figure(err), empty_figure(err), empty_figure(err), empty_figure(err)

    drivers = sorted(df['Driver'].dropna().unique().tolist())
    color_map = {d: DRIVER_PALETTE[i % len(DRIVER_PALETTE)] for i, d in enumerate(drivers)}

    # ── Lap Time Progression ──────────────────────────────────────────────
    laptimes_fig = empty_figure('No lap time data')
    try:
        # Filter out outlier laps (pit in/out + safety car)
        clean = df[df['LapTime'].notna()].copy()
        median_lt = clean['LapTime'].dt.total_seconds().median()
        clean = clean[clean['LapTime'].dt.total_seconds() < median_lt * 1.07]

        laptimes_fig = go.Figure()
        for driver in drivers:
            drv_laps = clean[clean['Driver'] == driver].sort_values('LapNumber')
            if drv_laps.empty:
                continue
            lt_sec = drv_laps['LapTime'].dt.total_seconds()
            laptimes_fig.add_trace(go.Scatter(
                x=drv_laps['LapNumber'],
                y=lt_sec,
                mode='lines+markers',
                name=driver,
                line=dict(color=color_map[driver], width=1.5),
                marker=dict(size=4, color=color_map[driver]),
                hovertemplate=f'<b>{driver}</b><br>Lap %{{x}}<br>%{{y:.3f}}s<extra></extra>',
            ))

        laptimes_fig.update_layout(
            **DARK_LAYOUT,
            height=420,
            title=dict(text='Lap Times Over Race Distance',
                       font=dict(family='Orbitron', size=13, color='#ffffff'),
                       x=0.5, xanchor='center'),
            xaxis=dict(title='Lap Number', **DARK_LAYOUT['xaxis']),
            yaxis=dict(title='Lap Time (s)', **DARK_LAYOUT['yaxis']),
            hovermode='x unified',
        )
    except Exception as e:
        laptimes_fig = empty_figure(f'Lap times error: {e}')

    # ── Position Evolution ────────────────────────────────────────────────
    positions_fig = empty_figure('No position data')
    try:
        pos_df = df[df['Position'].notna()].copy()
        pos_df['LapNumber'] = pos_df['LapNumber'].astype(int)
        pos_summary = (
            pos_df.groupby(['Driver', 'LapNumber'])['Position']
            .last()
            .reset_index()
        )

        positions_fig = go.Figure()
        for driver in drivers:
            drv_pos = pos_summary[pos_summary['Driver'] == driver].sort_values('LapNumber')
            if drv_pos.empty:
                continue
            positions_fig.add_trace(go.Scatter(
                x=drv_pos['LapNumber'],
                y=drv_pos['Position'],
                mode='lines+markers',
                name=driver,
                line=dict(color=color_map[driver], width=2),
                marker=dict(size=4, color=color_map[driver]),
                hovertemplate=f'<b>{driver}</b><br>Lap %{{x}}<br>P%{{y:.0f}}<extra></extra>',
            ))

        positions_fig.update_layout(
            **DARK_LAYOUT,
            height=400,
            title=dict(text='Position Evolution',
                       font=dict(family='Orbitron', size=13, color='#ffffff'),
                       x=0.5, xanchor='center'),
            xaxis=dict(title='Lap Number', **DARK_LAYOUT['xaxis']),
            yaxis=dict(
                title='Position',
                autorange='reversed',
                tickmode='linear', tick0=1, dtick=1,
                **DARK_LAYOUT['yaxis'],
            ),
            hovermode='x unified',
        )
    except Exception as e:
        positions_fig = empty_figure(f'Position error: {e}')

    # ── Tire Strategy ─────────────────────────────────────────────────────
    strategy_fig = empty_figure('No stint data')
    try:
        stints = (
            df[df['Compound'].notna()]
            .groupby(['Driver', 'Stint'])
            .agg(
                start_lap=('LapNumber', 'min'),
                end_lap=('LapNumber', 'max'),
                compound=('Compound', 'first'),
            )
            .reset_index()
        )

        strategy_fig = go.Figure()
        added_compounds = set()
        drv_order = sorted(drivers, reverse=True)  # bottom → top on chart

        for driver in drv_order:
            drv_stints = stints[stints['Driver'] == driver]
            for _, stint in drv_stints.iterrows():
                compound = stint['compound'].upper()
                color    = COMPOUND_COLORS.get(compound, '#888888')
                show_leg = compound not in added_compounds
                if show_leg:
                    added_compounds.add(compound)

                strategy_fig.add_trace(go.Bar(
                    x=[stint['end_lap'] - stint['start_lap'] + 1],
                    y=[driver],
                    base=[stint['start_lap'] - 1],
                    orientation='h',
                    marker=dict(color=color, line=dict(color='#0a0a0a', width=1.5)),
                    name=compound,
                    legendgroup=compound,
                    showlegend=show_leg,
                    hovertemplate=(
                        f'<b>{driver}</b><br>'
                        f'{compound}<br>'
                        f'Laps {int(stint["start_lap"])}–{int(stint["end_lap"])}'
                        f' ({int(stint["end_lap"] - stint["start_lap"] + 1)} laps)'
                        '<extra></extra>'
                    ),
                ))

        strategy_fig.update_layout(
            **DARK_LAYOUT,
            height=max(300, len(drv_order) * 28 + 80),
            barmode='overlay',
            bargap=0.3,
            title=dict(text='Tire Strategy',
                       font=dict(family='Orbitron', size=13, color='#ffffff'),
                       x=0.5, xanchor='center'),
            xaxis=dict(title='Lap Number', **DARK_LAYOUT['xaxis']),
            yaxis=dict(title='', tickfont=dict(color='#ffffff', family='Orbitron', size=9),
                       gridcolor='#2a2a2a', zerolinecolor='#2a2a2a'),
            margin=dict(l=72, r=120, t=48, b=48),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#a0a0a0'),
                        title=dict(text='Compound', font=dict(color='#666', family='Orbitron', size=9))),
        )
    except Exception as e:
        strategy_fig = empty_figure(f'Strategy error: {e}')

    # ── Gap to Leader ─────────────────────────────────────────────────────
    gap_fig = empty_figure('No gap data')
    try:
        # Build cumulative lap times per driver per lap
        clean2 = df[df['LapTime'].notna()].copy()
        clean2['lt_s'] = clean2['LapTime'].dt.total_seconds()
        clean2 = clean2[['Driver', 'LapNumber', 'lt_s']].dropna()
        clean2['LapNumber'] = clean2['LapNumber'].astype(int)

        # For each lap, keep only the last recorded lap time if duplicates exist
        clean2 = clean2.groupby(['Driver', 'LapNumber'])['lt_s'].last().reset_index()

        # Cumulative time
        clean2 = clean2.sort_values(['Driver', 'LapNumber'])
        clean2['cum_time'] = clean2.groupby('Driver')['lt_s'].cumsum()

        # Find max laps to align
        max_lap = int(clean2['LapNumber'].max())
        all_laps = pd.RangeIndex(1, max_lap + 1)

        # Leader at each lap = driver with minimum cumulative time
        lap_leaders = (
            clean2.groupby('LapNumber')['cum_time']
            .min()
            .rename('leader_time')
            .reset_index()
        )
        clean2 = clean2.merge(lap_leaders, on='LapNumber', how='left')
        clean2['gap'] = clean2['cum_time'] - clean2['leader_time']

        gap_fig = go.Figure()
        for driver in drivers:
            drv_gap = clean2[clean2['Driver'] == driver].sort_values('LapNumber')
            if drv_gap.empty:
                continue
            gap_fig.add_trace(go.Scatter(
                x=drv_gap['LapNumber'],
                y=drv_gap['gap'],
                mode='lines',
                name=driver,
                line=dict(color=color_map[driver], width=1.5),
                hovertemplate=f'<b>{driver}</b><br>Lap %{{x}}<br>+%{{y:.3f}}s<extra></extra>',
            ))

        gap_fig.update_layout(
            **DARK_LAYOUT,
            height=400,
            title=dict(text='Cumulative Gap to Leader',
                       font=dict(family='Orbitron', size=13, color='#ffffff'),
                       x=0.5, xanchor='center'),
            xaxis=dict(title='Lap Number', **DARK_LAYOUT['xaxis']),
            yaxis=dict(title='Gap (s)', **DARK_LAYOUT['yaxis']),
            hovermode='x unified',
        )
    except Exception as e:
        gap_fig = empty_figure(f'Gap error: {e}')

    return badge, laptimes_fig, positions_fig, strategy_fig, gap_fig
