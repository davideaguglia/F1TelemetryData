from dash import Dash, html, dcc, callback, Output, Input
import matplotlib.pyplot as plt
import numpy as np
import fastf1
import pandas as pd
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

app = Dash(__name__, external_stylesheets=[
    'https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@400;700&display=swap'
])

fastf1.Cache.enable_cache('./cache')

# Load event schedule and default session
event = fastf1.get_event_schedule(2024)
current_session_info = {"year": 2024, "event": 'China', "event_type": 'R'}
session = fastf1.get_session(current_session_info["year"], current_session_info["event"], current_session_info["event_type"])
session.load()
df = session.laps
circuit = session.get_circuit_info()
lap = session.laps.pick_fastest()
pos = lap.get_pos_data()

def rotate(xy, *, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)

app.layout = html.Div([
    html.Div([
        html.Img(src='/assets/logo.png', style={'height': '60px', 'marginRight': '20px'}),
        html.Div('Telemetry Data', style={
            'fontSize': '42px', 'fontFamily': 'Orbitron', 'fontWeight': 'bold', 'color': '#e10600'}),
    ], style={'display': 'flex', 'alignItems': 'center'}),

    html.Hr(style={'border': '1px solid #ccc', 'margin': '30px 0'}),

    html.Div([
        html.Label('Select the event', style={
            'fontFamily': 'Orbitron', 'fontSize': '16px', 'color': 'black'}),
        dcc.Dropdown(
            options=[{'label': i, 'value': i} for i in np.unique(event['OfficialEventName'])],
            id='controls-race-name',
            value=event['OfficialEventName'][0],
        ),
        html.Div(id="session-loaded")
    ], style={'width': '60%', 'marginBottom': '30px'}),

    html.Div([
        html.Div([
            html.Div([
            dcc.Graph(figure={}, id='circuit'),
            ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'width': '100%'}),
        ]), 

        html.Div([
            html.Div([
                dcc.Graph(figure={}, id='controls-and-graph'),
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                html.Label('Select Drivers', style={
                    'fontFamily': 'Orbitron', 'fontSize': '16px', 'color': 'black'}),
                dcc.Dropdown(id='controls-driver-item', multi=True),
                dcc.Graph(figure={}, id='controls-and-graph-driver')
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ])
    ]),

    html.Hr(style={'border': '1px solid #ccc', 'margin': '40px 0'}),

    html.Div([
        dcc.Graph(figure={}, id='telemetry'),
    ], style={'borderRadius': '10px', 'padding': '20px', 'backgroundColor': '#fff'})
])

@app.callback(
    Output('session-loaded', 'children'),
    Output('circuit', 'figure'),
    Output('controls-and-graph', 'figure'),
    Output('controls-and-graph-driver', 'figure'),
    Output('controls-driver-item', 'options'),
    Output('controls-driver-item', 'value'),
    Input('controls-race-name', 'value'),
    Input('controls-driver-item', 'value'),
)
def load_main_data(event_name, drivers):
    global session, df, circuit, lap, pos, current_session_info

    event_row = event[event['OfficialEventName'] == event_name].iloc[0]
    year = event_row['EventDate'].year
    location = event_row['Location']
    session_changed = (current_session_info["year"] != year or current_session_info["event"] != location)

    if session_changed:
        session = fastf1.get_session(year, location, 'R')
        session.load()
        df = session.laps
        circuit = session.get_circuit_info()
        lap = session.laps.pick_fastest()
        pos = lap.get_pos_data()
        current_session_info = {"year": year, "event": location, "event_type": 'R'}

    driver_options = [{'label': i, 'value': i} for i in np.unique(df['Driver'].dropna())]
    if session_changed or not drivers:
        default_drivers = list(df['Driver'].dropna().unique()[:2])
        drivers = default_drivers

    # Circuit map
    fig3 = go.Figure()
    track = pos.loc[:, ('X', 'Y')].to_numpy()
    # Close the loop by adding the first point to the end if it's not already there
    if not np.allclose(track[0], track[-1]):
        track = np.vstack([track, track[0]])

    track_angle = circuit.rotation / 180 * np.pi
    rotated_track = rotate(track, angle=track_angle)

    x_min, x_max = rotated_track[:, 0].min() - 2000, rotated_track[:, 0].max() + 2000
    y_min, y_max = rotated_track[:, 1].min() - 2000, rotated_track[:, 1].max() + 2000

    fig3.add_trace(go.Scatter(
        x=rotated_track[:, 0],
        y=rotated_track[:, 1],
        mode='lines',
        line=dict(color='purple', width=8),
        name='Track Line'))

    fig3.update_layout(
        width=800,
        height=700,
        template='plotly_white',
        title=dict(text='Circuit Track', x=0.5, xanchor='center'),
        title_font=dict(family='Orbitron', size=30, color='black', weight='bold'),
        font_family='Orbitron',
        xaxis=dict(range=[x_min, x_max], scaleanchor="y", showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[y_min, y_max], showgrid=False, zeroline=False, showticklabels=False),
        xaxis_fixedrange=True, yaxis_fixedrange=True)

    fig1 = go.Figure()
    summary = df.groupby('Driver').agg({'Position': lambda x: x.dropna().iloc[-1], 'LapTime': 'min'}).reset_index()
    summary['LapTime (s)'] = summary['LapTime'].dt.total_seconds()
    fig1.add_trace(go.Scatter(x=summary['Position'], y=summary['LapTime (s)'], mode='markers+text', text=summary['Driver'], textposition='top center', marker=dict(size=10)))
    fig1.update_layout(xaxis_title='Final Position', yaxis_title='Best Lap Time (s)', title='Final Position vs Best Lap Time', font_family="Orbitron")

    fig2 = go.Figure()
    for driver in drivers:
        try:
            lap = session.laps.pick_drivers(driver).pick_fastest()
            tel = lap.get_car_data().add_distance()
            fig2.add_trace(go.Scatter(x=tel['Distance'], y=tel['Speed'], mode='lines', name=driver))
        except Exception as e:
            print(f"Could not load telemetry for {driver}: {e}")
    #fig2.update_layout(xaxis_title='Distance', yaxis_title='Speed', font_family="Orbitron", title=dict(text='Driver Speed vs Distance', x=0.5))
    #fig2.update_layout(xaxis_title='Distance', yaxis_title='Speed', font_family="Orbitron")
    
    return "", fig3, fig1, fig2, driver_options, drivers

@app.callback(
    Output('telemetry', 'figure'),
    Input('controls-and-graph', 'hoverData'),
    Input('controls-driver-item', 'value'),
)
def update_telemetry_on_hover(hover_data, drivers):
    if hover_data is None:
        raise PreventUpdate

    driver_hovered = hover_data['points'][0]['text']
    fig4 = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025)

    try:
        lap = session.laps.pick_drivers(driver_hovered).pick_fastest()
        tel = lap.get_car_data().add_distance()
        tel['Brake'] = tel['Brake'].astype(int)*100

        # Speed
        fig4.add_trace(go.Scatter(x=tel['Distance'], y=tel['Speed'], mode='lines', name='Speed'), row=1, col=1)

        # Throttle and Brake in same subplot
        fig4.add_trace(go.Scatter(x=tel['Distance'], y=tel['Throttle'], mode='lines', name='Throttle', line=dict(color='green')), row=2, col=1)
        fig4.add_trace(go.Scatter(x=tel['Distance'], y=tel['Brake'], mode='lines', name='Brake', line=dict(color='red')), row=2, col=1)

        # Gear
        fig4.add_trace(go.Scatter(x=tel['Distance'], y=tel['nGear'], mode='lines', name='Gear'), row=3, col=1)

        # RPM
        fig4.add_trace(go.Scatter(x=tel['Distance'], y=tel['RPM'], mode='lines', name='RPM'), row=4, col=1)

    except Exception as e:
        print(f"Could not load telemetry for {driver_hovered}: {e}")

    fig4.update_layout(
        height=1100,
        title=dict(text=f'{driver_hovered}', x=0.5),
        title_font=dict(family='Orbitron', size=30, color='black', weight='bold'),
        font_family="Orbitron",
        showlegend=True
    )

    fig4.update_yaxes(title_text='Speed', row=1, col=1)
    fig4.update_yaxes(title_text='Throttle / Brake', row=2, col=1)
    fig4.update_yaxes(title_text='Gear', row=3, col=1)
    fig4.update_yaxes(title_text='RPM', row=4, col=1)
    fig4.update_xaxes(title_text='Distance', row=4, col=1)

    return fig4

if __name__ == '__main__':
    app.run(debug=True)