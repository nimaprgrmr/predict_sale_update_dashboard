# Import necessary libraries
import dash
from dash import html, dcc, State
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from utils import make_period_time
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from data_preprocessing import read_data


# Loading model and scaler
FILE_NAME_MODEL = 'models/bamland_predictor.pickle'
FILE_NAME_SCALER = 'models/bamland_scaler.pickle'
MODEL = pickle.load(open(FILE_NAME_MODEL, 'rb'))
SCALER = pickle.load(open(FILE_NAME_SCALER, 'rb'))
DATA = read_data()


def make_predictions(start_date: str, end_date: str, model: RandomForestRegressor, scaler: MinMaxScaler):

    start_date = list(start_date.split("-"))
    end_date = list(end_date.split("-"))

    start_date = [int(x) for x in start_date]
    end_date = [int(x) for x in end_date]

    # Create df between start date and end date
    input_features = make_period_time(start_date, end_date)
    scaler = scaler
    # Create scaled features
    scaled_input = scaler.transform(input_features)
    model = model
    # Make predictions
    predictions = model.predict(scaled_input)
    # Create start date from input dates for take actual sales
    start_year, start_month, start_day = input_features.iloc[0][['year', 'month', 'day']]
    # Create end date from input dates for take actual sales
    end_year, end_month, end_day = input_features.iloc[-1][['year', 'month', 'day']]
    # Create str date for set into data query
    start_tarikh = pd.to_datetime('-'.join([str(start_year), str(start_month), str(start_day)]))
    end_tarikh = pd.to_datetime('-'.join([str(end_year), str(end_month), str(end_day)]))
    # Add two days to end_date
    end_date_plus_1_days = end_tarikh + pd.Timedelta(days=1)
    actual = DATA[(DATA['date'] >= str(start_tarikh)) & (DATA['date'] <= str(end_date_plus_1_days))]['total_price']
    return predictions, actual


#MATERIA, MINTY, ZEPHYR, FLATLY, CERULEAN, QUARTZ
# Dark & ciculy : CYBORG
# Create a Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ])

# Month mapping for Persian calendar
persian_months = {
    "Farvardin": {"days": 31, "index": 1},
    "Ordibehesht": {"days": 31, "index": 2},
    "Khordad": {"days": 31, "index": 3},
    "Tir": {"days": 31, "index": 4},
    "Mordad": {"days": 31, "index": 5},
    "Shahrivar": {"days": 31, "index": 6},
    "Mehr": {"days": 30, "index": 7},
    "Aban": {"days": 30, "index": 8},
    "Azar": {"days": 30, "index": 9},
    "Dey": {"days": 30, "index": 10},
    "Bahman": {"days": 30, "index": 11},
    "Esfand": {"days": 30, "index": 12}
}

# Initialize default year and selected months list
selected_year = 1400
selected_months = []

# Define the layout using Dash Bootstrap Components
app.layout = dbc.Container([
    # set header
    dbc.Row([
        dbc.Col(html.H3("پیشبینی فروش", style={'margin-top': '20px'}), width=6),

        dbc.Col([
            dbc.Button('راهنما', id='help-button', n_clicks=0, color='primary', outline=True,
                       style={'background-color': 'rgb(6, 206, 255)', 'color': 'white'}),
        ], width=6, style={'text-align': 'left', 'margin-top': '10px', 'margin-bottom': '20px'}),
    ]),
    # Row 1: Year Selection
    dbc.Row([
        dbc.Col(html.Label('انتخاب سال'), width=12, style={'margin-bottom': '5px'}),
        dbc.Col(dbc.Button('1398', id='year-1398', n_clicks=0, size='md'), width=1),
        dbc.Col(dbc.Button('1399', id='year-1399', n_clicks=0), width=1),
        dbc.Col(dbc.Button('1400', id='year-1400', n_clicks=0), width=1),
        dbc.Col(dbc.Button('1401', id='year-1401', n_clicks=0), width=1),
        dbc.Col(dbc.Button('1402', id='year-1402', n_clicks=0), width=1),
        dbc.Col(dbc.Button('1403', id='year-1403', n_clicks=0), width=1),
        dbc.Col(dbc.Button('1404', id='year-1404', n_clicks=0), width=1),
        dbc.Col(dbc.Button('1405', id='year-1405', n_clicks=0), width=1),
        dbc.Col(dbc.Button('1406', id='year-1406', n_clicks=0), width=1),
    ], style={'margin-bottom': '10px'}),

    # Row 2: Month Selection
    dbc.Row([
        dbc.Col(html.Label('انتخاب ماه'), width=12, style={'margin-bottom': '5px'}),
        dbc.Col(dbc.Button('فروردین', id='month-farvardin', n_clicks=0, size='sm'), width=1, style={'text-align': 'right'}),
        dbc.Col(dbc.Button('اردیبهشت', id='month-ordibehesht', n_clicks=0, size='sm'), width=1, style={'text-align': 'right'}),
        dbc.Col(dbc.Button('خرداد', id='month-khordad', n_clicks=0, size='sm'), width=1, style={'text-align': 'right'}),
        dbc.Col(dbc.Button('تیر', id='month-tir', n_clicks=0, size='sm'), width=1, style={'text-align': 'right'}),
        dbc.Col(dbc.Button('مرداد', id='month-mordad', n_clicks=0, size='sm'), width=1, style={'text-align': 'right'}),
        dbc.Col(dbc.Button('شهریور', id='month-shahrivar', n_clicks=0, size='sm'), width=1, style={'text-align': 'right'}),
        dbc.Col(dbc.Button('مهر', id='month-mehr', n_clicks=0, size='sm'), width=1, style={'text-align': 'right'}),
        dbc.Col(dbc.Button('آبان', id='month-aban', n_clicks=0, size='sm'), width=1, style={'text-align': 'right'}),
        dbc.Col(dbc.Button('آذر', id='month-azar', n_clicks=0, size='sm'), width=1, style={'text-align': 'right'}),
        dbc.Col(dbc.Button('دی', id='month-dey', n_clicks=0, size='sm'), width=1, style={'text-align': 'right'}),
        dbc.Col(dbc.Button('بهمن', id='month-bahman', n_clicks=0, size='sm'), width=1, style={'text-align': 'right'}),
        dbc.Col(dbc.Button('اسفند', id='month-esfand', n_clicks=0, size='sm'), width=1, style={'text-align': 'right'}),
    ]),

    # Row 3: Output
    dbc.Row([
        dbc.Col(html.Div(id='output-container-date-range'), width=12, style={'margin-top': '15px'}),
    ]),
    dbc.Row([
            dbc.Col(dcc.Graph(id='prediction-plot'), width=12, style={'margin-top': '20px', 'margin-bottom': '50px',}),
        ]),
    dbc.Modal([
            dbc.ModalHeader("راهنمای کاربری", style={'direction': 'ltr'}),
            dbc.ModalBody(
                dcc.Markdown("""
                    راهنمای استفاده از داشبورد پیشبینی فروش:
    
                    1. با استفاده از دکمه های تعبیه شده برای مقدار سال و ماه تاریخ مورد نظر را انتخاب کنید.
                     2. میتوانید با انتخاب 2 ماه مختلف بازه ی طولانی تر از یک ماه داشته باشید. ماه اول تاریخ شروع و ماه دوم تاریخ پایان است.
                    3. توجه داشته باشید که این سیستم تمامی روز های بین تاریخ شروع و پایان حتی روز های تعطیل را محاسبه میکند. 
    
                    موفق باشید!
                """, style={'color': 'white'})
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="help-close-button", className="ml-auto")),
        ], id='help-modal', style={'display': 'none', 'direction': 'rtl'}),  # Make sure to set the display style
        html.Div(id='modal-background',
                 style={'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'height': '80%',
                        'background-color': 'rgba(6, 206, 255, 0.5)', 'display': 'none', 'color': 'rgb(6, 206, 255)'}),
], style={'direction': 'rtl'})


# Create a callback to update date range and button styles
@app.callback(
    [Output('output-container-date-range', 'children'),
     Output('year-1398', 'style'),
     Output('year-1399', 'style'),
     Output('year-1400', 'style'),
     Output('year-1401', 'style'),
     Output('year-1402', 'style'),
     Output('year-1403', 'style'),
     Output('year-1404', 'style'),
     Output('year-1405', 'style'),
     Output('year-1406', 'style'),
     Output('month-farvardin', 'style'),
     Output('month-ordibehesht', 'style'),
     Output('month-khordad', 'style'),
     Output('month-tir', 'style'),
     Output('month-mordad', 'style'),
     Output('month-shahrivar', 'style'),
     Output('month-mehr', 'style'),
     Output('month-aban', 'style'),
     Output('month-azar', 'style'),
     Output('month-dey', 'style'),
     Output('month-bahman', 'style'),
     Output('month-esfand', 'style'),
     Output('prediction-plot', 'figure'),],
    [Input('year-1398', 'n_clicks'),
     Input('year-1399', 'n_clicks'),
     Input('year-1400', 'n_clicks'),
     Input('year-1401', 'n_clicks'),
     Input('year-1402', 'n_clicks'),
     Input('year-1403', 'n_clicks'),
     Input('year-1404', 'n_clicks'),
     Input('year-1405', 'n_clicks'),
     Input('year-1406', 'n_clicks'),
     Input('month-farvardin', 'n_clicks'),
     Input('month-ordibehesht', 'n_clicks'),
     Input('month-khordad', 'n_clicks'),
     Input('month-tir', 'n_clicks'),
     Input('month-mordad', 'n_clicks'),
     Input('month-shahrivar', 'n_clicks'),
     Input('month-mehr', 'n_clicks'),
     Input('month-aban', 'n_clicks'),
     Input('month-azar', 'n_clicks'),
     Input('month-dey', 'n_clicks'),
     Input('month-bahman', 'n_clicks'),
     Input('month-esfand', 'n_clicks')]
)
def update_output(*button_clicks):
    global selected_year, selected_months  # To modify global variables
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'year-1402'  # Default to 1400 if no button clicked
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Update selected year and months based on button clicks
    if button_id.startswith('year-'):
        selected_year = int(button_id.split('-')[1])
    else:
        month_clicked = button_id.split('-')[1].capitalize()
        if month_clicked in selected_months:
            selected_months.remove(month_clicked)
        else:
            selected_months.append(month_clicked)

    # Sort selected months by their index in the year
    sorted_selected_months = sorted(selected_months, key=lambda x: persian_months[x]["index"])

    # Calculate start and end dates based on selected months
    if sorted_selected_months:
        start_month = sorted_selected_months[0]
        end_month = sorted_selected_months[-1]
        start_date = f'{selected_year:04d}-{persian_months[start_month]["index"]:02d}-01'
        end_date = f'{selected_year:04d}-{persian_months[end_month]["index"]:02d}-{persian_months[end_month]["days"]:02d}'
    else:
        # If no month is selected, default to Farvardin
        start_date = f'{selected_year:04d}-01-01'
        end_date = f'{selected_year:04d}-01-{persian_months["Farvardin"]["days"]:02d}'

    # Define default styles
    default_style = {'background-color': 'rgb(233, 39, 117)'}
    selected_style = {'background-color': 'lightblue'}

    # Set styles based on selected months
    year_styles = [selected_style if button_id == f'year-{selected_year}' else default_style for button_id in ['year-1398', 'year-1399', 'year-1400', 'year-1401', 'year-1402', 'year-1403',
                                                                                                               'year-1404', 'year-1405', 'year-1406']]
    month_styles = [selected_style if button_id.split('-')[1].capitalize() in selected_months else default_style for button_id in ['month-farvardin', 'month-ordibehesht', 'month-khordad', 'month-tir', 'month-mordad', 'month-shahrivar', 'month-mehr', 'month-aban', 'month-azar', 'month-dey', 'month-bahman', 'month-esfand']]

    predictions, actual = make_predictions(start_date=start_date, end_date=end_date, model=MODEL, scaler=SCALER)
    # Convert values by dividing by 10
    predictions = [value / 10 for value in predictions]
    actual = [value / 10 for value in actual]
    total_predictions, total_actual = sum(predictions), sum(actual)
    # Calculate the sum of predictions and actual values for every 7 records
    sum_predictions = [sum(predictions[i - 7:i]) if i >= 7 else None for i in range(7, len(predictions) + 1, 7)]
    sum_actual = [sum(actual[i - 7:i]) if i >= 7 else None for i in range(7, len(actual) + 1, 7)]

    # Create a Plotly bar plot
    fig = go.Figure()

    # Add a line trace for predictions
    fig.add_trace(go.Scatter(x=list(range(1, len(predictions) + 1)), y=predictions, mode='lines+markers',
                             name='Prediction'))

    # Add a line trace for actual values
    fig.add_trace(go.Scatter(x=list(range(1, len(actual) + 1)), y=actual, mode='lines+markers',
                             name='Actual'))

    # Add a bar trace for the sum of predictions
    fig.add_trace(go.Bar(x=list(range(7, len(predictions) + 1, 7)), y=sum_predictions, name='Sum of Predictions'))

    # Add a bar trace for the sum of actual values
    fig.add_trace(go.Bar(x=list(range(7, len(actual) + 1, 7)), y=sum_actual, name='Sum of Actual'))

    # Add an annotation
    fig.add_annotation(
        text=f"<b>مجموع فروش پیشبینی شده {total_predictions:,.0f} تومان</b>",
        xref="paper", yref="paper",
        x=0.8, y=0.95,
        showarrow=False,
        # font=dict(size=12, family="Arial, sans-serif",  # Specify the desired font family
        #           color="black"),
    )
    # Add an annotation
    fig.add_annotation(
        text=f"<b>مجموعه واقعی فروش {total_actual:,.0f} نومان</b>",
        xref="paper", yref="paper",
        x=0.8, y=0.90,
        showarrow=False,
        # font=dict(size=12, family="Arial, sans-serif",  # Specify the desired font family
        #           color="black"),
    )
    fig.update_layout(
        height=550,
        title="پیشبینی فروش شعبه باملند",
        title_font=dict(size=20),
        xaxis_title="روز",
        xaxis_title_font=dict(size=16),
        yaxis_title="مبلغ فروش",
        yaxis_title_font=dict(size=16),
        # font=dict(family="Arial", size=18, color="black"),
        paper_bgcolor="rgba(255,255,255,0.5)",
        plot_bgcolor="rgba(255,255,255,0.5)",
        xaxis=dict(tickfont=dict(size=15), gridcolor='white'),
        template='plotly',
        title_x=0.5,  # Set the x-coordinate of the title to the center
        title_y=0.9,  # Set the y-coordinate of the title
        title_xanchor='center',  # Set the x-anchor to center
        title_yanchor='top',  # Set the y-anchor to the top
        margin=dict(t=100, b=50, l=50, r=50)
        # yaxis=dict(gridcolor='white'),  # Color of y-axis grid lines
    )
    # Change the theme to "plotly_dark"
    # fig.update_layout(template="plotly_dark") # for changing the background of plot
    return f' تاریخ شروع: {start_date[5:7]}-{start_date[8:10]}-{start_date[0:4]} |  تاریخ پایان: {end_date[5:7]}-{end_date[8:10]}-{end_date[0:4]}', *year_styles, *month_styles, fig


@app.callback(
    Output('help-modal', 'is_open'),
    Output('modal-background', 'style'),
    Input('help-button', 'n_clicks'),
    Input('help-close-button', 'n_clicks'),
    State('help-modal', 'is_open'),
    prevent_initial_call=True
)
def toggle_modal(help_clicks, close_clicks, is_open):
    if help_clicks is None:
        help_clicks = 0

    if close_clicks is None:
        close_clicks = 0

    # Toggle the modal based on the sum of help and close clicks
    total_clicks = help_clicks + close_clicks

    if total_clicks % 2 == 1:  # Odd number of total clicks
        return not is_open, {'display': 'block'}
    else:
        return is_open, {'display': 'none'}


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
