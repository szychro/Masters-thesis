from dash import Dash, html, dcc, dash_table
import pandas as pd
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import umap.umap_ as umap
from sklearn.cluster import KMeans
import plotly.graph_objects as go

app = Dash(__name__)
#---------------------------------------------------------------------------------------------------
path = 'C:/Users/Szymon/Desktop/Praca/Seattle project/Public_datasets/public/SWAG_output/1_preprocessed_target.csv'
df = pd.read_csv(path, na_values='NULL')
df.set_index('Patient ID', inplace=True)
df['Patient ID'] = range(1, len(df) + 1)
PAGE_SIZE = 5
total_pages = len(df) // PAGE_SIZE
if len(df) % PAGE_SIZE != 0:
    total_pages += 1

# Set the page_current to the last page if there is data
page_current = max(total_pages - 1, 0)
# Create input fields for adding new rows
new_row_inputs = []
#---------------------------------------------------------------------------------------------------
table_mutations = dash_table.DataTable(
    id='datatable-paging',
    columns=[],  # Empty columns initially
    page_current=page_current,
    page_size=PAGE_SIZE,
    page_action='custom',
    editable=True,  # Allow editing of the table
    row_deletable=True,  # Allow deletion of rows
)

checklist_mutations = dcc.Checklist(
    id="select_feature_group_1",
    options=[{'label': x, 'value': x, 'disabled': False} for x in df],
    value=['Diagnosis Age', 'NPM Mutation', 'inv(16)', 'WT1 Mutation', 'FLT3-PM', 'Sex'],
    labelStyle=dict(display='inline')
)


UMAP = dcc.Graph(id='Umap_projection2d_live_adding', figure={})
dummy_div = html.Div(id='dummy-div', style={'display': 'none'})

app.layout = html.Div([
    html.H1("UMAP Projection for original data", title="With this application you can check to which risk group belong the specific patient.\n"
                                                       "You can mark any number of the variables and then write down your results.\n"
                                                       "Once you filled the cells with the information just press the button and wait for the results to appear.\n"
                                                       "If you want to update the results in the table just go to other page and come back.\n"
                                                       "If you do not know what the specific abbrevation mean, just hover on it",
            style={'text-align': 'left'}),
    checklist_mutations,
    table_mutations,
    html.H3("Enter new row data:"),
    html.Div(id='new-row-inputs-container'),  # Container for new row input fields
    dbc.Button('Add New Row', id='add-new-row-button', n_clicks=0, color='primary'),
    dcc.Graph(id='Umap_projection2d_live_adding', figure={}),  # Add the UMAP graph here
    dummy_div  # Include the dummy div in the layout
])

# Callback to dynamically update the new row input fields based on checklist selection
@app.callback(
    Output('new-row-inputs-container', 'children'),
    Input('select_feature_group_1', 'value'))

def update_new_row_inputs(group_1_selected):
    # Concatenate the selected features from all groups
    selected_features = group_1_selected
    # Create input fields for each selected feature with smaller cells
    new_row_inputs = [
        dbc.Input(
            id={'type': 'new-row-input', 'index': feature},
            type='text',
            placeholder=f'Enter {feature}',
            style={'width': '80px'}  # Adjust the width as desired
        )
        for feature in selected_features
    ]
    return new_row_inputs

# Callback to add a new row to the DataFrame when the "Add New Row" button is clicked
@app.callback(
    Output('dummy-div', 'children'),
    Input('add-new-row-button', 'n_clicks'),
    State({'type': 'new-row-input', 'index': ALL}, 'value'),
    State('select_feature_group_1', 'value')

)
def add_new_row(n_clicks, new_row_values, group_1_selected):
    if n_clicks > 0:
        selected_features = group_1_selected
        num_features_selected = len(selected_features)
        num_input_values = len(new_row_values)
        if num_input_values == num_features_selected:
            # Create a dictionary for the new row data
            new_row = {col: val for col, val in zip(selected_features, new_row_values)}
            # Convert the new row to a DataFrame
            new_row_df = pd.DataFrame([new_row])
            # Concatenate the new row DataFrame to the existing DataFrame
            global df
            df = pd.concat([df, new_row_df], ignore_index=True)

            # Sort DataFrame in reverse order based on the index column (Pat) and reset the index
            df = df.sort_values(by='Patient ID', ascending=False).reset_index(drop=True)

    return None

@app.callback(
    Output('datatable-paging', 'data'),
    Output('datatable-paging', 'columns'),
    Input('datatable-paging', "page_current"),
    Input('datatable-paging', "page_size"),
    Input('select_feature_group_1', 'value')
)
def update_table(page_current, page_size, group_1_selected):
    # Concatenate the selected features from all groups
    selected_features = group_1_selected
    # Filter the DataFrame based on selected features
    filtered_df = df[selected_features]
    # Generate columns dynamically based on selected features
    columns = [{"name": col, "id": col} for col in filtered_df.columns]

    start_index = page_current * page_size
    end_index = (page_current + 1) * page_size

    # Get the sliced data for the current page
    sliced_data = filtered_df.iloc[start_index:end_index].to_dict('records')

    return sliced_data, columns

@app.callback(
    Output('Umap_projection2d_live_adding', 'figure'),
    Input('select_feature_group_1', 'value'),
    Input('add-new-row-button', 'n_clicks'),  # Add the button as an input
)
def update_graph(group_1_selected, n_clicks):
    # Concatenate the selected features from all groups
    selected_features = group_1_selected
    # Filter the DataFrame based on selected features
    dff = df[selected_features]
    kmeans = KMeans(4)
    kmeans.fit(dff)
    clusters = dff.copy()
    clusters['clusters_pred'] = kmeans.fit_predict(dff)
    proj_2d = umap.UMAP(random_state=42).fit_transform(dff)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=proj_2d[:, 0],
        y=proj_2d[:, 1],
        mode='markers',
        marker=dict(
            size=3,
            color=clusters['clusters_pred'],
            colorscale='Viridis',
            showscale=True
        ),
        showlegend=False
    ))

    # Adding a red marker at the last point
    fig.add_trace(go.Scatter(
        x=[proj_2d[-1, 0]],
        y=[proj_2d[-1, 1]],
        mode='markers',
        marker=dict(size=10, color='red'),
        showlegend=False
    ))
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)