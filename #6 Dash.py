from dash import Dash, html, dcc, dash_table, dash
import pandas as pd
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import umap.umap_ as umap
from sklearn.cluster import KMeans
import plotly.graph_objects as go

#app = Dash(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

path = 'path_to_preprocessed_file_from_the_1st_module'
df = pd.read_csv(path, na_values='NULL')
df.set_index('Sample ID', inplace=True)

df['Sample ID'] = range(1, len(df) + 1)
PAGE_SIZE = 5
total_pages = len(df) // PAGE_SIZE
if len(df) % PAGE_SIZE != 0:
    total_pages += 1

# Set the page_current to the last page if there is data
page_current = max(total_pages - 1, 0)
# Create input fields for adding new rows
new_row_inputs = []
filtered_options = [{'label': x, 'value': x, 'disabled': False} for x in df if x not in ["Sample ID"]]
options_row1 = filtered_options[:11]  # First row of variables
options_row2 = filtered_options[11:23]  # Second row
options_row3 = filtered_options[23:] #Third row
CB_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#dede00']

#---------------------------------------------------------------------------------------------------
table_mutations = dash_table.DataTable(
    id='datatable-paging',
    columns=[],  # Empty columns initially
    page_current=page_current,
    page_size=PAGE_SIZE,
    page_action='custom',
    editable=True,
    row_deletable=False,
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(228, 236, 246)',
        }
    ],
    style_header={
            'backgroundColor': 'rgb(228, 236, 246)',
            'color': 'black',
            'fontWeight': 'bold',
            'text-align': 'center'
        }
)

checklist_mutations = dbc.Row([
    dbc.Col(
        dcc.Checklist(
            id="select_feature_group_1_row1",
            options=options_row1,
            value=['Diagnosis Age', 'C-Kit Mutation Exon 17', 'CEBPA Mutation'],
            labelStyle=dict(display='block'),
        ),
        width=4  # Adjust the width as needed
    ),
    dbc.Col(
        dcc.Checklist(
            id="select_feature_group_1_row2",
            options=options_row2,
            value=['FLT3-PM', 'Sex', 'NPM Mutation', 'NUP98-NSD1'],
            labelStyle=dict(display='block'),
        ),
        width=4  # Adjust the width as needed
    ),
    dbc.Col(
        dcc.Checklist(
            id="select_feature_group_1_row3",
            options=options_row3,
            value=[],
            labelStyle=dict(display='block'),
        ),
        width=4  # Adjust the width as needed
    ),
])


UMAP = dcc.Graph(id='Umap_projection2d_live_adding', figure={})
dummy_div = html.Div(id='dummy-div', style={'display': 'none'})

app.layout = html.Div([
    html.H1("UMAP Projection", title="With this application you can check to which risk group belong the specific patient.\n"
        "You can mark any number of the variables and then write down your results.\n"
        "Once you filled the cells with the information just press the button and wait for the results to appear.\n"
        "If you want to update the results in the table just go to the other page and come back.",
        style={'text-align': 'left', 'padding-left': '5px'}),
    table_mutations,
    html.H3("Enter a new row data:", style={'text-align': 'left', 'padding-left': '5px'}),
    html.Div(id='new-row-inputs-container'),  # Container for new row input fields
    dbc.Button('Add New Row', id='add-new-row-button', n_clicks=0, color='primary', size="sm"),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='Umap_projection2d_live_adding', figure={}),  # UMAP graph
            width=8,  # Adjust the width as needed
        ),
        dbc.Col(
            html.Div([
                html.H3("Variables"),
                checklist_mutations,
            ]),
            width=4,  # Adjust the width as needed
        ),
    ]),
    dummy_div  # Include the dummy div in the layout
])

# Callback to dynamically update the new row input fields based on checklist selection
@app.callback(
    Output('new-row-inputs-container', 'children'),
    Input('select_feature_group_1_row1', 'value'),
    Input('select_feature_group_1_row2', 'value'),
    Input('select_feature_group_1_row3', 'value')
)
def update_new_row_inputs(group_1_selected_row1, group_1_selected_row2, group_1_selected_row3):
    # Concatenate the selected features from all rows
    selected_features = group_1_selected_row1 + group_1_selected_row2 + group_1_selected_row3
    # Create input fields for each selected feature with smaller cells
    input_fields = [
        dbc.Col(
            dbc.Input(
                id={'type': 'new-row-input', 'index': feature},
                type='text',
                placeholder=f'{feature}',
                style={'width': '120px', 'font-size': '10px'}  # Adjust the width as desired
            ),
            width=1  # Adjust the column width as desired
        )
        for feature in selected_features
    ]

    # Wrap the input fields in a row
    new_row_inputs = dbc.Row(input_fields)

    return new_row_inputs

# Callback to add a new row to the DataFrame when the "Add New Row" button is clicked
@app.callback(
    Output('dummy-div', 'children'),
    Input('add-new-row-button', 'n_clicks'),
    State({'type': 'new-row-input', 'index': ALL}, 'value'),
    State('select_feature_group_1_row1', 'value'),
    State('select_feature_group_1_row2', 'value'),
    State('select_feature_group_1_row3', 'value')
)
def add_new_row(n_clicks, new_row_values, group_1_selected_row1, group_1_selected_row2, group_1_selected_row3):
    selected_features = group_1_selected_row1 + group_1_selected_row2 + group_1_selected_row3
    num_features_selected = len(selected_features)
    num_input_values = len(new_row_values)
    if num_input_values == num_features_selected:
        new_row = {col: val for col, val in zip(selected_features, new_row_values)}
        new_row_df = pd.DataFrame([new_row])
        global df
        df = pd.concat([df, new_row_df], ignore_index=True)
        df = df.sort_values(by='Sample ID', ascending=False).reset_index(drop=True)

    return n_clicks  # This will trigger the graph update

@app.callback(
    Output('datatable-paging', 'data'),
    Output('datatable-paging', 'columns'),
    Input('datatable-paging', "page_current"),
    Input('datatable-paging', "page_size"),
    Input('select_feature_group_1_row1', 'value'),
    Input('select_feature_group_1_row2', 'value'),
    Input('select_feature_group_1_row3', 'value')
)
def update_table(page_current, page_size, group_1_selected_row1, group_1_selected_row2, group_1_selected_row3):
    # Concatenate the selected features from all groups
    selected_features = group_1_selected_row1 + group_1_selected_row2 + group_1_selected_row3
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
    Input('dummy-div', 'children'),
    Input('select_feature_group_1_row1', 'value'),
    Input('select_feature_group_1_row2', 'value'),
    Input('select_feature_group_1_row3', 'value'),
)
def update_graph(dummy, group_1_selected_row1, group_1_selected_row2, group_1_selected_row3):
    if not dummy:  # Check if the dummy input has changed (i.e., a new row has been added)
        return go.Figure()  # Return an empty figure if the dummy input hasn't changed

    selected_features = group_1_selected_row1 + group_1_selected_row2 + group_1_selected_row3
    dff = df[selected_features]

    if len(selected_features) > 0:
        kmeans = KMeans(5)
        kmeans.fit(dff)
        clusters = dff.copy()
        clusters['clusters_pred'] = kmeans.fit_predict(dff)
        proj_2d = umap.UMAP(random_state=42).fit_transform(dff)

        fig = go.Figure()

        for cluster_id in range(0,5):
            cluster_points = proj_2d[clusters['clusters_pred'] == cluster_id]

            fig.add_trace(go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                marker=dict(
                    size=4,
                    color=CB_colors[cluster_id],
                ),
                showlegend=True,
                name=f'Cluster {cluster_id}'
            ))

        fig.add_trace(go.Scatter(
            x=[proj_2d[-1, 0]],
            y=[proj_2d[-1, 1]],
            mode='markers',
            marker=dict(size=12, color='red'),
            showlegend=False
        ))

        fig.update_layout(
            height=600,
            width=800,
            legend=dict(
                x=1.05,  # Adjust the x position of the legend
                y=0.5,  # Adjust the y position of the legend
                itemsizing='constant'  # Set the legend item size
            )
        )

        return fig
    else:
        return go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)
