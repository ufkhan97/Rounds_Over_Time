import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
import plotly.graph_objs as go
import plotly.express as px
import locale
import time
import networkx as nx
import utils

st.set_page_config(
    page_title="Round Performance Data - Gitcoin Grants",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)



with open("6433c5d029c6bb20c5f00bf8_GTC-Logotype-Dark.svg", "r") as file:
    svg_image = file.read().replace('<svg', '<svg style="max-width: 300px;"')
st.markdown(svg_image, unsafe_allow_html=True)
st.write('')

program_data = pd.read_csv("all_rounds.csv")
col1, col2 = st.columns(2)

program1 = st.selectbox('Select First Program', program_data['program'].unique())
program2 = st.selectbox('Select Second Program', [prog for prog in program_data['program'].unique() if prog != program1])

program_options = [program1, program2]

if "program_options" in st.session_state and st.session_state.program_options != program_options:
    st.session_state.data_loaded = False
st.session_state.program_options = program_options



if "data_loaded" in st.session_state and st.session_state.data_loaded:
    dfv = st.session_state.dfv
    dfp = st.session_state.dfp
    round_data = st.session_state.round_data
else:
    data_load_state = st.text('Loading data...')
    dfv, dfp, round_data = utils.load_round_data(program_options, "all_rounds.csv")
    data_load_state.text("")


def create_token_comparison_bar_chart(dfv):
    # Group by token_symbol and sum the amountUSD
    grouped_data = dfv.groupby('token_symbol')['amountUSD'].sum().reset_index()
    # Calculate the total amountUSD for percentage calculation
    total_amountUSD = grouped_data['amountUSD'].sum()
    # Calculate the percentage for each token
    grouped_data['percentage'] = (grouped_data['amountUSD'] / total_amountUSD) 
    # Create the bar chart with renamed axes and title
    fig = px.bar(grouped_data, x='token_symbol', y='amountUSD', 
                 title='Contributions (in USD) by Token', 
                 labels={'token_symbol': 'Token', 'amountUSD': 'Contribution (USD)'})
    # Update hover template to display clean USD numbers
    fig.update_traces(hovertemplate='Token: %{x}<br>Contribution: $%{y:,.2f}')
    fig.update_yaxes(tickprefix="$", tickformat="2s")
    # Add percentage as labels on the bars
    fig.update_traces(texttemplate='%{customdata:.2%}', textposition='outside', customdata=grouped_data['percentage'])
    # Add padding at the top of the function for the texttemplate and increase the text size
    fig.update_layout(
        autosize=False,
        height=600,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=10
        ),
        font=dict(
            size=14,
        )
    )
    return fig

def get_USD_by_round_chart(dfp, color_map):
    grouped_data = dfp.groupby('round_name')['amountUSD'].sum().reset_index().sort_values('amountUSD', ascending=False)
    fig = px.bar(grouped_data, y='round_name', x='amountUSD', title='Crowdfunded (in USD) by Round', 
                 color='round_name', labels={'amountUSD': 'Crowdfunded Amount (USD)', 'round_name': 'Round Name'}, 
                 color_discrete_map=color_map, orientation='h')
    fig.update_traces(hovertemplate='Amount: $%{x:,.2f}', texttemplate='$%{x:,.3s}', textposition='auto')
    fig.update_layout(showlegend=False, height=600)  # Expanded height
    fig.update_xaxes(tickprefix="$", tickformat="2s")
    return fig

def get_contributions_by_round_chart(dfp, color_map):
    grouped_data = dfp.groupby('round_name')['votes'].sum().reset_index().sort_values('votes', ascending=False)
    fig = px.bar(grouped_data, y='round_name', x='votes', title='Total Contributions (#) by Round', 
                 color='round_name', labels={'votes': 'Number of Contributions', 'round_name': 'Round Name'}, 
                 color_discrete_map=color_map, orientation='h')
    fig.update_traces(hovertemplate='Number of Contributions: %{x:,.2f}', texttemplate='%{x:,.3s}', textposition='auto')
    fig.update_layout(showlegend=False, height=600)  # Expanded height
    fig.update_xaxes(tickprefix="", tickformat="2s")
    return fig

def get_contribution_time_series_chart(dfv):
    dfv_count = dfv.groupby([dfv['block_timestamp'].dt.strftime('%m-%d-%Y %H')])['id'].nunique()
    dfv_count.index = pd.to_datetime(dfv_count.index)
    dfv_count = dfv_count.reindex(pd.date_range(start=dfv_count.index.min(), end=dfv_count.index.max(), freq='H'), fill_value=0)
    fig = px.bar(dfv_count, x=dfv_count.index, y='id', labels={'id': 'Number of Contributions', 'index': 'Time'}, title='Hourly Contributions over Time')
    fig.update_layout()
    return fig 

def get_cumulative_amountUSD_time_series_chart(dfv):
    dfv_cumulative = dfv.groupby([dfv['block_timestamp'].dt.strftime('%m-%d-%Y %H')])['amountUSD'].sum().cumsum()
    dfv_cumulative.index = pd.to_datetime(dfv_cumulative.index)
    dfv_cumulative = dfv_cumulative.reindex(pd.date_range(start=dfv_cumulative.index.min(), end=dfv_cumulative.index.max(), freq='H'), method='pad')
    fig = px.area(dfv_cumulative, x=dfv_cumulative.index, y='amountUSD', labels={'amountUSD': 'Total Donations (USD)', 'index': 'Time'}, title='Total Donations Over Time (USD)')
    fig.update_layout()
    fig.update_yaxes(tickprefix="$", tickformat="2s")
    return fig


def get_cumulative_amountUSD_time_series_comparison_chart(dfv):
    dfv_cumulative = dfv.groupby(['program', 'hours_since_start'])['amountUSD'].sum().groupby(level=[0]).cumsum().reset_index()
    # Create a side by side bar chart by setting the 'barmode' to 'group'
    fig = px.bar(dfv_cumulative, x='hours_since_start', y='amountUSD', color='program', 
                  labels={'amountUSD': 'Total Donations (USD)', 'hours_since_start': 'Hours Since Start of Round'}, 
                  title='Total Donations Over Time (USD)', barmode='group')
    fig.update_layout()
    fig.update_yaxes(tickprefix="$", tickformat="2s")
    return fig

def get_cumulative_amountUSD_time_series_comparison_chart_by_round(dfv):
    dfv['round_name'] = dfv['round_name'].str.strip()
    dfv['round_name'] = dfv['round_name'].str.replace('&', 'and')
    round_name = st.selectbox('Select a round', dfv['round_name'].unique())
    dfv = dfv[dfv['round_name'] == round_name]
    dfv_cumulative = dfv.groupby(['program', 'hours_since_start'])['amountUSD'].sum().groupby(level=[0]).cumsum().reset_index()
    # Create a side by side bar chart by setting the 'barmode' to 'group'
    fig = px.bar(dfv_cumulative, x='hours_since_start', y='amountUSD', color='program', 
                  labels={'amountUSD': 'Total Donations (USD)', 'hours_since_start': 'Hours Since Start of Round'}, 
                  title='Total Donations Over Time (USD)', barmode='group')
    fig.update_layout()
    fig.update_yaxes(tickprefix="$", tickformat="2s")
    return fig

def create_treemap(dfp):
    dfp['shortened_title'] = dfp['title'].apply(lambda x: x[:15] + '...' if len(x) > 20 else x)
    fig = px.treemap(dfp, path=['shortened_title'], values='amountUSD', hover_data=['title', 'amountUSD'])
    # Update hovertemplate to format the hover information
    fig.update_traces(
        texttemplate='%{label}<br>$%{value:.3s}',
        hovertemplate='<b>%{customdata[0]}</b><br>Amount: $%{customdata[1]:,.2f}',
        textposition='middle center',
        textfont_size=20
    )
    fig.update_traces(texttemplate='%{label}<br>$%{value:.3s}', textposition='middle center', textfont_size=20)
    fig.update_layout(font=dict(size=20))
    fig.update_layout(height=550)
    fig.update_layout(title_text="Donations by Grant")
    return fig


def generate_donation_difference_chart(dfv, round_names):
    dfv = dfv[dfv['round_name'].isin(round_names)]
    dfv_cumulative_at_min_max_hours = dfv.groupby(['round_name', 'program'])['amountUSD'].sum().reset_index()
    dfv_cumulative_at_min_max_hours.columns = ['Round Name', 'Program', 'Cumulative AmountUSD at Min of Max Hours']
    dfv_cumulative_at_min_max_hours_pivot = dfv_cumulative_at_min_max_hours.pivot(index='Round Name', columns='Program', values='Cumulative AmountUSD at Min of Max Hours')
    dfv_cumulative_at_min_max_hours_pivot['Percent Difference'] = dfv_cumulative_at_min_max_hours_pivot.apply(lambda row: (row[1] - row[0]) / row[0] * 100, axis=1)
    dfv_cumulative_at_min_max_hours_pivot.dropna(inplace=True)
    dfv_cumulative_at_min_max_hours_pivot.reset_index(inplace=True)
    dfv_cumulative_at_min_max_hours_pivot = dfv_cumulative_at_min_max_hours_pivot.sort_values(by='Percent Difference')
    fig = px.bar(dfv_cumulative_at_min_max_hours_pivot, y='Round Name', x='Percent Difference', 
                 title='Cumulative Donation Amount Percent Difference at Current Time by Round', 
                 labels={'Percent Difference': 'Percent Difference (%)', 'Round Name': 'Round Name'}, orientation='h')
                 #color='Percent Difference', color_continuous_scale='RdYlGn', color_continuous_midpoint=0) # Center the gradient at 0
    fig.update_layout(showlegend=False  ) 
    fig.update_xaxes(tickprefix="", tickformat=".2s%")
    fig.update_traces(texttemplate='%{x:.2f}%', textposition='inside', textfont_size=16)  # Move the texttemplate to the outside of the bars
    return fig

def generate_donor_churn_and_retention_chart(dfv, program1, program2):
    # Filter data for each program
    dfv_program1 = dfv[dfv['program'] == program1]
    dfv_program2 = dfv[dfv['program'] == program2]

    # Get unique donors for each program
    donors_program1 = set(dfv_program1['voter'].unique())
    donors_program2 = set(dfv_program2['voter'].unique())

    # Calculate new and returning donors for program1
    new_donors_program1 = donors_program1 - donors_program2
    returning_donors_program1 = donors_program1.intersection(donors_program2)

    # Calculate donors from program2 who did not return
    non_returning_donors_program2 = donors_program2 - donors_program1

    # Prepare data for visualization
    data = {
        'New Donors': len(new_donors_program1),
        'Returning Donors': len(returning_donors_program1),
        'Non-returning Donors': len(non_returning_donors_program2)
    }

    # Create a bar chart
    fig = px.bar(x=list(data.keys()), y=list(data.values()), title='Donor Churn and Retention', labels={'x': 'Category', 'y': 'Count'})

    return fig


max_hours_by_program = dfv.groupby('program')['hours_since_start'].max()
min_max_hours = min(max_hours_by_program)
st.title('Hours Since Start of Round: ' +  str(min_max_hours) )
if 'GG19' in program_options:
    target_time = datetime(2023, 11, 29, 23, 59, tzinfo=timezone.utc)
    time_left = utils.get_time_left(target_time)
    st.subheader("⏰ Time Left: " + (time_left))
st.write('This dashboard is to compare programs at similar points in time.')


dfv = dfv[dfv['hours_since_start'] <= min_max_hours]


columns = st.columns(3)
metrics = [ 'Total Donated', 'Total Donations', 'Unique Donors', 'Total Rounds', 'Avg. Amount per Donor', 'Avg. Projects per Donor']
for i, program in enumerate(program_options):
    dfv_program = dfv[dfv['program'] == program]
    dfp_program = dfp[dfp['program'] == program]
    round_data_program = round_data[round_data['program'] == program]
    columns[i].subheader(f'Summary for {program}')
    columns[i].metric(metrics[0], '${:,.2f}'.format(dfv_program['amountUSD'].sum()))
    columns[i].metric(metrics[1], '{:,.0f}'.format(dfv_program['id'].nunique()))
    columns[i].metric(metrics[2], '{:,.0f}'.format(dfv_program['voter'].nunique()))
    columns[i].metric(metrics[3], '{:,.0f}'.format(round_data_program['round_id'].nunique()))
    avg_amount_per_voter = dfv_program['amountUSD'].sum() / dfv_program['voter'].nunique()
    projects_per_voter = dfv_program.groupby('voter')['projectId'].nunique()
    avg_projects_per_voter = projects_per_voter.mean()
    columns[i].metric(metrics[4], '${:,.2f}'.format(avg_amount_per_voter))
    columns[i].metric(metrics[5], '{:,.2f}'.format(avg_projects_per_voter))

if len(program_options) == 2:
    dfv_program1 = dfv[dfv['program'] == program_options[0]]
    dfv_program2 = dfv[dfv['program'] == program_options[1]]
    round_data_program1 = round_data[round_data['program'] == program_options[0]]
    round_data_program2 = round_data[round_data['program'] == program_options[1]]
    columns[2].subheader('Percent Difference')
    columns[2].metric(metrics[0], '{:,.2f}%'.format((dfv_program1['amountUSD'].sum() - dfv_program2['amountUSD'].sum()) / dfv_program2['amountUSD'].sum() * 100))
    columns[2].metric(metrics[1], '{:,.2f}%'.format((dfv_program1['id'].nunique() - dfv_program2['id'].nunique()) / dfv_program2['id'].nunique() * 100))
    columns[2].metric(metrics[2], '{:,.2f}%'.format((dfv_program1['voter'].nunique() - dfv_program2['voter'].nunique()) / dfv_program2['voter'].nunique() * 100))
    columns[2].metric(metrics[3], '{:,.2f}%'.format((round_data_program1['round_id'].nunique() - round_data_program2['round_id'].nunique()) / round_data_program2['round_id'].nunique() * 100))
    avg_amount_per_voter1 = dfv_program1['amountUSD'].sum() / dfv_program1['voter'].nunique()
    avg_amount_per_voter2 = dfv_program2['amountUSD'].sum() / dfv_program2['voter'].nunique()
    columns[2].metric(metrics[4], '{:,.2f}%'.format((avg_amount_per_voter1 - avg_amount_per_voter2) / avg_amount_per_voter2 * 100))
    projects_per_voter1 = dfv_program1.groupby('voter')['projectId'].nunique()
    projects_per_voter2 = dfv_program2.groupby('voter')['projectId'].nunique()
    avg_projects_per_voter1 = projects_per_voter1.mean()
    avg_projects_per_voter2 = projects_per_voter2.mean()
    columns[2].metric(metrics[5], '{:,.2f}%'.format((avg_projects_per_voter1 - avg_projects_per_voter2) / avg_projects_per_voter2 * 100))


program_round_combinations = dfv[['program', 'round_name']].drop_duplicates()
round_counts = program_round_combinations['round_name'].value_counts()
multiple_round_names = round_counts[round_counts > 1].index.tolist()
dfv = dfv[dfv['round_name'].isin(multiple_round_names)]

round_names = st.multiselect('Select Recurring Rounds to See Percent Difference', dfv['round_name'].unique(), default=['Web3 Open Source Software', 'Web3 Community and Education', 'Ethereum Infrastructure'])
col1, col2 = st.columns(2)
col1.plotly_chart(generate_donation_difference_chart(dfv, round_names), use_container_width=True)
col2.plotly_chart(get_cumulative_amountUSD_time_series_comparison_chart(dfv), use_container_width=True)


