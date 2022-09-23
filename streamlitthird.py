import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
from mplsoccer import VerticalPitch
from mplsoccer.pitch import Pitch

###FUNCTIONS
##HEATMAP
def dfxg_preparation(df_input):
    df = df_input.copy()

    # selecting only shots
    df = df[df['type'] == 'Shot']
    df = df.reset_index(drop=True)
    # splitting coordinates
    for i in range(len(df)):
        df.loc[i, 'location'] = df.loc[i, 'location'][1:-1]
    df[['X', 'Y', 'Z']] = df.location.str.split(",", expand=True, )
    # converting splitted coordinates to float
    for column in ('X', 'Y', 'Z'):
        for i in range(len(df)):
            df.loc[i, column] = float(df.loc[i, column])

    return df
def xg_plot(df, fig, pitch, axs):
  df_H_goals = df[df['shot_outcome'] == 'Goal']
  df_H_nogoals = df[df['shot_outcome'] != 'Goal']

  pitch.draw(ax=axs)
  #plot the shots
  sc1 = pitch.scatter(df_H_goals["X"], df_H_goals["Y"], s=df_H_goals["shot_statsbomb_xg"]*500+100,
                      c="green", ax=axs)
  sc2 = pitch.scatter(df_H_nogoals["X"], df_H_nogoals["Y"], s=df_H_nogoals["shot_statsbomb_xg"]*500+100,
                    c="red", ax=axs)
def get_maxplayer(df):
      df = df[['player', 'shot_statsbomb_xg']]
      df = df.groupby('player').sum().sort_values(by=['shot_statsbomb_xg'], ascending=False).head(2)
      df = df.reset_index()
      return [df.loc[0, 'player'], round(df.loc[0, 'shot_statsbomb_xg'], 2)]
def xg_heatmap(df_input):
  df = dfxg_preparation(df_input)
  #get the player with most xg
  df2 = df[df['team'] == 'Houston Dash'].copy()
  df2_shots = len(df2)
  df2_xg = round(df2['shot_statsbomb_xg'].sum(), 2)
  top_xg2 = get_maxplayer(df2.copy())
  df1 = df[df['team'] != 'Houston Dash'].copy()
  df1['X'] = 120 - df1['X']
  df1['Y'] = 80 - df1['Y']
  df1_shots = len(df1)
  df1_xg = round(df1['shot_statsbomb_xg'].sum(), 2)
  top_xg1 = get_maxplayer(df1.copy())

  #fig and pitch parameters
  plt.rcParams["figure.figsize"] = (18,16)
  fig, axs = plt.subplots(1, 1)
  fig.patch.set_facecolor('black')
  fig.suptitle("OL Reign - Houston Dash 1-2", color="white", fontsize=20, x=0.26, y=0.83)
  axs.set_title("Expected goals:" + str(df1_xg) + ' - ' + str(df2_xg) + '  |  Total shots: ' + str(df1_shots) + ' - ' + str(df2_shots) +
               '  |  Most involved players: ' + top_xg1[0] + '(' + str(top_xg1[1]) +')' + ' - ' + top_xg2[0] + '(' + str(top_xg2[1]) +')',
                fontsize = 14, color = 'white', x = 0.44, y = 0.97)
  pitch = Pitch(pitch_color='black', line_color='white')

  xg_plot(df1, fig, pitch, axs)
  xg_plot(df2, fig, pitch, axs)

  return fig
##PASS NETWORK
def split_coordinates(df):
  df['location'] = df['location'].fillna('[0,0]')
  #splitting coordinates
  for i in range(len(df)):
    df.loc[i,'location'] = df.loc[i,'location'][1:-1]
  #for shots since they have sometimes 3 coordinates
  for i in range(len(df)):
    if(df.loc[i,'location'].count(',') > 1):
      df = df.drop([i])
  df = df.reset_index(drop=True)
  df[['pos_x','pos_y']] = df.location.str.split(",",expand=True)
  #converting splitted coordinates to float
  for column in ('pos_x', 'pos_y'):
    for i in range(len(df)):
      df.loc[i, column] = float(df.loc[i, column])
  df = df.replace(0, np.nan)
  return df
def pass_network(df_half):
  pass_raw = df_half[df_half['type'] == 'Pass']
  pass_number_raw = pass_raw[['timestamp', 'player', 'pass_recipient']]
  #identify the two players that has passed the ball between each other
  pass_number_raw['pair'] = pass_number_raw.player + pass_number_raw.pass_recipient
  #number of passes for every pair
  pass_count = pass_number_raw.groupby(['pair']).count().reset_index()
  pass_count = pass_count[['pair', 'timestamp']]
  pass_count.columns = ['pair', 'number_pass']
  #get the average location of each player
  avg_loc_df = pass_raw[['team', 'player', 'location']]
  avg_loc_df = avg_loc_df.reset_index()
  avg_loc_df = split_coordinates(avg_loc_df)
  avg_loc_df = avg_loc_df.groupby(['team','player']).agg({'pos_x': np.mean, 'pos_y': np.mean}).reset_index()
  #merge pairs of passes and drop duplicates
  pass_merge = pass_number_raw.merge(pass_count, on='pair')
  pass_merge = pass_merge[['player', 'pass_recipient', 'number_pass']]
  pass_merge = pass_merge.drop_duplicates()
  #merge passes and avg location
  avg_loc_df = avg_loc_df[['player', 'pos_x', 'pos_y']]
  pass_cleaned = pass_merge.merge(avg_loc_df, on='player')
  pass_cleaned.rename({'pos_x': 'pos_x_start', 'pos_y': 'pos_y_start'}, axis='columns', inplace=True)
  pass_cleaned = pass_cleaned.merge(avg_loc_df, left_on='pass_recipient', right_on='player', suffixes=['', '_end'])
  pass_cleaned.rename({'pos_x': 'pos_x_end', 'pos_y': 'pos_y_end'}, axis='columns', inplace=True)
  pass_cleaned = pass_cleaned.drop(['player_end'], axis=1)
  #get the 11 players with most minutes
  player_df = df_half[df_half.team == 'Houston Dash'].groupby('player').agg({'minute': [min, max]}).reset_index()
  player_df = pd.concat([player_df['player'], player_df['minute']], axis=1)
  player_df['minutes_played'] = player_df['max'] - player_df['min']
  player_df = player_df.sort_values('minutes_played', ascending=False)
  #get only the passes of that players
  player_names = player_df.player[:11].tolist()
  pass_H = pass_cleaned[pass_cleaned.player.isin(player_names)]
  pass_H = pass_H[pass_H.pass_recipient.isin(player_names)]
  #set width, will use it in the viz
  pass_H['width'] = pass_H['number_pass'] / pass_H['number_pass'].count()

  return pass_H
def plot_passmap(df):
  MIN_TRANSPARENCY = 0.3
  color = np.array(to_rgba('white'))
  color = np.tile(color, (len(df), 1))
  c_transparency = df.number_pass / df.number_pass.max()
  c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
  color[:, 3] = c_transparency

  pitch = Pitch(pitch_type='statsbomb', pitch_color='black', line_color='#c7d5cc')
  fig, axs = pitch.grid(figheight=10, title_height=0.08, endnote_space=0, axis=False,
                        title_space=0, grid_height=0.82, endnote_height=0.05)

  fig.set_facecolor("black")

  pass_lines = pitch.lines(df.pos_x_start, df.pos_y_start,
                          df.pos_x_end, df.pos_y_end, lw=df.width+0.8,
                          color=color, zorder=1, ax=axs['pitch'])

  pass_nodes = pitch.scatter(df.pos_x_start, df.pos_y_start, s=450,
                            color='darkorange', edgecolors='black', linewidth=1, alpha=1, ax=axs['pitch'])

  for index, row in df.iterrows():
      pitch.annotate(row.player, xy=(row.pos_x_start-3, row.pos_y_start-3), c='white', va='center',
                    ha='center', size=12, ax=axs['pitch'])

  axs['title'].text(0.5, 0.7, 'Passing Network Houston Dash', color='#c7d5cc',
                    va='center', ha='center', fontsize=30)
  axs['title'].text(0.5, 0.25, 'First Half of The Game', color='#c7d5cc',
                    va='center', ha='center', fontsize=18)
  return fig
##PASSING_INDEPTH-1
def dfpass_preparation(df_input, team_name):
  #selecting only passes
  df = df_input.copy()
  df = df[(df['team'] == team_name) & (df['type'] == 'Pass')]
  df = df.reset_index(drop=True)

  #splitting coordinates
  for i in range(len(df)):
    df.loc[i,'location'] = df.loc[i,'location'][1:-1]
  df[['X','Y']] = df.location.str.split(",",expand=True,)
  #converting splitted coordinates to float
  for column in ('X', 'Y'):
    for i in range(len(df)):
      df.loc[i, column] = float(df.loc[i, column])

  #splitting coordinates
  for i in range(len(df)):
    df.loc[i,'pass_end_location'] = df.loc[i,'pass_end_location'][1:-1]
  df[['end_X','end_Y']] = df.pass_end_location.str.split(",",expand=True,)
  #converting splitted coordinates to float
  for column in ('end_X', 'end_Y'):
    for i in range(len(df)):
      df.loc[i, column] = float(df.loc[i, column])

  return df
def passes_type(df_in, forwardpasses, backwardpasses, torightpasses, toleftpasses):
  df_out = df_in.copy()
  if forwardpasses:
    df1=df_in.copy()
    df_f = df1[(df1['end_X'] > df1['X'])]
    df_f = df_f.reset_index(drop=True)
    df_out = df_out.merge(df_f, how='inner')
  if backwardpasses:
    df2=df_in.copy()
    df_b = df2[df2['end_X'] < df2['X']]
    df_b = df_b.reset_index(drop=True)
    df_out = df_out.merge(df_b, how='inner')
  if torightpasses:
    df3=df_in.copy()
    df_r = df3[df3['end_Y'] > df3['Y']]
    df_r = df_r.reset_index(drop=True)
    df_out = df_out.merge(df_r, how='inner')
  if toleftpasses:
    df4=df_in.copy()
    df_l = df4[df4['end_Y'] < df4['Y']]
    df_l = df_l.reset_index(drop=True)
    df_out = df_out.merge(df_l, how='inner')
  return df_out
def zones_passes(df, zone1, zone2, pitch, axs, forwardpasses, backwardpasses, torightpasses, toleftpasses):
  df = passes_type(df, forwardpasses, backwardpasses, torightpasses, toleftpasses)
  for i in range(len(df)):
    if (df.loc[i, 'X'] <= zone1):
      x_values = [df.loc[i, 'X'], df.loc[i, 'end_X']]
      y_values = [df.loc[i, 'Y'], df.loc[i, 'end_Y']]

      if isinstance(df.loc[i,'pass_outcome'], str):
          pitch.plot(x_values, y_values, color='red', ax=axs[0])
          pitch.scatter(df.loc[i, 'end_X'],df.loc[i, 'end_Y'],color='red', ax = axs[0])
      else:
          pitch.plot(x_values, y_values, color='green', ax=axs[0])
          pitch.scatter(df.loc[i, 'end_X'],df.loc[i, 'end_Y'],color='green', ax = axs[0])
    elif (df.loc[i, 'X'] > zone1) & (df.loc[i, 'X'] <= zone2):
      x_values = [df.loc[i, 'X'], df.loc[i, 'end_X']]
      y_values = [df.loc[i, 'Y'], df.loc[i, 'end_Y']]

      if isinstance(df.loc[i,'pass_outcome'], str):
          pitch.plot(x_values, y_values, color='red', ax=axs[1])
          pitch.scatter(df.loc[i, 'end_X'],df.loc[i, 'end_Y'],color='red', ax = axs[1])
      else:
          pitch.plot(x_values, y_values, color='green', ax=axs[1])
          pitch.scatter(df.loc[i, 'end_X'],df.loc[i, 'end_Y'],color='green', ax = axs[1])
    elif (df.loc[i, 'X'] > zone2):
      x_values = [df.loc[i, 'X'], df.loc[i, 'end_X']]
      y_values = [df.loc[i, 'Y'], df.loc[i, 'end_Y']]

      if isinstance(df.loc[i,'pass_outcome'], str):
          pitch.plot(x_values, y_values, color='red', ax=axs[2])
          pitch.scatter(df.loc[i, 'end_X'],df.loc[i, 'end_Y'],color='red', ax = axs[2])
      else:
          pitch.plot(x_values, y_values, color='green', ax=axs[2])
          pitch.scatter(df.loc[i, 'end_X'],df.loc[i, 'end_Y'],color='green', ax = axs[2])
def zones_passmap(df_input, team_name, forwardpasses, backwardpasses, torightpasses, toleftpasses):

  df = dfpass_preparation(df_input, team_name)

  #fig and pitch parameters
  plt.rcParams["figure.figsize"] = (25,20)
  pitch = VerticalPitch(pitch_color='black', line_color='white')
  fig, axs = plt.subplots(nrows=1, ncols=3)
  fig.patch.set_facecolor('black')

  pitch.draw(ax=axs[0])
  axs[0].set_title("Defensive passes", color="white", fontsize=16)
  pitch.draw(ax=axs[1])
  axs[1].set_title("Midfield passes", color="white", fontsize=16)
  pitch.draw(ax=axs[2])
  axs[2].set_title("Offensive passes", color="white", fontsize=16)

  zones_passes(df, 25, 75, pitch, axs, forwardpasses, backwardpasses, torightpasses, toleftpasses)
  return fig
##PASSING_INDEPTH-2
def passmap_byzone(df_input, team_name, xin, yin, xend, yend):
    df = dfpass_preparation(df_input, team_name)

    #fig and pitch parameters
    plt.rcParams["figure.figsize"] = (20, 15)
    pitch = Pitch(pitch_color='black', line_color='white')
    fig, axs = plt.subplots(nrows=1, ncols=1)
    fig.patch.set_facecolor('black')
    pitch.draw(ax=axs)
    axs.set_title("Passmap - zone focus", color="white", fontsize=23, loc="left")

    for i in range(len(df)):
      if (xin[0] <= df.loc[i, 'X'] < xin[1]) and (yin[0] <= df.loc[i, 'Y'] <= yin[1]) and (xend[0] <= df.loc[i, 'end_X'] < xend[1]) and (yend[0] <= df.loc[i, 'end_Y'] < yend[1]):
        x_values = [df.loc[i, 'X'], df.loc[i, 'end_X']]
        y_values = [df.loc[i, 'Y'], df.loc[i, 'end_Y']]

        if isinstance(df.loc[i,'pass_outcome'], str):
            pitch.plot(x_values, y_values, color='red', ax=axs)
            pitch.scatter(df.loc[i, 'end_X'],df.loc[i, 'end_Y'],color='red', ax = axs)
        else:
            pitch.plot(x_values, y_values, color='green', ax=axs)
            pitch.scatter(df.loc[i, 'end_X'],df.loc[i, 'end_Y'],color='green', ax = axs)
    return fig
##PRESSURE HEATMAP
def df_forhm(df, team_name, event):
  df = df[(df['team'] == team_name) & (df['type'] == event)]
  df = df.reset_index(drop=True)

  for i in range(len(df)):
    df.loc[i,'location'] = df.loc[i,'location'][1:-1]
  df[['X','Y']] = df.location.str.split(",",expand=True)
  #converting splitted coordinates to float
  for column in ('X', 'Y'):
    for i in range(len(df)):
     df.loc[i, column] = float(df.loc[i, column])
  df = df.replace(0, np.nan)
  return df
def plot_eventhm(df, team_name, event):
  df = df_forhm(df, team_name, event)

  #fig and pitch parameters
  plt.rcParams["figure.figsize"] = (20, 15)
  pitch = Pitch(pitch_color='black', line_color='white')
  fig, axs = plt.subplots(nrows=1, ncols=1)
  fig.patch.set_facecolor('black')
  pitch.draw(ax=axs)
  axs.set_title(event + " " + team_name + " heatmap", color="white", fontsize=23, loc="left")

  sns.kdeplot(x = df['X'], y= df['Y'], fill = True, shade_lowest=False, alpha=.5, n_levels=10, cmap = 'magma')
  return  fig
###END FUNCTIONS


df = pd.read_csv('Reign_Dash.csv')
###DASHBOARD SCHEMA
st.title("Analysis of the match Houston Dash - OL Reign")
st.markdown("""General statistics displayed, explore other statistics clicking the buttons""")

##OPTIONS
st.sidebar.header('Analysis')
analysis = ['Match analysis', 'Players analysis']
selected_analysis = st.sidebar.selectbox('select the analysis to display', analysis)

if(selected_analysis == 'Match analysis'):
    st.sidebar.header('Options')
    match_analysis = ['Xgoals', 'Passing', 'Passing - in depth', 'Pressure', 'General actions']
    selected_match_analysis = st.sidebar.selectbox('select the analysis to display', match_analysis)

    if selected_match_analysis == 'Xgoals':
        st.pyplot(xg_heatmap(df))
    elif selected_match_analysis == 'Passing':
        first_half = df[df['period'] == 1].copy()
        second_half = df[df['period'] == 2].copy()
        st.pyplot(plot_passmap(pass_network(first_half)))
        st.pyplot(plot_passmap(pass_network(second_half)))
    elif selected_match_analysis == 'Passing - in depth':
        fw = st.checkbox('forward passes')
        bw = st.checkbox('backward passes')
        r = st.checkbox('to right passes')
        l = st.checkbox('to left passes')
        st.pyplot(zones_passmap(df, 'Houston Dash', fw, bw, r, l))
        x_start1, x_start2 = st.select_slider('start x', options=list(range(121)), value=(0, 60))
        x_end1, x_end2 = st.select_slider('end x', options=list(range(121)), value=(60, 120))
        y_start1, y_start2 = st.select_slider('start y', options=list(range(81)), value=(0, 80))
        y_end1, y_end2 = st.select_slider('end y', options=list(range(81)), value=(0, 80))
        st.pyplot(passmap_byzone(df, 'Houston Dash', [x_start1, x_start2], [y_start1, y_start2], [x_end1, x_end2], [y_end1, y_end2]))
    elif selected_match_analysis == 'Pressure':
        st.pyplot(plot_eventhm(df, "Houston Dash", "Pressure"))


else:
    st.sidebar.header('Options')
    player_analysis = ['Single player', 'Player Comparison']
    selected_player_analysis = st.sidebar.selectbox('select the analysis to display', player_analysis)