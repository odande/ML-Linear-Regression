

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
link = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/weather2019.csv"
df_weather = pd.read_csv(link)
df_weather

# Complete x, y and z in the script below:
px.scatter_3d(x = df_weather['MAX_TEMPERATURE_C'],
              y = df_weather['MIN_TEMPERATURE_C'] ,
              z= df_weather['SUNHOUR'],
              data_frame=df_weather)

# Your code here :
X =df_weather[['MIN_TEMPERATURE_C','SUNHOUR']]
y =df_weather[['MAX_TEMPERATURE_C']]

from sklearn.linear_model import LinearRegression

model1=LinearRegression()

model1.fit(X,y)

df_weather['predict']=model1.predict(df_weather[['MIN_TEMPERATURE_C','SUNHOUR']])

# Your code here :
px.scatter_3d(x = df_weather['predict'],
              y = df_weather['MIN_TEMPERATURE_C'] ,
              z= df_weather['SUNHOUR'],
              data_frame=df_weather)

#df_weather['DATE']=pd.to_datetime(df_weather['DATE'])
X=df_weather[['MIN_TEMPERATURE_C', 'WINDSPEED_MAX_KMH',	'TEMPERATURE_MORNING_C', 'TEMPERATURE_NOON_C', 'TEMPERATURE_EVENING_C', 'PRECIP_TOTAL_DAY_MM', 'HUMIDITY_MAX_PERCENT',
              'VISIBILITY_AVG_KM', 'WINDTEMP_MAX_C', 'WEATHER_CODE_MORNING', 'WEATHER_CODE_NOON',	'WEATHER_CODE_EVENING',	'TOTAL_SNOW_MM', 'UV_INDEX', 'SUNHOUR']]
y=df_weather['MAX_TEMPERATURE_C']

from sklearn.linear_model import LinearRegression

model3=LinearRegression()
model3.fit(X,y)

df_weather['predict2']=model3.predict(df_weather[['MIN_TEMPERATURE_C', 'WINDSPEED_MAX_KMH',	'TEMPERATURE_MORNING_C', 'TEMPERATURE_NOON_C', 'TEMPERATURE_EVENING_C', 'PRECIP_TOTAL_DAY_MM', 'HUMIDITY_MAX_PERCENT',
              'VISIBILITY_AVG_KM', 'WINDTEMP_MAX_C', 'WEATHER_CODE_MORNING', 'WEATHER_CODE_NOON',	'WEATHER_CODE_EVENING',	'TOTAL_SNOW_MM', 'UV_INDEX', 'SUNHOUR']])

df_weather['DATE']=pd.to_datetime(df_weather['DATE'])

sns.scatterplot(x=df_weather['DATE'],y=df_weather['MAX_TEMPERATURE_C'])
sns.scatterplot(x=df_weather['DATE'],y=df_weather['predict2'],color='orange')


model3.coef_