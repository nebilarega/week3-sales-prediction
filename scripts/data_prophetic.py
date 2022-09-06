import sys
import os
import json
import requests
import time
import datetime as dt
from datetime import date, timedelta, datetime

from itertools import repeat
import itertools

import numpy as np
import pandas as pd
import statistics as st
import scipy.stats as ss
from random import randint

import matplotlib.pyplot as plt
import seaborn as sns

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot, plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics

class Prophetic():
    def __init__(self, dfdict, tsdf):
        self.dfdict = dfdict
        self.tsdf = tsdf
         
    def df_to_prophet(self,name='train', 
                    timecol="Date",
                    targetcol="Sales",
                    ftransform=None, 
                    prefilter={}, 
                    postfilter={},
                    rs = '1D'):
            '''
            prepare data frame to prophet modelling
                dfin - input data frame
                timecol - the column name for time/date
                prefilter - a dictionary that contains column_name:value
                postfilter - a dictionary that contains column_name:value
                ftransform - a function to apply after prefiltering takes dfin as input
                rs - unit of time to resample time column 
            '''
            if name in ['train','test','sample']:
                df = self.dfdict[name].copy()
            else:
                print("only name=['train','test','sample'] are allowed")
                return pd.DataFrame()
            
            df['ts'] =  pd.to_datetime(df[timecol]).dt.tz_localize(None)
            df['ts'] = df['ts'].dt.to_pydatetime()

            #apply pre-filter
            for k, v in prefilter.items():
                print(f'Applying pre transform filter with column={k}, value={v}')
                df = df[df[k]==v]

            #transform
            if ftransform is not None:
                print(f'Applying functional transformation ..')
                df = ftransform(df)

            #apply post filter
            for k, v in prefilter.items():
                print(f'Applying post transform filter with column={k}, value={v}')
                df = df[df[k]==v]

            df = df.reset_index().set_index('ts').resample(rs).sum() 
            df = df.reset_index()

            df = df[['ts', targetcol]]
            df = df.rename(columns={"ts": "ds", targetcol: "y"})
            df = df.dropna()
            df.ds = pd.Series([v.to_pydatetime() for v in df.ds], dtype=object)
            
            self.tsdf = df
            return self.tsdf

    def plot_prophet(self, 
                    changepoint_prior_scale=0.001, 
                    seasonality_prior_scale=1.0, 
                    periods=10, 
                    split=0.8 ):

        model = (Prophet(changepoint_prior_scale=changepoint_prior_scale, 
                        seasonality_prior_scale=seasonality_prior_scale, 
                        interval_width=0.95, 
                        daily_seasonality=True, 
                        weekly_seasonality=True, 
                        yearly_seasonality=False) \
                .add_seasonality(name='monthly', period=30.5, fourier_order=5) \
                .fit(self.tsdf)
                )
        
        future = model.make_future_dataframe(periods)
        forecast = model.predict(future)
        components = model.plot_components(forecast)

        forecast.ds = pd.Series([v.to_pydatetime() for v in forecast.ds], dtype=object)

        split = 0.8
        threshold_date_train = self.tsdf.ds[ int(len(self.tsdf.ds)*split) ]
        threshold_date_forecast = forecast.ds[ int(len(forecast.ds)*split) ]

        forecast_train = forecast[ threshold_date_forecast >= forecast.ds ]
        forecast_test = forecast[ threshold_date_forecast < forecast.ds ]
        df_train = self.tsdf[ threshold_date_train >= self.tsdf.ds ]
        df_test = self.tsdf[ threshold_date_train < self.tsdf.ds ]

        fig, ax = plt.subplots(figsize=(20,10))
        sns.set_style('darkgrid', {'axes.facecolor': '.9'})
        sns.set_palette(palette='deep')
        sns_c = sns.color_palette(palette='deep')

        ax.fill_between( x=forecast['ds'], y1=forecast['yhat_lower'], y2=forecast['yhat_upper'],
            color=sns_c[2], alpha=0.25, label=r'0.95 credible_interval')

        sns.scatterplot(x='ds', y='y', label='real historic data', data=df_train, ax=ax, color='black')
        sns.scatterplot(x='ds', y='y', label='real test data', data=df_test, ax=ax, color = 'red')
        sns.lineplot(x='ds', y='yhat', label='historic fit', data=forecast_train, ax=ax, color = 'blue')
        sns.lineplot(x='ds', y='yhat', label='future prediction', data=forecast_test, ax=ax, color = 'orange')
        ax.axvline(threshold_date_train, color=sns_c[3], linestyle='--', label='80% train-test data split')
        ax.legend(loc='upper left')
        ax.set_xlabel('Date')
        ax.set_ylabel('Engagement rate')
        ax.tick_params(axis='x', rotation=45)
        ax.set(title='Engagement rate model fit & prediction for Campaign ID ')
        
        return fig, components
        