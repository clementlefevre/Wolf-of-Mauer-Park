# coding: utf-8
import pandas as pd
df = pd.read_csv('data/merged/raw.csv',sep=';')
df.head()
df.columns
df.corr()
