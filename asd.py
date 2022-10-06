import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data.csv', sep=';')
df = df.dropna().drop('Имя', axis=1).rename(columns={'Пол': 'sex', 'Высшая школа': 'high_school', 'Округ': 'region', 'Спорт': 'sport', 'Цвет глаз': 'eye_color', 'Во сколько встаете': 'time', 'Курение': 'smoking', 'К/ч': 'tea_or_coffee'})
df = (df-df.min())/(df.max()-df.min())
df = pd.read_csv('Normal_data.csv', encoding='UTF-8', sep=';').drop(['Отметка времени'], axis=1).apply(lambda x: pd.factorize(x)[0]).rename(columns={'Отметка времени': 'time', 'Высшая школа': 'high_school', 'Округ': 'region', 'Спорт': 'sport', 'Цвет глаз': 'eye_color', 'Во сколько встаешь': 'time','Курение':'smoke', 'Чай или кофе': 'tea_or_coffee'})
print(df)