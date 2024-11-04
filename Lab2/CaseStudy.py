# Case Study: Predicting Bicycle Traffic (Optional part)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd # Importing pandas to read the data files
import datetime as dt # Importing datetime to work with dates (pd.to_datetime didn't work like pd.datetime, so I used dt.datetime)

counts = pd.read_csv('FremontBridge.csv', index_col='Date', parse_dates=True)
weather = pd.read_csv('BicycleWeather.csv', index_col='DATE', parse_dates=True)

daily = counts.resample('d').sum() # Resampling the data to daily
daily['Total'] = daily.sum(axis=1) # Adding a column for total daily bicycle traffic
daily = daily[['Total']] # Removing other columns

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] # Days of the week
for i in range(7):
    daily[days[i]] = (daily.index.dayofweek == i).astype(float) # Adding columns for days of the week

from pandas.tseries.holiday import USFederalHolidayCalendar # Importing US Federal Holidays
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012', '2016') # Holidays between 2012 and 2016
daily = daily.join(pd.Series(1, index=holidays, name='holiday')) # Adding a column for holidays
daily['holiday'].fillna(0, inplace=True) # Filling NaN values with 0

def hours_of_daylight(date, axis=23.44, latitude=47.61): # Function to calculate hours of daylight
    days = (date - dt.datetime(2000, 12, 21)).days
    m = (1. - np.tan(np.radians(latitude))
         * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.

daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index)) # Adding a column for hours of daylight
daily[['daylight_hrs']].plot()
plt.title('Hours of Daylight')
plt.ylim(8, 17)
plt.show()

# Adding a column for average temperature
weather['TMIN'] /= 10
weather['TMAX'] /= 10
weather['Temp (C)'] = 0.5 * (weather['TMIN'] + weather['TMAX'])
# precip is in 1/10 mm; convert to inches
weather['PRCP'] /= 254
weather['dry day'] = (weather['PRCP'] == 0).astype(int)
daily = daily.join(weather[['PRCP', 'Temp (C)', 'dry day']]) # Adding columns for precipitation, temperature, and dry day

daily['annual'] = (daily.index - daily.index[0]).days / 365. # Adding a column for years since the start of the data
print(daily.head())

# Drop any rows with null values
daily.dropna(axis=0, how='any', inplace=True)
column_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'holiday', 'daylight_hrs', 'PRCP', 'Temp (C)', 'dry day', 'annual']
X = daily[column_names]
y = daily['Total']
model = LinearRegression(fit_intercept=False)
model.fit(X, y)
daily['predicted'] = model.predict(X)

daily[['Total', 'predicted']].plot(alpha=0.5)
plt.title('Linear Regression')
plt.show()

params = pd.Series(model.coef_, index=X.columns)
print(params)

from sklearn.utils import resample
np.random.seed(1)
err = np.std([model.fit(*resample(X, y)).coef_ for i in range(1000)], 0) 

print(pd.DataFrame({'effect': params.round(0), 'error': err.round(0)}))