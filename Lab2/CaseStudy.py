import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import numpy as np

# Case Study: Predicting Bicycle Traffic (Optional part)
import pandas as pd # Importing pandas to read the data files
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
    days = (date - pd.to_datetime(2000, 12, 21)).days
    m = (1. - np.tan(np.radians(latitude))
         * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.

daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index)) # Adding a column for hours of daylight
daily[['daylight_hrs']].plot()
plt.title('Hours of Daylight')
plt.ylim(8, 17)
plt.show()