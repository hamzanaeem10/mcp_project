import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 52.52,
	"longitude": 13.41,
	"daily": ["sunrise", "temperature_2m_max", "wind_speed_10m_max", "shortwave_radiation_sum", "sunset", "daylight_duration", "sunshine_duration", "weather_code", "rain_sum", "temperature_2m_mean"],
	"hourly": ["temperature_2m", "temperature_80m", "temperature_180m", "shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", "shortwave_radiation_instant", "direct_radiation_instant", "global_tilted_irradiance_instant", "direct_normal_irradiance_instant", "diffuse_radiation_instant", "terrestrial_radiation_instant"],
	"current": ["is_day", "rain"],
	"past_days": 92,
	"forecast_days": 14,
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process current data. The order of variables needs to be the same as requested.
current = response.Current()
current_is_day = current.Variables(0).Value()
current_rain = current.Variables(1).Value()

print(f"\nCurrent time: {current.Time()}")
print(f"Current is_day: {current_is_day}")
print(f"Current rain: {current_rain}")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_temperature_80m = hourly.Variables(1).ValuesAsNumpy()
hourly_temperature_180m = hourly.Variables(2).ValuesAsNumpy()
hourly_shortwave_radiation = hourly.Variables(3).ValuesAsNumpy()
hourly_direct_radiation = hourly.Variables(4).ValuesAsNumpy()
hourly_diffuse_radiation = hourly.Variables(5).ValuesAsNumpy()
hourly_direct_normal_irradiance = hourly.Variables(6).ValuesAsNumpy()
hourly_shortwave_radiation_instant = hourly.Variables(7).ValuesAsNumpy()
hourly_direct_radiation_instant = hourly.Variables(8).ValuesAsNumpy()
hourly_global_tilted_irradiance_instant = hourly.Variables(9).ValuesAsNumpy()
hourly_direct_normal_irradiance_instant = hourly.Variables(10).ValuesAsNumpy()
hourly_diffuse_radiation_instant = hourly.Variables(11).ValuesAsNumpy()
hourly_terrestrial_radiation_instant = hourly.Variables(12).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["temperature_80m"] = hourly_temperature_80m
hourly_data["temperature_180m"] = hourly_temperature_180m
hourly_data["shortwave_radiation"] = hourly_shortwave_radiation
hourly_data["direct_radiation"] = hourly_direct_radiation
hourly_data["diffuse_radiation"] = hourly_diffuse_radiation
hourly_data["direct_normal_irradiance"] = hourly_direct_normal_irradiance
hourly_data["shortwave_radiation_instant"] = hourly_shortwave_radiation_instant
hourly_data["direct_radiation_instant"] = hourly_direct_radiation_instant
hourly_data["global_tilted_irradiance_instant"] = hourly_global_tilted_irradiance_instant
hourly_data["direct_normal_irradiance_instant"] = hourly_direct_normal_irradiance_instant
hourly_data["diffuse_radiation_instant"] = hourly_diffuse_radiation_instant
hourly_data["terrestrial_radiation_instant"] = hourly_terrestrial_radiation_instant

hourly_dataframe = pd.DataFrame(data = hourly_data)
print("\nHourly data\n", hourly_dataframe)

# Process daily data. The order of variables needs to be the same as requested.
daily = response.Daily()
daily_sunrise = daily.Variables(0).ValuesInt64AsNumpy()
daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
daily_wind_speed_10m_max = daily.Variables(2).ValuesAsNumpy()
daily_shortwave_radiation_sum = daily.Variables(3).ValuesAsNumpy()
daily_sunset = daily.Variables(4).ValuesInt64AsNumpy()
daily_daylight_duration = daily.Variables(5).ValuesAsNumpy()
daily_sunshine_duration = daily.Variables(6).ValuesAsNumpy()
daily_weather_code = daily.Variables(7).ValuesAsNumpy()
daily_rain_sum = daily.Variables(8).ValuesAsNumpy()
daily_temperature_2m_mean = daily.Variables(9).ValuesAsNumpy()

daily_data = {"date": pd.date_range(
	start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
	end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = daily.Interval()),
	inclusive = "left"
)}

daily_data["sunrise"] = daily_sunrise
daily_data["temperature_2m_max"] = daily_temperature_2m_max
daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
daily_data["sunset"] = daily_sunset
daily_data["daylight_duration"] = daily_daylight_duration
daily_data["sunshine_duration"] = daily_sunshine_duration
daily_data["weather_code"] = daily_weather_code
daily_data["rain_sum"] = daily_rain_sum
daily_data["temperature_2m_mean"] = daily_temperature_2m_mean

daily_dataframe = pd.DataFrame(data = daily_data)
print("\nDaily data\n", daily_dataframe)