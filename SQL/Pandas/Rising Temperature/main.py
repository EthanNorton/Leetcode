## not solved, working through edge cases 
import pandas as pd

def rising_temperature(weather: pd.DataFrame) -> pd.DataFrame:
    # Shift the temperature column to get the previous day's temperature
    weather['prev_temp'] = weather['temperature'].shift(1)
    
    # Filter rows where the current temperature is greater than the previous day's temperature
    result = weather[weather['temperature'] > weather['prev_temp']]
    
    # Return the required columns
    return result[['id']]
