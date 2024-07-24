import pandas as pd
import requests
import time

# Load the CSV file
dataset = pd.read_csv("/Users/anav_sobti/Downloads/IndianWeatherRepository_cleaned_data.csv")

data = dataset.head(2000)
# Function to get altitude using Open-Meteo Elevation API
def get_altitude(lat, lon):
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json()
        return results['elevation']
    else:
        return None

# Apply the function to each row in the dataframe to get the altitude
for index, row in data.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    altitude = get_altitude(lat, lon)
    data.at[index, 'altitude'] = altitude
    # Adding a delay to avoid hitting the API rate limit
    time.sleep(0.1)

# Save the updated dataframe to a new CSV file
data.to_csv('/Users/anav_sobti/Downloads/IndianWeatherRepository_API_integrated_8000+8000.csv', index=False)

print("Altitude data fetched and saved successfully.")