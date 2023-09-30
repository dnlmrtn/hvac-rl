'''
This will get weather data for our simulation and store it in a CSV
Weather data for Toronto, ON is fetched using the OpenWeather API.
'''


import requests
import json
import time
import csv

# Replace YOUR_API_KEY with your actual API key from OpenWeatherMap
api_key = "65070adbd42189eacd304b226051e8b6"
lat = "43.6532"
lon = "-79.3832"

# Create or open a CSV file to store the data
with open("weather_data.csv", "w", newline="") as csvfile:
    fieldnames = ["date", "temperature", "humidity", "wind_speed"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()  # Write the headers

    # Loop through the required dates (For demonstration purposes, only one date is taken)
    for i in range(1):  # You would loop through 365 days
        # Unix time stamp for the date
        time_stamp = int(time.time()) - i*3600*24

        # API endpoint for hourly data
        url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={time_stamp}&appid={api_key}"

        response = requests.get(url)

        if response.status_code == 200:
            weather_data = json.loads(response.text)

            for hour in weather_data["hourly"]:
                writer.writerow({
                    "date": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(hour["dt"])),
                    "temperature": hour["temp"],
                    "humidity": hour["humidity"],
                    "wind_speed": hour["wind_speed"]
                })
        else:
            print(f"Failed to get data for date with time stamp: {time_stamp}")
            print("Status Code:", response.status_code)
            print("Response Text:", response.text)

        time.sleep(1)  # To avoid hitting rate limits
