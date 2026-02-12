import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import urllib.request
import ssl
import re

# FIX FOR MAC SSL ERROR
ssl._create_default_https_context = ssl._create_unverified_context

# 1. Download
url = "https://max.ge/aiml_final/b_babalashvili25_34521_server.log"
log_file = "server.log"
print("Downloading log...")
urllib.request.urlretrieve(url, log_file)

# 2. Extract timestamps using a more flexible Regex
# This looks for patterns like 12/Feb/2026:14:30:00 or 2026-02-12 14:30:00
timestamps = []
date_pattern = re.compile(r'(\d{1,4}[/-][A-Za-z0-9]{2,3}[/-]\d{2,4}[:\s]\d{2}:\d{2}:\d{2})')

with open(log_file, 'r') as f:
    lines = f.readlines()
    print(f"Total lines in file: {len(lines)}")
    for line in lines:
        match = date_pattern.search(line)
        if match:
            timestamps.append(match.group(1))

# 3. Process Data
df = pd.DataFrame(timestamps, columns=['time'])

# Try to convert whatever we found into a date
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.dropna().sort_values('time')

if df.empty:
    print("--- DEBUG: FAILED TO FIND DATES ---")
    print("Here are the first 3 lines of your log file. Please check the format:")
    with open(log_file, 'r') as f:
        for i in range(3): print(f.readline().strip())
else:
    print(f"Successfully found {len(df)} timestamps!")
    df['count'] = 1
    df.set_index('time', inplace=True)

    # 4. Aggregation & Regression
    data_per_minute = df['count'].resample('1min').count().reset_index()
    data_per_minute.columns = ['time', 'requests']

    X = np.array(data_per_minute.index).reshape(-1, 1)
    y = data_per_minute['requests'].values

    model = LinearRegression().fit(X, y)
    trend = model.predict(X)

    # 5. Detection (3x trend)
    ddos_events = data_per_minute[data_per_minute['requests'] > (trend * 3)]

    print("\n--- RESULTS FOR YOUR DDOS.MD ---")
    if not ddos_events.empty:
        print(ddos_events.to_string(index=False))
    else:
        print("No DDoS spikes detected above 3x the trend.")

    # 6. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(data_per_minute['time'], y, label='Actual Requests', color='blue', alpha=0.6)
    plt.plot(data_per_minute['time'], trend, color='red', ls='--', label='Regression Trend')
    plt.scatter(ddos_events['time'], ddos_events['requests'], color='orange', label='DDoS Spikes', zorder=5)
    plt.title("Task 3: Web Server Traffic Analysis")
    plt.xlabel("Time")
    plt.ylabel("Requests per Minute")
    plt.legend()
    plt.savefig('ddos_analysis.png')
    plt.show()