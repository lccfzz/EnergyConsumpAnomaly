# %%
# MSD Hometask
# Monte carlo that replicates the sthocastic behaviour of market demand

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv', delimiter=';')
print(df.head())

# Plot DEMAND/FORECAST vs MONTH for all MARKET_ID
market_1_filter = df[df['MARKET_ID'] == 1]

plt.figure(figsize=(8, 5))
plt.plot(market_1_filter["MONTH"], market_1_filter["FORECAST"], label="Forecast", linestyle="--", marker="o")
plt.plot(market_1_filter["MONTH"], market_1_filter["DEMAND"], label="Demand", marker="o")

plt.xlabel("Month")
plt.ylabel("Values")
plt.title(f"Product 1: Demand and Forecast vs Month")
plt.legend()
plt.grid(True)
plt.show()

market_2_filter = df[df['MARKET_ID'] == 2]
plt.figure(figsize=(8, 5))
plt.plot(market_2_filter["MONTH"], market_2_filter["FORECAST"], label="Forecast", linestyle="--", marker="o")
plt.plot(market_2_filter["MONTH"], market_2_filter["DEMAND"], label="Demand", marker="o")

plt.xlabel("Month")
plt.ylabel("Values")
plt.title(f"Product 2: Demand and Forecast vs Month")
plt.legend()
plt.grid(True)
plt.show()

market_3_filter = df[df['MARKET_ID'] == 3]
plt.figure(figsize=(8, 5))
plt.plot(market_3_filter["MONTH"], market_3_filter["FORECAST"], label="Forecast", linestyle="--", marker="o")
plt.plot(market_3_filter["MONTH"], market_3_filter["DEMAND"], label="Demand", marker="o")

plt.xlabel("Month")
plt.ylabel("Values")
plt.title(f"Product 3: Demand and Forecast vs Month")
plt.legend()
plt.grid(True)
plt.show()

# %%


# %%
# Step 1: fix data set - DONE

# Step 1.5: relationship between columns and other correlation?

# Step 2: what is the ideal distribution given this data?

# Step 2.5: parallization

# Step 3: Monte carlo set up

def MC_Main():
    decorrelation()
    run_sampling()
    write_output()
    diagnostics()

# Step 4: does it do a good job of replicating sthocastic behaviour of demand market?
# %%