# %% [markdown]
# # Task 12: Demand Seasonality and "Soft Weeks"
#
# **Problem statement:** Hotels plan staff, prices and promotions around predictable ups and
# downs in demand. How does occupancy change across the days of the week, and across the weeks of
# the season? Are there unusually weak weeks, and if so, do they hurt every city and hotel or only
# some?
#
# **Plain-English terms**
# - **Seasonality index:** a day's average occupancy divided by the overall average, times 100.
#   100 means average, 130 means 30% busier than average.
# - **Soft week:** a full calendar week whose occupancy is more than 5 points below the median week.
#
# This task uses "booked occupancy" (all bookings / rooms available), as in Tasks 3 to 5.
# The calendar table counts weeks from Sunday to Saturday, and this task follows it.
#
# **Run Task 1 first.**

# %%
# Tools: pandas works with tables, numpy does the maths, matplotlib and seaborn draw charts.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ---- Folder settings (usually nothing to change) ----
# These lines tell Python where your project folders are. Each name is a folder inside the
# main project folder, and the code must be run from that main folder.
# You only need to edit a name if you renamed the folder in your own copy of the project.
# (In Google Colab, use the notebooks: their first cell connects to your Google Drive folder for you.)
DATA_DIR = Path("data")             # folder holding the 6 ORIGINAL csv files (never changed)
CLEAN_DIR = Path("cleaned_data")    # folder for the cleaned files that the tasks create and reuse
OUT_DIR = Path("outputs")           # folder where the charts (PNG pictures) are saved
OUT_DIR.mkdir(exist_ok=True)        # create the outputs folder if it does not exist yet

# Chart look: one clean, colour-blind-friendly style used by every chart in this project.
INK, SOFT, GRID = "#0b0b0b", "#52514e", "#e3e2de"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a8a7a1"
HEAT = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
sns.set_theme(style="white", rc={
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
    "axes.labelcolor": SOFT, "text.color": INK, "xtick.color": SOFT, "ytick.color": SOFT,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left"})

# %% [markdown]
# ## Step 1: Build a daily demand table for the whole chain

# %%
agg = pd.read_csv(CLEAN_DIR / "clean_aggregated_bookings.csv", parse_dates=["check_in_date"])
hotels = pd.read_csv(DATA_DIR / "dim_hotels.csv")
calendar = pd.read_csv(DATA_DIR / "dim_date.csv")

# Calendar dates are text such as "01-May-22": convert them to real dates so they match.
calendar["date"] = pd.to_datetime(calendar["date"], format="%d-%b-%y")
calendar["day_type"] = calendar["day_type"].replace({"weekeday": "weekday"})   # fix the typo

data = agg.merge(hotels, on="property_id", how="left", validate="many_to_one")
data = data.merge(calendar[["date", "week no", "day_type"]], left_on="check_in_date",
                  right_on="date", how="left", validate="many_to_one").drop(columns="date")
assert data["week no"].notna().all()            # every day must be found in the calendar
data["weekday"] = data["check_in_date"].dt.day_name()

# One row per day: rooms booked and rooms available across all hotels.
daily = data.groupby(["check_in_date", "week no", "weekday", "day_type"], as_index=False).agg(
    booked=("successful_bookings", "sum"), rooms=("capacity", "sum")
)
daily["occ_pct"] = daily["booked"] / daily["rooms"] * 100

# Safety check: the calendar's "weekend" is exactly Saturday and Sunday.
assert set(daily.loc[daily["day_type"] == "weekend", "weekday"]) == {"Saturday", "Sunday"}
print("Days:", len(daily), "| Average booked occupancy:", round(daily["occ_pct"].mean(), 1), "%")

# %% [markdown]
# ## Step 2: Day-of-week pattern

# %%
ORDER = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
profile = daily.groupby("weekday")["occ_pct"].mean().reindex(ORDER)
season_index = profile / daily["occ_pct"].mean() * 100
print(pd.DataFrame({"avg_occupancy_pct": profile.round(1), "seasonality_index": season_index.round(0)}))

# %% [markdown]
# ## Step 3: Week-by-week view and soft weeks

# %%
# One row per calendar week (Sunday to Saturday). The last week holds only Sunday 31 July.
weekly = daily.groupby("week no").agg(
    start=("check_in_date", "min"), days=("check_in_date", "size"),
    booked=("booked", "sum"), rooms=("rooms", "sum"),
)
weekly["occ_pct"] = weekly["booked"] / weekly["rooms"] * 100
full = weekly[weekly["days"] == 7].copy()             # only complete weeks are judged

median_week = full["occ_pct"].median()
full["week_type"] = np.where(full["occ_pct"] < median_week - 5, "Soft week", "Normal week")
print(f"Median week: {median_week:.1f}%  | Soft weeks: {(full['week_type'] == 'Soft week').sum()} of {len(full)}")
print(full[["start", "occ_pct", "week_type"]].round({"occ_pct": 1}).to_string())

# %%
# How big is the dip, in occupancy points and in rooms?
level = full.groupby("week_type")["occ_pct"].mean()
dip_points = level["Normal week"] - level["Soft week"]
rooms_per_day = daily["rooms"].median()
print(f"Normal week: {level['Normal week']:.1f}% | Soft week: {level['Soft week']:.1f}% | dip: {dip_points:.1f} points")
print(f"Roughly {dip_points / 100 * rooms_per_day * 7:,.0f} fewer rooms booked in a soft week")

# %%
# Do soft weeks hurt every city? Compare each city's occupancy in normal and soft weeks.
data["week_type"] = data["week no"].map(full["week_type"])
city_view = data.dropna(subset=["week_type"]).groupby(["city", "week_type"]).agg(
    booked=("successful_bookings", "sum"), rooms=("capacity", "sum"))
city_view["occ_pct"] = city_view["booked"] / city_view["rooms"] * 100
city_table = city_view["occ_pct"].unstack()
city_table["dip_points"] = city_table["Normal week"] - city_table["Soft week"]
print(city_table.round(1))

hotel_view = data.dropna(subset=["week_type"]).groupby(["property_id", "week_type"]).agg(
    booked=("successful_bookings", "sum"), rooms=("capacity", "sum"))
hotel_dip = (hotel_view["booked"] / hotel_view["rooms"] * 100).unstack()
hotel_dip["dip_points"] = hotel_dip["Normal week"] - hotel_dip["Soft week"]
print(f"\nHotels hit by soft weeks: {int((hotel_dip['dip_points'] > 0).sum())} of {len(hotel_dip)} "
      f"(dip ranges from {hotel_dip['dip_points'].min():.1f} to {hotel_dip['dip_points'].max():.1f} points)")

# %% [markdown]
# ## Step 4: Peak days and high-demand days

# %%
cols = ["check_in_date", "weekday", "week no", "occ_pct"]
print("Five busiest days")
print(daily.nlargest(5, "occ_pct")[cols].round({"occ_pct": 1}).to_string(index=False))
print("\nFive quietest days")
print(daily.nsmallest(5, "occ_pct")[cols].round({"occ_pct": 1}).to_string(index=False))

# A "high-demand day" is a day with at least 75% of rooms booked chain-wide.
high = daily[daily["occ_pct"] >= 75]
print(f"\nHigh-demand days (75% or more): {len(high)} of {len(daily)}")
print(high["weekday"].value_counts().to_string())

# %% [markdown]
# ## Step 5: Charts

# %%
# Chart 1: day-of-week pattern (left) and every day of the season (right).
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
colours = [BLUE if day in ("Saturday", "Sunday") else GRAY for day in ORDER]
sns.barplot(x=[d[:3] for d in ORDER], y=season_index.to_numpy(), hue=[d[:3] for d in ORDER],
            palette=dict(zip([d[:3] for d in ORDER], colours)), legend=False, saturation=1, ax=ax1)
for bar, value in zip(ax1.patches, season_index):
    ax1.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.0f}", ha="center", fontsize=9.5)
ax1.axhline(100, color=SOFT, linestyle=":", linewidth=1.2)
ax1.set_ylabel("Seasonality index (100 = average day)")
weekend_lift = season_index[["Saturday", "Sunday"]].mean() - 100
ax1.set_title(f"Weekends are about {weekend_lift:.0f}% busier than an average day")

labels = [f"{w}  ({s:%d %b})" for w, s in zip(weekly.index, weekly["start"])]
grid = daily.pivot(index="week no", columns="weekday", values="occ_pct")[ORDER]
grid.index = labels
sns.heatmap(grid, annot=True, fmt=".0f", cmap=HEAT, vmin=40, vmax=82, linewidths=1.5,
            linecolor="#fcfcfb", cbar_kws={"label": "Booked occupancy (%)"}, annot_kws={"size": 8.5}, ax=ax2)
ax2.set_xticklabels([d[:3] for d in ORDER], rotation=0)
ax2.tick_params(axis="y", rotation=0)
ax2.set_ylabel("")
ax2.set_xlabel("")
ax2.set_title("Every day of the season: darker rows are stronger weeks")
fig.tight_layout()
fig.savefig(OUT_DIR / "task12_weekday_pattern_and_heatmap.png", dpi=150)
plt.show()

# %%
# Chart 2: average occupancy of each full week. Soft weeks are orange.
fig, ax = plt.subplots(figsize=(11, 4.2))
bar_colours = [ORANGE if t == "Soft week" else BLUE for t in full["week_type"]]
bars = ax.bar(full.index, full["occ_pct"], color=bar_colours, width=0.68)
ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=9.5)
ax.axhline(median_week, color=SOFT, linestyle=":", linewidth=1.2)
ax.set_ylim(0, 75)
ax.set_ylabel("Booked occupancy (%)")
ax.set_title(f"{(full['week_type'] == 'Soft week').sum()} of {len(full)} weeks were soft, about "
             f"{dip_points:.0f} points below a normal week")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE, label="Normal week"),
                   plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="Soft week")],
          frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.yaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "task12_weekly_demand.png", dpi=150)
plt.show()
