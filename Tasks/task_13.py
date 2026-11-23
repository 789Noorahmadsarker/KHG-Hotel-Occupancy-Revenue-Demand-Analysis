# %% [markdown]
# # Task 13: What Drives Guest Satisfaction?
#
# **Problem statement:** Guests rate their stay from 1 to 5. Average ratings differ between cities,
# but what really explains why one guest is happier than another: the city, the hotel, the room
# class, the booking channel, how early they booked, how long they stayed, or the size of their
# party? Knowing which factors matter tells the chain where service improvements will pay off.
#
# **Plain-English terms**
# - **Happy / unhappy guests:** ratings of 4 to 5 are "happy", ratings of 1 to 2 are "unhappy".
# - **Net Satisfaction Score (NSS):** % happy minus % unhappy, from -100 to +100.
# - **Explained share:** how much of the guest-to-guest variation in ratings a factor accounts for.
#   0% means the factor has no influence, 100% means it explains everything.
#
# Only guests who stayed and chose to leave a rating are included (about 60% of stays).
#
# **Run Task 2 first.**

# %%
# Tools: pandas works with tables, numpy does the statistics, matplotlib and seaborn draw charts.
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
# ## Step 1: Load the data and prepare guest segments

# %%
bookings = pd.read_csv(
    CLEAN_DIR / "clean_bookings.csv", parse_dates=["booking_date", "check_in_date", "checkout_date"]
)
hotels = pd.read_csv(DATA_DIR / "dim_hotels.csv")
rooms = pd.read_csv(DATA_DIR / "dim_rooms.csv")
data = bookings.merge(hotels, on="property_id", how="left", validate="many_to_one")
data = data.merge(rooms, left_on="room_category", right_on="room_id", how="left",
                  validate="many_to_one")

# Guest segments (same groupings as Tasks 9 and 10).
data["lead_days"] = (data["check_in_date"] - data["booking_date"]).dt.days
data["nights"] = (data["checkout_date"] - data["check_in_date"]).dt.days
data["lead_group"] = pd.cut(data["lead_days"], [-1, 0, 3, 7, np.inf],
                            labels=["Same day", "1-3 days", "4-7 days", "20-24 days"])
data["stay_group"] = pd.cut(data["nights"], [0, 1, 2, 4, 6],
                            labels=["1 night", "2 nights", "3-4 nights", "5-6 nights"])
data["party_group"] = pd.cut(data["no_guests"], [0, 1, 2, 4, 6],
                             labels=["Solo", "Couple", "3-4 guests", "5-6 guests"])
data["day_type"] = np.where(data["check_in_date"].dt.dayofweek >= 5, "Weekend", "Weekday")

# Only guests who stayed can rate, and only some of them do.
stayed = data[data["booking_status"] == "Checked Out"]
rated = data.dropna(subset=["ratings_given"]).copy()
rated["ratings_given"] = rated["ratings_given"].astype(int)
print(f"Guests who stayed: {len(stayed):,} | left a rating: {len(rated):,} "
      f"({len(rated) / len(stayed) * 100:.1f}%)")

# Do guests in some cities answer more often? (A big difference would bias the averages.)
response = stayed.groupby("city")["ratings_given"].apply(lambda s: s.notna().mean() * 100)
print("\nShare of guests who left a rating, by city (%):")
print(response.round(1).to_string())

# %% [markdown]
# ## Step 2: Satisfaction scorecards

# %%
def satisfaction(by):
    """Average rating, % happy, % unhappy and Net Satisfaction Score for each group."""
    ratings = rated.groupby(by, observed=True)["ratings_given"]
    table = pd.DataFrame({
        "ratings": ratings.size(),
        "avg_rating": ratings.mean(),
        "pct_happy": ratings.apply(lambda s: (s >= 4).mean() * 100),
        "pct_unhappy": ratings.apply(lambda s: (s <= 2).mean() * 100),
    })
    table["net_satisfaction"] = table["pct_happy"] - table["pct_unhappy"]
    return table


overall = satisfaction(np.full(len(rated), "All guests")).round(2)
print(overall.to_string())

by_city = satisfaction("city").sort_values("net_satisfaction", ascending=False)
print("\nBy city")
print(by_city.round(2))
print("\nBy room class")
print(satisfaction("room_class").sort_values("net_satisfaction", ascending=False).round(2))
print("\nBy hotel category")
print(satisfaction("category").round(2))

# %% [markdown]
# ## Step 3: Which factors really explain satisfaction?
#
# For each factor we measure the **explained share** (statisticians call it eta-squared): how much
# of the total variation in individual ratings is accounted for by the differences between the
# factor's groups.

# %%
def explained_share(column):
    """Share (%) of the variation in ratings that the groups of this column explain."""
    overall_mean = rated["ratings_given"].mean()
    groups = rated.groupby(column, observed=True)["ratings_given"].agg(["size", "mean"])
    between = (groups["size"] * (groups["mean"] - overall_mean) ** 2).sum()
    total = ((rated["ratings_given"] - overall_mean) ** 2).sum()
    return between / total * 100


factors = {
    "Hotel (property ID)": "property_id", "City": "city", "Room class": "room_class",
    "Hotel category": "category", "Booking channel": "booking_platform",
    "How early booked": "lead_group", "Length of stay": "stay_group",
    "Party size": "party_group", "Arrival day": "day_type",
}
drivers = pd.Series({name: explained_share(col) for name, col in factors.items()})
drivers = drivers.sort_values(ascending=False)
print("Share of rating variation explained (%)")
print(drivers.round(3).to_string())

# Straight-line relationships with the rating (0 = none, +1 / -1 = perfect).
numeric = rated[["ratings_given", "lead_days", "nights", "no_guests", "revenue_generated"]]
print("\nCorrelation of each number with the rating:")
print(numeric.corr()["ratings_given"].drop("ratings_given").round(3).to_string())

# %% [markdown]
# ## Step 4: City and room class together, and hotel by hotel

# %%
# Average rating for every city and room class, and how many ratings sit behind each cell.
cell_mean = rated.pivot_table(index="city", columns="room_class", values="ratings_given", aggfunc="mean")
cell_count = rated.pivot_table(index="city", columns="room_class", values="ratings_given", aggfunc="count")
ROOM_ORDER = ["Standard", "Elite", "Premium", "Presidential"]
cell_mean = cell_mean.loc[by_city.index, ROOM_ORDER]
print(cell_mean.round(2))
print(f"\nSmallest cell has {int(cell_count.to_numpy().min()):,} ratings")

# %%
# Each hotel's average rating with a 95% margin of error (wider when there are fewer ratings).
hotel_stats = rated.groupby(["property_id", "city"])["ratings_given"].agg(["mean", "std", "size"])
hotel_stats["margin"] = 1.96 * hotel_stats["std"] / np.sqrt(hotel_stats["size"])
hotel_stats = hotel_stats.sort_values("mean", ascending=False).reset_index()
print(hotel_stats.rename(columns={"mean": "avg_rating", "size": "ratings"}).round(2).head(5).to_string(index=False))
print("...")
print(hotel_stats.rename(columns={"mean": "avg_rating", "size": "ratings"}).round(2).tail(5).to_string(index=False))

# %% [markdown]
# ## Step 5: Charts

# %%
# Chart 1: how ratings are spread in each city (left) and what explains satisfaction (right).
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})

STAR_COLOURS = {1: "#e34948", 2: "#f0a3a2", 3: "#d9d8d3", 4: "#86b6ef", 5: "#2a78d6"}
share = pd.crosstab(rated["city"], rated["ratings_given"], normalize="index").loc[by_city.index] * 100
extent = 0
for y, city in enumerate(by_city.index):
    row = share.loc[city]
    start = -(row[1] + row[2] + row[3] / 2)            # the 3-star group is centred on zero
    extent = max(extent, -start, start + row.sum())
    for star in range(1, 6):
        ax1.barh(y, row[star], left=start, color=STAR_COLOURS[star], edgecolor="#fcfcfb",
                 linewidth=2, height=0.62)
        if row[star] >= 6:
            ax1.text(start + row[star] / 2, y, f"{row[star]:.0f}%", ha="center", va="center", fontsize=9,
                     color="white" if star in (1, 5) else INK)
        start += row[star]
limit = int(np.ceil(extent / 20) * 20)
ax1.axvline(0, color=SOFT, linewidth=1, zorder=0)       # drawn behind the bars so it never cuts through a label
ax1.set_yticks(range(len(by_city)), by_city.index)
ax1.invert_yaxis()
ax1.set_xlim(-limit, limit)
ax1.set_xticks(np.arange(-limit, limit + 1, 20), [str(abs(t)) for t in np.arange(-limit, limit + 1, 20)])
ax1.set_xlabel("% of ratings   (left: unhappy, right: happy)")
ax1.set_title("How guests rate each city", pad=28)
ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=STAR_COLOURS[s], label=f"{s} star") for s in range(1, 6)],
           frameon=False, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9,
           handlelength=1, columnspacing=1)

driver_chart = drivers.reset_index()
driver_chart.columns = ["factor", "explained"]
driver_chart["top"] = np.where(driver_chart.index < 1, "Largest", "Other")

# The chart title below states the main finding, so make the code stop loudly if the data ever
# disagrees with it (instead of silently printing a wrong headline).
assert drivers.index[0] == "Hotel (property ID)"
sns.barplot(data=driver_chart, y="factor", x="explained", hue="top", dodge=False, saturation=1,
            palette={"Largest": BLUE, "Other": GRAY}, legend=False, ax=ax2)
for bar in [p for p in ax2.patches if p.get_width() > 0]:
    ax2.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.2f}%",
             va="center", fontsize=9.5)
ax2.set_xlim(0, drivers.max() * 1.25)
ax2.set_xlabel("Share of rating variation explained (%)")
ax2.set_ylabel("")
ax2.set_title("What explains satisfaction? The hotel itself", pad=28)
fig.tight_layout()
fig.savefig(OUT_DIR / "task13_satisfaction_overview.png", dpi=150)
plt.show()

# %%
# Chart 2: average rating by city and room class (left) and hotel by hotel (right).
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.15]})
sns.heatmap(cell_mean, annot=True, fmt=".2f", cmap=HEAT, vmin=3.3, vmax=4.0, linewidths=2,
            linecolor="#fcfcfb", cbar_kws={"label": "Average rating"}, ax=ax1)
ax1.set_title("Average rating by city and room class", pad=14)
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.tick_params(axis="y", rotation=0)

plot_stats = hotel_stats.copy()
plot_stats["colour"] = GRAY
plot_stats.loc[plot_stats.index[:3], "colour"] = BLUE
plot_stats.loc[plot_stats.index[-3:], "colour"] = ORANGE
for y, row in plot_stats.iterrows():
    ax2.errorbar(row["mean"], y, xerr=row["margin"], fmt="o", color=row["colour"], ecolor=row["colour"],
                 elinewidth=1.4, capsize=3, markersize=6)
ax2.axvline(rated["ratings_given"].mean(), color=SOFT, linestyle=":", linewidth=1.2)
ax2.set_yticks(range(len(plot_stats)), [f"{r.property_id}  {r.city}" for r in plot_stats.itertuples()])
ax2.invert_yaxis()
ax2.set_xlabel("Average rating (with 95% margin of error)")
ax2.set_title("Hotel ratings: best three in blue, lowest three in orange", pad=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "task13_rating_heatmap_and_hotels.png", dpi=150)
plt.show()
