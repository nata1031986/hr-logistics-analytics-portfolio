"""
Generates the SYNTHETIC datasets used in this portfolio.

All companies, carriers, SKUs and drivers are fictional.
City names and coordinates are real US cities; road miles are
estimated as straight-line distance x 1.2 (a common rough routing factor).

Run:  python generate_data.py
Output: data/*.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Shared: distribution centers / cities
# ---------------------------------------------------------------------------
cities = pd.DataFrame([
    # city, state, region, lat, lon
    ("Chicago", "IL", "Midwest", 41.88, -87.63),
    ("Dallas", "TX", "South", 32.78, -96.80),
    ("Atlanta", "GA", "Southeast", 33.75, -84.39),
    ("Los Angeles", "CA", "West", 34.05, -118.24),
    ("Newark", "NJ", "Northeast", 40.74, -74.17),
    ("Columbus", "OH", "Midwest", 39.96, -83.00),
    ("Memphis", "TN", "Southeast", 35.15, -90.05),
    ("Denver", "CO", "West", 39.74, -104.99),
    ("Phoenix", "AZ", "West", 33.45, -112.07),
    ("Seattle", "WA", "West", 47.61, -122.33),
    ("Minneapolis", "MN", "Midwest", 44.98, -93.27),
    ("Boston", "MA", "Northeast", 42.36, -71.06),
    ("Houston", "TX", "South", 29.76, -95.37),
    ("Charlotte", "NC", "Southeast", 35.23, -80.84),
    ("Kansas City", "MO", "Midwest", 39.10, -94.58),
], columns=["city", "state", "region", "lat", "lon"])

def road_miles(a, b):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [a.lat, a.lon, b.lat, b.lon])
    d = 2 * R * np.arcsin(np.sqrt(np.sin((lat2 - lat1) / 2) ** 2 +
                                  np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2))
    return round(d * 1.2)

# ---------------------------------------------------------------------------
# PROJECT 1: shipments + carriers (calendar year 2025)
# ---------------------------------------------------------------------------
carriers = pd.DataFrame([
    # id, name (fictional), base on-time skill, cost factor, damage rate
    ("C01", "Prairie Line Freight", 0.93, 1.00, 0.004),
    ("C02", "Bluewater Transport",  0.91, 0.96, 0.006),
    ("C03", "Summit Haulers",       0.84, 0.88, 0.011),
    ("C04", "Redrock Logistics",    0.95, 1.12, 0.003),
    ("C05", "Ironbridge Carriers",  0.89, 0.93, 0.007),
    ("C06", "Gulfstar Trucking",    0.92, 0.98, 0.005),
    ("C07", "Northpoint Express",   0.87, 1.05, 0.006),
    ("C08", "Eagle Road Inc.",      0.90, 0.94, 0.008),
], columns=["carrier_id", "carrier_name", "_skill", "_cost", "_damage"])

N_SHIP = 12000
dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, N_SHIP), unit="D")
# more volume in Q4 (peak season)
peak_extra = pd.to_datetime("2025-10-15") + pd.to_timedelta(rng.integers(0, 70, 1800), unit="D")
dates = pd.DatetimeIndex(np.concatenate([dates.values, peak_extra.values]))
n = len(dates)

o_idx = rng.integers(0, len(cities), n)
d_idx = rng.integers(0, len(cities), n)
same = o_idx == d_idx
d_idx[same] = (d_idx[same] + rng.integers(1, len(cities) - 1, same.sum())) % len(cities)

car_idx = rng.choice(len(carriers), n, p=[.16, .15, .14, .09, .13, .12, .11, .10])
mode = rng.choice(["FTL", "LTL"], n, p=[0.62, 0.38])

rows = []
for i in range(n):
    o, d = cities.iloc[o_idx[i]], cities.iloc[d_idx[i]]
    c = carriers.iloc[car_idx[i]]
    miles = road_miles(o, d)
    # planned transit: ~ 500 road miles per day for FTL, slower for LTL
    per_day = 500 if mode[i] == "FTL" else 380
    planned = max(1, int(np.ceil(miles / per_day)) + (1 if mode[i] == "LTL" else 0))

    p_on_time = c._skill
    month = dates[i].month
    if month in (12, 1, 2) and (o.region in ("Midwest", "Northeast") or d.region in ("Midwest", "Northeast")):
        p_on_time -= 0.12          # winter weather
    if month in (11, 12):
        p_on_time -= 0.05          # peak season congestion
    if c.carrier_id == "C03" and d.region == "Northeast":
        p_on_time -= 0.15          # weak lane for Summit Haulers
    if mode[i] == "LTL":
        p_on_time -= 0.03
    on_time = rng.random() < p_on_time
    delay = 0 if on_time else int(rng.choice([1, 1, 1, 2, 2, 3, 4]))
    early = 1 if (on_time and planned > 1 and rng.random() < 0.08) else 0
    actual = planned + delay - early

    weight = int(rng.normal(38000, 4500)) if mode[i] == "FTL" else int(rng.gamma(2.2, 2200))
    weight = int(np.clip(weight, 150 if mode[i] == "LTL" else 20000, 44000))
    # cost: FTL ~ rate per mile; LTL ~ rate per cwt * distance factor
    if mode[i] == "FTL":
        rpm = rng.normal(2.45, 0.18) * c._cost
        if month in (11, 12):
            rpm *= 1.08
        cost = max(450, miles * rpm)
    else:
        cost = max(180, (weight / 100) * (14 + miles * 0.021) * c._cost * rng.normal(1, 0.06))
    damage = rng.random() < c._damage * (1.6 if mode[i] == "LTL" else 1)

    ship_date = dates[i]
    rows.append(dict(
        shipment_id=f"SH{100000 + i}",
        ship_date=ship_date.date(),
        promised_date=(ship_date + pd.Timedelta(days=planned)).date(),
        delivery_date=(ship_date + pd.Timedelta(days=actual)).date(),
        carrier_id=c.carrier_id,
        mode=mode[i],
        origin_city=o.city, origin_state=o.state, origin_region=o.region,
        dest_city=d.city, dest_state=d.state, dest_region=d.region,
        distance_miles=miles,
        weight_lbs=weight,
        freight_cost_usd=round(cost, 2),
        damage_claim=int(damage),
    ))

shipments = pd.DataFrame(rows).sort_values("ship_date").reset_index(drop=True)
shipments["shipment_id"] = [f"SH{100001 + i}" for i in range(len(shipments))]
shipments.to_csv(OUT / "shipments.csv", index=False)
carriers[["carrier_id", "carrier_name"]].to_csv(OUT / "carriers.csv", index=False)

# ---------------------------------------------------------------------------
# PROJECT 2: warehouse inventory (one DC, 2025 monthly sales, 240 SKUs)
# ---------------------------------------------------------------------------
cats = {
    "Packaging": (2, 12), "Pallets & Racking": (25, 140), "Safety Gear": (6, 45),
    "Shipping Supplies": (1, 8), "Handling Equipment": (80, 900), "Labels & Printing": (3, 30),
}
skus = []
k = 0
for cat, (lo, hi) in cats.items():
    for _ in range(40):
        k += 1
        unit_cost = round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), 2)
        # long-tailed demand: a few SKUs sell a lot
        base_demand = float(rng.lognormal(mean=4.2, sigma=1.15)) * (12 / np.sqrt(unit_cost))
        skus.append(dict(
            sku=f"SKU-{k:04d}", category=cat, unit_cost_usd=unit_cost,
            lead_time_days=int(rng.choice([7, 10, 14, 21, 30], p=[.25, .3, .25, .12, .08])),
            _base=max(2.0, base_demand),
        ))
products = pd.DataFrame(skus)

months = pd.period_range("2025-01", "2025-12", freq="M")
seas = np.array([0.85, 0.85, 0.95, 1.0, 1.0, 0.95, 0.95, 1.0, 1.05, 1.15, 1.3, 1.25])
sales_rows = []
for _, p in products.iterrows():
    for m_i, m in enumerate(months):
        units = int(rng.poisson(p._base * seas[m_i]))
        sales_rows.append(dict(sku=p.sku, month=str(m), units_sold=units))
sales = pd.DataFrame(sales_rows)

# current stock snapshot (as of 2025-12-31) + policy
avg_daily = sales.groupby("sku").units_sold.sum() / 365
products["avg_daily_demand"] = products.sku.map(avg_daily)
products["safety_stock_units"] = np.ceil(products.avg_daily_demand * products.lead_time_days * 0.5).astype(int)
products["reorder_point_units"] = np.ceil(products.avg_daily_demand * products.lead_time_days).astype(int) + products.safety_stock_units
cover_days = rng.choice([3, 8, 15, 30, 45, 70, 120, 200], len(products), p=[.07, .10, .14, .22, .18, .13, .09, .07])
products["on_hand_units"] = np.round(products.avg_daily_demand * cover_days * rng.uniform(0.8, 1.2, len(products))).astype(int)
dead = rng.choice(len(products), 10, replace=False)
products.loc[dead, "on_hand_units"] = products.loc[dead, "on_hand_units"] + rng.integers(150, 600, 10)
products["stockout_days_2025"] = np.where(cover_days <= 8, rng.integers(4, 25, len(products)),
                                          rng.integers(0, 4, len(products)))
products[["sku", "category", "unit_cost_usd", "lead_time_days", "safety_stock_units",
          "reorder_point_units", "on_hand_units", "stockout_days_2025"]].to_csv(OUT / "inventory_products.csv", index=False)
sales.to_csv(OUT / "inventory_monthly_sales.csv", index=False)

# ---------------------------------------------------------------------------
# PROJECT 3: truck drivers (hires 2023-01 .. 2025-12, snapshot 2025-12-31)
# ---------------------------------------------------------------------------
SNAP = pd.Timestamp("2025-12-31")
N_DRV = 1400
terminals = ["Dallas, TX", "Memphis, TN", "Columbus, OH", "Kansas City, MO", "Phoenix, AZ"]
hire = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 1095, N_DRV), unit="D")
route = rng.choice(["OTR", "Regional", "Local"], N_DRV, p=[0.45, 0.35, 0.20])
source = rng.choice(["Job board", "Employee referral", "Driving school", "Recruiting agency", "Walk-in"],
                    N_DRV, p=[0.35, 0.18, 0.22, 0.15, 0.10])
terminal = rng.choice(terminals, N_DRV)
exp_years = np.clip(np.round(rng.gamma(2.0, 3.0, N_DRV)), 0, 35).astype(int)
exp_years[source == "Driving school"] = rng.integers(0, 2, (source == "Driving school").sum())
age = np.clip(np.round(22 + exp_years * 0.9 + rng.normal(8, 7, N_DRV)), 21, 68).astype(int)

base_cpm = {"OTR": 0.58, "Regional": 0.60, "Local": 0.0}
cpm = np.array([base_cpm[r] for r in route]) + rng.normal(0, 0.035, N_DRV) + exp_years * 0.004
cpm = np.where(route == "Local", np.nan, np.round(cpm, 3))
hourly = np.where(route == "Local", np.round(rng.normal(27.5, 2.0, N_DRV), 2), np.nan)

weekly_miles = {"OTR": (2300, 350), "Regional": (1900, 300), "Local": (900, 200)}
miles = np.array([max(300, rng.normal(*weekly_miles[r])) for r in route]).round().astype(int)
home_time = {"OTR": (4, 1.5), "Regional": (8, 2), "Local": (26, 2)}
home = np.array([np.clip(rng.normal(*home_time[r]), 1, 30) for r in route]).round().astype(int)
weekly_pay = np.where(route == "Local", hourly * rng.normal(50, 4, N_DRV), np.nan_to_num(cpm) * miles)
weekly_pay = np.round(weekly_pay, 0)
dispatcher = rng.choice([f"D{i:02d}" for i in range(1, 16)], N_DRV)
bad_dispatchers = {"D04", "D11"}

# monthly hazard of quitting
hazard = np.full(N_DRV, 0.030)
hazard += np.where(route == "OTR", 0.020, np.where(route == "Regional", 0.008, -0.010))
hazard += np.where(home <= 4, 0.012, 0)
hazard += np.where(miles < np.array([weekly_miles[r][0] for r in route]) * 0.85, 0.015, 0)
hazard += np.where(source == "Employee referral", -0.015, 0)
hazard += np.where(source == "Recruiting agency", 0.010, 0)
hazard += np.where(np.isin(dispatcher, list(bad_dispatchers)), 0.014, 0)
hazard += np.where(exp_years < 1, 0.008, 0)
hazard = np.clip(hazard, 0.005, 0.2)

term_date, reason = [], []
reasons_vol = ["Pay", "Home time", "Better offer", "Dispatcher / management", "Personal", "Left trucking"]
for i in range(N_DRV):
    months_avail = (SNAP - hire[i]).days / 30.44
    t = 0.0
    left = None
    while t < months_avail:
        h = hazard[i] * (2.2 if t < 3 else 1.0)       # first 90 days are riskier
        if rng.random() < h:
            left = hire[i] + pd.Timedelta(days=int(t * 30.44 + rng.integers(0, 30)))
            break
        t += 1
    if left is not None and left > SNAP:
        left = None
    term_date.append(left.date() if left is not None else None)
    if left is None:
        reason.append(None)
    else:
        w = np.array([3, 2, 3, 1, 2, 1], float)
        if home[i] <= 4: w[1] += 4
        if dispatcher[i] in bad_dispatchers: w[3] += 4
        if miles[i] < weekly_miles[route[i]][0] * 0.85: w[0] += 4
        if rng.random() < 0.12:
            reason.append("Involuntary")
        else:
            reason.append(rng.choice(reasons_vol, p=w / w.sum()))

drivers = pd.DataFrame(dict(
    driver_id=[f"DRV{i:05d}" for i in range(1, N_DRV + 1)],
    hire_date=hire.date, termination_date=term_date, exit_reason=reason,
    terminal=terminal, route_type=route, recruiting_source=source,
    dispatcher_id=dispatcher, age=age, experience_years_at_hire=exp_years,
    pay_cpm_usd=cpm, pay_hourly_usd=hourly, avg_weekly_miles=miles,
    avg_weekly_pay_usd=weekly_pay, home_days_per_month=home,
))
drivers["status"] = np.where(drivers.termination_date.isna(), "Active", "Terminated")
drivers.sort_values("hire_date").to_csv(OUT / "drivers.csv", index=False)

# ---------------------------------------------------------------------------
# PROJECT 2 (HR): recruiting funnel, applications in 2025
# separate random stream so earlier datasets stay identical
# ---------------------------------------------------------------------------
r2 = np.random.default_rng(2026)
positions = {  # position: (share, screen pass base, days offer->start)
    "CDL-A Driver (OTR)": (0.24, 0.42, (10, 21)),
    "CDL-A Driver (Regional)": (0.16, 0.45, (10, 21)),
    "Local Driver": (0.10, 0.50, (7, 14)),
    "Warehouse Associate": (0.32, 0.60, (3, 10)),
    "Forklift Operator": (0.12, 0.52, (4, 10)),
    "Dispatcher": (0.06, 0.35, (10, 20)),
}
sources = {  # source: (share, screen mult, offer-accept mult, 90-day retention)
    "Job board": (0.38, 0.85, 1.00, 0.70),
    "Employee referral": (0.10, 1.45, 1.12, 0.86),
    "Driving school": (0.10, 1.10, 1.05, 0.66),
    "Recruiting agency": (0.12, 1.25, 0.95, 0.68),
    "Social media": (0.18, 0.70, 0.95, 0.71),
    "Company website": (0.12, 1.00, 1.05, 0.77),
}
N_APP = 8600
pos_names = list(positions); src_names = list(sources)
app_pos = r2.choice(pos_names, N_APP, p=[positions[k][0] for k in pos_names])
app_src = r2.choice(src_names, N_APP, p=[sources[k][0] for k in src_names])
# driving-school applicants only apply for driver roles
drv_roles = [k for k in pos_names if "Driver" in k]
bad = (app_src == "Driving school") & ~np.isin(app_pos, drv_roles)
app_pos[bad] = r2.choice(drv_roles, bad.sum())
app_date = pd.Timestamp("2025-01-01") + pd.to_timedelta(r2.integers(0, 330, N_APP), unit="D")
apps = []
for i in range(N_APP):
    pos, src = app_pos[i], app_src[i]
    rec = dict(application_id=f"APP{i + 1:05d}", apply_date=app_date[i].date(), position=pos, source=src,
               screen_date=None, interview_date=None, offer_date=None, offer_accepted=None,
               start_date=None, retained_90_days=None)
    d = app_date[i]
    if r2.random() < min(0.95, positions[pos][1] * sources[src][1]):
        d = d + pd.Timedelta(days=int(r2.integers(1, 7))); rec["screen_date"] = d.date()
        if r2.random() < 0.55:
            d = d + pd.Timedelta(days=int(r2.integers(2, 12))); rec["interview_date"] = d.date()
            if r2.random() < 0.58:
                d = d + pd.Timedelta(days=int(r2.integers(1, 10))); rec["offer_date"] = d.date()
                wait = (d - app_date[i]).days
                p_acc = (0.86 if wait <= 14 else 0.70 if wait <= 21 else 0.55) * sources[src][2]
                acc = r2.random() < min(0.97, p_acc)
                rec["offer_accepted"] = int(acc)
                if acc:
                    lo, hi = positions[pos][2]
                    d = d + pd.Timedelta(days=int(r2.integers(lo, hi + 1))); rec["start_date"] = d.date()
                    if d <= pd.Timestamp("2025-12-31") - pd.Timedelta(days=90):
                        rec["retained_90_days"] = int(r2.random() < sources[src][3])
    apps.append(rec)
applications = pd.DataFrame(apps)
applications.to_csv(OUT / "recruiting_applications.csv", index=False)
pd.DataFrame([
    ("Job board", "annual_fixed", 96000), ("Social media", "annual_fixed", 54000),
    ("Company website", "annual_fixed", 12000), ("Driving school", "annual_fixed", 30000),
    ("Recruiting agency", "per_hire", 3500), ("Employee referral", "per_hire", 1500),
], columns=["source", "cost_type", "amount_usd"]).to_csv(OUT / "recruiting_source_costs.csv", index=False)

# ---------------------------------------------------------------------------
# PROJECT 3 (HR): warehouse workforce — absence, overtime, pay (2025)
# ---------------------------------------------------------------------------
r3 = np.random.default_rng(7)
N_EMP = 650
sites = np.array(["Columbus, OH", "Memphis, TN", "Dallas, TX"])
emp_site = r3.choice(sites, N_EMP, p=[0.36, 0.30, 0.34])
gender = r3.choice(["F", "M"], N_EMP, p=[0.38, 0.62])
role = np.empty(N_EMP, dtype=object)
for i in range(N_EMP):
    p = [0.66, 0.14, 0.13, 0.07] if gender[i] == "F" else [0.46, 0.32, 0.14, 0.08]
    role[i] = r3.choice(["Associate", "Forklift Operator", "Team Lead", "Supervisor"], p=p)
shift = r3.choice(["Day", "Night", "Weekend"], N_EMP, p=[0.50, 0.35, 0.15])
hire_d = pd.Timestamp("2025-12-31") - pd.to_timedelta((r3.gamma(1.6, 2.2, N_EMP) * 365).astype(int), unit="D")
hire_d = hire_d.where(hire_d >= pd.Timestamp("2016-01-01"), pd.Timestamp("2016-01-01"))
tenure_y = np.clip(np.asarray((pd.Timestamp("2025-01-01") - hire_d).days) / 365, 0, None)
base = pd.Series(role).map({"Associate": 18.5, "Forklift Operator": 21.0, "Team Lead": 24.5, "Supervisor": 29.5}).values
rate = (base + np.minimum(tenure_y, 8) * 0.35 + np.where(shift == "Night", 1.5, 0)
        + pd.Series(emp_site).map({"Columbus, OH": 0.4, "Memphis, TN": -0.6, "Dallas, TX": 0.0}).values
        - np.where(gender == "F", 0.25, 0) + r3.normal(0, 0.6, N_EMP))
managers = {s: [f"M{j:02d}" for j in range(k, k + 6)] for s, k in zip(sites, [1, 7, 13])}
mgr = np.array([r3.choice(managers[s]) for s in emp_site])
engage = np.clip(np.round(r3.normal(3.6, 0.6, N_EMP) - np.where(np.isin(mgr, ["M09", "M15"]), 0.6, 0)
                          - np.where(shift == "Night", 0.2, 0), 1), 1, 5)
employees = pd.DataFrame(dict(employee_id=[f"E{i:04d}" for i in range(1, N_EMP + 1)], site=emp_site,
                              role=role, shift=shift, gender=gender, hire_date=hire_d.date,
                              hourly_rate_usd=np.round(rate, 2), manager_id=mgr, engagement_score=engage))
weeks = pd.date_range("2025-01-06", periods=52, freq="W-MON")
peak = np.array([1.8 if w.month in (11, 12) else 1.0 for w in weeks])
ts, term = [], []
for e in employees.itertuples():
    ot_base = 2.0 + (2.5 if e.site == "Memphis, TN" else 0) + (1.5 if e.shift == "Night" else 0)
    ot_base *= 0.5 if e.role == "Supervisor" else 1
    fatigue, left_week = 0.0, None
    risk = 0.0045 + (0.004 if e.engagement_score <= 3 else 0) + (0.003 if e.site == "Memphis, TN" else 0)
    for w_i, w in enumerate(weeks):
        if w < pd.Timestamp(e.hire_date):
            continue
        ot = max(0.0, r3.normal(ot_base * peak[w_i], 2.0))
        fatigue = 0.6 * fatigue + 0.4 * ot
        p_abs = 0.025 + 0.004 * fatigue + (0.02 if e.engagement_score <= 3 else 0) + (0.01 if e.manager_id in ("M09", "M15") else 0)
        absence = float(r3.choice([8, 8, 16, 24])) if r3.random() < p_abs else 0.0
        worked = 40 - absence + (ot if absence == 0 else 0)
        ts.append((e.employee_id, w.date(), 40, round(worked, 1), round(ot if absence == 0 else 0, 1), absence))
        if r3.random() < risk + 0.0015 * max(0, fatigue - 4):
            left_week = w; break
    term.append(left_week.date() if left_week is not None else None)
employees["termination_date"] = term
employees.to_csv(OUT / "warehouse_employees.csv", index=False)
pd.DataFrame(ts, columns=["employee_id", "week_start", "scheduled_hours", "worked_hours",
                          "overtime_hours", "unplanned_absence_hours"]).to_csv(OUT / "warehouse_timesheets.csv", index=False)

print("applications:", len(applications), "| hires:", applications.start_date.notna().sum(),
      "| warehouse employees:", len(employees), "| timesheet rows:", len(ts))
print("shipments:", len(shipments), "| products:", len(products), "| sales rows:", len(sales),
      "| drivers:", len(drivers), "| terminated:", (drivers.status == "Terminated").sum())
