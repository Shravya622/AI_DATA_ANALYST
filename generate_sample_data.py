"""
Generates tests/fixtures/sample_data.csv deterministically.
Run once: python generate_sample_data.py
"""
import random
import csv
import os
from datetime import date, timedelta

random.seed(42)

regions = ["North", "South", "East", "West"]
products = ["Widget", "Gadget", "Doohickey", "Gizmo"]

start_date = date(2024, 1, 1)
end_date = date(2024, 7, 18)
date_range_days = (end_date - start_date).days

# Pick 5 indices that will be outliers
outlier_indices = set(random.sample(range(200), 5))

rows = []
for i in range(200):
    d = start_date + timedelta(days=random.randint(0, date_range_days))
    region = random.choice(regions)
    product = random.choice(products)
    units_sold = random.randint(5, 50)
    discount_pct = round(random.uniform(0.0, 0.30), 4)

    if i in outlier_indices:
        # Intentional outlier: revenue > 10 000
        revenue = round(random.uniform(10001.0, 25000.0), 2)
    else:
        revenue = round(random.uniform(500.0, 3000.0), 2)

    rows.append([d.isoformat(), region, product, revenue, units_sold, discount_pct])

out_path = os.path.join(
    os.path.dirname(__file__), "tests", "fixtures", "sample_data.csv"
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "region", "product", "revenue", "units_sold", "discount_pct"])
    writer.writerows(rows)

# Verification
with open(out_path) as f:
    all_lines = f.readlines()

data_rows = all_lines[1:]  # skip header
outlier_rows = [r for r in data_rows if float(r.split(",")[3]) > 10000]

print(f"File written : {out_path}")
print(f"Total rows   : {len(data_rows)}")
print(f"Outliers (revenue > 10 000): {len(outlier_rows)}")
for r in outlier_rows:
    parts = r.strip().split(",")
    print(f"  date={parts[0]}  revenue={parts[3]}")
print("First 3 data rows:")
for r in data_rows[:3]:
    print(" ", r.strip())
