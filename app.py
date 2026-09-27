from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from preprocessing import load_and_clean_data, create_features
from inventory_logic import (
    calculate_inventory_decision,
    generate_dashboard_summary,
    generate_store_level_summary,
)
from forecasting import generate_best_forecast
import pandas as pd
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = "retailai_secret_key"

# ── Load dataset once at startup ──
df = load_and_clean_data("retail_store_inventory.csv")
df = create_features(df)

# ── Raw CSV for fast stats (no lag columns) ──
_raw = pd.read_csv("retail_store_inventory.csv", parse_dates=["Date"])


def get_initials(name):
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper() if len(name) >= 2 else name.upper()


def get_dashboard_stats():
    monthly = _raw.groupby(_raw['Date'].dt.to_period('M'))['Units Sold'].sum()
    last12  = monthly.tail(12)

    weekly  = _raw.groupby(_raw['Date'].dt.to_period('W'))['Units Sold'].sum()
    last13  = weekly.iloc[-14:-1]

    cat     = _raw.groupby('Category')['Units Sold'].sum()
    stores  = _raw.groupby('Store ID')['Units Sold'].sum()

    return {
        "monthly_labels": [str(p) for p in last12.index],
        "monthly_values": [int(v) for v in last12.values],
        "weekly_labels":  [str(p)[:10] for p in last13.index],
        "weekly_values":  [int(v) for v in last13.values],
        "cat_labels":     cat.index.tolist(),
        "cat_values":     [int(v) for v in cat.values],
        "store_labels":   stores.index.tolist(),
        "store_values":   [int(v) for v in stores.values],
        "total_units":    f"{int(_raw['Units Sold'].sum()):,}",
        "avg_daily":      f"{int(_raw.groupby('Date')['Units Sold'].sum().mean()):,}",
        "total_inv":      f"{int(_raw.groupby(['Store ID','Product ID'])['Inventory Level'].last().sum()):,}",
        "num_stores":     int(_raw['Store ID'].nunique()),
    }


def get_analytics_stats():
    """Fast analytics data — pure pandas, no ML."""
    store_cat  = _raw.groupby(['Store ID', 'Category'])['Units Sold'].sum().unstack(fill_value=0)
    categories = store_cat.columns.tolist()
    stores_list = store_cat.index.tolist()

    # Build per-store series lists for stacked bar
    cat_series = {cat: [int(store_cat.loc[s, cat]) for s in stores_list] for cat in categories}

    region     = _raw.groupby('Store ID')['Region'].first().to_dict()
    store_tot  = _raw.groupby('Store ID')['Units Sold'].sum().to_dict()
    store_avg  = _raw.groupby('Store ID')['Units Sold'].mean().round(1).to_dict()

    # Inventory status fast
    last_inv   = (_raw.sort_values('Date')
                      .groupby(['Store ID', 'Product ID'])
                      .last()[['Inventory Level']]
                      .reset_index())
    avg_d      = (_raw.groupby(['Store ID', 'Product ID'])['Units Sold']
                      .mean().reset_index()
                      .rename(columns={'Units Sold': 'avg_d'}))
    merged     = last_inv.merge(avg_d, on=['Store ID', 'Product ID'])
    merged['days_left'] = merged['Inventory Level'] / merged['avg_d'].clip(lower=1)
    merged['status']    = np.where(merged['days_left'] < 3, 'Critical',
                          np.where(merged['days_left'] < 7, 'Low Stock', 'Optimal'))

    store_status = {}
    for store in stores_list:
        sub = merged[merged['Store ID'] == store]['status'].value_counts()
        store_status[store] = {
            "critical": int(sub.get('Critical', 0)),
            "low":      int(sub.get('Low Stock', 0)),
            "optimal":  int(sub.get('Optimal', 0)),
            "region":   region.get(store, '—'),
            "total":    f"{int(store_tot.get(store, 0)):,}",
            "avg_daily": float(store_avg.get(store, 0)),
        }

    return {
        "stores_list": stores_list,
        "categories":  categories,
        "cat_series":  cat_series,
        "store_status": store_status,
    }


# ─────────────────────
#  ROUTES
# ─────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "User").strip() or "User"
        session["user"]     = username
        session["initials"] = get_initials(username)
        return redirect(url_for("home"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    stats = get_dashboard_stats()
    return render_template("main.html",
                           user=session["user"],
                           initials=session.get("initials", "U"),
                           **stats)


@app.route("/inventory")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    summary = generate_dashboard_summary(df)
    return render_template("Inventory.html",
                           total_stores=summary["Total_Stores"],
                           critical=summary["Critical"],
                           low_stock=summary["Low_Stock"],
                           optimal=summary["Optimal"],
                           user=session["user"],
                           initials=session.get("initials", "U"))


@app.route("/analytics")
def analytics():
    if "user" not in session:
        return redirect(url_for("login"))
    data = get_analytics_stats()
    return render_template("analytics.html",
                           user=session["user"],
                           initials=session.get("initials", "U"),
                           **data)


@app.route("/forecasting")
def forecasting_page():
    if "user" not in session:
        return redirect(url_for("login"))
    stores   = sorted(df['Store ID'].unique().tolist())
    products = sorted(df['Product ID'].unique().tolist())
    return render_template("forecasting.html",
                           stores=stores,
                           products=products,
                           user=session["user"],
                           initials=session.get("initials", "U"))


@app.route("/settings")
def settings():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("settings.html",
                           user=session["user"],
                           initials=session.get("initials", "U"))


# ── APIs ──

@app.route("/forecast/<store>/<product>")
def forecast(store, product):
    if "user" not in session:
        return jsonify({"error": "Unauthorized"})
    result = generate_best_forecast(df, store, product, steps=7)
    return jsonify(result)


@app.route("/inventory/<store>/<product>")
def inventory(store, product):
    if "user" not in session:
        return jsonify({"error": "Unauthorized"})
    result = calculate_inventory_decision(df, store, product)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
