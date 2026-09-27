import numpy as np
import pandas as pd
from forecasting import generate_best_forecast


# ─────────────────────────────────────────────
# Single product decision  (used by /inventory API)
# ─────────────────────────────────────────────
def calculate_inventory_decision(df, store_id, product_id, lead_time=7):
    """Run the ML forecast for ONE store+product and return a decision dict."""
    forecast_result  = generate_best_forecast(df, store_id, product_id, steps=7)
    forecast_values  = forecast_result["Forecast"]

    total_forecast_demand = sum(forecast_values)
    avg_daily_demand      = total_forecast_demand / 7

    current_inventory = df[
        (df['Store ID'] == store_id) &
        (df['Product ID'] == product_id)
    ]['Inventory Level'].iloc[-1]

    demand_std   = np.std(forecast_values)
    safety_stock = demand_std * 1.65
    reorder_point = (avg_daily_demand * lead_time) + safety_stock

    if current_inventory < reorder_point * 0.8:
        status           = "Critical"
        suggested_action = f"Urgent Reorder {int(reorder_point - current_inventory)} units"
    elif current_inventory < reorder_point:
        status           = "Low Stock"
        suggested_action = f"Reorder {int(reorder_point - current_inventory)} units"
    else:
        status           = "Optimal"
        suggested_action = "Monitor"

    return {
        "Store_ID":          store_id,
        "Product_ID":        product_id,
        "Current_Inventory": float(current_inventory),
        "Avg_Daily_Demand":  float(avg_daily_demand),
        "Forecast_7_Day":    float(total_forecast_demand),
        "Safety_Stock":      float(safety_stock),
        "Reorder_Point":     float(reorder_point),
        "Status":            status,
        "Suggested_Action":  suggested_action,
    }


# ─────────────────────────────────────────────
# FAST dashboard summary  (pure pandas, no ML)
# ─────────────────────────────────────────────
def _fast_status(df):
    """Return a DataFrame with [Store ID, Product ID, status] in ~0.07 s."""
    last = df.sort_values('Date').groupby(
        ['Store ID', 'Product ID']
    ).last().reset_index()[['Store ID', 'Product ID', 'Inventory Level']]

    avg = (df.groupby(['Store ID', 'Product ID'])['Units Sold']
             .mean()
             .reset_index()
             .rename(columns={'Units Sold': 'avg_daily'}))

    merged = last.merge(avg, on=['Store ID', 'Product ID'])
    merged['days_left'] = (merged['Inventory Level']
                           / merged['avg_daily'].clip(lower=1))

    merged['status'] = np.where(
        merged['days_left'] < 3, 'Critical',
        np.where(merged['days_left'] < 7, 'Low Stock', 'Optimal')
    )
    return merged


def generate_dashboard_summary(df):
    status_df = _fast_status(df)
    counts    = status_df['status'].value_counts()
    return {
        "Total_Stores": int(df['Store ID'].nunique()),
        "Critical":     int(counts.get('Critical',  0)),
        "Low_Stock":    int(counts.get('Low Stock',  0)),
        "Optimal":      int(counts.get('Optimal',    0)),
    }


def generate_store_level_summary(df):
    status_df    = _fast_status(df)
    store_counts = (status_df.groupby(['Store ID', 'status'])
                              .size()
                              .unstack(fill_value=0))

    for col in ['Critical', 'Low Stock', 'Optimal']:
        if col not in store_counts.columns:
            store_counts[col] = 0

    result = []
    for store, row in store_counts.iterrows():
        c  = int(row['Critical'])
        l  = int(row['Low Stock'])
        o  = int(row['Optimal'])
        st = 'Critical' if c > 0 else ('Low Stock' if l > 0 else 'Optimal')
        result.append({
            "Store_ID":       store,
            "Critical_Count": c,
            "Low_Count":      l,
            "Optimal_Count":  o,
            "Store_Status":   st,
        })
    return result
