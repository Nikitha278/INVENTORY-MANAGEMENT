import pandas as pd

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by Date
    df = df.sort_values(by='Date')

    return df

def create_features(df):
    
    # Time features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Group by Store + Product for proper time series
    df = df.sort_values(['Store ID', 'Product ID', 'Date'])

    # Lag features per group
    df['lag_1'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].shift(1)
    df['lag_7'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].shift(7)

    # Rolling mean per group
    df['rolling_mean_7'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'] \
                               .rolling(7).mean().reset_index(level=[0,1], drop=True)

    df['rolling_mean_14'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'] \
                                .rolling(14).mean().reset_index(level=[0,1], drop=True)

    df = df.dropna()

    return df

