import numpy as np
import pandas as pd


def make_year(column):
    year = column.split('-')[0]
    return int(year)


def make_month(column):
    month = column.split('-')[1]
    return int(month)


def make_day(column):
    day = column.split('-')[2][0:2]
    return int(day)


def make_date(column):
    date = column.split(" ")[0]
    return date


def convert_prd(column):
    if column == 401:
        return 1397
    if column == 901:
        return 1398
    if column == 1001:
        return 1399
    if column == 1101:
        return 1400
    if column == 1201:
        return 1401
    elif column == 1301:
        return 1402


def make_prd(column):
    id_prd = column.split('-')[3]
    return int(id_prd)


def make_new_date(column):
    year, month, day = column.split('-')[0:3]
    date = '-'.join([year, month, day])
    return date


def read_data(path='./datasets/data_sale_bamland.csv'):
    df = pd.read_csv(path, header=None)
    columns = ['date', 'id_prd_to_plc', 'id_br', 'amount', 'price']
    df.columns = columns

    df['total_price'] = df['price'] * df['amount']
    df['year'] = df['date'].apply(make_year)
    df['month'] = df['date'].apply(make_month)
    df['day'] = df['date'].apply(make_day)
    df['date'] = df['date'].apply(make_date)
    df['id_prd_to_plc'] = df['id_prd_to_plc'].apply(convert_prd)

    # Create date column for splitting
    df['date'] = df['date'] + '-' + df['id_prd_to_plc'].astype(str)

    # Group by day
    new_data = df.groupby(df['date'], as_index=False).sum(numeric_only=True)
    new_data['id_br'] = 51338  # id bamland branch

    # Create again columns after make new df for each day sales
    new_data['id_br'] = new_data['id_br'].astype(int)
    new_data = new_data.drop(['year', 'month', 'day'], axis=1)
    new_data['year'] = new_data['date'].apply(make_year)
    new_data['month'] = new_data['date'].apply(make_month)
    new_data['day'] = new_data['date'].apply(make_day)
    new_data['id_prd_to_plc'] = new_data['date'].apply(make_prd)
    new_data['date'] = new_data['date'].apply(make_new_date)
    # new_data['series'] = np.arange(1, len(new_data)+1)
    return new_data


def fill_days(new_data):
    new_data['date'] = pd.to_datetime(new_data['date'])
    # Initialize an empty DataFrame to store the appended rows
    appended_rows = pd.DataFrame(columns=new_data.columns)

    # Iterate through the DataFrame
    for i in range(len(new_data) - 1):
        current_date = new_data['date'].iloc[i]
        next_date = new_data['date'].iloc[i + 1]

        # Check if there is a gap between current_date and next_date
        if (next_date - current_date).days > 1 or (next_date - current_date).days < 29:
            # Append missing dates with the next day
            missing_dates = pd.date_range(current_date + pd.DateOffset(1), next_date - pd.DateOffset(1), freq='D')
            appended_rows = pd.concat([appended_rows, pd.DataFrame({'date': missing_dates})])

    # Append the missing rows to the original DataFrame
    new_data = pd.concat([new_data, appended_rows])

    # Sort the DataFrame by date again
    new_data = new_data.sort_values('date').reset_index(drop=True)
    new_data['total_price'] = new_data['total_price'].fillna(0)
    new_data['price'] = new_data['price'].fillna(0)
    new_data['amount'] = new_data['amount'].fillna(0)
    new_data['id_br'] = new_data['id_br'].fillna(51238)
    new_data['id_prd_to_plc'] = new_data['id_prd_to_plc'].fillna(0)
    new_data['date'] = new_data['date'].astype(str)
    new_data['year'] = new_data['date'].apply(make_year)
    new_data['month'] = new_data['date'].apply(make_month)
    new_data['day'] = new_data['date'].apply(make_day)
    new_data['series'] = np.arange(1, len(new_data) + 1)
    return new_data


def train_model(new_data):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    train = new_data.iloc[:-60]
    test = new_data.iloc[-60:-1]

    X = train[['series', 'year', 'month', 'day']]
    y = train['total_price']

    scaler = MinMaxScaler()
    scaler = scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.000001)
    X_test, y_test = test[['series', 'year', 'month', 'day']], test['total_price']
    X_test = scaler.transform(X_test)

    # model = RandomForestRegressor(n_estimators=1000, max_depth=20, criterion='absolute_error')
    model = xgb.XGBRegressor(n_estimators=15000, learning_rate=0.01, min_child_weight=50, max_depth=50)
    model.fit(X_train, y_train)

    return model, scaler


def save_model(model, path_model):
    import pickle

    pickle.dump(model, open(path_model, "wb"))


if __name__ == "__main__":

    new_data = read_data()

    new_df = fill_days(new_data)
    # print(new_df.info())
    model, scaler = train_model(new_df)

    save_model(model, path_model="./models/xgboost_predictor.pickle")
    save_model(scaler, path_model="./models/xgboost_scaler.pickle")
