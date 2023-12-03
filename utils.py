import pandas as pd
from datetime import date, timedelta
from data_preprocessing import read_data


def jalali_to_gregorian(jy, jm, jd):
    jy += 1595
    days = -355668 + (365 * jy) + ((jy // 33) * 8) + (((jy % 33) + 3) // 4) + jd
    if (jm < 7):
        days += (jm - 1) * 31
    else:
        days += ((jm - 7) * 30) + 186
    gy = 400 * (days // 146097)
    days %= 146097
    if (days > 36524):
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if (days >= 365):
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if (days > 365):
        gy += ((days - 1) // 365)
        days = (days - 1) % 365
    gd = days + 1
    if ((gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0)):
        kab = 29
    else:
        kab = 28
    sal_a = [0, 31, kab, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    gm = 0
    while (gm < 13 and gd > sal_a[gm]):
        gd -= sal_a[gm]
        gm += 1
    return [gy, gm, gd]


def create_date_dataframe(start_date, end_date) -> pd.DataFrame:
    '''
        Takes 2 date and return a dataframe from date 1 to date 2
        `Parameters`:
        `date1` : Start period time date that you want in shamsi hijri format
        `date2` : End period time date that you want in shamsi hijri format

        `return`: The dataframe contains all days from start_date to end_date
    '''

    # Convert the input date lists to datetime objects
    start_date = date(*start_date)
    end_date = date(*end_date)

    # Initialize an empty list to store date tuples
    date_list = []

    # Create a date range using timedelta and append each date as a tuple
    current_date = start_date
    while current_date <= end_date:
        id_prd = {'2018': 1397, '2019': 1398, '2020': 1399, '2021': 1400, '2022': 1401, '2023': 1402, '2024': 1403,
                  '2025': 1404, '2026': 1405}
        id_prd_to_plc = id_prd[str(current_date.year)]
        date_list.append((id_prd_to_plc, current_date.year, current_date.month, current_date.day))
        current_date += timedelta(days=1)

    # Convert the list of date tuples to a DataFrame
    date_df = pd.DataFrame(date_list, columns=["id_prd_to_plc", "year", "month", "day"])

    return date_df


def make_period_time(date1: list, date2: list = None):
    """
        Converts hijri shamsi date to gregorian and return dataframe using `create_date_dataframe`
    """
    date1_grg = jalali_to_gregorian(date1[0], date1[1], date1[2])
    date2_grg = jalali_to_gregorian(date2[0], date2[1], date2[2])

    df = create_date_dataframe(date1_grg, date2_grg)
    return df

# s = [1402, 1, 1]
# e = [1402, 2, 1]
#
# df = make_period_time(s, e)
# print(df.head())
