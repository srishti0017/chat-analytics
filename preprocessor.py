import re
import pandas as pd


def preprocess(data):
    pattern = '\d{1,2}\/\d{1,2}\/\d{2,4},\s\d{1,2}:\d{1,2}\s\w{1,2}\s-\s'
    msgs = re.split(pattern, data)[1:]  # discard 1st msg regarding encryption declaration by WhatsApp
    date = re.findall(pattern, data)
    ds = pd.DataFrame({'user_msg': msgs, 'date': date})
    # Clean the date strings by removing leading and trailing spaces
    ds['date'] = ds['date'].str.strip()
    # Convert msg data type
    ds['date'] = pd.to_datetime(ds['date'], format='%m/%d/%y, %I:%M %p -')
    # Rest of the code remains the same...

    users = []
    messages = []
    for message in ds['user_msg']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])
    ds['user'] = users
    ds['message'] = messages
    ds.drop(columns=['user_msg'], inplace=True)
    ds['only_date'] = ds['date'].dt.date
    ds['year'] = ds['date'].dt.year
    ds['month_num'] = ds['date'].dt.month
    ds['month'] = ds['date'].dt.month_name()
    ds['day'] = ds['date'].dt.day
    ds['day_name'] = ds['date'].dt.day_name()
    ds['hour'] = ds['date'].dt.hour
    ds['minute'] = ds['date'].dt.minute

    period = []
    for hour in ds[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    ds['period'] = period
    ds["row_id"] = ds.index + 1

    return ds