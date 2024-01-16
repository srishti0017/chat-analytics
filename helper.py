from urlextract import URLExtract
from wordcloud import WordCloud
extract = URLExtract()
import pandas as pd
from collections import Counter
import emoji
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = []
    for message in df['message']:
         words.extend(message.split())


    # if selected_user == 'Overall':
    #     #1st fetch no. of messages
    #     num_messages = df.shape[0]
    #     #2nd no.of words
    #     words = []
    #     for message in df['message']:
    #         words.extend(message.split())
    #     return num_messages,len(words)
    # else:
    #     new_df = df[df['user'] == selected_user]
    #     num_messages = new_df.shape[0]
    #     words = []
    #     for message in new_df['message']:
    #         words.extend(message.split())
    #     return num_messages,len(words)

    # fetch no. of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch no. of links
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words),num_media_messages,len(links)

def most_busy_user(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user, df):
    # if selected_user != 'Overall':
    #     df = df[df['user'] == selected_user]
    #
    # wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    # df_wc = wc.generate(df['message'].str.cat(sep=" "))
    # return df_wc
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

# import emoji
from collections import Counter
import pandas as pd

# def emoji_helper(selected_user, df):
#     if selected_user != 'Overall':
#         df = df[df['user'] == selected_user]
#
#     emojis = []
#     for message in df['message']:
#         emojis.extend(emoji.demojize(message).split())
#
#     emoji_df = pd.DataFrame(Counter(emojis).most_common(), columns=['Emoji', 'Count'])
#
#     return emoji_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

def sentiment_analysis(selected_user, df):
    df["row_id"] = df.index + 1
    ds_subset = df[['row_id', 'message']].copy()
    ds_subset['message'] = ds_subset['message'].str.replace("[^a-zA-Z#]", " ")
    ds_subset['message'] = ds_subset['message'].str.casefold()
    ds1 = pd.DataFrame()
    ds1['row_id'] = ['99999999999']
    ds1['sentiment_type'] = 'NA999NA'
    ds1['sentiment_score'] = 0
    sid = SentimentIntensityAnalyzer()
    t_ds = df
    for index, row in ds_subset.iterrows():
        scores = sid.polarity_scores(row[1])
        for key, value in scores.items():
            temp = [key, value, row[0]]
            ds1['row_id'] = row[0]
            ds1['sentiment_type'] = key
            ds1['sentiment_score'] = value
            t_ds = t_ds.append(ds1)
    # remove dummy row with row_id = 99999999999
    t_ds_cleaned = t_ds[t_ds.row_id != '99999999999']
    # remove duplicates if any exist
    t_ds_cleaned = t_ds_cleaned.drop_duplicates()
    # only keep rows where sentiment_type = compound
    t_ds_cleaned = t_ds[t_ds.sentiment_type == 'compound']

    return t_ds_cleaned

def sentiment_summary(selected_user, df):
    t_ds_cleaned = sentiment_analysis(selected_user, df)
    ds_output = pd.merge(df, t_ds_cleaned, on='row_id', how='inner')


    return ds_output[["sentiment_score"]].describe()

def graph(selected_user, df):
    graph = sentiment_analysis(selected_user, df)
    row_id = df['date']
    sentiment_score = graph['sentiment_score']
    return row_id, sentiment_score

def graph1(selected_user, df):
    graph = sentiment_analysis(selected_user, df)
    row_id = df['user']
    sentiment_score = graph['sentiment_score']
    return row_id, sentiment_score
