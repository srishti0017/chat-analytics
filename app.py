import streamlit as st
import preprocessor, helper
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

st.sidebar.title("WhatsApp Chat Analyser")


uploaded_file = st.sidebar.file_uploader("Choose a file")
# converting data from stream into string
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    #st.text(data)
    df = preprocessor.preprocess(data)
    st.dataframe(df)
    # fetch unique users
    user_list = df['user'].unique().tolist()

    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):

        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)

        with col2:
            st.header("Total Words")
            st.title(words)

        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)

        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # Monthly Timeline
        st.title("Monthly Timeline")
        monthly_timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(monthly_timeline['time'], monthly_timeline['message'], color = "Green")
        plt.xticks(rotation = 'vertical')
        st.pyplot(fig)

        # Daily Timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity Map
        st.title("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color="purple")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the active user of the group
        if selected_user == 'Overall':
            st.title("Most Active Users")
            x, new_df = helper.most_busy_user(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                # ax.bar(x.index, x.values, color='red')
                # plt.xticks(rotation='vertical')
                # st.pyplot(fig)
                ax.pie(x.values, labels=x.index, autopct='%1.1f%%', startangle=90,
                       colors=['red', 'green', 'blue'])  # Adjust colors as needed
                ax.axis('equal')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title('WordCloud')
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)



        # Most common Word
        most_common_df = helper.most_common_words(selected_user, df)

        # st.dataframe(most_common_df)
        st.title('Most Common Word')
        fig,ax = plt.subplots()
        ax.bar(most_common_df[0], most_common_df[1])

        plt.xticks(rotation = 'vertical')

        st.pyplot(fig)

        # # # Emoji analysis
        # emoji_df = helper.emoji_helper(selected_user,df)
        # st.title('Emoji Analysis')
        #
        # col1, col2 = st.columns(2)
        #
        # with col1:
        #     st.dataframe(emoji_df)
        # with col2:
        #     fig, ax = plt.subplots()
        #     ax.pie(emoji_df[1].head(10), labels=emoji_df[0].head(10), autopct="%0.2f")
        #     st.pyplot(fig)

        st.title("Sentiment Score")
        t_ds_cleaned = helper.sentiment_analysis(selected_user, df)
        st.dataframe(t_ds_cleaned)


        st.title("Sentiment Summary")
        graph = helper.sentiment_summary(selected_user, df)
        st.dataframe(graph)

        st.title(" Sentiment Graph")
        row_id, sentiment_analysis = helper.graph(selected_user, df)
        fig, ax = plt.subplots()
        ax.bar(row_id,sentiment_analysis)

        plt.xticks(rotation='vertical')

        st.pyplot(fig)

        st.title("Sentiment Graph wrt User")
        row_id, sentiment_analysis = helper.graph1(selected_user, df)
        fig, ax = plt.subplots()
        ax.bar(row_id, sentiment_analysis)

        plt.xticks(rotation='vertical')

        st.pyplot(fig)
