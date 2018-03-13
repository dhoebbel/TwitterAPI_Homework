
# Dependencies
import tweepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import rand
from itertools import cycle

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# Target User Account
target_user1 = "@bbcworld"

# Variables for holding sentiments
compound_list = []
positive_list = []
negative_list = []
neutral_list = []

# Loop through 10 pages of tweets (total 100 tweets)
for x in range(5):

    # Get all tweets from home feed
    public_tweets = api.user_timeline(target_user1)

    # Loop through all tweets
    for tweet in public_tweets:

        # Run Vader Analysis on each tweet
        results = analyzer.polarity_scores(tweet["text"])
        compound = results["compound"]
        pos = results["pos"]
        neu = results["neu"]
        neg = results["neg"]

        # Add each value to the appropriate list
        compound_list.append(compound)
        positive_list.append(pos)
        negative_list.append(neg)
        neutral_list.append(neu)
        
        # Print the Averages
print(f"User: {target_user1}")
print(f"Compound: {np.mean(compound_list):.3f}")
print(f"Positive: {np.mean(positive_list):.3f}")
print(f"Neutral: {np.mean(neutral_list):.3f}")
print(f"Negative: {np.mean(negative_list):.3f}")

# Convert sentiments to DataFrame
bbc_news = pd.DataFrame(
    {'Compound': compound_list,
     'Positive': positive_list,
     'Neutral': neutral_list,
     'Negative': negative_list
    })
bbc_news["Tweets Ago"] =  range(len(bbc_news))  
bbc_news["name"] = "BBC"
bbc_news.head()

# Target User Account
target_user2 = "@cbsnews"

# Variables for holding sentiments
compound_list2 = []
positive_list2 = []
negative_list2 = []
neutral_list2 = []

# Loop through 10 pages of tweets (total 100 tweets)
for x in range(5):

    # Get all tweets from home feed
    public_tweets2 = api.user_timeline(target_user2)

    # Loop through all tweets
    for tweet in public_tweets:

        # Run Vader Analysis on each tweet
        results2 = analyzer.polarity_scores(tweet["text"])
        compound = results2["compound"]
        pos = results2["pos"]
        neu = results2["neu"]
        neg = results2["neg"]

        # Add each value to the appropriate list
        compound_list2.append(compound)
        positive_list2.append(pos)
        negative_list2.append(neg)
        neutral_list2.append(neu)
        
        # Print the Averages
print(f"User: {target_user2}")
print(f"Compound: {np.mean(compound_list2):.3f}")
print(f"Positive: {np.mean(positive_list2):.3f}")
print(f"Neutral: {np.mean(neutral_list2):.3f}")
print(f"Negative: {np.mean(negative_list2):.3f}")

# Convert sentiments to DataFrame
cbs_news = pd.DataFrame(
    {'Compound': compound_list2,
     'Positive': positive_list2,
     'Neutral': neutral_list2,
     'Negative': negative_list2
    })
cbs_news["Tweets Ago"] =  range(len(cbs_news))  
cbs_news["name"] = "CBS"
cbs_news.head()

# Target User Account
target_user3 = "@cnn"

# Variables for holding sentiments
compound_list3 = []
positive_list3 = []
negative_list3 = []
neutral_list3 = []

# Loop through 10 pages of tweets (total 100 tweets)
for x in range(5):

    # Get all tweets from home feed
    public_tweets3 = api.user_timeline(target_user3)

    # Loop through all tweets
    for tweet in public_tweets3:

        # Run Vader Analysis on each tweet
        results3 = analyzer.polarity_scores(tweet["text"])
        compound = results3["compound"]
        pos = results3["pos"]
        neu = results3["neu"]
        neg = results3["neg"]

        # Add each value to the appropriate list
        compound_list3.append(compound)
        positive_list3.append(pos)
        negative_list3.append(neg)
        neutral_list3.append(neu)
        
        # Print the Averages
print(f"User: {target_user3}")
print(f"Compound: {np.mean(compound_list3):.3f}")
print(f"Positive: {np.mean(positive_list3):.3f}")
print(f"Neutral: {np.mean(neutral_list3):.3f}")
print(f"Negative: {np.mean(negative_list3):.3f}")

# Convert sentiments to DataFrame
cnn_news = pd.DataFrame(
    {'Compound': compound_list3,
     'Positive': positive_list3,
     'Neutral': neutral_list3,
     'Negative': negative_list3
    })
cnn_news["Tweets Ago"] =  range(len(cnn_news))  
cnn_news["name"] = "CNN"
cnn_news.head()

# Target User Account
target_user4 = "@foxnews"

# Variables for holding sentiments
compound_list4 = []
positive_list4 = []
negative_list4 = []
neutral_list4 = []

# Loop through 10 pages of tweets (total 100 tweets)
for x in range(5):

    # Get all tweets from home feed
    public_tweets4 = api.user_timeline(target_user4)

    # Loop through all tweets
    for tweet in public_tweets4:

        # Run Vader Analysis on each tweet
        results4 = analyzer.polarity_scores(tweet["text"])
        compound = results4["compound"]
        pos = results4["pos"]
        neu = results4["neu"]
        neg = results4["neg"]

        # Add each value to the appropriate list
        compound_list4.append(compound)
        positive_list4.append(pos)
        negative_list4.append(neg)
        neutral_list4.append(neu)
        
        # Print the Averages
print(f"User: {target_user4}")
print(f"Compound: {np.mean(compound_list4):.3f}")
print(f"Positive: {np.mean(positive_list4):.3f}")
print(f"Neutral: {np.mean(neutral_list4):.3f}")
print(f"Negative: {np.mean(negative_list4):.3f}")

# Convert sentiments to DataFrame
fox_news = pd.DataFrame(
    {'Compound': compound_list4,
     'Positive': positive_list4,
     'Neutral': neutral_list4,
     'Negative': negative_list4
    })
fox_news["Tweets Ago"] =  range(len(fox_news))  
fox_news["name"] = "Fox News"
fox_news.head()

# Target User Account
target_user5 = "@nytimes"

# Variables for holding sentiments
compound_list5 = []
positive_list5 = []
negative_list5 = []
neutral_list5 = []

# Loop through 10 pages of tweets (total 100 tweets)
for x in range(5):

    # Get all tweets from home feed
    public_tweets5 = api.user_timeline(target_user5)

    # Loop through all tweets
    for tweet in public_tweets5:

        # Run Vader Analysis on each tweet
        results5 = analyzer.polarity_scores(tweet["text"])
        compound = results5["compound"]
        pos = results5["pos"]
        neu = results5["neu"]
        neg = results5["neg"]

        # Add each value to the appropriate list
        compound_list5.append(compound)
        positive_list5.append(pos)
        negative_list5.append(neg)
        neutral_list5.append(neu)
        
        # Print the Averages
print(f"User: {target_user5}")
print(f"Compound: {np.mean(compound_list5):.3f}")
print(f"Positive: {np.mean(positive_list5):.3f}")
print(f"Neutral: {np.mean(neutral_list5):.3f}")
print(f"Negative: {np.mean(negative_list5):.3f}")

# Convert sentiments to DataFrame
nytimes_news = pd.DataFrame(
    {'Compound': compound_list5,
     'Positive': positive_list5,
     'Neutral': neutral_list5,
     'Negative': negative_list5
    })
nytimes_news["Tweets Ago"] =  range(len(nytimes_news))  
nytimes_news["name"] = "New York Times"
nytimes_news.head()

# Target User Account
target_user6 = "@theeconomist"

# Variables for holding sentiments
compound_list6 = []
positive_list6 = []
negative_list6 = []
neutral_list6 = []

# Loop through 10 pages of tweets (total 100 tweets)
for x in range(5):

    # Get all tweets from home feed
    public_tweets6 = api.user_timeline(target_user6)

    # Loop through all tweets
    for tweet in public_tweets6:

        # Run Vader Analysis on each tweet
        results6 = analyzer.polarity_scores(tweet["text"])
        compound = results6["compound"]
        pos = results6["pos"]
        neu = results6["neu"]
        neg = results6["neg"]

        # Add each value to the appropriate list
        compound_list6.append(compound)
        positive_list6.append(pos)
        negative_list6.append(neg)
        neutral_list6.append(neu)
        
        # Print the Averages
print(f"User: {target_user6}")
print(f"Compound: {np.mean(compound_list6):.3f}")
print(f"Positive: {np.mean(positive_list6):.3f}")
print(f"Neutral: {np.mean(neutral_list6):.3f}")
print(f"Negative: {np.mean(negative_list6):.3f}")

# Convert sentiments to DataFrame
economist_news = pd.DataFrame(
    {'Compound': compound_list6,
     'Positive': positive_list6,
     'Neutral': neutral_list6,
     'Negative': negative_list6
    })
economist_news["Tweets Ago"] =  range(len(economist_news))  
economist_news["name"] = "The Economist"
economist_news.head()

# Target User Account
target_user7 = "@msnbc"

# Variables for holding sentiments
compound_list7 = []
positive_list7 = []
negative_list7 = []
neutral_list7 = []

# Loop through 10 pages of tweets (total 100 tweets)
for x in range(5):

    # Get all tweets from home feed
    public_tweets7 = api.user_timeline(target_user7)

    # Loop through all tweets
    for tweet in public_tweets7:

        # Run Vader Analysis on each tweet
        results7 = analyzer.polarity_scores(tweet["text"])
        compound = results7["compound"]
        pos = results7["pos"]
        neu = results7["neu"]
        neg = results7["neg"]

        # Add each value to the appropriate list
        compound_list7.append(compound)
        positive_list7.append(pos)
        negative_list7.append(neg)
        neutral_list7.append(neu)
        
        # Print the Averages
print(f"User: {target_user7}")
print(f"Compound: {np.mean(compound_list7):.3f}")
print(f"Positive: {np.mean(positive_list7):.3f}")
print(f"Neutral: {np.mean(neutral_list7):.3f}")
print(f"Negative: {np.mean(negative_list7):.3f}")

# Convert sentiments to DataFrame
msnbc_news = pd.DataFrame(
    {'Compound': compound_list7,
     'Positive': positive_list7,
     'Neutral': neutral_list7,
     'Negative': negative_list7
    })
msnbc_news["Tweets Ago"] =  range(len(msnbc_news))  
msnbc_news["name"] = "MSNBC"
msnbc_news.head()

# Target User Account
target_user8 = "@abc"

# Variables for holding sentiments
compound_list8 = []
positive_list8 = []
negative_list8 = []
neutral_list8 = []

# Loop through 10 pages of tweets (total 100 tweets)
for x in range(5):

    # Get all tweets from home feed
    public_tweets8 = api.user_timeline(target_user8)

    # Loop through all tweets
    for tweet in public_tweets8:

        # Run Vader Analysis on each tweet
        results8 = analyzer.polarity_scores(tweet["text"])
        compound = results8["compound"]
        pos = results8["pos"]
        neu = results8["neu"]
        neg = results8["neg"]

        # Add each value to the appropriate list
        compound_list8.append(compound)
        positive_list8.append(pos)
        negative_list8.append(neg)
        neutral_list8.append(neu)
        
        # Print the Averages
print(f"User: {target_user8}")
print(f"Compound: {np.mean(compound_list8):.3f}")
print(f"Positive: {np.mean(positive_list8):.3f}")
print(f"Neutral: {np.mean(neutral_list8):.3f}")
print(f"Negative: {np.mean(negative_list8):.3f}")

# Convert sentiments to DataFrame
abc_news = pd.DataFrame(
    {'Compound': compound_list8,
     'Positive': positive_list8,
     'Neutral': neutral_list8,
     'Negative': negative_list8
    })
abc_news["Tweets Ago"] =  range(len(abc_news))  
abc_news["name"] = "ABC"
abc_news.head()

# Target User Account
target_user9 = "@newsweek"

# Variables for holding sentiments
compound_list9 = []
positive_list9 = []
negative_list9 = []
neutral_list9 = []

# Loop through 10 pages of tweets (total 100 tweets)
for x in range(5):

    # Get all tweets from home feed
    public_tweets9 = api.user_timeline(target_user9)

    # Loop through all tweets
    for tweet in public_tweets9:

        # Run Vader Analysis on each tweet
        results9 = analyzer.polarity_scores(tweet["text"])
        compound = results9["compound"]
        pos = results9["pos"]
        neu = results9["neu"]
        neg = results9["neg"]

        # Add each value to the appropriate list
        compound_list9.append(compound)
        positive_list9.append(pos)
        negative_list9.append(neg)
        neutral_list9.append(neu)
        
        # Print the Averages
print(f"User: {target_user9}")
print(f"Compound: {np.mean(compound_list9):.3f}")
print(f"Positive: {np.mean(positive_list9):.3f}")
print(f"Neutral: {np.mean(neutral_list9):.3f}")
print(f"Negative: {np.mean(negative_list9):.3f}")

# Convert sentiments to DataFrame
newsweek_news = pd.DataFrame(
    {'Compound': compound_list9,
     'Positive': positive_list9,
     'Neutral': neutral_list9,
     'Negative': negative_list9
    })
newsweek_news["Tweets Ago"] =  range(len(newsweek_news))  
newsweek_news["name"] = "NewsWeek"
newsweek_news.head()

# Target User Account
target_userz = "@washingtonpost"

# Variables for holding sentiments
compound_listz = []
positive_listz = []
negative_listz = []
neutral_listz = []

# Loop through 10 pages of tweets (total 100 tweets)
for x in range(5):

    # Get all tweets from home feed
    public_tweetsz = api.user_timeline(target_userz)

    # Loop through all tweets
    for tweet in public_tweetsz:

        # Run Vader Analysis on each tweet
        resultsz = analyzer.polarity_scores(tweet["text"])
        compound = resultsz["compound"]
        pos = resultsz["pos"]
        neu = resultsz["neu"]
        neg = resultsz["neg"]

        # Add each value to the appropriate list
        compound_listz.append(compound)
        positive_listz.append(pos)
        negative_listz.append(neg)
        neutral_listz.append(neu)

        
        # Print the Averages
print(f"User: {target_userz}")
print(f"Compound: {np.mean(compound_listz):.3f}")
print(f"Positive: {np.mean(positive_listz):.3f}")
print(f"Neutral: {np.mean(neutral_listz):.3f}")
print(f"Negative: {np.mean(negative_listz):.3f}")

# Convert sentiments to DataFrame
washpost_news = pd.DataFrame(
    {'Compound': compound_listz,
     'Positive': positive_listz,
     'Neutral': neutral_listz,
     'Negative': negative_listz
    })
washpost_news["Tweets Ago"] =  range(len(washpost_news))       
washpost_news["name"] = "Washington Post"
washpost_news

#create list of all dataframes
all_dfs = [bbc_news, cbs_news, cnn_news, fox_news, nytimes_news, economist_news, msnbc_news, abc_news, newsweek_news, washpost_news]

# Give all df's common column names
for df in all_dfs:
    df.columns = ['Compound', 'Negative', 'Neutral', 'Positive', 'Tweets Ago', 'Name']

#create new dataframe that combines all dataframes    
Major_News_Sentiment = pd.concat(all_dfs).reset_index(drop=True)
Major_News_Sentiment.head()
Major_News_Sentiment.to_csv("MajorNewsSentiment", encoding='utf-8', index=False)

mns_group = Major_News_Sentiment.groupby('Name')
mns_group.mean()

# Split up our data into groups based upon 'name'
news_names = Major_News_Sentiment.groupby('Name')

# Find out how many of each gender took bike trips
news_score = news_names['Compound'].mean()

# Chart our data, give it a title, and label the axes
news_chart = news_score.plot(kind="bar", title="Major Network Tweet Sentiment")
news_chart.set_xlabel("News Network")
news_chart.set_ylabel("Compound Sentiment Score")

fig.savefig('AggregateTweetSentiment.png')
plt.show()

# Plot and Formatting
x = Major_News_Sentiment['Tweets Ago']
y = Major_News_Sentiment['Compound']

fig, ax = plt.subplots()
for news in ['BBC', 'CBS', 'CNN', 'Fox', 'NYTimes', 'Economist', 'MSNBC', 'ABC', 'NewsWeek', 'WashPost']:
    ax.scatter(x, y, s=20, c=np.random.rand(3,), label=news,
               alpha=0.5, edgecolors='none')
plt.xlabel('Tweets Ago')
plt.ylabel('Compound Score')
ax.legend()
ax.grid(True)

fig2.savefig('TweetSentiment.png')
plt.show()

# 3 Observable Trends

# 1. New York Times' twitter account is relatively more negative than other news outlets
# 2. News Outlet twitter accounts tend to be negative in nature but to varying extents
# 3. ABC and The Economist represent the least negative news outlets when considering their last 100 tweets
