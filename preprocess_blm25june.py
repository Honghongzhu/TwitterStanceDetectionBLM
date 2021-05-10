import pandas as pd

blm25june_file = pd.read_csv('blm_25june.csv')

#remove unnecessary columns
blm25june_file.drop(['coordinates',
                     'media',
                     'urls',
                     'favorite_count',
                     'in_reply_to_screen_name',
                     'in_reply_to_user_id',
                     'place',
                     'possibly_sensitive',
                     'retweet_count',
                     'retweet_id',
                     'retweet_screen_name',
                     'source',
                     'tweet_url',
                     'user_created_at',
                     'user_screen_name',
                     'user_default_profile_image',
                     'user_description',
                     'user_favourites_count',
                     'user_followers_count',
                     'user_friends_count',
                     'user_listed_count',
                     'user_name',
                     'user_screen_name',
                     'user_statuses_count',
                     'user_time_zone',
                     'user_urls',
                     'user_verified',
                     'in_reply_to_status_id',
                     'user_screen_name.1'], axis=1, inplace=True)

#only keep rows with english
df = blm25june_file[blm25june_file.lang == 'en']

#remove duplicates
df_unique = df.drop_duplicates(subset=['text'],keep='first')

#remove enters
df_unique = df_unique.replace('\n',' ', regex=True)

#%%
df_unique.to_csv('blm_25june_unique.csv', encoding='utf-8-sig', index=False)

#%%
#only keep hashtag column and text column
df_unique.drop(['created_at','id','lang','user_location'], axis=1, inplace=True)

    
#%% Save as csv and keep emojis
df_unique.to_csv('blm_25june_tweets.csv', encoding='utf-8-sig', index=False)

#%%
print(df_unique.at[0, 'id'])