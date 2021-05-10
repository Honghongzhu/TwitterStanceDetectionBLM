import pandas as pd

#read file
data = pd.read_csv('blm_corpus.csv')

#remove every column except 'message_id' which represents the Tweet ID
data.drop(['user_id','blacklivesmatter','alllivesmatter','bluelivesmatter'], axis=1, inplace=True)

#write file to csv again
data.to_csv('blm_data.csv', header=False, index=False)

#%%
import pandas as pd

data = pd.read_csv('blm_data.csv')

data.columns = ['ID']

#%%
print(data.loc[11790000: 11790020, :])

#print(data.at[11780000, 'ID'])

#%%
data_25april = data.loc[11460000: 11470000, :]
data_25april.to_csv('blm25april.csv', header=False, index=False)

#%%
data_25may= data.loc[11780000: 11790000, :]
data_25may.to_csv('blm25may.csv', header=False, index=False)
#%%
data_25june = data.loc[39050000: 39600000, :]
data_25june.to_csv('blm25june.csv', header=False, index=False)