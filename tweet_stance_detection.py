import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.svm import SVC
import sklearn.metrics
from sklearn.model_selection import train_test_split
from string import punctuation
import numpy as np
import pandas as pd
import emoji
import re
import string
from collections import Counter
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

#Download punkt and stopwords
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

#load dataset
def load_data(dir_name):
    return pd.read_csv(dir_name, encoding='utf-8')

def preprocess_data(data):
    # remove URLs
    data['tweet'] = data['tweet'].str.replace(r"http[s]?:[a-zA-Z._0-9/]+[a-zA-Z0-9]", "")
    # remove RT
    data['tweet'] = data['tweet'].str.replace(r"RT @[\w]*: ", "")
    # remove twitter mentions
    data['tweet'] = data['tweet'].str.replace(r"@[\w]*", "")
    # normalize whitespace
    data['tweet'] = data['tweet'].str.replace(r" {2,}", " ")
    # remove digits
    data['tweet'] = data['tweet'].str.replace(r"\d+", "")

def split_data(data):
    X_train, X_test, y_train, y_test = train_test_split(data['tweet'], data['label'], test_size=0.1, random_state=42) 
    return X_train, X_test, y_train, y_test

def extract_features(tweet):
    features = []
    tknzr = TweetTokenizer()
    ps = SnowballStemmer("english")
    
    # HASHTAGS COUNT
    list_hashtag = re.findall(r'\B(\#[a-zA-Z]+\b)(?!;)', tweet)
    features.append(len(list_hashtag))
    
    # PUNCTUATION COUNT & PRESENCE
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    count_punct = count(tweet, set(punctuation))
    features.append(count_punct)
    features.append(count_punct!=0)
    
    # EXCLAMATION MARK
    list_exclamation = re.findall(r'\!', tweet)
    features.append(len(list_exclamation))
    features.append(list_exclamation!=0)
    
    # QUESTION MARK
    list_question = re.findall(r'\?', tweet)
    features.append(len(list_question))
    features.append(list_question!=0)
    
    # REMOVE PUNCTUATION
    tweet = tweet.translate(str.maketrans('', '', punctuation))
    
    # EMOJIS COUNT & PRESENCE
    demojized_tweet = emoji.demojize(tweet)
    emoji_list = re.findall(r'(:[^:]*:)', demojized_tweet)    
    list_emoji = [emoji.emojize(x) for x in emoji_list]
    features.append(len(emoji_list))
    features.append(len(emoji_list)!=0)
    
    # REMOVE EMOJIS
    for emo in list_emoji:
        if emo in tweet:
            tweet = tweet.replace(emo, "")       
    
    # WORD COUNT &LENGTH: number of words, number of words without stopwords, average word length
    tokens = [x for x in tknzr.tokenize(tweet)]
    bag_of_words = [x for x in tokens if x.lower() not in stop_words] 
    stemmed_text = [ps.stem(x) for x in bag_of_words if x.lower() not in stop_words]
    features.append(len(bag_of_words))
    features.append(float(sum(map(len, bag_of_words))) / len(bag_of_words))
    
    # SPECIFIC WORD COUNT
    favor_words = ['blm', 'blacklivesmatter']
    against_words = ['alllivesmatter', 'bluelivesmatter']   
    favor = [x for x in bag_of_words if x.lower() in favor_words]
    against = [x for x in bag_of_words if x.lower() in against_words]
    features.append(len(favor))
    features.append(len(favor)/len(bag_of_words))
    features.append(len(favor)!=0)
    features.append(len(against))
    features.append(len(against)/len(bag_of_words))
    features.append(len(against)!=0)
    
    # WORD COUNT UPPER
    upper_words = [x for x in bag_of_words if x.isupper()]
    features.append(len(upper_words)!=0)
    
    # CHECK FOR ELONGATED WORDS
    regex = re.compile(r"(.)\1{2}")
    elongated_words = [word for word in bag_of_words if regex.search(word)]
    features.append(len(elongated_words))
    features.append(len(elongated_words)!=0)
    
    # POS TAGS: frequency of each POS tag
    pos_tags = nltk.pos_tag(stemmed_text) 
    counts = Counter(tag for word, tag in pos_tags)
    features.append(counts['JJ'])
    features.append(counts['RB'])
    features.append(counts['VB'])
    features.append(counts['NN'])
    
    #Letter features: number of letters, frequency of each letter, frequency of each capital letter
    features.append(len(tweet))
    for letter in list(string.ascii_lowercase):
        features.append(len([x for x in tweet if x.lower() == letter]))
    for letter in list(string.ascii_uppercase):
        features.append(len([x for x in tweet if x == letter]))
    
    total_upper = sum(1 for x in tweet if x.isupper())
    total_lower = sum(1 for x in tweet if x.islower())
    features.append(total_upper/len(tweet))
    features.append(total_lower/len(tweet))
    
    return features


def classify(train_features, train_labels, test_features):
    clf = SVC(kernel='linear')
    clf.fit(train_features, train_labels)
    return clf.predict(test_features)

def evaluate(y_true, y_pred):
    recall = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    print("Recall: %f" % recall)

    precision = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    print("Precision: %f" % precision)

    f1_score = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    print("F1-score: %f" % f1_score)

    return recall, precision, f1_score

def train(X_train, y_train, features):
    kfold = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=1) #shuffle=True, random_state=1
    scores = []
    #for train_ix, val_ix in kfold.split(X_train, y_train):
    for fold_id, (train_ix, val_ix) in enumerate(kfold.split(X_train, y_train)):
        # Print the fold number
        print("Fold %d" % (fold_id + 1))
    
        # Collect the data for this train/validation split
        train_features = [features[x] for x in train_ix]
        train_labels = y_train.iloc[train_ix]
        val_features = [features[x] for x in val_ix]
        val_labels = y_train.iloc[val_ix]
        
        # print the division of data sets
        train_0, train_1, train_2 = len(train_labels[train_labels==0]), len(train_labels[train_labels==1]), len(train_labels[train_labels==2])
        val_0, val_1, val_2 = len(val_labels[val_labels==0]), len(val_labels[val_labels==1]), len(val_labels[val_labels==2])
        print(">Train: 0=%d, 1=%d, 2=%d, Val: 0=%d, 1=%d, 2=%d" % (train_0, train_1, train_2, val_0, val_1, val_2))
        
    
        # Classify and add the scores to be able to average later
        y_pred = classify(train_features, train_labels, val_features)
        scores.append(evaluate(val_labels, y_pred))
    
        # Print a newline
        print("")
    
    # Print the averaged score
    recall = sum([x[0] for x in scores]) / len(scores)
    print("Averaged total recall", recall)
    precision = sum([x[1] for x in scores]) / len(scores)
    print("Averaged total precision", precision)
    f_score = sum([x[2] for x in scores]) / len(scores)
    print("Averaged total f-score", f_score)
    print("")
    
def create_ablation_plot():
    f_measure = [0.5907564022759513, 0.5878337577692379 , 0.5857376243002016, 0.586035442196902, 0.5832685076075813, 0.5859473722361368, 0.5831515972370076, 0.4884564020644004, 0.5774334676262038]
    groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    
    y_pos = np.arange(len(groups))
    plt.bar(y_pos, f_measure)
    plt.xticks(y_pos, groups)
    plt.title('Ablation analysis')
    plt.ylabel('F-measure')
    plt.xlabel('Removed feature group')
    plt.show()

def plot_conmat(y_test, y_pred):
    cm = metrics.confusion_matrix(y_test, y_pred)
    ax = plt.axes()
    sns.heatmap(cm, annot=True, cmap='Blues')
    ax.set_title('Confusion matrix')
    plt.show()

def main():
    #train_data, val_data, test_data = load_data('all_tweets.csv')
    data = load_data("all_tweets.csv")
    preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data)
    features = list(map(extract_features, X_train))
    train(X_train, y_train, features)
    
    create_ablation_plot()
    
    #Test
    test_features = list(map(extract_features, X_test))
    y_pred = classify(features, y_train, test_features)
    evaluate(y_test, y_pred)
    plot_conmat(y_test, y_pred)
    
    # Retrieve list of files that are wrongly classified for failure analysis
    for test_data, prediction, label in zip(X_test, y_pred, y_test):
        if prediction != label:
            print(test_data, 'has been classified as ', prediction, 'and should be ', label)
    
if __name__ == '__main__':
    main()
