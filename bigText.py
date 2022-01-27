
path = '/content/drive/MyDrive/text2gif/'

import time
import nltk
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gc
from sklearn.preprocessing import minmax_scale

from gensim import corpora
import re

import math

#nltk.download('stopwords')

"""##Getting vectors for each tag"""

def tags_to_vec(tag, model):
    tt = nltk.TweetTokenizer()
    tags = ast.literal_eval(tag)
    words = []
    for i in tags:
        word = tt.tokenize(i)
        for j in word:
            words.append(j)
        
    blank = np.array(model['the']).shape[0]
    sent_vector = 0
    stop = set(nltk.corpus.stopwords.words("english"))
    words = [i for i in words if i not in stop]
    if len(words) == 0:
        words = tags
    for word in tags:
        if word not in model:
            word_vector = np.array(np.random.uniform(-1.0, 1.0, blank))
            model[word] = word_vector
        else:
            word_vector = model[word]
        sent_vector = sent_vector + word_vector
    return sent_vector

def get_sentence_vector(sentence, model):
    blank = np.array(model['the']).shape[0]
    sent_vector = 0
    stop = set(nltk.corpus.stopwords.words("english"))
    tt = nltk.TweetTokenizer()
    words = [i for i in tt.tokenize(sentence) if i not in stop]
    if len(words) == 0:
        words = tt.tokenize(sentence)
    for word in words:
        if word not in model:
            word_vector = np.array(np.random.uniform(-1.0, 1.0, blank))
            model[word] = word_vector
        else:
            word_vector = model[word]
        sent_vector = sent_vector + word_vector
    return sent_vector

def cosineValue(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    
    if type(v1) == int or type(v2) == int or v2 == [0.0]:
        return 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return float(sumxy/math.sqrt(sumxx*sumyy))
        #return dot(v1, v2)/(norm(v1)*norm(v2))

def cosine_sim(sentence, tags):
    vec1 = sentence
    vec2 = read_tags(tags)
    return cosineValue(vec1, vec2)

def read_tags(vec):
    tags = vec.replace('[','').replace(']','')
    tags = re.split(r"\s|(?<!\d)[,.]|[,.](?!\d)|[\n]", tags)
    i = 0
    new = []
    while i < len(tags):
        if tags[i] == "":
            tags.pop(i)
        else:
            new.append(float(tags[i]))
            i += 1
    return new

def tokenize_tweet(tweet):
    """
    Uses nltk's tweet tokenizer to get tokens from tweet,
    then converts back into string format so more preprocessing can be done
    """
    tt = nltk.TweetTokenizer()
    tokens = tt.tokenize(tweet)
    tokenized_tweet = " ".join(tokens)
    
    return tokenized_tweet

def lemmatize_tweet(tweet):
    from nltk import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tweet.split()]
    lemmatized_tweet = " ".join(lemmatized_tokens)

    return lemmatized_tweet

def preprocess_tweets(X_batch, y_batch):
    """
    preprocesses tweets using our tokenize_tweets function and returns data in a
    tensor format
    """
    # tokenization and lemmatization
    X_batch['content'] = X_batch.content.apply(tokenize_tweet)
    X_batch['content'] = X_batch.content.apply(lemmatize_tweet)
    #print(X_batch['content'].values)
    # splits our strings into a tensorflow readable format
    X_batch = tf.constant(X_batch.content, shape=[len(X_batch)])
    #print(X_batch)
    X_batch = tf.strings.split(X_batch)
    
    # converts to tensor data format
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch


def encode_tweets(X_batch, y_batch, table):
    return table.lookup(X_batch), y_batch

def build_dataset(X_batch, y_batch, table, batch_size=32):
    X_batch, y_batch = preprocess_tweets(X_batch, y_batch)
    X_batch, y_batch = encode_tweets(X_batch, y_batch, table)
    
    y_batch['sentiment'] = y_batch.sentiment.astype("category")
    y_batch = tf.one_hot(indices = y_batch.sentiment.cat.codes.values,
                        depth=7)

    combined_dataset = tf.data.Dataset.from_tensor_slices((X_batch, y_batch))
    combined_dataset = combined_dataset.shuffle(buffer_size=1024).batch(batch_size)
    combined_dataset = combined_dataset.prefetch(1)
    
    return combined_dataset, X_batch, y_batch

def get_score_df(string, model):
    data = pd.read_csv('tag_df.csv')
    '''data = pd.read_csv('gifgif_scores.csv')
    data = data.drop(columns=['Unnamed: 0'])
    data['tag_vecs'] = data.tags.apply(tags_to_vec)'''
    #data['input'] = [[get_sentence_vector(string, model)]]
    vec = get_sentence_vector(string, model)
    data['cos'] = data.apply(lambda x: cosine_sim(vec, x["tag_vecs"]), axis = 1)
    data[['cos']] = minmax_scale(data[['cos']])
    return data

def gif_link(content, cID):
    return "https://giphy.com/media/{}/giphy.gif?cid={}".format(content, cID)


def text_to_emotions(string, model, glove, table, emotion_dict):

    emotion_map = {
        'neutral' : 'neutral',
        'worry' : 'fear',
        'happiness' : 'happiness',
        'sadness' : 'sadness',
        'surprise' : 'surprise',
        'hate' : 'anger',
        'love' : 'contentment'
    }
    ts = time.time()

    score_df = get_score_df(string, glove)
    score_df['link'] = score_df.apply(lambda x: gif_link(x["content"], x["cID"]), axis = 1)

    model_time = time.time() - ts
    print("Score_df calculated in {} seconds".format(model_time))
    
    X = pd.DataFrame.from_dict({'content' : [string]})
    y = pd.DataFrame.from_dict({'sentiment' : ['']})
    comb, X, y = build_dataset(X, y, table)
    pred = model.predict(np.array(X,ndmin=2))[0]
    emotion_scores = {}
    for i in range(len(emotion_dict.keys())):
        mapped = emotion_map[list(emotion_dict.values())[i]]
        emotion_scores.update({mapped:pred[i].round(2)})
    return emotion_scores, score_df

##############################################

def big(text, glove_model):
    big_time = time.time()




    truncated_vocabulary = pickle.load(open("vocab.p", "rb"))

    """We will also replace the actual words with their word indexes for performance"""
    ts = time.time()
    words = tf.constant(truncated_vocabulary)
    word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
    vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
    num_oov_buckets = 1000
    table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)
    

    table.lookup(tf.constant(b'me')).numpy()

    model_time = time.time() - ts
    print("Table stuff loaded in {} seconds".format(model_time))

    #BUILD MODEL

    ts = time.time()
    rnn_model = tf.keras.models.load_model('rnn_model.h5')
    model_time = time.time() - ts
    print("RNN loaded in {} seconds".format(model_time))
    gc.collect()




    #data['sentiment'] = data['sentiment'].map(emotion_map)
    emotion_dict = pickle.load( open( "emotion_dict.p", "rb" ) )


    emotion_scores, score_df = text_to_emotions(text, rnn_model, glove_model, table, emotion_dict)
    print(emotion_scores)


    score_df = score_df.drop(columns=['amusement','contempt','disgust','embarrassment','excitement','guilt','pleasure','pride','relief','satisfaction','shame'])
    score_df['neutral'] = 0.5

    for emotion, score in emotion_scores.items():
        score_df[emotion] = score_df[emotion].mul(score).mul(score_df['cos'])

    score_df['max_score'] = score_df[['anger', 'contentment', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']].max(axis=1)


    score_df = score_df.sort_values('max_score', ascending=False)

    final_time = time.time() - big_time
    print("Program run in {} seconds".format(final_time))
    return score_df, emotion_scores


