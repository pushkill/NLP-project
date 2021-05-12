import pandas as pd
import seaborn as sns
import numpy as np
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import string
import re
from collections import Counter
from tqdm import tqdm
tqdm.pandas()


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


cwd = os.getcwd()
df = pd.read_csv( os.path.join(cwd, 'full_dataset_all_labels.csv'))
stop_words=set(stopwords.words('english') + list(string.punctuation))
stop_words.add('rt') # add word rt (meaning retweet) to stop words
df = pd.read_csv('full_dataset_all_labels.csv')
#df = df.sample(10000)


def print_some_texts(columns, df):
    text_idxs = [47, 7240, 7241, 8013, 14500, 16500, 16304, 18300,  21750, 34036, 45159, 71920]
    for i in text_idxs:
        for column in columns:
            print(df[column].iloc[i])
#print_some_texts(['text'])

def tokenize(text):
    #print(text)
    text = preprocess_text(text)
    #print(text)
    tokens = word_tokenize(text)
    filtered_tokens = []
    # Filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation). (adapted from lab example)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if token not in stop_words and len(token) > 2:
                filtered_tokens.append(token)
    return filtered_tokens
    

def preprocess_text(text):
    text = re.sub(r"http\S+", " ", text)            # remove urls
    text = re.sub("@[A-Za-z0-9]+","", text)         # remove twitter handle
    text = re.sub("&amp;","", text)                  # &amp; is a special character for ampersand
    text = re.sub('<USER>', '', text)               # remove '<USER>' as there are some such strings as user or url is masked with this string
    text = re.sub('<URL>', '', text)
    text = text.lower() 
    text = re.sub('[^a-zA-Z]', ' ', text)           # Remove punctuations
    text = text.lower()                             # Convert to lowercase
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)#remove tags
    text = re.sub("(\\d|\\W)+"," ",text)            # remove special characters and digits
    return text
    
    
def stemming(tokens):
    stemmer = SnowballStemmer("english")
    stems = [stemmer.stem(token) for token in tokens]
    return stems

def lemmatizing(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas


df['preprocessed_text']=df['text'].apply(preprocess_text)


df['appended'] = df['preprocessed_text']+', this is '+df['label']


from transformers import BertTokenizer, BertModel
import torch


model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True,)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model
    
    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids
    
    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token
    
    """
    
    
    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]
    
    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return np.array(list_token_embeddings)


def bert_text_preparation(text, tokenizer):
    """Preparing the input for BERT
    
    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.
    
    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids
        
    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids
    
    
    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return pd.Series([tokenized_text, tokens_tensor, segments_tensors], index = ['tokenized_text', 'tokens_tensor', 'segments_tensors'])


labels = df['label'].unique()
dict_labels_len = {}
for l in labels:
    x = bert_text_preparation(l, tokenizer)
    dict_labels_len[l] = len(x[0])-2


df[['tokenized_text', 'tokens_tensor', 'segments_tensors']] = df['appended'].progress_apply(bert_text_preparation, tokenizer = tokenizer)


df['leng'] = df['tokens_tensor'].apply(lambda x: x.size()[1])

df = df[df['leng']  <= 512]
df.drop(['leng'], axis = 1, inplace=True)


df['bert_emmbeding'] = df.progress_apply(lambda x: get_bert_embeddings(x['tokens_tensor'], x['segments_tensors'], model), axis=1)


df.progress_apply(lambda x: x['tokenized_text'][- dict_labels_len[x['label']]-1:-1], axis = 1)


df['bert_emmbeding'] = df.progress_apply(lambda x: np.average(x['bert_emmbeding'][- dict_labels_len[x['label']]-1:-1], axis = 0), axis = 1)
df.sample()


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('stsb-mpnet-base-v2')


sentences = df['preprocessed_text'].to_numpy()
sentence_embeddings = model.encode(sentences)


df['bert_sentence_emm'] = sentence_embeddings.tolist()


df = pd.read_pickle('first.pkl')


import tensorflow as tf
import tensorflow_hub as hub


elmo = hub.load("https://tfhub.dev/google/elmo/3").signatures['default']


dfs = df.drop(['tokens_tensor', 'segments_tensors', 'tokenized_text', 'text'], axis = 1)
dfs = df.sample(10000)
dfs


lst =  df['preprocessed_text'].tolist()
lst2 = df['appended'].tolist()
embeddings_words = elmo(tf.constant(lst))["elmo"]
embeddings_sent = elmo(tf.constant(lst2))["default"]
embeddings_words


embeddings_words.numpy().shape


embeddings_sent.numpy().shape


df['elmo_sentence'] = embeddings_sent.numpy().tolist()


df['idx'] = df.progress_apply(lambda x: len(x['appended'].split()), axis = 1)
df['shape'] = df.progress_apply(lambda x: len(x['elmo_word']), axis = 1)
df['idx'] = df.progress_apply(lambda x: min(x['idx'], x['shape']), axis = 1)


df.tail(2)


df['elmo_word'] = embeddings_words.numpy().tolist()
df['elmo_word'] = df.progress_apply(lambda x: x['elmo_word'][x['idx']-1], axis = 1)


df = df.drop(['elmo_words', 'idx', 'shape'], axis = 1)


df


df.to_pickle('elmobert.pkl')
