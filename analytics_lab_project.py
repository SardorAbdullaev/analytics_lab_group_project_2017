
# coding: utf-8

# # Identifying Duplicate Questions
#  
# Welcome to the Quora Question Pairs competition! Here, our goal is to identify which questions asked on [Quora](https://www.quora.com/), a quasi-forum website with over 100 million visitors a month, are duplicates of questions that have already been asked. This could be useful, for example, to instantly provide answers to questions that have already been answered. We are tasked with predicting whether a pair of questions are duplicates or not, and submitting a binary prediction against the logloss metric.
# 

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

pal = sns.color_palette()
#Pickle for later use
#Dumping:
#from cPickle import dump
#output = open('t2.pkl', 'wb')
#dump(t2, output, -1)
#output.close()
#Loading:
#from cPickle import load
#input = open('t2.pkl', 'rb')
#tagger = load(input)
#input.close()


# 
# ## Training set
# 
# 

# In[2]:

df_train = pd.read_csv('/Users/tiziankronsbein/Desktop/Quora/train.csv')
df_train.head()


# We are given a minimal number of data fields here, consisting of:
# 
# **`id`:** Looks like a simple rowID    
# **`qid{1, 2}`:** The unique ID of each question in the pair    
# **`question{1, 2}`:** The actual textual contents of the questions.    
# **`is_duplicate`:** The **label** that we are trying to predict - whether the two questions are duplicates of each other.
# 

# In[3]:

print('Total number of question pairs for training: {}'.format(len(df_train)))
print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean()*100, 2)))
qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
print('Total number of questions in the training data: {}'.format(len(
    np.unique(qids))))
print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))

plt.figure(figsize=(12, 5))
plt.hist(qids.value_counts(), bins=50)
plt.yscale('log', nonposy='clip')
plt.title('Log-Histogram of question appearance counts')
plt.xlabel('Number of occurences of question')
plt.ylabel('Number of questions')
print()


# 
# In terms of questions, everything looks as I would expect here. Most questions only appear a few times, with very few questions appearing several times (and a few questions appearing many times). One question appears more than 160 times, but this is an outlier.
#  
# We can see that we have a 37% positive class in this dataset. Since we are using the [LogLoss](https://www.kaggle.com/wiki/LogarithmicLoss) metric, and LogLoss looks at the actual predicts as opposed to the order of predictions, we should be able to get a decent score by creating a submission predicting the mean value of the label.
#  
# ## Test Submission

# In[4]:

from sklearn.metrics import log_loss

p = df_train['is_duplicate'].mean() # Our predicted probability
print('Predicted score:', log_loss(df_train['is_duplicate'], np.zeros_like(df_train['is_duplicate']) + p))

df_test = pd.read_csv('/Users/tiziankronsbein/Desktop/Quora/test.csv')
sub = pd.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': p})
sub.to_csv('naive_submission.csv', index=False)
sub.head()


#  **0.66 on the leaderboard! Score!**
# 
#  However, not all is well. The discrepancy between our local score and the LB one indicates that the distribution of values on the leaderboard is very different to what we have here, which could cause problems with validation later on in the competition.
#  
#  According to this [excellent notebook by David Thaler](www.kaggle.com/davidthaler/quora-question-pairs/how-many-1-s-are-in-the-public-lb/notebook), using our score and submission we can calculate that we have about 16.5% positives in the test set. This is quite surprising to see, so it'll be something that will need to be taken into account in machine learning models.
#  
#  Next, I'll take a quick peek at the statistics of the test data before we look at the text itself.

# ## Test Set

# In[5]:

df_test = pd.read_csv('/Users/tiziankronsbein/Desktop/Quora/test.csv')
df_test.head()


print('Total number of question pairs for testing: {}'.format(len(df_test)))


# Nothing out of the ordinary here. We are once again given rowIDs and the textual data of the two questions. It is worth noting that we are not given question IDs here however for the two questions in the pair.
#  
# It is also worth pointing out that the actual number of test rows are likely to be much lower than 2.3 million. According to the [data page](https://www.kaggle.com/c/quora-question-pairs/data), most of the rows in the test set are using auto-generated questions to pad out the dataset, and deter any hand-labelling. This means that the true number of rows that are scored could be very low.
#  
# We can actually see in the head of the test data that some of the questions are obviously auto-generated, as we get delights such as "How their can I start reading?" and "What foods fibre?". Truly insightful questions.
#  
# Now onto the good stuff - the text data!
# 
# 

# # Text Analysis

# First off, some quick histograms to understand what we're looking at. **Most analysis here will be only on the training set, to avoid the auto-generated questions**
# 
# 
# 
# 

# In[6]:

train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

dist_train = train_qs.apply(len)
dist_test = test_qs.apply(len)
plt.figure(figsize=(15, 10))
plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], normed=True, label='train')
plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='test')
plt.title('Normalised histogram of character count in questions', fontsize=15)
plt.legend()
plt.xlabel('Number of characters', fontsize=15)
plt.ylabel('Probability', fontsize=15)

print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 
                          dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))


# In[7]:

dist_train = train_qs.apply(lambda x: len(x.split(' ')))
dist_test = test_qs.apply(lambda x: len(x.split(' ')))

plt.figure(figsize=(15, 10))
plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], normed=True, label='train')
plt.hist(dist_test, bins=50, range=[0, 50], color=pal[1], normed=True, alpha=0.5, label='test')
plt.title('Normalised histogram of word count in questions', fontsize=15)
plt.legend()
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Probability', fontsize=15)

print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 
                          dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))


# We can see that most questions have anywhere from 15 to 150 characters in them. 
# It seems that the test distribution is a little different from the train one, 
# but not too much so (I can't tell if it is just the larger data reducing noise, 
# but it also seems like the distribution is a lot smoother in the test set).
#  
# One thing that catches my eye is the steep cut-off at 150 characters for the training set, 
# for most questions, while the test set slowly decreases after 150. Could this be some sort 
# of Quora question size limit?
#  
# It's also worth noting that I've truncated this histogram at 200 characters, and that the
# max of the distribution is at just under 1200 characters for both sets - although samples 
# with over 200 characters are very rare.
# 
# Let's do the same for word count. I'll be using a naive method for splitting words 
# (splitting on spaces instead of using a serious tokenizer), although this should still give 
# us a good idea of the distribution.

# We see a similar distribution for word count, with most questions being about 10 words long. It looks to me like the distribution of the training set seems more "pointy", while on the test set it is wider. Nevertheless, they are quite similar.
#  
# So what are the most common words? Let's take a look at a word cloud.

# In[8]:

from wordcloud import WordCloud
cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')



# # Sementic Analysis

# Next, I will take a look at usage of different punctuation in questions - this may form a basis for some interesting features later on.

# In[10]:

qmarks = np.mean(train_qs.apply(lambda x: '?' in x))
math = np.mean(train_qs.apply(lambda x: '[math]' in x))
fullstop = np.mean(train_qs.apply(lambda x: '.' in x))
capital_first = np.mean(train_qs.apply(lambda x: x[0].isupper()))
capitals = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))
numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))

print('Questions with question marks: {:.2f}%'.format(qmarks * 100))
print('Questions with [math] tags: {:.2f}%'.format(math * 100))
print('Questions with full stops: {:.2f}%'.format(fullstop * 100))
print('Questions with capitalised first letters: {:.2f}%'.format(capital_first * 100))
print('Questions with capital letters: {:.2f}%'.format(capitals * 100))
print('Questions with numbers: {:.2f}%'.format(numbers * 100))


# ## Normalization

# ### Stemming and Lemmatizing

# In[ ]:

#TODO Delete this line in production
df_train = df_train.head()

def extractStemList(x):
    porter = nltk.PorterStemmer()
    return [porter.stem(t) for t in word_tokenize(x)]

def extractLemmaList(x):
    wnl = nltk.WordNetLemmatizer()
    return [wnl.lemmatize(t) for t in word_tokenize(x)]

print(df_train)


# ### Tokenization

# In[ ]:

from nltk import word_tokenize
import nltk
for i in range(1,3):
    df_train['question1_stemmed'] = df_train['question1'].apply(extractStemList)
    df_train['question1_lemmed'] = df_train['question1'].apply(extractLemmaList)
    
for i in range(1,3):
    df_train['question2_stemmed'] = df_train['question2'].apply(extractStemList)
    df_train['question2_lemmed'] = df_train['question2'].apply(extractLemmaList)

df_train.head()


# ### POS Tagging

# In[ ]:

#some help funcs
#nltk.help.upenn_tagset('RB')
#nltk.name.readme()

def extractPOSList(x):
    return nltk.pos_tag(word_tokenize(x))

df_train['question1_tagged'] = df_train['question1'].apply(extractPOSList) 
df_train['question2_tagged']  = df_train['question2'].apply(extractPOSList)

print(df_train)


# ### Chunking 

# Now that we know the parts of speech, we can do what is called chunking, and group words into hopefully meaningful chunks. One of the main goals of chunking is to group into what are known as "noun phrases." These are phrases of one or more words that contain a noun, maybe some descriptive words, maybe a verb, and maybe something like an adverb. The idea is to group nouns with the words that are in relation to them.
# 
# In order to chunk, we combine the part of speech tags with regular expressions. Mainly from regular expressions, we are going to utilize the following:

# In[7]:

#first we need to define some rules
import nltk
from nltk import word_tokenize, pos_tag
grammar = r"""
Gerunds: {<DT>?<NN>?<VBG><NN>}
Coordinated noun: {<NNP><CC><NNP>|<DT><PRP\$><NNS><CC>
<NNS>|<NN><NNS> <CC><NNS>} """

cp = nltk.RegexpParser(grammar)

for i in df_train['question1_tagged']:
   tree = cp.parse(pos_tag(word_tokenize(df_train['question1_tagged'])))
   for subtree in tree.subtrees():
     if subtree.label()=='Gerunds': print(subtree)

print(cp.parse(pos_tag(word_tokenize(df_train['question1_tagged']))))


# ### Manual POS Tagging Adjustment

# In[ ]:

from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

patterns = [
	(r'.*ing$', 'VBG'), # gerunds
	(r'.*ed$', 'VBD'), # simple past
	(r'.*es$', 'VBZ'), # 3rd singular present
	(r'.*ould$', 'MD'), # modals
	(r'.*\'s$', 'NN$'), # possessive nouns
	(r'.*s$', 'NNS'), # plural nouns
	(r'^-?[0-9]+(.[0-9]+)?$', 'CD'), # cardinal numbers
	(r'.*', 'NN') # nouns (default)
]
regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(brown_sents[3])
regexp_tagger.evaluate(brown_tagged_sents)
i=0
np.sum([value[1]==brown_tagged_sents[3][i][1] for i,value in enumerate(nltk.pos_tag(brown_sents[3])) ])/len(brown_sents[3])


# In[ ]:


#for key,value in brown_tagged_sents[3]:
 #   print("%10s %10s" % (key, value))
    
#brown_tagged_sents[3] == nltk.pos_tag(brown_sents[3])


# ### WordNet

# In[ ]:

#nltk.app.wordnet()

#right = wn.synset('right_whale.n.01')
#orca = wn.synset('orca.n.01')
#minke = wn.synset('minke_whale.n.01')
#tortoise = wn.synset('tortoise.n.01')
#novel = wn.synset('novel.n.01')
#right.lowest_common_hypernyms(minke)
#[Synset('baleen_whale.n.01')]
#right.lowest_common_hypernyms(orca)
#[Synset('whale.n.02')]
#right.lowest_common_hypernyms(tortoise)
#[Synset('vertebrate.n.01')]
#right.lowest_common_hypernyms(novel)
#[Synset('entity.n.01')]
#right.path_similarity(minke)


# ### Named Entity Extraction

# In[ ]:

from nltk.tag import StanfordNERTagger

#st = StanfordNERTagger() #TODO
nltk.ne_chunk(df_train['question1'].apply(extractPOSList)[0],binary=True)



# # Initial Feature Analysis

# Before we create a model, we should take a look at how powerful some features are. I will start off with the word share feature from the benchmark model.
# 

# In[ ]:

from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

plt.figure(figsize=(15, 5))
train_word_match = df_train.apply(word_match_share, axis=1, raw=True)
plt.hist(train_word_match[df_train['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')
plt.hist(train_word_match[df_train['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Label distribution over word_match_share', fontsize=15)
plt.xlabel('word_match_share', fontsize=15)


# Here we can see that this feature has quite a lot of predictive power, as it is good at separating the duplicate questions from the non-duplicate ones. Interestingly, it seems very good at identifying questions which are definitely different, but is not so great at finding questions which are definitely duplicates.
# 
# 

# ## RTE

# In[ ]:

def rte_features(rtepair):
    extractor = nltk.RTEFeatureExtractor(rtepair)
    features = {}
    features['word_overlap'] = len(extractor.overlap('word'))
    features['word_hyp_extra'] = len(extractor.hyp_extra('word'))
    features['ne_overlap'] = len(extractor.overlap('ne'))
    features['ne_hyp_extra'] = len(extractor.hyp_extra('ne'))
    return features

rtepair = nltk.corpus.rte.pairs(['rte3_dev.xml'])[33]
extractor = nltk.RTEFeatureExtractor(rtepair)


# In[ ]:

print(nltk.corpus.rte.pairs(['rte3_dev.xml'])[33])


# ## TF - IDF

# I'm now going to try to improve this feature, by using something called TF-IDF (term-frequency-inverse-document-frequency). This means that we weigh the terms by how **uncommon** they are, meaning that we care more about rare words existing in both questions than common one. This makes sense, as for example we care more about whether the word "exercise" appears in both than the word "and" - as uncommon words will be more indicative of the content.
# 
# You may want to look into using sklearn's [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) to compute weights if you are implementing this yourself, but as I am too lazy to read the documentation I will write a version in pure python with a few changes which I believe should help the score.
# 
# 

# In[ ]:

from collections import Counter

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}


# In[ ]:

print('Most common words and weights: \n')
print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
print('\nLeast common words and weights: ')
(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])


# In[ ]:

def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


# In[ ]:

plt.figure(figsize=(15, 5))
tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)
plt.hist(tfidf_train_word_match[df_train['is_duplicate'] == 0].fillna(0), bins=20, normed=True, label='Not Duplicate')
plt.hist(tfidf_train_word_match[df_train['is_duplicate'] == 1].fillna(0), bins=20, normed=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Label distribution over tfidf_word_match_share', fontsize=15)
plt.xlabel('word_match_share', fontsize=15)


# In[ ]:

from sklearn.metrics import roc_auc_score
print('Original AUC:', roc_auc_score(df_train['is_duplicate'], train_word_match))
print('   TFIDF AUC:', roc_auc_score(df_train['is_duplicate'], tfidf_train_word_match.fillna(0)))



# So it looks like our TF-IDF actually got _worse_ in terms of overall AUC, which is a bit disappointing. (I am using the AUC metric since it is unaffected by scaling and similar, so it is a good metric for testing the predictive power of individual features.
#  
# However, I still think that this feature should provide some extra information which is not provided by the original feature. Our next job is to combine these features and use it to make a prediction. For this, I will use our old friend XGBoost to make a classification model.
#  
# ## Rebalancing the Data
# 
# However, before I do this, I would like to rebalance the data that XGBoost receives, since we have 37% positive class in our training data, and only 17% in the test data. By re-balancing the data so our training set has 17% positives, we can ensure that XGBoost outputs probabilities that will better match the data on the leaderboard, and should get a better score (since LogLoss looks at the probabilities themselves and not just the order of the predictions like AUC).
# 
# First we create our training and test data:
# 
# 

# In[ ]:

x_train = pd.DataFrame()
x_test = pd.DataFrame()
x_train['word_match'] = train_word_match
x_train['tfidf_word_match'] = tfidf_train_word_match
x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)

y_train = df_train['is_duplicate'].values


# In[ ]:

pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]


# Now we oversample the negative class.
# There is likely a much more elegant way to do this...

# In[ ]:

p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train) / (len(pos_train) + len(neg_train)))

x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train


# Finally, we split some of the data off for validation

# In[ ]:

from sklearn.cross_validation import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)


# ## XGBoost
#  
# Now we can finally run XGBoost on our data, in order to see the score on the leaderboard!

# In[ ]:

import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)


# In[ ]:

d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('simple_xgb.csv', index=False)

