import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
import nltk

# Corpora/stopwords not found when import nltk library download necesarry!!!
#nltk.download('stopwords')  
#nltk.download('punkt') same for these two to use tokeniser n .......
#nltk.download('wordnet')

#%matplotlib inline
#/kaggle/input/flipkart-products/flipkart_com-ecommerce_sample.csv

pre_df=pd.read_csv("..\\Dataset\\flipkart_data.csv", na_values=["No rating available"])

default_input = "FabHomeDecor Fabric Double Sofa Bed" # default if user inputs non existing product

user_input = input("\n ENTER THE PRODUCT FROM THE DATASET  \n\t:-")
if user_input not in pre_df.values: 
	user_input = default_input[:] # simply copying the defualt to x
    #print("\nTHE INPUT YOU GAVE IS NOT AVAILABLE IN DATASET!! REVERTING TO DEFAULT INPUT!\n")    
print( user_input )

#print(pre_df.head())
#pre_df.info()

#splits and removes the '>>'  and '[]' and ' "" ' (data cleaning!!)
pre_df['product_category_tree']=pre_df['product_category_tree'].map(lambda x:x.strip('[]'))
pre_df['product_category_tree']=pre_df['product_category_tree'].map(lambda x:x.strip('"'))
pre_df['product_category_tree']=pre_df['product_category_tree'].map(lambda x:x.split('>>'))


#simply deletes the unwated columns (add crawl_timestamp <diptayan sir > asked)
del_list=['crawl_timestamp','product_url','image',"retail_price","discounted_price","is_FK_Advantage_product","product_rating","overall_rating","product_specifications"]
pre_df=pre_df.drop(del_list,axis=1)

###removes words like (such as “the”, “a”, “an”, “in”)
from nltk.corpus import stopwords


from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) 
exclude = set(string.punctuation)
import string

#print(pre_df.head())
#print(pre_df.shape)

#dropping duplicate products <a lot of duplicates there so only same product> pops up again n again
smd=pre_df.copy()
smd.drop_duplicates(subset ="product_name", 
                     keep = "first", inplace = True)
#print(smd.shape)

print("\nRUMMAGING THE STOREROOM PLEASE WAIT!\n")
#lemmatise just brings the word to its root form eg: dogs->dog
#tokenising splits larger texts into smaller texts... refer tutorial point 
def filter_keywords(doc):
    doc=doc.lower()
    stop_free = " ".join([i for i in doc.split() if i not in stop_words])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    word_tokens = word_tokenize(punc_free)
    filtered_sentence = [(lem.lemmatize(w, "v")) for w in word_tokens]
    return filtered_sentence

#applying the filter <data cleaning! TBD>   

smd['product'] = smd['product_name'].apply(filter_keywords)
smd['description'] = smd['description'].astype("str").apply(filter_keywords)
smd['brand'] = smd['brand'].astype("str").apply(filter_keywords)

smd["all_meta"]=smd['product']+smd['brand']+ pre_df['product_category_tree']+smd['description']
smd["all_meta"] = smd["all_meta"].apply(lambda x: ' '.join(x))

#print(smd["all_meta"].head())

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['all_meta'])

#will give matrix of similarity
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#function for getting the desired output
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    product_indices = [i[0] for i in sim_scores]
    return titles.iloc[product_indices]

smd = smd.reset_index() #duplicates deletd hence indexing needs to be done again??
titles = smd['product_name']
indices = pd.Series(smd.index, index=smd['product_name'])

print(get_recommendations(user_input).head(50))
## take direct input from user at the start and pass it in recommendations
#run a check if its present in the dataframes or not! To be done june 11th

#change add seasonal sale ! search time_crawl! june 11th