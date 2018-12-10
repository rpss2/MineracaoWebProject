import pandas as pd
import numpy as np
import json
import re

from sklearn.feature_extraction.text import CountVectorizer#, TfidfTransformer
from sklearn.neighbors import NearestNeighbors
# from sklearn import neighbors
# from sklearn.metrics import classification_report

# from sklearn.metrics import accuracy_score
# from sklearn.metrics import mean_squared_error

with open('Data/reviews_Clothing_Shoes_and_Jewelry_5.json') as f:
    data = json.load(f)

df = pd.DataFrame.from_dict(data, orient='columns')

count = df.groupby('asin', as_index=False).count()
mean = df.groupby('asin', as_index=False).mean()
df_merged = pd.merge(df, count, how='right', on=['asin'])

df_merged['totalReviewers'] = df_merged['reviewerID_y']
df_merged["overallScore"] = df_merged["overall_x"]
df_merged["summaryReview"] = df_merged["summary_x"]

df_new = df_merged[['asin','summaryReview','overallScore',"totalReviewers"]]

df_merged = df_merged.sort_values(by='totalReviewers', ascending=False)
df_select = df_merged[df_merged.totalReviewers >= 10]

df_product_reviews = df_select.groupby('asin')['summaryReview'].apply(list)
df_product_reviews = pd.DataFrame(df_product_reviews)

df_mean = pd.merge(df_product_reviews, mean, on='asin', how='inner')
df_mean = df_mean[['asin', 'summaryReview', 'overall']]

regEx = re.compile('[^a-z]+')
def cleanReviews(reviewText):
    value = []
    for i in range(len(reviewText)):
        value.append(reviewText[i].lower())
        value[i] = regEx.sub(' ', value[i]).strip()
    return value

df_mean['summaryClean'] = df_mean['summaryReview'].apply(cleanReviews)

reviews = df_mean['summaryClean']
reviews = reviews.reset_index().summaryClean
countVector = CountVectorizer(max_features = 300, stop_words='english') 
corpus = []
for i in range(len(reviews)):
    string = ' '.join(reviews[i])
    corpus.append(string)
transformedReviews = countVector.fit_transform(corpus)

dfReviews = pd.DataFrame(transformedReviews.A, columns=countVector.get_feature_names())
dfReviews = dfReviews.astype(int)

X = np.array(dfReviews)
tpercent = 0.85
tsize = int(np.floor(tpercent * len(dfReviews)))
dfReviews_train = X[:tsize]
dfReviews_test = X[tsize:]
#len of train and test
lentrain = len(dfReviews_train)
lentest = len(dfReviews_test)

neighbor = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(dfReviews_train)

def calc_neighbors(element):
    a = neighbor.kneighbors([element])
    list_similar = a[1]
    
    first_similar = [item[0] for item in list_similar]
    first_similar = int(str(first_similar).strip('[]'))

    second_similar = [item[1] for item in list_similar]
    second_similar = int(str(second_similar).strip('[]'))
    
    third_similar = [item[2] for item in list_similar]
    third_similar = int(str(third_similar).strip('[]'))
    
    fourth_similar = [item[3] for item in list_similar]
    fourth_similar = int(str(fourth_similar).strip('[]'))
    
    return [first_similar, second_similar, third_similar, fourth_similar]