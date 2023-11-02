# Import necessary libraries and modules
import pandas as pd
from copy import deepcopy
from joblib import load, dump
from statistics import mean
from collections import Counter
import sys

# Add the './utils' directory to the sys path for custom utilities
sys.path.append('./utils')
from utils import review_feature
import argparse
import time

# Initialize review_feature object
rf = review_feature()

# Record the start time for measuring execution time
start = time.time()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Ranking Reviews')
parser.add_argument('--spell_threshold', action='store', type=int, default=0.9, help='Spell checking threshold')
parser.add_argument('--model_path', action='store', type=str, default='randomforest.joblib', help='Model Path')
parser.add_argument('--file_name', action='store', type=str, default='data/test.csv',
                    help='File to Rank (Product, Review)')
parser.add_argument('--testing', action='store', type=str, default='False', help='Get Ranking Score Test or Not (True/False)')
args = parser.parse_args()

# Load the classifier model
classifier = load(args.model_path)

# Read data from the specified CSV file
df = pd.read_csv(args.file_name)
print(df.head(5))

# Initialize sets to store different types of bad reviews
bad_reviews = set()
language_error = set()
gibberish = set()
swear = set()
company_tag = set()

# Iterate through the rows of the DataFrame
for indx in df.index:
    review = df.at[indx, 'answer_option']

    # Language Detection
    try:
        b = rf.language_detection(review)
        if b == 'hi' or b == 'mr':
            language_error.add(indx)
    except:
        language_error.add(indx)

    # Gibberish Detection
    if rf.gibberish_detection(review, prefix_path='utils'):
        gibberish.add(indx)

    # Swear Words Check
    if rf.english_swear_check(review) or rf.hindi_swear_check(review):
        swear.add(indx)

    # Identify reviews on Competitive Brands
    if rf.competitive_brand_tag(review):
        company_tag.add(indx)

# Print the counts of different types of bad reviews
print("Number of Bad Reviews for Language Error: {} \n Number of Bad Reviews for Gibberish: {} \n Number of Bad Reviews for Swear: {} \n Number of Bad Reviews for Competitive Brand: {}".format(len(language_error), len(gibberish), len(swear), len(company_tag)))

# Combine all types of bad reviews into a single list
bad_reviews = list(bad_reviews.union(swear, company_tag, gibberish, language_error))

# Print the deleted reviews
print("DELETED REVIEWS: \n", df[df.index.isin(bad_reviews)])

# Remove bad reviews from the DataFrame
df = df[~df.index.isin(bad_reviews)].reset_index(drop=True)

# Sort the DataFrame by the 'product' column
df = df.sort_values(by=['product'], ignore_index=True)

# Add additional columns for review length and sentiment analysis scores
df['review_len'] = df['answer_option'].apply(lambda x: len(x.split()))
df['Rn'] = 0.0
df['Rp'] = 0.0
df['Rs'] = 0.0
df['Rc'] = 0.0
df['Rd'] = 0.0
df['Rsc'] = 0.0

# Get the unique list of products in the DataFrame
product_list = df['product'].unique()

# Iterate through each product
for product in product_list:
    data = df[df['product'] == product]
    unique_bag = set()

    # Calculate a unique set of words for each product
    for review in data['answer_option']:
        review = review.lower()
        words = review.split()
        unique_bag = unique_bag.union(set(words))

    # Calculate various scores for each review
    for indx in data.index:
        review = data.at[indx, 'answer_option']
        df.at[indx, 'Rp'] = rf.polarity_sentiment(review)
        df.at[indx, 'Rs'] = rf.subjectivity_sentiment(review)
        df.at[indx, 'Rd'] = rf.service_tag(review)
        df.at[indx, 'Rsc'] = rf.slang_emoji_polarity_compoundscore(review)
        df.at[indx, 'Rc'] = float(len(set(review.split()))) / float(len(unique_bag))

    # Calculate noun scores for reviews in each product category
    df.loc[df['product'] == product, 'Rn'] = rf.noun_score(data['answer_option'].values).values

# Create columns for 'win', 'lose', and 'review_score'
product_list = df['product'].unique()
df['win'] = 0
df['lose'] = 0
df['review_score'] = 0.0
df.reset_index(inplace=True, drop=True)

# Define a function to give scores to reviews
def score_giver(C, D):
    E = pd.merge(C, D, how='outer', on='j')
    E.drop(columns=['j'], inplace=True)
    q = classifier.predict(E.values)
    return Counter(q)

# Iterate through products to calculate scores
for i, product in enumerate(product_list):
    data = df[df['product'] == product]
    for indx in data.index:
        review = df.loc[indx, ['review_len', 'Rn', 'Rp', 'Rs', 'Rc', 'Rd', 'Rsc']]
        review['j'] = 'jn'
        C = pd.DataFrame([review])
        D = data[data.index != indx].loc[:, ['review_len', 'Rn', 'Rp', 'Rs', 'Rc', 'Rd', 'Rsc']]
        D['j'] = 'jn'
        E = pd.merge(C, D, how='outer', on='j')
        score = score_giver(C, D)
        df.at[indx, 'win'] = 0 if score.get(1) is None else score.get(1)
        df.at[indx, 'lose'] = 0 if score.get(0) is None else score.get(0)
        df.at[indx, 'review_score'] = float(0 if score.get(1) is None else score.get(1)) / len(data) * 1.0
    print("Iteration: {} Reviews of Product: {} Ranked".format(i + 1, product))

# Sort the DataFrame by product and review_score in descending order
df = df.sort_values(by=['product', 'review_score'], ascending=False)

# If in testing mode, calculate and print rank accuracy
if args.testing == 'True':
    data_split = pd.crosstab(df['product'], df['label'])
    r_accuracy = []
    for product in product_list:
        x = data_split[data_split.index == product][1][0]
        number_of_1_in_x = Counter(df[df['product'] == product].iloc[:x, ]['label']).get(1)
        rank_accuracy = float(number_of_1_in_x * 1.0 / x * 1.0)
        print("Product: {} | Rank Accuracy: {}".format(product, rank_accuracy))
        r_accuracy.append(rank_accuracy)
    print("TEST DATA: Mean Rank Accuracy: {}".format(mean(r_accuracy)))

# Print the resulting DataFrame with product, answer_option, and review_score
print(df[['product', 'answer_option', 'review_score']])

# Save the resulting DataFrame to a CSV file
df[['product', 'answer_option', 'review_score']].to_csv('data/test_ranked_output.csv', index=False)

# Print the total time taken for the ranking process
print('RANKING COMPLETE TIME TAKEN: {}'.format(time.time() - start))
