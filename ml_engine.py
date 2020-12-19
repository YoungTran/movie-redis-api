import asyncio
import json
from io import StringIO

import pandas as pd
import redis
from flask import Flask, request
from quart import Quart
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


async def init_features(df):
  features = ['overview', 'original_title', 'genre_ids', 'title']
  for feature in features:
    df[feature] = df[feature].fillna('')

def combine_features(row):
  try:
    return row['overview'] + " " + row['original_title'] + " " + row['genre_ids'] + " " + row['title']
  except:
    print("Error: " + row)

async def apply_features(df):
  df['combinedFeatures'] = df.apply(combine_features,axis=1)

def get_title_from_index(index,df):
   return df[df.index == index]["title"].values[0]

def get_index_from_title(title, df):
   return df[df.title == title]['title'].index.values[0]

def get_similar_movies(movie_user_picked, df):
   cv = CountVectorizer()
   count_matrix = cv.fit_transform(df['combinedFeatures'])

   cosine_sim = cosine_similarity(count_matrix)
   
   movie_index = get_index_from_title(movie_user_picked, df)

   similar_movies = list(enumerate(cosine_sim[movie_index]))

  ## Step 7: Get a list of similar movies in descending order of similarity score
   sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True
   )
   return sorted_similar_movies
