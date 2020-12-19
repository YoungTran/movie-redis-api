import asyncio
import json
from io import StringIO

import pandas as pd
import redis
from quart import Quart, request
from quart_cors import cors

from ml_engine import (apply_features, get_index_from_title,
                       get_similar_movies, get_title_from_index, init_features)

app = Quart(__name__)
app = cors(app)

r = redis.Redis(host='localhost', port=6379, db=0)

# set movies json to cache
@app.route('/set_movie_cache',methods=['POST'])
def set_movie_cached():
  movies_json = request.json
  r.set('movies', json.dumps(movies_json))
  print("Successfully set movies to cache")
  return r.get('movies')

# return all movies json from cache
@app.route('/get_movies',methods=['GET'])
async def get_movies():
   return r.get('movies')

# get cached movie csv string
async def get_cached_csv():
    return r.get('movies.csv')

# get all similar movies based on movie_user_picked
@app.route('/similar_movie',methods=['GET'])
async def get_similar():
   
   movie_user_picked =  request.args.get("movie_user_picked")
   
   cached_csv = await get_cached_csv()
   csv_data = cached_csv.decode("utf-8")

   df = pd.read_csv(StringIO(csv_data))

   await init_features(df)
   await apply_features(df)
   sorted_similar_movies = get_similar_movies(movie_user_picked, df)

   i = 0
   arr = []
   
   # return 50 similar movies
   for movie in sorted_similar_movies:
     arr.append(get_title_from_index(movie[0], df))

     i = i + 1
     if i > 50:
       break
   return json.dumps(arr)
  
if __name__ == '__main__':
  app.run(debug=False, use_reloader=True)
