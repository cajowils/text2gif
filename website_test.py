import time
from text2gif import *
import pandas as pd
import random

sentence = "I like warm cookies"

ts = time.time()
glove_model = pickle.load( open( "glove_model.p", "rb" ) )
model_time = time.time() - ts

link_df, emotion_scores = text2gif(sentence, glove_model)

print(emotion_scores)