## text2gif
#### Recommending GIFs Using Emotion and Word Embeddings

Made by Calder Wilson, Brandon Luu, and Samuel Walsh

### Using the app

To run this Flask application, use the command ```flask run```

### About
In our project we sought out to build an NLP system that would take an input text, extract the emotion and topic of input, and output an animated GIF related to those two features. We found this topic to be super rewarding because seeing our system display GIFs made it fun; increasingly so as our system got better at recommending them. 

This project is similar to what you might find on a GIF search on a social media platform or in a messenger app. Our methodology might be a little bit different in that we are only matching keywords to get a related topic, but we use a model to extract the emotion.

We were able to use MIT Media Lab’s GIFGIF API to retrieve emotion scores (amusement, anger, happiness, sadness, etc) for thousands of GIFs. We then built a recurrent neural network that performed emotion analysis on two different datasets with labels mapped to various tweets and sentences. 

Our recurrent neural network consisted of an embedding layer that was trained on our data, two Gated Recurrent Units (GRU; similar to Long Short-Term Memory layers but generally more efficient) with 128 nodes each, and a Dense layer output with 7 nodes for each of our emotion classes (sadness, worry, surprise, love, hate, happiness, neutral).
![](/assets/model_architecture.png)
We attempted to use a Bidirectional layer to implement attention to our model, but we found that performance decreased for some reason. We also attempted to use pre-trained GloVe twitter word embeddings as our embedding layer, but this also decreased performance.

Before training the neural network we had to do a bit of preprocessing, which involved mapping the labeled emotions between the datasets, tokenization, lemmatization, padding, and converting the data to a tensor format. For the mapping of emotions we equated joy to happiness, anger to hate, and fear to worry. Similarly, when pulling GIFs from the GIFGIF API we mapped love to contentment, since the GIFs in that category were the closest representation to love we could find. There are definitely some drawbacks to this methodology, however we wanted to keep as many categories of emotions as possible, so that our emotion classifier would have a broad enough range of emotions to predict.

One problem we discovered in our initial training of the model was that it was predicting the majority classes much more often than the minority classes as a strategy of minimizing its loss function.
![](/assets/class_imbalance.png)
As you can see in the top plot, our original tweet dataset had very few samples that were labeled as hate or surprise. When we tested our original model on new samples such as “my cat is so annoying I hate her”, it misclassified them as one of the majority classes such as neutral, worry, sadness, or happiness. Our first method to solve this issue was to undersample our dataset, taking a randomly sampled dataset that included about 1000 samples per emotion. This solved our problem of minority classes being ignored, at the cost of losing lots of valuable data.

 Our second method to solve this issue was to collect more data to balance out our classes. We added a text dataset which had many sentences in a format akin to “i am feeling ____”. As shown above, combining these two datasets somewhat balanced our classes, but surprise and hate still lacked the amount of samples as our other classes.
 
Our model performed much better on the combined dataset, but we wanted to avoid the problem of our model underperforming on minority classes. To do so, we implemented class weighting in our model. This made it so our model’s loss function would penalize mislabeling minority classes more than when it penalized a majority class sample being mislabeled.

In GIFGIF, each GIF also has tags that we wanted to use to match the output GIF to the subject of the input text. To do so we summed GloVe word embeddings for both the input sentence and the tags and computed the cosine similarity of the two resulting vectors. To combine our scores for the emotion classifier and the cosine similarity of the input and the GIF tags we first multiplied our classifiers prediction probabilities for each emotion on the input by the score for the matching emotion on each GIF pulled from the GIFGIF API. This gave us a score that told us how well each GIF matched each emotion probability outputted by the emotion classifier. Then we multiplied the highest composite score by the cosine similarity of the tags and the input sentence. This final composite score would give us a metric that we could use to evaluate how well each GIF matched the predicted emotion of our input and the topic.

For our emotion classifier we used a simple accuracy score on the validation set to evaluate how well it was performing. Evaluating our tag-input similarity score was a bit harder because there were no “right answers” to compare our scores to. We found that many of the high similarity scores were related to the emotion of the input. For example, on the input “my cat is so annoying I hate her” we found that many of the high similarity scores had tags such as angry instead of our desired subject: cat.

Generating better similarity scores is one of the most important things we want to improve in our system. As of now, our system is pretty good at finding GIFs that match the emotion of our input, but not very good at discerning the topic. Another thing we could improve upon is our preprocessing, especially for tweets. We probably should have links, usernames, and erroneous punctuation from our data so that these things do not introduce bias in our model. Also, we recently found that Google AI has just released GoEmotion, an extremely large dataset with labeled emotions that is included in the TensorFlow datasets package. We think that including this dataset would greatly improve our model’s performance.

With that being said, we are proud of what we have accomplished and excited to keep iterating on what for most of us was our most in-depth and rewarding coding project at the time.
