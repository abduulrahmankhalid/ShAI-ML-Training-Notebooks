# Movie Reviews Sentiment Prediction Competition

## Kaggle Notebook: https://www.kaggle.com/code/abduulrahmankhalid/end-to-end-movie-reviews-sentiment-prediction


- The main objective of this competition was to predict the sentiment of movie reviews. To accomplish this, we embarked on a comprehensive data exploration phase to gain a better understanding of the dataset. Subsequently, we conducted data cleaning and preprocessing using various text cleaning techniques. Additionally, we employed processing techniques like TF-IDF vectorization and Keras tokenizer to prepare the data for training our machine learning and deep learning models.

- To streamline the preprocessing process, we developed a dedicated pipeline that encompassed text cleaning, tokenization, and preprocessing steps. We then experimented with different ML models and found that linear models yielded superior results. Building upon this, we fine-tuned our linear models and created a boosting model with our models. Additionally, we developed two deep learning models, one utilizing CNN and the other employing Bidirectional LSTM. These DL models performed comparably to our boosting ML model.

- To facilitate deployment, we constructed two prediction pipelines that integrated our ML and DL preprocessing pipelines along with the corresponding models. These pipelines were deployed using FastAPI. As a result of our efforts, we achieved a commendable position within the top 10 scores among 20 participating teams in the competition.

- This project not only allowed us to develop effective models for sentiment prediction but also provided valuable experience in deploying pipelines using FastAPI and competing in a challenging competition.






