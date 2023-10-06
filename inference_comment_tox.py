import os
import pandas as pd
import tensorflow as tf
import numpy as np


new_model =  tf.keras.models.load_model("Model_comments_toxicity/Comments_Toxicity_2")
print("loaded successfully")
# MAX_FEATURES = 200000 # number of words in the vocab
# vectorizer = tf.keras.layers.TextVectorization(max_tokens=MAX_FEATURES,
#                                output_sequence_length=1800,
#                                output_mode='int')

# df = pd.read_csv(os.path.join('dataset_tox','train.csv'))

# X = df['comment_text']

# vectorizer.adapt(X.values)

def textClassifier(text):
   
    input_text = text

    print(input_text)
   
    res = new_model.predict(np.expand_dims(input_text,0))

    print(res)

    flat_array = np.array(res).flatten()

    output_array = [1 if value > 0.2 else 0 for value in  flat_array]

    return  output_array



