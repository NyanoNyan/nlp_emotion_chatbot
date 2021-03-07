import numpy as np
from flask import Flask, request, jsonify, render_template, flash, url_for
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

## Load the model
model = tf.keras.models.load_model('emotions.h5', custom_objects={
    'Adam': lambda **kwargs: hvd.DistributedOptimizer(keras.optimizers.Adam(**kwargs))
})
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)

## Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features =  [request.form['review']]

    final_features = request.form['review']

    sample_sequences = tokenizer.texts_to_sequences(int_features)
    fakes_padded = pad_sequences(sample_sequences, padding='post', maxlen=50) 
    
    output = model.predict(fakes_padded)
    print(type(output))
    if output>0.5:
        sentiment_pred = 'Joy'
    elif output<0.4:
        sentiment_pred = 'Sadness'
    else:
        sentiment_pred = 'Neutral'


    for x in range(len(int_features)):
        print(int_features[x])
        print(output[x])
        print('\n')
        

    return render_template('index.html', prediction_text='Predicted Emotion {}'.format(output), text=final_features, sentiment=sentiment_pred)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)