
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

 <h1>Emotion Detection Text Classification</h1>
    <p>
        This project focuses on building a text classification system to detect emotions from text data. 
        The IEMOCAP dataset is used to create a structured dataset, followed by data preprocessing, 
        model development, and evaluation. The project is implemented using Python, TensorFlow, and Keras.
    </p>
 <h2>Project Overview</h2>
<h4>Objective:</h4>
<p>Develop a text classification system that detects emotions in real-time data using the IEMOCAP dataset.</p>
<h4>Dataset Preparation:</h4>
<p></p>The raw data from IEMOCAP is processed and converted into a structured format, resulting in a CSV file containing text and corresponding emotion labels.</p>
<h2>Directory Structure</h2>
        <pre>
├── data/
│   ├── s1/
│   ├── s2/
│   ├── s3/
│   ├── s4/
│   └── s5/
├── output.csv
├── model/
│   └── best_model.h5
├── README.md
└── main.py
        </pre>
    <h2>Data Preprocessing</h2>
    <ul> <li><strong>Text Cleaning:</strong> The text is cleaned by converting it to lowercase, removing punctuation, tokenizing, removing stopwords, and lemmatizing words.</li>
        <li><strong>Label Encoding:</strong> Emotion labels are encoded into numerical values using <code>LabelEncoder</code>.</li>
    </ul>
    <h2>Model Development</h2>
    <p>The model is developed using the following architecture:</p>
    <ul>
        <li><strong>Embedding Layer:</strong> Converts text data into dense vectors.</li>
        <li><strong>Bidirectional LSTM Layers:</strong> Captures sequential dependencies in both forward and backward directions.</li>
        <li><strong>Dropout Layer:</strong> Prevents overfitting.</li>
        <li><strong>Dense Layer with Softmax Activation:</strong> Classifies the text into one of the predefined emotion categories.</li>
    </ul>
    <h2>Installation and Setup</h2>
    <ol>
        <li><strong>Clone the repository:</strong>
            <div class="code-block">
                <pre><code>git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection</code></pre>
            </div>
        </li>
        <li><strong>Install dependencies:</strong>
            <div class="code-block">
                <pre><code>pip install -r requirements.txt</code></pre>
            </div>
        </li>
        <li><strong>Download NLTK stopwords and wordnet:</strong>
            <div class="code-block">
                <pre><code>import nltk
nltk.download('stopwords')
nltk.download('wordnet')</code></pre>
            </div>
        </li>
        <li><strong>Run the project:</strong>
            <div class="code-block">
                <pre><code>python main.py</code></pre>
            </div>
        </li>
    </ol>
    <h2>Code Snippets</h2>
    <h3>Preprocess Text</h3>
    <div class="code-block">
        <pre><code>import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)</code></pre>
    </div>
    <h3>Model Definition</h3>
    <div class="code-block">
        <pre><code>from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, Bidirectional
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=100, input_length=100))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(8, activation='softmax'))  # 8 emotions</code></pre>
   </div>
    <h3>Model Training</h3>
    <div class="code-block">
        <pre><code>from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
checkpoint = ModelCheckpoint('model/best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint, earlyStopping, reduce_lr])</code></pre>
    </div>
   <h3>Text Classification Example</h3>
    <div class="code-block">
        <pre><code>def classify_text(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=100)
    prediction = model.predict(padded_sequence)
    emotion = label_encoder.inverse_transform([prediction.argmax()])
    return emotion[0]
new_text = "I love spending time with you!"
predicted_emotion = classify_text(new_text)
print(f"Text: {new_text}")
print(f"Predicted Emotion: {predicted_emotion}")</code></pre>
    </div>
    <h2>Evaluation and Results</h2>
    <p>
        The model's performance is evaluated on a test set, and accuracy is reported. The system can classify new text inputs into predefined emotion categories such as "joy," "sadness," "anger," etc.
    </p>
    <h2>Example Usage</h2>
    <div class="code-block">
        <pre><code>new_text = "I love spending time with you!"
predicted_emotion = classify_text(new_text)
print(f"Text: {new_text}")
print(f"Predicted Emotion: {predicted_emotion}")</code></pre>
    </div>
    <h2>Contributing</h2>
    <p> Contributions are welcome! Please create a pull request or open an issue for any suggestions or improvements.
    </p>
    <h2>License</h2>
    <p> This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.
    </p>

</body>
</html>
