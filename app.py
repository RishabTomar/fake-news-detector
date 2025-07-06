from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']
    data = vectorizer.transform([text])
    prediction = model.predict(data)
    result = "REAL NEWS ✅" if prediction[0] == 1 else "FAKE NEWS ❌"
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
