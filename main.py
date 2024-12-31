from flask import Flask, request, jsonify, render_template
from ML_Project import URLPhishingDetector, IntegratedPhishingDetector
import os

app = Flask(__name__)
detector = URLPhishingDetector()
llm_detector = IntegratedPhishingDetector(gemini_key=os.getenv('GEMINI_KEY'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    url = request.form.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        # Analyze with ML model
        result = detector.predict(url)

        if 'error' in result:
            return jsonify({'error': result['error']}), 500

        # Extract features and get LLM reasoning
        features = llm_detector.extract_features(url)
        if not features:
            result['llm_reasoning'] = 'Failed to extract features for LLM reasoning.'
        else:
            llm_reasoning = llm_detector.get_llm_prediction(url, features)
            result['llm_reasoning'] = llm_reasoning

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    model_file = 'random_forest_phishing_detector.pkl'
    if os.path.exists(model_file):
        detector.load_model(model_file)
    else:
        detector.train()

    app.run(debug=True)
