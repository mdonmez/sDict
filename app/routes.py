from flask import jsonify, request
from app import app
from app.main import TurkishLanguageAnalyzer

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    
    if not data or 'sentence' not in data or 'word' not in data:
        return jsonify({'error': 'Missing required parameters'}), 400

    analyzer = TurkishLanguageAnalyzer()
    result = analyzer.analyze(data['sentence'], data['word'])
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200
