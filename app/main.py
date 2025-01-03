import orjson
from typing import Dict, Any, Optional
import tdk.gts
from openai import OpenAI
from dotenv import load_dotenv
import os
from flask import Flask, jsonify, request
from dataclasses import dataclass
from http import HTTPStatus

# Load environment variables
load_dotenv()

@dataclass
class Config:
    API_KEY: str = os.getenv("API_KEY", "")
    BASE_URL: str = os.getenv("BASE_URL", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "")
    SYSTEM_PROMPT: str = """
    You are a Turkish language analysis assistant. Analyze given Turkish sentences and words.
    Return a JSON with:
    {
        "emotion": "Single English emotion word",
        "meaning_index": "Single number",
        "contextual_meaning": "Turkish explanation",
        "synonym": "Single Turkish word",
        "antonym": "Single Turkish word",
        "example": "New Turkish example sentence",
        "alternative": "Rephrased Turkish sentence"
    }
    """

class TDKService:
    @staticmethod
    def get_data(word: str) -> Dict[str, Any]:
        tdk_data = tdk.gts.search(word)
        if not tdk_data:
            raise ValueError(f"No TDK data found for: {word}")
        
        return {
            "definitions": [
                {
                    "number": meaning.order,
                    "meaning": meaning.meaning,
                    "properties": [prop.value.full_name for prop in meaning.properties 
                                if hasattr(prop, 'value')]
                }
                for entry in tdk_data
                for meaning in entry.meanings
            ],
            "origin": tdk_data[0].origin_language.name if tdk_data[0].origin_language else "Unspecified",
            "plurality": "Plural" if tdk_data[0].plural else "Singular",
            "proper_noun": "Proper Noun" if tdk_data[0].proper else "Common Noun"
        }

class AIService:
    def __init__(self, config: Config):
        self.client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
        )
        self.config = config

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.config.MODEL_NAME,
            messages=[
                {"role": "system", "content": self.config.SYSTEM_PROMPT},
                {"role": "user", "content": orjson.dumps(context).decode('utf-8')}
            ],
            response_format={"type": "json_object"}
        )
        return orjson.loads(response.choices[0].message.content)

class LanguageAnalyzer:
    def __init__(self, config: Config):
        self.tdk_service = TDKService()
        self.ai_service = AIService(config)

    def analyze(self, sentence: str, word: str) -> Dict[str, Any]:
        try:
            tdk_data = self.tdk_service.get_data(word)
            
            context = {
                "sentence": sentence,
                "word": word,
                "dictionary": {
                    "definitions": [
                        {
                            "number": d["number"],
                            "meaning": d["meaning"]
                        } for d in tdk_data["definitions"]
                    ]
                }
            }
            
            ai_response = self.ai_service.analyze(context)
            
            return {
                **ai_response,
                "dictionary": {
                    "definitions": tdk_data["definitions"],
                    "origin": tdk_data["origin"],
                    "plurality": tdk_data["plurality"],
                    "proper_noun": tdk_data["proper_noun"]
                }
            }
            
        except Exception as e:
            raise AnalysisError(str(e))

class AnalysisError(Exception):
    pass

def create_app(config: Config) -> Flask:
    app = Flask(__name__)
    analyzer = LanguageAnalyzer(config)

    @app.route('/analyze', methods=['POST'])
    def analyze():
        try:
            data = request.get_json()
            if not data or 'sentence' not in data or 'word' not in data:
                return jsonify({'error': 'Missing required parameters'}), HTTPStatus.BAD_REQUEST

            result = analyzer.analyze(data['sentence'], data['word'])
            return jsonify(result), HTTPStatus.OK

        except AnalysisError as e:
            return jsonify({'error': str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR

    @app.route('/')
    def home():
        return "Turkish Language Analyzer API is running"

    return app

if __name__ == "__main__":
    config = Config()
    app = create_app(config)
    app.run(debug=True)
