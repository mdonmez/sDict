import orjson
import tdk.gts
from openai import OpenAI
from dotenv import load_dotenv
import os
from flask import Flask, jsonify, request, render_template

# Load environment variables
load_dotenv()

app = Flask(__name__, 
           template_folder='../templates',
           static_folder='../static')


class TurkishLanguageAnalyzer:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
        )

    MODEL = os.getenv("MODEL_NAME")
        
    SYSTEM_PROMPT = """
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

    def get_tdk_data(self, word):
        tdk_data = tdk.gts.search(word)
        if not tdk_data:
            raise ValueError(f"No TDK data found for: {word}")
            
        definitions = []
        for entry in tdk_data:
            for meaning in entry.meanings:
                definitions.append({
                    "number": meaning.order,
                    "meaning": meaning.meaning,
                    "properties": [prop.value.full_name for prop in meaning.properties 
                                 if hasattr(prop, 'value')]
                })
                
        return {
            "definitions": definitions,
            "origin": tdk_data[0].origin_language.name if tdk_data[0].origin_language else "Unspecified",
            "plurality": "Plural" if tdk_data[0].plural else "Singular",
            "proper_noun": "Proper Noun" if tdk_data[0].proper else "Common Noun"
        }

    def get_ai_analysis(self, sentence, word, tdk_data):
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
        
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": orjson.dumps(context).decode('utf-8')}
            ],
            response_format={"type": "json_object"}
        )
        
        return orjson.loads(response.choices[0].message.content)

    def analyze(self, sentence, word):
        try:
            # Step 1: Get TDK data
            tdk_data = self.get_tdk_data(word)
            
            # Step 2: Get AI analysis
            ai_response = self.get_ai_analysis(sentence, word, tdk_data)
            
            # Step 3: Create ordered response
            final_response = {
                "emotion": ai_response.get("emotion"),
                "meaning_index": ai_response.get("meaning_index"),
                "contextual_meaning": ai_response.get("contextual_meaning"),
                "synonym": ai_response.get("synonym"),
                "antonym": ai_response.get("antonym"),
                "example": ai_response.get("example"),
                "alternative": ai_response.get("alternative"),
                "dictionary": {
                    "definitions": tdk_data["definitions"],
                    "origin": tdk_data["origin"],
                    "plurality": tdk_data["plurality"],
                    "proper_noun": tdk_data["proper_noun"]
                }
            }
            
            return final_response
                
        except Exception as e:
            return {"error": str(e)}


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    
    if not data or 'sentence' not in data or 'word' not in data:
        return jsonify({'error': 'Missing required parameters'}), 400

    analyzer = TurkishLanguageAnalyzer()
    result = analyzer.analyze(data['sentence'], data['word'])
    return jsonify(result)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
