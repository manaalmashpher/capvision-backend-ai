from flask import Flask, request, jsonify, send_file
import tensorflow as tf
from modulecomponents.model_architecture import CNN_Encoder, RNN_Decoder, BahdanauAttention
from modulecomponents.preprocessing import load_image, tokenizer, evaluate, image_features_extract_model
from modulecomponents.config import EMBEDDING_DIM, UNITS, VOCAB_SIZE, MAX_LENGTH
import os
from PIL import Image
import io
from google.cloud import texttospeech
from google.oauth2 import service_account
from dotenv import load_dotenv
from google import genai
from google.genai import types
import base64

load_dotenv()
app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

with open("tokenizer.json", "r") as f:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())

# Load models
encoder = tf.keras.models.load_model("saved_models/encoderNEW.keras", custom_objects={"CNN_Encoder": CNN_Encoder})
decoder = tf.keras.models.load_model("saved_models/decoderNEW.keras", custom_objects={
    "RNN_Decoder": RNN_Decoder,
    "BahdanauAttention": BahdanauAttention
})

# Initialize TTS client
CREDENTIALS_PATH = "credentials.json"
credentials = service_account.Credentials.from_service_account_file(
    CREDENTIALS_PATH,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
tts_client = texttospeech.TextToSpeechClient(credentials=credentials)

def generate_tts(text, output_file="output.mp3"):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-IN",
        name="en-IN-Standard-C",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    
    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    
    with open(output_file, "wb") as out:
        out.write(response.audio_content)
    return output_file

def generate_caption_gemini(img):
    try:
        #img = Image.open(image_path)
        if not isinstance(img, Image.Image):
            raise ValueError("Input must be a PIL Image")
        # Convert the image to a base64-encoded string
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
        response = client.models.generate_content(model='gemini-2.0-flash-lite', contents=["Generate a detailed image caption in 40 words or less:", image_part])
        print("DONE 1")
        return response.text
    except Exception as e:
        return jsonify({'error': 'API error'}), 400

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded file temporarily
    temp_image_path = "temp_upload.jpg"
    file.save(temp_image_path)
    
    try:
        # Generate caption
        result = evaluate(temp_image_path)

        # Flatten nested lists if needed
        if result and isinstance(result[0], list):
            result = [word for sublist in result for word in sublist]

        caption = ' '.join(result).replace('<start>', '').replace('<end>', '').strip()
        
        # Generate TTS
        audio_file = generate_tts(caption)
        
        # Return results
        return jsonify({
            'caption': caption,
            'audio_url': '/get_audio'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary files
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

@app.route('/gemini_caption', methods=['POST'])
def upload_gemini():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded file temporarily
    img_bytes = file.read()
    if not img_bytes:
            return jsonify({'error': 'Empty image file'}), 400
    img = Image.open(io.BytesIO(img_bytes))

    try:
        # Generate caption
        result = generate_caption_gemini(img)
        # Generate TTS
        audio_file = generate_tts(result)
        
        # Return results
        return jsonify({
            'caption': result,
            'audio_url': '/get_audio'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    # finally:
        # Clean up temporary files
        # if os.path.exists(temp_image_path):
        #     os.remove(temp_image_path)

@app.route('/get_audio')
def get_audio():
    return send_file("output.mp3", mimetype="audio/mpeg")

if __name__ == '__main__':
    app.run(debug=True)