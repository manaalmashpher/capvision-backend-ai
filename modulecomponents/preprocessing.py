import tensorflow
import keras
# from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from modulecomponents.model_architecture import CNN_Encoder, RNN_Decoder, BahdanauAttention

def load_image(image_path):
    img = tensorflow.io.read_file(image_path)
    img = tensorflow.image.decode_jpeg(img, channels=3)
    img = tensorflow.image.resize(img, (299, 299))
    return tensorflow.keras.applications.inception_v3.preprocess_input(img), image_path

# Load models
encoder = tensorflow.keras.models.load_model("saved_models/encoderNEW.keras", custom_objects={"CNN_Encoder": CNN_Encoder})
decoder = tensorflow.keras.models.load_model("saved_models/decoderNEW.keras", custom_objects={
    "RNN_Decoder": RNN_Decoder,
    "BahdanauAttention": BahdanauAttention
})

# Initialize tokenizer (adjust paths as needed)
with open("tokenizer.json") as f:
    tokenizer = tensorflow.keras.preprocessing.text.tokenizer_from_json(f.read())

image_model = tensorflow.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tensorflow.keras.Model(new_input, hidden_layer)

def evaluate(image):
    attention_plot = np.zeros((40, 64))
    
    hidden = decoder.reset_state(batch_size=1)

    temp_input = tensorflow.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tensorflow.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tensorflow.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(40):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tensorflow.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tensorflow.argmax(predictions[0]).numpy()
        predicted_word = tokenizer.index_word[predicted_id]
        result.append(predicted_word)
        if predicted_word == '<end>':
            break

        dec_input = tensorflow.expand_dims([predicted_id], 0)
    attention_plot = attention_plot[:len(result), :]
    return result
