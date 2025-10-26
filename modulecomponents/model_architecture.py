import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim, **kwargs):
        super(CNN_Encoder, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim  # NEW
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,  #new
            # "fc": tf.keras.layers.serialize(self.fc)
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
@tf.keras.utils.register_keras_serializable()
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units, name=None, **kwargs):
        super(BahdanauAttention, self).__init__()
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        batch_size = tf.shape(features)[0] # Use dynamic batch size
        hidden_with_time_axis = tf.expand_dims(hidden, axis=1) #NEW
        score = (tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = tf.reduce_sum(attention_weights * features, axis=1)
        return context_vector, attention_weights

    def get_config(self): #new func
        config = super().get_config()
        config.update({
            "units": self.units,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_build_config(self):
        # Return a dictionary of layer configurations
        return {
            "W1": tf.keras.layers.serialize(self.W1),
            "W2": tf.keras.layers.serialize(self.W2),
            "V": tf.keras.layers.serialize(self.V),
        }

    def build_from_config(self, config):
        # Rebuild the model from the config
        self.W1 = tf.keras.layers.deserialize(config["W1"])
        self.W2 = tf.keras.layers.deserialize(config["W2"])
        self.V = tf.keras.layers.deserialize(config["V"])
        return self
    
@tf.keras.utils.register_keras_serializable()
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size, **kwargs): 
        super(RNN_Decoder, self).__init__(**kwargs) 
        self.embedding_dim = embedding_dim 
        self.units = units
        self.vocab_size = vocab_size
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)
    
    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        
        x = self.embedding(x)
        context_vector = tf.expand_dims(context_vector, axis=1)  # Shape: (8, 1, 256)
        x = tf.concat([context_vector, x], axis=-1)  # Shape: (8, 10, 512)
        output, state = self.gru(x)

        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "units": self.units,
            "vocab_size": self.vocab_size,
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_build_config(self):
        # Return a dictionary of layer configurations
        return {
            "embedding": tf.keras.layers.serialize(self.embedding),
            "gru": tf.keras.layers.serialize(self.gru),
            "fc1": tf.keras.layers.serialize(self.fc1),
            "fc2": tf.keras.layers.serialize(self.fc2),
            "attention": tf.keras.layers.serialize(self.attention),
        }

    def build_from_config(self, config):
        # Rebuild the model from the config
        self.embedding = tf.keras.layers.deserialize(config["embedding"])
        self.gru = tf.keras.layers.deserialize(config["gru"])
        self.fc1 = tf.keras.layers.deserialize(config["fc1"])
        self.fc2 = tf.keras.layers.deserialize(config["fc2"])
        self.attention = tf.keras.layers.deserialize(config["attention"])
        return self
    
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
