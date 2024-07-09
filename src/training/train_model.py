import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
import json

# Function to preprocess data
def preprocess_data(data):
    m_points_all = []
    q_points_all = []
    stroke_widths_all = []
    classes = []

    max_points = 0  # Variable to track maximum number of points

    # Loop through each class data
    for entry in data:
        class_name = entry['class']
        classes.append(class_name)
        images_data = entry['images']
        
        # Initialize lists for each image's data
        class_m_points = []
        class_q_points = []
        class_stroke_widths = []

        # Find maximum points for padding
        for image_data in images_data:
            m_points = np.array(image_data['m_points'])
            q_points = np.array(image_data['q_points'])
            stroke_widths = np.array(image_data['stroke_widths'])
            
            class_m_points.append(m_points)
            class_q_points.append(q_points)
            class_stroke_widths.append(stroke_widths)
            
            max_points = max(max_points, len(m_points))

        # Pad m_points and q_points to ensure consistent dimensions
        for i in range(len(class_m_points)):
            m_points_padded = pad_sequence(class_m_points[i], max_points)
            q_points_padded = pad_sequence(class_q_points[i], max_points)
            stroke_widths = np.array(class_stroke_widths[i])[:, np.newaxis]  # reshape stroke widths
            
            m_points_all.append(m_points_padded)
            q_points_all.append(q_points_padded)
            stroke_widths_all.append(stroke_widths)

    # Convert lists to numpy arrays
    m_points_all = np.array(m_points_all)
    q_points_all = np.array(q_points_all, dtype=object)  # dtype=object for ragged arrays
    stroke_widths_all = np.concatenate(stroke_widths_all, axis=0)  # Concatenate all stroke widths
    
    return classes, m_points_all, q_points_all, stroke_widths_all

# Function to pad sequences to a fixed length
def pad_sequence(sequence, max_length):
    padded_sequence = np.zeros((max_length, sequence.shape[1]), dtype=sequence.dtype)
    padded_sequence[:sequence.shape[0], :] = sequence
    return padded_sequence

# Define the VAE model architecture
def build_vae(input_shape):
    # Encoder
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(256, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    z_mean = Dense(64, name='z_mean')(x)
    z_log_var = Dense(64, name='z_log_var')(x)
    
    # Reparameterization trick
    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    z = Lambda(sampling, output_shape=(64,), name='z')([z_mean, z_log_var])
    
    # Decoder
    decoder_inputs = Input(shape=(64,), name='z_sampling')
    y = Dense(128, activation='relu')(decoder_inputs)
    y = Dense(256, activation='relu')(y)
    outputs = Dense(input_shape[0], activation='sigmoid')(y)
    
    # Models
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(decoder_inputs, outputs, name='decoder')
    vae_outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, vae_outputs, name='vae')
    
    # VAE Loss
    reconstruction_loss = mse(inputs, vae_outputs)
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    return vae, encoder, decoder

# Load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Train VAE
def train_vae(data_file):
    # Load and preprocess data
    data = load_json_data(data_file)
    classes, m_points_all, q_points_all, stroke_widths_all = preprocess_data(data)
    
    # Define input shape
    input_shape = m_points_all.shape[1:]
    
    # Build VAE model
    vae, encoder, decoder = build_vae(input_shape)
    
    # Compile and train model
    vae.compile(optimizer='adam')
    vae.fit(m_points_all, m_points_all, epochs=10, batch_size=32)
    
    # Save models
    vae.save('vae_model.h5')
    encoder.save('encoder_model.h5')
    decoder.save('decoder_model.h5')

# Example usage
if __name__ == "__main__":
    data_file = '/home/abhishek/svg-model-project/data/processed/extracted_features.json'
    train_vae(data_file)
