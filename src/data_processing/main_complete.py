import json
import numpy as np
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam
import numpy as np
import svgwrite
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, concatenate, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, concatenate, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model


def load_json_file(json_file):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    return json_data

def preprocess_data(json_data, max_seq_length):
    X_m_points = []
    X_q_points = []
    X_stroke_widths = []
    y = []

    for entry in json_data:
        class_label = entry['class']
        for image_data in entry['images']:
            m_points = np.array(image_data['m_points'])
            q_points = np.array(image_data['q_points'])
            stroke_widths = np.array(image_data['stroke_widths'])
            
            # Skip data if any sequence length is greater than max_seq_length
            if len(m_points) > max_seq_length or len(q_points) > max_seq_length:
                continue
            
            # Pad sequences to max_seq_length
            pad_length_m = max_seq_length - len(m_points)
            pad_length_q = max_seq_length - len(q_points)
            m_points = np.pad(m_points, ((0, pad_length_m), (0, 0)), mode='constant')
            q_points = np.pad(q_points, ((0, pad_length_q), (0, 0)), mode='constant')
            stroke_widths = np.pad(stroke_widths, (0, max_seq_length - len(stroke_widths)), mode='constant')
            
            X_m_points.append(m_points)
            X_q_points.append(q_points)
            X_stroke_widths.append(stroke_widths)
            y.append(class_label)
    
    X_m_points = np.array(X_m_points)
    X_q_points = np.array(X_q_points)
    X_stroke_widths = np.array(X_stroke_widths)
    y = np.array(y)
    
    return X_m_points, X_q_points, X_stroke_widths, y

# Example usage
json_file = '/home/abhishek/svg-model-project/data/processed/extracted_features.json'
max_seq_length = 100  # Example max sequence length for padding

# Load JSON data
json_data = load_json_file(json_file)

# Preprocess data
X_m_points, X_q_points, X_stroke_widths, y = preprocess_data(json_data, max_seq_length)



# Parameters
input_dim_m_points = (100, 2)
input_dim_q_points = (100, 4)
input_dim_stroke_widths = (100,)
latent_dim = 32


# Normalize data
X_m_points = (X_m_points - np.mean(X_m_points)) / np.std(X_m_points)
X_q_points = (X_q_points - np.mean(X_q_points)) / np.std(X_q_points)
X_stroke_widths = (X_stroke_widths - np.mean(X_stroke_widths)) / np.std(X_stroke_widths)



# Encoder
m_points_input = Input(shape=input_dim_m_points, name='m_points')
q_points_input = Input(shape=input_dim_q_points, name='q_points')
stroke_widths_input = Input(shape=input_dim_stroke_widths, name='stroke_widths')

x = concatenate([Flatten()(m_points_input), Flatten()(q_points_input), stroke_widths_input])
x = Dense(512)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)
x = Dense(256)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)
x = Dense(128)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

encoder = Model([m_points_input, q_points_input, stroke_widths_input], [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# Decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')

x_decoded = Dense(128)(latent_inputs)
x_decoded = BatchNormalization()(x_decoded)
x_decoded = LeakyReLU()(x_decoded)
x_decoded = Dense(256)(x_decoded)
x_decoded = BatchNormalization()(x_decoded)
x_decoded = LeakyReLU()(x_decoded)
x_decoded = Dropout(0.5)(x_decoded)
x_decoded = Dense(512)(x_decoded)
x_decoded = BatchNormalization()(x_decoded)
x_decoded = LeakyReLU()(x_decoded)
x_decoded = Dropout(0.5)(x_decoded)
x_decoded = Dense(np.prod(input_dim_m_points) + np.prod(input_dim_q_points) + np.prod(input_dim_stroke_widths), activation='sigmoid')(x_decoded)

m_points_decoded = Reshape(input_dim_m_points)(x_decoded[:, :np.prod(input_dim_m_points)])
q_points_decoded = Reshape(input_dim_q_points)(x_decoded[:, np.prod(input_dim_m_points):np.prod(input_dim_m_points) + np.prod(input_dim_q_points)])
stroke_widths_decoded = x_decoded[:, np.prod(input_dim_m_points) + np.prod(input_dim_q_points):]

decoder = Model(latent_inputs, [m_points_decoded, q_points_decoded, stroke_widths_decoded], name='decoder')
decoder.summary()

# VAE Model
outputs = decoder(encoder([m_points_input, q_points_input, stroke_widths_input])[2])
vae = Model([m_points_input, q_points_input, stroke_widths_input], outputs, name='vae')

# Loss
reconstruction_loss_m_points = mse(K.flatten(m_points_input), K.flatten(outputs[0]))
reconstruction_loss_q_points = mse(K.flatten(q_points_input), K.flatten(outputs[1]))
reconstruction_loss_stroke_widths = mse(K.flatten(stroke_widths_input), K.flatten(outputs[2]))
reconstruction_loss = reconstruction_loss_m_points + reconstruction_loss_q_points + reconstruction_loss_stroke_widths

kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5

vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0))
vae.summary()

# Checking for NaN values in data
print("Any NaN in X_m_points:", np.isnan(X_m_points).any())
print("Any NaN in X_q_points:", np.isnan(X_q_points).any())
print("Any NaN in X_stroke_widths:", np.isnan(X_stroke_widths).any())

# Ensure y is numeric and check for NaN values
try:
    y = y.astype(float)
    print("Any NaN in y:", np.isnan(y).any())
except ValueError as e:
    print(f"Error converting y to float: {e}")
    print("y contains non-numeric values or is already in the correct type. Checking unique values instead.")
    print("Unique values in y:", np.unique(y))

# Clipping values to avoid extreme values
X_m_points = np.clip(X_m_points, -1e15, 1e15)
X_q_points = np.clip(X_q_points, -1e15, 1e15)
X_stroke_widths = np.clip(X_stroke_widths, -1e15, 1e15)

# Ensure y is converted to integers if they are not
y = y.astype(int)
print("Unique labels in y after conversion to int:", np.unique(y))

# Map class labels to indices
class_label_mapping = {0: 0, 1: 1, 2: 2, 7: 3}
try:
    y_mapped = np.array([class_label_mapping[label] for label in y])
except KeyError as e:
    print(f"Label {e} not found in class_label_mapping")
    raise

# Convert y to one-hot encoding
num_classes = 4  # Classes 0, 1, 2, 7 mapped to indices 0, 1, 2, 3
y_one_hot = tf.keras.utils.to_categorical(y_mapped, num_classes=num_classes)

print(y_one_hot.shape)  # Should print (798, 4)

# Train the VAE
vae.fit([X_m_points, X_q_points, X_stroke_widths], epochs=500, batch_size=32)


def generate_new_data(class_label, num_samples=1):
    class_vector = np.zeros((num_samples, num_classes))
    class_vector[:, class_label_mapping[class_label]] = 1
    latent_samples = np.random.normal(size=(num_samples, latent_dim))
    generated_m_points, generated_q_points, generated_stroke_widths = decoder.predict(latent_samples)
    return generated_m_points, generated_q_points, generated_stroke_widths

# Generate data for class 7
m_points_generated, q_points_generated, stroke_widths_generated = generate_new_data(class_label=0, num_samples=1)

# Convert generated data to SVG
def create_svg(m_points, q_points, stroke_widths, filename='generated0_image.svg', width=25, height=25):
    dwg = svgwrite.Drawing(filename, profile='tiny', size=(width, height))

    for i in range(len(m_points)):
        start = (m_points[i][0] * width, m_points[i][1] * height)
        if i < len(q_points):
            control1 = (q_points[i][0] * width, q_points[i][1] * height)
            control2 = (q_points[i][2] * width, q_points[i][3] * height)
            end = (m_points[(i+1) % len(m_points)][0] * width, m_points[(i+1) % len(m_points)][1] * height)
            dwg.add(dwg.path(d='M {} {} Q {} {} {} {}'.format(start[0], start[1], control1[0], control1[1], control2[0], control2[1], end[0], end[1]),
                             stroke='black', fill='none', stroke_width=str(stroke_widths[i])))
        else:
            end = (m_points[(i+1) % len(m_points)][0] * width, m_points[(i+1) % len(m_points)][1] * height)
            dwg.add(dwg.line(start=start, end=end, stroke='black', stroke_width=str(stroke_widths[i])))

    dwg.save()

# Directory to save the SVG files
output_directory = '/home/abhishek/svg-model-project/data/generated'

create_svg(m_points_generated[0], q_points_generated[0], stroke_widths_generated[0], directory=output_directory, width=25, height=25)
