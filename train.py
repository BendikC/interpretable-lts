# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers
tf.keras.backend.clear_session()  # For easy reset of notebook state.
import soundfile as sf

import random
import numpy as np

import os, sys, argparse, time
from pathlib import Path

import librosa
import configparser
import random
import json
import matplotlib.pyplot as plt

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default ='./configs/default.ini' , help='path to the config file')
args = parser.parse_args()

#Get configs
config_path = args.config
config = configparser.ConfigParser(allow_no_value=True)
try: 
  config.read(config_path)
except FileNotFoundError:
  print('Config File Not Found at {}'.format(config_path))
  sys.exit()

#import audio configs 
sample_rate = config['audio'].getint('sample_rate')
hop_length = config['audio'].getint('hop_length')
bins_per_octave = config['audio'].getint('bins_per_octave')
num_octaves = config['audio'].getint('num_octaves')
n_bins = int(num_octaves * bins_per_octave)
n_iter = config['audio'].getint('n_iter')

#dataset
dataset = Path(config['dataset'].get('datapath'))
if not dataset.exists():
    raise FileNotFoundError(dataset.resolve())

cqt_dataset = config['dataset'].get('cqt_dataset')

if config['dataset'].get('workspace') != None:
  workspace = Path(config['dataset'].get('workspace'))

run_number = config['dataset'].getint('run_number')
my_cqt = dataset / cqt_dataset
if not my_cqt.exists():
    raise FileNotFoundError(my_cqt.resolve())

my_audio = dataset / 'audio'
    
#Training configs
epochs = config['training'].getint('epochs')
learning_rate = config['training'].getfloat('learning_rate')
batch_size = config['training'].getint('batch_size')
train_buf = config['training'].getint('buffer_size')
buffer_size_dataset = config['training'].getboolean('buffer_size_dataset')
max_to_keep = config['training'].getint('max_ckpts_to_keep')
ckpt_epochs = config['training'].getint('checkpoint_epochs')
continue_training = config['training'].getboolean('continue_training')
learning_schedule = config['training'].getboolean('learning_schedule')
save_best_only = config['training'].getboolean('save_best_only')
early_patience_epoch = config['training'].getint('early_patience_epoch')
early_delta = config['training'].getfloat('early_delta')
adam_beta_1 = config['training'].getfloat('adam_beta_1')
adam_beta_2 = config['training'].getfloat('adam_beta_2')

#Model configs
latent_dim = config['VAE'].getint('latent_dim')
n_units = config['VAE'].getint('n_units')
kl_beta = config['VAE'].getfloat('kl_beta')
batch_normalization = config['VAE'].getboolean('batch_norm')
VAE_output_activation = config['VAE'].get('output_activation')

# Spectral centroid configs
spectral_centroid_weight = config['VAE'].getfloat('spectral_centroid_weight', fallback=0.0)
centroid_dim = config['VAE'].getint('centroid_dim', fallback=0)

#etc
example_length = config['extra'].getint('example_length')
normalize_examples = config['extra'].getboolean('normalize_examples')
plot_model = config['extra'].getboolean('plot_model')

desc = config['extra'].get('description')
start_time = time.time()
config['extra']['start'] = time.asctime( time.localtime(start_time) )

AUTOTUNE = tf.data.experimental.AUTOTUNE

#Create workspace

if not continue_training:
    run_id = run_number
    while True:
        try:
            my_runs = dataset / desc
            run_name = 'run-{:03d}'.format(run_id)
            workdir = my_runs / run_name 
            os.makedirs(workdir)

            break
        except OSError:
            if workdir.is_dir():
                run_id = run_id + 1
                continue
            raise

    config['dataset']['workspace'] = str(workdir.resolve())
else:
    workdir = config['dataset'].get('workspace')

print("Workspace: {}".format(workdir))

#create the dataset
print('creating the dataset...')
training_array = []
new_loop = True

for f in os.listdir(my_cqt): 
    if f.endswith('.npy'):
        print('adding-> %s' % f)
        file_path = my_cqt / f
        new_array = np.load(file_path)
        if new_loop:
            training_array = new_array
            new_loop = False
        else:
            training_array = np.concatenate((training_array, new_array), axis=0)

total_cqt = len(training_array)
print('Total number of CQT frames: {}'.format(total_cqt))
config['dataset']['total_frames'] = str(total_cqt)

print("saving initial configs...")
config_path = workdir / 'config.ini'
with open(config_path, 'w') as configfile:
  config.write(configfile)

if buffer_size_dataset:
  train_buf = len(training_array)

# Spectral centroid computation function
def compute_spectral_centroid_tf(cqt_magnitude, fmin=32.7, bins_per_octave=48):
    """Compute spectral centroid using proper frequency mapping."""

    # first we ensure non-negative and add small epsilon
    cqt_magnitude = tf.abs(cqt_magnitude) + 1e-8

    # we get the number of bins, and we define the indices as floats
    n_bins = tf.shape(cqt_magnitude)[1]
    bin_indices = tf.cast(tf.range(n_bins), tf.float32)
    
    # cqt bins are logarithmically spaced
    # this means that frequency for each bin is:
    # f = fmin * 2^(bin_index / bins_per_octave)
    frequencies = fmin * tf.pow(2.0, bin_indices / bins_per_octave)

    # we expand the dims to match the batch size
    frequencies = tf.expand_dims(frequencies, 0)
    
    # we compute the sum of frequency * magnitude
    weighted_freq = tf.reduce_sum(cqt_magnitude * frequencies, axis=1)

    # then we compute the total magnitude
    total_magnitude = tf.reduce_sum(cqt_magnitude, axis=1)

    # spectral centroid = weighted frequency sum / total magnitude
    centroid_hz = weighted_freq / (total_magnitude + 1e-8)
    
    # Clamp centroid to valid frequency range before log
    fmax = fmin * tf.pow(2.0, tf.cast(n_bins, tf.float32) / bins_per_octave)
    centroid_hz = tf.clip_by_value(centroid_hz, fmin, fmax)
    
    # Normalize to [0, 1] using logarithmic scale
    # log(centroid/fmin) / log(fmax/fmin)
    log_centroid = tf.math.log(centroid_hz / fmin + 1e-8)
    log_range = tf.math.log(fmax / fmin + 1e-8)
    centroid_normalized = log_centroid / log_range
    
    # Final clamp to [0, 1] (safety net)
    centroid_normalized = tf.clip_by_value(centroid_normalized, 0.0, 1.0)
    
    return centroid_normalized

#Define Sampling Layer
class Sampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Train
model_dir = workdir / "model"
os.makedirs(model_dir,exist_ok=True)

log_dir = workdir / 'logs'
os.makedirs(log_dir, exist_ok=True)

if not continue_training:
  # Define encoder model.
  original_dim = n_bins
  original_inputs = tf.keras.Input(shape=(original_dim,), name='encoder_input')
  x = layers.Dense(n_units, activation='relu')(original_inputs)
  z_mean = layers.Dense(latent_dim, name='z_mean')(x)
  z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
  z = Sampling()((z_mean, z_log_var))
  encoder = tf.keras.Model(inputs=original_inputs, outputs=[z_mean, z_log_var, z], name='encoder')
  encoder.summary()

  # Define decoder model.
  latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
  x = layers.Dense(n_units, activation='relu')(latent_inputs)
  outputs = layers.Dense(original_dim, activation=VAE_output_activation)(x)
  decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')
  decoder.summary()

  # Define VAE model with custom loss including spectral centroid
  class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, kl_beta=1.0, spectral_centroid_weight=0.0, 
                 fmin=32.7, bins_per_octave=48, **kwargs):
      super(VAE, self).__init__(**kwargs)
      self.encoder = encoder
      self.decoder = decoder
      self.kl_beta = kl_beta
      self.spectral_centroid_weight = spectral_centroid_weight
      self.centroid_dim = centroid_dim
      self.fmin = fmin
      self.bins_per_octave = bins_per_octave
      
      self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
      self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
      self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
      self.centroid_loss_tracker = tf.keras.metrics.Mean(name="centroid_loss")

    @property
    def metrics(self):
      return [
        self.total_loss_tracker,
        self.reconstruction_loss_tracker,
        self.kl_loss_tracker,
        self.centroid_loss_tracker,
      ]

    def call(self, inputs):
      z_mean, z_log_var, z = self.encoder(inputs)
      reconstruction = self.decoder(z)
      return reconstruction


    def train_step(self, data):
      with tf.GradientTape() as tape:
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        
        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(
          tf.keras.losses.mse(data, reconstruction)
        )
        
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
          z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        
        # Spectral centroid loss (BOTH jobs in ONE loss!)
        centroid_loss = tf.constant(0.0)
        if self.spectral_centroid_weight > 0:
          # Compute actual centroid from input
          input_centroid = compute_spectral_centroid_tf(
            data, 
            fmin=self.fmin, 
            bins_per_octave=self.bins_per_octave
          )
          
          # Predict centroid from latent space
          z_centroid_dim = z_mean[:, self.centroid_dim]
          # Apply sigmoid to map to [0, 1] (same range as centroid)
          predicted_centroid = tf.nn.sigmoid(z_centroid_dim)

          # compute centroid from reconstruction
          output_centroid = compute_spectral_centroid_tf(
            reconstruction, 
            fmin=self.fmin, 
            bins_per_octave=self.bins_per_octave
          )
          
          # Combined centroid loss
          centroid_loss = (
            tf.reduce_mean(tf.square(predicted_centroid - input_centroid)) + 
            tf.reduce_mean(tf.square(predicted_centroid - output_centroid))
          )
          centroid_loss = tf.clip_by_value(centroid_loss, 0.0, 2.0)
        
        # Total loss
        total_loss = (
          reconstruction_loss + 
          self.kl_beta * kl_loss +
          self.spectral_centroid_weight * centroid_loss
        )
      
      # Compute gradients
      grads = tape.gradient(total_loss, self.trainable_weights)
      self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
      
      # Update metrics
      self.total_loss_tracker.update_state(total_loss)
      self.reconstruction_loss_tracker.update_state(reconstruction_loss)
      self.kl_loss_tracker.update_state(kl_loss)
      self.centroid_loss_tracker.update_state(centroid_loss)
      
      return {
        "loss": self.total_loss_tracker.result(),
        "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        "kl_loss": self.kl_loss_tracker.result(),
        "centroid_loss": self.centroid_loss_tracker.result(),
      }

    def test_step(self, data):
      z_mean, z_log_var, z = self.encoder(data, training=False)
      reconstruction = self.decoder(z, training=False)
      
      reconstruction_loss = tf.reduce_mean(
        tf.keras.losses.mse(data, reconstruction)
      )
      kl_loss = -0.5 * tf.reduce_mean(
        z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
      )
      
      centroid_loss = tf.constant(0.0)
      if self.spectral_centroid_weight > 0:
        input_centroid = compute_spectral_centroid_tf(
          data, 
          fmin=self.fmin, 
          bins_per_octave=self.bins_per_octave
        )
        output_centroid = compute_spectral_centroid_tf(
          reconstruction, 
          fmin=self.fmin, 
          bins_per_octave=self.bins_per_octave
        )
        centroid_loss = tf.reduce_mean(tf.square(input_centroid - output_centroid))
        centroid_loss = tf.clip_by_value(centroid_loss, 0.0, 1.0)
      
      total_loss = (
        reconstruction_loss + 
        self.kl_beta * kl_loss +
        self.spectral_centroid_weight * centroid_loss
      )
      
      self.total_loss_tracker.update_state(total_loss)
      self.reconstruction_loss_tracker.update_state(reconstruction_loss)
      self.kl_loss_tracker.update_state(kl_loss)
      self.centroid_loss_tracker.update_state(centroid_loss)
      
      return {
        "loss": self.total_loss_tracker.result(),
        "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        "kl_loss": self.kl_loss_tracker.result(),
        "centroid_loss": self.centroid_loss_tracker.result(),
      }

  # Calculate fmin from CQT configuration
  # Standard MIDI note 0 (C-1) is 8.176 Hz
  # For typical CQT, fmin = C1 (32.7 Hz) or use config if available
  fmin = 32.7  # You can make this configurable if needed
  
  # Create VAE instance with spectral centroid
  vae = VAE(
    encoder, 
    decoder, 
    kl_beta=kl_beta, 
    spectral_centroid_weight=spectral_centroid_weight,
    fmin=fmin,
    bins_per_octave=bins_per_octave
  )
  vae.build(input_shape=(None, original_dim))
  vae.summary()
  
  print(f"\n=== VAE Configuration ===")
  print(f"KL beta: {kl_beta}")
  print(f"Spectral centroid weight: {spectral_centroid_weight}")
  print(f"Frequency range: {fmin:.2f} Hz - {fmin * (2 ** num_octaves):.2f} Hz")

  if plot_model:
    tf.keras.utils.plot_model(
      encoder,
      to_file= workdir.joinpath('model_encoder.jpg'),
      show_shapes=True,
      show_layer_names=True,
      rankdir='TB',
      expand_nested=True,
      dpi=300
    )

    tf.keras.utils.plot_model(
      decoder,
      to_file=workdir.joinpath('model_decoder.jpg'),
      show_shapes=True,
      show_layer_names=True,
      rankdir='TB',
      expand_nested=True,
      dpi=300
    )

  if learning_schedule:
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      learning_rate*100,
      decay_steps=int(epochs*0.8),
      decay_rate=0.96,
      staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule, beta_1=adam_beta_1, beta_2=adam_beta_2)
  else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=adam_beta_1, beta_2=adam_beta_2)

  vae.compile(optimizer=optimizer)

else: 
  # Load the model - this part needs updating too
  print("Loading existing model...")
  my_model_path = workdir / 'model' / 'mymodel_last.h5'
  
  # For now, you'll need to recreate and load weights
  print("⚠️  Note: Loading saved VAE not yet implemented for custom model")
  print("   Please retrain from scratch or implement custom loading")
  sys.exit(1)

modelpath = model_dir.joinpath('mymodel_last.h5')

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
      filepath= str(modelpath),
      # Path where to save the model
      # The two parameters below mean that we will overwrite
      # the current checkpoint if and only if
      # the `val_loss` score has improved.
      save_best_only=save_best_only,
      monitor='loss',
      verbose=1),         
    tf.keras.callbacks.EarlyStopping(
      # Stop training when `val_loss` is no longer improving
      monitor='loss',
      # "no longer improving" being defined as "no better than 1e-2 less"
      min_delta=early_delta,
      # "no longer improving" being further defined as "for at least 2 epochs"
      patience=early_patience_epoch,
      verbose=1),
    tf.keras.callbacks.TensorBoard(
      log_dir=str(log_dir), 
      histogram_freq=1)
]

history = vae.fit(training_array, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

print('\nhistory dict:', history.history)

with open(workdir.joinpath('my_history.json'), 'w') as json_file:
  json.dump(history.history, json_file)

print("Finished training...")
end_time = time.time()
config['extra']['end'] = time.asctime( time.localtime(end_time) )
time_elapsed = end_time - start_time
config['extra']['time_elapsed'] = str(time_elapsed)
config['extra']['total_epochs'] = str(len(history.history['loss']))
with open(workdir.joinpath('config.ini'), 'w') as configfile:
  config.write(configfile)

# Generate examples 
print("Generating examples...")
my_examples_folder = workdir.joinpath('audio_examples')
audio_list = os.listdir(my_audio)
os.makedirs(my_examples_folder, exist_ok=True)
# take a random subset of audio_list
random.seed(42)
random_subset = random.sample(audio_list, min(len(audio_list), 30))  # for example, take 10 random files
for f in random_subset:
  print("Examples for {}".format(os.path.splitext(f)[0])) 
  file_path = my_audio.joinpath(f) 
  s, fs = librosa.load(file_path, duration=example_length, sr=None)
  # Get the CQT magnitude
  print("Calculating CQT")
  C_complex = librosa.cqt(y=s, sr=fs, hop_length= hop_length, bins_per_octave=bins_per_octave, n_bins=n_bins)
  C = np.abs(C_complex)

  C_32 = C.astype('float32')
  y_inv_32 = librosa.griffinlim_cqt(C, sr=fs, n_iter=n_iter, hop_length=hop_length, bins_per_octave=bins_per_octave, dtype=np.float32)
  
  ## Generate the same CQT using the model
  my_array = np.transpose(C_32)
  test_dataset = tf.data.Dataset.from_tensor_slices(my_array).batch(batch_size).prefetch(AUTOTUNE)
  output = tf.constant(0., dtype='float32', shape=(1,n_bins))
  print("Working on regenerating cqt magnitudes with the DL model")
  for step, x_batch_train in enumerate(test_dataset):
    reconstructed = vae(x_batch_train)
    output = tf.concat([output, reconstructed], 0)

  output_np = np.transpose(output.numpy())
  output_inv_32 = librosa.griffinlim_cqt(output_np[1:], 
    sr=fs, n_iter=n_iter, hop_length=hop_length, bins_per_octave=bins_per_octave, dtype=np.float32)
  if normalize_examples:
    output_inv_32 = librosa.util.normalize(output_inv_32)
  print("Saving audio files...")
  my_audio_out_fold = my_examples_folder / os.path.splitext(f)[0]
  os.makedirs(my_audio_out_fold, exist_ok=True)

  sf.write(my_audio_out_fold / 'original.wav', s, sample_rate)
  sf.write(my_audio_out_fold / 'original-icqt+gL.wav', y_inv_32, sample_rate)
  sf.write(my_audio_out_fold / 'VAE-output+gL.wav', output_inv_32, sample_rate)

#Generate plots for losses 
print("Generating loss plots...")
history_dict = history.history

# Create figure with subplots (now 4 subplots to include centroid)
fig, axes = plt.subplots(4, 1, figsize=(10, 15))

# Plot 1: Total Loss
axes[0].plot(history_dict['loss'], linewidth=2, color='blue')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Total Loss')
axes[0].set_title('Total Loss over Training')
axes[0].grid(True, alpha=0.3)

# Plot 2: Reconstruction Loss
if 'reconstruction_loss' in history_dict:
    axes[1].plot(history_dict['reconstruction_loss'], linewidth=2, color='green')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss over Training')
    axes[1].grid(True, alpha=0.3)

# Plot 3: KL Loss
if 'kl_loss' in history_dict:
    axes[2].plot(history_dict['kl_loss'], linewidth=2, color='red')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('KL Loss')
    axes[2].set_title('KL Divergence Loss over Training')
    axes[2].grid(True, alpha=0.3)

# Plot 4: Centroid Loss
if 'centroid_loss' in history_dict:
    axes[3].plot(history_dict['centroid_loss'], linewidth=2, color='orange')
    axes[3].set_xlabel('Epochs')
    axes[3].set_ylabel('Centroid Loss')
    axes[3].set_title('Spectral Centroid Loss over Training')
    axes[3].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(workdir.joinpath('my_history_plot.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved loss plot to: {workdir / 'my_history_plot.pdf'}")

# Also create a combined plot
fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(history_dict['loss'], label='Total Loss', linewidth=2)
if 'reconstruction_loss' in history_dict:
    ax.plot(history_dict['reconstruction_loss'], label='Reconstruction Loss', linewidth=2)
if 'kl_loss' in history_dict:
    # Scale KL loss by beta for visualization
    kl_scaled = np.array(history_dict['kl_loss']) * kl_beta
    ax.plot(kl_scaled, label=f'KL Loss × β ({kl_beta})', linewidth=2)
if 'centroid_loss' in history_dict and spectral_centroid_weight > 0:
    centroid_scaled = np.array(history_dict['centroid_loss']) * spectral_centroid_weight
    ax.plot(centroid_scaled, label=f'Centroid Loss × weight ({spectral_centroid_weight})', linewidth=2)

ax.set_xlabel('Epochs', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('VAE Training Loss Components', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig2.savefig(workdir.joinpath('my_history_plot_combined.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved combined loss plot to: {workdir / 'my_history_plot_combined.pdf'}")

# Print final loss values
print("\n=== Final Loss Values ===")
print(f"Total Loss: {history_dict['loss'][-1]:.6f}")
if 'reconstruction_loss' in history_dict:
    print(f"Reconstruction Loss: {history_dict['reconstruction_loss'][-1]:.6f}")
if 'kl_loss' in history_dict:
    print(f"KL Loss: {history_dict['kl_loss'][-1]:.6f}")
    print(f"KL Loss × β: {history_dict['kl_loss'][-1] * kl_beta:.6f}")
if 'centroid_loss' in history_dict:
    print(f"Centroid Loss: {history_dict['centroid_loss'][-1]:.6f}")
    print(f"Centroid Loss × weight: {history_dict['centroid_loss'][-1] * spectral_centroid_weight:.6f}")

plt.close('all')

print('bye...')