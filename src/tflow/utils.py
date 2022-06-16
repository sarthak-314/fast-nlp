import tensorflow_addons as tfa
import tensorflow as tf 

from tqdm.auto import tqdm 
import gc

from ..core import green
import wandb

def get_model_average(model, weight_files): 
    model_weights = []
    for weight_file in tqdm(weight_files): 
        model.load_weights(weight_file)
        model_weights.append(model.get_weights())
    
    model.set_weights([
        (sum(w)/len(weight_files))
        for w in zip(*model_weights)
    ])
    del model_weights; _ = gc.collect()
    return model

def build_hidden_layer(hidden_layer_units=[], hidden_dropout=0.10, activation_str='mish', name='hidden_layer'): 
    if not hidden_layer_units:
        return tf.keras.layers.Lambda(lambda x: x)
    activation_fn = {'mish': tfa.activations.mish, None: None, 'gelu': tf.keras.activations.gelu}[activation_str]
    hidden_layers = []
    for units in hidden_layer_units: 
        hidden_layers.append(tf.keras.layers.Dropout(hidden_dropout))
        hidden_layers.append(tf.keras.layers.Dense(units=units, activation=activation_fn))
    return tf.keras.Sequential(hidden_layers, name=name)

def load_model_weights(model, model_weights): 
    if model_weights is None:
        return
    weights_file, run_name = model_weights
    print(f'Loading model weights from {green(weights_file)}')
    weights_path = wandb.restore(weights_file, run_name).name
    model.load_weights(weights_path)