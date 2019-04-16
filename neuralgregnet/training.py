import os,glob
from tensorflow import keras
import tensorflow as tf
import pickle
import numpy as np
def training(model,in_set, out_set,model_name,nb_epochs,learning_rate=0.001,batch_size=None,loss="mse",validation_split=None,clear=False):
    """
    Handles the training of a model and sves a backup file in the networks/ directory
    
    Args:
        model: model to train
        in_set: input data used for the training
        out_set: output set used for the training
        model_name: name of the model. The backup file will be called after this name (model_name.ckpt)
        nb_epochs: Number of epochs
        learning_rate: Learning rate of the Adam optimizer
        batch_size:
        loss: Loss function
        clear: If true, delete existing old backups.
    
    Return:
        Trained model
    """
    if not os.path.exists("networks"):
        os.mkdir("networks") 
    backup_name = "networks/{}".format(model_name)
    backups = glob.glob("{}*".format(backup_name))
    if not len(backups) or clear:
        if clear:
            for ff in backups:
                os.remove(ff)
        model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),loss=loss)
        shuffle = np.random.permutation(len(in_set))
        model.fit(in_set[shuffle],out_set[shuffle],epochs=nb_epochs,batch_size=batch_size,validation_split=validation_split)
        save_model(model,backup_name)
    else:
        model.load_weights(backup_name+"weights")
    return model

def save_model(model,model_name, withWeights=True):
    model_yaml = model.to_yaml()
    with open("{}.yaml".format(model_name), "w") as yaml_file:
        yaml_file.write(model_yaml)
    if withWeights:
        model.save_weights(model_name+"weights")
            
def load_model(model_name, withWeights=True):
    with open('{}.yaml'.format(model_name), 'r') as yaml_file: 
        model = keras.models.model_from_yaml(yaml_file.read())
    if withWeights:
        model.load_weights(model_name+"weights")
    return model
