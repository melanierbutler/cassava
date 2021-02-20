import numpy as np
import pandas as pd
from seaborn import heatmap
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adamax


class EfficientNetB4Model(object):
    
    def __init__(self, load_fp=None, img_size=380, optimizer=Adamax(lr=0.0001), loss=CategoricalCrossentropy()):
        
        if load_fp is None:
            print('Building new model.')
            self.img_size = img_size
            self.loss = loss
            self.optimizer = optimizer
            self.model = self.build_model()
            self.compiled = False
            self.freeze_change = False
            
        else:
            print('Loading model from load_fp.')
            self.model = load_model(load_fp)
            self.optimizer = self.model.optimizer
            self.loss = self.model.loss
            self.img_size = int(self.model.input.shape[1])
            if self.optimizer is None:
                self.compiled = False
            else:
                self.compiled = True
            self.freeze_change = False

        self.history = None
        self.callbacks = None
        self.data_loader = None

    def build_model(self):
        
        base = EfficientNetB4(include_top=False, weights='imagenet', input_shape=(self.img_size, self.img_size, 3))
        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        preds = Dense(5, activation='softmax')(x)
        
        return Model(inputs=base.input, outputs=preds)
        
    def freeze_all_but_top(self):
        for layer in self.model.layers[:-6]:
            layer.trainable = False
        self.freeze_change = True
            
    def unfreeze_all_layers(self):
        for layer in self.model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = True
        self.freeze_change = True
    
    def freeze_up_to_block(self, block_num:str):
        
        """Freezes all layers up to and not including the given block number.
        Block number must be entered as a string, number only."""
        
        self.unfreeze_all_layers()
        blocks = [layer.name.split('_')[0].strip('block')[0] for layer in self.model.layers]
        l = blocks.index(block_num)
        for layer in self.model.layers[:l]:
            layer.trainable = False
        print("First", l, "layers frozen.")
        
        self.freeze_change = True
    
    def compile_model(self):
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['acc'])
        self.compiled = True
        self.freeze_change = False
           
    def init_callbacks(self):
        model_save = ModelCheckpoint("Model", 
                             save_best_only = True, 
                             save_weights_only = True,
                             monitor = 'val_loss', 
                             mode = 'min', verbose = 1)
        
        early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, 
                           patience = 5, mode = 'min', verbose = 1,
                           restore_best_weights = True)
        
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
                              patience = 3, min_delta = 0.001, 
                              mode = 'min', verbose = 1)
        
        return [model_save, early_stop, reduce_lr]

        
    def train(self, data_loader=None, epochs=10):

      #check if ready to train
        if (self.data_loader is None) or (data_loader is not None):
            self.data_loader = data_loader
        if self.history is None:
            self.history = list()
        if self.callbacks is None:
            self.callbacks = self.init_callbacks()
        if not self.compiled:
            raise NotImplementedError('Model not compiled!')
        if self.freeze_change:
            raise NotImplementedError('Layers frozen or unfrozen since compilation! Recompile or set model.freeze_change to False to ignore this error.')

        self.history.append(
            self.model.fit(self.data_loader.train_gen,
                      batch_size=self.data_loader.batch_size,
                      epochs=epochs,
                      steps_per_epoch=self.data_loader.train_gen.samples // self.data_loader.train_gen.batch_size,
                      callbacks=self.callbacks,
                      validation_data=self.data_loader.val_gen,
                      validation_steps=self.data_loader.val_gen.samples // self.data_loader.val_gen.batch_size
                     )
        )

    def save_to_disk(self, save_dir, model_name='model.h5'):
        self.model.save(save_dir + model_name)
        print('Model saved to', save_dir + model_name)
        
    def hists_to_df(self):
        
        hist_df = pd.DataFrame()
        for hist in self.history:
            temp = pd.DataFrame(hist.history)
            hist_df = pd.concat([hist_df, temp])
        
        hist_df = hist_df.reset_index(drop=True)
        
        return hist_df
        
    def plot_history(self):
        hist_df = self.hists_to_df()
        hist_df[['loss', 'val_loss']].plot()
        hist_df[['acc', 'val_acc']].plot()

    def evaluate(self, test_set=False):
        if not test_set:
            gen = self.data_loader.eval_gen
        else:
            try:
                gen = self.data_loader.test_gen
            except:
                raise NotImplementedError('Data loader has no test generator! Use validation set or reinitialize Data Loader with test set.')
        gen.reset()
        preds = self.model.predict(gen, 
                                   batch_size = gen.batch_size, 
                                   steps = gen.samples // gen.batch_size + int(gen.samples % gen.batch_size > 0))
        pred_classes = np.argmax(preds, axis=1)
        true_classes = gen.classes
        class_labels = self.data_loader.label_key.values

        print(classification_report(true_classes, pred_classes, target_names=class_labels))
        heatmap(confusion_matrix(true_classes, pred_classes, normalize='true'), cmap='Blues', annot=True,xticklabels=class_labels,yticklabels=class_labels)
        plt.xlabel('predicted')
        plt.ylabel('true')
        plt.show()

        