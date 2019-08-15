import numpy as np
import pandas as pd
import random
import re

import os
os.environ['KERAS_BACKEND'] = 'theano'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model

from underthesea import word_tokenize

pd.options.mode.chained_assignment = None
class preprocessing:
    def __init__(self, datasetpath=''):
        self.datasetpath = datasetpath

    def load_dataset(self,filename):
        ddata = pd.read_excel(filename ,sheet_name='dataset_XY')
        dlabel = pd.read_excel(filename,sheet_name='Label_tag')

        intent = ddata['Label']
        for idx, i in enumerate(intent):
            intent[idx] = dlabel.loc[i-1,"Tag"]
        unique_intent = list(dlabel["Tag"])
        sentences = list(ddata["Questions"])

        return (intent, unique_intent, sentences)
    # This will return from your dataset
    
    def viToken(self,word, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', use_filter=True):
        if use_filter == False:
            return word_tokenize(word.lower(), format="text")
        else:
            table = str.maketrans(filters, len(filters)*" ")
            word = word.translate(table)
            return word_tokenize(word.lower(), format="text")

    def preprocessData(self, sentences):
        words = []
        for s in sentences:
            w = self.viToken(s)
            words.append(w)
        return words 
    
    # This will clean your sentence from your shit, for that the code can Tokenize it easier
    
    def create_tokenizer(self, words):
        token = Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@[\]^`{|}~')
        token.fit_on_texts(words)
        return token
    def max_length(self, cleaned_words):
        return(len(max(cleaned_words, key = len).split()))
    
    # We will create token, and find out how many token and which is the maximum lenght in all your dataset
    
    def encoding_doc(self, token, words):
        return(token.texts_to_sequences(words))
    
    # Why encoding ? Cause we want this : [1] not this [hộ chiếu] to feed into our model    
    def padding_doc(self, encoded_doc, max_length):
        return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))
    
    #W Why padding ? Cause we want all sentence we feed have THE SAME LENGHT
    def one_hot(self, encode):
        o = OneHotEncoder(sparse = False)
        return(o.fit_transform(encode))
    def datapreprocessed_1(self):
        #Get the main idea of data
        intent, unique_intent, sentences = self.load_dataset(self.datasetpath)
        # Clean training data
        cleaned_words = self.preprocessData(sentences)
        word_tokenizer = self.create_tokenizer(cleaned_words)
        vocab_size = len(word_tokenizer.word_index) + 1
        max_length = self.max_length(cleaned_words)
        # Encode training data
        # encoded_doc = self.encoding_doc(word_tokenizer, cleaned_words)
        # padded_doc = self.padding_doc(encoded_doc, max_length)
        # # Encode label
        # output_tokenizer = self.create_tokenizer(unique_intent)
        # encoded_output = self.encoding_doc(output_tokenizer, intent)
        # encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)
        # output_one_hot = self.one_hot(encoded_output)
        # We now have the train, val, test as 60%, 20%, 20% 
        # train_X, _, train_Y, _ = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0,random_state = 1)
        # _, test_X, _, test_Y = train_test_split(train_X, train_Y, shuffle = True, test_size = 0.2,random_state = 2)
        # _, val_X, _, val_Y = train_test_split(train_X, train_Y, shuffle = True, test_size = 0.2,random_state = 3)
        return word_tokenizer, vocab_size, max_length, unique_intent
    
class load_models:
    def __init__(self, model_path):
        self.model_path = model_path

    def get_model(self, model_name):         
        for file in os.listdir(self.model_path):
            if file.endswith(".h5"):
                print("Models found in ", self.model_path, " : ", os.path.join(file))
            else:
            	print("Not found model *.h5 in ", self.model_path)   

        print("Using model: ", model_name)
        model = load_model(self.model_path + model_name)
        
        return model

class predict:
    def __init__(self, my_model, unique_intent, word_tokenizer, max_length, preprocess):
        self.unique_intent = unique_intent
        self.word_tokenizer = word_tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.my_model= my_model
    
    def predictions(self,text):
        test_word = self.preprocess.viToken(text)
        test_ls = self.preprocess.encoding_doc(self.word_tokenizer,[test_word])
        print("test_word: ", test_word)
        print("test_ls: ", test_ls)
        x = self.preprocess.padding_doc(test_ls, self.max_length)
        pred = self.my_model.predict_proba(x)
        return pred

    def get_final_output(self, pred, classes):
        predictions = pred[0]
        classes = np.array(classes)
        ids = np.argsort(-predictions)
        classes = classes[ids]
        predictions = -np.sort(-predictions)

        for i in range(3):
            print("%s has confidence = %s" % (classes[i], (predictions[i])))
        return classes[0], predictions[0]
    #     response(classes[0])
    #     print(classes[0])

    def get_intent_confidence(self, text):
        pred = self.predictions(text)        
        intent, confidence = self.get_final_output(pred, self.unique_intent)
        return intent, confidence

my_model = load_model('keras_model/'+'model_lambda2.h5') #let's load model outside first. So for every prediction, you dont have to load_model again.
# my_model._make_predict_function()
datasetpath='data/dataset_XY_XLS_updatedbytho_ver2_2.xlsx'
preprocess = preprocessing(datasetpath)
(word_tokenizer, vocab_size, max_length, unique_intent)= preprocess.datapreprocessed_1()

def ic_predict(message):
        preprocess = preprocessing()
        pred = predict(my_model, unique_intent, word_tokenizer, max_length, preprocess)
        intent, confidence = pred.get_intent_confidence(message)

        return intent, confidence


