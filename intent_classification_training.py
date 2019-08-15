"""
A few words
Is this a perfect intent classification in NLP for Vietnamese language ????
Yes n No !
Why Yes ? Well, cause so fall it done everything that I expect
Why No ? Cause I'm not a professional in Python language nor expert in ML/DL/AI, I'm just an amatuer, try to bring what my best for this world...And obviously there are "some" copy paste while I wander on the Internet (Well, We all do !)
That's all for shitty thing
"""
"""
There are 4 module in this particular code:
1. preprocessing
2. train_model (obviously it come along with the save_model)
3. load_model
4. predict
"""
# Why we need to import all this shit ? Well you can delete all of this shit if you wish !
import numpy as np
import pandas as pd
import random
import re
import sklearn
import os
from underthesea import word_tokenize

from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
from keras import regularizers

from distutils.version import LooseVersion, StrictVersion
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from IPython.display import display, Image


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

class preprocessing:
	def __init__(self, datasetpath):
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
	def datapreprocessed(self):
		# Get the main idea of data
		intent, unique_intent, sentences = self.load_dataset(self.datasetpath)
		# Clean training data
		cleaned_words = self.preprocessData(sentences)
		word_tokenizer = self.create_tokenizer(cleaned_words)
		vocab_size = len(word_tokenizer.word_index) + 1
		max_length = self.max_length(cleaned_words)
		# Encode training data
		encoded_doc = self.encoding_doc(word_tokenizer, cleaned_words)
		padded_doc = self.padding_doc(encoded_doc, max_length)
		# Encode label
		output_tokenizer = self.create_tokenizer(unique_intent)
		encoded_output = self.encoding_doc(output_tokenizer, intent)
		encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)
		output_one_hot = self.one_hot(encoded_output)
		# We now have the train, val, test as 60%, 20%, 20% 
		train_X, test_X, train_Y, test_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.2,random_state = 1)
		train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, shuffle = True, test_size = 0.25,random_state = 1)
		return word_tokenizer, vocab_size, max_length, train_X, train_Y, test_X, test_Y, val_X, val_Y, unique_intent

class train_model:
	def __init__(self, model_name,list_dropout_1, list_dropout_2 ,model_pretrain, model_path, DIM_SIZE ,word_tokenizer, vocab_size, max_length, train_X, train_Y, test_X, test_Y, epoch, batch_size,
		useDropoutList):
		self.model_name = model_name
		self.list_dropout_1 = list_dropout_1
		self.list_dropout_2 = list_dropout_2
		self.model_pretrain = model_pretrain
		self.model_path = model_path
		self.DIM_SIZE = DIM_SIZE
		self.word_tokenizer = word_tokenizer
		self.vocab_size = vocab_size
		self.max_length = max_length
		self.train_X = train_X
		self.train_Y = train_Y
		self.val_X = val_X
		self.val_Y = val_Y
		self.test_X = test_X
		self.test_Y = test_Y
		self.epoch = epoch
		self.batch_size = batch_size
		self.useDropoutList = useDropoutList

	def WE_model(self, number_words, dim, word_vectors, word_tokenizer):
		from sklearn.decomposition import PCA
		MAX_NB_WORDS = number_words
		WV_DIM = 400
		nb_words = min(MAX_NB_WORDS, len(word_vectors.vocab))
		# print(nb_words)
		# we initialize the matrix with random numbers
		random.seed(1)
		wv_matrix = (np.random.rand(nb_words, WV_DIM) - 0.5) / 5.0
		j = 0
		for word, i in word_tokenizer.word_index.items():
			if i >= MAX_NB_WORDS:
				break
			try:
				embedding_vector = word_vectors[word]
				# words not found in embedding index will be all-zeros.
				wv_matrix[i] = embedding_vector[:WV_DIM]
				j += 1
			except KeyError:
				print("There is a word non-exist in Vocab of EM: {0}".format(word))
				pass        
		print("There is {0} words in dataset match with {1} words in Embedding matrix".format(j,len(word_vectors.vocab)))
		pca = PCA(n_components = dim)
		wv_matrix = pca.fit(wv_matrix.T)
		wv_matrix = np.transpose(wv_matrix)
		return wv_matrix
	def init_EM(self):
		import gensim
		model_WE = model_pretrain
		if LooseVersion(gensim.__version__)>=LooseVersion("1.0.1"):
			from gensim.models import KeyedVectors
			word2vec_model = KeyedVectors.load_word2vec_format(model_WE, binary=True)
		else:
			from gensim.models import Word2Vec
			word2vec_model = Word2Vec.load_word2vec_format(model_WE, binary = True)
		word_vectors = word2vec_model.wv
		# We create Embedding matrix with (how much unique word in your dataset x which dim you want - MAX = 400 )
		wv_matrix = self.WE_model(len(word_tokenizer.word_index) + 1, DIM_SIZE, word_vectors, word_tokenizer)
		return wv_matrix
	def create_model_2(self,vocab_size, wv_dim, weight_matrix ,max_length, regularize_term_input,regularize_term_hidden):
		if self.useDropoutList == False:
			dropout_input = 0.5
			dropout_hidden = 0.5
		else:
			dropout_input = regularize_term_input
			dropout_hidden = regularize_term_hidden
		model = Sequential()
		model.add(Embedding(vocab_size, wv_dim, weights=[weight_matrix], input_length = max_length, trainable=False))
		model.add(Dropout(dropout_input))
		model.add(Bidirectional(LSTM(wv_dim)))
		# model.add(LSTM(wv_dim, return_sequences = True))
		# model.add(LSTM(wv_dim))
		model.add(Dropout(dropout_hidden))
		model.add(Dense(32, activation = "relu"))
		model.add(Dense(len(train_Y[0]), activation = "softmax"))

		return model
	def create_model(self,vocab_size, wv_dim, weight_matrix ,max_length, regularize_term):
		if self.useDropoutList == False:
			dropout = 0.5
		else:
			dropout = regularize_term
		model = Sequential()
		model.add(Embedding(vocab_size, wv_dim, weights=[weight_matrix], input_length = max_length, trainable=False))
		# model.add(Bidirectional(LSTM(wv_dim)))
		model.add(LSTM(wv_dim))
		model.add(Dropout(dropout))
		model.add(Dense(32, activation = "relu"))
		model.add(Dense(len(train_Y[0]), activation = "softmax"))

		return model
	# Use this to save your graph to enjoy how stupid your model is
	def save_graph_model(self,hist, filename):
	# Plot training & validation accuracy values
		import matplotlib.pyplot as plt
		plt.subplot(211)
		plt.plot(hist.history['acc'])
		plt.plot(hist.history['val_acc'])
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Val'], loc='upper left')
		# plt.savefig(filename + ' Accuracy.png')

		# Plot training & validation loss values
		plt.subplot(212)
		plt.plot(hist.history['loss'])
		plt.plot(hist.history['val_loss'])
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Val'], loc='upper left')
		# plt.savefig(filename +' Loss.png')
		plt.subplots_adjust(hspace = 0.4,right =1.5,top =1.75)
		# plt.subplots_adjust(hspace = 0.7,top =1.25)
		plt.savefig(filename+ '.png',bbox_inches = 'tight')
		plt.clf()
	#     plt.show()
	def show_model(self, model):
		model.summary()
		plot_model(model, to_file= self.model_path + 'model.png', show_shapes=True,show_layer_names=True)
		# display(Image(filename='model.png'))

	def train_model_f(self, train_X, train_Y, val_X, val_Y, epoch, batch):
		wv_matrix = self.init_EM()
		nb_words, WV_DIM = wv_matrix.shape
		model = self.create_model_2(nb_words ,WV_DIM, wv_matrix, max_length, self.list_dropout_1, self.list_dropout_2)

		model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
		filename = self.model_path + self.model_name + '_dim150_do0.5.h5'
		self.show_model(model)
		# decide = input("Do you satisfy with your model ?(y/n): ")
		# if decide == 'y' :
		checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
		hist = model.fit(train_X, train_Y, epochs = epoch, batch_size = batch, validation_data = (val_X, val_Y), callbacks = [checkpoint])
		self.save_graph_model(hist, filename)

	# def train_model_w_regu(self, list_regu, train_X, train_Y, val_X, val_Y, epoch, batch):
	#     #Choices model with Embedding matrix = 500 x 200
	#     wv_matrix = self.init_EM()
	#     nb_words, WV_DIM = wv_matrix.shape
	#     for i in range(len(list_regu)):
	#         model = self.create_model(nb_words ,WV_DIM, wv_matrix, max_length, list_regu[i])
	#         model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
			
	#         filename = self.model_path + 'model_lambda{0}.h5'.format(i)
	#         checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

	#         hist = model.fit(train_X, train_Y, epochs = epoch, batch_size = batch, validation_data = (val_X, val_Y), callbacks = [checkpoint])
	#         self.save_graph_model(hist, filename)
	def train_model_w_2_dropouts(self,train_X, train_Y, val_X, val_Y, epoch, batch):
		wv_matrix = self.init_EM()
		nb_words, WV_DIM = wv_matrix.shape
		for i in range(len(self.list_dropout_1)):
			model = self.create_model_2(nb_words ,WV_DIM, wv_matrix, max_length, self.list_dropout_1[i], self.list_dropout_2[i])
			model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
			
			filename = self.model_path + self.model_name + '_do1_{0}_do2_{1}.h5'.format(self.list_dropout_1[i], self.list_dropout_2[i])
			checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

			hist = model.fit(train_X, train_Y, epochs = epoch, batch_size = batch, validation_data = (val_X, val_Y), callbacks = [checkpoint])
			self.save_graph_model(hist, filename)

	# def train_model_w_dropouts(self,train_X, train_Y, val_X, val_Y, epoch, batch):
	#     wv_matrix = self.init_EM()
	#     nb_words, WV_DIM = wv_matrix.shape
	#     for i in range(len(self.list_dropout)):
	#         model = self.create_model(nb_words ,WV_DIM, wv_matrix, max_length, self.list_dropout[i])
	#         model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
			
	#         filename = self.model_path + self.model_name[i] + '.h5'
	#         checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

	#         hist = model.fit(train_X, train_Y, epochs = epoch, batch_size = batch, validation_data = (val_X, val_Y), callbacks = [checkpoint])
	#         self.save_graph_model(hist, filename)
	def init_train_w_dropout(self):
		self.train_model_w_2_dropouts(self.train_X, self.train_Y, self.val_X, self.val_Y, self.epoch, self.batch_size)
	def init_train(self):
		self.train_model_f(self.train_X, self.train_Y, self.val_X, self.val_Y, self.epoch, self.batch_size)

class load_models:
	def __init__(self, model_path, train_X, train_Y, test_X, test_Y, val_X, val_Y):
		self.model_path = model_path
		self.train_X = train_X
		self.train_Y = train_Y
		self.test_X = test_X
		self.test_Y = test_Y
		self.val_X = val_X
		self.val_Y = val_Y

	def check_all_model(self):
		decision = input("Do you want to run test check on all model(y/n):")
		if decision == 'y':
			for file in os.listdir(model_path):
				if file.endswith(".h5"):
					model = load_model(model_path + file)
					print("_____load " + file)
					print(model.metrics_names)
					print(model.test_on_batch(self.train_X, self.train_Y, sample_weight=None))
					print(model.test_on_batch(self.val_X,  self.val_Y, sample_weight=None))
					print(model.test_on_batch(self.test_X,  self.test_Y, sample_weight=None))
					# print("F1 score on test:")
					# y_pred = model.predict(self.test_X)
					# y_pred = (y_pred > 0.5)
					# watch = sklearn.metrics.classification_report(self.test_Y, y_pred)
					# print(watch)
	def get_model(self):         
		for file in os.listdir(model_path):
			if file.endswith(".h5"):
				print(os.path.join(file))    
		model_name = input("Which model you want to load and run test(type specific model.ex: model.h5): ")
		model = load_model(self.model_path + model_name)
		print("_____load" + model_name)
		print(model.metrics_names)
		print("On train set:")
		print(model.test_on_batch(self.train_X, self.train_Y, sample_weight=None))
		print("On val set:")
		print(model.test_on_batch(self.val_X, self.val_Y, sample_weight=None))
		print("On test set:")
		print(model.test_on_batch(self.test_X, self.test_Y, sample_weight=None))
		print("F1 score on test:")
		y_pred = model.predict(self.test_X)
		y_pred = (y_pred > 0.5)
		watch = sklearn.metrics.classification_report(self.test_Y, y_pred)
		print(watch)
		return model
class predict:
	def __init__(self, my_model,datasetpath, unique_intent, word_tokenizer, max_length, preprocess):
		self.datasetpath = datasetpath
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
	# def load_response_dataset(filename):
	#     df_res = pd.read_excel(filename, sheet_name="tam_bo", encoding="utf8") 
	#     return df_res

	# def response(classes):
	#     df_res = load_response_dataset(datasetpath)
	#     s_res = random.choice(list(df_res['Response'][df_res['Tag']==classes]))
	#     print(s_res)

	def get_final_output(self,pred, classes):
		predictions = pred[0]
		classes = np.array(classes)
		ids = np.argsort(-predictions)
		classes = classes[ids]
		predictions = -np.sort(-predictions)

		for i in range(3):
			print("%s has confidence = %s" % (classes[i], (predictions[i])))
	#     response(classes[0])
	#     print(classes[0])
	def run(self):
		print('You are in test zone,if you want to exit type q:')
		while(1):
			text = input()    
			if text == 'q':
				break
			pred = self.predictions(text)
			self.get_final_output(pred, self.unique_intent)

class response:
	def __init__(self, datasetpath, intent, locate, asklocate):
		self.datasetpath = datasetpath
		self.intent = intent
		self.locate = locate
		self.asklocate = asklocate
	def get_response(self, filename, intent):
		dresponse = pd.read_excel(filename, sheet_name ='Tag_Response')
		response = dresponse[intent][0]
		return response
	def get_location(self, locate):
		dlocation = pd.read_excel(filename, sheet_name = 'listofWorkingplace')
		return dlocation.loc[dlocation.index[dlocation['Name'] == locate].tolist()[0],'Address']
	def return_response(self):
		if self.asklocate == True:
			return "Nếu bạn ở {0} bạn có thể tới {1} để làm hộ chiếu".format(self.locate,self.get_location(self.locate))
		else:
			return self.get_response(self.datasetpath, self.intent)


# A few thing you need to list by hand( cause I can't know what is in your god damn mind !)
"""
Your data set must be excel or csv, and it contain 3 sheet : dataset_XY, Label_tag, Response
dataset_XY: 
Questions -- Label
Your shit -- Your shit

Label_tag:
Label     -- Tag
Your shit -- Your shit

Response:
Locate    -- Address
Your shit -- Your shit
"""
datasetpath='data/dataset_XY_XLS_updatedbytho_ver2_2.xlsx'  # str - that path or file that contain your dataset, and we only accept csv of excel format, if you use any other format, re-format it or don't use this code ! ex: 'data/dataset_XY_XLS_updatedbytho_ver2_1.xlsx'
model_pretrain ='data/baomoi.model.bin'  # We will use w2v model pretrained ex:'data/baomoi.model.bin'

model_path = 'keras_model/'# The model and path which you want to save ex:"keras_model/"

model_name = 'model_w_PCA' # The name of model which you decide, if you want to use list dropout and save many model,
# Put names you want in to a list. ex: model_name= ['model_this', 'model_that']
useDropoutList = False
list_dropout_input = [ 0.2, 0.5, 0.7, 0.8, 0.2, 0.2, 0.8, 0.5]
list_dropout_hidden = [ 0.2, 0.5, 0.7, 0.8, 0.8, 0.5, 0.5, 0.8]
DIM_SIZE = 100 #You need to decide how large is your vector represent each word, is pretrained model max is 400, I only use first 200 dimension, cause I ran into issue call: out OF memory which lead to my god damn laptop just stop, like Thanos just make a snap on my shitty laptop (WHYYYYYY, WHAT DID MY LAPTOP EVER DONE TO YOU MY LORD ????)
epoch = 500 # How many time you want your model train over again your data

intent = 'Chào_hỏi'
locate = 'An_Giang' 
# if not use locate, set False
asklocate = True
"""
When train : comment model and prediction
When test model: comment train_model and predict and use check_all_model to run all model you have, or just get specific model to predcit or do some shit
When predict model : comment train_model, when being ask:"Do you want to run test check on all model(y/n):" type 'n' and
then select which model you want to use
"""      

preprocess = preprocessing(datasetpath)
(word_tokenizer, vocab_size, max_length, train_X, train_Y, test_X, test_Y, val_X, val_Y ,unique_intent)= preprocess.datapreprocessed()
batch_size = train_X.shape[0] # How many samples next you want to update the weight
training = train_model(model_name,
						list_dropout_input, 
						list_dropout_hidden, 
						model_pretrain,
						model_path, 
						DIM_SIZE ,
						word_tokenizer, 
						vocab_size,
						max_length, 
						train_X, train_Y, test_X, test_Y, epoch, batch_size, useDropoutList= useDropoutList)
training.init_train()
# training.init_train_w_dropout()

# model = load_models(model_path,train_X, train_Y, test_X, test_Y, val_X, val_Y)
# model.check_all_model()
# my_model = model.get_model()

# prediction = predict(my_model, datasetpath, unique_intent, word_tokenizer, max_length, preprocess)
# prediction.run
# responses = response(datasetpath, intent ,locate, asklocate )


