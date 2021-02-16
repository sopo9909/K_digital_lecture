import pandas as pd
import numpy as np
import re
import time
import tensorflow_datasets as tfds
import tensorflow as tf
import json
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from preprocessing import *
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.random.set_seed(99)
index_inputs = np.load(open('train_inputs.npy','rb'), allow_pickle=True)
index_outputs = np.load(open('train_outputs.npy','rb'), allow_pickle=True)
index_targets = np.load(open('train_targets.npy','rb'), allow_pickle=True)
prepro_configs = json.load(open('data_configs.json'))
BATCH_SIZE = 2 
MAX_SEQUENCE =25
EPOCH =30
UNITS =1024
EMBEDDING_DIM = 256
VALIDATION_SPLIT = 0.1
char2idx = prepro_configs['char2idx']
idx2char = prepro_configs['idx2char']
std_index = prepro_configs['std_symbol']
end_index = prepro_configs['end_symbol']
vocab_size = prepro_configs['vocab_size']
class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder,self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, 
                                         return_sequences= True,
                                         return_state= True,
                                         recurrent_initializer= 'glorot_uniform'
                                        )
    def call(self,x,hidden): 
        x = self.embedding(x)
        output,state = self.gru(x, initial_state = hidden)
        return output, state
    def initialize_hidden_state(self, inp):
        return tf.zeros((tf.shape(inp)[0],self.enc_units))
class BandanauAttention(tf.keras.layers.Layer):
    def __init__(self,units):  
        super(BandanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    def call(self, query, values): 
        hidden_with_time_axis =  tf.expand_dims(query,1)
        score = self.V(tf.nn.tanh(self.W1(values)+self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score,axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis =1) 
        return context_vector, attention_weights
class Decoder(tf.keras.layers.Layer):
    def __init__(self,vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        
        self.batch_size = batch_size
        self.dec_units =  dec_units
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences = True,
                                        return_state = True,
                                        recurrent_initializer = 'glorot_uniform'
                                       )
        self.fc = tf.keras.layers.Dense(self.vocab_size)
        self.attention = BandanauAttention(self.dec_units)
        
    def call(self, x, hidden, enc_output):
        context_vector,attention_weights = self.attention(hidden, enc_output) 
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector,1),x], axis =-1)  
        output,state = self.gru(x)
        output = tf.reshape(output, (-1,output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights
optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction= 'none')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'accuracy')

def loss(real, pred):  
    mask = tf.math.logical_not(tf.math.equal(real,0)) 
    loss_ = loss_object(real,pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

def accuracy(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    mask = tf.expand_dims(tf.cast(mask, dtype = pred.dtype), axis = -1)
    pred *= mask
    acc = train_accuracy(real, pred)
    return tf.reduce_mean(acc)
class seq2seq(tf.keras.Model):
    def __init__(self,vocab_size, embedding_dim, enc_units, dec_units, batch_size, end_token_idx = 2):
        super(seq2seq, self).__init__()
        self.end_token_idx = end_token_idx
        self.encoder = Encoder(vocab_size, embedding_dim, enc_units, batch_size)
        self.decoder = Decoder(vocab_size, embedding_dim, dec_units, batch_size)
    def call(self,x): 
        inp, tar = x
        enc_hidden = self.encoder.initialize_hidden_state(inp)
        enc_output, enc_hidden = self.encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        predict_tokens  = list()
        for t in range(0, tar.shape[1]):
            dec_input = tf.dtypes.cast(tf.expand_dims(tar[:,t],1),tf.float32) #특정 state 디코더 입력값
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            predict_tokens.append(tf.dtypes.cast(predictions, tf.float32))
        result = tf.stack(predict_tokens, axis = 1)
        return result
    def inference(self, x): 
        inp = x
        enc_hidden = self.encoder.initialize_hidden_state(inp)
        enc_output,enc_hidden = self.encoder(inp,enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([char2idx[std_index]],1)
        predict_tokens = list()
        for t in range(0, MAX_SEQUENCE):
            predictions,dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            predict_token = tf.argmax(predictions[0])
            if predict_token == self.end_token_idx : 
                break
            predict_tokens.append(predict_token)
            dec_input = tf.dtypes.cast(tf.expand_dims([predict_token],0),tf.float32)
        return tf.stack(predict_tokens, axis =0).numpy()
model = seq2seq(vocab_size, EMBEDDING_DIM, UNITS, UNITS,BATCH_SIZE, char2idx[end_index])
model.compile(loss = loss, optimizer= tf.keras.optimizers.Adam(1e-3), metrics =  [accuracy])
path = 'data_out/seq2seq_ban'
if not(os.path.isdir(path)):
    os.makedirs(os.path.join(path))
chk_path = path + '/weights.h5'
callback = ModelCheckpoint( chk_path, monitor = 'val_accuracy', verbose =1, save_best_only= True,
                            save_weights_only =True)
earlystop = EarlyStopping(monitor ='val_accuracy', min_delta = 0.001, patience =10)

history = model.fit([index_inputs, index_outputs], index_targets,
                   batch_size =BATCH_SIZE,
                   epochs = EPOCH,
                   validation_split= 0.2,
                   callbacks = [earlystop, callback])
SAVE_FILE_NM = "weights.h5"
FILTERS = "([~.,!?\"':;)(])"
CHANGE_FILTER = re.compile(FILTERS) # 미리 Complie
PAD, PAD_INDEX = "<PAD>", 0 # 패딩 토큰
STD, STD_INDEX = "<SOS>", 1 # 시작 토큰
END, END_INDEX = "<END>", 2 # 종료 토큰
UNK, UNK_INDEX = "<UNK>", 3 # 사전에 없음
MARKER = [PAD,STD,END,UNK]
MAX_SEQUNECE = 25
model.load_weights(os.path.join('data_out/seq2seq_ban/weights.h5'))
def enc_processing(value, dictionary):
    sequences_input_index = []
    sequences_length = []
    for sequence in value :
        sequence = re.sub(CHANGE_FILTER,"",sequence)
        sequence_index = []
        for word in sequence.split(): # 공백 기준으로 word를 구분
            if dictionary.get(word) is not None : # 사전에 있으면
                sequence_index.extend([dictionary[word]]) # index 값 쓰고
            else:
                sequence_index.extend([dictionary[UNK]])
        # 길이 제한
        if len(sequence_index) > MAX_SEQUNECE:
            sequence_index = sequence_index[:MAX_SEQUNECE]

        sequences_length.append(len(sequence_index)) # 이 문장의 길이 저장
        # Padding 추가
        # "안녕"  → "안녕,<PAD>,<PAD>,<PAD>,<PAD>"
        
        sequence_index += (MAX_SEQUNECE - len(sequence_index))*[dictionary[PAD]]
        
        sequences_input_index.append(sequence_index)

    return np.asarray(sequences_input_index), sequences_length
#query = "뭐야?"#질문
query = input('이이')
test_index_inputs , _ = enc_processing([query],char2idx)
predict_tokens =  model.inference(test_index_inputs)
print(' '.join([idx2char['%s'%t] for  t in predict_tokens]))#대답