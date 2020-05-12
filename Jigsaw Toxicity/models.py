from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, CuDNNLSTM, BatchNormalization, Dropout, concatenate, dot
from keras.optimizers import RMSprop


def get_model(embedding,
              num_lstm=1,
              num_hidden=1,
              dim_hidden=128,
              num_classes=2,
              max_seq_len=35,
              dropout_rate=0.5,
              lr=1e-3,
              clipping=5.0,
              use_cudnn=True):
    model_in = Input(shape=(max_seq_len,), dtype='int32')

    embedding_layer = Embedding(embedding.shape[0],
                                embedding.shape[1],
                                mask_zero=False,
                                weights=[embedding],
                                trainable=False,
                                input_length=max_seq_len)
    hidden = embedding_layer(model_in)

    for j in range(num_lstm):
        lstm_cell = CuDNNLSTM if use_cudnn else LSTM
        lstm_layer = lstm_cell(dim_hidden, return_sequences=(j != num_lstm - 1))
        hidden = lstm_layer(hidden)

    for _ in range(num_hidden):
        hidden = Dense(dim_hidden, activation='relu')(hidden)
        hidden = BatchNormalization()(hidden)
    hidden = Dropout(dropout_rate)(hidden)

    model_out = Dense(num_classes, activation='softmax')(hidden)

    ret_model = Model(model_in, model_out)
    ret_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr, clipnorm=clipping), metrics=['acc'])

    return ret_model
