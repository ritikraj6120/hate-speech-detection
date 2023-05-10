import numpy as np
import logging
import keras.backend as K
import tensorflow as tf

logger = logging.getLogger(__name__)

import keras.backend as K
import tensorflow as tf
from tensorflow.keras import initializers, regularizers

from tensorflow.keras import layers
# from keras.layers.convolutional import Convolution1D
import tensorflow.keras as keras
from tensorflow.keras.layers import Flatten, GlobalMaxPooling1D, Dense, Conv1D, Dropout, LSTM,GlobalAveragePooling1D, Concatenate, Layer, Add,MaxPool1D


def create_model(args, overal_maxlen, ruling_dim, category_dim, vocab, num_class):
    from tensorflow.keras.models import Model, Sequential
    import keras.backend as K
    import keras
    from tensorflow.keras.layers import Embedding, Dense, Dropout, Activation, Concatenate, Input, GlobalMaxPooling1D, GlobalAveragePooling1D, Add
    from tensorflow.keras import Sequential, Model
    from my_layers import HSMMBottom, HSMMTower, MultiHeadAttention
    ###############################################################################################################################

    ###############################################################################################################################
    # Create Model
    #

    if args.model_type == 'cls':
        raise NotImplementedError

    elif args.model_type == 'HHMM_transformer':
        import tensorflow as tf
        from tensorflow.keras.layers import Embedding, Input, concatenate
        logger.info('Building a HHMM_transfermer')

        task_num = 2 
        sequence_input_word = Input(
            shape=(overal_maxlen,), dtype='int32', name='sequence_input')
        embedded_sequences_word = Embedding(
            len(vocab), args.emb_dim, name='word_emb')(sequence_input_word)
        ruling_input = Input(shape=(ruling_dim,),
                             dtype='float32', name='rule_input')
        emb_ruling = Embedding(5, 50, name='rule_emb')(ruling_input)
        category_input = Input(shape=(category_dim,),
                               dtype='float32', name='category_input')
        emb_category = Embedding(
            5, 50, name='category_emb')(category_input)
        emb_output = concatenate(
            [embedded_sequences_word, emb_ruling, emb_category], axis=-1)  # 1 50 300 1 50 100 --->>>  batch_size 50 400
        taskid_input = Input(
            shape=(task_num,), dtype='float32', name='taskid_input')

        # xs=Permute((3,1,2))(embedded_sequences)
        # embedded_sequences_char=Dropout(0.1)(embedded_sequences_char)

        tower_outputs = []

        task_num = 2  # 
        conv_bigram = Conv1D(filters=256, kernel_size=2, activation='relu',name='conv_bigram',padding='same')(embedded_sequences_word)    #  b * 50 *256
        max_pool_bigram=MaxPool1D(pool_size=2)(conv_bigram)
        conv_trigram = Conv1D(filters=256, kernel_size=3, activation='relu',padding='same')(embedded_sequences_word)  # b * 50 * 256
        max_pool_trigram=MaxPool1D(pool_size=2)(conv_trigram)
        conv_fourgram = Conv1D(filters=256, kernel_size=4, activation='relu',padding='same')(embedded_sequences_word)# b * 50 * 256
        max_pool_fourgram=MaxPool1D(pool_size=2)(conv_fourgram)
        # Bi-GRU layers
        lstm_bigram=LSTM(units=128, dropout=0.2, return_sequences=True)(max_pool_bigram)  # b * 25 * 128
        lstm_trigram=LSTM(units=128, dropout=0.2, return_sequences=True)(max_pool_trigram) # b * 25 * 128
        lstm_fourgram= LSTM(units=128, dropout=0.2, return_sequences=True)(max_pool_fourgram) # b * 25 * 128

        # attention_bigram = SingleAttention(74)(lstm_bigram) # b * 74 * 74
        # attention_trigram =SingleAttention(74)(lstm_trigram) # b * 73 * 73
        # attention_fourgram =SingleAttention(74)(lstm_fourgram) # b * 72 * 72
        # Concatenation layer
        out = Concatenate(axis=1)([lstm_bigram,lstm_trigram,lstm_fourgram]) # b* 75 *128
        out2=Flatten()(out)
        dense1=Dense(512,kernel_regularizer=regularizers.l2(0.01),activation="relu")(out2)
        dropout1=Dropout(0.5)(dense1)
        # dense2= Dense(128,kernel_regularizer=regularizers.l2(0.01),activation="relu")(dropout1)
        outputDupliacets = tf.stack([dropout1, dropout1], axis=1)  # batch * 2 * 512
        # model = tf.keras.Model(
        #             inputs=sequence_input_word, outputs=finalout)(emb_output)
        # out = HSMMTower(units=[50,2])(expert_outputs)
        for i in range(task_num):
            tower_outputs.append(
                HSMMTower(units=[50, 2])(outputDupliacets[:, i, :])
                ) #[ b * 2, b*2]
        out = tf.matmul(tf.stack(tower_outputs, axis=-1),tf.expand_dims(taskid_input, -1))  # multiplication of 2*2 with 2* 1 --> 2*1 
        pred = tf.squeeze(out, axis=-1)  # change shape to 1 *2
        # pred = tf.nn.softmax(out,axis=-1)
        # , ruling_input, category_input
        model = Model(
            inputs=[sequence_input_word, taskid_input], outputs=pred)
        model.emb_index = 0
        model.summary()

    elif args.model_type == 'Trm':
        logger.info("Building a Simple Word Embedding Model")
        # from keras.layers import Input, Dense, concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D, Add
        input = Input(shape=(overal_maxlen,), dtype='int32')
        # input2 = Input(shape=(dim_ruling,300), dtype='float32')
        emb_output = Embedding(len(vocab), args.emb_dim,
                               name='word_emb')(input)
        # mlp_output = Self_Attention(300)(emb_output)
        mlp_output = MultiHeadAttention(300)(emb_output)
        mlp_output = Dense(300, activation='relu')(
            mlp_output)  # mlp_output = Dropout(0.2)(mlp_output)
        avg = GlobalAveragePooling1D()(mlp_output)
        max1 = GlobalMaxPooling1D()(mlp_output)
        concat = concatenate([avg, max1], axis=-1)
        dense1 = Dense(50, activation='relu')(concat)
        dense2 = Dense(50, activation='relu')(dense1)
        dropout = Dropout(0.5)(dense2)
        output = Dense(num_class, activation='softmax')(dropout)
        model = Model(inputs=input, outputs=output)
        model.emb_index = 1
        model.summary()

    logger.info('  Done')

    ###############################################################################################################################
    # Initialize embeddings if requested
    if args.emb_path and args.model_type not in {'FNN', 'CNN', 'HHMM'}:
        from w2vEmbReader import W2VEmbReader as EmbReader
        logger.info('Initializing lookup table')
        emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
        # embedding_matrix = emb_reader.get_emb_matrix_given_vocab(vocab)
        # model.layers[model.emb_index].W.set_value(emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[model.emb_index].W.get_value()))
        # model.get_layer(name='category_emb').set_weights(model.get_layer(name='rule_emb').get_weights())
        model.get_layer(name='word_emb').set_weights(emb_reader.get_emb_matrix_given_vocab(
            vocab, model.get_layer(name='word_emb').get_weights()))
        # w = model.get_layer(name='rule_emb').get_weights()
        # print(type(w))
        # z=tf.convert_to_tensor(w)
        # print("printing weight shape of rule")
        # print(z.shape)
        # print(z[0][5])
        logger.info('  Done')
    return model


def expand_dim(x):
    return K.expand_dims(x, 1)


def matmul(conv_output, swem_output, gate_output):
    K.dot(K.stack([conv_output, swem_output], axis=1), gate_output)
