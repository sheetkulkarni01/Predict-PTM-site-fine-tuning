import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Layer, GlobalAveragePooling1D, Input


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.01):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=rate)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(2 * ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs, training):
        attn_output, attention_scores = self.att(inputs, inputs, inputs, return_attention_scores=True, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), attention_scores


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim, mask_zero=True)
        
    def get_config(self):
        config = super().get_config()
        return config

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def transformer_model():
    vocab_size = 21
    embed_dim = 128 #config["embedding_dim"]
    ff_dim = 128 #config["feed_forward_dim"]
    max_len = 33 #config["maximum_path_length"]
    dropout = 0.01 #config["dropout"]
    n_heads = 2

    inputs = Input(shape=(max_len,))
    embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, n_heads, ff_dim)
    x, weights = transformer_block(x)
    x = GlobalAveragePooling1D()(x) 
    x = Dropout(dropout)(x)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=[outputs])

def train_transformer(pre_train_y_cnn, pre_valid_y_cnn, pre_train_y_cnn_labels, pre_valid_y_cnn_labels):
    all_train_losses = []
    all_train_accuracies = []
    all_val_losses = []
    all_val_accuracies = []
    saved_results_list2 = []
    n_epo_transformer = 20
    max_len2 = 0
    metrics = ['accuracy', tf.keras.metrics.AUC(name='auc_roc')]

    print("PRE Training Transformer model...")
    for i in range(5):
        print(f"\nTraining Loop {i + 1}")

        model_transformer = transformer_model()
        model_transformer.compile(optimizer=Adam(learning_rate=1e-5), loss=BinaryCrossentropy(), metrics=metrics)

        checkpointer = ModelCheckpoint(filepath="./pre_model_transformer.h5",
                                    monitor = "val_accuracy",
                                    verbose = 0,
                                    save_weights_only=True,
                                    save_best_only=True)
        
        pre_history_transformer = model_transformer.fit(pre_train_y_cnn, pre_train_y_cnn_labels, batch_size=32, epochs=n_epo_transformer, verbose=1, callbacks=[checkpointer],
            validation_data=(pre_valid_y_cnn, pre_valid_y_cnn_labels))

        all_train_losses.append(pre_history_transformer.history['loss'])
        all_train_accuracies.append(pre_history_transformer.history['accuracy'])
        all_val_losses.append(pre_history_transformer.history['val_loss'])
        all_val_accuracies.append(pre_history_transformer.history['val_accuracy'])

        saved_results2 = {
        'train_losses': all_train_losses.copy(),
        'train_accuracies': all_train_accuracies.copy(),
        'val_losses': all_val_losses.copy(),
        'val_accuracies': all_val_accuracies.copy()
        }
        saved_results_list2.append(saved_results2)
        np.save(f'saved_results_Transformerloop_{i + 1}.py', saved_results2)

        max_len2 = max(max_len2, len(all_train_losses[-1]))

    for result in saved_results_list2:
        for key in result:
            result[key] = [np.pad(lst, (0, max_len2 - len(lst))) for lst in result[key]]

    # Trim or zero-pad the lists to the maximum length
    all_train_losses = [lst[:max_len2] for lst in all_train_losses]
    all_train_accuracies = [lst[:max_len2] for lst in all_train_accuracies]
    all_val_losses = [lst[:max_len2] for lst in all_val_losses]
    all_val_accuracies = [lst[:max_len2] for lst in all_val_accuracies]

    return pre_history_transformer, all_train_accuracies, all_val_accuracies

