# Train the autoencoder
# Source: https://github.com/fchollet/keras/issues/358
from keras.layers import containers
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, AutoEncoder, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

random.seed(3)
np.random.seed(3)



nb_epoch_pretraining = 10
batch_size_pretraining = 500


# Layer-wise pretraining
encoders = []
decoders = []
nb_hidden_layers = [train_x.shape[1], 500, 2]
X_train_tmp = np.copy(train_x)

dense_layers = []

for i, (n_in, n_out) in enumerate(zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]), start=1):
    print('Training the layer {}: Input {} -> Output {}'.format(i, n_in, n_out))
    # Create AE and training
    ae = Sequential()
    if n_out >= 100:
        encoder = containers.Sequential([Dense(output_dim=n_out, input_dim=n_in, activation='tanh'), Dropout(0.5)])
    else:
        encoder = containers.Sequential([Dense(output_dim=n_out, input_dim=n_in, activation='tanh')])
    decoder = containers.Sequential([Dense(output_dim=n_in, input_dim=n_out, activation='tanh')])
    ae.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))

    sgd = SGD(lr=2, decay=1e-6, momentum=0.0, nesterov=True)
    ae.compile(loss='mse', optimizer='adam')
    ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size_pretraining, nb_epoch=nb_epoch_pretraining, verbose = True, shuffle=True)
    # Store trainined weight and update training data
    encoders.append(ae.layers[0].encoder)
    decoders.append(ae.layers[0].decoder)

    X_train_tmp = ae.predict(X_train_tmp)





##############





#End to End Autoencoder training
if len(nb_hidden_layers) > 2:
    full_encoder = containers.Sequential()
    for encoder in encoders:
        full_encoder.add(encoder)

    full_decoder = containers.Sequential()
    for decoder in reversed(decoders):
        full_decoder.add(decoder)

    full_ae = Sequential()
    full_ae.add(AutoEncoder(encoder=full_encoder, decoder=full_decoder, output_reconstruction=False))
    full_ae.compile(loss='mse', optimizer='adam')

    print "Pretraining of full AE"
    full_ae.fit(train_x, train_x, batch_size=batch_size_pretraining, nb_epoch=nb_epoch_pretraining, verbose = True, shuffle=True)

