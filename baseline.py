from keras.layers import containers
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, AutoEncoder, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

random.seed(1)
np.random.seed(1)

nb_epoch = 50
batch_size = 100
nb_labels = 10

train_subset_y_cat = np_utils.to_categorical(train_subset_y, nb_labels)
dev_y_cat = np_utils.to_categorical(dev_y, nb_labels)
test_y_cat = np_utils.to_categorical(test_y, nb_labels)

model = Sequential()
model.add(Dense(1000, input_dim=train_x.shape[1], activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_labels, activation='softmax'))




model.compile(loss='categorical_crossentropy', optimizer='Adam')
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0)

print('Start training')
model.fit(train_subset_x, train_subset_y_cat, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=True, validation_data=(dev_x, dev_y_cat), callbacks=[earlyStopping])

score = model.evaluate(test_x, test_y_cat, show_accuracy=True, verbose=False)
print('Test accuracy:', score[1])