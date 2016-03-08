nb_epoch = 50
batch_size = 100

model = Sequential()
for encoder in encoders:
    model.add(encoder)


model.add(Dense(output_dim=nb_labels, activation='softmax'))

train_subset_y_cat = np_utils.to_categorical(train_subset_y, nb_labels)
test_y_cat = np_utils.to_categorical(test_y, nb_labels)




model.compile(loss='categorical_crossentropy', optimizer='Adam')
score = model.evaluate(test_x, test_y_cat, show_accuracy=True, verbose=0)
print('Test score before fine turning:', score[0])
print('Test accuracy before fine turning:', score[1])
model.fit(train_subset_x, train_subset_y_cat, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, validation_data=(dev_x, dev_y_cat), shuffle=True)
score = model.evaluate(test_x, test_y_cat, show_accuracy=True, verbose=0)
print('Test score after fine turning:', score[0])
print('Test accuracy after fine turning:', score[1])