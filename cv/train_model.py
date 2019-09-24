from skimage import io, feature, filters, exposure, color
from sklearn.model_selection import train_test_split
import numpy as np
import re
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import keras

model_path = 'models/name_model3.h5'
#people = ["R", "V", "S", "A"]
people = {"R":0, "V":1, "S":2, "A":3}

def imread_convert(f):
    return io.imread(f).astype(np.uint8)

def load_data_from_folder(dir):
    ic = io.ImageCollection(dir+"*.jpg", load_func=imread_convert)
    data = io.concatenate_images(ic)
    labels = np.array(ic.files)
    for i, f in enumerate(labels):
        print(f)
        m = f.find("_")
        lbl = f[len(dir):m]
        # m = re.search("_", f)
        #lbl = f[len(dir):m.start()]
        # if lbl not in people:
        #     lbl = "Rando"
        # labels[i] = lbl
        labels[i] = people.get(lbl, 4)
    return(data,labels)

(X, Y) = load_data_from_folder('./images/')
X = np.reshape(X, (1350, 48*48))
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)
one_hot_labels_train = to_categorical(y_train, num_classes=5)
one_hot_labels_test = to_categorical(y_test, num_classes=5)

model = Sequential()
model.add(Dense(32, input_dim = (48*48)))
model.add(Activation('sigmoid'))
model.add(Dense(5))
model.add(Activation('softmax'))

adam = keras.optimizers.Adam(lr=0.2, decay=0.0,)
model.compile(optimizer=adam,loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, one_hot_labels_train, epochs=10, batch_size=32)

model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, one_hot_labels_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
