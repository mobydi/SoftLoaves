from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
import keras
from keras.models import load_model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU, Bidirectional, Dropout, SpatialDropout1D, \
    Embedding
from keras.models import Model


class NNModel():
    def __init__(self, model_path, tokenizer, input_length, y_transformer=None,
                 fit_epochs=0, max_features=20000,
                 regr_path="./pretrain/small_regr.h5"):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.y_transformer = y_transformer
        self.fit_epochs = fit_epochs
        self.max_features = max_features
        self.regr_path = regr_path

    def build_classification_model(self):
        model = Sequential()
        model.add(
            Embedding(self.max_features, 50, input_length=self.input_length))
        model.add(SpatialDropout1D(0.3))
        model.add(Bidirectional(
            GRU(64, return_sequences=True, recurrent_dropout=0.3, dropout=0.3)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(
            GRU(64, return_sequences=False, recurrent_dropout=0.3,
                dropout=0.3)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        # model.add(Dense(1))
        # model.compile(loss='mse',
        #     optimizer='adam')

        model.add(Dense(5, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        self.model = model

    def build_regression_model(self):
        model_regression = Sequential()
        model_regression.add(Dense(32, activation='relu', input_shape=(5,)))
        model_regression.add(Dense(1))

        model_regression.compile(loss='mse',
                                 optimizer='adam')

        self.model_regr = model_regression

    def pretrain_clf_model(self, x_train, y_train_onehot, x_test,
                           y_test_onehot):

        self.build_classification_model()
        self.model.fit(x_train, y_train_onehot,
                       batch_size=128,
                       epochs=self.fit_epochs,
                       validation_data=(x_test, y_test_onehot),
                       callbacks=[
                           keras.callbacks.ModelCheckpoint(self.model_path,
                                                           save_best_only=True)])

    def preprocess(self, x_data, y_data=None):
        x_data = pad_sequences(self.tokenizer.texts_to_sequences(x_data),
                               maxlen=self.input_length, truncating='post',
                               padding='post')
        if y_data is not None:
            y_data = np.array(y_data).reshape(-1, 1)
            y_data = self.y_transformer.transform(y_data)
        return x_data, y_data

    def fit(self, x_train, y_train):
        self.model = load_model(self.model_path)
        self.build_regression_model()

        if self.fit_epochs > 0:
            x_train, y_train_clf = self.preprocess(x_train, y_train)
            self.model.fit(x_train, y_train_clf, batch_size=128,
                           epochs=self.fit_epochs)
            # self.model = Model(input = self.model.get_input_at(0), output = self.model.layers[-2].get_output_at(0))
            x_train_clf = self.model.predict(x_train, verbose=1)
            self.model_regr.fit(x_train_clf, y_train, batch_size=128, epochs=10,
                                validation_split=0.2)

        else:
            x_train, y_train_clf = self.preprocess(x_train, y_train)
            x_train_clf = self.model.predict(x_train, verbose=1)
            self.model_regr.fit(x_train_clf, y_train, batch_size=128, epochs=10,
                                validation_split=0.2)
            self.save_models()

    def predict(self, x_data, verbose=0):
        x_data, _ = self.preprocess(x_data)
        x_clf = self.model.predict(x_data, verbose=verbose)
        prediction = self.model_regr.predict(x_clf, verbose=verbose)
        return prediction

    def save_models(self):
        self.model.save(self.model_path)
        self.model_regr.save(self.regr_path)
        return self

    def define_for_flow(self):
        self.model = load_model(self.model_path)
        self.model_regr = load_model(self.regr_path)
        return self

