import pathlib

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import sklearn.model_selection as sk
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.utils import plot_model
from sklearn.metrics import precision_recall_fscore_support, classification_report

import helper
import load_folder
import network


def my_plots(results, n_folds):
    plt.title('Accuracy / Epochs')
    for i in range(n_folds):
        plt.plot(results[i].history['acc'], label="Training Fold " + str(i + 1))
    plt.legend()
    plt.draw()
    plt.savefig("Accuracies_vs_Epochs.png", bbox_inches=None, pad_inches=0)
    plt.close('all')

    plt.title('Train Accuracy vs Val Accuracy')
    plt.plot(results[0].history['acc'], label='Train Accuracy Fold 1', color='blue')
    plt.plot(results[0].history['val_acc'], label='Val Accuracy Fold 1', color='blue', linestyle="dashdot")
    plt.plot(results[1].history['acc'], label='Train Accuracy Fold 2', color='red', )
    plt.plot(results[1].history['val_acc'], label='Val Accuracy Fold 2', color='red', linestyle="dashdot")
    plt.legend()
    plt.draw()
    plt.savefig("Train Accuracy vs Val Accuracy.png", bbox_inches=None, pad_inches=0)
    plt.close('all')


def fit(x_train, x_test, y_train, y_test, early_stopping, model_checkpoint, EPOCHS, BATCH_SIZE):
    model = network.build_model()
    results = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1, validation_split=0.1)
    return results


def evaluate(model, x_test, y_test):
    y_prob = model.predict_proba(x_test, verbose=0)
    y_pred = y_prob.argmax(axis=-1)
    y_true = y_test

    conf = sklearn.metrics.confusion_matrix(y_test, y_pred)
    print("conf:", conf)

    label_names = sorted(item.name for item in pathlib.Path("data").glob('*/') if item.is_dir())
    helper.plot_confusion_matrix(y_test, y_pred, classes=label_names)

    score, accuracy = model.evaluate(x_test, y_test, batch_size=32)
    print("\nAccuracy = {:.2f}".format(accuracy))

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='micro')
    print("Precision: ", p)
    print("Recall: ", r)
    print("F-Score: ", round(f, 2))

    plot_model(model, to_file='model.png')


def main():
    features, labels = load_folder.load_all_folds("data/processed_data/")

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # define the model checkpoint callback -> this will keep on saving the model as a physical file
    model_checkpoint = ModelCheckpoint('ckpt.h5', verbose=1, save_best_only=True)

    n_folds = 10
    epochs = 4
    batch_size = 20

    results = []
    x_test = None
    y_test = None

    for i in range(n_folds):
        print("Training on Fold: ", i + 1)
        x_train, x_test, y_train, y_test = sk.train_test_split(features, labels, test_size=0.1,
                                                               random_state=np.random.randint(1, 1000, 1)[0])
        results.append(
            fit(x_train, x_test, y_train, y_test, early_stopping, model_checkpoint, epochs, batch_size))
        print("=======" * 12, end="\n\n\n")

    my_plots(results, n_folds)

    model = load_model('ckpt.h5')
    model.evaluate(x_test, y_test)
    y_prob = model.predict(x_test)
    y_pred = y_prob.argmax(axis=-1)

    print(classification_report(y_test, y_pred))
    label_names = sorted(item.name for item in pathlib.Path("data").glob('*/') if item.is_dir())
    helper.plot_confusion_matrix(y_test, y_pred, classes=label_names)


if __name__ == "__main__":
    main()
