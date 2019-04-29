import helper
import network
from keras.callbacks import EarlyStopping
import load_folder
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import sklearn.model_selection as sk
import sklearn.metrics
import pathlib
from keras.utils import plot_model

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


def main ():
    train = False
    features, labels = load_folder.load_all_folds("data/processed_data/")

    x_train, x_test, y_train, y_test = sk.train_test_split(features, labels, shuffle=True, train_size=0.8, random_state=42)

    print("Building model...")
    model = network.build_model()

    # a stopping function to stop training before we excessively overfit to the training set
    earlystop = EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')

    if train:
        print("Training model...")
        model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[earlystop], batch_size=20, epochs=3)
        model.save_weights("trained_model")
    else:
        model.load_weights("trained_model")

    print("Evaluating model...")
    evaluate(model, x_test, y_test)

if __name__ == "__main__":
    main()
