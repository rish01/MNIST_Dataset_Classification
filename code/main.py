import gzip
import os
import pickle
import time

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D
# from keras.utils import np_utils

from knn import KNN
from linear_model import LeastSquaresL2, LinearClassifierRobust, leastSquaresClassifier
from kernelRegression import kernelLinearClassifier, kernel_poly, kernel_RBF
from svm_multiclass import SVM_Multiclass_Sum_Loss, SVM_Multiclass_Max_Loss
from neural_net import NeuralNet
from cnn import BasicCNN


def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-q','--question', required=True)
    #
    # io_args = parser.parse_args()
    # question = io_args.question\

    question = "1"

    # ######################################## IMPORTANT INPUT ####################################################### #
    model = "CNN"   # Input the model name to run here. Choose from KNN, LINEAR REGRESSION, SVM, MLP and CNN

    if question == "1":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        # Eliminate zero columns from X matrix and remove those same columns from Xtest matrix
        non_zero_cols = ~np.all(X == 0, axis=0)
        X = X[:, non_zero_cols]
        Xtest = Xtest[:, non_zero_cols]

        # Create k-folds for cross-validation
        kf = KFold(n_splits=5)

        ###############################################################################################################
        # --------------------------------------------- 1.1 KNN ---------------------------------------------------- #
        ###############################################################################################################
        if model == "KNN":
            random_rows = np.random.choice(X.shape[0], size=20000, replace=False)
            X_subset = X[random_rows]
            y_subset = y[random_rows]

            k_values = np.arange(1, 10)
            y_pred = np.zeros(k_values.size)
            min_validation_err = 1
            best_k = 3

            if best_k is None:
                for k in k_values:
                    val_errors_per_fold = []
                    for train, validate in kf.split(X_subset, y_subset):
                        X_train = X_subset[train]
                        y_train = y_subset[train]
                        X_validate = X_subset[validate]
                        y_validate = y_subset[validate]

                        knn_model = KNN(k)
                        knn_model.fit(X_train, y_train)

                        # Validation Error for this fold
                        y_pred = knn_model.predict(X_validate)
                        val_errors_per_fold.append(np.mean(y_pred != y_validate))

                    avg_validation_error = np.average(np.asarray(val_errors_per_fold))
                    print("Validation test error: {:.4f} and k = {}".format(avg_validation_error, k))

                    if avg_validation_error < min_validation_err:
                        min_validation_err = avg_validation_error
                        best_k = k

            print(f"Best k-value: {best_k}")

            if best_k is not None:
                knn_model = KNN(k)
                knn_model.fit(X, y)

                # Training Error
                y_pred = knn_model.predict(X)
                tr_error = np.mean(y_pred != y)
                print("KNN training error: {:.3f}".format(tr_error))

                # Test Error
                y_pred = knn_model.predict(Xtest)
                test_error = np.mean(y_pred != ytest)
                print("KNN test error: {:.3f} and k = {}".format(test_error, k))

        ###############################################################################################################
        # --------------------------------------- 1.2 LINEAR REGRESSION --------------------------------------------- #
        ###############################################################################################################
        elif model == "LINEAR_REGRESSION":
            # -------------------- 1.2.1 LEAST SQUARES LINEAR REGRESSION WITH L2 REGULARIZATION --------------------- #
            least_squares_L2_model = LeastSquaresL2()
            least_squares_L2_model.fit(X, y, lammy=2)
            y_pred = np.round(least_squares_L2_model.predict(X))
            tr_error = np.mean(y_pred != y)
            print(f"Least Squares L2 Training Error: {tr_error}")

            y_pred = np.round(least_squares_L2_model.predict(Xtest))
            test_error = np.mean(y_pred != ytest)
            print(f"Least Squares L2 Test Error: {test_error}")

            # ------------------------------- 1.2.2 ROBUST LINEAR CLASSIFIER ---------------------------------------- #
            robust_linear_classifier_model = LinearClassifierRobust(lammy=0)
            robust_linear_classifier_model.fit(X, y)
            y_pred = robust_linear_classifier_model.predict(X)
            tr_error = np.mean(y_pred != y)
            print("Robust Linear Classifier training error = {}".format(tr_error))

            y_pred = robust_linear_classifier_model.predict(Xtest)
            test_error = np.mean(y_pred != ytest)
            print("Robust Linear Classifier test error = {}".format(test_error))

            # -------------------------------- 1.2.3 LEAST SQUARES CLASSIFIER --------------------------------------- #
            lammy_m = np.arange(-4, 1, dtype=float)
            best_lammy_mm = 0
            min_validation_err = 1

            if best_lammy_mm is None:
                for lammy_mm in lammy_m:
                    val_errors_per_fold = []
                    for train, validate in kf.split(X, y):
                        X_train = X[train]
                        y_train = y[train]
                        X_validate = X[validate]
                        y_validate = y[validate]

                        least_squares_classifier_model = leastSquaresClassifier(lammy=10**lammy_mm)
                        least_squares_classifier_model.fit(X_train, y_train)
                        y_pred = least_squares_classifier_model.predict(X_validate)
                        val_errors_per_fold.append(np.mean(y_pred != y_validate))

                    avg_validation_error = np.average(np.asarray(val_errors_per_fold))
                    print(f"Least Squares Classifier Validation Error: {avg_validation_error} with lammy: {10**lammy_mm}")

                    if avg_validation_error < min_validation_err:
                        min_validation_err = avg_validation_error
                        best_lammy_mm = lammy_mm

            if best_lammy_mm is not None:
                print(f"Least Squares Classifier best lammy value: {10 ** best_lammy_mm}")
                least_squares_classifier_model = leastSquaresClassifier(lammy=10 ** best_lammy_mm)
                least_squares_classifier_model.fit(X, y)
                y_pred = least_squares_classifier_model.predict(X)
                tr_error = np.mean(y_pred != y)
                print(f"Least Squares Classifier Training Error: {tr_error} with lammy: {10 ** best_lammy_mm}")

                y_pred = least_squares_classifier_model.predict(Xtest)
                test_error = np.mean(y_pred != ytest)
                print(f"Least Squares Classifier Test Error: {test_error} with lammy: {10 ** best_lammy_mm}")
            
            # ---------------------------- 1.2.4 KERNEL_REGRESSION - POLYNOMIAL ------------------------------------- #
            lammy_m = np.arange(-4, 1, dtype=float)
            p = np.arange(1, 7)
            random_rows = np.random.choice(X.shape[0], size=2500, replace=False)
            X_subset = X[random_rows]
            y_subset = y[random_rows]

            best_pp = 3
            best_lammy_mm = 0
            min_validation_err = 1

            if best_pp is None or best_lammy_mm is None:
                for pp in p:
                    for lammy_mm in lammy_m:
                        val_errors_per_fold = []
                        for train, validate in kf.split(X_subset, y_subset):
                            X_train = X_subset[train]
                            y_train = y_subset[train]
                            X_validate = X_subset[validate]
                            y_validate = y_subset[validate]

                            poly_kernel = kernelLinearClassifier(kernel_fun=kernel_poly, lammy=lammy_mm, p=pp)
                            poly_kernel.fit(X_train, y_train)
                            val_errors_per_fold.append(np.mean(poly_kernel.predict(X_validate) != y_validate))

                        avg_validation_error = np.average(np.asarray(val_errors_per_fold))
                        print(
                            f"Polynomial Kernel Validation Error: {avg_validation_error} with lammy: {10**lammy_mm} "
                            f"and polynomial degree: {pp}")

                        if avg_validation_error < min_validation_err:
                            min_validation_err = avg_validation_error
                            best_pp = pp
                            best_lammy_mm = lammy_mm

            random_rows = np.random.choice(X.shape[0], size=20000, replace=False)
            if best_pp is not None and best_lammy_mm is not None:
                print(f"Poly Kernel best degree: {best_pp} and best lammy: {10**best_lammy_mm}")
                poly_kernel = kernelLinearClassifier(kernel_fun=kernel_poly, lammy=10**best_lammy_mm, p=best_pp)
                poly_kernel.fit(X[random_rows], y[random_rows])

                test_error = np.mean(poly_kernel.predict(Xtest) != ytest)
                print(f"Test Error Poly Kernel: {test_error} for p = {best_pp} and lammy = {10**best_lammy_mm}")

            # ------------------------------ 1.2.4 KERNEL_REGRESSION - RBF ------------------------------------------ #
            lammy_m = np.arange(-4, 1, dtype=float)
            sigma_m = np.arange(-2, 3, dtype=float)
            random_rows = np.random.choice(X.shape[0], size=2500, replace=False)
            X_subset = X[random_rows]
            y_subset = y[random_rows]

            best_sigma_mm = 1
            best_lammy_mm = -2
            min_validation_err = 1

            if best_sigma_mm is None and best_lammy_mm is None:
                for sigma_mm in sigma_m:
                    for lammy_mm in lammy_m:
                        val_errors_per_fold = []
                        for train, validate in kf.split(X_subset, y_subset):
                            X_train = X_subset[train]
                            y_train = y_subset[train]
                            X_validate = X_subset[validate]
                            y_validate = y_subset[validate]

                            RBF_kernel = kernelLinearClassifier(kernel_fun=kernel_RBF, lammy=10**lammy_mm, sigma=10**sigma_mm)
                            RBF_kernel.fit(X_train, y_train)
                            val_errors_per_fold.append(np.mean(RBF_kernel.predict(X_validate) != y_validate))

                        avg_validation_error = np.average(np.asarray(val_errors_per_fold))
                        print(
                            f"RBF Kernel Validation Error: {avg_validation_error} with sigma: {10 ** sigma_mm} "
                            f"and lammy: {10**lammy_mm}")

                        if avg_validation_error < min_validation_err:
                            min_validation_err = avg_validation_error
                            best_sigma_mm = sigma_mm
                            best_lammy_mm = lammy_mm

            random_rows = np.random.choice(X.shape[0], size=5000, replace=False)
            if best_sigma_mm is not None and best_lammy_mm is not None:
                print(f"RBF Kernel best sigma: {10**best_sigma_mm} and best lammy: {10 ** best_lammy_mm}")
                RBF_kernel = kernelLinearClassifier(kernel_fun=kernel_RBF, lammy=10**best_lammy_mm,
                                                    sigma=10**best_sigma_mm)
                RBF_kernel.fit(X[random_rows], y[random_rows])
                tr_error = np.mean(RBF_kernel.predict(X) != y)
                print("Training Error RBF Kernel {} for sigma = {} and lammy = {}".format(tr_error,
                                                                                          10 ** best_sigma_mm,
                                                                                          10 ** best_lammy_mm))
                test_error = np.mean(RBF_kernel.predict(Xtest) != ytest)
                print("Test Error RBF Kernel {} for sigma = {} and lammy = {}".format(test_error, 10 ** best_sigma_mm,
                                                                                      10 ** best_lammy_mm))

        ###############################################################################################################
        # ----------------------------------------------- 1.3 SVM --------------------------------------------------- #
        ###############################################################################################################
        elif model == "SVM":
            lammy_m = np.arange(-4, 1, dtype=float)     # Trialing different values of lammy_m
            # lammy_m = [-2]

            # -------------------- 1.3.1 MULTI-CLASS SVM - SUM LOSS WITH L2 REGULARIZATION -------------------------- #
            best_lammy_mm = -2
            min_validation_err = 1

            if best_lammy_mm is None:
                for lammy_mm in lammy_m:
                    val_errors_per_fold = []
                    for train, validate in kf.split(X, y):
                        X_train = X[train]
                        y_train = y[train]
                        X_validate = X[validate]
                        y_validate = y[validate]

                        multi_class_SVM_Sum_Loss_model = SVM_Multiclass_Sum_Loss(lammy=10 ** lammy_mm, epoch=0.1, verbose=0)
                        multi_class_SVM_Sum_Loss_model.fit(X_train, y_train)
                        val_errors_per_fold.append(np.mean(multi_class_SVM_Sum_Loss_model.predict(X_validate) != y_validate))

                    avg_validation_error = np.average(np.asarray(val_errors_per_fold))
                    print(f"SVM SUM Loss Validation Error: {avg_validation_error} with lammy: {10 ** lammy_mm}")

                    if avg_validation_error < min_validation_err:
                        min_validation_err = avg_validation_error
                        best_lammy_mm = lammy_mm

            if best_lammy_mm is not None:
                print(f"SVM SUM Loss best lammy: {10**best_lammy_mm}")
                multi_class_SVM_Sum_Loss_model = SVM_Multiclass_Sum_Loss(lammy=10 ** best_lammy_mm, epoch=5, verbose=0)
                multi_class_SVM_Sum_Loss_model.fit(X, y)
                tr_error = np.mean(multi_class_SVM_Sum_Loss_model.predict(X) != y)
                test_error = np.mean(multi_class_SVM_Sum_Loss_model.predict(Xtest) != ytest)
                print("Training Error SVM - SUM Loss: {} with lammy: {}".format(tr_error, 10 ** best_lammy_mm))
                print("Test Error SVM - SUM Loss: {} with lammy: {}".format(test_error, 10 ** best_lammy_mm))

            # --------------------- 1.3.2 MULTI-CLASS SVM - MAX LOSS WITH L2 REGULARIZATION ------------------------- #
            best_lammy_mm = -1
            min_validation_err = 1

            if best_lammy_mm is None:
                for lammy_mm in lammy_m:
                    val_errors_per_fold = []
                    for train, validate in kf.split(X, y):
                        X_train = X[train]
                        y_train = y[train]
                        X_validate = X[validate]
                        y_validate = y[validate]

                        multi_class_SVM_Max_Loss_model = SVM_Multiclass_Max_Loss(lammy=10 ** lammy_mm, epoch=0.1,
                                                                                 verbose=0)
                        multi_class_SVM_Max_Loss_model.fit(X_train, y_train)
                        val_errors_per_fold.append(
                            np.mean(multi_class_SVM_Max_Loss_model.predict(X_validate) != y_validate))

                    avg_validation_error = np.average(np.asarray(val_errors_per_fold))
                    print(f"SVM SUM Loss Validation Error: {avg_validation_error} with lammy: {10 ** lammy_mm}")

                    if avg_validation_error < min_validation_err:
                        min_validation_err = avg_validation_error
                        best_lammy_mm = lammy_mm

            if best_lammy_mm is not None:
                print(f"SVM MAX Loss best lammy: {10 ** best_lammy_mm}")
                multi_class_SVM_Max_Loss_model = SVM_Multiclass_Max_Loss(lammy=10 ** best_lammy_mm, epoch=5, verbose=0)
                multi_class_SVM_Max_Loss_model.fit(X, y)
                tr_error = np.mean(multi_class_SVM_Max_Loss_model.predict(X) != y)
                test_error = np.mean(multi_class_SVM_Max_Loss_model.predict(Xtest) != ytest)
                print("Training Error SVM - MAX Loss: {} with lammy: {}".format(tr_error, 10 ** best_lammy_mm))
                print("Test Error SVM - MAX Loss: {} with lammy: {}".format(test_error, 10 ** best_lammy_mm))


        ###############################################################################################################
        # ----------------------------------------------- 1.4 MLP --------------------------------------------------- #
        ###############################################################################################################
        elif model == "MLP":
            hidden_layer_sizes = [50]
            mlp_model = NeuralNet(hidden_layer_sizes, learning_rate_decay=False, max_iter=500)

            t = time.time()
            # mlp_model.fit(X, Y)
            mlp_model.fitWithSGD(X, Y, epoch=40, minibatch_size=2500)
            print("Fitting took %d seconds" % (time.time() - t))

            # Compute training error
            yhat = mlp_model.predict(X)
            tr_error = np.mean(yhat != y)
            print("Training error = ", tr_error)

            # Compute test error
            yhat = mlp_model.predict(Xtest)
            test_error = np.mean(yhat != ytest)
            print("Test error     = ", test_error)


        ###############################################################################################################
        # ----------------------------------------------- 1.5 CNN --------------------------------------------------- #
        ###############################################################################################################

        # CITATION: The code for CNN is adapted from Mahan Fathi's Stanford's CS231 Assignment 2 repo
        # https://github.com/MahanFathi/CS231/blob/master/assignment2/cs231n/

        elif model == "CNN":
            # Determine whether the model needs to be trained and/or used for prediction
            overfit_testing = True
            training = False
            predicting = False

            cnn_model_weights_file = os.path.join(os.path.abspath(__file__), '..', 'data', 'cnn_weights.npy')

            # Reinitializing X, y, Xtest and ytest so that the original dimensions of images are preserved
            X, y = train_set
            Xtest, ytest = test_set
            random_rows = np.random.choice(X.shape[0], size=40, replace=False)
            X_overfit = X[random_rows]
            y_overfit = y[random_rows]
            Y_overfit = Y[random_rows]

            cnn_model = BasicCNN(
                input_dim=(1, 28, 28),
                lammy=0,
                epoch=0.2,
                minibatch_size=40,
                verbose=1,
                learning_rate_decay=False
            )

            # Overfit on a small sub-sample to ensure that the model and SGD is working as expected
            if overfit_testing:
                print("Beginning overfit testing.")
                cnn_model.epoch = 100
                cnn_model.fit(X_overfit, Y_overfit)
                y_pred = cnn_model.predict(X_overfit)
                overfit_training_error = np.mean(y_pred != Y_overfit)
                print(f"CNN Overfit training error: {overfit_training_error}")

            if training:
                cnn_model.fit(X, Y)
                np.save(cnn_model_weights_file, cnn_model.weights)

            # Compute training and test errors by predicting a limited number of rows per iteration to avoid value error
            if predicting:
                cnn_model.weights = np.load(cnn_model_weights_file, allow_pickle=True)
                i_train = 0
                i_test = 0
                num_folds = 500
                yhat_train = np.zeros(y.size)
                yhat_test = np.zeros(ytest.size)
                num_rows_train = int(X.shape[0] / num_folds)
                num_rows_test = int(Xtest.shape[0] / num_folds)

                for fold in range(num_folds):
                    yhat_train[i_train: i_train + num_rows_train] = cnn_model.predict(X[i_train: i_train + num_rows_train])
                    yhat_test[i_test: i_test+num_rows_train] = cnn_model.predict(Xtest[i_test: i_test+num_rows_train])
                    i_train += num_rows_train
                    i_test += num_rows_test

                tr_error = np.mean(yhat_train != y)
                print("Training error = ", tr_error)

                test_error = np.mean(yhat_test != ytest)
                print("Test error     = ", test_error)

        else:
            print("Unknown model: %s" % model)

    else:
        print("Unknown question: %s" % question)