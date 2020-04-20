import gzip
import os
import pickle
import time

import numpy as np
from sklearn.preprocessing import LabelBinarizer

from knn import KNN
from linear_model import LeastSquaresL2, LinearClassifierRobust, leastSquaresClassifier
from kernelRegression import kernelLinearClassifier, kernel_poly, kernel_RBF
from svm_multiclass import SVM_Multiclass_Sum_Loss, SVM_Multiclass_Max_Loss
from neural_net import NeuralNet


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
    model = "MLP"

    if question == "1":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

        X, y = train_set
        Xtest, ytest = test_set

        # Convert row vectors to column vectors
        # y = y[:, None]
        # ytest = ytest[:, None]

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        # Eliminate zero columns from X matrix and remove those same columns from Xtest matrix
        non_zero_cols = ~np.all(X == 0, axis=0)
        X = X[:, non_zero_cols]
        Xtest = Xtest[:, non_zero_cols]

        # --------------------------------------------- 1.1 KNN ---------------------------------------------------- #
        if model == "KNN":
            k_values = np.asarray([1, 3, 5, 7])
            y_pred = np.zeros(k_values.size)

            for k in k_values:
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

        # --------------------------------------- 1.2 LINEAR REGRESSION -------------------------------------------- #
        elif model == "LINEAR_REGRESSION":
            # ------------------------------- 1.2.1 ROBUST LINEAR CLASSIFIER
            robust_linear_classifier_model = LinearClassifierRobust(lammy=0)
            robust_linear_classifier_model.fit(X, y)
            y_pred = robust_linear_classifier_model.predict(X)
            tr_error = np.mean(y_pred != y)
            print("Robust Linear Classifier training error = {}".format(tr_error))

            y_pred = robust_linear_classifier_model.predict(Xtest)
            test_error = np.mean(y_pred != ytest)
            print("Robust Linear Classifier test error = {}".format(test_error))

            # -------------------------------- 1.2.2 LEAST SQUARES CLASSIFIER
            lammy_m = np.arange(0, 1, dtype=float)

            for lammy_mm in lammy_m:
                least_squares_classifier_model = leastSquaresClassifier(lammy=10**lammy_mm)
                least_squares_classifier_model.fit(X, y)
                y_pred = least_squares_classifier_model.predict(X)
                tr_error = np.mean(y_pred != y)
                print(f"Least Squares Classifier Training Error: {tr_error} with lammy: {10**lammy_mm}")

                y_pred = least_squares_classifier_model.predict(Xtest)
                test_error = np.mean(y_pred != ytest)
                print(f"Least Squares Classifier Test Error: {test_error} with lammy: {10**lammy_mm}")

            # -------------------- 1.2.3 LEAST SQUARES LINEAR REGRESSION WITH L2 REGULARIZATION
            least_squares_L2_model = LeastSquaresL2()
            least_squares_L2_model.fit(X, y, lammy=2)
            y_pred = np.round(least_squares_L2_model.predict(X))
            tr_error = np.mean(y_pred != y)
            print(f"Least Squares L2 Training Error: {tr_error}")

            y_pred = np.round(least_squares_L2_model.predict(Xtest))
            test_error = np.mean(y_pred != ytest)
            print(f"Least Squares L2 Test Error: {test_error}")

            # ---------------------------- 1.2.4 KERNEL_REGRESSION - POLYNOMIAL
            sigma_m = np.arange(-2, 3, dtype=float)
            lammy_m = np.arange(-4, 1, dtype=float)
            p = np.arange(1, 7)
            random_rows = np.random.choice(X.shape[0], size=2500, replace=False)

            # for pp in p:
            #   for lammy in lammies:
            #     poly_kernel = kernelLinearClassifier(kernel_fun=kernel_poly, lammy=lammy, p=pp)
            #     poly_kernel.fit(X[:5000], y[:5000])
            #     test_error = np.mean(poly_kernel.predict(Xtest) != ytest)
            #     print("Validation Error Poly Kernel {} for p = {} and lammy = {}".format(test_error, pp, lammy))
            pp = 3
            lammy_mm = 0

            poly_kernel = kernelLinearClassifier(kernel_fun=kernel_poly, lammy=10**lammy_mm, p=pp)
            poly_kernel.fit(X[random_rows, :], y[random_rows])
            tr_error = np.mean(poly_kernel.predict(X) != y)
            print("Training Error Poly Kernel {} for p = {} and lammy = {}".format(tr_error, pp, 10 ** lammy_mm))

            test_error = np.mean(poly_kernel.predict(Xtest) != ytest)
            print("Test Error Poly Kernel {} for p = {} and lammy = {}".format(test_error, pp, 10**lammy_mm))

            # ------------------------------ 1.2.4 KERNEL_REGRESSION - RBF
            # for sigma_mm in sigma_m:
            #   for lammy_mm in lammy_m:
            #     RBF_kernel = kernelLinearClassifier(kernel_fun=kernel_RBF, lammy=10**lammy_mm, sigma=10**sigma_mm)
            #     RBF_kernel.fit(X[:4500], y[:4500])
            #     test_error = np.mean(RBF_kernel.predict(Xtest) != ytest)
            #     print("Test Error RBF Kernel {} for sigma = {} and lammy = {}".format(test_error, 10 ** sigma_mm,
            #                                                                           10 ** lammy_mm))

            lammy_mm = -2.0
            sigma_mm = 1.0
            RBF_kernel = kernelLinearClassifier(kernel_fun=kernel_RBF, lammy=10**lammy_mm, sigma=10**sigma_mm)
            RBF_kernel.fit(X[random_rows, :], y[random_rows])
            tr_error = np.mean(RBF_kernel.predict(X) != y)
            print("Training Error RBF Kernel {} for sigma = {} and lammy = {}".format(test_error, 10 ** sigma_mm,
                                                                                  10 ** lammy_mm))
            test_error = np.mean(RBF_kernel.predict(Xtest) != ytest)
            print("Test Error RBF Kernel {} for sigma = {} and lammy = {}".format(test_error, 10**sigma_mm, 10**lammy_mm))

        # ----------------------------------------------- 1.3 SVM --------------------------------------------------- #
        elif model == "SVM":
            # lammy_m = np.arange(-4, 1, dtype=float)     # Trialing different values of lammy_m
            lammy_m = [-2]

            # -------------------- 1.3.1 MULTI-CLASS SVM - SUM LOSS WITH L2 REGULARIZATION
            for lammy_mm in lammy_m:
                multi_class_SVM_Sum_Loss_model = SVM_Multiclass_Sum_Loss(lammy=10 ** lammy_mm, epoch=2, verbose=0)
                multi_class_SVM_Sum_Loss_model.fit(X, y)
                tr_error = np.mean(multi_class_SVM_Sum_Loss_model.predict(X) != y)
                test_error = np.mean(multi_class_SVM_Sum_Loss_model.predict(Xtest) != ytest)
                print("Training Error SVM - SUM Loss = {} with lammy = {}".format(tr_error, 10**lammy_mm))
                print("Test Error SVM - SUM Loss = {} with lammy = {}".format(test_error, 10**lammy_mm))

            # --------------------- 1.3.2 MULTI-CLASS SVM - MAX LOSS WITH L2 REGULARIZATION
            for lammy_mm in lammy_m:
                multi_class_SVM_Max_Loss_model = SVM_Multiclass_Max_Loss(lammy=10 ** lammy_mm, epoch=1, verbose=0)
                multi_class_SVM_Max_Loss_model.fit(X, y)
                tr_error = np.mean(multi_class_SVM_Max_Loss_model.predict(X) != y)
                test_error = np.mean(multi_class_SVM_Max_Loss_model.predict(Xtest) != ytest)
                print("Training Error SVM - MAX Loss = {} with lammy = {}".format(tr_error, 10**lammy_mm))
                print("Test Error SVM - MAX Loss = {} with lammy = {}".format(test_error, 10**lammy_mm))

        # ----------------------------------------------- 1.4 MLP --------------------------------------------------- #
        elif model == "MLP":
            hidden_layer_sizes = [50]
            mlp_model = NeuralNet(hidden_layer_sizes, learning_rate_MLP_SGD=True, max_iter=500)

            t = time.time()
            # mlp_model.fit(X, Y)
            mlp_model.fitWithSGD(X, Y, epoch=40, minibatch_size=2500)
            print("Fitting took %d seconds" % (time.time() - t))

            # Compute training error
            yhat = mlp_model.predict(X)
            trainError = np.mean(yhat != y)
            print("Training error = ", trainError)

            # Compute test error
            yhat = mlp_model.predict(Xtest)
            testError = np.mean(yhat != ytest)
            print("Test error     = ", testError)

    else:
        print("Unknown question: %s" % question)