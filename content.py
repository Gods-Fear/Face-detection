import numpy as np


def sigmoid(x):
    """
    :param x: wektor wartości *x* do zaaplikowania funkcji sigmoidalnej Nx1
    :return: wektor wartości funkcji sigmoidalnej dla wartości *x* Nx1
    """
    return 1 / (1 + np.exp(-x))  # exp = e^-x


def logistic_cost_function(w, x_train, y_train):
    """
    :param w: wektor parametrów modelu Mx1
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :return: krotka (log, grad), gdzie *log* to wartość funkcji logistycznej,
        a *grad* jej gradient po parametrach *w* Mx1
    """
    sigma = sigmoid(x_train @ w)
    N = x_train.shape[0]
    out_arr = -np.sum(y_train * np.log(sigma) + (1 - y_train) * np.log(1 - sigma))
    grad = x_train.transpose() @ (sigma - y_train) / N

    return out_arr / N, grad


def gradient_descent(obj_fun, w0, epochs, eta):
    """
    :param obj_fun: optymalizowana funkcja celu, przyjmująca jako argument
        wektor parametrów *w* [wywołanie *val, grad = obj_fun(w)*]
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok algorytmu gradientu prostego
    :param eta: krok uczenia
    :return: krotka (w, log_values), gdzie *w* to znaleziony optymalny
        punkt *w*, a *log_values* to lista wartości funkcji celu w każdej
        epoce (lista o długości *epochs*)
    """
    w = w0
    wA = []
    _, grad = obj_fun(w0)
    for i in range(epochs):
        w = w - eta * grad
        val, grad = obj_fun(w)
        wA.append(val)
    return w, np.reshape(np.array(wA), epochs)


def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    """
    :param obj_fun: optymalizowana funkcja celu, przyjmująca jako argumenty
        wektor parametrów *w*, paczkę danych składających się z danych
        treningowych *x* i odpowiadających im etykiet *y*
        [wywołanie *val, grad = obj_fun(w, x, y)*]
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok stochastycznego algorytmu gradientu prostego
    :param eta: krok uczenia 
    :param mini_batch: rozmiar paczki danych / mini-batcha
    :return: krotka (w, log_values), gdzie *w* to znaleziony optymalny
        punkt *w*, a *log_values* to lista wartości funkcji celu dla całego
        zbioru treningowego w każdej epoce (lista o długości *epochs*)
    """
    M = int(y_train.shape[0] / mini_batch)
    x_mini_batch = np.vsplit(x_train, M)
    y_mini_batch = np.vsplit(y_train, M)

    w = w0
    wA = []
    for i in range(epochs):
        for x, y in zip(x_mini_batch, y_mini_batch):
            grad = obj_fun(w, x, y)[1]
            w = w - eta * grad
        wA.append(obj_fun(w, x_train, y_train)[0])
    return w, np.array(wA).reshape(epochs)


def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    """

    :param w: wektor parametrów modelu Mx1
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :param regularization_lambda: parametr regularyzacji l2
    :return: krotka (log, grad), gdzie *log* to wartość funkcji logistycznej
        z regularyzacją l2, a *grad* jej gradient po parametrach *w* Mx1
    """
    ws = np.delete(w, 0)
    sigArr = sigmoid(x_train @ w)
    w = w.transpose()

    norm = regularization_lambda / 2 * (np.linalg.norm(ws) ** 2)
    outArr = np.divide(y_train * np.log(sigArr) + (1 - y_train) * np.log(1 - sigArr), -1 * sigArr.shape[0])
    wz = w.copy().transpose()
    wz[0] = 0
    grad = (x_train.transpose() @ (sigArr - y_train)) / sigArr.shape[0] + regularization_lambda * wz
    return np.sum(outArr) + norm, grad


def prediction(x, w, theta):
    """
.
    :param x: macierz obserwacji NxM
    :param w: wektor parametrów modelu Mx1
    :param theta: próg klasyfikacji z przedziału [0,1]
    :return: wektor predykowanych etykiet ze zbioru {0, 1} Nx1
    """
    return np.vectorize(lambda s: s >= theta)(sigmoid(x @ w))


def f_measure(y_true, y_pred):
    """
    :param y_true: wektor rzeczywistych etykiet Nx1
    :param y_pred: wektor etykiet predykowanych przed model Nx1
    :return: wartość miary F (F-measure)
    """
    TP = np.sum(y_true & y_pred)
    FP_plus_FN = np.sum(y_true ^ y_pred) / 2
    return TP / (TP + FP_plus_FN)


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    """
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :param x_val: zbiór danych walidacyjnych NxM
    :param y_val: etykiety klas dla danych walidacyjnych Nx1
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok stochastycznego algorytmu gradientu prostego
    :param eta: krok uczenia
    :param mini_batch: rozmiar paczki danych / mini-batcha
    :param lambdas: lista wartości parametru regularyzacji l2 *lambda*,
        które mają być sprawdzone
    :param thetas: lista wartości progów klasyfikacji *theta*,
        które mają być sprawdzone
    :return: krotka (regularization_lambda, theta, w, F), gdzie
        *regularization_lambda* to wartość regularyzacji *lambda* dla
        najlepszego modelu, *theta* to najlepszy próg klasyfikacji,
        *w* to parametry najlepszego modelu, a *F* to macierz wartości miary F
        dla wszystkich par *(lambda, theta)* #lambda x #theta
    """
    F = np.zeros(shape=(len(lambdas), len(thetas)))
    max_f = [0, 0, 0, -1]
    for i, λ in enumerate(lambdas):
        obj_fun = lambda w, x, y: regularized_logistic_cost_function(w, x, y, λ)
        w, func_values = stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch)
        for j, θ in enumerate(thetas):
            f = f_measure(y_val, prediction(x_val, w, θ))
            F[i, j] = f
            if f > max_f[3]: max_f = [λ, θ, w, f]
    max_f[3] = F
    return tuple(max_f)
