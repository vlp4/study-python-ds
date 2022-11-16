# 1. Для реализованной в методичке модели градиентного бустинга построить зависимости ошибки от количества деревьев при
# разных значениях шага градиента на одном графике и для разной глубины деревьев на другом. Сделать выводы о
# зависимости ошибки от этих гиперпараметров (шаг градиента, максимальная глубина деревьев, количество деревьев).
# Подобрать оптимальные значения этих гиперпараметров (минимум ошибки на тесте при отсутствии переобучения).
#
# 2. Модифицируйте реализованный алгоритм, чтобы получился стохастический градиентный бустинг. Размер подвыборки примите
# равным 0.5. Сравните на одном графике кривые изменения ошибки на тестовой выборке в зависимости от числа итераций.
#
# 3. Модифицируйте алгоритм градиентного бустинга, взяв за основу реализацию решающего дерева из ДЗ_4 (для задачи
# регрессии). Сделать выводы о качестве алгоритма по сравнению с реализацией из п.1.

import random
from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_diabetes

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib


def gb_predict(X, trees_list, coef_list, eta):
    # Реализуемый алгоритм градиентного бустинга будет инициализироваться нулевыми значениями,
    # поэтому все деревья из списка trees_list уже считаются дополнительными и при предсказании прибавляются с шагом eta
    return np.array([sum([eta * coef * alg.predict([x])[0] for alg, coef in zip(trees_list, coef_list)]) for x in X])


def mean_squared_error(y_real, prediction):
    return (sum((y_real - prediction) ** 2)) / len(y_real)


def bias(y, z):
    return (y - z)


def identity(x, y):
    return x, y


def random_samples(x, y, n_samples):
    data_len = len(x)
    result_x = np.zeros((n_samples, x.shape[1]))
    result_y = np.zeros(n_samples)
    for i in range(n_samples):
        rand_index = random.randint(0, data_len - 1)
        result_x[i] = x[rand_index]
        result_y[i] = y[rand_index]
    return result_x, result_y


def gb_fit(n_trees, max_depth, X_train0, X_test, y_train0, y_test, coefs, eta, sample_choice=identity):
    # Деревья будем записывать в список
    trees = []

    # Будем записывать ошибки на обучающей и тестовой выборке на каждой итерации в список
    train_errors = []
    test_errors = []

    for i in range(n_trees):
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)

        # Подготовим выборку для обучения
        X_train, y_train = sample_choice(X_train0, y_train0)

        # инициализируем бустинг начальным алгоритмом, возвращающим ноль,
        # поэтому первый алгоритм просто обучаем на выборке и добавляем в список
        if len(trees) == 0:
            # обучаем первое дерево на обучающей выборке
            tree.fit(X_train, y_train)

            train_errors.append(mean_squared_error(y_train, gb_predict(X_train, trees, coefs, eta)))
            test_errors.append(mean_squared_error(y_test, gb_predict(X_test, trees, coefs, eta)))
        else:
            # Получим ответы на текущей композиции
            target = gb_predict(X_train, trees, coefs, eta)

            # алгоритмы начиная со второго обучаем на сдвиг
            tree.fit(X_train, bias(y_train, target))

            train_errors.append(mean_squared_error(y_train, gb_predict(X_train, trees, coefs, eta)))
            test_errors.append(mean_squared_error(y_test, gb_predict(X_test, trees, coefs, eta)))

        trees.append(tree)

    return trees, train_errors, test_errors


def evaluate_alg(X_train, X_test, y_train, y_test, trees, coefs, eta):
    train_prediction = gb_predict(X_train, trees, coefs, eta)

    print(f'Ошибка алгоритма из {n_trees} деревьев глубиной {max_depth} \
    с шагом {eta} на тренировочной выборке: {mean_squared_error(y_train, train_prediction)}')

    test_prediction = gb_predict(X_test, trees, coefs, eta)

    print(f'Ошибка алгоритма из {n_trees} деревьев глубиной {max_depth} \
    с шагом {eta} на тестовой выборке: {mean_squared_error(y_test, test_prediction)}')


def get_error_plot(n_trees, train_err, test_err):
    plt.xlabel('Iteration number')
    plt.ylabel('MSE')
    plt.xlim(0, n_trees)
    plt.plot(list(range(n_trees)), train_err, label='train error')
    plt.plot(list(range(n_trees)), test_err, label='test error')
    plt.legend(loc='upper right')
    plt.show()


random.seed(42)
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

n_trees = 14
coefs = [1] * n_trees

# fig, axes = plt.subplots(2, 2, figsize=(16, 9))
# axes_list = axes.flatten()
fig = plt.figure(constrained_layout=True, figsize=(16, 9))
gs = GridSpec(2, 2, figure=fig)

max_depth = 3
ax = fig.add_subplot(gs[0, 0])
ax.set_title(f'Test error, max_depth={max_depth}')
ax.set_xlabel('Iteration number')
for eta in [1, 0.5, 0.2, 0.1, 0.05, 0.01]:
    trees, train_errors, test_errors = gb_fit(n_trees, max_depth, X_train, X_test, y_train, y_test, coefs, eta)
    # ax.plot(list(range(n_trees)), train_errors, label=f'Train eta={eta}')
    ax.plot(list(range(n_trees)), test_errors, label=f'eta={eta}')
    ax.legend(loc='upper right')

eta = 0.1
ax = fig.add_subplot(gs[0, 1])
ax.set_title(f'Test error, eta={eta}')
ax.set_xlabel('Iteration number')
for depth in [1, 5, 10]:
    trees, train_errors, test_errors = gb_fit(n_trees, depth, X_train, X_test, y_train, y_test, coefs, eta)
    # ax.plot(list(range(n_trees)), train_errors, label=f'Train depth={depth}')
    ax.plot(list(range(n_trees)), test_errors, label=f'max_depth={depth}')
    ax.legend(loc='upper right')

eta = 0.2
n_trees = 16
max_depth = 3
trees_plain, train_errors_plain, test_errors_plain = gb_fit(n_trees, max_depth, X_train, X_test, y_train, y_test, coefs, eta)
trees_stohastic, train_errors_stohastic, test_errors_stohastic = gb_fit(n_trees, max_depth, X_train, X_test, y_train, y_test, coefs, eta,
                                                lambda x, y: random_samples(x, y, int(len(x) / 2)))
ax = fig.add_subplot(gs[1, :])
ax.set_title(f'Stohastic vs normal gradient boosting eta={eta}, max_depth={max_depth}')
ax.set_xlabel('Iteration number')
ax.plot(list(range(n_trees)), train_errors_plain, label=f'Normal')
ax.plot(list(range(n_trees)), train_errors_stohastic, label=f'Stohastic')
ax.legend(loc='upper right')

plt.show()

input('Press enter')
