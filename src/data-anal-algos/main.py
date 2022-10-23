import random
from sklearn import datasets
from sklearn import model_selection
import numpy as np


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
# %matplotlib inline


# Введём функцию подсчёта точности, как доли правильных ответов
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def mse_metric(actual, predicted):
    se = (np.square(actual - predicted)).mean()
    return se

def get_meshgrid(data, step=0.01, border=1.2):
    x_min, x_max = data[:, 0].min() - border, data[:, 0].max() + border
    y_min, y_max = data[:, 1].min() - border, data[:, 1].max() + border
    return np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))


# Расчёт критерия Джини
def gini(labels):
    #  подсчёт количества объектов разных классов
    classes = {}
    for label in labels:
        if label not in classes:
            classes[label] = 0
        classes[label] += 1

    #  расчёт критерия
    impurity = 1  # "impurity" - "нечистота", степень неопределённости
    for label in classes:
        p = classes[label] / len(labels)
        impurity -= p ** 2

    return impurity


# Расчёт качества
def quality(left_labels, right_labels, current_gini):
    # доля выборки, ушедшей в левое поддерево
    p = float(left_labels.shape[0]) / (left_labels.shape[0] + right_labels.shape[0])
    return current_gini - p * gini(left_labels) - (1 - p) * gini(right_labels)


# Разбиение датасета в узле
def split(data, labels, index, t):
    left = np.where(data[:, index] <= t)
    right = np.where(data[:, index] > t)
    true_data = data[left]
    false_data = data[right]
    true_labels = labels[left]
    false_labels = labels[right]
    return true_data, false_data, true_labels, false_labels


class QualityCriteria:
    def measure_befire_split(self, y_values):
        return 0

    def quality(self, left_values, right_values, before_split_measure):
        return 0


# Оценка качества критерием Gini
class GiniCriteria(QualityCriteria):
    def measure_befire_split(self, y_values):
        return gini(y_values)

    def quality(self, left_values, right_values, before_split_measure):
        return quality(left_values, right_values, before_split_measure)


class VarianceCriteria(QualityCriteria):
    def quality(self, left_values, right_values, before_split_measure):
        v1 = np.var(left_values)
        v2 = np.var(right_values)
        return -(v1 + v2)


class Params:

    def __init__(self, min_leaf=1, max_depth=1000000, quality_criteria=GiniCriteria()):
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.quality_criteria = quality_criteria

    def __str__(self):
        return f'Params(min_leaf={self.min_leaf} max_depth={self.max_depth}, criteria={self.quality_criteria})'


class BaseNode:
    def print_tree(self, spacing=""):
        pass

    def classify(self, obj):
        pass


# Реализуем класс узла
class Node(BaseNode):

    def __init__(self, index, t, true_branch, false_branch):
        self.index = index  # индекс признака, по которому ведётся сравнение с порогом в этом узле
        self.t = t  # значение порога
        self.true_branch = true_branch  # поддерево, удовлетворяющее условию в узле
        self.false_branch = false_branch  # поддерево, не удовлетворяющее условию в узле

    def classify_all(self, objects):
        return list(map(lambda obj: self.classify(obj).prediction, objects))

    def regress_all(self, objects):
        return list(map(lambda obj: self.classify(obj).mean, objects))

    def classify(self, obj):
        if obj[self.index] <= self.t:
            return self.true_branch.classify(obj)
        else:
            return self.false_branch.classify(obj)

    # Напечатаем ход нашего дерева
    def print_tree(self, spacing=""):
        # Выведем значение индекса и порога на этом узле
        print(spacing + 'Индекс:', str(self.index), '  Порог:', str(self.t))

        # Рекурсионный вызов функции на положительном поддереве
        print(spacing + '--> True:')
        self.true_branch.print_tree(spacing + "  ")

        # Рекурсионный вызов функции на положительном поддереве
        print(spacing + '--> False:')
        self.false_branch.print_tree(spacing + "  ")


# И класс терминального узла (листа)
class Leaf(BaseNode):

    def __init__(self, data, y_true):
        self.data = data
        self.y_true = y_true  # y_true
        self.prediction = self.predict(y_true)  # y_pred
        self.mean = np.mean(y_true)

    def classify(self, obj):
        return self

    def predict(self, y_true):
        # подсчёт количества объектов разных классов
        classes = {}  # сформируем словарь "класс: количество объектов"
        for label in y_true:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1
        #  найдём класс, количество объектов которого будет максимальным в этом листе, и вернём его
        prediction = max(classes, key=classes.get)
        return prediction

    def print_tree(self, spacing=""):
        print(f'{spacing} Прогноз: {self.prediction},  Среднее: {self.mean},  Объектов: {len(self.data)}')


# Нахождение наилучшего разбиения
def find_best_split(data, labels, params):
    min_leaf = params.min_leaf

    criteria = params.quality_criteria

    current_metric = criteria.measure_befire_split(labels)

    best_quality = 0
    best_t = None
    best_index = None

    n_features = data.shape[1]

    for index in range(n_features):
        t_values = [row[index] for row in data]

        for t in t_values:
            true_data, false_data, true_labels, false_labels = split(data, labels, index, t)
            #  пропускаем разбиения, где в узле остаётся менее min_leaf объектов
            if len(true_data) < min_leaf or len(false_data) < min_leaf:
                continue

            current_quality = criteria.quality(true_labels, false_labels, current_metric)

            #  выбираем порог, на котором получается максимальный прирост качества
            if (best_t == None) or (current_quality > best_quality):
                best_quality, best_t, best_index = current_quality, t, index

    return best_t, best_index


# Построение дерева посредством рекурсивной функции

def build_tree(data, labels, params, current_depth=1):
    return_leaf = False
    t = None
    index = None

    if current_depth > params.max_depth:
        return_leaf = True
    else:
        t, index = find_best_split(data, labels, params)

    if return_leaf or (t == None):
        return Leaf(data, labels)

    true_data, false_data, true_labels, false_labels = split(data, labels, index, t)

    # Рекурсивно строим два поддерева
    true_branch = build_tree(true_data, true_labels, params, current_depth + 1)
    false_branch = build_tree(false_data, false_labels, params, current_depth + 1)

    # Возвращаем класс узла со всеми поддеревьями, то есть целого дерева
    return Node(index, t, true_branch, false_branch)


def test_tree(params, train_data, train_labels, test_data, test_labels):
    # Построим дерево по обучающей выборке
    my_tree = build_tree(train_data, train_labels, params)
    print('Дерево по параметрам:', params)
    my_tree.print_tree()

    # Получим ответы для обучающей и тестовой выборки
    train_answers = my_tree.classify_all(train_data)
    answers = my_tree.classify_all(test_data)

    # Точность на обучающей и тестовой выборке
    train_accuracy = accuracy_metric(train_labels, train_answers)
    test_accuracy = accuracy_metric(test_labels, answers)

    print(f'Точность на обучающей выборке: {train_accuracy}\nТочность на тестовой выборке: {test_accuracy}')

    # Визуализируем дерево на графике
    plt.figure(figsize = (16, 7))

    # график обучающей выборки
    plt.subplot(1,2,1)
    xx, yy = get_meshgrid(train_data)
    mesh_predictions = np.array(my_tree.classify_all(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.pcolormesh(xx, yy, mesh_predictions, cmap = light_colors)
    plt.scatter(train_data[:, 0], train_data[:, 1], c = train_labels, cmap = colors)
    plt.title(f'Train accuracy={train_accuracy:.2f}')

    # график тестовой выборки
    plt.subplot(1,2,2)
    plt.pcolormesh(xx, yy, mesh_predictions, cmap = light_colors)
    plt.scatter(test_data[:, 0], test_data[:, 1], c = test_labels, cmap = colors)
    plt.title(f'Test accuracy={test_accuracy:.2f}')


def classification():
    # сгенерируем данные
    classification_data, classification_labels = datasets.make_classification(
        n_features = 2,
        n_informative = 2,
        n_classes = 2,
        n_redundant=0,
        n_clusters_per_class=2,
        random_state=8
        )

    # визуализируем сгенерированные данные
    colors = ListedColormap(['red', 'blue'])
    light_colors = ListedColormap(['lightcoral', 'lightblue'])
    plt.figure(figsize=(8,8))
    plt.scatter(list(map(lambda x: x[0], classification_data)), list(map(lambda x: x[1], classification_data)),
                  c=classification_labels, cmap=colors)

    # Разобьём выборку на обучающую и тестовую
    train_data, test_data, train_labels, test_labels = model_selection.train_test_split(classification_data,
                                                                                         classification_labels,
                                                                                         test_size = 0.3,
                                                                                         random_state = 1)


    # Проверим результаты построения дерева с разными параметрами:
    test_tree(Params(min_leaf=10), train_data, train_labels, test_data, test_labels)


# Регрессия
def regression():
    data, target, coef = datasets.make_regression(n_samples = 100, n_features = 2, n_informative = 2, n_targets = 1,
                                                  noise = 10, coef = True, random_state = 42)
    # plt.scatter(data[:,0], data[:,1], c=target)

    train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data, target, test_size = 0.3, random_state = 1)

    params = Params(max_depth=5, quality_criteria=VarianceCriteria())
    my_tree = build_tree(train_data, train_labels, params)
    print('Дерево по параметрам:', params)
    my_tree.print_tree()

    # Получим ответы для обучающей и тестовой выборки
    train_answers = my_tree.classify_all(train_data)
    answers = my_tree.classify_all(test_data)

    # Точность на обучающей и тестовой выборке
    train_accuracy = mse_metric(train_labels, train_answers)
    test_accuracy = mse_metric(test_labels, answers)

    print(f'MSE на обучающей выборке: {train_accuracy}\nMSE на тестовой выборке: {test_accuracy}')

    # визуализируем сгенерированные данные
    colors = ListedColormap(['red', 'blue'])
    light_colors = ListedColormap(['lightcoral', 'lightblue'])

    # Визуализируем дерево на графике
    plt.figure(figsize=(16, 7))

    # график обучающей выборки
    plt.subplot(1, 2, 1)
    xx, yy = get_meshgrid(train_data)
    mesh_predictions = np.array(my_tree.classify_all(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    plt.pcolormesh(xx, yy, mesh_predictions, cmap=light_colors)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap=colors)
    plt.title(f'Train accuracy={train_accuracy:.2f}')

    # график тестовой выборки
    plt.subplot(1, 2, 2)
    plt.pcolormesh(xx, yy, mesh_predictions, cmap=light_colors)
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, cmap=colors)
    plt.title(f'Test accuracy={test_accuracy:.2f}')

    input("Press Enter to continue...")


regression()

