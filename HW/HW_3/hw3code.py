import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator


def compute_bias_variance(
        regressor,
        dependence_fun,
        x_generator=np.random.uniform,
        noise_generator=np.random.uniform,
        sample_size=300,
        samples_num=300,
        objects_num=200,
        seed=1234):

    np.random.seed(seed)
    X_arr = []
    noise_arr = []
    for i in range(samples_num):
        X = x_generator(size=sample_size)
        noise = noise_generator(size=sample_size)
        X_arr.append(X)
        noise_arr.append(noise)

    objects = x_generator(size=objects_num)
    mean_noise = np.mean(noise)
    bias, variance = compute_bias_variance_fixed_samples(
        regressor, dependence_fun, X_arr, objects, noise_arr, mean_noise)
    return bias, variance


def compute_bias_variance_fixed_samples(
        regressor,
        dependence_fun,
        samples,
        objects,
        noise,
        mean_noise):

    E_y = dependence_fun(objects) + mean_noise
    preds_array = []
    samples_num, samples_size = np.array(samples).shape
    objects_for_pred = objects[:, np.newaxis]
    for i in range(samples_num):
        x = samples[i]
        y = dependence_fun(x) + noise[i]
        x = x.reshape(-1, 1)
        regressor.fit(x, y)
        preds = regressor.predict(objects_for_pred)
        preds_array.append(preds)
    E_X_mu = np.mean(preds_array, axis=0)

    bias = np.mean((E_X_mu - E_y)**2)
    temp_E = np.mean((preds_array - E_X_mu)**2, axis=0)
    variance = np.mean(temp_E)
    return bias, variance


def find_best_split(feature_vector, target_vector):
    # feature_vector, target_vector = zip(*sorted(zip(feature_vector, target_vector)))

    # ----
    cutoffs = np.unique(feature_vector)
    thresholds = ((np.roll(cutoffs, -1) + cutoffs) / 2.)[:-1]
    # ---

    a_u, a_c = np.unique(feature_vector, return_counts=True)
    a_temp = np.append(feature_vector, a_u)
    target_temp = np.append(target_vector, np.ones(len(a_u)))

    _, a_by_target = np.unique(a_temp[np.where(target_temp == 1)], return_counts=True)
    var_cumsum = a_by_target - 1

    a_unique, value_count = np.unique(feature_vector, return_counts=True)

    H_l_1 = (np.cumsum(var_cumsum) * 1. / np.cumsum(value_count))[:-1] ** 2
    H_l_0 = (np.cumsum(value_count - var_cumsum) * 1. / np.cumsum(value_count))[:-1] ** 2

    H_r_1 = (np.cumsum(var_cumsum[::-1]) * 1. / np.cumsum(value_count[::-1]))[:-1][::-1] ** 2
    H_r_0 = (np.cumsum((value_count - var_cumsum)[::-1]) * 1. / np.cumsum(value_count[::-1]))[:-1][::-1] ** 2

    left_delitel = (np.cumsum(value_count) * 1. / len(target_vector))[:-1]
    right_delitel = 1 - left_delitel

    Gini_l = (1 - (H_l_1 + H_l_0)) * left_delitel
    Gini_r = (1 - (H_r_1 + H_r_0)) * right_delitel
    ginis = -Gini_l - Gini_r

    if len(np.unique(feature_vector)) == 1:
        return None, None, None, -1
    else:
        return thresholds, ginis, thresholds[np.argmax(ginis)], np.max(ginis)


class DecisionTree(BaseEstimator):
    BaseEstimator()
    def __init__(self, feature_types, max_depth=None,
                 min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self.feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            # !!!!
            # print feature
            # !!!!
            feature_type = self.feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
                feature_vector = np.array(feature_vector)  # !!!
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click/current_count
                sorted_categories = list(map(lambda x: x[0],
                                             sorted(ratio.items(),
                                                    key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories,
                                          list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(feature_vector) == 0:
                continue
            # ---
            if (sub_y[0] != 0 and sub_y[0] != 1):
                b = np.unique(sub_y)
                sub_y[sub_y == b[0]] = 1
                sub_y[sub_y == b[1]] = 0
                sub_y = sub_y.astype(float)
            # ---
            _, _, threshold, gini = find_best_split(feature_vector, np.array(sub_y))
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    # ---
                    # q = np.unique(sub_y)
                    # s = {q[0]: 0, q[1]: 1}
                    # sub_y = map(sub_y, s)
                    # a = np.array(['e', 'p', 'p', 'e', 'p'])

                    # ---
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold,
                                                     categories_map.items())))
                else:
                    raise ValueError

        if gini_best == -1:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)],
                       sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        x = np.array(x)
        tree = node
        class_i = None
        for i in range(100):
            if tree['type'] == 'terminal':
                class_i = tree['class']
                break
            else:
                feature_split_i = tree['feature_split']
                if self.feature_types[tree['feature_split']] == 'real':
                    real_split_i = tree['threshold']
                    if x[feature_split_i] < real_split_i:
                        tree = tree['left_child']
                    else:
                        tree = tree['right_child']
                else:
                    categories_split_i = tree['categories_split']
                    if x[feature_split_i] in categories_split_i:
                        tree = tree['left_child']
                    else:
                        tree = tree['right_child']
        return class_i

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
