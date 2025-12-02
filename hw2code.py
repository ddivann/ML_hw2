import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    
    sorted_indices = np.argsort(feature_vector)
    feature_vector = feature_vector[sorted_indices]
    target_vector = target_vector[sorted_indices]

    unique_values, unique_indices = np.unique(feature_vector, return_index=True)
    
    if len(unique_values) < 2:
        return [], [], None, None

    thresholds = (unique_values[:-1] + unique_values[1:]) / 2
    
    split_indices = unique_indices[1:]
    
    R = len(target_vector)
    R_l = split_indices
    R_r = R - R_l
    
    cumsum = np.cumsum(target_vector)
    h_l = cumsum[split_indices - 1]
    
    total_1 = cumsum[-1]
    h_r = total_1 - h_l
    
    p1_l = h_l / R_l
    p0_l = 1 - p1_l
    
    p1_r = h_r / R_r
    p0_r = 1 - p1_r
    
    H_l = 1 - p1_l**2 - p0_l**2
    H_r = 1 - p1_r**2 - p0_r**2
    
    ginis = - (R_l / R) * H_l - (R_r / R) * H_r
    
    best_idx = np.argmax(ginis)
    gini_best = ginis[best_idx]
    threshold_best = thresholds[best_idx]
    
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
            
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    current_click = clicks.get(key, 0)
                    ratio[key] = current_click / current_count
                sorted_categories = [cat for cat, _ in sorted(ratio.items(), key=lambda x: x[1])]
                categories_map = {cat: idx for idx, cat in enumerate(sorted_categories)}

                feature_vector = np.array([categories_map[val] for val in sub_X[:, feature]])
            else:
                raise ValueError

            if len(np.unique(feature_vector)) < 2:
                continue

            thresholds_vec, ginis_vec, _, _ = find_best_split(feature_vector, sub_y)
            
            if self._min_samples_leaf is not None:
                best_gini_local = None
                best_threshold_local = None
                
                for thr, g in zip(thresholds_vec, ginis_vec):
                    mask = feature_vector < thr
                    if np.sum(mask) >= self._min_samples_leaf and np.sum(~mask) >= self._min_samples_leaf:
                        if best_gini_local is None or g > best_gini_local:
                            best_gini_local = g
                            best_threshold_local = thr
                
                if best_gini_local is None:
                    continue
                
                gini = best_gini_local
                threshold = best_threshold_local
            else:
                best_idx = np.argmax(ginis_vec)
                gini = ginis_vec[best_idx]
                threshold = thresholds_vec[best_idx]

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature = node["feature_split"]
        feature_type = self._feature_types[feature]
        
        if feature_type == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
