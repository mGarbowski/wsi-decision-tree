import math
from dataclasses import dataclass
from typing import TypeVar

from dataset import Dataset, RowAttributes, Label

T = TypeVar('T')


def entropy(dataset: Dataset) -> float:
    unique_labels = set(dataset.labels)
    entropy_value = 0
    for unique_label in unique_labels:
        label_frequency = dataset.labels.count(unique_label) / dataset.size()
        entropy_value -= label_frequency * math.log(label_frequency, 2)

    return entropy_value


def entropy_after_split(dataset: Dataset, split_attribute_idx: int) -> float:
    return sum(
        partitioned_dataset.size() / dataset.size() * entropy(partitioned_dataset)
        for partitioned_dataset
        in dataset.split_by_attribute(split_attribute_idx)
    )


def information_gain(dataset: Dataset, split_attribute_idx: int) -> float:
    return entropy(dataset) - entropy_after_split(dataset, split_attribute_idx)


def best_split_idx(dataset: Dataset, unused_attribute_idxs: set[int]) -> int:
    return max(
        unused_attribute_idxs,
        key=lambda idx: information_gain(dataset, idx)
    )


def most_common_element(elements: list[T]) -> T:
    counts = {}
    for element in elements:
        if element not in counts:
            counts[element] = 1
        else:
            counts[element] += 1

    elem, count = max(counts.items(), key=lambda item: item[1])
    return elem


@dataclass
class Node:
    children: dict[str, 'Node']
    label: str | None
    split_attribute_idx: int | None

    def is_leaf(self) -> bool:
        return self.label is not None

    @classmethod
    def leaf(cls, label: str) -> 'Node':
        return cls(
            children={},
            label=label,
            split_attribute_idx=None
        )


class DecisionTreeClassifier:
    _root: Node

    def __init__(self, root: Node):
        self._root = root

    @classmethod
    def train(cls, dataset: Dataset) -> 'DecisionTreeClassifier':
        attribute_idxs = set(range(len(dataset.attributes[0])))
        root = build_decision_tree(dataset, attribute_idxs)
        return cls(root)

    def predict_single(self, row_attributes: RowAttributes) -> Label:
        """Predict label based on attributes"""
        node = self._root
        while not node.is_leaf():
            attribute_value = row_attributes[node.split_attribute_idx]
            node = node.children[attribute_value]

        return node.label

    def predict(self, attributes: list[RowAttributes]) -> list[Label]:
        """Predict label based on attributes for each row"""
        return [self.predict_single(row_attributes) for row_attributes in attributes]

    def evaluate(self, test_set: Dataset) -> float:
        """Ratio of correct predictions to all predictions"""
        actual = test_set.labels
        predicted = self.predict(test_set.attributes)
        correct_predictions = 0
        for a, p in zip(actual, predicted):
            if a == p:
                correct_predictions += 1

        return correct_predictions / len(actual)


def build_decision_tree(
        training_set: Dataset,
        unused_attribute_idxs: set[int],
) -> Node:
    if training_set.is_empty():
        raise ValueError("Training set cannot be empty")

    _, final_label = training_set[0]
    if all(label == final_label for label in training_set.labels):
        return Node.leaf(final_label)

    if len(unused_attribute_idxs) == 0:
        most_common_label = most_common_element(training_set.labels)
        return Node.leaf(most_common_label)

    split_attribute_idx = best_split_idx(training_set, unused_attribute_idxs)
    dataset_partition = training_set.split_by_attribute(split_attribute_idx)
    unused_attribute_idxs.remove(split_attribute_idx)
    children = {
        partitioned_set.attributes[0][split_attribute_idx]: build_decision_tree(
            training_set=partitioned_set,
            unused_attribute_idxs=unused_attribute_idxs.copy(),  # pass down a copy of a mutable set
        )
        for partitioned_set in dataset_partition
    }

    return Node(
        children=children,
        label=None,
        split_attribute_idx=split_attribute_idx
    )
