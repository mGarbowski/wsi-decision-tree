import math
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar('T')


class Dataset:
    """Represents a dataset

    A single datapoint is represented by a tuple of attributes and a class
    tuple and class are located at corresponding indices in the lists
    All attributes and classes are strings

    attributes and classes are of equal length
    """
    _attributes: list[tuple[str, ...]]
    _classes: list[str]

    def __init__(self, attributes: list[tuple[str, ...]] = None, classes: list[str] = None):
        if attributes is None and classes is None:
            self._attributes = []
            self._classes = []
            return

        if len(attributes) != len(classes):
            raise ValueError("Attributes and classes must have equal length")

        self._attributes = attributes if attributes is not None else []
        self._classes = classes if classes is not None else []

    def __eq__(self, other: 'Dataset') -> bool:
        return self.attributes == other.attributes and self.classes == other.classes

    def __getitem__(self, index: int) -> tuple[tuple[str, ...], str]:
        return self._attributes[index], self._classes[index]

    @property
    def attributes(self) -> list[tuple[str, ...]]:
        return self._attributes

    @property
    def classes(self) -> list[str]:
        return self._classes

    def is_empty(self) -> bool:
        return len(self._attributes) == 0

    def size(self):
        return len(self._attributes)

    def add_row(self, row_attributes: tuple[str, ...], row_class: str):
        self._attributes.append(row_attributes)
        self._classes.append(row_class)

    def split_by_attribute(self, attribute_idx: int) -> list['Dataset']:
        unique_attribute_values = set(data_point[attribute_idx] for data_point in self._attributes)
        new_datasets = {attr_value: Dataset() for attr_value in unique_attribute_values}

        for row_attributes, row_class in zip(self._attributes, self._classes):
            split_attribute = row_attributes[attribute_idx]
            new_datasets[split_attribute].add_row(row_attributes, row_class)

        return list(new_datasets.values())


def load_dataset(file_path: str, class_idx: int = 0) -> Dataset:
    with open(file_path, mode="rt", encoding="utf-8") as file:
        lines = file.readlines()
        attributes = []
        classes = []
        for line in lines:
            values = line.strip().split(",")
            cls = values.pop(class_idx)
            attrs = tuple(values)
            attributes.append(attrs)
            classes.append(cls)

        return Dataset(attributes, classes)


def entropy(dataset: Dataset) -> float:
    unique_classes = set(dataset.classes)
    entropy_value = 0
    for unique_class in unique_classes:
        class_frequency = dataset.classes.count(unique_class) / dataset.size()
        entropy_value -= class_frequency * math.log(class_frequency, 2)

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


def build_decision_tree(
        training_set: Dataset,
        unused_attribute_idxs: set[int],
) -> Node:
    if training_set.is_empty():
        raise ValueError("Training set cannot be empty")

    _, final_label = training_set[0]
    if all(label == final_label for label in training_set.classes):
        return Node.leaf(final_label)

    if len(unused_attribute_idxs) == 0:
        most_common_label = most_common_element(training_set.classes)
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
