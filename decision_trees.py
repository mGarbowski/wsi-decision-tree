import math


class Dataset:
    """Represents a dataset

    A single datapoint is represented by a tuple of attributes and a class
    tuple and class are located at corresponding indices in the lists
    All attributes and classes are strings

    attributes and classes are of equal length
    """
    _attributes: list[tuple[str, ...]]
    _classes: list[str]

    def __init__(self, attributes: list[tuple[str, ...]], classes: list[str]):
        if len(attributes) != len(classes):
            raise ValueError("Attributes and classes must have equal length")

        self._attributes = attributes
        self._classes = classes

    @property
    def attributes(self) -> list[tuple[str, ...]]:
        return self._attributes

    @property
    def classes(self) -> list[str]:
        return self._classes

    def size(self):
        return len(self._attributes)


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
    entropy = 0
    for unique_class in unique_classes:
        class_frequency = dataset.classes.count(unique_class) / dataset.size()
        entropy -= class_frequency * math.log(class_frequency, 2)

    return entropy
