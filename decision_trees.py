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

    @property
    def attributes(self) -> list[tuple[str, ...]]:
        return self._attributes

    @property
    def classes(self) -> list[str]:
        return self._classes

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
    pass
