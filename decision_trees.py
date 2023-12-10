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
