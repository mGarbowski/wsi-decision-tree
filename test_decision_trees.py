from decision_trees import Dataset, entropy, most_common_element


def test_zero_entropy():
    dataset = Dataset(
        [("1", "2"), ("5", "1"), ("6", "3"), ("1", "2")],
        ["A", "A", "A", "A"]
    )

    assert entropy(dataset) == 0


def test_max_entropy():
    dataset = Dataset(
        [("1", "2"), ("5", "1"), ("6", "3"), ("1", "2")],
        ["A", "A", "B", "B"]
    )

    assert entropy(dataset) == 1


def test_split_dataset():
    dataset = Dataset(
        [("1", "2"), ("5", "1"), ("6", "3"), ("1", "2")],
        ["A", "A", "B", "B"]
    )

    expected_1 = Dataset(
        [("1", "2"), ("1", "2")],
        ["A", "B"]
    )

    expected_5 = Dataset(
        [("5", "1")],
        ["A"]
    )

    expected_6 = Dataset(
        [("6", "3")],
        ["B"]
    )

    after_split = dataset.split_by_attribute(attribute_idx=0)
    assert len(after_split) == 3
    assert expected_1 in after_split
    assert expected_5 in after_split
    assert expected_6 in after_split


def test_most_common_element():
    elements = [1, 6, 4, 2, 6, 1, 2, 2]
    assert most_common_element(elements) == 2
