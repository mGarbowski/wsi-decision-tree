from decision_trees import Dataset, entropy


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
