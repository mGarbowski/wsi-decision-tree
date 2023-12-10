from decision_trees import load_dataset, DecisionTreeClassifier


def main():
    dataset = load_dataset("data/mushroom/agaricus-lepiota.data")
    model = DecisionTreeClassifier.train(dataset)
    pass


if __name__ == '__main__':
    main()
