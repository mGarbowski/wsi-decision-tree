from dataset import Dataset
from decision_trees import DecisionTreeClassifier


def evaluate_on_mushroom_dataset():
    dataset = Dataset.load_from_file("data/mushroom/agaricus-lepiota.data")
    train_set, test_set = dataset.train_test_split(0.6)
    model = DecisionTreeClassifier.train(train_set)
    accuracy = model.evaluate(test_set)

    print(f"Accuracy on mushroom dataset is {accuracy * 100:.2f}%")


def evaluate_on_breast_cancer_dataset():
    dataset = Dataset.load_from_file("data/breast+cancer/breast-cancer.data")
    train_set, test_set = dataset.train_test_split(0.6)
    model = DecisionTreeClassifier.train(train_set)
    accuracy = model.evaluate(test_set)

    print(f"accuracy on breast cancer dataset is {accuracy * 100:.2f}%")


def main():
    evaluate_on_mushroom_dataset()
    evaluate_on_breast_cancer_dataset()


if __name__ == '__main__':
    main()
