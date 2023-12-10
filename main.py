from dataset import Dataset
from decision_trees import DecisionTreeClassifier


def evaluate_on_dataset(file_path: str, n_times: int = 25):
    dataset = Dataset.load_from_file(file_path)
    accuracies = []
    for _ in range(n_times):
        train_set, test_set = dataset.train_test_split(0.6)
        model = DecisionTreeClassifier.train(train_set)
        accuracy = model.evaluate(test_set)
        accuracies.append(accuracy)

    avg_accuracy = sum(accuracies) / len(accuracies)

    print(f"Average accuracy over {n_times} runs on {file_path} dataset is {avg_accuracy * 100:.2f}%")


def main():
    evaluate_on_dataset("data/mushroom/agaricus-lepiota.data")
    evaluate_on_dataset("data/breast+cancer/breast-cancer.data")


if __name__ == '__main__':
    main()
