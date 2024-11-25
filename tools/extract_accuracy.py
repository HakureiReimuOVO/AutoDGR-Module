import os
import json
import csv

def find_best_accuracy(directory):
    results = []

    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            best_accuracy = 0.0
            dataset_model = dir_name
            json_files = [f for f in os.listdir(os.path.join(root, dir_name)) if f.endswith('.json')]

            for json_file in json_files:
                with open(os.path.join(root, dir_name, json_file), 'r') as f:
                    lines = f.readlines()[1:]  # Skip the first line
                    for line in lines:
                        data = json.loads(line)
                        accuracy = data.get("accuracy_top-1", 0.0)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy

            results.append((dataset_model, best_accuracy))

    return results

def write_csv(results, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Dataset', 'Model', 'Best Accuracy'])
        for dataset_model, best_accuracy in results:
            parts = dataset_model.split('_')
            dataset = '_'.join(parts[:2]) + '.py'
            model = parts[2]+'.py'
            csvwriter.writerow([dataset, model, best_accuracy])

if __name__ == "__main__":
    directory = 'checkpoints/'
    output_file = 'best_accuracy.csv'
    results = find_best_accuracy(directory)
    write_csv(results, output_file)