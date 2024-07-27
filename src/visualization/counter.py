import csv
from collections import Counter

def count_numbers_in_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        numbers = [int(row[0]) for row in reader if row]  # Assuming a single column of numeric values

    counts = Counter(numbers)
    
    for number, count in counts.items():
        print(f"Number {number}: {count} times")

if __name__ == "__main__":
    # Replace 'numbers.csv' with the path to your CSV file
    count_numbers_in_csv(r'C:\Users\Headwall\Desktop\BeetleClassifier\trainingData\labels.csv')