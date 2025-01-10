from datasets import load_dataset
import pandas as pd

# Load the IMDB dataset
dataset = load_dataset("stanfordnlp/imdb")

# Check the dataset structure
print(dataset)

# Access the training set
train_data = dataset['train']

# Access the test set
test_data = dataset['test']

# Access the validation set (if available)
# Uncomment if the dataset has a validation split
# val_data = dataset['validation']

# View the first few samples
print(train_data[0])

# Iterate through a few examples
for example in train_data[:5]:
    print(example)

# Convert to pandas DataFrame
train_df = pd.DataFrame(train_data)

# Print the first few rows
print(train_df.head())
