import os
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

# Define the relative path to the dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root of the project
DATA_PATH = os.path.join(BASE_DIR, "data", "raw")  # Path to raw data

# Load the dataset
dataset = load_dataset("stanfordnlp/imdb", cache_dir=DATA_PATH)

# Access the train, test, or validation splits
train_data = dataset['train']
test_data = dataset['test']

# Convert the train dataset to a pandas DataFrame
train_df = pd.DataFrame(dataset['train'])

# Display the first few rows
print(train_df.head())

# Check the number of rows and columns
print(train_df.shape)

# Inspect the columns and their data types
print(train_df.info())

# Generate basic statistics (for numeric data like labels)
print(train_df.describe())

# Count the occurrences of each label
label_counts = train_df['label'].value_counts()

# Plot the distribution
label_counts.plot(kind='bar', title='Sentiment Label Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'], rotation=0)
plt.show()




