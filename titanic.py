import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



# Load the dataset
df = pd.read_csv('train.csv')

# Print the first 5 rows of the dataset
print(df.head())

# Check the size of the dataset
print('Number of rows:', df.shape[0])
print('Number of columns:', df.shape[1])

# Check the data types of each column(getting information)
print(df.dtypes)

# Check the number of missing values in each column
print(df.isnull().sum())

# Drop unnecessary columns
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Print the first 5 rows of the cleaned dataset
print(df.head())

# Fill in missing values for the Age column with the mean age
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)

# Check the number of missing values in each column again
print(df.isnull().sum())

# Fill missing values in 'Embarked' with the mode (most common value)
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)

# Verify there are no missing values left
print(df.isnull().sum())
import matplotlib.pyplot as plt

# Calculate the survival rate
survival_rate = df['Survived'].value_counts(normalize=True) * 100

# Create a bar plot of the survival rate
survival_rate.plot(kind='bar')
plt.xlabel('Survived')
plt.ylabel('Percentage')
plt.title('Survival Rate of Passengers')
plt.xticks(rotation=0)
plt.show()

# Calculate the survival rate by gender
gender_survival_rate = df.groupby(['Sex'])['Survived'].value_counts(normalize=True) * 100
 #Convert categorical data into numerical data
# Label encoding for 'Sex'
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-Hot Encoding for 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')
# Separate target and features
X = df.drop('Survived', axis=1)  # Features
y = df['Survived']               # Target

# Print the first 5 rows of features and target to verify
print("Features (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())
# Print the first 5 rows of the dataset after encoding
print(df.head())
# Create a stacked bar plot of the survival rate by gender
gender_survival_rate.unstack().plot(kind='bar', stacked=True)
plt.xlabel('Sex')
plt.ylabel('Percentage')
plt.title('Survival Rate of Passengers by Gender')
plt.xticks(rotation=0)
plt.legend(['Not Survived', 'Survived'], loc='upper right')
plt.show()
import seaborn as sns

# Create a histogram of the age distribution
sns.histplot(data=df, x='Age', hue='Survived', kde=True)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution of Passengers by Survival')
plt.show()



# Load the dataset
df = pd.read_csv('train.csv')

# Drop unnecessary columns
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')

# Separate target and features
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model

#creating the model/lassify data into one of two categories survied or no survival
model = LogisticRegression(max_iter=1000)
# Teaching the model using x train(features)and y train(target)
#X_train: The input data (features) used to train the model.
#y_train: The target values (outcomes) that the model learns to predict based on X_train.
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

