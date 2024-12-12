import pandas as pd
df = pd.read_csv("C:/harB00967434/harts.csv")
# Display basic information about the dataset
df_info = df.info()
df_head = df.head()
df_summary = df.describe(include='all')

print(df_info)
print(df_head)
print(df_summary)

print(df.columns)


print(df['activity'].value_counts())

data_cleaned = df.dropna()


data_cleaned = data_cleaned.drop_duplicates()


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
numeric_features = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
data_cleaned[numeric_features] = scaler.fit_transform(data_cleaned[numeric_features])



X = data_cleaned.drop(['activity'], axis=1)  # Replace 'Activity' with the actual target column
y = data_cleaned['activity']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report

# Make predictions using the trained model
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print accuracy as a percentage (100%)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print detailed classification report
print(classification_report(y_test, y_pred))


