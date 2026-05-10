import pandas as pd
import numpy as np

df = pd.read_csv(
    r"C:\Users\Admin\Desktop\Projects\AIML\Prison Overcrowding Predictor\Main\prisons.csv",
    encoding='latin1',
    on_bad_lines='skip',
    delimiter='\t'
)

print(f"Original shape: {df.shape}")

required_columns = ['FACILITYID', 'POPULATION', 'CAPACITY', 'STATUS']
existing_columns = [col for col in required_columns if col in df.columns]
df = df[existing_columns]

df = df[df['STATUS'] == 'OPEN']

df = df.drop('STATUS', axis=1)


df['POPULATION'] = df['POPULATION'].replace(-999, np.nan)
df['CAPACITY'] = df['CAPACITY'].replace(-999, np.nan)

df['POPULATION'] = df['POPULATION'].fillna(df['POPULATION'].median())
df['CAPACITY'] = df['CAPACITY'].fillna(df['CAPACITY'].median())

df['POPULATION'] = df['POPULATION'].astype(int)
df['CAPACITY'] = df['CAPACITY'].astype(int)

df['available_slots'] = df['CAPACITY'] - df['POPULATION']

df['is_overcrowded'] = (df['POPULATION'] > df['CAPACITY']).astype(int)
overcrowded_count = df['is_overcrowded'].sum()
not_overcrowded_count = len(df) - overcrowded_count

print(f"\nFinal shape: {df.shape}")
print(f"Final columns: {df.columns.tolist()}")

print(f"\nOvercrowded Facilities: {overcrowded_count}")
print(f"Not Overcrowded Facilities: {not_overcrowded_count}")

df.to_excel(r'C:\Users\Admin\Desktop\preprocessed.xlsx', index=False)

print(f"\nFile saved")

print(df.head(10))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X = df[['POPULATION', 'CAPACITY']]
y = df['is_overcrowded']

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


rm = RandomForestClassifier()
rm.fit(xtrain, ytrain)
ypred = rm.predict(xtest)
print("Accuracy of Random:",accuracy_score(ytest,ypred))

import joblib

# Save the model
joblib.dump(rm, r'C:\Users\Admin\Desktop\Projects\AIML\Prison Overcrowding Predictor\random_forest_model.pkl')
print("✅ Model saved as 'random_forest_model.pkl'")
