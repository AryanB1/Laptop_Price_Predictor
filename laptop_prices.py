import pandas as pd
import mlcroissant as mlc
from sklearn.preprocessing import OneHotEncoder

# Download Dataset
croissant_dataset = mlc.Dataset('https://www.kaggle.com/datasets/asinow/laptop-price-dataset/croissant/download')
df = pd.DataFrame(croissant_dataset.records(record_set='laptop_prices.csv'))

# Clean Data
df.columns = [col.split('/')[-1].replace('+', ' ') for col in df.columns]
string_cols = df.select_dtypes(include=['object']).columns
for col in string_cols:
    df[col] = df[col].str.decode('utf-8')

# One Hot Encoding
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df[string_cols])
feature_names = encoder.get_feature_names_out(string_cols)
encoded_df = pd.DataFrame(encoded_data, columns=feature_names)
num_df = df.select_dtypes(exclude=['object'])
final_df = pd.concat([num_df, encoded_df], axis=1)
print(final_df.head())