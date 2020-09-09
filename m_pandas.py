import pandas as pd

# read a file
df = pd.read_csv("file_name.csv")

print(df)
print(df.head())
print(df.head(3))

print(df.shape)

# save a file
p = model.predict(X_test)
df['new_column'] = p
d.to_csv("name.csv")
d.to_csv("name.csv",index = False)

#check null values
inputs.columns[inputs.isna().any()]

# mean of a column
s = df.column_name.mean()

#column element name
df['new_column_name'] = df.column_name.apply(lambda x: iris.target_names[x])

#get_dummies
dummies = df.get_dummies(df.column_name)
merged = pd.concat([df,dummies],axis='columns')
final = merged.drop(['column_name1','column_name2'], axis='columns')


