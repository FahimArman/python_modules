import pickle

# save

with open('model_save_name','wb') as f:
	pickle.dump(model,f)

# load

with open('model_file_name', 'rb') as f:
	mp = pickle.load(f)

# use in machine learning
# mp.predict([[5000]]) # LinearRegression()

