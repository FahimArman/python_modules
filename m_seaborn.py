import seaborn as sn
import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))
#heat-map
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
