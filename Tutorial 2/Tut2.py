#Tahniat Khayyam 577608
#For Deep Learning Class (Spring 2026)
#%%

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier    
from sklearn.metrics import classification_report,accuracy_score
import matplotlib.pyplot as plt

iris=load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3,random_state=42)

# Standardize mean=0 var 1
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# MLP classifier
mlp=MLPClassifier(hidden_layer_sizes=(10,10),max_iter=1000,random_state=42,learning_rate_init=0.001)

# train
mlp.fit(X_train_scaled,y_train)
# preds + eval
y_pred=mlp.predict(X_test_scaled)
# eval
accuracy=accuracy_score(y_test, y_pred)
print(f'Accuracy:  {accuracy:.2f}')

# classification report
print("Classification Report: \n",classification_report(y_test,y_pred))

# disp struct of MLP
print("\nMLP Structure: ")
print(f'No of layers: {mlp.n_layers_}')
print(f'No of outputs: {mlp.n_outputs_}')
print(f'Activition Function: {mlp.activation}')
print(f'Output Activition Function: {mlp.out_activation_}')
print(f'No of epochs: {mlp.n_iter_}')


# visualize the curve
# plot the loss curve
plt.figure(figsize=(8,6))
plt.plot(mlp.loss_curve_, label="Training Loss")
plt.title("MLP Classifier Learning Curve")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# INcreasing the no. of layers to 3
print("\n\nIncreasing the no. of layers to 3")
# MLP classifier
mlp2=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000,random_state=42,learning_rate_init=0.001)

# train
mlp2.fit(X_train_scaled,y_train)
# preds + eval
y_pred=mlp2.predict(X_test_scaled)
# eval
accuracy=accuracy_score(y_test, y_pred)
print(f'Accuracy:  {accuracy:.2f}')
# visualize the curve
# plot the loss curve
plt.figure(figsize=(8,6))
plt.plot(mlp2.loss_curve_, label="Training Loss ")
plt.title("MLP Classifier Learning Curve with 3 hidden layers")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# INcreasing the no. of neurons to 20 
print("\n\nIncreasing the no. of neurons to 20 ")
# MLP classifier
mlp2=MLPClassifier(hidden_layer_sizes=(20,10),max_iter=1000,random_state=42,learning_rate_init=0.001)

# train
mlp2.fit(X_train_scaled,y_train)
# preds + eval
y_pred=mlp2.predict(X_test_scaled)
# eval
accuracy=accuracy_score(y_test, y_pred)
print(f'Accuracy:  {accuracy:.2f}')
# visualize the curve
# plot the loss curve
plt.figure(figsize=(8,6))
plt.plot(mlp2.loss_curve_, label="Training Loss ")
plt.title("MLP Classifier Learning Curve with 20 neurons in 1st hidden layer and 10 in 2nd hidden layer")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Decreasing Learning rate of MLP 
print("\n\n Decreasing Learning rate to 0.0001  ")
# MLP classifier
mlp2=MLPClassifier(hidden_layer_sizes=(10,10),max_iter=1000,random_state=42,learning_rate_init=0.0001)


# train
mlp2.fit(X_train_scaled,y_train)
# preds + eval
y_pred=mlp2.predict(X_test_scaled)
# eval
accuracy=accuracy_score(y_test, y_pred)
print(f'Accuracy:  {accuracy:.2f}')
# visualize the curve
# plot the loss curve
plt.figure(figsize=(8,6))
plt.plot(mlp2.loss_curve_, label="Training Loss ")
plt.title("MLP Classifier with LR of 0.0001")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

print(mlp.best_loss_)