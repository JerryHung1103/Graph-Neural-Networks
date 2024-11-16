import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('../CNN/') 
from featurizer import get_GraphSAGE_embedding_feature
from graph_sage_embedding import GraphSAGE
from graph_sage_embedding import length_of_embedding

import sys
sys.path.append('../GNN/') 
from data_preprocessing import MoleculeDataset


train = MoleculeDataset(root="../data/", filename="HIV_train_oversampled.csv")
test = MoleculeDataset(root="../data/", filename="test.csv", test=True)

model = GraphSAGE(in_channels=train.num_node_features, hidden_channels=128, out_channels=length_of_embedding)
print("========== wait for converting graph to embedding ==========")
X_train, y_train = get_GraphSAGE_embedding_feature(model, train)
X_test, y_test = get_GraphSAGE_embedding_feature(model, test)

print("shape of X_train:", X_train.shape)
print("shape of y_train:", y_train.shape)

print("shape of X_test:", X_test.shape)
print("shape of y_test:", y_test.shape)


X_train=X_train.detach().numpy()
y_train = y_train.detach().numpy()

X_test=X_test.detach().numpy()
y_test=y_test.detach().numpy()


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC

svm = SVC(kernel='rbf') 
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)


accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)


conf_matrix = confusion_matrix(y_test, y_pred_svm)
print('Confusion Matrix:')
print(conf_matrix)


class_report = classification_report(y_test, y_pred_svm)
print('Classification Report:')
print(class_report)


precision = precision_score(y_test, y_pred_svm, average='weighted')
recall = recall_score(y_test, y_pred_svm, average='weighted')
f1 = f1_score(y_test, y_pred_svm, average='weighted')

print(f'Weighted Precision: {precision}')
print(f'Weighted Recall: {recall}')
print(f'Weighted F1 Score: {f1}')
