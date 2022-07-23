import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from mydatasets import load_dataset
from utils import train, evaluate
from plots import plot_learning_curves, plot_confusion_matrix
from mymodels import MyMLP, MyCNN
import shap
import matplotlib.pyplot as plt
from focal_loss import FocalLoss, reweight

#TODO: https://pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection

# Set a correct path to the seizure data file you downloaded
PATH_FILE = "../data/compiled_FC_75.csv"

# Path for saving model
PATH_OUTPUT = "../output/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

# Some parameters
MODEL_TYPE = 'CNN'  # TODO: Change this to 'MLP' or 'CNN'
NUM_EPOCHS = 500
BATCH_SIZE = 512
USE_CUDA = True  # Set 'True' if you want to use GPU
NUM_WORKERS = 0  # Number of threads used by DataLoader. You can adjust this according to your machine spec.

device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
if device.type == "cuda":
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

train_dataset, valid_dataset, test_dataset, features, test_data = load_dataset(PATH_FILE, MODEL_TYPE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


if MODEL_TYPE == 'MLP':
	model = MyMLP()
	save_file = 'MyMLP.pth'
elif MODEL_TYPE == 'CNN':
	model = MyCNN()
	save_file = 'MyCNN.pth'
else:
	raise AssertionError("Wrong Model Type!")

cls_num_list = [18930, 5297, 22783, 14090, 10341]

criterion = nn.CrossEntropyLoss()
#criterion = FocalLoss(weight=reweight(cls_num_list), gamma=1)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay = 0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)

best_val_acc = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []

for epoch in range(NUM_EPOCHS):
	train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
	valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

	train_losses.append(train_loss)
	valid_losses.append(valid_loss)

	train_accuracies.append(train_accuracy)
	valid_accuracies.append(valid_accuracy)

	is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
	if is_best:
		best_val_acc = valid_accuracy
		torch.save(model, os.path.join(PATH_OUTPUT, save_file), _use_new_zipfile_serialization=False)

plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)

best_model = torch.load(os.path.join(PATH_OUTPUT, save_file))
test_loss, test_accuracy, test_results = evaluate(best_model, device, test_loader, criterion)

class_names = ['Awake', 'N1', 'N2', 'N3', 'REM']
plot_confusion_matrix(test_results, class_names)

# test_data_new = []

# for i, (inp, target) in enumerate(test_loader):
#   test_data_new.append(inp)

# #SHAP computation

# tests = torch.cat(test_data_new, dim=0).to(device)

# e = shap.DeepExplainer(model, tests)
# shap_values = e.shap_values(tests)
# shap.summary_plot(shap_values=shap_values, features=tests, feature_names=features, max_display=10)
# # plt.savefig("ShapPlot.jpg")
# plt.clf()
