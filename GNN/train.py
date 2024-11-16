import torch
from data_preprocessing import MoleculeDataset
from model import GNN
from torch_geometric.data import DataLoader
from utils import count_parameters, calculate_metrics
from tqdm import tqdm
import numpy as np
from test import test
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x.float(), batch.edge_attr.float(),batch.edge_index,batch.batch)
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss/step

def run_one_training(params):
    params = params[0]
    print("Loading dataset...")
    train_dataset = MoleculeDataset(root="../data/", filename="HIV_train_oversampled.csv")
    test_dataset = MoleculeDataset(root="../data/", filename="test.csv", test=True)
    params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]


    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=True)

    print("Loading model...")
    model_params = {k: v for k, v in params.items() if k.startswith("model_")}
    model = GNN(feature_size=train_dataset[0].x.shape[1], model_params=model_params)
    model = model.to(device)
    print(f"Number of parameters: {count_parameters(model)}")
   
    weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=params["learning_rate"],
                                momentum=params["sgd_momentum"],
                                weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])

    best_loss = 1000
    early_stopping_counter = 0
    for epoch in range(300):
            if early_stopping_counter <= 10: # = x * 5
                model.train()
                loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
                print(f"Epoch {epoch} | Train Loss {loss}")
                model.eval()
                if epoch % 5 == 0:
                    loss = test(epoch, model, test_loader, loss_fn)
                    print(f"Epoch {epoch} | Test Loss {loss}")

                    if float(loss) < best_loss:
                        best_loss = loss

                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                scheduler.step()
            else:
                print("Early stopping due to no improvement.")
                return [best_loss]
    print(f"Finishing training with best test loss: {best_loss}")
    return [best_loss]
