import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from dataset import WakeWordData, collate_fn
from model import LSTMWakeWord
from sklearn.metrics import classification_report
from tabulate import tabulate

def save_checkpoint(checkpoint_path, model, optimizer, scheduler, model_params, notes=None):
    torch.save({
        "notes": notes,
        "model_params": model_params,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }, checkpoint_path)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    acc = rounded_preds.eq(y.view_as(rounded_preds)).sum().item() / len(y)
    return acc

def test(test_loader, model, device, epoch):
    print("\nStarting test for epoch %s" % epoch)
    accs = []
    preds = []
    labels = []
    with torch.no_grad():
        for idx, (mfcc, label) in enumerate(test_loader):
            mfcc, label = mfcc.to(device), label.to(device)
            output = model(mfcc)
            pred = torch.sigmoid(output)
            acc = binary_accuracy(pred, label)
            preds += torch.flatten(torch.round(pred)).cpu()
            labels += torch.flatten(label).cpu()
            accs.append(acc)
            print("Iter: {}/{}, accuracy: {}".format(idx, len(test_loader), acc), end="\r")
    average_acc = sum(accs) / len(accs) 
    print('Average test Accuracy:', average_acc, "\n")
    report = classification_report(labels, preds)
    print(report)
    return average_acc, report

def train(train_loader, model, optimizer, loss_fn, device, epoch):
    print("\nStarting train for epoch %s" % epoch)
    losses = []
    preds = []
    labels = []
    for idx, (mfcc, label) in enumerate(train_loader):
        mfcc, label = mfcc.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(mfcc)
        loss = loss_fn(torch.flatten(output), label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pred = torch.sigmoid(output)
        preds += torch.flatten(torch.round(pred)).cpu()
        labels += torch.flatten(label).cpu()
        print("Epoch: {}, Iter: {}/{}, loss: {}".format(epoch, idx, len(train_loader), loss.item()), end="\r")
    avg_train_loss = sum(losses) / len(losses)
    acc = binary_accuracy(torch.Tensor(preds), torch.Tensor(labels))
    print('Average train loss:', avg_train_loss, "Average train acc:", acc)
    report = classification_report(torch.Tensor(labels).numpy(), torch.Tensor(preds).numpy())
    print(report)
    return acc, report

def main(sample_rate, epochs, batch_size, eval_batch_size, lr, model_name, save_checkpoint_path, train_data_json, test_data_json, no_cuda, num_workers, hidden_size):
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_dataset = WakeWordData(data_json=train_data_json, sample_rate=sample_rate, valid=False)
    test_dataset = WakeWordData(data_json=test_data_json, sample_rate=sample_rate, valid=True)

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn=collate_fn,
                                   **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=eval_batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  **kwargs)

    model_params = {
        "num_classes": 1, "feature_size": 40, "hidden_size": hidden_size,
        "num_layers": 1, "dropout": 0.1, "bidirectional": False
    }
    model = LSTMWakeWord(**model_params, device=device)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_train_acc, best_train_report = 0, None
    best_test_acc, best_test_report = 0, None
    best_epoch = 0
    for epoch in range(epochs):
        print("\nStarting training with learning rate", optimizer.param_groups[0]['lr'])
        train_acc, train_report = train(train_loader, model, optimizer, loss_fn, device, epoch)
        test_acc, test_report = test(test_loader, model, device, epoch)

        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if save_checkpoint_path and test_acc >= best_test_acc:
            checkpoint_path = os.path.join(save_checkpoint_path, model_name + ".pt")
            print("Found best checkpoint. Saving model as", checkpoint_path)
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, model_params,
                notes="train_acc: {}, test_acc: {}, epoch: {}".format(best_train_acc, best_test_acc, epoch),
            )
            best_train_report = train_report
            best_test_report = test_report
            best_epoch = epoch

        table = [["Train ACC", train_acc], ["Test ACC", test_acc],
                 ["Best Train ACC", best_train_acc], ["Best Test ACC", best_test_acc],
                 ["Best Epoch", best_epoch]]
        print(tabulate(table))

        scheduler.step(train_acc)

    print("Done Training...")
    print("Best Model Saved to", checkpoint_path)
    print("Best Epoch", best_epoch)
    print("\nTrain Report \n")
    print(best_train_report)
    print("\nTest Report\n")
    print(best_test_report)

if __name__ == "__main__":
    # Input values directly
    sample_rate = 8000
    epochs = 100
    batch_size = 32
    eval_batch_size = 32
    lr = 1e-4
    model_name = "wakeword"
    save_checkpoint_path = "final_model"
    train_data_json = "VoiceAssistant/jsons/train.json"
    test_data_json = "VoiceAssistant/jsons/test.json"
    no_cuda = True
    num_workers = 1
    hidden_size = 128

    main(sample_rate, epochs, batch_size, eval_batch_size, lr, model_name, save_checkpoint_path, train_data_json, test_data_json, no_cuda, num_workers, hidden_size)
