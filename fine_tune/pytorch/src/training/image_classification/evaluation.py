import torch


def total_loss(model, loss_fn, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    total_loss = 0
    for inputs, labels in loader:
        outputs = model(inputs.cuda())
        total_loss += loss_fn(outputs, labels.cuda()).item()
    return total_loss / len(dataset)


def accuracy(model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    total_correct = 0
    for inputs, labels in loader:
        outputs = model(inputs.cuda())
        total_correct += (outputs.argmax(dim=1) == labels.cuda()).sum().item()
    return total_correct / len(dataset)