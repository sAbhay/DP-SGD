import torch
import copy


def train_sgd(model, train_loader, sgd, loss_function, epochs, loss_reduction='mean'):
    epoch_loss = ValueError()
    accuracy = ValueError()
    for e in range(epochs):
        epoch_loss = 0
        accuracy = 0
        for X_batch, y_batch in train_loader:
            sgd.zero_grad()
            y_pred = model(X_batch)
            accuracy += torch.sum(torch.round(y_pred) == torch.round(y_batch))
            loss = loss_function(y_pred, y_batch)
            if loss_reduction == 'mean':
                epoch_loss += loss.item() * X_batch.shape[0]
            else:
                epoch_loss += loss.item()
            loss.backward()
            sgd.step()
        accuracy = accuracy.item() / len(train_loader.dataset)
        if loss_reduction == 'mean':
            epoch_loss *= 1 / len(train_loader.dataset)
    return epoch_loss, accuracy


def train_dpsgd(model, train_loader, dpsgd, loss_function, epochs, loss_reduction='mean'):
    epoch_loss = ValueError()
    accuracy = ValueError()
    for e in range(epochs):
        epoch_loss = 0
        accuracy = 0
        for X_batch, y_batch in train_loader:
            dpsgd.zero_grad()
            for x, y in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_batch, y_batch)):
                y_pred = model(x)
                accuracy += int(torch.round(y_pred) == torch.round(y))
                loss = loss_function(y_pred, y)
                epoch_loss += loss.item()
                loss.backward()
                dpsgd.per_sample_gradient_clip()
            dpsgd.step()
        if loss_reduction == 'mean':
            epoch_loss /= len(train_loader.dataset)
        accuracy = accuracy / len(train_loader.dataset)
    return epoch_loss, accuracy


def evaluate(model, test_loader, loss_function, loss_reduction='mean'):
    test_loss = 0
    accuracy = 0
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        loss = loss_function(y_pred, y_batch)
        accuracy += torch.sum(torch.round(y_pred) == torch.round(y_batch))
        if loss_reduction == 'mean':
            test_loss += loss.item() * X_batch.shape[0]
        else:
            test_loss += loss.item()
    if loss_reduction == 'mean':
        test_loss /= len(test_loader.dataset)
    accuracy = accuracy.item() / len(test_loader.dataset)
    return test_loss, accuracy


def run_experiment(master_model, train, optimizer_func, optimizer_params, train_loader, test_loader, X, Y, loss_function, epochs, loss_reduction='mean'):
    model = copy.deepcopy(master_model)
    assert (loss_function(model(X), Y) - loss_function(master_model(X), Y) == 0)

    optimizer = optimizer_func(params=model.parameters(), **optimizer_params)
    train_loss, train_accuracy = train(model, train_loader, optimizer, loss_function, epochs)
    test_loss, test_accuracy = evaluate(model, test_loader, loss_function, loss_reduction=loss_reduction)

    return (train_loss, train_accuracy), (test_loss, test_accuracy)


def run_experiments(master_model, train_funcs, optimizers, train_loader, test_loader, X, Y, loss_functions, epochs):
    results = {}
    i = 0
    for optimizer_name, (optimizer_func, optimizer_params) in optimizers.items():
        for loss_name, loss_function in loss_functions.items():
            (train_loss, train_accuracy), (test_loss, test_accuracy) = run_experiment(master_model, train_funcs[optimizer_name], optimizer_func, optimizer_params, train_loader, test_loader, X, Y, loss_function, epochs)
            results[(optimizer_name, loss_name, epochs)] = {"train_loss": train_loss, "test_loss": test_loss, "train_accuracy": train_accuracy, "test_accuracy": test_accuracy}
            i += 1
            print(i, optimizer_name, loss_name)
    return results



