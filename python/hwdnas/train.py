import torch


def train_weights_one_step(model,
                           device,
                           data_loader,
                           optimizer,
                           criterion,
                           train=True):

    if (train):
        model.train()
    else:
        model.eval()

    acc_loss = 0
    correct = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        data, labels = data.to(device), labels.to(device)

        logits = model(data)
        loss = criterion(logits, labels)

        if (train):

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc_loss += loss.item()
        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    mean_loss = acc_loss / len(data_loader)
    accuracy = correct / len(data_loader.dataset)

    return mean_loss, accuracy


def train_network_one_step(model,
                           device,
                           data_loader,
                           optimizer,
                           criterion,
                           latency_predictor,
                           lambda_latency=0.1,
                           train=True):

    if (train):
        model.train()
    else:
        model.eval()

    acc_loss = 0
    acc_loss_cls = 0
    acc_loss_latency = 0
    correct = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        data, labels = data.to(device), labels.to(device)
        # -- Forward pass --
        logits = model(data)
        loss_cls = criterion(logits, labels)

        #  (Hardware-aware) Latency penalty
        lat_pred = torch.tensor(0.0).to(device)
        for arch_param in model.arch_parameters:
            lat_pred += latency_predictor(device, arch_param)  # this is differentiable wrt alpha
        loss_latency = lat_pred

        # Combine them into a total loss
        loss = loss_cls + lambda_latency * loss_latency

        if (train):
            # -- Backward pass for weights and alpha (in real DARTS, these are done on separate splits) --
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        acc_loss += loss.item()
        acc_loss_cls += loss_cls.item()
        acc_loss_latency += loss_latency.item()
        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    mean_loss = acc_loss / len(data_loader)
    mean_loss_cls = acc_loss_cls / len(data_loader)
    mean_loss_latency = acc_loss_latency / len(data_loader)
    accuracy = correct / len(data_loader.dataset)

    return mean_loss, mean_loss_cls, mean_loss_latency, accuracy


def train_all_one_step(model,
                       device,
                       data_loader,
                       optimizer,
                       criterion,
                       lambda_latency=0.0,
                       lambda_slices=0.1,
                       lambda_implementability=100.0,
                       train=True):

    if (train):
        model.train()
    else:
        model.eval()

    acc_loss = 0
    acc_loss_cls = 0
    acc_loss_latency = 0
    correct = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        data, labels = data.to(device), labels.to(device)
        # -- Forward pass --
        logits = model(data)
        loss_cls = criterion(logits, labels)

        #  (Hardware-aware) Latency penalty
        loss_latency = model.getLatency()
        loss_slices = model.getSlices()
        loss_imp = model.getImplementability()

        # Combine them into a total loss
        loss = loss_cls + lambda_latency * loss_latency + lambda_slices * loss_slices + lambda_implementability * loss_imp

        if (train):
            # -- Backward pass for weights and alpha (in real DARTS, these are done on separate splits) --
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc_loss += loss.item()
        acc_loss_cls += loss_cls.item()
        acc_loss_latency += loss_latency.item()
        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    mean_loss = acc_loss / len(data_loader)
    mean_loss_cls = acc_loss_cls / len(data_loader)
    mean_loss_latency = acc_loss_latency / len(data_loader)
    accuracy = correct / len(data_loader.dataset)

    return mean_loss, mean_loss_cls, mean_loss_latency, accuracy
