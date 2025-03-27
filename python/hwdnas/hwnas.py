import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import layers as layers
import network as network
import load_dataset as load_dataset
import train as train
import hardware as hardware


def main(iterations=50):

    batch_size = 32

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using divice ", device)

    # dataset = load_dataset.mnist_dataset(batch_size)
    dataset = load_dataset.cifar10_dataset(batch_size)
    train_data_loader = dataset.train_loader
    test_data_loader = dataset.test_loader

    # Create a tiny DARTS network
    # model = network.TinyDARTSNetwork(num_classes=len(dataset.classes),
    #                                 input_shape=dataset.input_size)
    # model = network.VerySimpleModel(num_classes=len(dataset.classes),
    #                                input_shape=dataset.input_size)
    model = network.VerySimpleDARTSNetwork(num_classes=len(dataset.classes),
                                           input_shape=dataset.input_size)

    model = model.to(device)

    # Separate optimizers:
    #  - One for model weights
    #  - One for architecture params (alphas)
    # weight_params = [p for n, p in model.named_parameters() if 'alpha' not in n]
    # arch_params = [p for n, p in model.named_parameters() if 'alpha' in n]
    # arch_params = model.arch_parameters  # i.e., [mixed_op.alpha]

    MixedOpLayersList = []
    for name, layer in model.named_modules():
        if isinstance(layer, layers.MixedConv2D):
            MixedOpLayersList.append(layer)

    # optimizer_weights = optim.Adam(weight_params, lr=0.001)
    # optimizer_weights = optim.SGD(weight_params, lr=0.01, momentum=0.9)
    # if (len(arch_params) > 0):
    #    optimizer_alpha = optim.Adam(arch_params, lr=0.001)

    all_params = model.parameters()
    optimizer_all = optim.Adam(all_params, lr=0.001)

    criterion = nn.CrossEntropyLoss()

    # Run a few steps of training
    for step in range(iterations):
        """
        train_loss_total, train_accuracy = train.train_weights_one_step(
                model,
                device,
                train_data_loader,
                optimizer_weights,
                criterion,
                train=True
            )

        if (len(arch_params) > 0):
            train_loss_total, train_loss_cls, train_loss_latency, \
                train_accuracy = train.train_network_one_step(
                        model,
                        device,
                        test_data_loader,
                        optimizer_alpha,
                        criterion,
                        latency_predictor=hardware.latency_predictor,
                        train=True
                    )

        test_loss_total, test_accuracy = train.train_weights_one_step(
                model,
                device,
                test_data_loader,
                optimizer_weights,
                criterion,
                train=False
            )
        """

        if (len(MixedOpLayersList) == 0):
            train_loss_total, train_accuracy = train.train_weights_one_step(
                model,
                device,
                train_data_loader,
                optimizer_all,
                criterion,
                train=True
            )
            test_loss_total, test_accuracy = train.train_weights_one_step(
                model,
                device,
                test_data_loader,
                optimizer_all,
                criterion,
                train=False
            )
        else:
            train_loss_total, train_loss_cls, train_loss_latency, \
                train_accuracy = train.train_all_one_step(
                        model,
                        device,
                        train_data_loader,
                        optimizer_all,
                        criterion,
                        lambda_latency=1e-7,  # max(step*0.01, 0.1),
                        lambda_implementability=1.0,
                        train=True
                    )
            test_loss_total, test_loss_cls, test_loss_latency, \
                test_accuracy = train.train_all_one_step(
                        model,
                        device,
                        test_data_loader,
                        optimizer_all,
                        criterion,
                        lambda_latency=1e-7,  # max(step*0.01, 0.1),
                        lambda_implementability=1.0,
                        train=False
                    )

        if step % 1 == 0:
            print(f"Step {step:02d}")
            print(f"Train: |"
                  f"Loss: {train_loss_total:.4f} | "
                  f"Accuracy: {train_accuracy:.4f}")
            print(f"Test: |"
                  f"Loss: {test_loss_total:.4f} | "
                  f"Accuracy: {test_accuracy:.4f}")
            if (len(MixedOpLayersList) > 0):
                print(f"Cls loss: {train_loss_cls:.4f} |"
                      f"Lat loss: {train_loss_latency:.4f}")
                for index, layer in enumerate(MixedOpLayersList):
                    print(f"Alpha {index}: {layer.alpha.data}")
            print("hardware results: ", model.getHardwareResults())
            print("isImplementable: ", model.getImplementability())

    if (len(MixedOpLayersList) > 0):
        for index, layer in enumerate(MixedOpLayersList):
            # Inspect final architecture parameters
            # final_alpha = model.mixed_op_1.alpha.detach()
            # final_alpha_2 = model.mixed_op_2.alpha.detach()
            # final_alpha_3 = model.mixed_op_3.alpha.detach()
            # final_weights_1 = F.softmax(final_alpha_1, dim=0)
            # final_weights_2 = F.softmax(final_alpha_2, dim=0)
            # final_weights_3 = F.softmax(final_alpha_3, dim=0)
            final_alpha = layer.alpha.detach()
            final_weights = F.softmax(final_alpha, dim=0)
            print("\nAlpha ", index)
            print(f"Final architecture parameters: {final_alpha}")  #, " ", final_alpha_2, " ", final_alpha_3)
            print(f"Softmax over alpha (operation weights): {final_weights}")  #, " ", final_weights_2, " ", final_weights_3)


if __name__ == "__main__":
    main(200)
