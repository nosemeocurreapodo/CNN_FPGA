import torch
import torch.nn.functional as F


def layer_latency_predictor(device, mixed_op_param):
    """
    A toy differentiable latency predictor.

    Suppose we measured (offline) average latencies (ms) for each op:
      Conv3x3 = 0.8 ms
      Conv5x5 = 1.2 ms
      Skip    = 0.1 ms

    We'll combine them using the softmax weights of alpha.
    """
    # Hardcoded latencies for the example
    op_latencies = torch.tensor([0.8, 1.2, 0.1]).to(device)  # shape [num_ops]

    weights = F.softmax(mixed_op_param, dim=0)  # shape [num_ops]
    predicted_latency = torch.sum(weights * op_latencies)
    return predicted_latency


def act_latency_predictor(device, mixed_op_param):
    """
    A toy differentiable latency predictor.

    Suppose we measured (offline) average latencies (ms) for each op:
      Conv3x3 = 0.8 ms
      Conv5x5 = 1.2 ms
      Skip    = 0.1 ms

    We'll combine them using the softmax weights of alpha.
    """
    # Hardcoded latencies for the example
    op_latencies = torch.tensor([0.8, 1.2, 0.1]).to(device)  # shape [num_ops]

    weights = F.softmax(mixed_op_param, dim=0)  # shape [num_ops]
    predicted_latency = torch.sum(weights * op_latencies)
    return predicted_latency