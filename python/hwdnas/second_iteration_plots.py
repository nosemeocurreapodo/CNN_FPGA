import numpy as np
import matplotlib.pyplot as plt

#Step 199
#Train: |Loss: 0.4088 | Accuracy: 0.9565
#Test: |Loss: 1.2039 | Accuracy: 0.8078
#Cls loss: 0.1231 |Lat loss: 2857344.5536
#Alpha 0: tensor([-2.9695, -1.1371, -3.0107, -2.6385, -0.8047,  1.7255, -1.5931,  0.7069,
#         1.2491], device='cuda:0')
#Weights 0:tensor([0.0042, 0.0261, 0.0040, 0.0058, 0.0364, 0.4575, 0.0166, 0.1652, 0.2841],
#       device='cuda:0')
#Alpha 1: tensor([-1.3092, -3.8584,  0.3536, -1.7044,  1.0154, -0.3627, -0.3421,  0.9695,
#        -1.1710], device='cuda:0')
#Weights 1:tensor([0.0300, 0.0023, 0.1581, 0.0202, 0.3064, 0.0772, 0.0788, 0.2926, 0.0344],
#       device='cuda:0')
#Alpha 2: tensor([-1.0670, -2.9765,  0.4982, -0.9545,  1.0866,  0.4797, -0.8200,  0.2590,
#         1.1864], device='cuda:0')
#Weights 2:tensor([0.0286, 0.0042, 0.1370, 0.0320, 0.2467, 0.1344, 0.0367, 0.1078, 0.2726],
#       device='cuda:0')
#Alpha 3: tensor([-1.5356,  1.0744, -0.8110, -0.5805,  1.4650, -2.7208, -0.3723,  0.8188,
#        -2.4298], device='cuda:0')
#Weights 3:tensor([0.0186, 0.2528, 0.0384, 0.0483, 0.3735, 0.0057, 0.0595, 0.1957, 0.0076],
#       device='cuda:0')
#Alpha 4: tensor([-1.0214, -0.3125, -0.6304, -0.2392,  0.3289,  0.2619,  0.0464,  0.6846,
#         0.7703], device='cuda:0')
#Weights 4:tensor([0.0350, 0.0711, 0.0517, 0.0765, 0.1350, 0.1263, 0.1018, 0.1927, 0.2099],
#       device='cuda:0')
#Alpha 5: tensor([-1.2495, -1.1481,  5.5232, -0.9244, -0.4293, -0.5423, -0.8479, -0.3901,
#        -0.3610], device='cuda:0')
#Weights 5:tensor([0.0011, 0.0012, 0.9841, 0.0016, 0.0026, 0.0023, 0.0017, 0.0027, 0.0027],
#       device='cuda:0')
#hardware results:  tensor([2.8573e+06, 2.3313e+01, 0.0000e+00, 6.0027e+03, 6.4696e+03, 2.3340e+03,
#        0.0000e+00, 0.0000e+00], device='cuda:0', grad_fn=<AddBackward0>)
#isImplementable:  tensor(0., device='cuda:0', grad_fn=<MaxBackward1>)


# Define the custom x-axis labels
x_labels = ["w2_a2", "w2_a4", "w2_a8", "w4_a2", "w4_a4", "w4_a8", "w8_a2", "w8_a4", "w8_a8"]

# Define the data for each pair (converted from PyTorch tensors to numpy arrays)
alpha0 = np.array([-2.7962, -1.0707, -2.6068, -2.4500, -0.6418,  1.7040, -1.5464,  0.7042,1.2017])
weight0 = np.array([0.0050, 0.0282, 0.0061, 0.0071, 0.0433, 0.4524, 0.0175, 0.1665, 0.2738])

alpha1 = np.array([-1.2598, -3.6659,  0.2873, -1.5749,  1.0096, -0.2890, -0.3450,  0.9680, -1.0181])
weight1 = np.array([0.0314, 0.0028, 0.1474, 0.0229, 0.3034, 0.0828, 0.0783, 0.2911, 0.0399])

alpha2 = np.array([-0.9871, -2.4805,  0.4321, -0.9048,  1.1058,  0.4855, -0.7818,  0.3017, 1.1261])
weight2 = np.array([0.0312, 0.0070, 0.1290, 0.0339, 0.2530, 0.1361, 0.0383, 0.1132, 0.2582])

alpha3 = np.array([-1.4160,  1.0428, -0.6917, -0.5862,  1.4260, -2.3974, -0.3866,  0.8090, -2.1445])
weight3 = np.array([0.0212, 0.2482, 0.0438, 0.0487, 0.3641, 0.0080, 0.0594, 0.1964, 0.0102])

alpha4 = np.array([-0.9779, -0.3195, -0.5466, -0.2449,  0.3413,  0.2649,  0.0342,  0.6781, 0.7390])
weight4 = np.array([0.0366, 0.0707, 0.0564, 0.0762, 0.1369, 0.1269, 0.1007, 0.1918, 0.2038])

alpha5 = np.array([-1.1784, -0.9063,  5.0057, -0.8363, -0.3286, -0.4518, -0.7429, -0.3213, -0.2925])
weight5 = np.array([0.0020, 0.0026, 0.9711, 0.0028, 0.0047, 0.0041, 0.0031, 0.0047, 0.0049])

# Group pairs for iteration
pairs = [(alpha0, weight0), (alpha1, weight1), (alpha2, weight2),
         (alpha3, weight3), (alpha4, weight4), (alpha5, weight5)]

# Create a 2x3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

for i, (alpha, weight) in enumerate(pairs):
    ax = axs[i // 3, i % 3]
    indices = np.arange(len(alpha))
    
    # Plot alpha values on the primary y-axis
    ln1 = ax.plot(indices, alpha, marker='o', color='blue', label=f'Alpha {i}')[0]
    ax.set_xlabel('Measurement')
    ax.set_ylabel('Alpha', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    
    # Replace x-axis ticks with custom labels
    ax.set_xticks(indices)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Plot weight values on a secondary y-axis
    ax2 = ax.twinx()
    ln2 = ax2.plot(indices, weight, marker='x', color='red', label=f'Weight {i}')[0]
    ax2.set_ylabel('Weight', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax.set_title(f'Mixed layer {i}')
    
    # Combine legends from both y-axes
    lns = [ln1, ln2]
    labels = [ln.get_label() for ln in lns]
    ax.legend(lns, labels, loc='upper right')

plt.tight_layout()
#plt.show()
plt.savefig("second_iteration_alpha_weights.png")
print("Plot saved as my_plot.png")
