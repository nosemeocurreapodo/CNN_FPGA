data_type_list = ["ap_fixed<2,1>",
                  "ap_fixed<4,2>",
                  "ap_fixed<8,4>"]
batch_size_list = [32, 64, 128, 256]
conv_param_list = [{"in_channels": 1, "out_channels": 32,
                    "in_height": 32, "in_width": 32,
                    "padding": 1},
                   {"in_channels": 32, "out_channels": 64,
                    "in_height": 16, "in_width": 16,
                    "padding": 1},
                   {"in_channels": 64, "out_channels": 128,
                    "in_height": 8, "in_width": 8,
                    "padding": 1}]
clock_list = ["10", "20"]
part_list = ['xc7z020clg400-1']

data_type_dict = {"ap_fixed<1,1>": 1,
                  "ap_fixed<2,1>": 2,
                  "ap_fixed<4,2>": 4,
                  "ap_fixed<8,4>": 8}
