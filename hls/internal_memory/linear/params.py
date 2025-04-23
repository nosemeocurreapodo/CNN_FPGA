data_type_list = ["ap_fixed<2,1>",
                  "ap_fixed<4,2>",
                  "ap_fixed<8,4>"]
batch_size_list = [1]
linear_param_list = [{"in_size": 2, "out_size": 2},
                     {"in_size": 4, "out_size": 4},
                     {"in_size": 8, "out_size": 8},
                     {"in_size": 16, "out_size": 16},
                     {"in_size": 32, "out_size": 32},]
clock_list = ["10"]
part_list = ['xc7z020clg400-1']

data_type_dict = {"ap_fixed<1,1>": 1,
                  "ap_fixed<2,1>": 2,
                  "ap_fixed<4,2>": 4,
                  "ap_fixed<8,4>": 8}
