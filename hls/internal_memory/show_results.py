import os

from params import data_type_list, batch_size_list, conv_param_list, clock_list, part_list, data_type_dict
from parse_hls_xml import parse_xml_reports

cwd = os.getcwd()+'/'
workspace_path = cwd + "/vitis_workspace/"


for part in part_list:
    for clock in clock_list:
        for w_data_type_name in data_type_list:
            for w_data_type_name in data_type_list:
                for a_data_type_name in data_type_list:
                    for batch_size in batch_size_list:
                        for conv_param in conv_param_list:

                            w_data_type = data_type_dict[w_data_type_name]
                            a_data_type = data_type_dict[a_data_type_name]

                            in_channels = conv_param["in_channels"]
                            out_channels = conv_param["out_channels"]
                            in_height = conv_param["in_height"]
                            in_width = conv_param["in_width"]
                            padding = conv_param["padding"]
                            use_relu = 1

                            component_name = (f"conv2d3x3_p{clock}_"
                                            f"W{w_data_type}_"
                                            f"A{a_data_type}_"
                                            f"BATCH{batch_size}_"
                                            f"{in_channels}x{out_channels}x{in_height}x{in_width}_"
                                            f"PAD{padding}_"
                                            f"RELU{use_relu}")

                            component_path = workspace_path + component_name + "/"

                            ip_path = component_path + component_name
                            data = parse_xml_reports(ip_path)
                            data["part"] = part
                            data["period"] = clock
                            data["weight_bits"] = w_data_type
                            data["act_bits"] = a_data_type
                            data["kernel_size"] = 3
                            data["in_channels"] = in_channels
                            data["out_channels"] = out_channels
                            data["in_height"] = in_height
                            data["in_width"] = in_width
                            data["padding"] = padding
                            print(data)
