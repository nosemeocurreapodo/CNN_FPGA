import vitis
import os

from data_type_dict import data_type_dict

data_type_list = ["ap_fixed<2,1>",
                  "ap_fixed<4,2>",
                  "ap_fixed<8,4>"]
batch_size_list = [1]
conv_param_list = [{"in_channels": 1, "out_channels": 32,
                    "in_height": 32, "in_width": 32,
                    "padding": 1,
                    "use_relu": 1},
                   {"in_channels": 32, "out_channels": 32,
                    "in_height": 32, "in_width": 32,
                    "padding": 1,
                    "use_relu": 1},
                   {"in_channels": 32, "out_channels": 64,
                    "in_height": 16, "in_width": 16,
                    "padding": 1,
                    "use_relu": 1},
                   {"in_channels": 64, "out_channels": 64,
                    "in_height": 16, "in_width": 16,
                    "padding": 1,
                    "use_relu": 1},
                   {"in_channels": 64, "out_channels": 128,
                    "in_height": 8, "in_width": 8,
                    "padding": 1,
                    "use_relu": 1},
                   {"in_channels": 128, "out_channels": 128,
                    "in_height": 8, "in_width": 8,
                    "padding": 1,
                    "use_relu": 1}]
clock_list = ["10"]
part_list = ['xc7z020clg400-1']

cwd = os.getcwd()+'/'
workspace_path = cwd + "/vitis_workspace/"

# Initialize session
client = vitis.create_client()
client.set_workspace(path=workspace_path)

for part in part_list:
    for clock in clock_list:
        for batch_size in batch_size_list:
            for w_data_type_name in data_type_list:
                for in_data_type_name in data_type_list:
                    for out_data_type_name in data_type_list:
                        for conv_param in conv_param_list:

                            w_data_type = data_type_dict[w_data_type_name]
                            in_data_type = data_type_dict[in_data_type_name]
                            out_data_type = data_type_dict[out_data_type_name]

                            in_channels = conv_param["in_channels"]
                            out_channels = conv_param["out_channels"]
                            in_height = conv_param["in_height"]
                            in_width = conv_param["in_width"]
                            padding = conv_param["padding"]
                            use_relu = conv_param["use_relu"]

                            component_name = (f"Conv2D3x3_p{clock}_"
                                              f"W{w_data_type}_"
                                              f"I{in_data_type}_"
                                              f"O{out_data_type}_"
                                              f"BATCH{batch_size}_"
                                              f"{in_channels}x{out_channels}x{in_height}x{in_width}_"
                                              f"PAD{padding}")

                            component_path = workspace_path + component_name + "/"

                            # Delete the component if it already exists
                            if os.path.exists(component_path):
                                # client.delete_component(name=component_name)
                                continue

                            # Create component.
                            # Create new config file in the component folder
                            # of the workspace
                            comp = client.create_hls_component(name=component_name,
                                                            cfg_file=['hls_config.cfg'],
                                                            template='empty_hls_component')

                            # Get handle of config file, then programmatically set desired options
                            cfg_file = client.get_config_file(path=component_path + "hls_config.cfg")
                            cfg_file.set_value(key='part', value=part) 
                            cfg_file.set_values(section='hls', key='syn.file',
                                                values=[cwd+'conv2D_3x3_IM.cpp',
                                                        cwd+'conv2D_3x3_IM.h',
                                                        cwd+'conv2D_3x3_IM_base.h',
                                                        cwd+'conv2D_3x3_IM_params.h',
                                                        cwd+'conv2D_3x3_IM_weights.h'])
                            cfg_file.set_value(section='hls', key='syn.file_cflags',
                                            value=cwd+(f"conv2D_3x3_IM.cpp, "
                                                        f"-DTOP_NAME={component_name} "
                                                        f"-DUSE_RELU={use_relu} "
                                                        f"-DW_DATA_TYPE={w_data_type} "
                                                        f"-DIN_DATA_TYPE={in_data_type} "
                                                        f"-DOUT_DATA_TYPE={out_data_type} "
                                                        f"-DBATCH_SIZE={batch_size} "
                                                        f"-DIN_CHANNELS={in_channels} "
                                                        f"-DOUT_CHANNELS={out_channels} "
                                                        f"-DIN_HEIGHT={in_height} "
                                                        f"-DIN_WIDTH={in_width} "
                                                        f"-DPADDING={padding}"))
                            cfg_file.set_values(section='hls', key='tb.file',
                                                values=[cwd+'conv2D_3x3_IM_test.cpp'])
                            cfg_file.set_value(section='hls', key='tb.file_cflags',
                                            value=cwd+(f"conv2D_3x3_IM_test.cpp, "
                                                        f"-DTOP_NAME={component_name} "
                                                        f"-DUSE_RELU={use_relu} "
                                                        f"-DW_DATA_TYPE={w_data_type} "
                                                        f"-DIN_DATA_TYPE={in_data_type} "
                                                        f"-DOUT_DATA_TYPE={out_data_type} "
                                                        f"-DBATCH_SIZE={batch_size} "
                                                        f"-DIN_CHANNELS={in_channels} "
                                                        f"-DOUT_CHANNELS={out_channels} "
                                                        f"-DIN_HEIGHT={in_height} "
                                                        f"-DIN_WIDTH={in_width} "
                                                        f"-DPADDING={padding} "
                                                        "-I/usr/include/opencv4 "
                                                        "-lopencv_core "
                                                        "-lopencv_highgui "
                                                        "-lopencv_imgcodecs"))
                            cfg_file.set_value(section='hls', key='syn.top',
                                            value=component_name)
                            cfg_file.set_value(section='hls', key='clock', value=clock)
                            cfg_file.set_value(section='hls', key='flow_target',
                                            value='vivado')
                            # cfg_file.set_value (section = 'hls',
                            # key = 'package.output.syn',    value = '0')
                            # cfg_file.set_value (section = 'hls',
                            # key = 'package.output.format', value = 'rtl')
                            # cfg_file.set_value (section = 'hls',
                            # key = 'package.output.format', value = 'ip_catalog')
                            # cfg_file.set_value (section = 'hls',
                            # key = 'package.ip.display_name',   value = component_name)
                            # cfg_file.set_value (section = 'hls',
                            # key = 'package.ip.name',   value = component_name)
                            # cfg_file.set_value (section = 'hls',
                            # key = 'csim.code_analyzer',    value = '0')

                            # cfg_file.set_value (section = 'hls',
                            # key = 'syn.compile.pipeline_style',    value = 'frp')
                            # cfg_file.set_value (section = 'hls',
                            # key = 'syn.dataflow.default_channel',    value = 'fifo')
                            # cfg_file.set_value (section = 'hls',
                            # key = 'syn.dataflow.fifo_depth',    value = '16')
                            # cfg_file.set_value (section = 'hls',
                            # key = 'syn.directive.interface',    value = 'free_pipe_mult mode=ap_fifo B')
                            # cfg_file.set_value (section = 'hls',
                            # key = 'syn.directive.interface',    value = 'free_pipe_mult mode=ap_fifo out')

                            # Run flow steps
                            comp = client.get_component(name=component_name)
                            # comp.run(operation='C_SIMULATION')
                            comp.run(operation='SYNTHESIS')
                            # comp.run(operation='CO_SIMULATION')
                            comp.run(operation='IMPLEMENTATION')
