import vitis
import os

from params import data_type_list, batch_size_list, linear_param_list, clock_list, part_list, data_type_dict

cwd = os.getcwd()+'/'
workspace_path = cwd + "/vitis_workspace/"

# Initialize session
client = vitis.create_client()
client.set_workspace(path=workspace_path)

for part in part_list:
    for clock in clock_list:
        for batch_size in batch_size_list:
            for w_data_type_name in data_type_list:
                for b_data_type_name in data_type_list:
                    for in_data_type_name in data_type_list:
                        for out_data_type_name in data_type_list:
                            for linear_param in linear_param_list:

                                w_data_type = data_type_dict[w_data_type_name]
                                b_data_type = data_type_dict[b_data_type_name]
                                in_data_type = data_type_dict[in_data_type_name]
                                out_data_type = data_type_dict[out_data_type_name]

                                in_size = linear_param["in_size"]
                                out_size = linear_param["out_size"]
                                use_relu = 1

                                component_name = (f"Linear_"
                                                  f"period_{clock}_"
                                                  f"W{w_data_type}_"
                                                  f"B{b_data_type}_"
                                                  f"IN{in_data_type}_"
                                                  f"OUT{out_data_type}_"
                                                  f"batch_size_{batch_size}_"
                                                  f"shape_{in_size}x{out_size}_")

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
                                                            f"-DB_DATA_TYPE={b_data_type} "
                                                            f"-DIN_DATA_TYPE={in_data_type} "
                                                            f"-DOUT_DATA_TYPE={out_data_type} "
                                                            f"-DBATCH_SIZE={batch_size} "
                                                            f"-DIN_SIZE={in_size} "
                                                            f"-DOUT_SIZE={out_size} "))
                                cfg_file.set_values(section='hls', key='tb.file',
                                                    values=[cwd+'conv2D_3x3_IM_test.cpp'])
                                cfg_file.set_value(section='hls', key='tb.file_cflags',
                                                value=cwd+(f"conv2D_3x3_IM_test.cpp, "
                                                            f"-DTOP_NAME={component_name} "
                                                            f"-DUSE_RELU={use_relu} "
                                                            f"-DW_DATA_TYPE={w_data_type} "
                                                            f"-DB_DATA_TYPE={b_data_type} "
                                                            f"-DIN_DATA_TYPE={in_data_type} "
                                                            f"-DOUT_DATA_TYPE={out_data_type} "
                                                            f"-DBATCH_SIZE={batch_size} "
                                                            f"-DIN_SIZE={in_size} "
                                                            f"-DOUT_SIZE={out_size} "
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
