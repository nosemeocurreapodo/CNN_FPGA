import vitis
import os

from parse_hls_xml import parse_hls_xml

cwd = os.getcwd()+'/'
workspace_path = cwd + "/vitis_workspace/"

data_type_list = ["float", "half",
                  "ap_fixed<32,16>", "ap_fixed<16,8>",
                  "ap_fixed<8,4>", "ap_fixed<4,2>"]
conv_param_list = [{"in_channels": 1, "out_channels": 32,
                    "in_height": 32, "in_width": 32,
                    "padding": 1},
                   {"in_channels": 32, "out_channels": 32,
                    "in_height": 32, "in_width": 32,
                    "padding": 1},
                   {"in_channels": 32, "out_channels": 64,
                    "in_height": 16, "in_width": 16,
                    "padding": 1},
                   {"in_channels": 64, "out_channels": 64,
                    "in_height": 16, "in_width": 16,
                    "padding": 1},
                   {"in_channels": 64, "out_channels": 128,
                    "in_height": 8, "in_width": 8,
                    "padding": 1},
                   {"in_channels": 128, "out_channels": 128,
                    "in_height": 8, "in_width": 8,
                    "padding": 1}]
clock_list = ["10", "20", "30", "40", "50"]
part_list = ['xc7z020clg400-1']

data_type_dict = {"float": 0,
                  "half": 1,
                  "ap_fixed<32,16>": 2,
                  "ap_fixed<16,8>": 3,
                  "ap_fixed<8,4>": 4,
                  "ap_fixed<4,2>": 5}

for part in part_list:
    for clock in clock_list:
        for data_type_name in data_type_list:
            for conv_param in conv_param_list:

                data_type = data_type_dict[data_type_name]

                in_channels = conv_param["in_channels"]
                out_channels = conv_param["out_channels"]
                in_height = conv_param["in_height"]
                in_width = conv_param["in_width"]
                padding = conv_param["padding"]
                use_relu = 1

                component_name = f"conv2D_3x3_IM_period_{clock}_dtype_{data_type}_shape_{in_channels}x{out_channels}x{in_height}x{in_width}_padding_{padding}"

                component_path = workspace_path + component_name + "/"

                # Initialize session
                client = vitis.create_client()
                client.set_workspace(path=workspace_path)

                # Delete the component if it already exists
                if os.path.exists(component_path):
                    client.delete_component(name=component_name)

                # Create component. Create new config file in the component folder of the workspace
                comp = client.create_hls_component(name=component_name, cfg_file = ['hls_config.cfg'], template = 'empty_hls_component')

                # Get handle of config file, then programmatically set desired options
                cfg_file = client.get_config_file(path = component_path + "hls_config.cfg")
                cfg_file.set_value (                 key = 'part',                  value = part) 
                cfg_file.set_values(section = 'hls', key = 'syn.file',              values = [cwd+'conv2D_3x3_IM.cpp', cwd+'conv2D_3x3_IM.h', cwd+'conv2D_3x3_IM_base.h', cwd+'conv2D_3x3_IM_params.h', cwd+'conv2D_3x3_IM_weights.h'])
                cfg_file.set_value (section = 'hls', key = 'syn.file_cflags',       value = cwd+f'conv2D_3x3_IM.cpp, -DTOP_NAME={component_name} -DUSE_RELU={use_relu} -DDATA_TYPE={data_type} -DIN_CHANNELS={in_channels} -DOUT_CHANNELS={out_channels} -DIN_HEIGHT={in_height} -DIN_WIDTH={in_width} -DPADDING={padding}')
                cfg_file.set_values(section = 'hls', key = 'tb.file',               values = [cwd+'conv2D_3x3_IM_test.cpp'])
                cfg_file.set_value (section = 'hls', key = 'tb.file_cflags',        value  = cwd+f'conv2D_3x3_IM_test.cpp, -DTOP_NAME={component_name} -DUSE_RELU={use_relu} -DDATA_TYPE={data_type} -DIN_CHANNELS={in_channels} -DOUT_CHANNELS={out_channels} -DIN_HEIGHT={in_height} -DIN_WIDTH={in_width} -DPADDING={padding} -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs')
                cfg_file.set_value (section = 'hls', key = 'syn.top',               value = component_name)
                cfg_file.set_value (section = 'hls', key = 'clock',                 value = clock) # 250MHz
                cfg_file.set_value (section = 'hls', key = 'flow_target',           value = 'vivado')
                # cfg_file.set_value (section = 'hls', key = 'package.output.syn',    value = '0')
                # cfg_file.set_value (section = 'hls', key = 'package.output.format', value = 'rtl')
                # cfg_file.set_value (section = 'hls', key = 'package.output.format', value = 'ip_catalog')
                # cfg_file.set_value (section = 'hls', key = 'package.ip.display_name',   value = component_name)
                # cfg_file.set_value (section = 'hls', key = 'package.ip.name',   value = component_name)
                # cfg_file.set_value (section = 'hls', key = 'csim.code_analyzer',    value = '0')

                # cfg_file.set_value (section = 'hls', key = 'syn.compile.pipeline_style',    value = 'frp')
                # cfg_file.set_value (section = 'hls', key = 'syn.dataflow.default_channel',    value = 'fifo')
                # cfg_file.set_value (section = 'hls', key = 'syn.dataflow.fifo_depth',    value = '16')
                # cfg_file.set_value (section = 'hls', key = 'syn.directive.interface',    value = 'free_pipe_mult mode=ap_fifo B')
                # cfg_file.set_value (section = 'hls', key = 'syn.directive.interface',    value = 'free_pipe_mult mode=ap_fifo out')

                # Run flow steps
                comp = client.get_component(name=component_name)
                # comp.run(operation='C_SIMULATION')
                comp.run(operation='SYNTHESIS')
                # comp.run(operation='CO_SIMULATION')

for part in part_list:
    for clock in clock_list:
        for data_type_name in data_type_list:
            for conv_param in conv_param_list:

                data_type = data_type_dict[data_type_name]

                in_channels = conv_param["in_channels"]
                out_channels = conv_param["out_channels"]
                in_height = conv_param["in_height"]
                in_width = conv_param["in_width"]
                padding = conv_param["padding"]
                use_relu = 1

                component_name = f"conv2D_3x3_IM_period_{clock}_dtype_{data_type}_shape_{in_channels}x{out_channels}x{in_height}x{in_width}_padding_{padding}"

                component_path = workspace_path + component_name + "/"

                xml_path = component_path + component_name + \
                    "/hls/syn/report/" + component_name + "_csynth.xml"
                data = parse_hls_xml(xml_path)
                print(data)
