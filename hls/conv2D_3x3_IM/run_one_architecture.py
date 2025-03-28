import vitis
import os

from params import data_type_dict

part = 'xc7z020clg400-1'
clock = "10"
batch_size = 1
layers_list = [{"w_data_type": "ap_fixed<4,2>",
                "a_data_type": "ap_fixed<8,4>",
                "in_channels": 1, "out_channels": 32,
                "in_height": 32, "in_width": 32,
                "padding": 1},
               {"w_data_type": "ap_fixed<4,2>",
                "a_data_type": "ap_fixed<4,2>",
                "in_channels": 32, "out_channels": 32,
                "in_height": 32, "in_width": 32,
                "padding": 1},
               {"w_data_type": "ap_fixed<4,2>",
                "a_data_type": "ap_fixed<4,2>",
                "in_channels": 32, "out_channels": 64,
                "in_height": 16, "in_width": 16,
                "padding": 1},
               {"w_data_type": "ap_fixed<4,2>",
                "a_data_type": "ap_fixed<4,2>",
                "in_channels": 64, "out_channels": 64,
                "in_height": 16, "in_width": 16,
                "padding": 1},
               {"w_data_type": "ap_fixed<8,4>",
                "a_data_type": "ap_fixed<8,4>",
                "in_channels": 64, "out_channels": 128,
                "in_height": 8, "in_width": 8,
                "padding": 1},
               {"w_data_type": "ap_fixed<2,1>",
                "a_data_type": "ap_fixed<8,4>",
                "in_channels": 128, "out_channels": 128,
                "in_height": 8, "in_width": 8,
                "padding": 1}]

cwd = os.getcwd()+'/'
workspace_path = cwd + "/vitis_workspace/"

# Initialize session
client = vitis.create_client()
client.set_workspace(path=workspace_path)

for layer in layers_list:

    w_data_type_name = layer["w_data_type"]
    a_data_type_name = layer["a_data_type"]
    in_channels = layer["in_channels"]
    out_channels = layer["out_channels"]
    in_height = layer["in_height"]
    in_width = layer["in_width"]
    padding = layer["padding"]
    use_relu = 1

    w_data_type = data_type_dict[w_data_type_name]
    a_data_type = data_type_dict[a_data_type_name]

    component_name = (f"conv2D_3x3_IM_period_{clock}_"
                      f"W_{w_data_type}_"
                      f"A_{a_data_type}_"
                      f"batch_size_{batch_size}_"
                      f"shape_{in_channels}x{out_channels}x{in_height}x{in_width}_"
                      f"padding_{padding}")

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
                                  f"-DA_DATA_TYPE={a_data_type} "
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
                                  f"-DA_DATA_TYPE={a_data_type} "
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
