import vitis
import os

from data_type_dict import data_type_dict

part = 'xc7z020clg400-1'
clock = "10"
batch_size = 1
layers_list = [{"w_data_type": "Posit<16,2>",
                "in_data_type": "Posit<16,2>",
                "out_data_type": "Posit<16,2>",
                "in_channels": 1, "out_channels": 32,
                "in_height": 32, "in_width": 32,
                "padding": 1,
                "use_relu": 1},
               {"w_data_type": "Posit<16,2>",
                "in_data_type": "Posit<16,2>",
                "out_data_type": "Posit<16,2>",
                "in_channels": 32, "out_channels": 32,
                "in_height": 32, "in_width": 32,
                "padding": 1,
                "use_relu": 1},]

cwd = os.getcwd()+'/'
workspace_path = cwd + "/vitis_workspace/"

# Initialize session
client = vitis.create_client()
client.set_workspace(path=workspace_path)

for layer in layers_list:

    w_data_type_name = layer["w_data_type"]
    in_data_type_name = layer["in_data_type"]
    out_data_type_name = layer["out_data_type"]
    in_channels = layer["in_channels"]
    out_channels = layer["out_channels"]
    in_height = layer["in_height"]
    in_width = layer["in_width"]
    padding = layer["padding"]
    use_relu = layer["use_relu"]

    w_data_type = data_type_dict[w_data_type_name]
    in_data_type = data_type_dict[in_data_type_name]
    out_data_type = data_type_dict[out_data_type_name]

    component_name = (f"Conv2d3x3_"
                      f"p{clock}_"
                      f"W{w_data_type}_"
                      f"I{in_data_type}_"
                      f"O{out_data_type}_"
                      f"BATCH{batch_size}_"
                      f"{in_channels}x{out_channels}x{in_height}x{in_width}_"
                      f"PAD{padding}_"
                      f"RELU{use_relu}")

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
                        values=[cwd+'conv2d_3x3/conv2d_3x3_generic_name.cpp',
                                cwd+'conv2d_3x3/conv2d_3x3_base.h',
                                cwd+'conv2d_3x3/conv2d_3x3_params.h',
                                cwd+'conv2d_3x3/conv2d_3x3_weights.h'])
    cfg_file.set_value(section='hls', key='syn.file_cflags',
                       value=(cwd+f"conv2d_3x3/conv2d_3x3_generic_name.cpp, "
                                  f"-DTOP_NAME={component_name} "
                                  f"-DUSE_RELU={use_relu} "
                                  f"-DW_DATA_TYPE={w_data_type} "
                                  f"-DB_DATA_TYPE={w_data_type} "
                                  f"-DIN_DATA_TYPE={in_data_type} "
                                  f"-DOUT_DATA_TYPE={out_data_type} "
                                  f"-DBATCH_SIZE={batch_size} "
                                  f"-DIN_CHANNELS={in_channels} "
                                  f"-DOUT_CHANNELS={out_channels} "
                                  f"-DIN_HEIGHT={in_height} "
                                  f"-DIN_WIDTH={in_width} "
                                  f"-DPADDING={padding} "
                                  f"-I{cwd}../../HLSLinearAlgebra/src"))
    cfg_file.set_values(section='hls', key='tb.file',
                        values=[cwd+'conv2d_3x3/conv2d_3x3_test.cpp'])
    cfg_file.set_value(section='hls', key='tb.file_cflags',
                       value=(cwd+f"conv2d_3x3/conv2d_3x3_test.cpp, "
                                  f"-DUSE_RELU={use_relu} "
                                  f"-DW_DATA_TYPE={w_data_type} "
                                  f"-DB_DATA_TYPE={w_data_type} "
                                  f"-DIN_DATA_TYPE={in_data_type} "
                                  f"-DOUT_DATA_TYPE={out_data_type} "
                                  f"-DBATCH_SIZE={batch_size} "
                                  f"-DIN_CHANNELS={in_channels} "
                                  f"-DOUT_CHANNELS={out_channels} "
                                  f"-DIN_HEIGHT={in_height} "
                                  f"-DIN_WIDTH={in_width} "
                                  f"-DPADDING={padding} "
                                  f"-I{cwd}../../HLSLinearAlgebra/src "
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
