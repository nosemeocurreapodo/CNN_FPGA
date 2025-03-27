# ------------------------------------------------------------------
# File: run_hls.tcl
# ------------------------------------------------------------------

if {![info exists ::period]} {
    set ::period 15
}
puts "PERIOD set to $::period"

# 1. Open a new project
# open_project conv2d_project
open_component -reset conv2D_3x3_IM_component -flow_target vivado

# 2. Add your source files (and header if needed)
add_files ./conv2D_3x3_IM_params.h
add_files ./conv2D_3x3_IM_weights.h
add_files ./conv2D_3x3_IM_base.h
add_files ./conv2D_3x3_IM.cpp
add_files ./conv2D_3x3_IM.h
add_files ../common/floatX.h
add_files ../common/types.h
add_files -tb ./conv2D_3x3_IM_test.cpp -csimflags "-I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs"

# 3. Set top function
set_top conv2D_3x3_IM

# 4. Create the first solution
# open_solution -name solution1
#    - For example, pick a device part or set it from your board files
set_part {xc7z020clg400-1}      ; # for a Zynq-7000 SoC device
create_clock -period $::period -name default   ; # 100 MHz

# 5. Run C synthesis on solution1
csynth_design

exit