# ------------------------------------------------------------------------------
# File: chain_hls_cores.tcl
# ------------------------------------------------------------------------------
# Usage (from a terminal or script):
#   vivado -mode batch -source chain_hls_cores.tcl
#     (or run "source chain_hls_cores.tcl" from within the Vivado Tcl shell)
# ------------------------------------------------------------------------------

# 1) Create or open a Vivado project
create_project chain_cores ./chain_cores -part xc7z020clg400-1  ;# Example device
# If you already have a project, you can do "open_project my_existing_project"

# 2) Add directory paths where your HLS IPs are located
#    Typically each HLS IP is packaged as an XCI or directory with component.xml.
#    E.g., from "export_design" in Vitis HLS.
set ip_repo_1 "./my_hls_ip1"   ;# directory containing IP #1
set ip_repo_2 "./my_hls_ip2"   ;# directory containing IP #2
# Add more if needed...

# Add them to IP repository list
set_property ip_repo_paths [list $ip_repo_1 $ip_repo_2] [current_project]
update_ip_catalog

# 3) Create a new Block Design
create_bd_design "myBlockDesign"

# 4) Instantiate your HLS IP blocks
#    The 'create_bd_cell' command needs the exact 'VLNV' (vendor:library:name:version)
#    that matches how the IP was packaged. If uncertain, open "IP Catalog" in GUI
#    or examine the IP's component.xml.
create_bd_cell -type ip -vlnv hls.vendor:IP1:1.0 ip1_0
create_bd_cell -type ip -vlnv hls.vendor:IP2:1.0 ip2_0

# If you have more IPs, keep creating them similarly:
# create_bd_cell -type ip -vlnv hls.vendor:IP3:1.0 ip3_0
# ...

# 5) Connect the interfaces or signals
#    - The exact pin names and interfaces depend on your HLS IP settings
#    - For example, if each IP uses AXI Stream out -> AXI Stream in for chaining:

# Example: Connect M_AXIS from the first IP to S_AXIS of the second IP
connect_bd_intf_net \
  [get_bd_intf_pins ip1_0/M_AXIS] \
  [get_bd_intf_pins ip2_0/S_AXIS]

# Also, connect the clock and reset for each IP. Often, if you have a shared AXI-lite
# control interface or shared clock/reset, you'd do something like:
# create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ctrl_0
# create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0
# Then connect these signals to each IP's clock/reset pins.
#
# For simplicity, let's assume you have "ap_clk" and "ap_rst_n" in each IP:
connect_bd_net \
  [get_bd_pins clk_wiz_0/clk_out1] \
  [get_bd_pins ip1_0/ap_clk] \
  [get_bd_pins ip2_0/ap_clk]

connect_bd_net \
  [get_bd_pins rst_ctrl_0/peripheral_aresetn] \
  [get_bd_pins ip1_0/ap_rst_n] \
  [get_bd_pins ip2_0/ap_rst_n]

# If your HLS IP uses AXI-Lite for control, you also need to connect the M_AXI from a Zynq PS
# or a MicroBlaze to their S_AXI control interface. Example:
# connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] \
#                    [get_bd_intf_pins ip1_0/S_AXI_CONTROL]

# 6) Validate the Block Design
validate_bd_design [get_bd_design "myBlockDesign"]

# 7) (Optional) Generate output products for block design (DCP, HDL, etc.)
generate_target all [get_files [get_bd_designs myBlockDesign]]

# 8) Optionally create a top-level wrapper, synthesize, implement, and generate bitstream
# create_bd_wrapper [get_bd_design "myBlockDesign"] -fileset sources_1
# synth_design -top myBlockDesign_wrapper -part xc7z020clg400-1
# opt_design
# place_design
# route_design
# write_bitstream -force myBlockDesign.bit

# 9) Save and close if you want
save_project_as ./chain_cores/chain_cores.xpr
close_project
