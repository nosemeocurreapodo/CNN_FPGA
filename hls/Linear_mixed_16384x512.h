#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"

#include "Linear_base.h"

typedef ap_fixed<4, 1> weight_data_type;
typedef ap_fixed<4, 1> bias_data_type;
typedef ap_fixed<4, 1> input_data_type;
typedef ap_fixed<4, 1> output_data_type;

typedef hls::axis<weight_data_type, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> weight_packet_type;
typedef hls::axis<bias_data_type, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> bias_packet_type;
typedef hls::axis<input_data_type, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> input_packet_type;
typedef hls::axis<output_data_type, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> output_packet_type;

extern int Linear_mixed_16384x512(hls::stream<weight_packet_type> &weights_s, hls::stream<bias_packet_type> &bias_s, hls::stream<input_packet_type> &input_s, hls::stream<output_packet_type> &output_s);
