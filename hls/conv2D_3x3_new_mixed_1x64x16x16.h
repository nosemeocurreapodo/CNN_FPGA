#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"
#include "types.h"
#include "conv2D_3x3_new_base.h"

typedef ap_fixed<4, 1> weight_data_type;
typedef ap_fixed<4, 1> input_data_type;
typedef ap_fixed<4, 1> output_data_type;

typedef hls::axis<weight_data_type, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> weight_packet_type;
typedef hls::axis<input_data_type, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> input_packet_type;
typedef hls::axis<output_data_type, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false> output_packet_type;

extern int conv2D_3x3_new_mixed_1x64x16x16(hls::stream<input_packet_type> &input, hls::stream<output_packet_type> &output, hls::stream<weight_packet_type> &weights);
