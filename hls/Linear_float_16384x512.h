#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"

#include "Linear.h"

typedef ap_fixed<4, 1> weight_type;
typedef ap_fixed<4, 1> bias_type;
typedef ap_fixed<4, 1> input_type;
typedef ap_fixed<4, 1> output_type;

typedef hls::axis<weight_type, 0, 0, 0> weight_packet;
typedef hls::axis<bias_type, 0, 0, 0> bias_packet;
typedef hls::axis<input_type, 0, 0, 0> input_packet;
typedef hls::axis<output_type, 0, 0, 0> output_packet;

extern int Linear_16384x512(hls::stream<weight_packet> &weights_s, hls::stream<bias_packet> &bias_s, hls::stream<input_packet> &input_s, hls::stream<output_packet> &output_s);
