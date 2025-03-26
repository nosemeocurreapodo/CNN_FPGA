#pragma once

#include "ap_axi_sdata.h"

// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "../common/floatX.h"

#if DATA_TYPE == 1
using data_type = ap_fixed<1, 1>;
#elif DATA_TYPE == 2
using data_type = ap_fixed<2, 1>;
#elif DATA_TYPE == 4
using data_type = ap_fixed<4, 2>;
#elif DATA_TYPE == 8
using data_type = ap_fixed<8, 4>;
#elif DATA_TYPE == 16
using data_type = ap_fixed<16, 8>;
#elif DATA_TYPE == 32
using data_type = ap_fixed<32, 16>;
#else
#error "Unsupported DATA_TYPE"
#endif

using packet_type = hls::axis<data_type, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false>;

const int in_channels = IN_CHANNELS;
const int out_channels = OUT_CHANNELS;
const int in_height = IN_HEIGHT;
const int in_width = IN_WIDTH;
const int padding = PADDING;
const bool use_relu = USE_RELU;
