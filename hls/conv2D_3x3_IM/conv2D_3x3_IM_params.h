#pragma once

#include "ap_axi_sdata.h"

// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "../common/floatX.h"

#if DATA_TYPE == 0
using data_type = float;
#elif DATA_TYPE == 1
using data_type = half;
#elif DATA_TYPE == 2
using data_type = ap_fixed<32, 16>;
#elif DATA_TYPE == 3
using data_type = ap_fixed<16, 8>;
#elif DATA_TYPE == 4
using data_type = ap_fixed<8, 4>;
#elif DATA_TYPE == 5
using data_type = ap_fixed<4, 2>;
#else
#error "Unsupported DATA_TYPE"
#endif

using packet_type = hls::axis<data_type, 0, 0, 0, (AXIS_ENABLE_KEEP | AXIS_ENABLE_LAST | AXIS_ENABLE_STRB), false>;

const int in_channels = IN_CHANNELS;
const int out_channels = OUT_CHANNELS;
const int height = HEIGHT;
const int width = WIDTH;
const int padding = PADDING;
