#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "../common/floatX.h"

#include "conv2D_3x3_IM_base.h"
#include "conv2D_3x3_IM_params.h"
#include "conv2D_3x3_IM_weights.h"

int TOP_NAME(hls::stream<packet_type> &input, hls::stream<packet_type> &output)
{

#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = return

    data_type weights[in_channels * out_channels * 3 * 3];
    conv2D_3x3_IM_weights(weights);

    return conv2D_3x3_IM_base<data_type,
                              packet_type,
                              in_channels,
                              out_channels,
                              height,
                              width,
                              padding>(input, output, weights);
}