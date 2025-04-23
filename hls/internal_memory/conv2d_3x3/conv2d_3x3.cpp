#include "hls_stream.h"

#include "conv2d_3x3_base.h"
#include "conv2d_3x3_params.h"
#include "conv2d_3x3_weights.h"

int TOP_NAME(hls::stream<packet_type> &input, hls::stream<packet_type> &output)
{

#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = return

    //data_type weights[in_channels * out_channels * 3 * 3];
    mat3<w_data_type> weights[in_channels][out_channels];
    #pragma HLS ARRAY_PARTITION variable = weights complete dim = 0

    Conv2d3x3Weights(weights);

    return Conv2D3x3IMBase<w_data_type,
                              a_data_type,
                              packet_type,
                              use_relu,
                              batch_size,
                              in_channels,
                              out_channels,
                              in_height,
                              in_width,
                              padding>(input, output, weights);
}