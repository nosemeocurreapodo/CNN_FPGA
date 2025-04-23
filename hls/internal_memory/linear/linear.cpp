#include "hls_stream.h"

#include "linear_base.h"
#include "linear_params.h"
#include "linear_weights_bias.h"

int TOP_NAME(hls::stream<in_packet_type> &input, hls::stream<out_packet_type> &output)
{

#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = return

    // data_type weights[in_channels * out_channels * 3 * 3];
    w_data_type weights[in_size][out_size];
    b_data_type bias[out_size];

#pragma HLS ARRAY_PARTITION variable = weights complete dim = 0
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 0

    LinearRandomWeights(weights);
    LinearRandomBias(bias);

    return LinearBase<w_data_type,
                      b_data_type,
                      in_data_type,
                      out_data_type,
                      in_packet_type,
                      out_packet_type,
                      use_relu,
                      in_size,
                      out_size>(input, output, weights, bias);
}