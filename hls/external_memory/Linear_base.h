#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"

template <typename weight_data_type, typename weight_packet_type, typename bias_data_type, typename bias_packet_type, typename input_data_type, typename input_packet_type, typename output_data_type, typename output_packet_type, int in_size, int out_size>
int Linear_base(hls::stream<weight_packet_type> &weights_s, hls::stream<bias_packet_type> &bias_s, hls::stream<input_packet_type> &input_s, hls::stream<output_packet_type> &output_s)
{
#pragma HLS INTERFACE axis port = weights_s
#pragma HLS INTERFACE axis port = bias_s
#pragma HLS INTERFACE axis port = input_s
#pragma HLS INTERFACE axis port = output_s
#pragma HLS INTERFACE s_axilite port = return

    output_data_type output[out_size];

in_size_loop:
    for (int i = 0; i < in_size; i++)
    {
// #pragma HLS PIPELINE II=1

        input_packet_type input_packet;
        input_s.read(input_packet);
        input_data_type input = input_data_type(input_packet.data);

    out_size_loop:
        for (int j = 0; j < out_size; j++)
        {
            weight_packet_type weights_packet;
            weights_s.read(weights_packet);
            weight_data_type weight = weight_data_type(weights_packet.data);

            output_data_type mul = weight * input;
            if (i == 0)
                output[j] = mul;
            else
                output[j] += mul;
        }
    }

write_res_loop:
    for (int i = 0; i < out_size; i++)
    {
        bias_packet_type bias_packet;
        bias_s.read(bias_packet);
        bias_data_type bias = bias_data_type(bias_packet.data);

        output_packet_type out_packet;
        out_packet.data = output[i] + output_data_type(bias);
        out_packet.keep = -1;
        out_packet.strb = -1;

        if (i == out_size - 1)
            out_packet.last = true;
        else
            out_packet.last = false;

        output_s.write(out_packet);
    }

    return 0;
}
