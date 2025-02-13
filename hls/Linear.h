#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"

template <typename weight_type, typename weight_packet, typename bias_type, typename bias_packet, typename input_type, typename input_packet, typename output_type, typedef output_packet, int in_size, int out_size>
int Linear(hls::stream<weight_packet> &weights_s, hls::stream<bias_packet> &bias_s, hls::stream<input_packet> &input_s, hls::stream<output_packet> &output_s)
{
#pragma HLS INTERFACE axis port = weights_s
#pragma HLS INTERFACE axis port = bias_s
#pragma HLS INTERFACE axis port = input_s
#pragma HLS INTERFACE axis port = output_s
#pragma HLS INTERFACE s_axilite port = return

    output_type output[out_size];

in_size_loop:
    for (int i = 0; i < in_size; i++)
    {
// #pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min = 12544 max = 12544

        hls::axis<input_type, 0, 0, 0> input_packet;
        input_s.read(input_packet);
        input_type input = data_type(input_packet.data);

    out_size_loop:
        for (int j = 0; j < out_size; j++)
        {
            hls::axis<weight_type, 0, 0, 0> weights_packet;
            weights_s.read(weights_packet);
            weight_type weight = data_type(weights_packet.data);

            output_type mul = weight * input;
            if (i == 0)
                output[j] = mul;
            else
                output[j] += mul;
        }
    }

write_res_loop:
    for (int i = 0; i < out_size; i++)
    {
        hls::axis<bias_type, 0, 0, 0> bias_packet;
        bias_s.read(bias_packet);
        bias_type bias = data_type(bias_packet.data);

        output_type out_packet;
        out_packet.data = float(output[i] + bias);
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
