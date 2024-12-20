#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"

template <typename data_type, typename packet_type, int in_size, int out_size>
int Linear(hls::stream<packet_type> &weights_s, hls::stream<packet_type> &bias_s, hls::stream<packet_type> &input_s, hls::stream<packet_type> &output_s)
{
#pragma HLS INTERFACE axis port = weights_s
#pragma HLS INTERFACE axis port = bias_s
#pragma HLS INTERFACE axis port = input_s
#pragma HLS INTERFACE axis port = output_s
#pragma HLS INTERFACE s_axilite port = return

    data_type output[out_size];

in_size_loop:
    for (int i = 0; i < in_size; i++)
    {
// #pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min = 12544 max = 12544

        packet_type input_packet;
        input_s.read(input_packet);
        data_type input = data_type(input_packet.data);

    out_size_loop:
        for (int j = 0; j < out_size; j++)
        {
            packet_type weights_packet;
            weights_s.read(weights_packet);
            data_type weight = data_type(weights_packet.data);

            data_type mul = weight * input;
            if (i == 0)
                output[j] = mul;
            else
                output[j] += mul;
        }
    }

write_res_loop:
    for (int i = 0; i < out_size; i++)
    {
        packet_type bias_packet;
        bias_s.read(bias_packet);
        data_type bias = data_type(bias_packet.data);

        packet_type out_packet;
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
