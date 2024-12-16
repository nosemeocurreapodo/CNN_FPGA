#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"

template <typename data_type, typename packet_type>
int ReLU(hls::stream<packet_type> &input, hls::stream<packet_type> &output, int &data_size)
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = data_size
#pragma HLS INTERFACE s_axilite port = return

main_loop:
    for (int i = 0; i < data_size; i++)
    {
// #pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min = 307200 max = 307200

        packet_type in_packet;
        input.read(in_packet);
        data_type in_data = data_type(in_packet.data);

        data_type out_data = data_type(0.0f);
        if (in_data > data_type(0.0f))
            out_data = in_data;

        packet_type out_packet;
        out_packet.data = float(out_data);
        out_packet.keep = -1;
        out_packet.strb = -1;
        if (i == data_size - 1)
            out_packet.last = true;
        else
            out_packet.last = false;

        output.write(out_packet);

        // if (in_packet.last == 1)
        //     break;
    }

    return 0;
}