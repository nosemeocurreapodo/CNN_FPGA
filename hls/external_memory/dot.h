#pragma once

#include "ap_axi_sdata.h"
#include "hls_stream.h"
// #include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "floatX.h"

template <typename data_type, typename packet_type, int ppbuf_size>
int dot(hls::stream<packet_type> &input1, hls::stream<packet_type> &input2, float &result, int &in_size)
{
#pragma HLS INTERFACE axis port = input1
#pragma HLS INTERFACE axis port = input2
#pragma HLS INTERFACE s_axilite port = in_size
#pragma HLS INTERFACE s_axilite port = return

    data_type res[ppbuf_size];

init_loop:
    for (int i = 0; i < ppbuf_size; i++)
        res[i] = data_type(0.0f);

main_loop:
    for (int i = 0; i < in_size; i++)
    {
// #pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min = 307200 max = 307200

        packet_type in_packet1;
        input1.read(in_packet1);
        data_type in_data1 = data_type(in_packet1.data);

        packet_type in_packet2;
        input2.read(in_packet2);
        data_type in_data2 = data_type(in_packet2.data);

        data_type mul = in_data1 * in_data2;
        res[i % ppbuf_size] += mul;

        // if (in_packet.last == 1)
        //     break;
    }

    data_type res2 = data_type(0.0f);

final_loop:
    for (int i = 0; i < ppbuf_size; i++)
        res2 += res[i];

    result = float(res2);

    return 0;
}
