#include "dot.h"

int dot(hls::stream<dot_packet> &input1, hls::stream<dot_packet> &input2, float &result, int &in_size)
{
#pragma HLS INTERFACE axis port = input1
#pragma HLS INTERFACE axis port = input2
#pragma HLS INTERFACE s_axilite port = result
#pragma HLS INTERFACE s_axilite port = in_size
#pragma HLS INTERFACE s_axilite port = return


dot_data_type res[DOT_PPBUFF_SIZE];

init_loop:
for(int i = 0; i < DOT_PPBUFF_SIZE; i++)
    res[i] = dot_data_type(0.0f);

main_loop:
    for (int i = 0; i < in_size; i++)
    {
// #pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min = 307200 max = 307200

        dot_packet in_packet1;
        input1.read(in_packet1);
        dot_data_type in_data1 = dot_data_type(in_packet1.data);

        dot_packet in_packet2;
        input2.read(in_packet2);
        dot_data_type in_data2 = dot_data_type(in_packet2.data);

        dot_data_type mul = in_data1*in_data2;
        res[i % DOT_PPBUFF_SIZE] += mul;

        // if (in_packet.last == 1)
        //     break;
    }

    dot_data_type res2 = dot_data_type(0.0f);

    final_loop:
    for(int i = 0; i < DOT_PPBUFF_SIZE; i++)
        res2 += res[i];

    result = float(res2);

    return 0;
}