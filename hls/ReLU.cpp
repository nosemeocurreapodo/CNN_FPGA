#include "ReLU.h"


int ReLU(hls::stream<relu_packet> &input, hls::stream<relu_packet> &output, int &data_size)
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = data_size
#pragma HLS INTERFACE s_axilite port = return

main_loop:
    for(int i = 0; i < data_size; i++)
    {
// #pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min = 307200 max = 307200

        relu_packet in_packet;
        input.read(in_packet);
        relu_data_type in_data = relu_data_type(in_packet.data);
        
        relu_data_type out_data = relu_data_type(0.0f);
        if(in_data > relu_data_type(0.0f))
            out_data = in_data;

        relu_packet out_packet;
        out_packet.data = float(out_data);
        out_packet.keep = -1;
        out_packet.strb = -1;
        if(i == data_size - 1)
            out_packet.last = true;
        else
            out_packet.last = false;

        output.write(out_packet);

        //if (in_packet.last == 1)
        //    break;
    }

    return 0;
}