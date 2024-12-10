#include "Linear.h"

int Linear(hls::stream<linear_packet> &input, hls::stream<linear_packet> &output, int &in_size, int &out_size, int &max_width, float weights[LINEAR_MAX_WIDTH])
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = in_size
#pragma HLS INTERFACE s_axilite port = out_size
#pragma HLS INTERFACE s_axilite port = max_width
#pragma HLS INTERFACE s_axilite port = weights
#pragma HLS INTERFACE s_axilite port = return

    max_width = LINEAR_MAX_WIDTH;

    linear_data_type output_data[LINEAR_MAX_WIDTH];

set_zero_loop:
    for (int i = 0; i < out_size; i++)
    {
        output_data[i] = linear_data_type(0.0f);
    }

main_loop:
    for (int j = 0; j < in_size; j++)
    {
// #pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min = 307200 max = 307200

        linear_packet in_packet;
        input.read(in_packet);
        linear_data_type in_data = linear_data_type(in_packet.data);

    dot_product_loop:
        for (int i = 0; i < out_size; i++)
        {
            output_data[i] += in_data * weights[i];
        }
        // if (in_packet.last == 1)
        //     break;
    }

write_output_loop:
    for (int i = 0; i < out_size; i++)
    {
        linear_packet out_packet;
        out_packet.data = float(output_data[i]);
        output.write(out_packet);
    }

    return 0;
}