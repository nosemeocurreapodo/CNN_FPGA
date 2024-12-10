#include "conv2D_3x3.h"

int conv2D_3x3(hls::stream<conv_packet> &input, hls::stream<conv_packet> &output, int &in_width, int &in_height, float weights[3 * 3])
{
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = in_width
#pragma HLS INTERFACE s_axilite port = in_height
#pragma HLS INTERFACE s_axilite port = weights
#pragma HLS INTERFACE s_axilite port = return

    shift_register<conv_data_type, CONV_MAX_WIDTH * 3> shift_reg;
    // #pragma HLS ARRAY_PARTITION variable=shift_reg.data dim=1 factor=5 type=cyclic
    //  #pragma HLS ARRAY_PARTITION variable = in_buffer complete dim = 2

    mat3<conv_data_type> kernel;
#pragma HLS ARRAY_PARTITION variable = kernel.data dim = 0 type = complete

    // initialize buffers

init_kernel_y_loop:
    for (int y = 0; y < 3; y++)
    {
#pragma HLS UNROLL
    init_kernel_x_loop:
        for (int x = 0; x < 3; x++)
        {
#pragma HLS UNROLL
            kernel.data[y][x] = conv_data_type(weights[x + y * 3]);
        }
    }

init_buffer_loop:
    for (int i = 0; i < 3 * in_width; i++)
    {
        conv_packet in_packet;
        input.read(in_packet);
        conv_data_type in_data = conv_data_type(in_packet.data);
        shift_reg.shift_down(in_data);
    }

    bool last_was_read = false;

main_loop:
    for (int i = 0; i < in_width * in_height; i++)
    {
// #pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min = 307200 max = 307200

        conv_data_type in_data = conv_data_type(0.0f);
        if (!last_was_read)
        {
            conv_packet in_packet;
            input.read(in_packet);
            in_data = conv_data_type(in_packet.data);
            if (in_packet.last == 1)
                last_was_read = true;
        }

        shift_reg.shift_down(in_data);

        mat3<conv_data_type> data = shift_reg.getMat3(in_width);
        conv_data_type conv = data.mul_v2(kernel);

        conv_packet out_packet;
        out_packet.data = float(conv);
        output.write(out_packet);
    }

    return 0;
}