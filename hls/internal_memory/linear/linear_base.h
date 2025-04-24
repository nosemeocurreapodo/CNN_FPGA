#pragma once

template <typename w_data_type, typename b_data_type, typename in_data_type, typename out_data_type, typename in_packet_type, typename out_packet_type, bool use_relu, int in_size, int out_size>
int LinearBase(hls::stream<in_packet_type> &input, hls::stream<out_packet_type> &output, const w_data_type weights[in_size][out_size], const b_data_type bias[out_size])
{
#pragma HLS INTERFACE axis port = weights
#pragma HLS INTERFACE axis port = bias
#pragma HLS INTERFACE axis port = input
#pragma HLS INTERFACE axis port = output
#pragma HLS INTERFACE s_axilite port = return

    out_data_type output_data[out_size];

in_size_loop:
    for (int i = 0; i < in_size; i++)
    {
        // #pragma HLS PIPELINE II=1

        in_packet_type input_packet;
        input.read(input_packet);
        in_data_type in_data = in_data_type(input_packet.data);

    out_size_loop:
        for (int j = 0; j < out_size; j++)
        {
            out_data_type mul = out_data_type(weights[i][j] * in_data);
            if (i == 0)
                output_data[j] = mul;
            else
                output_data[j] += mul;
        }
    }

write_res_loop:
    for (int i = 0; i < out_size; i++)
    {
        out_data_type out_data = out_data_type(output_data[i] + bias[i]);
        if (use_relu)
            if (out_data < out_data_type(0.0f))
                out_data = out_data_type(0.0f);

        out_packet_type out_packet;
        out_packet.data = out_data;
        out_packet.keep = -1;
        out_packet.strb = -1;

        if (i == out_size - 1)
            out_packet.last = true;
        else
            out_packet.last = false;

        output.write(out_packet);
    }

    return 0;
}
