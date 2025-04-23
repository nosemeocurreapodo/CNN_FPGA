#pragma once

#include "linear_params.h"

inline void LinearRandomWeights(w_data_type weights[in_size][out_size])
{
    for (int i = 0; i < in_size; i++)
    {
        for (int j = 0; j < out_size; j++)
        {
            int index = i * out_size + j;
            weights[i][j] = w_data_type(float(index) / 10.0);
        }
    }
}

inline void LinearRandomBias(b_data_type bias[out_size])
{
    for (int j = 0; j < out_size; j++)
    {
        bias[j] = b_data_type(float(j) / 10.0);
    }
}
