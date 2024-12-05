#include <iostream>
#include <stdio.h>
#include <iomanip>

#include "floatX.h"

#define mantissa_size 10
#define exponent_size 5

int main(void)
{
    float sum_max_diff = 0.0001f;
    float res_max_diff = 0.0001f;
    float mul_max_diff = 0.0001f;
    float div_max_diff = 4.00001f;

    for(float a = -128.0f; a < 128.0f; a+=1.0f)//pow(2.0f, -5.0f))
    {
        for(float b = -128.0f; b < 128.0f; b+=0.01f)//pow(2.0f, -5.0f))
        {
            bool failed = false;

            floatX<mantissa_size, exponent_size> a_mixed(a);
            floatX<mantissa_size, exponent_size> b_mixed(b);

            std::cout << std::setprecision(20) << std::fixed;

            /*
            if(a != a_mixed.to_float())
            {
                std::cout << "a is not equal: " << a << " != " << a_mixed.to_float() << std::endl;
                failed = true;
            }

            if(b != b_mixed.to_float())
            {
                std::cout << "b is not equal: " << b << " != " << b_mixed.to_float() << std::endl;
                failed = true;
            }
            */

            double sum = double(a) + double(b);
            double res = double(a) - double(b);
            double mul = double(a) * double(b);
            double div = double(a) / double(b);

            floatX<mantissa_size, exponent_size> sum_mixed = a_mixed + b_mixed;
            floatX<mantissa_size, exponent_size> res_mixed = a_mixed - b_mixed;
            floatX<mantissa_size, exponent_size> mul_mixed = a_mixed * b_mixed;
            floatX<mantissa_size, exponent_size> div_mixed = a_mixed / b_mixed;

            float sum_diff = fabs(sum - sum_mixed.to_float());
            if(sum_diff > sum_max_diff)
            {
                std::cout << "diff: " << sum_diff << " in: " << a << " + " << b << " = " << sum << std::endl;
                std::cout << a_mixed.to_float() << " + " << b_mixed.to_float() << " = " << sum_mixed.to_float() << std::endl;
                failed = true;
            }

            float mul_diff = fabs(mul - mul_mixed.to_float());
            if(mul_diff > mul_max_diff)
            {
                std::cout << "diff: " << mul_diff << " in: " << a << " * " << b << " = " << mul << std::endl;
                std::cout << a_mixed.to_float() << " * " << b_mixed.to_float() << " = " << mul_mixed.to_float() << std::endl;
                failed = true;
            }
        
            float res_diff = fabs(res - res_mixed.to_float());
            if(res_diff > res_max_diff)
            {
                std::cout << "diff: " << res_diff << " in: " << a << " - " << b << " = " << res << std::endl;
                std::cout << a_mixed.to_float() << " - " << b_mixed.to_float() << " = " << res_mixed.to_float() << std::endl;
                failed = true;
            }
            
            float div_diff = fabs(div - div_mixed.to_float());
            if(div_diff > div_max_diff)
            {
                std::cout << "diff: " << div_diff << " in: " << a << " / " << b << " = " << div << std::endl;
                std::cout << a_mixed.to_float() << " / " << b_mixed.to_float() << " = " << div_mixed.to_float() << std::endl;
                failed = true;
            }

            if(failed)
                return 1;
        }
    }

    return 0;
}
