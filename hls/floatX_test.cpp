#include <iostream>
#include <stdio.h>
#include <iomanip>

#include "floatX.h"

int main(void)
{
    float sum_max_diff = 0.001f;
    float res_max_diff = 0.001f;
    float mul_max_diff = 0.001f;
    float div_max_diff = 0.001f;

    for(double a = -32.0f; a < 32.0f; a+=1.0f)//pow(2.0f, -5.0f))
    {
        for(double b = -32.0f; b < 32.0f; b+=0.01f)//pow(2.0f, -5.0f))
        {
            bool failed = false;

            floatX<23, 8> a_mixed(a);
            floatX<23, 8> b_mixed(b);

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

            double sum = a + b;
            double res = a - b;
            double mul = a * b;
            double div = a / b;

            floatX<23, 8> sum_mixed = a_mixed + b_mixed;
            floatX<23, 8> res_mixed = a_mixed - b_mixed;
            floatX<23, 8> mul_mixed = a_mixed * b_mixed;
            floatX<23, 8> div_mixed = a_mixed / b_mixed;

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
