#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>


static void print_usage_message()
{
    const char* msg =
    "Parâmetros incorretos. Uso:\n"
    "  main <NUM>\n"
    "onde:\n"
    "  <NUM>      Número a checar por primalidade.\n"
    "\n";

    printf(msg);
}

static uint64_t parse_arg(int argc, const char* argv[])
{
    uint64_t ret;

    if (argc != 2 || sscanf(argv[1], "%lu", &ret) != 1)
    {
        print_usage_message();
        exit(1);
    }

    return ret;
}

bool is_prime(uint64_t number)
{
    uint64_t root, i;
    volatile bool result = true;

    if (number < 2)
        return false;

    if (number == 2)
        return true;

    if (number % 2 == 0)
        return false;

    root = sqrtl(number);

    #pragma omp parallel private(i)
    {
        #pragma omp for
        for (i = 3; i <= root; i += 2)
        {
            if (number % i == 0)
            {
                #pragma omp critical
                {
                    result = false;
                }
                #pragma omp cancel for
            }
        }
    }
    return result;
}

int main(int argc, const char* argv[])
{
    uint64_t number = parse_arg(argc, argv);

    if (is_prime(number))
        printf("É primo.\n");
    else
        printf("Não primo.\n");

    return 0;
}
