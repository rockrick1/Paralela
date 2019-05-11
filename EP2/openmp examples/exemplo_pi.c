#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void print_usage_message()
{
    const char* msg =
    "Parâmetros incorretos. Uso:\n"
    "  main <NUM>\n"
    "onde:\n"
    "  <NUM>      Número  de pontos.\n"
    "\n";

    printf(msg);
}

static unsigned parse_arg(int argc, const char* argv[])
{
    unsigned ret;

    if (argc != 2 || sscanf(argv[1], "%u", &ret) != 1)
    {
        print_usage_message();
        exit(1);
    }

    return ret;
}


double calculate_pi(unsigned n)
{
	double acc = 0;
	double interval_size = 1.0 / n; // The circle radius is 1.0
	unsigned i;

	// Integrates f(x) = sqrt(1 - x^2)
    #pragma omp parallel for private(i) reduction(+:acc)
 	for(i = 0; i < n; ++i)
	{
		double x = (i * interval_size) + interval_size / 2;
		acc += sqrt(1 - (x * x)) * interval_size;
 	}

	return 4*acc;
}

int main(int argc, const char* argv[])
{
    unsigned n = parse_arg(argc, argv);
    double pi;

    pi = calculate_pi(n);

    printf("pi = %lf\n", pi);

    return 0;
}
