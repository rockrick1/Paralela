
#include <iostream>
#include <string>
#include <complex>

using namespace std;

template<class real> //para trocar entre float e double
void mandelbrot(char *argv[]){
	complex<real> c0 (stod(argv[1]), stod(argv[2]));
	complex<real> c1 (stod(argv[3]), stod(argv[4]));

	int W = stoi(argv[5]);
	int H = stoi(argv[6]);

	string CPU_GPU = argv[7];

	int threads = stoi(argv[8]);

	string saida = argv[9]; 
}

int main(int argc, char *argv[]){
	//processar os args
	//mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>

	if(argc < 10){
		cout << "Incorrect Number of Args" << endl;
		cout << "Usage:" << endl;
		cout << "mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>" << endl;
		return 0;
	}

	mandelbrot<float>(argv);
	return 0;

}