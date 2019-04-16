#include <pthread.h>
#include <dirent.h>
#include <regex.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <queue>

using namespace std;

pthread_mutex_t semaforo = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t change_list = PTHREAD_MUTEX_INITIALIZER;

char QUERY[100];
string DIRECTORY;
queue<string> directories;
bool is_search_done = false;


void *pgrep(void *arg) {

	while(true){

		// espera chegar mais arquivos pra processar
		while(directories.empty() and !is_search_done) {
			pthread_yield();
		}


		pthread_mutex_lock(&change_list);
		string path;

		// começa a processar um arquivo, caso tenha um
		if(!directories.empty()) {
			path  = directories.front(); //pega uma palavra e vai
			directories.pop();

			// a partir daqui podemos sair da seção critica, para outras
			// threads tambem poderem trabalhar
			pthread_mutex_unlock(&change_list);




			string line;
			queue<string> matches;
			size_t len = 0;

			regex_t regex;
			int reti;
			string msgbuf;
			ifstream file(path);

			reti = regcomp(&regex, QUERY, 0);
			if (reti) {
				cout << "Could not compile regex" << endl;
				exit(1);
			}

			else if (reti == REG_NOMATCH) {
				cout << "No match" << endl;
			}

			if (file) {
				int linenum = 0;
				while (getline(file, line)) {
					reti = regexec(&regex, line.c_str(), 0, NULL, 0);
					if (!reti) {
						stringstream texto;
						texto << path << ": " << linenum << endl;

						pthread_mutex_lock(&semaforo);
						matches.push(texto.str());
						pthread_mutex_unlock(&semaforo);
					}
					else if (reti != REG_NOMATCH) {
						char *errbuf;
						strcpy(errbuf, msgbuf.c_str());
						regerror(reti, &regex, errbuf, sizeof(errbuf));
						cerr << "Regex match failed: " << errbuf << endl;
						exit(1);
					}
					linenum++;
				}
			}
			else
				cout << "Couldn't open file: " << path << endl;

			// imprime os matches, numa seção crítica
			pthread_mutex_lock(&semaforo);
			while(!matches.empty()) {
				cout << matches.front().erase(0,2);
				matches.pop();
			}
			pthread_mutex_unlock(&semaforo);

		}

		// acabaram os arquivos, todas as threads terminam
		else if(is_search_done and directories.empty()) {
			pthread_mutex_unlock(&change_list);
			return 0;
		}

		// não há nada a fazer ainda
		else {
			pthread_mutex_unlock(&change_list);
			continue;
		}
	}
}

// preenche a lista global de diretorios recusivamente
void list_dir (const char * dir_name) {
	DIR *d;

	d = opendir (dir_name);

	if (!d) {
		cerr << "Cannot open directory '" << dir_name << "': " << strerror(errno) << endl;
		exit (EXIT_FAILURE);
	}

	while (1) {

		struct dirent *entry;
		const char *d_name;

		entry = readdir (d);
		if (! entry) {
			break;
		}

		d_name = entry->d_name;

		// se nao for um diretorio, adiciona ele na lista de arquivos
		if (! (entry->d_type & DT_DIR)) {
			// printf ("%s/%s\n", dir_name, d_name);
			stringstream full_path;
			full_path << dir_name << "/" << d_name;
			while (directories.size() >= 1000)  pthread_yield();
			pthread_mutex_lock(&change_list);
			directories.push(full_path.str());
			pthread_mutex_unlock(&change_list);
		}

		// caso contrario, é um diretorio, entao faz a recursao
		if (entry->d_type & DT_DIR) {
			// nao pega .. ou .
			if (strcmp (d_name, "..") != 0 && strcmp (d_name, ".") != 0) {
				int path_length;
				char path[PATH_MAX];

				path_length = snprintf(path, PATH_MAX, "%s/%s", dir_name, d_name);
				if (path_length >= PATH_MAX) {
					cerr << "Path length has got too long." << endl;
					exit (EXIT_FAILURE);
				}
				list_dir (path);
			}
		}
	}

	if (closedir (d)) {
		cerr << "Could not close '" << dir_name << "': "<< strerror(errno) << endl;
		exit (EXIT_FAILURE);
	}
}

int main(int argc, char **argv) {
	if (argc != 4) {
		cout << "modo de uso: " << argv[0] << " num_threads regex file_path" << endl;
		return 0;
	}

	int MAX_THREADS = stoi(argv[1]);
	int th;

	strcpy(QUERY, argv[2]);

	DIRECTORY = argv[3];

	pthread_t threads[MAX_THREADS];

	for (int i = 0; i < MAX_THREADS;) {
		if ( ( th = pthread_create(&threads[i], NULL, pgrep, NULL) ) )
			cout << "Failed to create thread " << th << endl;
		i++;
	}

	// processa os diretorios e guarda eles numa lista de stirngs global
	// directories. Aqui temos o produtor, e as threads serão os consumidores.
	list_dir(DIRECTORY.c_str());
	is_search_done = true;

	// mata todas as threads
	for (int i = 0; i < MAX_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}

	return 0;
}