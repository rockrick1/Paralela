#include <pthread.h>
#include <dirent.h>
#include <regex.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

class StringList{

	public:

		StringList(int max): head(nullptr), tail(nullptr), size(0), maxsize(max){}

		void push(string s){
			if (size == maxsize){
				cerr << "tried to push into a full list" << endl;
				exit(1);
			}

			if(size == 0){
				head = tail = new Node(s);
			}
			else{
				tail->next = new Node(s);
				tail = tail->next;
			}
			size++;
		}

		string pop(){

			if (size == 0) {
				cerr << "tried to pop an empty list" << endl;
				exit(1);
			}

			string popped = head->frase;

			Node* old_head = head;
			head = head->next;
			delete old_head;

			size--;

			if(size == 0){
				head = tail = nullptr;
			}

			return popped;
		}

		bool is_empty(){
			return size == 0;
		}

		bool is_full(){
			return size == maxsize;
		}

	private:

		class Node{

			public:

				Node(string f): frase(f), next(nullptr) {}

				string frase;
				Node* next;
		};

		Node* head;
		Node* tail;
		int size;
		int maxsize;

};

pthread_mutex_t semaforo = PTHREAD_MUTEX_INITIALIZER;

char QUERY[100];
StringList directories(1000); // seila um tamanho bom > ENTAO BORA FAZER UMA LISTA
unsigned INDEX = 0;
unsigned WORKING_THREADS = 0;
bool is_search_done = false;


// char *process_regex()
void *pgrep(void *arg) {
	/*char path[200];
	strcpy(path, (char*) arg);*/

	while(true){

		while(!is_search_done or directories.is_empty()) pthread_yield();

		pthread_mutex_lock(&semaforo);
		WORKING_THREADS++; //precisa disso?
		
		string path;

		if(!directories.is_empty()){
			path  = directories.pop(); //pega uma palavra e vai
		}
		else if(is_search_done){
			return 0; //acabou a pesquisa e não tem mais nada na lista
		}
		else{
			continue; //não tem nada na lista, mas a pesquisa continua
		}

		pthread_mutex_unlock(&semaforo);

		string line;
		StringList matches(-1); //hack para deixar ela de tamanho ilimitado
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
					texto << path << ": " << linenum << ": " << line.c_str() << endl;
					matches.push(texto.str());
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
			cout << "cago" << endl;

		pthread_mutex_lock(&semaforo);
		while(!matches.is_empty()) cout << matches.pop();
		WORKING_THREADS--;
		pthread_mutex_unlock(&semaforo);
		// precisa disso?
		// regfree(&regex);
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

		while(directories.is_full()) pthread_yield();

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
			directories.push(full_path.str());
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
		cout << "da os argumento direito pora" << endl;
		return 0;
	}

	int MAX_THREADS = stoi(argv[1]);

	strcpy(QUERY, argv[2]);

	char DIRECTORY[200];
	strcpy(DIRECTORY, argv[3]);

	pthread_t threads[MAX_THREADS];

	// processa os diretorios e guarda eles numa lista de stirngs
	list_dir(DIRECTORY);

	int th;
	// O i começa em 1 pois teoricamente a primeira thread está pegando a lista
	for (int i = 1; i < MAX_THREADS;) {
		// nao permite que mais que MAX_THREADS trabalhem
		if ( ( th = pthread_create(&threads[i], NULL, pgrep, NULL) ) ) //Nem precisa de argumento na real
			cout << "Failed to create thread " << th << endl;
		i++;
	}

	// mata todo mundo
	for (int i = 0; i < INDEX; i++)
		pthread_join(threads[i], NULL);

	return 0;
}