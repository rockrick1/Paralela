#include <string>
#include <sstream>
#include <thread>
#include <mutex>
#include <dirent.h>
#include <regex.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

class StringList{

    public:

        StringList(): head(nullptr), tail(nullptr), size(0) {}

        void push(string s){
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

};

std::mutex semaforo;

char QUERY[100];
StringList directories; // seila um tamanho bom > ENTAO BORA FAZER UMA LISTA
unsigned INDEX = 0;
unsigned WORKING_THREADS = 0;


// char *process_regex()
void *pgrep(string arg) {
    char path[arg.size()+1];
    arg.copy(path, arg.size()+1);
    path[arg.size()] = '\0';

    StringList matches;
    string line;
    size_t len = 0;

    regex_t regex;
    int reti;
    string msgbuf;
    ifstream file(path);

    //cout << "oi eu vo processar " << path << endl;
    semaforo.lock();
    WORKING_THREADS++;
    semaforo.unlock();

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

    semaforo.lock();
    while(!matches.is_empty()) cout << matches.pop();
    WORKING_THREADS--;
    //cout << "oi eu processei " << path << " e acabei." << endl;
    semaforo.unlock();
    
    // precisa disso?
    // regfree(&regex);
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
        cout << "modo de uso: ./" << argv[0] << " num_threads regex file_path" << endl;
        return 0;
    }

    int MAX_THREADS = stoi(argv[1]);

    strcpy(QUERY, argv[2]);

    char DIRECTORY[200];
    strcpy(DIRECTORY, argv[3]);

    // processa os diretorios e guarda eles numa lista de stirngs
    list_dir(DIRECTORY);

    /*for (int i = 0; i < INDEX; i++)
        cout << "d:        " << directories[i].c_str() << endl; cant do this anymore */

    while(!directories.is_empty()) {
        // nao permite que mais que MAX_THREADS trabalhem
        while(WORKING_THREADS >= MAX_THREADS) this_thread::yield();
            
        thread(pgrep,directories.pop()).detach();
        
    }

    while(WORKING_THREADS > 0) this_thread::yield();

    return 0;
}