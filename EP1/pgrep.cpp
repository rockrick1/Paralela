#include <string.h>
#include <pthread.h>
#include <dirent.h>
#include <regex.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;


pthread_mutex_t semaforo = PTHREAD_MUTEX_INITIALIZER;

char QUERY[100];
string directories[300]; // seila um tamanho bom
unsigned INDEX = 0;
unsigned WORKING_THREADS = 0;


// char *process_regex()
void *pgrep(void *arg) {
    char path[200];
    strcpy(path, (char*) arg);

    string line;
    size_t len = 0;

    regex_t regex;
    int reti;
    string msgbuf;
    ifstream file(path);

    cout << "oi eu vo processar " << path << endl;
    pthread_mutex_lock(&semaforo);
    WORKING_THREADS++;
    pthread_mutex_unlock(&semaforo);

    reti = regcomp(&regex, QUERY, 0);
    if (reti) {
        cout << "Could not compile regex" << endl;
        exit(1);
    }

    else if (reti == REG_NOMATCH) {
        puts("No match");
    }

    if (file) {
        int linenum = 0;
        while (getline(file, line)) {
            reti = regexec(&regex, line.c_str(), 0, NULL, 0);
            if (!reti) {
                pthread_mutex_lock(&semaforo);
                cout << path << ": " << linenum << ": " << line.c_str() << endl;
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
        printf("cago\n");
    cout << "oi eu processei " << path << " e acabei." << endl;
    pthread_mutex_lock(&semaforo);
    WORKING_THREADS--;
    pthread_mutex_unlock(&semaforo);
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
            directories[INDEX].append(dir_name);
            directories[INDEX].append("/");
            directories[INDEX].append(d_name);
            INDEX++;
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

    for (int i = 0; i < INDEX; i++)
        cout << "d:        " << directories[i].c_str() << endl;

    int th;
    char path[200];
    for (int i = 0; i < INDEX;) {
        // nao permite que mais que MAX_THREADS trabalhem
        if (WORKING_THREADS < MAX_THREADS) {
            if ((th = pthread_create(&threads[i], NULL, pgrep,
                (void *) directories[i].c_str())))
                cout << "Failed to create thread" << th << endl;
            i++;
        }
    }

    // mata todo mundo
    for (int i = 0; i < INDEX; i++)
        pthread_join(threads[i], NULL);

    return 0;
}