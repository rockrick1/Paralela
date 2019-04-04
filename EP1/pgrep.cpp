#include <string.h>
#include <pthread.h>
#include <dirent.h>
#include <regex.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

string QUERY; // melhor ser const?


// arg da thread vai ser um arquivo
void *thread_work(void *arg) {

}

// char *process_regex()
void *pgrep(void *arg) {
    string path = (string*) arg;
    string line;
    size_t len = 0;

    regex_t regex;
    int reti;
    string msgbuf;
    ifstream file(path);

    reti = regcomp(&regex, QUERY.c_str(), 0);
    if (reti) {
        printf("Could not compile regex\n");
        exit(1);
    }

    else if (reti == REG_NOMATCH) {
        puts("No match");
    }

    if (file) {
        int linenum = 0;
        while (getline(file, line)) {
            // printf("%s\n", line.c_str());
            reti = regexec(&regex, line.c_str(), 0, NULL, 0);
            if (!reti) {
                pthread_mutex_lock(&mutex);
                printf("%s: %d: %s\n", path.c_str(), linenum, line.c_str());
                pthread_mutex_unlock(&mutex);
            }
            else if (reti != REG_NOMATCH) {
                char *errbuf;
                strcpy(errbuf, msgbuf.c_str());
                regerror(reti, &regex, errbuf, sizeof(errbuf));
                fprintf(stderr, "Regex match failed: %s\n", errbuf);
                exit(1);
            }
            linenum++;
        }
    }
    else
        printf("cago\n");
    // precisa disso?
    // regfree(&regex);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("da os argumento direito pora\n");
        return 0;
    }

    int MAX_THREADS = stoi(argv[1]);
    QUERY = argv[2];
    string DIRECTORY = argv[3];

    pthread_t threads[MAX_THREADS];

    DIR *pDir;
    struct dirent *pDirent;

    // começa a processar o diretorio
    pDir = opendir(DIRECTORY.c_str());
    if (pDir == NULL) {
        printf ("Cannot open directory '%s'\n", DIRECTORY.c_str());
        return 1;
    }

    int i = 0;
    int th;
    while ((pDirent = readdir(pDir)) != NULL) {
        string path;
        path = DIRECTORY + "/" + pDirent->d_name;
        if ((th = pthread_create(threads[i], NULL, pgrep, (void *) path)))
            printf("Failed to create thread %d\n", th);
        // pgrep(path);
    }
    closedir(pDir);


    // pthread_t thread;
    return 0;
}