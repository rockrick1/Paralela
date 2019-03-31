#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <dirent.h>
#include <regex.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

// char *process_regex()
void pgrep(ifstream &f, string query, string path) {
    string line;
    size_t len = 0;

    regex_t regex;
    int reti;
    string msgbuf;

    reti = regcomp(&regex, query.c_str(), 0);
    if (reti) {
        printf("Could not compile regex\n");
        exit(1);
    }

    else if (reti == REG_NOMATCH) {
        puts("No match");
    }

    if (f) {
        int linenum = 0;
        while (getline(f, line)) {
            linenum++;
            // printf("%s\n", line.c_str());
            reti = regexec(&regex, line.c_str(), 0, NULL, 0);
            if (!reti) {
                printf("%s: %d: %s\n", path.c_str(), linenum, line.c_str());
            }
            else if (reti != REG_NOMATCH) {
                char *errbuf;
                strcpy(errbuf, msgbuf.c_str());
                regerror(reti, &regex, errbuf, sizeof(errbuf));
                fprintf(stderr, "Regex match failed: %s\n", errbuf);
                exit(1);
            }
        }
    }
    else
        printf("cago\n");
    // precisa disso?
    // regfree(&regex);
    return;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("da os argumento direito pora\n");
        return 0;
    }

    int MAX_THREADS = stoi(argv[1]);
    string QUERY = argv[2];
    string DIRECTORY = argv[3];

    DIR *pDir;
    struct dirent *pDirent;

    // começa a processar o diretorio
    pDir = opendir(DIRECTORY.c_str());
    if (pDir == NULL) {
        printf ("Cannot open directory '%s'\n", DIRECTORY.c_str());
        return 1;
    }
    while ((pDirent = readdir(pDir)) != NULL) {
        string path;
        path = DIRECTORY + "/" + pDirent->d_name;
        ifstream file(path);
        pgrep(file, QUERY, path);
    }
    closedir(pDir);


    // pthread_t thread;
    return 0;
}