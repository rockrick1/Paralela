#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <dirent.h>
#include <regex.h>

// char *process_regex()
void pgrep(FILE *f, char query[], char path[]) {
    char *line;
    line = malloc(200*sizeof(char));
    size_t len = 0;

    regex_t regex;
    int reti;
    char msgbuf[100];

    reti = regcomp(&regex, query, 0);
    if (reti) {
        printf("Could not compile regex\n");
        exit(1);
    }

    else if (reti == REG_NOMATCH) {
        puts("No match");
    }

    if (f) {
        int linenum = 0;
        while (getline(&line, &len, f) != -1) {
            linenum++;
            // printf("%s", line);
            reti = regexec(&regex, line, 0, NULL, 0);
            if (!reti) {
                printf("%s: %d: %s\n", path, linenum, line);
            }
            else if (reti != REG_NOMATCH) {
                regerror(reti, &regex, msgbuf, sizeof(msgbuf));
                fprintf(stderr, "Regex match failed: %s\n", msgbuf);
                exit(1);
            }
        }
    }
    else
        printf("cago\n");
    // printf("%s\n", pDirent->d_name);
    free(line);
    regfree(&regex);
    return;
}

int main(int argc, char **argv) {
    if (argc <= 3) {
        printf("da os argumento pora\n");
        return 0;
    }

    int MAX_THREADS = atoi(argv[1]);
    char QUERY[100];
    char DIRECTORY[100];

    strcpy(QUERY, argv[2]);

    strcpy(DIRECTORY, argv[3]);

    DIR *pDir;
    struct dirent *pDirent;

    FILE *f = fopen(DIRECTORY, "r");

    // se for só um arquivo
    if (f == NULL) {
        printf("oie\n" );
        // pgrep(f, QUERY);
    }

    // se for um diretorio cheio
    else {
        pDir = opendir(DIRECTORY);
        if (pDir == NULL) {
            printf ("Cannot open directory '%s'\n", DIRECTORY);
            return 1;
        }

        while ((pDirent = readdir(pDir)) != NULL) {
            char path[200];

            strcpy(path, DIRECTORY);
            strcat(path, "/");
            strcat(path, pDirent->d_name);

            FILE *f = fopen(path, "r");

            pgrep(f, QUERY, path);
        }
        closedir(pDir);
    }





    // pthread_t thread;
    return 0;
}