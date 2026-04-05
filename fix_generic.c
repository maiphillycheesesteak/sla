#include <stdio.h>

int main() {
    FILE *f = fopen("sla.h", "r");
    FILE *fout = fopen("sla.h.tmp", "w");
    char line[1024];
    while(fgets(line, sizeof(line), f)) {
        // Replace _Generic((type)0 with _Generic((type*)0
        char *pos = line;
        while ((pos = strstr(pos, "_Generic((type)0"))) {
            pos[13] = '*'; // Change ')' to '*'
            pos[14] = ')'; // Add ')'
            pos[15] = '0';
            // Simple replace is tricky if sizes change, let's just do a sed replacement in bash instead
        }
        fputs(line, fout);
    }
    fclose(f);
    fclose(fout);
    return 0;
}
