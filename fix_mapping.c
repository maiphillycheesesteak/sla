#include <stdio.h>
#include <string.h>

int main() {
    FILE *f = fopen("sla.h", "r");
    FILE *fout = fopen("sla.h.tmp", "w");
    char line[2048];
    while(fgets(line, sizeof(line), f)) {
        // Find map(from: res.data...) and replace with map(alloc: res.data...)
        char *pos = strstr(line, "map(from: res.data");
        if (pos) {
            pos[4] = 'a';
            pos[5] = 'l';
            pos[6] = 'l';
            pos[7] = 'o';
            pos[8] = 'c';
            // Need to insert an enter data mapping BEFORE the target loop
            // This is complex via C string manipulation. Let's just fix the macros directly via sed.
        }
        fputs(line, fout);
    }
    fclose(f);
    fclose(fout);
    return 0;
}
