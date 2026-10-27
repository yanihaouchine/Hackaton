/* Pilote de test de portabilité : lit un lot et une configuration binaires, exécute le moteur et écrit
 * les features. Compilé pour x86-64 et pour aarch64 (exécuté sous qemu), il vérifie que le même code donne
 * les mêmes features sur les deux architectures. Format : voir tests/test_portability.py. */
#include <stdio.h>
#include <stdlib.h>

#include "../aml/engine/aml_engine.h"

static void *rd(FILE *f, size_t sz, size_t n) {
    void *p = malloc(sz * (n ? n : 1));
    if (!p || fread(p, sz, n, f) != n) {
        fprintf(stderr, "lecture incomplète\n");
        exit(2);
    }
    return p;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage : %s entree.bin sortie.bin\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    aml_config *cfg = rd(f, sizeof(aml_config), 1);
    int64_t *n = rd(f, sizeof(int64_t), 1);
    size_t na = (size_t)cfg->n_accounts, nt = (size_t)*n;
    aml_kyc kyc = {rd(f, 4, na), rd(f, 4, na), rd(f, 1, na), rd(f, 4, na), rd(f, 2, na)};
    aml_batch b = {*n, rd(f, 4, nt), rd(f, 4, nt), rd(f, 4, nt), rd(f, 4, nt), rd(f, 4, nt),
                   rd(f, 8, nt), rd(f, 1, nt), rd(f, 1, nt), rd(f, 1, nt)};
    fclose(f);
    aml_engine *e = aml_new(cfg, &kyc);
    if (!e) return 3;
    int32_t nf = aml_n_features();
    float *out = malloc(sizeof(float) * nt * (size_t)nf);
    if (!out || aml_process(e, &b, out) != 0) return 4;
    FILE *g = fopen(argv[2], "wb");
    if (!g) return 5;
    fwrite(&nf, 4, 1, g);
    fwrite(out, sizeof(float), nt * (size_t)nf, g);
    fclose(g);
    aml_free(e);
    return 0;
}
