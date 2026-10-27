/* Moteur de features LCB-FT en flux : C99 portable, sans dépendance, sans intrinsèques.
 *
 * Chaque transaction est traitée dans l'ordre : ses features sont calculées à partir de l'état
 * (transactions antérieures uniquement), puis l'état est mis à jour avec elle. Aucune étiquette
 * n'entre dans cette API. La mémoire est bornée : état par compte, table des paires à éviction,
 * anneau d'arêtes de taille fixe.
 */
#ifndef AML_ENGINE_H
#define AML_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t n_accounts;
    int32_t win_short, win_mid, win_long; /* minutes */
    int32_t motif_fmt_mask;               /* bit f : format f inclus dans le graphe de motifs */
    int32_t ach_code;                     /* code du format ACH */
    int32_t ring_bits;                    /* anneau d'arêtes : 2^ring_bits emplacements */
    int32_t pair_bits;                    /* table des paires : 2^pair_bits emplacements */
    int32_t pair_ttl;                     /* éviction des paires inactives (minutes) */
    int32_t scan_cap;                     /* arêtes lues au plus par liste d'adjacence */
    int32_t fan_cap;                      /* voisins examinés pour les motifs à deux sauts */
    int32_t cycle_max_len;                /* longueur max des cycles (arêtes) */
    int32_t cycle_budget;                 /* expansions de DFS au plus par transaction */
    float chain_tol;                      /* écart relatif max pour chaîner un passage de fonds */
} aml_config;

/* Lot de transactions (tableaux parallèles, triés par t). */
typedef struct {
    int64_t n;
    const int32_t *t; /* minutes depuis t0 */
    const int32_t *src, *dst;
    const float *usd_paid, *usd_recv;
    const double *amt_paid;
    const uint8_t *cur_paid, *cur_recv, *fmt;
} aml_batch;

/* KYC statique par compte (copié à la création). */
typedef struct {
    const int32_t *bank;
    const int32_t *entity;
    const uint8_t *etype;
    const int32_t *nacc;
    const int16_t *country;
} aml_kyc;

/* Forêt d'arbres de décision aplatie (exportée de LightGBM) : nœud interne si feat >= 0,
 * x[feat] <= thr → left, NaN → default_left ; score brut = somme des feuilles, proba = sigmoïde. */
typedef struct {
    int32_t n_trees, n_nodes;
    const int32_t *root;  /* nœud racine de chaque arbre */
    const int32_t *feat;  /* index de feature du moteur, -1 pour une feuille */
    const double *thr;
    const int32_t *left, *right;
    const uint8_t *default_left;
    const double *value;  /* valeur de feuille */
} aml_forest;

typedef struct aml_engine aml_engine;

aml_engine *aml_new(const aml_config *cfg, const aml_kyc *kyc);
void aml_free(aml_engine *e);

int32_t aml_n_features(void);
const char *aml_feature_name(int32_t i);

/* Traite un lot : écrit n * aml_n_features() float (ligne par transaction) dans out.
 * Si une forêt est attachée et scores != NULL, écrit aussi la probabilité de chaque transaction,
 * calculée juste après ses features (donc avant toute transaction ultérieure). */
int aml_process(aml_engine *e, const aml_batch *b, float *out);
int aml_process_scored(aml_engine *e, const aml_batch *b, float *out, double *scores);

/* Attache une forêt (copiée) ; renvoie 0 si succès. */
int aml_set_forest(aml_engine *e, const aml_forest *f);

/* Cascade : si le score brut du mini-modèle `pre` est < raw_cut, la forêt complète n'est pas évaluée et la
 * transaction reçoit le score du mini-modèle. pre = NULL désactive la cascade. */
int aml_set_cascade(aml_engine *e, const aml_forest *pre, double raw_cut);

/* Score brut d'une matrice de features (n lignes de stride float) sans moteur. */
void aml_forest_raw(const aml_forest *f, const float *x, int64_t n, int32_t stride, double *out);

/* Latences par transaction (ns) du dernier aml_process* si l'instrumentation est active. */
void aml_set_timing(aml_engine *e, int64_t *ns_per_tx);

/* Repère de séquence de chaque transaction (à conserver pour la revue J+1). */
void aml_set_seq_out(aml_engine *e, int64_t *seq_at);

/* Revue J+1 : features rétrospectives d'un lot déjà traité, avec l'état courant (instant de décision). */
int32_t aml_n_hindsight(void);
const char *aml_hindsight_name(int32_t i);
int aml_hindsight(aml_engine *e, const aml_batch *b, const int64_t *seq_at, float *out);

/* Instantané de l'état (comptes, anneau, paires ; sans la forêt) : octets écrits, < 0 si erreur.
 * La restauration exige le même moteur et la même configuration. */
int64_t aml_snapshot(const aml_engine *e, const char *path);
int aml_restore(aml_engine *e, const char *path);

/* Octets alloués par l'état (comptabilité exacte des allocations). */
int64_t aml_state_bytes(const aml_engine *e);

/* Compteurs internes (voir AML_STAT_* dans aml_engine.c) ; renvoie le nombre écrit. */
int32_t aml_n_stats(void);
const char *aml_stat_name(int32_t i);
void aml_stats(const aml_engine *e, int64_t *out);

#ifdef __cplusplus
}
#endif
#endif
