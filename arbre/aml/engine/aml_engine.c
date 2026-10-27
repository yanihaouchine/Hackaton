/* Moteur de features LCB-FT en flux (voir aml_engine.h).
 *
 * Familles de features (toutes calculées AVANT la mise à jour de l'état par la transaction) :
 *   F0 tx_*   intrinsèques            F1 kyc_* KYC statique
 *   F2 s_* et r_* : profils émetteur et destinataire (EWMA multi-échelles, Welford)
 *   F3 p_*    relation (paire, arête inverse)
 *   F4 g_*    degrés fenêtrés du graphe de motifs (fan-in, fan-out)
 *   F6 f_*    conservation des flux : appariement de montants, profondeur de chaîne de passage
 *   F5 m_*    motifs : cycles temporels, scatter-gather, chemins à deux sauts, recouvrement biparti
 *   F7 z_*    propagation : scores (étape 1) des transactions passées des deux comptes, si une forêt
 *             est attachée ; 0 sinon. Seuls des scores antérieurs sont utilisés (causal).
 */
#define _POSIX_C_SOURCE 199309L /* clock_gettime */
#include "aml_engine.h"

#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#define NONE_T INT32_MIN
#define N_SCALES 3
#define NO_MATCH 9.0f
#define MAX_PATH 40
#define CYCLE_FOUND_CAP 16

/* ---- registre des features (source unique des noms) ------------------------------------------ */
#define PROFILE(X, p)                                                                              \
    X(p##_age) X(p##_dt_out) X(p##_dt_in) X(p##_n_out) X(p##_n_in)                                 \
    X(p##_ewc_out_s) X(p##_ewc_out_m) X(p##_ewc_out_l) X(p##_ewa_out_s) X(p##_ewa_out_m)           \
    X(p##_ewa_out_l) X(p##_ewc_in_s) X(p##_ewc_in_m) X(p##_ewc_in_l) X(p##_ewa_in_s)               \
    X(p##_ewa_in_m) X(p##_ewa_in_l) X(p##_z_out) X(p##_z_in) X(p##_ach_out) X(p##_ach_in)          \
    X(p##_dist_out) X(p##_dist_in) X(p##_flow) X(p##_n_self)

#define FEATURES(X)                                                                                \
    X(tx_log_usd) X(tx_fmt) X(tx_is_ach) X(tx_cur) X(tx_xcur) X(tx_self) X(tx_same_bank)           \
    X(tx_hour) X(tx_round)                                                                         \
    X(kyc_s_etype) X(kyc_r_etype) X(kyc_s_nacc) X(kyc_r_nacc) X(kyc_same_entity)                   \
    X(kyc_s_country) X(kyc_r_country) X(kyc_cross_border)                                          \
    PROFILE(X, s) PROFILE(X, r)                                                                    \
    X(p_cnt) X(p_dt) X(p_rev_cnt) X(p_rev_dt)                                                      \
    X(g_s_outdeg_m) X(g_s_outdeg_l) X(g_s_outcnt_l) X(g_s_indeg_l)                                 \
    X(g_r_indeg_m) X(g_r_indeg_l) X(g_r_incnt_l) X(g_r_outdeg_l)                                   \
    X(f_best_err) X(f_best_lag) X(f_n_match2) X(f_n_match10) X(f_chain_depth) X(f_chain_age)      \
    X(f_in_sum_l) X(f_out_sum_l) X(f_split) X(f_r_ret_err)                                         \
    X(m_cyc_n) X(m_cyc_minlen) X(m_cyc_drift) X(m_cyc4_n) X(m_sg) X(m_2hop) X(m_bip)            \
    X(z_s_in_max) X(z_s_out_max) X(z_r_in_max) X(z_r_out_max)                                     \
    X(z_s_susp_out) X(z_s_susp_in) X(z_r_susp_out) X(z_r_susp_in)

enum {
#define X(name) F_##name,
    FEATURES(X)
#undef X
        F_COUNT
};

static const char *FEATURE_NAMES[] = {
#define X(name) #name,
    FEATURES(X)
#undef X
};

/* décalages à l'intérieur d'un bloc PROFILE (même ordre que la macro) */
enum {
    P_age, P_dt_out, P_dt_in, P_n_out, P_n_in,
    P_ewc_out, P_ewa_out = P_ewc_out + N_SCALES, P_ewc_in = P_ewa_out + N_SCALES,
    P_ewa_in = P_ewc_in + N_SCALES, P_z_out = P_ewa_in + N_SCALES, P_z_in, P_ach_out, P_ach_in,
    P_dist_out, P_dist_in, P_flow, P_n_self, P_COUNT
};
typedef char profile_layout_check[(F_s_n_self - F_s_age + 1 == P_COUNT) ? 1 : -1];

/* ---- compteurs internes ---------------------------------------------------------------------- */
#define STATS(X)                                                                                   \
    X(tx) X(edges) X(ring_overflow) X(recycled_links) X(scan_cap_hits) X(dfs_expansions)              \
    X(cycle_budget_hits) X(pair_used) X(pair_evicted) X(cascade_skips)
enum {
#define X(name) ST_##name,
    STATS(X)
#undef X
        ST_COUNT
};
static const char *STAT_NAMES[] = {
#define X(name) #name,
    STATS(X)
#undef X
};

/* ---- structures -------------------------------------------------------------------------------- */
typedef struct {
    int64_t seq;      /* numéro global (validité après recouvrement de l'anneau) */
    int64_t prev_out; /* arête sortante précédente de src (seq), -1 */
    int64_t prev_in;  /* arête entrante précédente de dst (seq), -1 */
    int32_t t, src, dst;
    int32_t chain_start; /* minute de début de la chaîne de passage */
    float usd_out, usd_in;
    float score;        /* score de l'étape 1 de la transaction (0 sans forêt) */
    uint8_t fmt, depth; /* profondeur de chaîne (0 = pas de passage apparié) */
} edge_t;

typedef struct {
    uint64_t key; /* 0 = vide */
    int32_t last_t;
    uint32_t cnt;
} pair_t;

typedef struct { /* ensemble d'entiers sans remise à zéro (génération) + liste d'insertion */
    int32_t *key;
    uint32_t *gen;
    int32_t *list;
    uint32_t cur;
    int32_t mask, n;
} iset;

typedef struct { /* état d'un compte : 136 octets contigus (≈ 2 lignes de cache) */
    int32_t first_t, last_out_t, last_in_t, last_upd_t;
    float ew[N_SCALES * 4]; /* [échelle*4 + {nb_out, usd_out, nb_in, usd_in}] */
    int32_t n_out, n_in, n_self, n_ach_out, n_ach_in, dist_out, dist_in;
    float mu_out, m2_out, mu_in, m2_in;
    int64_t head_out, head_in; /* têtes des listes d'adjacence (seq), -1 */
    float susp_out, susp_in;   /* somme décroissante (96 h) des scores envoyés / reçus */
} acct_t;

/* Nœud compact (12 octets) en ordre préfixe : le fils gauche est le nœud suivant, une visite touche une
 * seule ligne de cache. Seuil float32 arrondi vers le bas : pour une entrée float32, x <= seuil32 ⇔
 * x <= seuil double (décision identique à LightGBM). Pour une feuille : feat = -1, right = index de valeur. */
typedef struct {
    float thr;
    int32_t right;
    int16_t feat;
    uint8_t dl;
    uint8_t pad;
} cnode_t;

typedef struct {
    int32_t n_trees, n_nodes;
    int32_t *root;
    cnode_t *nodes;
    double *leaf; /* valeurs de feuilles en double : somme identique à LightGBM */
} cforest;

typedef struct { /* KYC statique d'un compte */
    int32_t bank, entity, nacc;
    int16_t country;
    uint8_t etype;
} kyc_t;

struct aml_engine {
    aml_config cfg;
    int32_t n_acc;
    int64_t bytes;
    kyc_t *kyc;
    acct_t *acc;
    /* anneau d'arêtes du graphe de motifs */
    edge_t *ring;
    int64_t ring_cap, ring_mask, seq_next;
    /* table des paires */
    pair_t *pairs;
    int64_t pair_cap, pair_mask;
    /* décroissance exponentielle tabulée par minute */
    float *decay[N_SCALES];
    int32_t decay_len;
    /* ensembles de travail */
    iset s_out_src, s_in_src, s_in_dst, s_out_dst, s_tmp;
    int64_t stat[ST_COUNT];
    /* forêt attachée (forme compacte possédée) et instrumentation des latences */
    cforest forest;
    int has_forest;
    int64_t forest_bytes;
    /* cascade (Viola-Jones) : un mini-modèle écarte d'abord les transactions manifestement licites */
    cforest pre;
    int has_pre;
    double pre_raw_cut; /* score brut du mini-modèle sous lequel on n'évalue pas la forêt complète */
    int64_t pre_bytes;
    int64_t *timing;
    int64_t *seq_out; /* seq_next avant chaque transaction (repère « après » pour la revue J+1) */
};

/* ---- utilitaires ------------------------------------------------------------------------------- */
static void *xcalloc(aml_engine *e, size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz);
    if (p) e->bytes += (int64_t)(n * sz);
    return p;
}

static inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t pair_key(int32_t a, int32_t b) {
    return ((uint64_t)(uint32_t)(a + 1) << 32) | (uint32_t)(b + 1);
}

static int iset_init(aml_engine *e, iset *s, int32_t max_items) {
    int32_t cap = 16;
    while (cap < 4 * max_items) cap <<= 1;
    s->key = xcalloc(e, (size_t)cap, sizeof(int32_t));
    s->gen = xcalloc(e, (size_t)cap, sizeof(uint32_t));
    s->list = xcalloc(e, (size_t)cap, sizeof(int32_t));
    s->cur = 1;
    s->mask = cap - 1;
    s->n = 0;
    return s->key && s->gen && s->list;
}

static void iset_free(iset *s) {
    free(s->key);
    free(s->gen);
    free(s->list);
}

static inline void iset_clear(iset *s) {
    s->n = 0;
    if (++s->cur == 0) {
        memset(s->gen, 0, (size_t)(s->mask + 1) * sizeof(uint32_t));
        s->cur = 1;
    }
}

static inline int iset_add(iset *s, int32_t k) { /* 1 si nouveau */
    uint32_t h = ((uint32_t)k * 2654435761u) & (uint32_t)s->mask;
    while (s->gen[h] == s->cur) {
        if (s->key[h] == k) return 0;
        h = (h + 1) & (uint32_t)s->mask;
    }
    if (s->n >= s->mask / 2) return 0; /* plein : on ignore (ne se produit pas avec les plafonds) */
    s->gen[h] = s->cur;
    s->key[h] = k;
    s->list[s->n++] = k;
    return 1;
}

static inline int iset_has(const iset *s, int32_t k) {
    uint32_t h = ((uint32_t)k * 2654435761u) & (uint32_t)s->mask;
    while (s->gen[h] == s->cur) {
        if (s->key[h] == k) return 1;
        h = (h + 1) & (uint32_t)s->mask;
    }
    return 0;
}

static inline edge_t *edge_at(const aml_engine *e, int64_t s) {
    if (s < 0 || s < e->seq_next - e->ring_cap) return NULL;
    return &e->ring[s & e->ring_mask];
}

static inline float decay(const aml_engine *e, int k, int32_t dt) {
    if (dt <= 0) return 1.f;
    if (dt < e->decay_len) return e->decay[k][dt];
    return 0.f;
}

/* ---- table des paires -------------------------------------------------------------------------- */
static pair_t *pair_find(const aml_engine *e, uint64_t key) {
    uint64_t i = mix64(key) & (uint64_t)e->pair_mask;
    while (e->pairs[i].key) {
        if (e->pairs[i].key == key) return &e->pairs[i];
        i = (i + 1) & (uint64_t)e->pair_mask;
    }
    return NULL;
}

static int pair_rebuild(aml_engine *e, int32_t t_min) {
    pair_t *old = e->pairs, *nw = calloc((size_t)e->pair_cap, sizeof(pair_t));
    if (!nw) return -1;
    int64_t kept = 0;
    for (int64_t j = 0; j < e->pair_cap; j++) {
        if (!old[j].key) continue;
        if (old[j].last_t < t_min) {
            e->stat[ST_pair_evicted]++;
            continue;
        }
        uint64_t i = mix64(old[j].key) & (uint64_t)e->pair_mask;
        while (nw[i].key) i = (i + 1) & (uint64_t)e->pair_mask;
        nw[i] = old[j];
        kept++;
    }
    free(old);
    e->pairs = nw;
    e->stat[ST_pair_used] = kept;
    return 0;
}

static pair_t *pair_upsert(aml_engine *e, uint64_t key, int32_t t, int *is_new) {
    pair_t *p = pair_find(e, key);
    *is_new = 0;
    if (p) return p;
    if (e->stat[ST_pair_used] + 1 > (e->pair_cap * 7) / 10) {
        int32_t ttl = e->cfg.pair_ttl;
        do { /* éviction des paires inactives, en resserrant le délai si nécessaire */
            if (pair_rebuild(e, t - ttl) != 0) return NULL;
            ttl /= 2;
        } while (e->stat[ST_pair_used] > e->pair_cap / 2 && ttl >= 60);
    }
    uint64_t i = mix64(key) & (uint64_t)e->pair_mask;
    while (e->pairs[i].key) i = (i + 1) & (uint64_t)e->pair_mask;
    e->pairs[i].key = key;
    e->pairs[i].cnt = 0;
    e->pairs[i].last_t = t;
    e->stat[ST_pair_used]++;
    *is_new = 1;
    return &e->pairs[i];
}

/* ---- création / destruction -------------------------------------------------------------------- */
int32_t aml_n_features(void) { return F_COUNT; }
const char *aml_feature_name(int32_t i) { return (i >= 0 && i < F_COUNT) ? FEATURE_NAMES[i] : NULL; }
int32_t aml_n_stats(void) { return ST_COUNT; }
const char *aml_stat_name(int32_t i) { return (i >= 0 && i < ST_COUNT) ? STAT_NAMES[i] : NULL; }
int64_t aml_state_bytes(const aml_engine *e) { return e->bytes; }
void aml_stats(const aml_engine *e, int64_t *out) { memcpy(out, e->stat, sizeof(e->stat)); }

void aml_free(aml_engine *e) {
    if (!e) return;
    free(e->kyc);
    free(e->acc);
    free(e->ring);
    free(e->pairs);
    if (e->has_forest) {
        free(e->forest.root);
        free(e->forest.nodes);
        free(e->forest.leaf);
    }
    if (e->has_pre) {
        free(e->pre.root);
        free(e->pre.nodes);
        free(e->pre.leaf);
    }
    for (int k = 0; k < N_SCALES; k++) free(e->decay[k]);
    iset_free(&e->s_out_src);
    iset_free(&e->s_in_src);
    iset_free(&e->s_in_dst);
    iset_free(&e->s_out_dst);
    iset_free(&e->s_tmp);
    free(e);
}

aml_engine *aml_new(const aml_config *cfg, const aml_kyc *kyc) {
    if (!cfg || cfg->n_accounts <= 0 || cfg->ring_bits < 4 || cfg->ring_bits > 30 ||
        cfg->pair_bits < 4 || cfg->pair_bits > 31 || cfg->scan_cap <= 0 || cfg->fan_cap <= 0 ||
        cfg->cycle_max_len < 2 || cfg->cycle_max_len > MAX_PATH - 2)
        return NULL;
    aml_engine *e = calloc(1, sizeof(aml_engine));
    if (!e) return NULL;
    e->cfg = *cfg;
    int32_t n = cfg->n_accounts;
    e->n_acc = n;
    size_t un = (size_t)n;
    int ok = 1;
#define ALLOC(ptr, count) ok &= ((ptr) = xcalloc(e, (count), sizeof(*(ptr)))) != NULL
    ALLOC(e->kyc, un);
    ALLOC(e->acc, un);
    e->ring_cap = (int64_t)1 << cfg->ring_bits;
    e->ring_mask = e->ring_cap - 1;
    ALLOC(e->ring, (size_t)e->ring_cap);
    e->pair_cap = (int64_t)1 << cfg->pair_bits;
    e->pair_mask = e->pair_cap - 1;
    ALLOC(e->pairs, (size_t)e->pair_cap);
    e->decay_len = 8 * cfg->win_long;
    int32_t taus[N_SCALES] = {cfg->win_short, cfg->win_mid, cfg->win_long};
    for (int k = 0; k < N_SCALES; k++) {
        ALLOC(e->decay[k], (size_t)e->decay_len);
        if (ok)
            for (int32_t dt = 0; dt < e->decay_len; dt++)
                e->decay[k][dt] = (float)exp(-(double)dt / (double)taus[k]);
    }
#undef ALLOC
    int32_t m = cfg->scan_cap > cfg->fan_cap ? cfg->scan_cap : cfg->fan_cap;
    ok &= iset_init(e, &e->s_out_src, m) && iset_init(e, &e->s_in_src, m) &&
          iset_init(e, &e->s_in_dst, m) && iset_init(e, &e->s_out_dst, m) &&
          iset_init(e, &e->s_tmp, m);
    if (!ok) {
        aml_free(e);
        return NULL;
    }
    for (int32_t a = 0; a < n; a++) {
        acct_t *c = &e->acc[a];
        c->first_t = c->last_out_t = c->last_in_t = c->last_upd_t = NONE_T;
        c->head_out = c->head_in = -1;
        kyc_t *k = &e->kyc[a];
        k->bank = kyc && kyc->bank ? kyc->bank[a] : 0;
        k->entity = kyc && kyc->entity ? kyc->entity[a] : -1;
        k->nacc = kyc && kyc->nacc ? kyc->nacc[a] : 1;
        k->etype = kyc && kyc->etype ? kyc->etype[a] : 0;
        k->country = kyc && kyc->country ? kyc->country[a] : 0;
    }
    for (int64_t j = 0; j < e->ring_cap; j++) e->ring[j].seq = -1;
    return e;
}

/* ---- profils de compte ------------------------------------------------------------------------- */
static void touch(aml_engine *e, int32_t a, int32_t t) { /* décroissance paresseuse jusqu'à t */
    acct_t *c = &e->acc[a];
    if (c->last_upd_t == NONE_T) {
        c->last_upd_t = t;
        return;
    }
    int32_t dt = t - c->last_upd_t;
    if (dt <= 0) return;
    for (int k = 0; k < N_SCALES; k++) {
        float d = decay(e, k, dt);
        for (int q = 0; q < 4; q++) c->ew[k * 4 + q] *= d;
    }
    const float dl = decay(e, N_SCALES - 1, dt);
    c->susp_out *= dl;
    c->susp_in *= dl;
    c->last_upd_t = t;
}

static inline float zscore(int32_t n, float mu, float m2, float x) {
    if (n < 2) return 0.f;
    float z = (x - mu) / sqrtf(m2 / (float)(n - 1) + 1e-3f);
    return z > 50.f ? 50.f : (z < -50.f ? -50.f : z);
}

static void profile(const aml_engine *e, int32_t a, int32_t t, float lx, float *p) {
    const acct_t *c = &e->acc[a];
    p[P_age] = c->first_t == NONE_T ? -1.f : (float)(t - c->first_t);
    p[P_dt_out] = c->last_out_t == NONE_T ? -1.f : (float)(t - c->last_out_t);
    p[P_dt_in] = c->last_in_t == NONE_T ? -1.f : (float)(t - c->last_in_t);
    p[P_n_out] = (float)c->n_out;
    p[P_n_in] = (float)c->n_in;
    for (int k = 0; k < N_SCALES; k++) {
        p[P_ewc_out + k] = c->ew[k * 4 + 0];
        p[P_ewa_out + k] = log1pf(c->ew[k * 4 + 1]);
        p[P_ewc_in + k] = c->ew[k * 4 + 2];
        p[P_ewa_in + k] = log1pf(c->ew[k * 4 + 3]);
    }
    p[P_z_out] = zscore(c->n_out, c->mu_out, c->m2_out, lx);
    p[P_z_in] = zscore(c->n_in, c->mu_in, c->m2_in, lx);
    p[P_ach_out] = c->n_out ? (float)c->n_ach_out / (float)c->n_out : -1.f;
    p[P_ach_in] = c->n_in ? (float)c->n_ach_in / (float)c->n_in : -1.f;
    p[P_dist_out] = (float)c->dist_out;
    p[P_dist_in] = (float)c->dist_in;
    const int L = (N_SCALES - 1) * 4;
    p[P_flow] = logf((c->ew[L + 1] + 1.f) / (c->ew[L + 3] + 1.f));
    p[P_n_self] = (float)c->n_self;
}

static inline void welford(int32_t *n, float *mu, float *m2, float x) {
    (*n)++;
    float d = x - *mu;
    *mu += d / (float)(*n);
    *m2 += d * (x - *mu);
}

static float roundness(double amt) { /* nombre de zéros finaux du montant en centimes */
    double c = floor(amt * 100.0 + 0.5);
    if (c <= 0.0) return 0.f;
    int z = 0;
    while (z < 14 && fmod(c, 10.0) == 0.0) {
        c /= 10.0;
        z++;
    }
    return (float)z;
}

/* ---- cycles temporels : DFS bornée (chemin dst → … → src, temps croissants, avant t) ----------- */
typedef struct {
    int32_t target, start;
    float x, lo, hi;
    int guided, max_len, found, min_len;
    float drift;
    int64_t budget;
    int truncated;
    int32_t path[MAX_PATH];
    int plen;
} dfs_ctx;

static void dfs(aml_engine *e, dfs_ctx *c, int32_t node, int32_t t_prev, int depth, float first_amt) {
    int64_t s = e->acc[node].head_out;
    for (int k = 0; k < e->cfg.scan_cap && s >= 0; k++) {
        edge_t *ed = edge_at(e, s);
        if (!ed) {
            e->stat[ST_recycled_links]++;
            return;
        }
        if (ed->t < t_prev) return; /* liste par temps décroissant : le reste est plus ancien */
        s = ed->prev_out;
        if (--c->budget < 0) {
            c->truncated = 1;
            return;
        }
        e->stat[ST_dfs_expansions]++;
        if (c->guided && (ed->usd_out < c->lo || ed->usd_out > c->hi)) continue;
        float fa = depth == 0 ? ed->usd_out : first_amt;
        int32_t y = ed->dst;
        if (y == c->target) {
            int len = depth + 2;
            c->found++;
            if (c->min_len == 0 || len < c->min_len) {
                c->min_len = len;
                c->drift = fa > 0.f ? fabsf(c->x / fa - 1.f) : NO_MATCH;
            }
            if (c->found >= CYCLE_FOUND_CAP) {
                c->budget = -1;
                return;
            }
            continue;
        }
        if (depth + 3 > c->max_len || y == c->start) continue;
        int on_path = 0;
        for (int j = 0; j < c->plen; j++)
            if (c->path[j] == y) {
                on_path = 1;
                break;
            }
        if (on_path) continue;
        c->path[c->plen++] = y;
        dfs(e, c, y, ed->t, depth + 1, fa);
        c->plen--;
        if (c->budget < 0) return;
    }
}

static void cycles(aml_engine *e, int32_t a, int32_t d, int32_t t, float x, int guided, int max_len,
                   int64_t budget, int *found, int *min_len, float *drift) {
    dfs_ctx c;
    memset(&c, 0, sizeof(c));
    c.target = a;
    c.start = d;
    c.x = x;
    c.lo = x / 1.3f;
    c.hi = x * 1.3f;
    c.guided = guided;
    c.max_len = max_len;
    c.drift = NO_MATCH;
    c.budget = budget;
    c.path[c.plen++] = d;
    dfs(e, &c, d, t - e->cfg.win_long, 0, 0.f);
    if (c.truncated) e->stat[ST_cycle_budget_hits]++;
    *found = c.found;
    *min_len = c.min_len;
    *drift = c.drift;
}

/* ---- calcul des features d'une transaction ---------------------------------------------------- */
typedef struct {
    uint8_t depth;
    int32_t chain_start;
} chain_info;

static void features(aml_engine *e, const aml_batch *b, int64_t i, float *f, chain_info *ch) {
    const int32_t t = b->t[i], a = b->src[i], d = b->dst[i];
    const float xp = b->usd_paid[i];
    const float lx = log1pf(xp);
    const uint8_t fm = b->fmt[i];
    const int self = (a == d);
    const int in_mask = !self && ((e->cfg.motif_fmt_mask >> fm) & 1);
    const int32_t t_l = t - e->cfg.win_long, t_m = t - e->cfg.win_mid;
    const int cap = e->cfg.scan_cap, fcap = e->cfg.fan_cap;

    touch(e, a, t);
    if (!self) touch(e, d, t);

    /* F0 */
    f[F_tx_log_usd] = lx;
    f[F_tx_fmt] = (float)fm;
    f[F_tx_is_ach] = (float)(fm == e->cfg.ach_code);
    f[F_tx_cur] = (float)b->cur_paid[i];
    f[F_tx_xcur] = (float)(b->cur_paid[i] != b->cur_recv[i]);
    f[F_tx_self] = (float)self;
    const kyc_t *ka = &e->kyc[a], *kd = &e->kyc[d];
    f[F_tx_same_bank] = (float)(ka->bank == kd->bank);
    f[F_tx_hour] = (float)((t % 1440) / 60);
    f[F_tx_round] = roundness(b->amt_paid[i]);
    /* F1 */
    f[F_kyc_s_etype] = (float)ka->etype;
    f[F_kyc_r_etype] = (float)kd->etype;
    f[F_kyc_s_nacc] = (float)ka->nacc;
    f[F_kyc_r_nacc] = (float)kd->nacc;
    f[F_kyc_same_entity] = (float)(!self && ka->entity >= 0 && ka->entity == kd->entity);
    f[F_kyc_s_country] = (float)ka->country;
    f[F_kyc_r_country] = (float)kd->country;
    f[F_kyc_cross_border] = (float)(ka->country != kd->country);
    /* F2 */
    profile(e, a, t, lx, f + F_s_age);
    profile(e, d, t, lx, f + F_r_age);

    /* valeurs par défaut des familles relationnelles */
    f[F_p_cnt] = 0.f;
    f[F_p_dt] = -1.f;
    f[F_p_rev_cnt] = 0.f;
    f[F_p_rev_dt] = -1.f;
    for (int k = F_g_s_outdeg_m; k <= F_g_r_outdeg_l; k++) f[k] = 0.f;
    f[F_f_best_err] = NO_MATCH;
    f[F_f_best_lag] = -1.f;
    f[F_f_n_match2] = f[F_f_n_match10] = f[F_f_chain_depth] = 0.f;
    f[F_f_chain_age] = -1.f;
    f[F_f_in_sum_l] = f[F_f_out_sum_l] = f[F_f_split] = 0.f;
    f[F_f_r_ret_err] = NO_MATCH;
    f[F_m_cyc_n] = f[F_m_cyc_minlen] = f[F_m_cyc4_n] = 0.f;
    f[F_m_cyc_drift] = NO_MATCH;
    f[F_m_sg] = f[F_m_2hop] = f[F_m_bip] = 0.f;
    f[F_z_s_in_max] = f[F_z_s_out_max] = f[F_z_r_in_max] = f[F_z_r_out_max] = 0.f;
    f[F_z_s_susp_out] = e->acc[a].susp_out;
    f[F_z_s_susp_in] = e->acc[a].susp_in;
    f[F_z_r_susp_out] = e->acc[d].susp_out;
    f[F_z_r_susp_in] = e->acc[d].susp_in;
    float zso = 0.f, zsi = 0.f, zri = 0.f, zro = 0.f;
    ch->depth = 0;
    ch->chain_start = t;
    if (self) return;

    /* F3 */
    const pair_t *pp = pair_find(e, pair_key(a, d));
    if (pp) {
        f[F_p_cnt] = (float)pp->cnt;
        f[F_p_dt] = (float)(t - pp->last_t);
    }
    const pair_t *pr = pair_find(e, pair_key(d, a));
    if (pr) {
        f[F_p_rev_cnt] = (float)pr->cnt;
        f[F_p_rev_dt] = (float)(t - pr->last_t);
    }

    /* F4 + F6 : arêtes sortantes de src (fan-out, volume déjà sorti) */
    double out_sum = 0.0, in_sum = 0.0;
    iset *so = &e->s_out_src, *si = &e->s_in_src, *sd = &e->s_in_dst, *sdo = &e->s_out_dst;
    iset_clear(so);
    int outcnt = 0, outdeg_m = 0, k = 0;
    int64_t s = e->acc[a].head_out;
    for (; k < cap && s >= 0; k++) {
        edge_t *ed = edge_at(e, s);
        if (!ed) {
            e->stat[ST_recycled_links]++;
            break;
        }
        if (ed->t < t_l) break;
        outcnt++;
        out_sum += ed->usd_out;
        if (ed->score > zso) zso = ed->score;
        if (iset_add(so, ed->dst) && ed->t >= t_m) outdeg_m++;
        s = ed->prev_out;
    }
    if (k == cap) e->stat[ST_scan_cap_hits]++;
    f[F_g_s_outdeg_m] = (float)outdeg_m;
    f[F_g_s_outdeg_l] = (float)so->n;
    f[F_g_s_outcnt_l] = (float)outcnt;

    /* arêtes entrantes de src : fan-in de l'émetteur + appariement de montants (passage de fonds) */
    iset_clear(si);
    float best = NO_MATCH;
    const edge_t *best_ed = NULL;
    int n2 = 0, n10 = 0;
    s = e->acc[a].head_in;
    for (k = 0; k < cap && s >= 0; k++) {
        edge_t *ed = edge_at(e, s);
        if (!ed) {
            e->stat[ST_recycled_links]++;
            break;
        }
        if (ed->t < t_l) break;
        iset_add(si, ed->src);
        in_sum += ed->usd_in;
        if (ed->score > zsi) zsi = ed->score;
        if (ed->usd_in > 0.f) {
            float err = fabsf(xp / ed->usd_in - 1.f);
            if (err < best) {
                best = err;
                best_ed = ed;
            }
            n2 += err <= 0.02f;
            n10 += err <= 0.10f;
        }
        s = ed->prev_in;
    }
    if (k == cap) e->stat[ST_scan_cap_hits]++;
    f[F_g_s_indeg_l] = (float)si->n;
    f[F_f_best_err] = best;
    if (best_ed) {
        f[F_f_best_lag] = (float)(t - best_ed->t);
        if (best <= e->cfg.chain_tol) {
            int dep = best_ed->depth + 1;
            ch->depth = (uint8_t)(dep > 255 ? 255 : dep);
            ch->chain_start = best_ed->chain_start;
            f[F_f_chain_depth] = (float)ch->depth;
            f[F_f_chain_age] = (float)(t - best_ed->chain_start);
        }
    }
    f[F_f_n_match2] = (float)n2;
    f[F_f_n_match10] = (float)n10;
    f[F_f_in_sum_l] = (float)log1p(in_sum);
    f[F_f_out_sum_l] = (float)log1p(out_sum);
    f[F_f_split] = (float)log((out_sum + xp + 1.0) / (in_sum + 1.0));

    /* arêtes entrantes de dst : fan-in du destinataire (ensemble H réutilisé par les motifs) */
    iset_clear(sd);
    int incnt = 0, indeg_m = 0;
    s = e->acc[d].head_in;
    for (k = 0; k < cap && s >= 0; k++) {
        edge_t *ed = edge_at(e, s);
        if (!ed) {
            e->stat[ST_recycled_links]++;
            break;
        }
        if (ed->t < t_l) break;
        incnt++;
        if (iset_add(sd, ed->src) && ed->t >= t_m) indeg_m++;
        if (ed->score > zri) zri = ed->score;
        s = ed->prev_in;
    }
    if (k == cap) e->stat[ST_scan_cap_hits]++;
    f[F_g_r_indeg_m] = (float)indeg_m;
    f[F_g_r_indeg_l] = (float)sd->n;
    f[F_g_r_incnt_l] = (float)incnt;

    /* arêtes sortantes de dst : fan-out du destinataire, retour d'un montant similaire */
    iset_clear(sdo);
    float ret = NO_MATCH;
    s = e->acc[d].head_out;
    for (k = 0; k < cap && s >= 0; k++) {
        edge_t *ed = edge_at(e, s);
        if (!ed) {
            e->stat[ST_recycled_links]++;
            break;
        }
        if (ed->t < t_l) break;
        iset_add(sdo, ed->dst);
        if (ed->score > zro) zro = ed->score;
        if (ed->usd_out > 0.f) {
            float err = fabsf(xp / ed->usd_out - 1.f);
            if (err < ret) ret = err;
        }
        s = ed->prev_out;
    }
    if (k == cap) e->stat[ST_scan_cap_hits]++;
    f[F_g_r_outdeg_l] = (float)sdo->n;
    f[F_f_r_ret_err] = ret;
    f[F_z_s_in_max] = zsi;
    f[F_z_s_out_max] = zso;
    f[F_z_r_in_max] = zri;
    f[F_z_r_out_max] = zro;

    if (!in_mask) return;

    /* F5 : chemins à deux sauts src → w → dst */
    int two = 0;
    for (int j = 0; j < so->n; j++) {
        int32_t w = so->list[j];
        if (w != d && w != a && iset_has(sd, w)) two++;
    }
    f[F_m_2hop] = (float)two;

    /* scatter-gather (cette arête = collecte) : sources s → src, s → w', w' → dst */
    int sg = 0;
    iset *tmp = &e->s_tmp;
    for (int j = 0; j < si->n && j < fcap; j++) {
        int32_t src0 = si->list[j];
        iset_clear(tmp);
        s = e->acc[src0].head_out;
        for (k = 0; k < fcap && s >= 0; k++) {
            edge_t *ed = edge_at(e, s);
            if (!ed || ed->t < t_l) break;
            int32_t w = ed->dst;
            if (w != a && w != d && iset_has(sd, w) && iset_add(tmp, w)) sg++;
            s = ed->prev_out;
        }
    }
    f[F_m_sg] = (float)sg;

    /* recouvrement biparti : autres bénéficiaires r de src qui partagent des émetteurs avec dst */
    int bip = 0;
    for (int j = 0; j < so->n && j < fcap; j++) {
        int32_t r = so->list[j];
        if (r == d) continue;
        iset_clear(tmp);
        s = e->acc[r].head_in;
        for (k = 0; k < fcap && s >= 0; k++) {
            edge_t *ed = edge_at(e, s);
            if (!ed || ed->t < t_l) break;
            int32_t s2 = ed->src;
            if (s2 != a && s2 != d && iset_has(sd, s2) && iset_add(tmp, s2)) bip++;
            s = ed->prev_in;
        }
    }
    f[F_m_bip] = (float)bip;

    /* cycles temporels fermés par cette arête : guidés par le montant (±30 %) puis libres (≤ 4) */
    int found, minlen;
    float drift;
    cycles(e, a, d, t, xp, 1, e->cfg.cycle_max_len, e->cfg.cycle_budget, &found, &minlen, &drift);
    f[F_m_cyc_n] = (float)found;
    f[F_m_cyc_minlen] = (float)minlen;
    f[F_m_cyc_drift] = drift;
    cycles(e, a, d, t, xp, 0, 4, e->cfg.cycle_budget / 4, &found, &minlen, &drift);
    f[F_m_cyc4_n] = (float)found;
}

/* ---- mise à jour de l'état --------------------------------------------------------------------- */
static void update(aml_engine *e, const aml_batch *b, int64_t i, const chain_info *ch, float score) {
    const int32_t t = b->t[i], a = b->src[i], d = b->dst[i];
    const uint8_t fm = b->fmt[i];
    acct_t *ca = &e->acc[a], *cd = &e->acc[d];
    if (ca->first_t == NONE_T) ca->first_t = t;
    if (cd->first_t == NONE_T) cd->first_t = t;
    if (a == d) {
        ca->n_self++;
        return;
    }
    const float xp = b->usd_paid[i], xr = b->usd_recv[i];
    for (int k = 0; k < N_SCALES; k++) {
        ca->ew[k * 4 + 0] += 1.f;
        ca->ew[k * 4 + 1] += xp;
        cd->ew[k * 4 + 2] += 1.f;
        cd->ew[k * 4 + 3] += xr;
    }
    ca->susp_out += score;
    cd->susp_in += score;
    welford(&ca->n_out, &ca->mu_out, &ca->m2_out, log1pf(xp));
    welford(&cd->n_in, &cd->mu_in, &cd->m2_in, log1pf(xr));
    if (fm == e->cfg.ach_code) {
        ca->n_ach_out++;
        cd->n_ach_in++;
    }
    ca->last_out_t = t;
    cd->last_in_t = t;

    int is_new;
    pair_t *p = pair_upsert(e, pair_key(a, d), t, &is_new);
    if (p) {
        p->cnt++;
        p->last_t = t;
        if (is_new) {
            ca->dist_out++;
            cd->dist_in++;
        }
    }

    if (!((e->cfg.motif_fmt_mask >> fm) & 1)) return;
    int64_t seq = e->seq_next++;
    edge_t *ed = &e->ring[seq & e->ring_mask];
    if (ed->seq >= 0 && ed->t >= t - e->cfg.win_long) e->stat[ST_ring_overflow]++;
    ed->seq = seq;
    ed->t = t;
    ed->src = a;
    ed->dst = d;
    ed->usd_out = xp;
    ed->usd_in = xr;
    ed->fmt = fm;
    ed->depth = ch->depth;
    ed->score = score;
    ed->chain_start = ch->chain_start;
    ed->prev_out = ca->head_out;
    ed->prev_in = cd->head_in;
    ca->head_out = seq;
    cd->head_in = seq;
    e->stat[ST_edges]++;
}

/* ---- forêt d'arbres ------------------------------------------------------------------------------ */
static double forest_raw1(const cforest *f, const float *x) {
    double raw = 0.0;
    for (int32_t k = 0; k < f->n_trees; k++) {
        int32_t n = f->root[k];
        const cnode_t *nd = &f->nodes[n];
        while (nd->feat >= 0) {
            float v = x[nd->feat];
            int go_left = (v <= nd->thr) || (isnan(v) && nd->dl);
            n = go_left ? n + 1 : nd->right;
            nd = &f->nodes[n];
        }
        raw += f->leaf[nd->right];
    }
    return raw;
}

static void cforest_free(cforest *c) {
    free(c->root);
    free(c->nodes);
    free(c->leaf);
    memset(c, 0, sizeof(*c));
}

/* Conversion vers la forme compacte ; exige l'ordre préfixe (gauche = nœud suivant). */
static int cforest_build(const aml_forest *f, cforest *c) {
    memset(c, 0, sizeof(*c));
    size_t nt = (size_t)f->n_trees, nn = (size_t)f->n_nodes;
    c->n_trees = f->n_trees;
    c->n_nodes = f->n_nodes;
    c->root = malloc(nt * sizeof(int32_t));
    c->nodes = malloc(nn * sizeof(cnode_t));
    c->leaf = malloc(nn * sizeof(double));
    if (!c->root || !c->nodes || !c->leaf) {
        cforest_free(c);
        return -3;
    }
    memcpy(c->root, f->root, nt * sizeof(int32_t));
    int32_t n_leaf = 0;
    for (size_t j = 0; j < nn; j++) {
        cnode_t *nd = &c->nodes[j];
        nd->pad = 0;
        if (f->feat[j] < 0) {
            nd->feat = -1;
            nd->thr = 0.f;
            nd->dl = 0;
            nd->right = n_leaf;
            c->leaf[n_leaf++] = f->value[j];
            continue;
        }
        if (f->left[j] != (int32_t)j + 1 || f->feat[j] > 32767) {
            cforest_free(c);
            return -4; /* ordre préfixe requis */
        }
        float t = (float)f->thr[j];
        if ((double)t > f->thr[j]) t = nextafterf(t, -INFINITY); /* plus grand float32 <= seuil */
        nd->thr = t;
        nd->feat = (int16_t)f->feat[j];
        nd->dl = f->default_left[j];
        nd->right = f->right[j];
    }
    return 0;
}

void aml_forest_raw(const aml_forest *f, const float *x, int64_t n, int32_t stride, double *out) {
    cforest c;
    if (cforest_build(f, &c) != 0) {
        for (int64_t i = 0; i < n; i++) out[i] = NAN;
        return;
    }
    for (int64_t i = 0; i < n; i++) out[i] = forest_raw1(&c, x + i * (int64_t)stride);
    cforest_free(&c);
}

int aml_set_forest(aml_engine *e, const aml_forest *f) {
    if (!e || !f || f->n_trees <= 0 || f->n_nodes <= 0) return -1;
    for (int32_t j = 0; j < f->n_nodes; j++)
        if (f->feat[j] >= F_COUNT) return -2; /* forêt incompatible avec ce moteur */
    cforest c;
    int rc = cforest_build(f, &c);
    if (rc != 0) return rc;
    if (e->has_forest) {
        cforest_free(&e->forest);
        e->bytes -= e->forest_bytes;
    }
    e->forest = c;
    e->has_forest = 1;
    e->forest_bytes = (int64_t)((size_t)c.n_trees * 4 + (size_t)c.n_nodes * (sizeof(cnode_t) + 8));
    e->bytes += e->forest_bytes;
    return 0;
}

int aml_set_cascade(aml_engine *e, const aml_forest *pre, double raw_cut) {
    if (!e) return -1;
    if (e->has_pre) {
        cforest_free(&e->pre);
        e->bytes -= e->pre_bytes;
        e->has_pre = 0;
    }
    if (!pre) return 0; /* désactivation */
    for (int32_t j = 0; j < pre->n_nodes; j++)
        if (pre->feat[j] >= F_COUNT) return -2;
    int rc = cforest_build(pre, &e->pre);
    if (rc != 0) return rc;
    e->has_pre = 1;
    e->pre_raw_cut = raw_cut;
    e->pre_bytes = (int64_t)((size_t)e->pre.n_trees * 4 + (size_t)e->pre.n_nodes * (sizeof(cnode_t) + 8));
    e->bytes += e->pre_bytes;
    return 0;
}

void aml_set_timing(aml_engine *e, int64_t *ns_per_tx) { e->timing = ns_per_tx; }
void aml_set_seq_out(aml_engine *e, int64_t *seq_at) { e->seq_out = seq_at; }

static inline int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int aml_process_scored(aml_engine *e, const aml_batch *b, float *out, double *scores) {
    if (!e || !b || !out) return -1;
    if (scores && !e->has_forest) return -3;
    for (int64_t i = 0; i < b->n; i++) {
        if (b->src[i] < 0 || b->src[i] >= e->n_acc || b->dst[i] < 0 || b->dst[i] >= e->n_acc ||
            b->fmt[i] >= 32)
            return -2;
        int64_t t0 = e->timing ? now_ns() : 0;
        if (e->seq_out) e->seq_out[i] = e->seq_next;
        chain_info ch;
        float *x = out + i * (int64_t)F_COUNT;
        features(e, b, i, x, &ch);
        /* score de l'étape 1 calculé avant la mise à jour : il ne sert qu'aux transactions suivantes */
        double sc = 0.0;
        if (e->has_forest) {
            double r0 = e->has_pre ? forest_raw1(&e->pre, x) : 0.0;
            if (e->has_pre && r0 < e->pre_raw_cut) {
                sc = 1.0 / (1.0 + exp(-r0)); /* écartée par la cascade : score du mini-modèle (bas) */
                e->stat[ST_cascade_skips]++;
            } else {
                sc = 1.0 / (1.0 + exp(-forest_raw1(&e->forest, x)));
            }
        }
        if (scores) scores[i] = sc;
        update(e, b, i, &ch, (float)sc);
        e->stat[ST_tx]++;
        if (e->timing) e->timing[i] = now_ns() - t0;
    }
    return 0;
}

int aml_process(aml_engine *e, const aml_batch *b, float *out) {
    return aml_process_scored(e, b, out, NULL);
}

/* ---- revue J+1 : features rétrospectives -----------------------------------------------------------
 * Pour une transaction déjà traitée, recalculées à l'instant de décision avec l'état courant : ce qui
 * s'est passé APRÈS elle (et avant la décision). Les arêtes « après » ont seq >= seq_at, hors l'arête
 * de la transaction elle-même. Aucune donnée postérieure à la décision n'est dans l'état.
 */
#define HINDSIGHT(X)                                                                               \
    X(h_s_out_after) X(h_s_outcnt_after) X(h_r_in_after) X(h_r_out_after) X(h_r_fwd_err)           \
    X(h_r_fwd_lag) X(h_r_fwd_n10) X(h_r_split) X(h_cyc_n) X(h_cyc_minlen) X(h_sg_scatter)          \
    X(h_s_in_after)
enum {
#define X(name) H_##name,
    HINDSIGHT(X)
#undef X
        H_COUNT
};
static const char *HINDSIGHT_NAMES[] = {
#define X(name) #name,
    HINDSIGHT(X)
#undef X
};
int32_t aml_n_hindsight(void) { return H_COUNT; }
const char *aml_hindsight_name(int32_t i) { return (i >= 0 && i < H_COUNT) ? HINDSIGHT_NAMES[i] : NULL; }

/* DFS statique (sans ordre temporel) dans la fenêtre [t_lo, maintenant], en évitant l'arête own */
static void sdfs(aml_engine *e, dfs_ctx *c, int32_t node, int32_t t_lo, int depth, int64_t own) {
    int64_t s = e->acc[node].head_out;
    for (int k = 0; k < e->cfg.scan_cap && s >= 0; k++) {
        edge_t *ed = edge_at(e, s);
        if (!ed || ed->t < t_lo) return;
        s = ed->prev_out;
        if (ed->seq == own) continue;
        if (--c->budget < 0) {
            c->truncated = 1;
            return;
        }
        if (ed->usd_out < c->lo || ed->usd_out > c->hi) continue;
        int32_t y = ed->dst;
        if (y == c->target) {
            int len = depth + 2;
            c->found++;
            if (c->min_len == 0 || len < c->min_len) c->min_len = len;
            if (c->found >= CYCLE_FOUND_CAP) {
                c->budget = -1;
                return;
            }
            continue;
        }
        if (depth + 3 > c->max_len || y == c->start) continue;
        int on_path = 0;
        for (int j = 0; j < c->plen; j++)
            if (c->path[j] == y) {
                on_path = 1;
                break;
            }
        if (on_path) continue;
        c->path[c->plen++] = y;
        sdfs(e, c, y, t_lo, depth + 1, own);
        c->plen--;
        if (c->budget < 0) return;
    }
}

int aml_hindsight(aml_engine *e, const aml_batch *b, const int64_t *seq_at, float *out) {
    if (!e || !b || !seq_at || !out) return -1;
    const int cap = e->cfg.scan_cap, fcap = e->cfg.fan_cap;
    for (int64_t i = 0; i < b->n; i++) {
        float *h = out + i * (int64_t)H_COUNT;
        for (int k = 0; k < H_COUNT; k++) h[k] = 0.f;
        h[H_h_r_fwd_err] = NO_MATCH;
        h[H_h_r_fwd_lag] = -1.f;
        const int32_t t = b->t[i], a = b->src[i], d = b->dst[i];
        if (a == d || a < 0 || a >= e->n_acc || d < 0 || d >= e->n_acc) continue;
        const int own_edge = (e->cfg.motif_fmt_mask >> b->fmt[i]) & 1;
        const int64_t s_i = seq_at[i], own = own_edge ? s_i : -2;
        const float xr = b->usd_recv[i];
        const int32_t t_lo = t - e->cfg.win_long;
#define AFTER(ed) ((ed)->seq >= s_i && (ed)->seq != own)
        /* émetteur : bénéficiaires et émetteurs apparus après */
        iset *so = &e->s_out_src, *sd = &e->s_out_dst, *tmp = &e->s_tmp;
        iset_clear(so);
        int cnt = 0;
        int64_t s = e->acc[a].head_out;
        for (int k = 0; k < cap && s >= 0; k++) {
            edge_t *ed = edge_at(e, s);
            if (!ed || !AFTER(ed)) break; /* liste par seq décroissant : le reste est antérieur */
            cnt++;
            iset_add(so, ed->dst);
            s = ed->prev_out;
        }
        h[H_h_s_out_after] = (float)so->n;
        h[H_h_s_outcnt_after] = (float)cnt;
        iset_clear(tmp);
        s = e->acc[a].head_in;
        for (int k = 0; k < cap && s >= 0; k++) {
            edge_t *ed = edge_at(e, s);
            if (!ed || !AFTER(ed)) break;
            iset_add(tmp, ed->src);
            s = ed->prev_in;
        }
        h[H_h_s_in_after] = (float)tmp->n;
        /* destinataire : réception et réémission après (passage de fonds en aval) */
        iset_clear(tmp);
        s = e->acc[d].head_in;
        for (int k = 0; k < cap && s >= 0; k++) {
            edge_t *ed = edge_at(e, s);
            if (!ed || !AFTER(ed)) break;
            iset_add(tmp, ed->src);
            s = ed->prev_in;
        }
        h[H_h_r_in_after] = (float)tmp->n;
        iset_clear(sd);
        float best = NO_MATCH;
        int32_t best_t = 0, n10 = 0;
        double fwd_sum = 0.0;
        s = e->acc[d].head_out;
        for (int k = 0; k < cap && s >= 0; k++) {
            edge_t *ed = edge_at(e, s);
            if (!ed || !AFTER(ed)) break;
            iset_add(sd, ed->dst);
            fwd_sum += ed->usd_out;
            if (ed->usd_out > 0.f) {
                float err = fabsf(ed->usd_out / xr - 1.f);
                if (err <= best) { /* à égalité : la plus ancienne (parcours du plus récent) */
                    best = err;
                    best_t = ed->t;
                }
                n10 += err <= 0.10f;
            }
            s = ed->prev_out;
        }
        h[H_h_r_out_after] = (float)sd->n;
        h[H_h_r_fwd_err] = best;
        if (best < NO_MATCH) h[H_h_r_fwd_lag] = (float)(best_t - t);
        h[H_h_r_fwd_n10] = (float)n10;
        h[H_h_r_split] = xr > 0.f ? (float)(fwd_sum / xr) : 0.f;
        /* scatter-gather où cette arête est une dispersion : a → w' → b avec b parmi les bénéficiaires
         * ultérieurs de d, pour d'autres w' payés par a dans la fenêtre */
        int sg = 0;
        if (sd->n) {
            iset_clear(so);
            s = e->acc[a].head_out;
            for (int k = 0; k < fcap && s >= 0; k++) {
                edge_t *ed = edge_at(e, s);
                if (!ed || ed->t < t_lo) break;
                s = ed->prev_out;
                int32_t w = ed->dst;
                if (w == d || !iset_add(so, w)) continue;
                int64_t s2 = e->acc[w].head_out;
                for (int k2 = 0; k2 < fcap && s2 >= 0; k2++) {
                    edge_t *e2 = edge_at(e, s2);
                    if (!e2 || e2->t < t_lo) break;
                    if (iset_has(sd, e2->dst)) {
                        sg++;
                        break;
                    }
                    s2 = e2->prev_out;
                }
            }
        }
        h[H_h_sg_scatter] = (float)sg;
        /* cycle statique contenant l'arête : chemin d → … → a dans la fenêtre, montants à ±30 % */
        if (own_edge) {
            dfs_ctx c;
            memset(&c, 0, sizeof(c));
            c.target = a;
            c.start = d;
            c.x = b->usd_paid[i];
            c.lo = c.x / 1.3f;
            c.hi = c.x * 1.3f;
            c.max_len = e->cfg.cycle_max_len;
            c.budget = e->cfg.cycle_budget;
            c.path[c.plen++] = d;
            sdfs(e, &c, d, t_lo, 0, own);
            h[H_h_cyc_n] = (float)c.found;
            h[H_h_cyc_minlen] = (float)c.min_len;
        }
#undef AFTER
    }
    return 0;
}

/* ---- instantané de l'état (sans la forêt) ---------------------------------------------------------- */
typedef struct {
    char magic[8];
    int32_t version, f_count, n_acc, pad;
    int64_t ring_cap, pair_cap, seq_next;
    aml_config cfg;
} snap_hdr;

static int wr(FILE *f, const void *p, size_t sz, size_t n) { return fwrite(p, sz, n, f) == n ? 0 : -1; }
static int rd(FILE *f, void *p, size_t sz, size_t n) { return fread(p, sz, n, f) == n ? 0 : -1; }

int64_t aml_snapshot(const aml_engine *e, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    snap_hdr h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, "AMLSNAP", 8);
    h.version = 1;
    h.f_count = F_COUNT;
    h.n_acc = e->n_acc;
    h.ring_cap = e->ring_cap;
    h.pair_cap = e->pair_cap;
    h.seq_next = e->seq_next;
    h.cfg = e->cfg;
    size_t na = (size_t)e->n_acc;
    int rc = wr(f, &h, sizeof(h), 1) | wr(f, e->kyc, sizeof(kyc_t), na) | wr(f, e->acc, sizeof(acct_t), na) |
             wr(f, e->ring, sizeof(edge_t), (size_t)e->ring_cap) |
             wr(f, e->pairs, sizeof(pair_t), (size_t)e->pair_cap) | wr(f, e->stat, sizeof(int64_t), ST_COUNT);
    long sz = ftell(f);
    fclose(f);
    return rc ? -2 : (int64_t)sz;
}

int aml_restore(aml_engine *e, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    snap_hdr h;
    int rc = rd(f, &h, sizeof(h), 1);
    if (rc || memcmp(h.magic, "AMLSNAP", 8) != 0 || h.version != 1 || h.f_count != F_COUNT ||
        h.n_acc != e->n_acc || h.ring_cap != e->ring_cap || h.pair_cap != e->pair_cap ||
        memcmp(&h.cfg, &e->cfg, sizeof(aml_config)) != 0) {
        fclose(f);
        return -2; /* instantané incompatible avec ce moteur */
    }
    size_t na = (size_t)e->n_acc;
    rc = rd(f, e->kyc, sizeof(kyc_t), na) | rd(f, e->acc, sizeof(acct_t), na) |
         rd(f, e->ring, sizeof(edge_t), (size_t)e->ring_cap) |
         rd(f, e->pairs, sizeof(pair_t), (size_t)e->pair_cap) | rd(f, e->stat, sizeof(int64_t), ST_COUNT);
    fclose(f);
    if (rc) return -3;
    e->seq_next = h.seq_next;
    return 0;
}
