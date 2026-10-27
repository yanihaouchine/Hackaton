"""Préparation hors ligne : CSV IBM AML → flux encodé, trié et causal.

- tri stable sur l'horodatage (le CSV n'est pas trié) ;
- un compte = (code banque brut, numéro) ; ids int32 attribués par ordre de première apparition ;
- montants convertis en USD avec des taux dérivés des transactions multi-devises du **train** ;
- KYC statique (accounts.csv) : type d'entité, taille de l'entité, pays de la banque ;
- étiquettes (Is Laundering, tentative, typologie) rangées à part : le moteur ne les reçoit jamais.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from aml.config import (
    CURRENCIES,
    DATA_PROC,
    DATA_RAW,
    ENTITY_TYPES,
    FORMATS,
    SPLITS,
    TYPOLOGIES,
    minutes_since,
)

TX_COLS = [
    "ts", "src_bank", "src_acc", "dst_bank", "dst_acc",
    "amt_recv", "cur_recv", "amt_paid", "cur_paid", "fmt", "y",
]  # fmt: skip
MAX_CSV_BYTES = 4 << 30  # au-delà (variantes Large), il faudrait un chemin par blocs
COUNTRY_RE = re.compile(r"^(?P<c>.+) Bank #\d+$")


def _read_tx(path: Path):
    import polars as pl

    # codes banque et comptes en chaîne (zéros de tête) ; la 2e colonne « Account » est dédoublée par polars
    as_str = {c: pl.Utf8 for c in ("Timestamp", "From Bank", "Account", "To Bank", "Account_duplicated_0")}
    df = pl.read_csv(path, has_header=True, schema_overrides=as_str)
    if len(df.columns) != len(TX_COLS):
        raise ValueError(f"colonnes inattendues : {df.columns}")
    df.columns = TX_COLS  # renommage positionnel : les deux « Account » deviennent src_acc / dst_acc
    return df.with_columns(pl.col("amt_recv").cast(pl.Float64), pl.col("amt_paid").cast(pl.Float64),
                           pl.col("y").cast(pl.Int8))  # fmt: skip


def _read_patterns(path: Path):
    """Patterns.txt → une ligne par transaction de motif, avec n° de tentative et typologie."""
    import polars as pl

    rows, attempt, typ = [], -1, None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            attempt += 1
            typ = line.split(" - ", 1)[1].split(":", 1)[0].strip()
            continue
        if not line or line.startswith("END"):
            continue
        p = line.split(",")
        rows.append((p[0], p[1], p[2], p[3], p[4], float(p[5]), p[6], float(p[7]), p[8], p[9], attempt, typ))
    cols = TX_COLS[:-1] + ["attempt", "typ"]
    return pl.DataFrame(rows, schema=cols, orient="row")


def _fx_rates(df, train_end: int, min_n: int = 20) -> tuple[np.ndarray, dict]:
    """Taux (unités de devise pour 1 USD) dérivés des transactions multi-devises du train.

    Médiane directe des paires USD→X (ou X→USD) quand elles sont assez nombreuses : le simulateur
    utilise des taux fixes par rapport à l'USD. Repli : moindres carrés sur toutes les paires.
    """
    import polars as pl

    x = df.filter((pl.col("cur_paid") != pl.col("cur_recv")) & (pl.col("t") < train_end))
    g = (
        x.with_columns(lr=(pl.col("amt_recv") / pl.col("amt_paid")).log())
        .group_by("cur_paid", "cur_recv")
        .agg(lr=pl.col("lr").median(), n=pl.len(), cv=pl.col("lr").exp().std() / pl.col("lr").exp().mean())
    )
    k = len(CURRENCIES)
    a_rows, b, w = [], [], []
    for cp, cr, lr, n, _ in g.iter_rows():
        row = np.zeros(k)
        row[CURRENCIES.index(cr)] += 1.0
        row[CURRENCIES.index(cp)] -= 1.0
        a_rows.append(row)
        b.append(lr)
        w.append(np.sqrt(n))
    a, b, w = np.array(a_rows), np.array(b), np.array(w)
    seen = set(g["cur_paid"].to_list()) | set(g["cur_recv"].to_list())
    missing = [c for c in CURRENCIES if c not in seen]
    if missing:
        raise ValueError(f"devises sans transaction multi-devises dans le train : {missing}")
    sol, *_ = np.linalg.lstsq(a[:, 1:] * w[:, None], b * w, rcond=None)  # log(rate[USD]) = 0
    log_rate = np.concatenate([[0.0], sol])
    method = {}
    usd = CURRENCIES[0]
    for i, c in enumerate(CURRENCIES[1:], start=1):
        fwd = g.filter((pl.col("cur_paid") == usd) & (pl.col("cur_recv") == c))
        bwd = g.filter((pl.col("cur_paid") == c) & (pl.col("cur_recv") == usd))
        nf = fwd["n"].item() if fwd.height else 0
        nb = bwd["n"].item() if bwd.height else 0
        if max(nf, nb) >= min_n:
            log_rate[i] = fwd["lr"].item() if nf >= nb else -bwd["lr"].item()
            method[c] = f"médiane USD ({max(nf, nb)} tx)"
        else:
            method[c] = "moindres carrés"
    resid = b - a @ log_rate
    info = {
        "rates_per_usd": dict(zip(CURRENCIES, np.exp(log_rate).round(8).tolist(), strict=True)),
        "method": method,
        "n_pairs": int(len(b)),
        "n_tx": int(x.height),
        "median_abs_log_residual": float(np.median(np.abs(resid))),
    }
    return np.exp(log_rate), info


def _kyc(keys, accounts_path: Path):
    """Attributs KYC statiques par compte (ordre des ids)."""
    import polars as pl

    acc = pl.read_csv(
        accounts_path,
        schema_overrides={"Bank ID": pl.Utf8, "Account Number": pl.Utf8, "Entity ID": pl.Utf8},
    ).rename(
        {"Bank Name": "bank_name", "Bank ID": "bank_id", "Account Number": "acc", "Entity ID": "ent",
         "Entity Name": "ent_name"}
    )  # fmt: skip
    acc = acc.with_columns(
        bank_int=pl.col("bank_id").cast(pl.Int64),
        n_acc=pl.len().over("ent"),
        etype=pl.col("ent_name").str.replace(r" #\d+$", ""),
    )
    dup = acc.height - acc.unique(subset=["bank_int", "acc"]).height
    if dup:
        raise ValueError(f"{dup} doublons (banque, compte) dans accounts.csv")
    k = keys.with_columns(bank_int=pl.col("bank").cast(pl.Int64)).join(
        acc.select("bank_int", "acc", "bank_name", "ent", "ent_name", "n_acc", "etype"),
        on=["bank_int", "acc"],
        how="left",
    )
    k = k.sort("acc_id")
    n_missing = k["ent"].null_count()

    def country(name: str | None) -> str:
        if name is None:
            return "Inconnu"
        m = COUNTRY_RE.match(name)
        if not m:
            return "USA"  # banques nommées à l'américaine, sans « #n »
        c = m.group("c")
        return "Crypto" if c == "Crytpo" else c

    countries = [country(n) for n in k["bank_name"].to_list()]
    vocab = sorted(set(countries))
    ent_codes = {e: i for i, e in enumerate(sorted(set(x for x in k["ent"].to_list() if x is not None)))}
    bank_codes = {b: i for i, b in enumerate(sorted(set(k["bank"].to_list())))}  # code brut, zéros compris
    out = {
        "acc_bank": np.array([bank_codes[b] for b in k["bank"].to_list()], dtype=np.int32),
        "acc_entity": np.array([ent_codes.get(e, -1) for e in k["ent"].to_list()], dtype=np.int32),
        "acc_etype": np.array(
            [ENTITY_TYPES.index(t) if t in ENTITY_TYPES else len(ENTITY_TYPES) for t in k["etype"].to_list()],
            dtype=np.uint8,
        ),
        "acc_nacc": k["n_acc"].fill_null(1).to_numpy().astype(np.int32),
        "acc_country": np.array([vocab.index(c) for c in countries], dtype=np.int16),
    }
    names = k.select("acc_id", "key", "bank", "acc", "bank_name", "ent_name")
    return out, names, vocab, n_missing


def prepare(variant: str, out_dir: Path | None = None) -> Path:
    import polars as pl

    t_start = time.time()
    src = DATA_RAW / f"{variant}_Trans.csv"
    if src.stat().st_size > MAX_CSV_BYTES:
        raise NotImplementedError("variante trop grosse pour la préparation en mémoire (chemin par blocs à écrire)")
    out = out_dir or DATA_PROC / variant
    out.mkdir(parents=True, exist_ok=True)

    df = _read_tx(src).with_row_index("row")
    ts = df["ts"].str.strptime(pl.Datetime, "%Y/%m/%d %H:%M")
    t0 = datetime.combine(ts.min().date(), datetime.min.time())
    df = df.with_columns(t=((ts - pl.lit(t0)).dt.total_minutes()).cast(pl.Int32))

    pat = _read_patterns(DATA_RAW / f"{variant}_Patterns.txt")
    n_pat = pat.height
    pat = pat.unique(subset=TX_COLS[:-1], keep="first")
    df = df.join(pat, on=TX_COLS[:-1], how="left").sort("row")
    n_matched = int(df["attempt"].is_not_null().sum())
    bad = int(df.filter(pl.col("attempt").is_not_null() & (pl.col("y") == 0)).height)
    if bad:
        raise ValueError(f"{bad} transactions licites appariées à un motif")

    df = df.sort("t", maintain_order=True)  # tri stable : l'ordre du fichier départage les ex aequo

    # ids de comptes par ordre de première apparition dans le flux trié (émetteur avant destinataire)
    n = df.height
    sk = pl.concat_str(["src_bank", "src_acc"], separator=":")
    dk = pl.concat_str(["dst_bank", "dst_acc"], separator=":")
    both = pl.concat(
        [
            df.select(
                pl.int_range(n).alias("i"),
                pl.lit(0).alias("side"),
                sk.alias("key"),
                pl.col("src_bank").alias("bank"),
                pl.col("src_acc").alias("acc"),
            ),
            df.select(
                pl.int_range(n).alias("i"),
                pl.lit(1).alias("side"),
                dk.alias("key"),
                pl.col("dst_bank").alias("bank"),
                pl.col("dst_acc").alias("acc"),
            ),
        ]  # fmt: skip
    ).sort(["i", "side"])
    keys = both.unique(subset="key", keep="first", maintain_order=True).drop("i", "side")
    keys = keys.with_row_index("acc_id").with_columns(pl.col("acc_id").cast(pl.Int32))
    kmap = keys.select("key", "acc_id")
    df = df.with_columns(sk.alias("sk"), dk.alias("dk"))
    df = df.join(kmap.rename({"key": "sk", "acc_id": "src"}), on="sk", how="left", maintain_order="left")
    df = df.join(kmap.rename({"key": "dk", "acc_id": "dst"}), on="dk", how="left", maintain_order="left")

    train_end = minutes_since(t0, SPLITS["train"][1])
    rates, fx_info = _fx_rates(df, train_end)
    cur_p = df["cur_paid"].replace_strict(CURRENCIES, list(range(len(CURRENCIES))), return_dtype=pl.UInt8)
    cur_r = df["cur_recv"].replace_strict(CURRENCIES, list(range(len(CURRENCIES))), return_dtype=pl.UInt8)
    fmt = df["fmt"].replace_strict(FORMATS, list(range(len(FORMATS))), return_dtype=pl.UInt8)
    cp, cr = cur_p.to_numpy(), cur_r.to_numpy()
    amt_paid = df["amt_paid"].to_numpy()
    amt_recv = df["amt_recv"].to_numpy()

    arrays = {
        "t": df["t"].to_numpy().astype(np.int32),
        "src": df["src"].to_numpy().astype(np.int32),
        "dst": df["dst"].to_numpy().astype(np.int32),
        "usd_paid": (amt_paid / rates[cp]).astype(np.float32),
        "usd_recv": (amt_recv / rates[cr]).astype(np.float32),
        "amt_paid": amt_paid.astype(np.float64),
        "amt_recv": amt_recv.astype(np.float64),
        "cur_paid": cp.astype(np.uint8),
        "cur_recv": cr.astype(np.uint8),
        "fmt": fmt.to_numpy().astype(np.uint8),
        "row": df["row"].to_numpy().astype(np.int32),
    }
    labels = {
        "y": df["y"].to_numpy().astype(np.int8),
        "attempt": df["attempt"].fill_null(-1).to_numpy().astype(np.int32),
        "typ": df["typ"]
        .replace_strict(TYPOLOGIES, list(range(len(TYPOLOGIES))), default=-1, return_dtype=pl.Int8)
        .to_numpy()
        .astype(np.int8),
    }
    kyc, names, countries, n_missing = _kyc(keys, DATA_RAW / f"{variant}_accounts.csv")

    for k, v in arrays.items():
        np.save(out / f"{k}.npy", v)
    (out / "labels").mkdir(exist_ok=True)
    for k, v in labels.items():
        np.save(out / "labels" / f"{k}.npy", v)
    for k, v in kyc.items():
        np.save(out / f"{k}.npy", v)
    names.write_parquet(out / "accounts.parquet")
    # mêmes informations en tableaux ASCII mmap-ables : l'inférence n'a pas besoin de polars
    for col, fname in (("key", "acc_key"), ("bank_name", "acc_bank_name"), ("ent_name", "acc_ent_name")):
        vals = [(v or "").encode("ascii", "replace") for v in names[col].to_list()]
        np.save(out / f"{fname}.npy", np.array(vals, dtype=f"S{max(len(v) for v in vals)}"))

    meta = {
        "variant": variant,
        "t0": t0.isoformat(),
        "n_tx": n,
        "n_accounts": keys.height,
        "n_illicit": int(labels["y"].sum()),
        "n_pattern_rows": n_pat,
        "n_pattern_matched": n_matched,
        "n_accounts_without_kyc": int(n_missing),
        "countries": countries,
        "fx": fx_info,
        "splits_minutes": {
            k: [minutes_since(t0, a), None if b is None else minutes_since(t0, b)] for k, (a, b) in SPLITS.items()
        },
        "source_bytes": src.stat().st_size,
        "seconds": round(time.time() - t_start, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return out


@dataclass
class Stream:
    """Flux préparé, chargé en mmap (RAM bornée) ; les étiquettes restent dans `labels`."""

    meta: dict
    t: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    usd_paid: np.ndarray
    usd_recv: np.ndarray
    amt_paid: np.ndarray
    amt_recv: np.ndarray
    cur_paid: np.ndarray
    cur_recv: np.ndarray
    fmt: np.ndarray
    row: np.ndarray
    acc_bank: np.ndarray
    acc_entity: np.ndarray
    acc_etype: np.ndarray
    acc_nacc: np.ndarray
    acc_country: np.ndarray
    path: Path

    @property
    def n(self) -> int:
        return len(self.t)

    def split_mask(self, name: str) -> np.ndarray:
        a, b = self.meta["splits_minutes"][name]
        m = self.t >= a
        if b is not None:
            m &= self.t < b
        return m

    def split_range(self, name: str) -> tuple[int, int]:
        """Indices [début, fin) d'un découpage (le flux est trié par temps)."""
        a, b = self.meta["splits_minutes"][name]
        lo = int(np.searchsorted(self.t, a, side="left"))
        hi = self.n if b is None else int(np.searchsorted(self.t, b, side="left"))
        return lo, hi


def load(variant: str, mmap: bool = True) -> Stream:
    d = DATA_PROC / variant
    meta = json.loads((d / "meta.json").read_text())
    mode = "r" if mmap else None
    fields = {
        k: np.load(d / f"{k}.npy", mmap_mode=mode) for k in Stream.__dataclass_fields__ if k not in ("meta", "path")
    }
    return Stream(meta=meta, path=d, **fields)


def load_labels(variant: str) -> dict[str, np.ndarray]:
    d = DATA_PROC / variant / "labels"
    return {k: np.load(d / f"{k}.npy") for k in ("y", "attempt", "typ")}


class AccountNames:
    """Accès par id aux libellés des comptes (clé, banque, titulaire), en mmap : aucune table en RAM."""

    def __init__(self, variant: str):
        d = DATA_PROC / variant
        self.key = np.load(d / "acc_key.npy", mmap_mode="r")
        self.bank = np.load(d / "acc_bank_name.npy", mmap_mode="r")
        self.ent = np.load(d / "acc_ent_name.npy", mmap_mode="r")

    def __getitem__(self, i: int) -> dict:
        return {"key": self.key[i].decode(), "bank_name": self.bank[i].decode() or None,
                "ent_name": self.ent[i].decode() or None}  # fmt: skip
