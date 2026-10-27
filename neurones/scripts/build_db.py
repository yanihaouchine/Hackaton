"""Construit la base DuckDB d'exploration et d'évaluation à partir de data/raw/.

Usage (depuis neurones/) : python scripts/build_db.py [--variant HI-Small] [--out CHEMIN]

Tables produites :
- transactions : triées (tri stable sur Timestamp), sans étiquette, avec la colonne split ;
- labels       : étiquette Is Laundering et typologie éventuelle, par tx_id ;
- accounts     : comptes, avec bank_key normalisé pour la jointure ;
- attempts     : tentatives de Patterns.txt (typologie, détail, split de départ) ;
- attempt_tx   : lien tentative -> tx_id.

Sert à l'exploration et à l'évaluation, jamais à l'inférence en flux.
"""

import argparse
import re
import time
from pathlib import Path

import duckdb

# Racine du dépôt : data/ y est partagé entre les solutions.
REPO = Path(__file__).resolve().parents[2]

# Découpage commun du dépôt (CLAUDE.md de la racine). Le protocole littérature
# correspond à split IN ('test', 'tail').
SPLIT_SQL = """
CASE
    WHEN ts < TIMESTAMP '2022-09-07' THEN 'train'
    WHEN ts < TIMESTAMP '2022-09-09' THEN 'val'
    WHEN ts < TIMESTAMP '2022-09-11' THEN 'test'
    ELSE 'tail'
END
"""

TRANS_COLUMNS = {
    "ts": "TIMESTAMP",
    "from_bank": "VARCHAR",
    "from_account": "VARCHAR",
    "to_bank": "VARCHAR",
    "to_account": "VARCHAR",
    "amount_received": "DOUBLE",
    "currency_received": "VARCHAR",
    "amount_paid": "DOUBLE",
    "currency_paid": "VARCHAR",
    "payment_format": "VARCHAR",
    "is_laundering": "TINYINT",
}

BEGIN_RE = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z-]+)(?::\s*(.*))?$")


def bank_key(col: str) -> str:
    """Code banque sans zéros en tête (Trans.csv en a, accounts.csv non)."""
    return f"coalesce(nullif(ltrim({col}, '0'), ''), '0')"


def load_transactions(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    # header=false + noms explicites : les deux colonnes « Account » sont renommées.
    con.execute(
        """
        CREATE TEMP TABLE raw AS
        SELECT * FROM read_csv(?, header = false, skip = 1, columns = ?,
                               timestampformat = '%Y/%m/%d %H:%M')
        """,
        [str(path), TRANS_COLUMNS],
    )
    # rowid suit l'ordre du fichier (preserve_insertion_order) : tri stable.
    con.execute(
        """
        CREATE TEMP TABLE ordered AS
        SELECT row_number() OVER (ORDER BY ts, rowid) - 1 AS tx_id, *
        FROM raw
        """
    )
    con.execute(
        f"""
        CREATE TABLE transactions AS
        SELECT tx_id, ts, from_bank, from_account, to_bank, to_account,
               amount_received, currency_received, amount_paid, currency_paid,
               payment_format,
               (from_bank = to_bank AND from_account = to_account) AS is_self_transfer,
               {SPLIT_SQL} AS split
        FROM ordered ORDER BY tx_id
        """
    )
    con.execute("CREATE TABLE labels AS SELECT tx_id, is_laundering FROM ordered ORDER BY tx_id")
    con.execute("DROP TABLE raw")


def load_accounts(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    con.execute(
        f"""
        CREATE TABLE accounts AS
        SELECT "Bank Name" AS bank_name,
               {bank_key('"Bank ID"')} AS bank_key,
               "Account Number" AS account,
               "Entity ID" AS entity_id,
               "Entity Name" AS entity_name
        FROM read_csv(?, all_varchar = true)
        """,
        [str(path)],
    )


def parse_patterns(path: Path) -> tuple[list[tuple], list[tuple]]:
    attempts, lines = [], []
    attempt_id = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("BEGIN"):
            m = BEGIN_RE.match(line)
            if m is None:
                raise ValueError(f"ligne BEGIN inattendue : {line}")
            attempt_id = len(attempts)
            attempts.append((attempt_id, m.group(1), (m.group(2) or "").strip()))
        elif line.startswith("END"):
            attempt_id = None
        else:
            if attempt_id is None:
                raise ValueError(f"transaction hors tentative : {line}")
            lines.append((attempt_id, *line.split(",")))
    return attempts, lines


def load_patterns(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    attempts, lines = parse_patterns(path)
    con.execute("CREATE TABLE attempts (attempt_id INTEGER, typology VARCHAR, detail VARCHAR)")
    con.executemany("INSERT INTO attempts VALUES (?, ?, ?)", attempts)
    con.execute(
        "CREATE TEMP TABLE pattern_lines (attempt_id INTEGER, ts VARCHAR, "
        "from_bank VARCHAR, from_account VARCHAR, to_bank VARCHAR, to_account VARCHAR, "
        "amount_received DOUBLE, currency_received VARCHAR, amount_paid DOUBLE, "
        "currency_paid VARCHAR, payment_format VARCHAR, is_laundering TINYINT)"
    )
    con.executemany("INSERT INTO pattern_lines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", lines)
    # Jointure sur tous les champs : une ligne de motif peut correspondre à
    # plusieurs transactions identiques, on les garde toutes et on le signale.
    con.execute(
        """
        CREATE TABLE attempt_tx AS
        SELECT DISTINCT p.attempt_id, o.tx_id
        FROM pattern_lines p
        JOIN ordered o
          ON o.ts = strptime(p.ts, '%Y/%m/%d %H:%M')
         AND o.from_bank = p.from_bank AND o.from_account = p.from_account
         AND o.to_bank = p.to_bank AND o.to_account = p.to_account
         AND o.amount_received = p.amount_received
         AND o.currency_received = p.currency_received
         AND o.amount_paid = p.amount_paid AND o.currency_paid = p.currency_paid
         AND o.payment_format = p.payment_format
        ORDER BY p.attempt_id, o.tx_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE attempts AS
        SELECT a.*, t.split AS start_split
        FROM attempts a
        LEFT JOIN (
            SELECT attempt_id, arg_min(split, ts) AS split
            FROM attempt_tx JOIN transactions USING (tx_id)
            GROUP BY attempt_id
        ) t USING (attempt_id)
        ORDER BY attempt_id
        """
    )
    # Typologie par transaction ; NULL si la transaction n'est dans aucun motif.
    con.execute(
        """
        CREATE OR REPLACE TABLE labels AS
        SELECT l.tx_id, l.is_laundering, m.attempt_id, m.typology
        FROM labels l
        LEFT JOIN (
            SELECT tx_id, min(attempt_id) AS attempt_id,
                   min(typology) AS typology
            FROM attempt_tx JOIN attempts USING (attempt_id)
            GROUP BY tx_id
        ) m USING (tx_id)
        ORDER BY l.tx_id
        """
    )
    con.execute("DROP TABLE pattern_lines")
    n_lines = len(lines)
    n_matched = con.execute("SELECT count(DISTINCT tx_id) FROM attempt_tx").fetchone()[0]
    print(f"motifs : {len(attempts)} tentatives, {n_lines} lignes, {n_matched} transactions retrouvées")


def report(con: duckdb.DuckDBPyConnection) -> None:
    print(
        con.sql(
            """
            SELECT t.split, count(*) AS n, sum(l.is_laundering) AS illicit,
                   round(100 * avg(l.is_laundering), 3) AS pct_illicit,
                   min(t.ts) AS first_ts, max(t.ts) AS last_ts
            FROM transactions t JOIN labels l USING (tx_id)
            GROUP BY t.split ORDER BY min(t.ts)
            """
        )
    )
    print(
        con.sql(
            """
            SELECT start_split, count(*) AS attempts
            FROM attempts GROUP BY start_split ORDER BY min(attempt_id)
            """
        )
    )
    unmatched = con.execute(
        """
        SELECT count(*) FROM transactions t
        WHERE NOT EXISTS (SELECT 1 FROM accounts a
                          WHERE a.bank_key = {} AND a.account = t.from_account)
        """.format(bank_key("t.from_bank"))
    ).fetchone()[0]
    print(f"transactions dont l'émetteur est absent de accounts : {unmatched}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", default="HI-Small")
    parser.add_argument("--raw", type=Path, default=REPO / "data" / "raw")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "neurones" / "aml.duckdb")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="2GB")
    args = parser.parse_args()

    start = time.perf_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.unlink(missing_ok=True)
    con = duckdb.connect(str(args.out))
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute("SET preserve_insertion_order = true")

    load_transactions(con, args.raw / f"{args.variant}_Trans.csv")
    load_accounts(con, args.raw / f"{args.variant}_accounts.csv")
    load_patterns(con, args.raw / f"{args.variant}_Patterns.txt")
    con.execute("DROP TABLE ordered")
    report(con)
    con.close()
    size_mb = args.out.stat().st_size / 2**20
    print(f"{args.out} : {size_mb:.0f} Mo, {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
