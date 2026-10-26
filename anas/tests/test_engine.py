from aml.data import LABEL_COLUMNS, STREAM_COLUMNS
from aml.engine import StreamDetector, Thresholds

TH = Thresholds(fan_out=3, fan_in=99, chain_depth=99, structuring_count=99, risky_formats=("ACH",))
RATES = {"US Dollar": 1.0}


def fan_out_rows(fmt: str = "ACH") -> list[tuple]:
    # Un émetteur, 4 bénéficiaires distincts, une transaction toutes les 10 minutes.
    return [(i, 10 * i, "001", "SRC", "002", f"DST{i}", 500.0, "US Dollar", fmt) for i in range(4)]


def test_stream_never_reads_labels() -> None:
    assert not set(LABEL_COLUMNS) & set(STREAM_COLUMNS)


def test_fan_out_alert_fires_at_threshold() -> None:
    det = StreamDetector(TH, RATES)
    alerts = [a for row in fan_out_rows() for a in det.process(*row)[1]]
    assert len(alerts) == 1  # la 4e transaction est en période de « cooldown »
    a = alerts[0]
    assert a.typology == "fan_out" and a.trigger_tx_id == 2
    assert a.subject.key() == "001/SRC"
    assert a.tx_ids == (0, 1, 2)
    assert a.reasons[0].value == 3 and a.reasons[0].threshold == 3


def test_rules_only_on_risky_formats() -> None:
    det = StreamDetector(TH, RATES)
    assert not [a for row in fan_out_rows("Cheque") for a in det.process(*row)[1]]


def test_window_expires() -> None:
    det = StreamDetector(TH, RATES)
    rows = fan_out_rows()
    rows[2] = (2, 25 * 60, *rows[2][2:])  # 3e virement 25 h après : hors fenêtre de 24 h
    rows[3] = (3, 25 * 60 + 1, *rows[3][2:])
    feats = [det.process(*row)[0] for row in rows]
    # à 25 h, les deux premiers virements sont sortis de la fenêtre
    assert [f.fan_out for f in feats] == [1, 2, 1, 2]


def test_features_are_causal() -> None:
    """Les indicateurs d'une transaction ne changent pas si on ajoute des transactions futures."""
    rows = fan_out_rows() + [(9, 100, "002", "DST0", "001", "SRC", 480.0, "US Dollar", "ACH")]
    first = StreamDetector(TH, RATES)
    short = [first.process(*r)[0] for r in rows[:3]]
    det = StreamDetector(TH, RATES)
    full = [det.process(*r)[0] for r in rows][:3]
    assert short == full


def test_rapid_chain() -> None:
    th = Thresholds(fan_out=99, fan_in=99, chain_depth=3, structuring_count=99)
    det = StreamDetector(th, RATES)
    hops = [("A", "B", 1000.0), ("B", "C", 950.0), ("C", "D", 900.0)]
    alerts = []
    for i, (s, d, amount) in enumerate(hops):
        alerts += det.process(i, 60 * i, "001", s, "001", d, amount, "US Dollar", "ACH")[1]
    assert [a.typology for a in alerts] == ["rapid_chain"]
    assert alerts[0].subject.account == "D" and alerts[0].tx_ids == (0, 1, 2)
