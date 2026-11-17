"""
USTEC Breakout Strategy Hub - Web Dashboard (Streamlit)
Platform analisis visual intraday range breakout USTEC / US Tech 100 CFD.
"""

import os
import sys
import io
import time
from types import SimpleNamespace
from itertools import product
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

# Tambahkan direktori kerja ke sys.path agar modul ustec_range_mae dapat diimpor
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# --- CORE USTEC RANGE & RISK ENGINE (EMBEDDED) ---
DEFAULT_RANGES = {
    1: "1256-2023",
    2: "0830-2000",
    3: "0404-0944",
    4: "2152-1804",
    5: "2156-0705",
    6: "0808-1833",
    7: "0010-0101",
}


# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def hhmm_to_min(s):
    s = s.strip()
    return int(s[:2]) * 60 + int(s[2:4])


def fmt_time_min(m):
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"


def first_true(c):
    """Kembalikan index pertama yang True pada boolean array c, atau -1 jika tidak ada."""
    if c.size == 0:
        return -1
    i = int(np.argmax(c))
    return i if c[i] else -1


def parse_capital(s):
    """Parse string modal seperti '8jt', '8 juta', '8000000', '8m' menjadi float."""
    if s is None:
        return None
    s = str(s).lower().replace(".", "").replace(",", "").replace("rp", "").strip()
    if "jt" in s or "juta" in s:
        val = float(s.replace("jt", "").replace("juta", "").strip()) * 1_000_000
    elif "k" in s or "rb" in s:
        val = float(s.replace("k", "").replace("rb", "").strip()) * 1_000
    else:
        val = float(s)
    return val


def fmt_idr(val, with_sign=True):
    sign = "+" if (with_sign and val > 0) else ("-" if (with_sign and val < 0) else "")
    return f"{sign}Rp {abs(val):,.0f}".replace(",", ".")


def fmt_duration(dur_s):
    if pd.isna(dur_s) or dur_s is None:
        return "-"
    if dur_s < 60:
        return f"{dur_s:.1f}s"
    m = int(dur_s // 60)
    s = int(dur_s % 60)
    return f"{m}m {s}s ({dur_s:.0f}s)"


# --------------------------------------------------------------------------- #
# Load tick
# --------------------------------------------------------------------------- #
def read_one(path, chunk_rows=2_000_000):
    """Baca CSV tick per potongan (chunk) supaya file besar (GB) tidak menghabiskan RAM."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        first = f.readline()
    sep = "\t" if "\t" in first else (";" if ";" in first else ",")
    hdr = [c.strip().strip("<>").strip().lower() for c in first.rstrip("\r\n").split(sep)]
    if "bid" not in hdr or "ask" not in hdr:
        sys.exit(f"[{path}] kolom BID dan ASK tidak ditemukan. Header terbaca: {hdr}")
    if not (("date" in hdr and "time" in hdr) or "datetime" in hdr or "time" in hdr):
        sys.exit(f"[{path}] kolom tanggal/waktu tidak ditemukan. Header terbaca: {hdr}")

    need = [c for c in hdr if c in ("date", "time", "datetime", "bid", "ask")]
    parts, total = [], 0
    reader = pd.read_csv(path, sep=sep, header=0, names=hdr, usecols=need, dtype=str,
                         encoding_errors="replace", chunksize=chunk_rows)
    for df in reader:
        if "date" in hdr and "time" in hdr:
            ds = df["date"].str.replace(".", "-", regex=False).str.strip() + " " + df["time"].str.strip()
        elif "datetime" in hdr:
            ds = df["datetime"].str.replace(".", "-", n=2, regex=False)
        else:
            ds = df["time"].str.replace(".", "-", n=2, regex=False)
        try:
            dt = pd.to_datetime(ds, format="ISO8601", errors="coerce")
        except (ValueError, TypeError):
            dt = pd.to_datetime(ds, errors="coerce")
        parts.append(pd.DataFrame({
            "dt": dt,
            "bid": pd.to_numeric(df["bid"], errors="coerce"),
            "ask": pd.to_numeric(df["ask"], errors="coerce"),
        }))
        total += len(df)
        print(f"  {path}: {total:,} baris dibaca ...", flush=True)
    return pd.concat(parts, ignore_index=True)


def load_ticks(files, server_offset_hours, tz_mode="utc-5", chunk_rows=2_000_000):
    frames = [read_one(p, chunk_rows) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["dt"])
    df["bid"] = df["bid"].where(df["bid"] > 0)
    df["ask"] = df["ask"].where(df["ask"] > 0)
    df = df.sort_values("dt", kind="stable").reset_index(drop=True)
    df[["bid", "ask"]] = df[["bid", "ask"]].ffill()
    df = df.dropna(subset=["bid", "ask"])

    df["dt"] = df["dt"] - pd.Timedelta(hours=server_offset_hours)
    if tz_mode == "utc-5":
        days = df["dt"].dt.normalize().values.astype("datetime64[ns]")
        uniq = np.unique(days)
        noon = pd.DatetimeIndex(uniq) + pd.Timedelta(hours=12)
        loc = noon.tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward")
        dst_flag = np.array([x.dst().total_seconds() > 0 for x in loc])
        row_dst = dst_flag[np.searchsorted(uniq, days)]
        df["dt"] = df["dt"] - pd.to_timedelta(row_dst.astype(int), unit="h")
    t = df["dt"].values.astype("datetime64[ns]")
    return t, df["bid"].to_numpy(float), df["ask"].to_numpy(float)


# --------------------------------------------------------------------------- #
# Simulasi Range & Trade
# --------------------------------------------------------------------------- #
def simulate_range(rid, sess, t, bid, ask, args):
    a, b = sess.split("-")
    sm, em = hhmm_to_min(a), hhmm_to_min(b)

    d0 = pd.Timestamp(t[0]).normalize() - pd.Timedelta(days=1)
    d1 = pd.Timestamp(t[-1]).normalize()
    days = pd.date_range(d0, d1, freq="D")
    rows = []

    for D in days:
        start = D + pd.Timedelta(minutes=sm)
        end = D + pd.Timedelta(minutes=em)
        if em <= sm:
            end += pd.Timedelta(days=1)
        nxt = start + pd.Timedelta(days=1)

        i0 = np.searchsorted(t, start.to_datetime64(), side="left")
        i1 = np.searchsorted(t, end.to_datetime64(), side="left")
        j1 = np.searchsorted(t, nxt.to_datetime64(), side="left")
        if i1 - i0 < 1 or j1 - i1 < 1:
            continue

        rb = bid[i0:i1] if args.range_price == "bid" else (bid[i0:i1] + ask[i0:i1]) / 2.0
        hi, lo = float(rb.max()), float(rb.min())
        buy_lvl, sell_lvl = hi + args.offset, lo - args.offset

        wb, wa = bid[i1:j1], ask[i1:j1]
        bi = first_true(wa >= buy_lvl)
        si = first_true(wb <= sell_lvl)
        if bi < 0 and si < 0:
            continue

        if si < 0 or (bi >= 0 and bi <= si):
            direction = "Buy"
            k = bi
            fill = float(wa[k]) if args.fill == "tick" else buy_lvl
        else:
            direction = "Sell"
            k = si
            fill = float(wb[k]) if args.fill == "tick" else sell_lvl

        start_idx = i1 + k
        end_idx = j1
        if end_idx <= start_idx:
            continue

        if direction == "Buy":
            stream_prices = bid[start_idx:end_idx]
            pnl_stream = stream_prices - fill
        else:
            stream_prices = ask[start_idx:end_idx]
            pnl_stream = fill - stream_prices

        if len(pnl_stream) == 0:
            continue

        mfe_session = max(0.0, float(pnl_stream.max()))
        mae_full = max(0.0, float(-pnl_stream.min()))
        pnl_session_close = float(pnl_stream[-1])

        # ------------------------------------------------------------- #
        # Analisis Sesi Penuh: Apakah harga pernah tembus TP di sesi ini?
        # ------------------------------------------------------------- #
        hit_tp_idx_session = first_true(pnl_stream >= args.tp)
        reached_tp_eventually = (hit_tp_idx_session >= 0)

        if reached_tp_eventually:
            # Floating minus terdalam SEBELUM menyentuh target TP
            mae_before_tp = max(0.0, float(-pnl_stream[: hit_tp_idx_session + 1].min()))
            dur_to_tp_s = float((t[start_idx + hit_tp_idx_session] - t[start_idx]) / np.timedelta64(1, "s"))
        else:
            mae_before_tp = mae_full
            dur_to_tp_s = np.nan

        # ------------------------------------------------------------- #
        # Evaluasi Hasil Trade berdasarkan Target SL dan TP
        # ------------------------------------------------------------- #
        hit_sl_idx = first_true(pnl_stream <= -args.sl) if args.sl else -1

        dur_s = np.nan
        dur_type = ""
        if args.sl:
            if hit_tp_idx_session >= 0 and (hit_sl_idx < 0 or hit_tp_idx_session < hit_sl_idx):
                result = "TP"
                pnl = args.tp
                mae = max(0.0, float(-pnl_stream[: hit_tp_idx_session + 1].min()))
                dur_s = dur_to_tp_s
                dur_type = "TP"
            elif hit_sl_idx >= 0 and (hit_tp_idx_session < 0 or hit_sl_idx <= hit_tp_idx_session):
                result = "SL"
                pnl = -args.sl
                mae = args.sl
                dur_s = float((t[start_idx + hit_sl_idx] - t[start_idx]) / np.timedelta64(1, "s"))
                dur_type = "SL"
            else:
                result = "Open"
                pnl = pnl_session_close
                mae = mae_full
                dur_type = "Open"
        else:
            if hit_tp_idx_session >= 0:
                result = "TP"
                pnl = args.tp
                mae = max(0.0, float(-pnl_stream[: hit_tp_idx_session + 1].min()))
                dur_s = dur_to_tp_s
                dur_type = "TP"
            else:
                result = "No TP"
                pnl = pnl_session_close
                mae = mae_full
                dur_type = "Open"

        spread_at_entry = float(ask[start_idx] - bid[start_idx])

        # Status "No Floating": langsung menyentuh TP tanpa pernah minus di bawah harga entry
        is_no_floating = (result == "TP" and mae <= 0.0001)

        # Durasi label (jelaskan durasi exit ke TP atau ke SL)
        if pd.notna(dur_s):
            dur_label = fmt_duration(dur_s) if dur_type == "TP" else f"{fmt_duration(dur_s)} ({dur_type})"
        else:
            dur_label = "-"

        rows.append({
            "range": rid,
            "session_date": start.date(),
            "entry_time": pd.Timestamp(t[start_idx]),
            "dir": direction,
            "range_high": hi,
            "range_low": lo,
            "entry": fill,
            "tp": fill + args.tp if direction == "Buy" else fill - args.tp,
            "sl": (fill - args.sl if direction == "Buy" else fill + args.sl) if args.sl else np.nan,
            "spread_at_entry": spread_at_entry,
            "result": result,
            "pnl": pnl,
            "mae": mae,
            "mae_full": mae_full,
            "mfe": mfe_session,
            "pnl_session_close": pnl_session_close,
            "dur_s": dur_s,
            "dur_type": dur_type,
            "dur_label": dur_label,
            "is_no_floating": is_no_floating,
            "reached_tp_eventually": reached_tp_eventually,
            "mae_before_tp": mae_before_tp,
            "dur_to_tp_s": dur_to_tp_s,
            "start_idx": start_idx,
            "end_idx": end_idx,
        })
    return rows


# --------------------------------------------------------------------------- #
# Ringkasan Statistik
# --------------------------------------------------------------------------- #
def summarize(trades, pct, sl_list, sess_label, sess_col, use_sl, tp_only):
    out = []
    for (rid, d), g in trades.groupby(["range", "dir"]):
        n = len(g)
        n_tp = int((g["result"] == "TP").sum())
        n_sl = int((g["result"] == "SL").sum())
        n_open = n - n_tp - n_sl
        base = g[g["result"] == "TP"] if tp_only else g

        row = {
            "Range": f"R{rid}",
            sess_col: sess_label[rid],
            "Arah": d,
            "Trade": n,
            "TP": n_tp,
        }
        if use_sl:
            row["SL"] = n_sl
        row["NoTP" if not use_sl else "Open"] = n_open
        row["TP%"] = round(n_tp / n * 100, 1)
        if use_sl:
            row["SL%"] = round(n_sl / n * 100, 1)

        tot_pnl = g["pnl"].sum() if len(g) else 0.0
        exp_trd = g["pnl"].mean() if len(g) else 0.0
        row["TotPnL"] = tot_pnl
        row["Exp/Trd"] = exp_trd

        # Analisis Floating Drawdown (MAE)
        row["AvgMAE"] = base["mae"].mean() if len(base) else np.nan
        row["MedMAE"] = base["mae"].median() if len(base) else np.nan
        row[f"P{pct}_MAE"] = np.percentile(base["mae"], pct) if len(base) else np.nan

        # Analisis "No Floating" & Durasi Detik ke TP
        no_float_count = int(((g["result"] == "TP") & (g["mae"] <= 0.0001)).sum())
        row["NoFloat#"] = no_float_count
        row["FastTP(s)"] = g.loc[g["result"] == "TP", "dur_s"].min() if n_tp else np.nan
        row["MedDur(s)"] = g.loc[g["result"] == "TP", "dur_s"].median() if n_tp else np.nan

        # Analisis Potensi Profit (MFE)
        row["AvgMFE"] = g["mfe"].mean() if len(g) else np.nan
        row[f"P{pct}_MFE"] = np.percentile(g["mfe"], 100 - pct) if len(g) else np.nan
        row["MaxMFE"] = g["mfe"].max() if len(g) else np.nan

        row["AvgSpread"] = g["spread_at_entry"].mean()

        for x in sl_list:
            hit = int((g["mae_full"] >= x).sum())
            row[f"SL{x:g}"] = hit
            row[f"SL{x:g}%"] = round(hit / n * 100, 1)
        out.append(row)

    df = pd.DataFrame(out)
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(2)
    return df


# --------------------------------------------------------------------------- #
# Streak & Durasi Performa (Win Streak & Loss Streak)
# --------------------------------------------------------------------------- #
def compute_streaks(trades):
    if len(trades) == 0:
        return 0, 0, 0
    trades_sorted = trades.sort_values("entry_time").reset_index(drop=True)
    cur_win, max_win = 0, 0
    cur_loss, max_loss = 0, 0
    last_streak = 0
    for pnl in trades_sorted["pnl"]:
        if pnl > 0.0001:
            cur_win += 1
            cur_loss = 0
            if cur_win > max_win:
                max_win = cur_win
            last_streak = cur_win
        elif pnl < -0.0001:
            cur_loss += 1
            cur_win = 0
            if cur_loss > max_loss:
                max_loss = cur_loss
            last_streak = -cur_loss
        else:
            cur_win = 0
            cur_loss = 0
            last_streak = 0
    return max_win, max_loss, last_streak


# --------------------------------------------------------------------------- #
# Simulasi Modal & Ketahanan Risiko (Rupiah & Lot)
# --------------------------------------------------------------------------- #
def simulate_capital(trades, initial_capital, lot, contract_size=1.0, usd_rate=16000.0):
    if len(trades) == 0 or initial_capital is None:
        return None

    point_val_idr = lot * contract_size * usd_rate
    trades_sorted = trades.sort_values("entry_time").reset_index(drop=True)
    balance = float(initial_capital)
    balance_history = [balance]
    trade_pnls_idr = []

    peak_balance = balance
    max_dd_idr = 0.0
    max_dd_pct = 0.0
    mc_trade_idx = None

    for idx, row in trades_sorted.iterrows():
        pnl_pts = float(row["pnl"])
        pnl_idr = pnl_pts * point_val_idr
        trade_pnls_idr.append(pnl_idr)

        balance += pnl_idr
        balance_history.append(balance)

        if balance > peak_balance:
            peak_balance = balance

        dd_idr = peak_balance - balance
        dd_pct = (dd_idr / peak_balance * 100.0) if peak_balance > 0 else 0.0

        if dd_idr > max_dd_idr:
            max_dd_idr = dd_idr
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

        if balance <= 0 and mc_trade_idx is None:
            mc_trade_idx = idx + 1

    final_balance = balance
    net_profit_idr = final_balance - initial_capital
    roi_pct = (net_profit_idr / initial_capital) * 100.0

    wins = [p for p in trade_pnls_idr if p > 0]
    losses = [p for p in trade_pnls_idr if p < 0]

    max_win_streak, max_loss_streak, last_streak = compute_streaks(trades_sorted)

    return {
        "initial_capital": initial_capital,
        "lot": lot,
        "contract_size": contract_size,
        "usd_rate": usd_rate,
        "point_val_idr": point_val_idr,
        "final_balance": final_balance,
        "net_profit_idr": net_profit_idr,
        "roi_pct": roi_pct,
        "max_dd_idr": max_dd_idr,
        "max_dd_pct": max_dd_pct,
        "balance_history": np.array(balance_history),
        "trade_pnls_idr": np.array(trade_pnls_idr),
        "avg_win_idr": float(np.mean(wins)) if len(wins) else 0.0,
        "avg_loss_idr": float(np.mean(losses)) if len(losses) else 0.0,
        "max_win_idr": float(np.max(wins)) if len(wins) else 0.0,
        "max_loss_idr": float(np.min(losses)) if len(losses) else 0.0,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "last_streak": last_streak,
        "mc_hit": mc_trade_idx is not None,
        "mc_trade_idx": mc_trade_idx,
    }


def print_capital_report(cap_sim, trades, tp, sl):
    if not cap_sim:
        return

    print("=" * 86)
    print("           SIMULASI PERTUMBUHAN MODAL (CAPITAL & RISK SIMULATION)")
    print("=" * 86)

    init_usd = cap_sim["initial_capital"] / cap_sim["usd_rate"]
    final_usd = cap_sim["final_balance"] / cap_sim["usd_rate"]
    profit_usd = cap_sim["net_profit_idr"] / cap_sim["usd_rate"]

    print(f"Modal Awal               : {fmt_idr(cap_sim['initial_capital'], with_sign=False)} (US$ {init_usd:,.2f})")
    print(f"Ukuran Transaksi         : {cap_sim['lot']:.2f} Lot  (Nilai 1 Poin = {fmt_idr(cap_sim['point_val_idr'], with_sign=False)} / US$ {cap_sim['lot']*cap_sim['contract_size']:.2f})")
    print("-" * 86)
    print(f"Saldo Akhir              : {fmt_idr(cap_sim['final_balance'], with_sign=False)} (US$ {final_usd:,.2f})")
    print(f"Net Profit Bersih        : {fmt_idr(cap_sim['net_profit_idr'])} (US$ {profit_usd:+,.2f})  -->  ROI: {cap_sim['roi_pct']:+.1f}%")

    if tp:
        tp_val = tp * cap_sim["point_val_idr"]
        print(f"Target Profit per TP     : +{fmt_idr(tp_val, with_sign=False)} (+{tp:g} poin x {cap_sim['lot']:.2f} lot)")
    if sl:
        sl_val = sl * cap_sim["point_val_idr"]
        risk_pct = (sl_val / cap_sim["initial_capital"]) * 100.0
        print(f"Risiko Rugi per SL       : -{fmt_idr(sl_val, with_sign=False)} (-{sl:g} poin x {cap_sim['lot']:.2f} lot)  -->  Risiko: {risk_pct:.1f}% dari modal")

    print(f"Rata-rata Menang (Win)   : {fmt_idr(cap_sim['avg_win_idr'])}")
    print(f"Rata-rata Kalah (Loss)   : {fmt_idr(cap_sim['avg_loss_idr'])}")
    print(f"Longest Win Streak       : {cap_sim['max_win_streak']} Trade Berturut-turut (Rekor Kemenangan Beruntun)")
    print(f"Longest Loss Streak      : {cap_sim['max_loss_streak']} Trade Berturut-turut (Maksimal Kalah Beruntun)")
    print(f"Max Drawdown (Penurunan) : {fmt_idr(cap_sim['max_dd_idr'], with_sign=False)} ({cap_sim['max_dd_pct']:.1f}% dari saldo tertinggi)")

    if cap_sim["mc_hit"]:
        print(f"\n[PERINGATAN RISIKO] Akun mengalami MARGIN CALL pada trade ke-{cap_sim['mc_trade_idx']}! Lot terlalu besar untuk modal ini.")
    else:
        if cap_sim["max_dd_pct"] < 15.0:
            status_text = "SANGAT AMAN / KONSERVATIF (Drawdown < 15%)"
        elif cap_sim["max_dd_pct"] < 30.0:
            status_text = "MODERAT / SEHAT (Drawdown 15-30%)"
        else:
            status_text = "AGRESIF / RISIKO TINGGI (Drawdown > 30%)"
        print(f"Status Ketahanan Modal   : {status_text}")
    print("=" * 86 + "\n")


# --------------------------------------------------------------------------- #
# Grid Matrix Optimizer (SL vs TP)
# --------------------------------------------------------------------------- #
def run_grid_optimization(trades, bid, ask, grid_tp, grid_sl, sort_by="expectancy", top_n=15, eod_mode="close", cap_sim_meta=None):
    n_trades = len(trades)
    if n_trades == 0:
        return None, None, None

    s_indices = trades["start_idx"].values
    e_indices = trades["end_idx"].values
    fills = trades["entry"].values
    is_buys = (trades["dir"].values == "Buy")
    mfes = trades["mfe"].values
    maes = trades["mae_full"].values
    close_pnls = trades["pnl_session_close"].values

    results = []
    matrix_exp = {}

    for sl in grid_sl:
        for tp in grid_tp:
            tp_count = 0
            sl_count = 0
            open_count = 0
            pnls = np.empty(n_trades, dtype=float)

            for i in range(n_trades):
                m = mfes[i]
                a = maes[i]

                if m < tp and a < sl:
                    open_count += 1
                    pnls[i] = close_pnls[i] if eod_mode == "close" else 0.0
                elif m >= tp and a < sl:
                    tp_count += 1
                    pnls[i] = tp
                elif a >= sl and m < tp:
                    sl_count += 1
                    pnls[i] = -sl
                else:
                    s_idx, e_idx = s_indices[i], e_indices[i]
                    fill = fills[i]
                    stream = (bid[s_idx:e_idx] - fill) if is_buys[i] else (fill - ask[s_idx:e_idx])

                    c_tp = stream >= tp
                    c_sl = stream <= -sl
                    i_tp = int(np.argmax(c_tp)) if c_tp.any() else -1
                    i_sl = int(np.argmax(c_sl)) if c_sl.any() else -1

                    if i_tp >= 0 and (i_sl < 0 or i_tp < i_sl):
                        tp_count += 1
                        pnls[i] = tp
                    elif i_sl >= 0 and (i_tp < 0 or i_sl <= i_tp):
                        sl_count += 1
                        pnls[i] = -sl
                    else:
                        open_count += 1
                        pnls[i] = close_pnls[i] if eod_mode == "close" else 0.0

            win_rate = (tp_count / n_trades) * 100.0
            loss_rate = (sl_count / n_trades) * 100.0
            total_pnl = float(np.sum(pnls))
            expectancy = total_pnl / n_trades

            pos = pnls[pnls > 0]
            neg = pnls[pnls < 0]
            gross_profit = float(np.sum(pos)) if len(pos) else 0.0
            gross_loss = float(abs(np.sum(neg))) if len(neg) else 0.0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)

            cum = np.cumsum(pnls)
            peaks = np.maximum.accumulate(cum)
            dd = peaks - cum
            max_dd = float(np.max(dd)) if len(dd) else 0.0

            row_data = {
                "TP": tp,
                "SL": sl,
                "RRR": round(tp / sl, 2),
                "Trades": n_trades,
                "TP#": tp_count,
                "SL#": sl_count,
                "Open#": open_count,
                "WinRate%": round(win_rate, 1),
                "LossRate%": round(loss_rate, 1),
                "ProfitFactor": round(profit_factor, 2),
                "TotalPnL": round(total_pnl, 1),
                "Exp/Trade": round(expectancy, 2),
                "MaxDD": round(max_dd, 1),
                "_pnls": pnls.copy(),
            }

            if cap_sim_meta:
                pt_val = cap_sim_meta["point_val_idr"]
                net_idr = total_pnl * pt_val
                row_data["NetProfit(IDR)"] = fmt_idr(net_idr)
                row_data["ROI%"] = round((net_idr / cap_sim_meta["initial_capital"]) * 100, 1)

            results.append(row_data)
            matrix_exp[(sl, tp)] = expectancy

    df_res = pd.DataFrame(results)

    sort_cols = {
        "expectancy": ["Exp/Trade", "ProfitFactor", "TotalPnL"],
        "pnl": ["TotalPnL", "Exp/Trade", "ProfitFactor"],
        "pf": ["ProfitFactor", "Exp/Trade", "TotalPnL"],
        "winrate": ["WinRate%", "Exp/Trade", "ProfitFactor"],
    }
    df_sorted = df_res.sort_values(sort_cols.get(sort_by, ["Exp/Trade"]), ascending=False).reset_index(drop=True)

    best_info = None
    if len(df_sorted) > 0:
        top_row = df_sorted.iloc[0]
        best_equity = np.cumsum(top_row["_pnls"])
        best_label = f"TP={top_row['TP']:g}, SL={top_row['SL']:g} (Exp: +{top_row['Exp/Trade']:g} pts/trd)"
        best_info = (best_equity, best_label)

    df_sorted = df_sorted.drop(columns=["_pnls"])
    df_sorted.index = df_sorted.index + 1
    df_sorted.index.name = "Rank"

    mat_data = {}
    for tp in grid_tp:
        col_name = f"TP_{tp:g}"
        mat_data[col_name] = [matrix_exp.get((sl, tp), np.nan) for sl in grid_sl]
    df_matrix = pd.DataFrame(mat_data, index=[f"SL_{sl:g}" for sl in grid_sl])
    df_matrix = df_matrix.round(2)

    return df_sorted, df_matrix, best_info


# --------------------------------------------------------------------------- #
# Analisis & Rekomendasi Sweet Spot
# --------------------------------------------------------------------------- #
def print_recommendations(trades, grid_sorted, pct):
    print("=" * 86)
    print("          REKOMENDASI PARAMETER SL & TP (SWEET SPOT) BERDASARKAN DATA TICK")
    print("=" * 86)

    tp_trades = trades[trades["result"] == "TP"]
    base_mae = tp_trades["mae"] if len(tp_trades) > 0 else trades["mae_full"]

    med_mae = base_mae.median()
    p80_mae = np.percentile(base_mae, 80)
    p90_mae = np.percentile(base_mae, 90)
    max_mae = base_mae.max()

    print(f"\n[1] REKOMENDASI STOP LOSS (Berdasarkan Toleransi Floating MAE):")
    print(f"    - Median Floating Drawdown : {med_mae:.2f} poin")
    print(f"    - Persentil 80% (P80 MAE)   : {p80_mae:.2f} poin")
    print(f"    - Persentil 90% (P90 MAE)   : {p90_mae:.2f} poin")
    print(f"    - Floating Terjauh (Pemenang): {max_mae:.2f} poin")
    print(f"    -> REKOMENDASI SL IDEAL    : {p80_mae:.1f} - {p90_mae:.1f} poin")
    print(f"       (Alasan: 80-90% trade yang berhasil mencapai target tidak pernah floating")
    print(f"        melebihi batas ini. Memasang SL di atas {p90_mae:.1f} poin hanya menambah risiko")
    print(f"        tanpa menambah persentase kemenangan secara signifikan).")

    all_mfe = trades["mfe"]
    med_mfe = all_mfe.median()
    p80_mfe = np.percentile(all_mfe, 20)
    max_mfe = all_mfe.max()

    print(f"\n[2] REKOMENDASI TAKE PROFIT MAKSIMAL (Berdasarkan Potensi MFE Breakout):")
    print(f"    - Target Konservatif (80% trade tembus) : {p80_mfe:.2f} poin (Scalping / High Winrate)")
    print(f"    - Target Moderat     (50% trade tembus) : {med_mfe:.2f} poin (Balanced Swing)")
    print(f"    - Potensi Lari Maksimal (Peak MFE)      : {max_mfe:.2f} poin")
    print(f"    -> REKOMENDASI TP MAKSIMAL : {p80_mfe:.1f} poin (Konservatif) s/d {med_mfe:.1f} poin (Optimal)")

    if grid_sorted is not None and len(grid_sorted) > 0:
        best_exp = grid_sorted.iloc[0]
        rrr_ge_1 = grid_sorted[grid_sorted["RRR"] >= 1.0]
        best_rrr = rrr_ge_1.iloc[0] if len(rrr_ge_1) > 0 else best_exp

        print(f"\n[3] KOMBINASI TERBAIK BERDASARKAN MATRIX OPTIMIZER:")
        print(f"    * Juara 1 Expectancy (Nilai Harapan Tertinggi per Trade):")
        print(f"      -> SL: {best_exp['SL']} poin | TP: {best_exp['TP']} poin | RRR 1:{best_exp['RRR']}")
        print(f"         Expectancy : +{best_exp['Exp/Trade']} poin/trade")
        print(f"         Win Rate   : {best_exp['WinRate%']}% | Profit Factor: {best_exp['ProfitFactor']}")
        print(f"         Total PnL  : +{best_exp['TotalPnL']} poin | Max Drawdown: {best_exp['MaxDD']} poin")

        if best_rrr["TP"] != best_exp["TP"] or best_rrr["SL"] != best_exp["SL"]:
            print(f"    * Juara Alternatif (Healthy RRR >= 1.0):")
            print(f"      -> SL: {best_rrr['SL']} poin | TP: {best_rrr['TP']} poin | RRR 1:{best_rrr['RRR']}")
            print(f"         Expectancy : +{best_rrr['Exp/Trade']} poin/trade | Win Rate: {best_rrr['WinRate%']}% | PF: {best_rrr['ProfitFactor']}")

    print("=" * 86 + "\n")


# --------------------------------------------------------------------------- #
# Laporan PDF Visual Profesional
# --------------------------------------------------------------------------- #
def generate_pdf_report(pdf_path, meta, trades, summary_df, grid_sorted=None, df_matrix=None, best_info=None, cap_sim=None, args=None, ranges=None, sess_label=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("\n[Peringatan] Library matplotlib tidak ditemukan. PDF tidak dapat dibuat.")
        print("Silakan jalankan: pip install matplotlib")
        return

    print(f"\nMembuat laporan PDF visual: {pdf_path} ...", flush=True)

    with PdfPages(pdf_path) as pdf:
        # =====================================================================
        # HALAMAN 1: Executive Summary, Parameter Input, Performa Modal & History
        # =====================================================================
        fig1 = plt.figure(figsize=(11.69, 8.27), dpi=150)
        gs1 = gridspec.GridSpec(4, 2, height_ratios=[0.88, 2.30, 2.30, 2.30], hspace=0.36, wspace=0.25)

        # 1. Header Banner
        ax_hdr = fig1.add_subplot(gs1[0, :])
        ax_hdr.axis("off")
        ax_hdr.add_patch(plt.Rectangle((0, 0), 1, 1, color="#0F172A", transform=ax_hdr.transAxes, zorder=1))
        ax_hdr.text(0.5, 0.65, "USTEC RANGE BREAKOUT - LAPORAN ANALISIS KUANTITATIF",
                    ha="center", va="center", color="white", fontsize=13.5, fontweight="bold", transform=ax_hdr.transAxes, zorder=2)

        subtitle = f"Periode: {meta.get('date_range', '-')}  |  Total Data Ticks: {meta.get('total_ticks', 0):,}  |  Total Trades: {len(trades):,}  |  Zona Waktu: {meta.get('tzname', 'UTC-5')}"
        ax_hdr.text(0.5, 0.25, subtitle, ha="center", va="center", color="#94A3B8", fontsize=8.5, transform=ax_hdr.transAxes, zorder=2)

        # 2. Card Kiri: Parameter Input (Syntax Python yang Diinput)
        ax_prm = fig1.add_subplot(gs1[1, 0])
        ax_prm.axis("off")
        ax_prm.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.2, transform=ax_prm.transAxes))
        ax_prm.text(0.04, 0.88, "PARAMETER YANG DIINPUT (COMMAND SYNTAX)", fontsize=9.5, fontweight="bold", color="#1E293B", transform=ax_prm.transAxes)

        if args and args.only:
            try:
                r_num = int(args.only)
                r_sess = sess_label.get(r_num, "")
                a, b = DEFAULT_RANGES[r_num].split("-")
                sm, em = hhmm_to_min(a), hhmm_to_min(b)
                wsm, wem = (sm + 720) % 1440, (em + 720) % 1440
                range_str = f"Range {r_num} ({r_sess} NY / {fmt_time_min(wsm)}-{fmt_time_min(wem)} WIB)"
            except Exception:
                range_str = f"Range {args.only}"
        else:
            range_str = "Semua Range (Range 1 s/d 7)"

        sl_str = f"SL {args.sl:g} Poin" if (args and args.sl) else "Tanpa SL (Biarkan sampai akhir sesi)"
        tp_str = f"TP {args.tp:g} Poin" if (args and args.tp) else "Default 8.0 Poin"
        rrr_calc = f"(RRR 1 : {args.tp/args.sl:.2f})" if (args and args.sl and args.sl > 0) else ""

        cap_str = fmt_idr(cap_sim["initial_capital"], with_sign=False) if cap_sim else "Tidak ditentukan"
        lot_str = f"{cap_sim['lot']:.2f} Lot" if cap_sim else "-"
        pt_val_str = fmt_idr(cap_sim["point_val_idr"], with_sign=False) if cap_sim else "-"

        prm_text = (
            f"• Sesi / Range      : {range_str}\n"
            f"• Target TP & SL    : {tp_str}  |  {sl_str} {rrr_calc}\n"
            f"• Simulasi Modal    : {cap_str}  (Ukuran: {lot_str})\n"
            f"• Nilai Per 1 Poin  : {pt_val_str} per poin USTEC\n"
            f"• Mode Eksekusi     : Fill {args.fill.upper() if args else 'TICK'} (Real Slippage)  |  Offset: {args.offset if args else 0.0:g} poin\n"
            f"• Jendela Transaksi : OCO Pending Stop (Maks. 1 posisi per sesi)"
        )
        ax_prm.text(0.04, 0.44, prm_text, fontsize=8.0, color="#334155", va="center", transform=ax_prm.transAxes, linespacing=1.28)

        # 3. Card Kanan: Performa Modal Nyata & Karakteristik Trade
        ax_prf = fig1.add_subplot(gs1[1, 1])
        ax_prf.axis("off")
        ax_prf.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.2, transform=ax_prf.transAxes))
        ax_prf.text(0.04, 0.88, "PERFORMA MODAL & KARAKTERISTIK TRADE", fontsize=9.5, fontweight="bold", color="#1E293B", transform=ax_prf.transAxes)

        no_float_total = int(trades["is_no_floating"].sum()) if "is_no_floating" in trades.columns else 0
        no_float_pct = (no_float_total / len(trades) * 100.0) if len(trades) > 0 else 0.0

        tp_hits = trades[trades["result"] == "TP"]
        fastest_tp = tp_hits["dur_s"].min() if len(tp_hits) > 0 else np.nan
        med_tp = tp_hits["dur_s"].median() if len(tp_hits) > 0 else np.nan

        # Analisis SL yang berbalik tembus TP
        sl_all = trades[trades["result"] == "SL"]
        sl_saved_all = sl_all[sl_all["reached_tp_eventually"]]
        n_saved = len(sl_saved_all)
        avg_saved_mae = sl_saved_all["mae_before_tp"].mean() if n_saved else 0.0

        avg_peak_all_pdf = trades["mfe"].mean() if len(trades) else 0.0
        max_peak_all_pdf = trades["mfe"].max() if len(trades) else 0.0
        max_win_pdf, max_loss_pdf, _ = compute_streaks(trades)

        if cap_sim:
            risk_pct = (args.sl * cap_sim["point_val_idr"] / cap_sim["initial_capital"] * 100.0) if (args and args.sl) else 0.0
            prf_text = (
                f"• Saldo Akhir       : {fmt_idr(cap_sim['final_balance'], with_sign=False)} ({fmt_idr(cap_sim['net_profit_idr'])} / ROI: {cap_sim['roi_pct']:+.1f}%)\n"
                f"• Target Profit TP  : +{fmt_idr(args.tp * cap_sim['point_val_idr'], with_sign=False)} (+{args.tp:g} poin x {cap_sim['lot']:.2f} lot)\n"
                f"• Risiko Rugi SL    : -{fmt_idr(args.sl * cap_sim['point_val_idr'] if args and args.sl else 0, with_sign=False)} ({risk_pct:.1f}% risiko modal per trade)\n"
                f"• Max Drawdown      : {fmt_idr(cap_sim['max_dd_idr'], with_sign=False)} ({cap_sim['max_dd_pct']:.1f}% dari puncak saldo - AMAN)\n"
                f"• Win / Loss Streak : Rekor Win: {max_win_pdf}x beruntun  |  Loss: {max_loss_pdf}x beruntun\n"
                f"• Rata2 Peak Profit : +{avg_peak_all_pdf:.1f} poin (Tertinggi: +{max_peak_all_pdf:.1f} poin)\n"
                f"• Rekor No Floating : {no_float_total} Trade ({no_float_pct:.1f}% langsung tembus TP tanpa minus!)\n"
                f"• SL Tembus TP Nanti: {n_saved} Trade SL sebenarnya tembus TP (Rata-rata koreksi: {avg_saved_mae:.1f} poin)"
            )
        else:
            base_mae = tp_hits["mae"] if len(tp_hits) > 0 else trades["mae_full"]
            p80_mae = np.percentile(base_mae, 80) if len(base_mae) else 0.0
            p80_mfe = np.percentile(trades["mfe"], 20) if len(trades) else 0.0
            prf_text = (
                f"• Total Keuntungan  : {trades['pnl'].sum():+.2f} Poin (Exp: {trades['pnl'].mean():+.2f} pts/trd)\n"
                f"• Win Rate Strategi : {(trades['result'] == 'TP').sum() / len(trades) * 100:.1f}%\n"
                f"• Win / Loss Streak : Rekor Win: {max_win_pdf}x beruntun  |  Loss: {max_loss_pdf}x beruntun\n"
                f"• Rata2 Peak Profit : +{avg_peak_all_pdf:.1f} poin (Tertinggi: +{max_peak_all_pdf:.1f} poin)\n"
                f"• Rekor No Floating : {no_float_total} Trade ({no_float_pct:.1f}% langsung tembus TP tanpa minus!)\n"
                f"• Kecepatan Hit TP  : Rata-rata: {fmt_duration(med_tp)}  |  Tercepat: {fmt_duration(fastest_tp)}\n"
                f"• SL Tembus TP Nanti: {n_saved} Trade SL sebenarnya tembus TP (Rata-rata koreksi: {avg_saved_mae:.1f} poin)\n"
                f"• Batas TP Maksimal : P80 MFE = {p80_mfe:.1f} poin (Target tembus 80% trade)"
            )
        ax_prf.text(0.04, 0.44, prf_text, fontsize=8.0, color="#334155", va="center", transform=ax_prf.transAxes, linespacing=1.28)

        # 4. Table 1: Ringkasan Per Range & Arah (Tengah Bawah)
        ax_tbl1 = fig1.add_subplot(gs1[2, :])
        ax_tbl1.axis("off")
        ax_tbl1.text(0.0, 1.05, "Ringkasan Statistik Performa per Range & Arah", fontsize=9.5, fontweight="bold", color="#1E293B", transform=ax_tbl1.transAxes)

        t1_cols = ["Range", meta.get("sess_col", "Sesi UTC-5"), "Arah", "Trade", "TP", "SL", "TP%", "TotPnL", "Exp/Trd", "NoFloat#", "FastTP(s)", "MedDur(s)", "P80_MAE", "P80_MFE"]
        avail_cols = [c for c in t1_cols if c in summary_df.columns]
        sub_tbl1 = summary_df[avail_cols].copy()

        rename_map = {
            meta.get("sess_col", "Sesi UTC-5"): "Sesi (NY)",
            "Trade": "Trades",
            "TP%": "Win%",
            "TotPnL": "Net PnL",
            "NoFloat#": "NoFloat#",
            "FastTP(s)": "Fast TP",
            "MedDur(s)": "Med Dur",
            "P80_MAE": "P80 MAE",
            "P80_MFE": "P80 MFE",
        }
        sub_tbl1.rename(columns=rename_map, inplace=True)

        tbl1 = ax_tbl1.table(cellText=sub_tbl1.values, colLabels=sub_tbl1.columns, loc="center", cellLoc="center")
        tbl1.auto_set_font_size(False)
        tbl1.set_fontsize(7.5)
        tbl1.scale(1.0, 1.35)
        for (r, c), cell in tbl1.get_celld().items():
            if r == 0:
                cell.set_facecolor("#0F172A")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#F8FAFC" if r % 2 == 0 else "white")
                cell.set_edgecolor("#E2E8F0")

        # 5. Table 2: Sample History Detail Trade Terakhir (Durasi Exit, Floating Posisi, Max Float sblm TP)
        ax_tbl2 = fig1.add_subplot(gs1[3, :])
        ax_tbl2.axis("off")
        ax_tbl2.text(0.0, 1.05, "Sample History Detail Trade (Durasi Exit, Floating Posisi & Max Float sebelum Tembus TP)", fontsize=9.5, fontweight="bold", color="#1E293B", transform=ax_tbl2.transAxes)

        sample_trades = trades.tail(6).iloc[::-1].copy()
        if len(sample_trades) > 0:
            sample_trades["Waktu Entry"] = sample_trades["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
            sample_trades["Entry"] = sample_trades["entry"].round(1)
            sample_trades["TP"] = sample_trades["tp"].round(1)
            sample_trades["Hasil"] = sample_trades["result"]
            sample_trades["Durasi Exit"] = sample_trades["dur_label"]

            def get_fpos(r):
                if r["result"] == "TP":
                    return "No Floating" if r["mae"] <= 0.0001 else f"-{r['mae']:.1f} poin"
                elif r["result"] == "SL":
                    return f"-{args.sl:.1f} (SL)" if args and args.sl else f"-{r['mae']:.1f}"
                else:
                    return f"-{r['mae_full']:.1f} poin"

            def get_fbefore_tp(r):
                if r["result"] == "TP":
                    return "No Floating" if r["mae"] <= 0.0001 else f"-{r['mae']:.1f} poin"
                elif r["reached_tp_eventually"]:
                    return f"-{r['mae_before_tp']:.1f} (Tembus TP)"
                else:
                    return f"Gagal ke TP (+{r['mfe']:.1f})"

            sample_trades["Floating Posisi"] = sample_trades.apply(get_fpos, axis=1)
            sample_trades["Max Float sblm TP"] = sample_trades.apply(get_fbefore_tp, axis=1)
            sample_trades["Peak Profit"] = "+" + sample_trades["mfe"].round(1).astype(str) + " pts"
            sample_trades["Spread"] = sample_trades["spread_at_entry"].round(1)

            cols_h_show = ["Waktu Entry", "dir", "Entry", "TP", "Hasil", "Durasi Exit", "Floating Posisi", "Max Float sblm TP", "Peak Profit", "Spread"]
            cols_h_rename = {"dir": "Arah"}
            sub_tbl2 = sample_trades[cols_h_show].rename(columns=cols_h_rename)

            tbl2 = ax_tbl2.table(cellText=sub_tbl2.values, colLabels=sub_tbl2.columns, loc="center", cellLoc="center")
            tbl2.auto_set_font_size(False)
            tbl2.set_fontsize(7.2)
            tbl2.scale(1.0, 1.35)
            for (r, c), cell in tbl2.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#1E293B")
                    cell.set_text_props(color="white", fontweight="bold")
                else:
                    row_idx = r - 1
                    is_nf = sample_trades.iloc[row_idx]["is_no_floating"]
                    is_saved = (sample_trades.iloc[row_idx]["result"] == "SL" and sample_trades.iloc[row_idx]["reached_tp_eventually"])
                    if is_nf:
                        cell.set_facecolor("#F0FDF4")  # Hijau lembut No Floating
                    elif is_saved:
                        cell.set_facecolor("#FEF3C7")  # Kuning lembut Tembus TP Nanti
                    else:
                        cell.set_facecolor("#F8FAFC" if r % 2 == 0 else "white")
                    cell.set_edgecolor("#E2E8F0")

            if n_saved > 0:
                pct_saved = (n_saved / len(sl_all) * 100.0) if len(sl_all) > 0 else 0.0
                p80_saved_mae = np.percentile(sl_saved_all["mae_before_tp"], 80) if n_saved else 0.0
                note_sl = (
                    f"* Analisis Koreksi & Peak Profit: Dari {len(sl_all)} trade SL, {n_saved} trade ({pct_saved:.1f}%) berbalik tembus TP (Rata-rata koreksi: -{avg_saved_mae:.1f} poin, P80: -{p80_saved_mae:.1f} poin). "
                    f"Rata-rata Peak Profit (MFE): +{avg_peak_all_pdf:.1f} poin (Maks: +{max_peak_all_pdf:.1f} poin). "
                    f"Baris kuning = trade yang sebenarnya tembus TP jika SL > {p80_saved_mae:.1f} poin."
                )
                ax_tbl2.text(0.0, -0.09, note_sl, fontsize=7.2, color="#B45309", fontweight="bold", transform=ax_tbl2.transAxes)

        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        # =====================================================================
        # HALAMAN 2: Grid Matrix Optimization (SL vs TP) jika aktif
        # =====================================================================
        if grid_sorted is not None and df_matrix is not None:
            fig2 = plt.figure(figsize=(11.69, 8.27), dpi=150)
            gs2 = gridspec.GridSpec(3, 2, height_ratios=[0.88, 2.8, 3.2], width_ratios=[1.2, 1.0], hspace=0.36, wspace=0.25)

            ax_hdr2 = fig2.add_subplot(gs2[0, :])
            ax_hdr2.axis("off")
            ax_hdr2.add_patch(plt.Rectangle((0, 0), 1, 1, color="#0F172A", transform=ax_hdr2.transAxes, zorder=1))
            ax_hdr2.text(0.5, 0.65, "GRID MATRIX OPTIMIZER - PETA KOMBINASI SL vs TP TERBAIK",
                         ha="center", va="center", color="white", fontsize=13.5, fontweight="bold", transform=ax_hdr2.transAxes, zorder=2)

            sub_grid = f"Total Kombinasi Diuji: {len(df_matrix.columns)*len(df_matrix.index)} Matrix Pairs  |  Disortir Berdasarkan: {args.grid_by.upper() if args else 'EXPECTANCY'}"
            if cap_sim:
                sub_grid += f"  |  Modal Acuan: {fmt_idr(cap_sim['initial_capital'], with_sign=False)} ({cap_sim['lot']:.2f} Lot)"
            ax_hdr2.text(0.5, 0.25, sub_grid, ha="center", va="center", color="#94A3B8", fontsize=8.5, transform=ax_hdr2.transAxes, zorder=2)

            # 1. Heatmap Expectancy Matrix (Kiri, Full Height)
            ax_heat = fig2.add_subplot(gs2[1:, 0])
            mat_vals = df_matrix.values
            v_max = max(abs(np.nanmin(mat_vals)), abs(np.nanmax(mat_vals)), 1.0)
            im = ax_heat.imshow(mat_vals, cmap="RdYlGn", aspect="auto", vmin=-v_max, vmax=v_max)

            ax_heat.set_xticks(range(len(df_matrix.columns)))
            ax_heat.set_xticklabels(df_matrix.columns, fontsize=8, rotation=35)
            ax_heat.set_yticks(range(len(df_matrix.index)))
            ax_heat.set_yticklabels(df_matrix.index, fontsize=8)
            ax_heat.set_title("Heatmap Expectancy (Poin / Trade) - Area Hijau = Paling Menguntungkan", fontsize=9.5, fontweight="bold", pad=8)
            ax_heat.set_xlabel("Target Take Profit (TP)", fontsize=8.5)
            ax_heat.set_ylabel("Stop Loss (SL)", fontsize=8.5)

            for i in range(len(df_matrix.index)):
                for j in range(len(df_matrix.columns)):
                    val = mat_vals[i, j]
                    if not np.isnan(val):
                        sign = "+" if val > 0 else ""
                        color = "white" if abs(val) > v_max * 0.65 else "black"
                        ax_heat.text(j, i, f"{sign}{val:.1f}", ha="center", va="center", color=color, fontsize=7.5, fontweight="bold")

            cbar = fig2.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.03)
            cbar.ax.tick_params(labelsize=8)

            # 2. Kurva Equity Curve Juara 1 (Kanan Atas)
            ax_eq = fig2.add_subplot(gs2[1, 1])
            if cap_sim and cap_sim["balance_history"] is not None:
                bal_history_jt = cap_sim["balance_history"] / 1_000_000
                ax_eq.plot(bal_history_jt, color="#2563EB", lw=2, label="Saldo Akun")
                ax_eq.fill_between(range(len(bal_history_jt)), bal_history_jt, cap_sim["initial_capital"]/1_000_000,
                                   color="#2563EB", alpha=0.15)
                ax_eq.axhline(cap_sim["initial_capital"]/1_000_000, color="gray", linestyle="--", lw=1)
                ax_eq.set_title(f"Kurva Pertumbuhan Modal (Awal: {fmt_idr(cap_sim['initial_capital'], with_sign=False)})", fontsize=9.5, fontweight="bold", color="#1E293B")
                ax_eq.set_xlabel("Urutan Trade", fontsize=8)
                ax_eq.set_ylabel("Saldo (Juta Rp)", fontsize=8)
                ax_eq.grid(True, linestyle="--", alpha=0.4)
                ax_eq.tick_params(labelsize=8)
            elif best_info and best_info[0] is not None and len(best_info[0]) > 0:
                eq_curve = best_info[0]
                ax_eq.plot(eq_curve, color="#2563EB", lw=2, label="Kumulatif PnL (Poin)")
                ax_eq.fill_between(range(len(eq_curve)), eq_curve, color="#2563EB", alpha=0.15)
                ax_eq.axhline(0, color="gray", linestyle="--", lw=1)
                ax_eq.set_title(f"Kurva Pertumbuhan Akun (Juara 1: {best_info[1]})", fontsize=9.5, fontweight="bold", color="#1E293B")
                ax_eq.set_xlabel("Urutan Trade", fontsize=8)
                ax_eq.set_ylabel("Net PnL (Poin)", fontsize=8)
                ax_eq.grid(True, linestyle="--", alpha=0.4)
                ax_eq.tick_params(labelsize=8)
            else:
                ax_eq.axis("off")

            # 3. Top 10 Kombinasi Table (Kanan Bawah)
            ax_top = fig2.add_subplot(gs2[2, 1])
            ax_top.axis("off")
            ax_top.text(0.0, 1.05, "Top 10 Kombinasi Parameter Terbaik", fontsize=9.5, fontweight="bold", color="#1E293B", transform=ax_top.transAxes)

            top_cols = ["TP", "SL", "RRR", "WinRate%", "ProfitFactor", "TotalPnL", "Exp/Trade", "MaxDD"]
            if "NetProfit(IDR)" in grid_sorted.columns:
                top_cols = ["TP", "SL", "RRR", "WinRate%", "ProfitFactor", "NetProfit(IDR)", "Exp/Trade", "MaxDD"]

            top_show = grid_sorted.head(10)[top_cols].copy()
            if "NetProfit(IDR)" in top_show.columns:
                top_show.columns = ["TP", "SL", "RRR", "Win%", "PF", "Profit(IDR)", "Exp/Trd", "MaxDD"]
            else:
                top_show.columns = ["TP", "SL", "RRR", "Win%", "PF", "TotPnL", "Exp/Trd", "MaxDD"]

            top_tbl = ax_top.table(cellText=top_show.values, colLabels=top_show.columns, loc="center", cellLoc="center")
            top_tbl.auto_set_font_size(False)
            top_tbl.set_fontsize(7.5)
            top_tbl.scale(1.0, 1.35)

            for (r, c), cell in top_tbl.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#1E293B")
                    cell.set_text_props(color="white", fontweight="bold")
                else:
                    if r == 1:
                        cell.set_facecolor("#DCFCE7")
                        cell.set_text_props(fontweight="bold")
                    elif r % 2 == 0:
                        cell.set_facecolor("#F8FAFC")
                    else:
                        cell.set_facecolor("white")
                    cell.set_edgecolor("#E2E8F0")

            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

    print(f"Laporan PDF berhasil dibuat: {pdf_path}")


# --------------------------------------------------------------------------- #
# Main Entry Point
# --------------------------------------------------------------------------- #
# --- END CORE ENGINE ---



# -----------------------------------------------------------------------------
# KONFIGURASI HALAMAN STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="USTEC Breakout Hub - Strategy Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling CSS
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 18px;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-val {
        font-size: 22px;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 12px;
        color: #334155;
        margin-top: 2px;
    }
    .badge-win {
        background-color: #DCFCE7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
    .badge-loss {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
    .badge-saved {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        font-weight: 600;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# FUNGSI CACHE DATA TICK
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="⚡ Membaca & memproses jutaan tick ke memori...")
def get_cached_ticks(file_path, server_offset, tz_mode):
    t, bid, ask = load_ticks([file_path], server_offset, tz_mode)
    return t, bid, ask


# -----------------------------------------------------------------------------
# SIDEBAR KONTROL & PARAMETER
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ USTEC Strategy Hub")
    st.caption("Intraday Range Breakout & Risk Engine")
    st.markdown("---")

    # 1. Sumber Data CSV Tick
    st.markdown("#### 📂 Sumber Data Tick")
    default_csv = "ticks_USTEC.csv"
    data_mode = st.radio(
        "Pilih Data:",
        ["Gunakan File Lokal di Folder", "Upload File CSV Baru"],
        index=0,
        label_visibility="collapsed"
    )

    tick_file_to_load = None
    if data_mode == "Gunakan File Lokal di Folder":
        local_path = st.text_input("Path File CSV:", value=default_csv)
        if os.path.exists(local_path):
            tick_file_to_load = local_path
            file_sz_mb = os.path.getsize(local_path) / (1024 * 1024)
            st.success(f"File ditemukan: {os.path.basename(local_path)} ({file_sz_mb:.1f} MB)")
        else:
            st.error(f"File '{local_path}' tidak ditemukan di folder kerja.")
    else:
        uploaded_file = st.file_uploader("Upload CSV Ticks MT5:", type=["csv"])
        if uploaded_file is not None:
            # Simpan temporary
            temp_path = os.path.join(CURRENT_DIR, "_uploaded_ticks.csv")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            tick_file_to_load = temp_path
            st.success(f"File berhasil diunggah: {uploaded_file.name}")

    st.markdown("---")

    # 2. Pemilihan Range
    st.markdown("#### 🕒 Pemilihan Range")
    range_options = {
        4: "Range 4: 21:52 - 18:04 (+1d) [09:52 - 06:04 WIB] (Paling Populer)",
        1: "Range 1: 00:00 - 08:30 [12:00 - 20:30 WIB]",
        2: "Range 2: 08:30 - 09:30 [20:30 - 21:30 WIB] (US Open)",
        3: "Range 3: 09:30 - 16:00 [21:30 - 04:00 WIB]",
        5: "Range 5: 18:04 - 19:33 [06:04 - 07:33 WIB]",
        6: "Range 6: 19:33 - 21:13 [07:33 - 09:13 WIB]",
        7: "Range 7: 21:13 - 21:52 [09:13 - 09:52 WIB]",
        0: "Semua Range (Range 1 s/d 7)",
    }
    selected_range_key = st.selectbox(
        "Pilih Sesi / Range:",
        options=list(range_options.keys()),
        format_func=lambda k: range_options[k],
        index=0  # Default Range 4
    )

    st.markdown("---")

    # 3. Parameter Strategi (TP, SL, Fill)
    st.markdown("#### 🎯 Parameter Target TP & SL")
    col_tp, col_sl = st.columns(2)
    with col_tp:
        tp_input = st.number_input("Target TP (Poin):", min_value=1.0, max_value=200.0, value=15.0, step=1.0)
    with col_sl:
        use_sl = st.checkbox("Pakai Stop Loss", value=True)
        sl_input = st.number_input("Target SL (Poin):", min_value=1.0, max_value=100.0, value=8.0, step=1.0, disabled=not use_sl)

    fill_mode = st.selectbox("Mode Eksekusi Fill:", ["tick (Real Slippage)", "level (Pending Level)"], index=0)
    fill_arg = "tick" if "tick" in fill_mode else "level"

    st.markdown("---")

    # 4. Simulasi Modal & Lot (Rupiah)
    st.markdown("#### 💰 Simulasi Modal & Ukuran Lot")
    capital_input = st.number_input("Modal Akun (Rupiah):", min_value=1_000_000, max_value=1_000_000_000, value=8_000_000, step=1_000_000, format="%d")
    col_lot, col_kurs = st.columns(2)
    with col_lot:
        lot_input = st.number_input("Ukuran Lot:", min_value=0.01, max_value=100.0, value=3.0, step=0.1)
    with col_kurs:
        kurs_input = st.number_input("Kurs USD/IDR:", min_value=10_000, max_value=25_000, value=16_000, step=500)

    st.markdown("---")

    # 5. Fitur Optimizer & PDF
    st.markdown("#### 🧩 Fitur Analisis Tambahan")
    opt_grid = st.checkbox("Jalankan Grid Matrix Optimizer (SL vs TP)", value=True)
    opt_pdf = st.checkbox("Buat File Laporan PDF Siap Cetak", value=True)

    btn_run = st.button("🚀 JALANKAN ANALISIS", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# MAIN DASHBOARD AREA
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">⚡ USTEC Breakout Strategy Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Platform Kuantitatif & Simulator Risiko Intraday Range Breakout US Tech 100 (Exness/CFD)</div>', unsafe_allow_html=True)

if not tick_file_to_load or not os.path.exists(tick_file_to_load):
    st.info("👋 Silakan pilih atau upload file CSV tick di menu samping kiri untuk memulai analisis.")
    st.stop()

# Siapkan parameter backtest
args = SimpleNamespace(
    tp=tp_input,
    sl=sl_input if use_sl else None,
    fill=fill_arg,
    offset=0.0,
    range_price="bid",
    sl_test="",
    float_tp_only=False,
    pct=80,
    tz="utc-5",
    server_offset=0,
    chunk_rows=2_000_000,
    grid_by="expectancy",
    grid_top=15,
    grid_eod="close",
    grid_tp="5,8,10,12,15,20,25,30,40,50",
    grid_sl="5,8,10,15,20,25,30,40,50",
    last=0
)

# Tentukan range yang diuji
ranges = dict(DEFAULT_RANGES)
if selected_range_key != 0:
    ranges = {selected_range_key: DEFAULT_RANGES[selected_range_key]}

sess_label = {}
for rid, sess in ranges.items():
    a, b = sess.split("-")
    sm, em = hhmm_to_min(a), hhmm_to_min(b)
    sess_label[rid] = f"{(sm//60)%24:02d}:{sm%60:02d}-{(em//60)%24:02d}:{em%60:02d}"

# Load Ticks (Menggunakan cache cepat)
try:
    t, bid, ask = get_cached_ticks(tick_file_to_load, args.server_offset, args.tz)
except Exception as e:
    st.error(f"Gagal memproses file tick: {e}")
    st.stop()

# Eksekusi Simulasi Range
with st.spinner("Menghitung backtest dan logika eksekusi real tick order..."):
    rows = []
    for rid, sess in sorted(ranges.items()):
        rows += simulate_range(rid, sess, t, bid, ask, args)

if not rows:
    st.warning("Tidak ada trade yang terpicu pada rentang waktu atau data yang dipilih.")
    st.stop()

trades = pd.DataFrame(rows).sort_values(["range", "entry_time"]).reset_index(drop=True)
cap_sim = simulate_capital(trades, capital_input, lot_input, contract_size=1.0, usd_rate=kurs_input)
summary_df = summarize(trades, args.pct, [], sess_label, "Sesi UTC-5", bool(args.sl), False)

# Hitung Metrik Utama
total_trades = len(trades)
tp_trades = trades[trades["result"] == "TP"]
sl_trades = trades[trades["result"] == "SL"]
win_rate = (len(tp_trades) / total_trades * 100.0) if total_trades else 0.0

no_float_total = int(trades["is_no_floating"].sum())
no_float_pct = (no_float_total / total_trades * 100.0) if total_trades else 0.0

sl_saved = sl_trades[sl_trades["reached_tp_eventually"]]
n_sl_saved = len(sl_saved)
pct_sl_saved = (n_sl_saved / len(sl_trades) * 100.0) if len(sl_trades) else 0.0
avg_saved_mae = sl_saved["mae_before_tp"].mean() if n_sl_saved else 0.0
p80_saved_mae = np.percentile(sl_saved["mae_before_tp"], 80) if n_sl_saved else 0.0

avg_peak_all = trades["mfe"].mean() if total_trades else 0.0
max_peak_all = trades["mfe"].max() if total_trades else 0.0
max_win_streak, max_loss_streak, last_streak = compute_streaks(trades)

# Optimasi Grid Matrix jika diaktifkan
grid_sorted, df_matrix, best_info = None, None, None
if opt_grid:
    with st.spinner("⚡ Menghitung matriks optimasi kombinasi SL vs TP..."):
        grid_tp = [float(x) for x in args.grid_tp.split(",") if x.strip()]
        grid_sl = [float(x) for x in args.grid_sl.split(",") if x.strip()]
        cap_meta = {
            "initial_capital": capital_input,
            "point_val_idr": cap_sim["point_val_idr"],
        }
        grid_sorted, df_matrix, best_info = run_grid_optimization(
            trades, bid, ask, grid_tp, grid_sl,
            sort_by=args.grid_by, top_n=args.grid_top, eod_mode=args.grid_eod, cap_sim_meta=cap_meta
        )

# -----------------------------------------------------------------------------
# TOP METRICS CARDS (BARIS 1 & 2)
# -----------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Net Profit & ROI</div>
        <div class="metric-val" style="color:{'#16A34A' if cap_sim['net_profit_idr'] >= 0 else '#DC2626'}">{fmt_idr(cap_sim['net_profit_idr'])}</div>
        <div class="metric-sub">ROI: <b>{cap_sim['roi_pct']:+.1f}%</b> ({fmt_idr(cap_sim['final_balance'], with_sign=False)})</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Win Rate Strategi</div>
        <div class="metric-val">{win_rate:.1f}%</div>
        <div class="metric-sub">{len(tp_trades)} Menang TP / {len(sl_trades)} Kalah SL</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Max Drawdown</div>
        <div class="metric-val" style="color:#D97706">{fmt_idr(cap_sim['max_dd_idr'], with_sign=False)}</div>
        <div class="metric-sub"><b>{cap_sim['max_dd_pct']:.1f}%</b> dari puncak saldo</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    streak_cur_lbl = f"+{last_streak} Win" if last_streak > 0 else (f"{last_streak} Loss" if last_streak < 0 else "0")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Rekor Beruntun (Streak)</div>
        <div class="metric-val" style="color:#2563EB"><span style="color:#16A34A">{max_win_streak}W</span> / <span style="color:#DC2626">{max_loss_streak}L</span></div>
        <div class="metric-sub">Longest Win: <b>{max_win_streak}x</b> | Longest Loss: <b>{max_loss_streak}x</b></div>
    </div>
    """, unsafe_allow_html=True)

c5, c6, c7 = st.columns(3)

with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Rekor No Floating</div>
        <div class="metric-val" style="color:#059669">{no_float_total} Trade</div>
        <div class="metric-sub"><b>{no_float_pct:.1f}%</b> langsung tembus TP tanpa minus</div>
    </div>
    """, unsafe_allow_html=True)

with c6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">SL Tembus TP Nanti (Penyelamatan)</div>
        <div class="metric-val" style="color:#B45309">{n_sl_saved} Trade</div>
        <div class="metric-sub"><b>{pct_sl_saved:.1f}%</b> SL berbalik tembus TP (Rata2 koreksi: {avg_saved_mae:.1f} pts)</div>
    </div>
    """, unsafe_allow_html=True)

with c7:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Rata2 Peak Profit (Daya Ledak)</div>
        <div class="metric-val" style="color:#7C3AED">+{avg_peak_all:.1f} pts</div>
        <div class="metric-sub">Puncak Runner Terjauh: <b>+{max_peak_all:.1f} pts</b></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB KONTEN UTAMA
# -----------------------------------------------------------------------------
tab_chart, tab_history, tab_grid, tab_export = st.tabs([
    "📈 Pertumbuhan Modal & Statistik",
    "📋 Detail Riwayat Trade & Analisis Stop-Out",
    "🧩 Grid Matrix Optimizer (SL vs TP)",
    "📥 Download Laporan PDF & CSV"
])


# =============================================================================
# TAB 1: GRAFIK & STATISTIK PERFORMA
# =============================================================================
with tab_chart:
    col_g1, col_g2 = st.columns([1.6, 1.0])

    with col_g1:
        st.markdown("##### 📈 Grafik Pertumbuhan Saldo Akun (Equity Curve)")
        fig_eq, ax_eq = plt.subplots(figsize=(8.5, 4.2), dpi=130)
        ax_eq.plot(cap_sim["balance_history"], color="#2563EB", linewidth=2.2, label="Saldo Akun (IDR)")
        ax_eq.axhline(capital_input, color="#94A3B8", linestyle="--", linewidth=1.2, label=f"Modal Awal ({fmt_idr(capital_input, with_sign=False)})")
        ax_eq.fill_between(range(len(cap_sim["balance_history"])), capital_input, cap_sim["balance_history"],
                           where=np.array(cap_sim["balance_history"]) >= capital_input, facecolor="#DCFCE7", alpha=0.5)
        ax_eq.fill_between(range(len(cap_sim["balance_history"])), capital_input, cap_sim["balance_history"],
                           where=np.array(cap_sim["balance_history"]) < capital_input, facecolor="#FEE2E2", alpha=0.5)
        ax_eq.set_title("Pertumbuhan Saldo Realistis (Akumulasi PnL)", fontsize=11, fontweight="bold", pad=10)
        ax_eq.set_xlabel("Jumlah Trade Berjalan", fontsize=9)
        ax_eq.set_ylabel("Saldo (IDR)", fontsize=9)
        ax_eq.grid(True, linestyle=":", alpha=0.6)
        ax_eq.legend(loc="upper left", fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_eq)
        plt.close(fig_eq)

    with col_g2:
        st.markdown("##### 🎯 Rekomendasi Parameter Sweet Spot")
        p80_mae = np.percentile(trades["mae_full"], 80) if len(trades) else 0.0
        p90_mae = np.percentile(trades["mae_full"], 90) if len(trades) else 0.0
        p80_mfe = np.percentile(trades["mfe"], 20) if len(trades) else 0.0

        st.markdown(f"""
        <div style="background:#F1F5F9; border-radius:8px; padding:14px; border:1px solid #CBD5E1; font-size:13px; line-height:1.6;">
            <b>🛡️ Rekomendasi Stop Loss (MAE Toleransi):</b><br>
            • P80 MAE: <b>{p80_mae:.1f} poin</b><br>
            • P90 MAE: <b>{p90_mae:.1f} poin</b><br>
            <i>👉 80-90% trade yang menang tidak pernah floating minus lebih dari batas ini. SL ideal: <b>{p80_mae:.1f} - {p90_mae:.1f} poin</b>.</i>
            <hr style="margin:10px 0;">
            <b>🚀 Rekomendasi Take Profit (MFE Breakout):</b><br>
            • Target Konservatif (80% trade tembus): <b>{p80_mfe:.1f} poin</b><br>
            • Rata-rata Lari Breakout (MFE): <b>+{avg_peak_all:.1f} poin</b><br>
            • Runner Maksimal: <b>+{max_peak_all:.1f} poin</b><br>
            <i>👉 Target TP aman: <b>{p80_mfe:.1f} poin</b>, target agresif: <b>+{avg_peak_all:.1f} poin</b>.</i>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#EFF6FF; border-radius:8px; padding:12px; border:1px solid #BFDBFE; font-size:12.5px;">
            <b>⚙️ Spesifikasi Nilai Transaksi & Rekor:</b><br>
            • Ukuran Lot: <b>{lot_input:.2f} Lot</b>  |  Nilai 1 Poin: <b>{fmt_idr(cap_sim['point_val_idr'], with_sign=False)}</b><br>
            • Target Profit per TP: <b style="color:#16A34A">+{fmt_idr(args.tp * cap_sim['point_val_idr'], with_sign=False)}</b><br>
            • Risiko Rugi per SL: <b style="color:#DC2626">-{fmt_idr((args.sl or 0) * cap_sim['point_val_idr'], with_sign=False)}</b> ({(args.sl or 0)*cap_sim['point_val_idr']/capital_input*100:.1f}% risiko modal)<br>
            • Rekor Menang (Win Streak): <b style="color:#16A34A">{max_win_streak} Trade Beruntun</b><br>
            • Rekor Kalah (Loss Streak): <b style="color:#DC2626">{max_loss_streak} Trade Beruntun</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##### 📊 Ringkasan Statistik Per Sesi & Arah")
    st.dataframe(summary_df, use_container_width=True)


# =============================================================================
# TAB 2: HISTORY DETAIL TRADE & ANALISIS STOP-OUT
# =============================================================================
with tab_history:
    st.markdown("##### 📋 Riwayat Detail Setiap Trade")

    # Format dataframe untuk tampilan interaktif
    h = trades.iloc[::-1].copy()
    h["Waktu Entry"] = h["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
    h["Arah"] = h["dir"]
    h["Entry"] = h["entry"].round(1)
    h["TP"] = h["tp"].round(1)
    h["Hasil"] = h["result"]
    h["Durasi Exit"] = h["dur_label"]

    def get_fpos(r):
        if r["result"] == "TP":
            return "No Floating" if r["mae"] <= 0.0001 else f"-{r['mae']:.1f} poin"
        elif r["result"] == "SL":
            return f"-{args.sl:.1f} (SL)" if args and args.sl else f"-{r['mae']:.1f} (SL)"
        else:
            return f"-{r['mae_full']:.1f} poin"

    def get_fbefore_tp(r):
        if r["result"] == "TP":
            return "No Floating" if r["mae"] <= 0.0001 else f"-{r['mae']:.1f} poin"
        elif r["reached_tp_eventually"]:
            return f"-{r['mae_before_tp']:.1f} (Tembus TP)"
        else:
            return f"Gagal ke TP (+{r['mfe']:.1f})"

    h["Floating Posisi"] = h.apply(get_fpos, axis=1)
    h["Max Float sblm TP"] = h.apply(get_fbefore_tp, axis=1)
    h["Peak Profit"] = "+" + h["mfe"].round(1).astype(str) + " pts"
    h["Spread"] = h["spread_at_entry"].round(1)

    # Filter pilihan
    f_filter = st.radio("Tampilkan:", ["Semua Trade", "Hanya Kena TP", "Hanya Kena SL", "Kena SL Tapi Tembus TP Nanti (Penyelamatan)"], horizontal=True)
    if f_filter == "Hanya Kena TP":
        h_show = h[h["Hasil"] == "TP"]
    elif f_filter == "Hanya Kena SL":
        h_show = h[h["Hasil"] == "SL"]
    elif f_filter == "Kena SL Tapi Tembus TP Nanti (Penyelamatan)":
        h_show = h[(h["Hasil"] == "SL") & (h["reached_tp_eventually"])]
    else:
        h_show = h

    cols_table = ["Waktu Entry", "Arah", "Entry", "TP", "Hasil", "Durasi Exit", "Floating Posisi", "Max Float sblm TP", "Peak Profit", "Spread"]
    st.dataframe(h_show[cols_table], use_container_width=True, height=360)

    # Box Kesimpulan Analisis Stop-Out Prematur
    st.markdown("---")
    st.markdown("#### 💡 Kesimpulan Analisis Floating Sebelum TP & Runner Potential")

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        avg_tp_mae = tp_trades["mae"].mean() if len(tp_trades) else 0.0
        avg_tp_mfe = tp_trades["mfe"].mean() if len(tp_trades) else 0.0
        st.markdown(f"""
        <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:14px; font-size:13px; line-height:1.6;">
            <b style="color:#166534; font-size:14px;">1. Analisis Trade Pemenang (Langsung Tembus TP):</b><br>
            • Total Trade Menang: <b>{len(tp_trades)} trade</b> ({win_rate:.1f}%)<br>
            • Rata-rata Floating (MAE) sebelum TP: <b>{avg_tp_mae:.2f} poin</b><br>
            • Trade 'No Floating' (0 Drawdown): <b>{no_float_total} trade</b> ({no_float_pct:.1f}% melesat tanpa minus)<br>
            • Rata-rata Peak Profit Pemenang: <b>+{avg_tp_mfe:.2f} poin</b> (Harga terus berlari jauh setelah TP {args.tp:g} pts)
        </div>
        """, unsafe_allow_html=True)

    with col_k2:
        if len(sl_trades) > 0:
            avg_sl_mfe_saved = sl_saved["mfe"].mean() if n_sl_saved else 0.0
            st.markdown(f"""
            <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:8px; padding:14px; font-size:13px; line-height:1.6;">
                <b style="color:#92400E; font-size:14px;">2. Analisis Trade Kena SL (Potensi Terselamatkan):</b><br>
                • Total Kena SL: <b>{len(sl_trades)} trade</b><br>
                • Kena SL Tapi Akhirnya Tembus TP: <b style="color:#B45309;">{n_sl_saved} trade ({pct_sl_saved:.1f}%)</b><br>
                • Rata-rata Koreksi sebelum Tembus TP: <b>{avg_saved_mae:.2f} poin</b> (P80: <b>{p80_saved_mae:.2f} poin</b>)<br>
                • Rata-rata Peak Profit SL Terselamatkan: <b>+{avg_sl_mfe_saved:.2f} poin</b><br>
                <i>👉 Sebanyak <b>{pct_sl_saved:.1f}%</b> trade yang terkena SL sebenarnya tembus TP jika SL dilonggarkan ke <b>{p80_saved_mae:.1f} poin</b>!</i>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("Tidak ada trade yang terkena Stop Loss pada pengujian ini.")


# =============================================================================
# TAB 3: GRID MATRIX OPTIMIZER
# =============================================================================
with tab_grid:
    if opt_grid and df_matrix is not None and grid_sorted is not None:
        col_m1, col_m2 = st.columns([1.2, 1.0])
        with col_m1:
            st.markdown("##### 🗺️ Heatmap Expectancy Matrix (SL vs TP)")
            fig_hm, ax_hm = plt.subplots(figsize=(6.5, 4.6), dpi=140)
            mat_vals = df_matrix.values
            v_max = max(abs(np.nanmin(mat_vals)), abs(np.nanmax(mat_vals)), 1.0)
            im = ax_hm.imshow(mat_vals, cmap="RdYlGn", aspect="auto", vmin=-v_max, vmax=v_max)
            ax_hm.set_xticks(range(len(df_matrix.columns)))
            ax_hm.set_xticklabels(df_matrix.columns, fontsize=8, rotation=35)
            ax_hm.set_yticks(range(len(df_matrix.index)))
            ax_hm.set_yticklabels(df_matrix.index, fontsize=8)
            ax_hm.set_title("Heatmap Expectancy (Poin / Trade) - Hijau = Paling Untung", fontsize=9.5, fontweight="bold", pad=8)
            ax_hm.set_xlabel("Target Take Profit (TP)", fontsize=8.5)
            ax_hm.set_ylabel("Stop Loss (SL)", fontsize=8.5)

            for i in range(len(df_matrix.index)):
                for j in range(len(df_matrix.columns)):
                    val = mat_vals[i, j]
                    if not np.isnan(val):
                        sign = "+" if val > 0 else ""
                        color = "white" if abs(val) > v_max * 0.65 else "black"
                        ax_hm.text(j, i, f"{sign}{val:.1f}", ha="center", va="center", color=color, fontsize=7.5, fontweight="bold")

            fig_hm.colorbar(im, ax=ax_hm, fraction=0.035, pad=0.03)
            plt.tight_layout()
            st.pyplot(fig_hm)
            plt.close(fig_hm)

        with col_m2:
            st.markdown(f"##### 🏆 Top {args.grid_top} Kombinasi SL vs TP Terbaik")
            st.dataframe(grid_sorted.head(args.grid_top), use_container_width=True, height=380)

    else:
        st.info("Aktifkan centang 'Jalankan Grid Matrix Optimizer (SL vs TP)' di menu samping untuk melihat peta kombinasi.")


# =============================================================================
# TAB 4: DOWNLOAD CENTER
# =============================================================================
with tab_export:
    st.markdown("##### 📥 Download Hasil Analisis & Laporan Lengkap")

    c_d1, c_d2 = st.columns(2)

    with c_d1:
        st.markdown("#### 📄 Laporan PDF Visual Profesional")
        st.caption("Laporan visual 2 halaman format A4 landscape lengkap dengan diagram, parameter input, performa modal, dan heatmap.")
        if opt_pdf:
            pdf_out_path = os.path.join(CURRENT_DIR, "laporan_ustec_terbaru.pdf")
            meta_pdf = {
                "date_range": f"{pd.Timestamp(t[0]).strftime('%Y-%m-%d')} s/d {pd.Timestamp(t[-1]).strftime('%Y-%m-%d')}",
                "total_ticks": len(t),
                "tzname": "UTC-5",
                "sess_col": "Sesi UTC-5"
            }
            with st.spinner("Menyiapkan dokumen PDF visual..."):
                generate_pdf_report(
                    pdf_out_path, meta_pdf, trades, summary_df,
                    grid_sorted=grid_sorted if opt_grid else None,
                    df_matrix=df_matrix if opt_grid else None,
                    best_info=best_info if opt_grid else None,
                    cap_sim=cap_sim,
                    args=args,
                    ranges=ranges,
                    sess_label=sess_label
                )

            if os.path.exists(pdf_out_path):
                with open(pdf_out_path, "rb") as f_pdf:
                    st.download_button(
                        label="📥 Download Laporan PDF (Visual)",
                        data=f_pdf.read(),
                        file_name="Laporan_USTEC_Breakout.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
        else:
            st.warning("Aktifkan centang 'Buat File Laporan PDF Siap Cetak' di menu samping.")

    with c_d2:
        st.markdown("#### 📊 Ekspor Riwayat Trade (CSV)")
        st.caption("File spreadsheet berisi seluruh data tick entry, exit, MAE, MFE, dan durasi setiap posisi.")
        csv_buffer = io.StringIO()
        export_cols = [c for c in trades.columns if c not in ("start_idx", "end_idx")]
        trades[export_cols].to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Riwayat Trade (CSV)",
            data=csv_buffer.getvalue(),
            file_name="riwayat_trade_ustec.csv",
            mime="text/csv"
        )
