#!/usr/bin/env python3
"""
USTEC range breakout - analisis floating (MAE) dan potensi profit (MFE) memakai data TICK bid/ask dari MT5.

Fitur:
  1. Analisis MFE (Maximum Favorable Excursion) & MAE: mengukur floating minus terdalam dan potensi TP maksimal.
  2. Durasi Hit TP & Status "No Floating": mencatat waktu tempuh ke TP (dalam detik) dan menandai trade yang langsung TP tanpa floating.
  3. Analisis Premature Stop-Out (Highest Float sblm TP): melacak floating minus terdalam sebelum menyentuh TP untuk trade yang kena SL dulu.
  4. Grid Matrix Optimizer (--grid-opt): menguji matriks kombinasi SL vs TP (Winrate, Expectancy, Profit Factor, Max Drawdown).
  5. Rekomendasi Sweet Spot Otomatis: rekomendasi batas SL & TP optimal berdasarkan distribusi MAE/MFE dan ekspektansi.
  6. Generate Laporan PDF (--pdf): laporan visual profesional yang rapi, mencerminkan seluruh parameter input syntax dan simulasi modal.
  7. Simulasi Pertumbuhan Modal (--capital): menghitung pertumbuhan saldo nyata dalam Rupiah (ROI, Rupiah Profit/Loss, Drawdown %).

Zona waktu sesi (opsi --tz):
  utc-5 (default) : jam tetap UTC-5 sepanjang tahun, tanpa penyesuaian DST.
  ny              : waktu New York biasa (UTC-5 musim dingin, UTC-4 musim panas).

Contoh Penggunaan:
  python ustec_range_mae.py ticks_USTEC.csv --only 4 --tp 15 --sl 8 --capital 8jt --lot 3 --pdf
  python ustec_range_mae.py ticks_USTEC.csv --only 4 --grid-opt --capital 8jt --lot 3 --sl 8 --tp 15 --last 50
"""
import argparse
from itertools import product
import sys

import numpy as np
import pandas as pd

# Sesi default (waktu New York), sama dengan input indikator Pine versi terakhir
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
def main():
    ap = argparse.ArgumentParser(description="Analisis floating (MAE) dan potensi profit (MFE) per range, data tick MT5.")
    ap.add_argument("files", nargs="+", help="File CSV tick hasil ekspor MT5 (boleh beberapa file)")
    ap.add_argument("--tp", type=float, default=8.0, help="Take profit dalam poin harga (default 8)")
    ap.add_argument("--offset", type=float, default=0.0, help="Offset entry dari High/Low dalam poin (default 0)")
    ap.add_argument("--server-offset", type=float, default=7.0,
                    help="Selisih jam server MT5 terhadap New York (default 7: server = NY + 7 jam)")
    ap.add_argument("--tz", choices=["utc-5", "ny"], default="utc-5",
                    help="Zona waktu sesi: utc-5 = jam tetap UTC-5 (default); ny = New York dengan DST")
    ap.add_argument("--ranges", default=None,
                    help='Override sesi, mis. "1=1256-2023,7=0010-0101". Default: Range 1-7 dari indikator.')
    ap.add_argument("--only", default=None, help="Hanya range tertentu, mis. 3,7")
    ap.add_argument("--pct", type=int, default=80, help="Persentil MAE & MFE (default 80)")
    ap.add_argument("--sl", type=float, default=None,
                    help="Stop Loss dalam poin. Trade dihitung SL jika floating mencapai nilai ini sebelum TP.")
    ap.add_argument("--sl-test", default="", help="Bandingkan beberapa nilai SL (jumlah & persen kena SL), mis. 10,15,20")
    ap.add_argument("--float-tp-only", action="store_true",
                    help="Statistik floating hanya dari trade yang kena TP (default: semua trade terpicu)")
    ap.add_argument("--fill", choices=["tick", "level"], default="tick",
                    help="tick: isi di harga tick pemicu (realistis); level: isi tepat di level pending")
    ap.add_argument("--range-price", choices=["bid", "mid"], default="bid",
                    help="Harga untuk High/Low range (default bid, seperti chart)")
    ap.add_argument("--from", dest="date_from", default=None, help="Tanggal mulai (YYYY-MM-DD, pada zona waktu --tz)")
    ap.add_argument("--to", dest="date_to", default=None, help="Tanggal akhir (YYYY-MM-DD, pada zona waktu --tz)")
    ap.add_argument("--chunk-rows", type=int, default=2_000_000,
                    help="Jumlah baris per potongan saat membaca file besar (kecilkan jika RAM terbatas)")
    ap.add_argument("--out", default=None, help="Simpan semua trade ke CSV")
    ap.add_argument("--history", type=int, default=None, help="Tampilkan history trade untuk range ini (1-7)")
    ap.add_argument("--last", type=int, default=15, help="Jumlah trade history (default 15, gunakan 0 untuk semua)")

    # Argumen Tambahan: Grid Optimizer & Sweet Spot
    ap.add_argument("--grid-opt", action="store_true",
                    help="Jalankan Grid Matrix Optimizer untuk mencari kombinasi SL & TP terbaik.")
    ap.add_argument("--grid-tp", default="5,8,10,12,15,20,25,30,40,50",
                    help="Daftar TP yang diuji di grid, pisahkan koma (default: 5,8,10,12,15,20,25,30,40,50)")
    ap.add_argument("--grid-sl", default="5,8,10,15,20,25,30,40,50",
                    help="Daftar SL yang diuji di grid, pisahkan koma (default: 5,8,10,15,20,25,30,40,50)")
    ap.add_argument("--grid-by", choices=["expectancy", "pnl", "pf", "winrate"], default="expectancy",
                    help="Kriteria pengurutan tabel grid (default: expectancy)")
    ap.add_argument("--grid-top", type=int, default=15,
                    help="Jumlah peringkat teratas yang ditampilkan di tabel grid (default: 15)")
    ap.add_argument("--grid-eod", choices=["close", "ignore"], default="close",
                    help="Trade yang belum kena SL/TP saat jendela berakhir: close (tutup di market) / ignore (default close)")
    ap.add_argument("--grid-per-range", action="store_true",
                    help="Tampilkan optimasi grid terpisah untuk masing-masing range.")

    # Argumen Tambahan: Export PDF Report
    ap.add_argument("--pdf", nargs="?", const="ustec_analysis_report.pdf", default=None,
                    help="Generate laporan PDF lengkap (contoh: --pdf laporan.pdf atau cukup --pdf)")

    # Argumen Tambahan: Simulasi Modal & Lot (Rupiah)
    ap.add_argument("--capital", default=None,
                    help="Simulasi modal awal, mis. 8000000 atau 8jt (dalam Rupiah)")
    ap.add_argument("--lot", type=float, default=1.0,
                    help="Ukuran lot transaksi untuk simulasi modal (default: 1.0)")
    ap.add_argument("--contract-size", type=float, default=1.0,
                    help="Nilai 1 poin per 1 lot dalam USD (default: 1.0 untuk Exness/broker CFD USTEC)")
    ap.add_argument("--usd-rate", type=float, default=16000.0,
                    help="Kurs konversi 1 USD ke IDR (default: 16000)")

    args = ap.parse_args()

    ranges = dict(DEFAULT_RANGES)
    if args.ranges:
        if "=" not in args.ranges:
            print("Catatan: --ranges dipakai untuk mengganti jam sesi (contoh: 1=1256-2023). "
                  "Untuk memilih range gunakan --only. Dianggap --only " + args.ranges + "\n")
            args.only = args.only or args.ranges
        else:
            for part in args.ranges.split(","):
                if "=" not in part:
                    sys.exit(f"Format --ranges salah di '{part}'. Contoh yang benar: --ranges 1=1256-2023,7=0010-0101")
                k, v = part.split("=")
                ranges[int(k)] = v.strip()
    if args.only:
        keep = {int(x) for x in args.only.split(",")}
        ranges = {k: v for k, v in ranges.items() if k in keep}
    sl_list = [float(x) for x in args.sl_test.split(",") if x.strip()]

    initial_capital = parse_capital(args.capital)

    print("Membaca tick ...", flush=True)
    t, bid, ask = load_ticks(args.files, args.server_offset, args.tz, args.chunk_rows)

    if args.date_from:
        i = np.searchsorted(t, np.datetime64(args.date_from), side="left")
        t, bid, ask = t[i:], bid[i:], ask[i:]
    if args.date_to:
        i = np.searchsorted(t, np.datetime64(args.date_to) + np.timedelta64(1, "D"), side="left")
        t, bid, ask = t[:i], bid[:i], ask[:i]
    if len(t) == 0:
        sys.exit("Tidak ada tick pada rentang yang dipilih.")

    tzname = "UTC-5" if args.tz == "utc-5" else "New York (DST)"
    sess_col = "Sesi " + ("UTC-5" if args.tz == "utc-5" else "NY")

    def fmt(m):
        return f"{(m // 60) % 24:02d}:{m % 60:02d}"

    sess_label = {}
    for rid, sess in ranges.items():
        a, b = sess.split("-")
        sm, em = hhmm_to_min(a), hhmm_to_min(b)
        sess_label[rid] = f"{fmt(sm)}-{fmt(em)}"

    print(f"{len(t):,} tick  |  {pd.Timestamp(t[0])}  ->  {pd.Timestamp(t[-1])}  (waktu {tzname})")
    print(f"Target TP: {args.tp:g} poin | Offset: {args.offset:g} | Fill: {args.fill} | " +
          (f"Target SL: {args.sl:g} poin" if args.sl else "Tanpa SL"))
    if initial_capital:
        print(f"Simulasi Modal: {fmt_idr(initial_capital, with_sign=False)} | Lot: {args.lot:.2f} | Kurs: Rp {args.usd_rate:,.0f}/USD")
    print()

    print(f"=== JAM RANGE (zona waktu {tzname}) ===")
    for rid in sorted(ranges):
        a, b = ranges[rid].split("-")
        sm, em = hhmm_to_min(a), hhmm_to_min(b)
        line = f"R{rid}: {fmt(sm)} - {fmt(em)}"
        if em <= sm:
            line += " (+1 hari)"
        if args.tz == "utc-5":
            wsm, wem = (sm + 720) % 1440, (em + 720) % 1440
            line += f"   | setara WIB (UTC+7): {fmt(wsm)} - {fmt(wem)}"
        print(line)
    print()

    rows = []
    for rid, sess in sorted(ranges.items()):
        rows += simulate_range(rid, sess, t, bid, ask, args)
    if not rows:
        sys.exit("Tidak ada trade yang terpicu. Periksa --server-offset dan rentang data.")

    trades = pd.DataFrame(rows).sort_values(["range", "entry_time"]).reset_index(drop=True)

    cap_sim = simulate_capital(trades, initial_capital, args.lot, args.contract_size, args.usd_rate)

    basis = "trade yang kena TP saja" if args.float_tp_only else "semua trade yang terpicu"
    print(f"=== RINGKASAN per Range & Arah (MAE dari {basis}, MFE dari seluruh jendela trade) ===")
    summary_df = summarize(trades, args.pct, sl_list, sess_label, sess_col, bool(args.sl), args.float_tp_only)
    print(summary_df.to_string(index=False))
    print("\nKeterangan Kolom:")
    print("  Trade       = Total trade yang terpicu (Buy Stop / Sell Stop tersentuh).")
    print("  TotPnL      = Total akumulasi keuntungan/kerugian bersih (dalam poin harga).")
    print("  Exp/Trd     = Expectancy (rata-rata keuntungan poin per trade). Wajib POSITIF agar profit jangka panjang.")
    print("  NoFloat#    = Jumlah trade yang langsung mencapai TP TANPA FLOATING MINUS sama sekali.")
    print("  FastTP(s)   = Waktu tercepat mencapai TP sejak posisi terpicu (dalam detik).")
    print("  MedDur(s)   = Nilai tengah (median) durasi mencapai TP (dalam detik).")
    print("  MAE         = Maximum Adverse Excursion (floating minus terdalam sebelum TP atau akhir jendela).")
    print("  MFE         = Maximum Favorable Excursion (keuntungan puncak tertinggi yang sempat tercapai).")
    print(f"  P{args.pct}_MAE     = {args.pct}% trade memiliki floating minus TIDAK LEBIH BESAR dari nilai ini (Batas SL Optimal).")
    print(f"  P{args.pct}_MFE     = {args.pct}% trade MAMPU MENCAPAI keuntungan minimal sebesar nilai ini (Batas TP Konservatif).")
    print()

    if cap_sim:
        print_capital_report(cap_sim, trades, args.tp, args.sl)
    else:
        mw, ml, _ = compute_streaks(trades)
        print(f"Rekor Beruntun (Streak) : Longest Win Streak = {mw} trade | Longest Loss Streak = {ml} trade\n")

    grid_sorted, df_matrix, best_info = None, None, None
    if args.grid_opt:
        grid_tp = [float(x) for x in args.grid_tp.split(",") if x.strip()]
        grid_sl = [float(x) for x in args.grid_sl.split(",") if x.strip()]

        cap_meta = None
        if cap_sim:
            cap_meta = {
                "initial_capital": initial_capital,
                "point_val_idr": cap_sim["point_val_idr"],
            }

        print(f"Menjalankan Grid Matrix Optimizer ({len(grid_tp)} TP x {len(grid_sl)} SL = {len(grid_tp)*len(grid_sl)} kombinasi) ...")
        grid_sorted, df_matrix, best_info = run_grid_optimization(
            trades, bid, ask, grid_tp, grid_sl,
            sort_by=args.grid_by, top_n=args.grid_top, eod_mode=args.grid_eod, cap_sim_meta=cap_meta
        )

        print(f"\n=== TOP {args.grid_top} KOMBINASI SL vs TP TERBAIK (Diurutkan berdasarkan {args.grid_by.upper()}) ===")
        print(grid_sorted.head(args.grid_top).to_string())
        print("\nKeterangan Metrik Grid:")
        print("  Exp/Trade    : Rata-rata poin keuntungan bersih per trade (Expectancy). Wajib bernilai POSITIF.")
        print("  ProfitFactor : Total Profit / Total Loss (di atas 1.3 bagus, di atas 1.5 sangat kuat).")
        print("  MaxDD        : Penurunan akumulatif terbesar dari puncak (Max Drawdown dalam poin).")
        if cap_meta:
            print(f"  Profit(IDR)  : Estimasi total keuntungan rupiah bersih dengan modal {fmt_idr(initial_capital, with_sign=False)} dan {args.lot:.2f} lot.")

        print("\n=== HEATMAP MATRIX: EXPECTANCY (Poin / Trade) ===")
        print("Baris = Stop Loss (SL), Kolom = Take Profit (TP):")
        print(df_matrix.to_string())
        print()

        if args.grid_per_range:
            print("=== KOMBINASI TERBAIK PER RANGE ===")
            for rid in sorted(trades["range"].unique()):
                sub = trades[trades["range"] == rid].reset_index(drop=True)
                s_res, _, _ = run_grid_optimization(sub, bid, ask, grid_tp, grid_sl, sort_by=args.grid_by, top_n=3)
                if s_res is not None and len(s_res) > 0:
                    top_one = s_res.iloc[0]
                    print(f"  Range {rid} ({sess_label.get(rid, '')}): Best TP={top_one['TP']}, SL={top_one['SL']} "
                          f"| Exp: +{top_one['Exp/Trade']} pts/trd | WinRate: {top_one['WinRate%']}% | PF: {top_one['ProfitFactor']}")
            print()

    print_recommendations(trades, grid_sorted, args.pct)

    # Tampilkan History Trade (Otomatis tampil jika --only 1 range atau --history ditentukan)
    history_target = args.history
    if history_target is None and args.only and "," not in str(args.only):
        try:
            history_target = int(args.only)
        except ValueError:
            history_target = None

    if history_target:
        sub_h = trades[trades["range"] == history_target]
        h = sub_h.iloc[::-1].copy() if args.last <= 0 else sub_h.tail(args.last).iloc[::-1].copy()
        if len(h) > 0:
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

            cols_show = ["Waktu Entry", "Arah", "Entry", "TP", "Hasil", "Durasi Exit", "Floating Posisi", "Max Float sblm TP", "Peak Profit", "Spread"]
            label_count = f"semua {len(h)} trade" if (args.last <= 0 or len(h) < args.last) else f"{len(h)} trade terakhir"
            print(f"=== HISTORY DETAIL TRADE Range {history_target} {sess_label.get(history_target, '')} {tzname} ({label_count}) ===")
            print(h[cols_show].to_string(index=False))
            print()

            # ----------------------------------------------------------------- #
            # KESIMPULAN ANALISIS FLOATING SEBELUM TP (PREMATURE STOP-OUT)
            # ----------------------------------------------------------------- #
            tp_sub = h[h["result"] == "TP"]
            sl_sub = h[h["result"] == "SL"]

            avg_tp_mae = tp_sub["mae"].mean() if len(tp_sub) else 0.0
            med_tp_mae = tp_sub["mae"].median() if len(tp_sub) else 0.0
            no_float_tp = (tp_sub["mae"] <= 0.0001).sum() if len(tp_sub) else 0

            n_sl = len(sl_sub)
            sl_saved = sl_sub[sl_sub["reached_tp_eventually"]]
            n_sl_saved = len(sl_saved)
            pct_saved = (n_sl_saved / n_sl * 100.0) if n_sl > 0 else 0.0

            avg_sl_mae_to_tp = sl_saved["mae_before_tp"].mean() if n_sl_saved else 0.0
            med_sl_mae_to_tp = sl_saved["mae_before_tp"].median() if n_sl_saved else 0.0
            max_sl_mae_to_tp = sl_saved["mae_before_tp"].max() if n_sl_saved else 0.0
            p80_sl_mae_to_tp = np.percentile(sl_saved["mae_before_tp"], 80) if n_sl_saved else 0.0

            # Statistik Peak Profit (MFE)
            avg_peak_all = h["mfe"].mean() if len(h) else 0.0
            med_peak_all = h["mfe"].median() if len(h) else 0.0
            max_peak_all = h["mfe"].max() if len(h) else 0.0
            p80_peak_all = np.percentile(h["mfe"], 80) if len(h) else 0.0

            avg_peak_tp = tp_sub["mfe"].mean() if len(tp_sub) else 0.0
            avg_peak_sl_saved = sl_saved["mfe"].mean() if n_sl_saved else 0.0

            print("=" * 96)
            print("     KESIMPULAN ANALISIS FLOATING SEBELUM TP & PEAK PROFIT (PREMATURE STOP-OUT)")
            print("=" * 96)
            h_win_s, h_loss_s, _ = compute_streaks(h)
            print(f"Dari {len(h)} trade pada riwayat di atas ({len(tp_sub)} Menang TP, {n_sl} Kena SL | Longest Win Streak: {h_win_s}x, Loss Streak: {h_loss_s}x):\n")
            print("1. ANALISIS TRADE PEMENANG (KENA TP LANGSUNG):")
            print(f"   • Rata-rata Floating (MAE) sebelum TP : {avg_tp_mae:.2f} poin  (Median: {med_tp_mae:.2f} poin)")
            print(f"   • Trade 'No Floating' (0 Drawdown)    : {no_float_tp} trade ({(no_float_tp/max(1, len(tp_sub))*100):.1f}% langsung tembus tanpa minus)")
            print(f"   • Rata-rata Peak Profit Trade TP      : +{avg_peak_tp:.2f} poin (Potensi lanjutan setelah hit TP {args.tp:g} pts)\n")

            if n_sl > 0:
                print("2. ANALISIS TRADE YANG KENA SL (POTENSI TERSALAMATKAN):")
                print(f"   • Total Trade Kena SL                 : {n_sl} trade")
                print(f"   • Kena SL Tapi Akhirnya Tembus TP     : {n_sl_saved} trade ({pct_saved:.1f}% dari trade yang SL!)")
                if n_sl_saved > 0:
                    print(f"   • Rata-rata Max Float sebelum Tembus TP: {avg_sl_mae_to_tp:.2f} poin  (Median: {med_sl_mae_to_tp:.2f} poin)")
                    print(f"   • Persentil 80% (P80 Float sblm TP)   : {p80_sl_mae_to_tp:.2f} poin")
                    print(f"   • Floating Terjauh sebelum Tembus TP  : {max_sl_mae_to_tp:.2f} poin")
                    print(f"   • Rata-rata Peak Profit SL Terselamatkan: +{avg_peak_sl_saved:.2f} poin")
                print(f"   • Trade Murni Gagal (Tidak Tembus TP) : {n_sl - n_sl_saved} trade\n")

            print("3. ANALISIS DAYA LEDAK PEAK PROFIT (RUNNER POTENTIAL):")
            print(f"   • Rata-rata Peak Profit Keseluruhan   : +{avg_peak_all:.2f} poin  (Median: +{med_peak_all:.2f} poin)")
            print(f"   • Persentil 80% (P80 Peak Profit)     : +{p80_peak_all:.2f} poin")
            print(f"   • Peak Profit Tertinggi (Max Runner)  : +{max_peak_all:.2f} poin")
            if cap_sim:
                print(f"   • Nilai Rupiah Rata-rata Peak Profit  : +{fmt_idr(avg_peak_all * cap_sim['point_val_idr'], with_sign=False)} per trade ({cap_sim['lot']:.2f} lot)")

            if n_sl > 0 and n_sl_saved > 0 and args.sl:
                print(f"\n   💡 KESIMPULAN REKOMENDASI:")
                print(f"      Sebanyak {n_sl_saved} trade yang kena SL ({pct_saved:.1f}%) SEBENARNYA AKHIRNYA MENYENTUH TARGET TP {args.tp:g} POIN!")
                print(f"      Harga hanya mengalami tarikan koreksi rata-rata {avg_sl_mae_to_tp:.2f} poin (terjauh {max_sl_mae_to_tp:.2f} poin) sebelum melesat ke TP.")
                print(f"      Bahkan setelah koreksi tersebut, rata-rata harga melesat hingga +{avg_peak_sl_saved:.2f} poin!")
                print(f"      -> Jika SL Anda dilonggarkan dari {args.sl:g} poin ke kisaran {p80_sl_mae_to_tp:.1f} - {max_sl_mae_to_tp:.1f} poin,")
                print(f"         maka {n_sl_saved} trade yang tadinya rugi ini AKAN BERUBAH MENJADI PROFIT!")
            print("=" * 96 + "\n")

    if args.out:
        export_cols = [c for c in trades.columns if c not in ("start_idx", "end_idx")]
        trades[export_cols].to_csv(args.out, index=False)
        print(f"Semua trade ({len(trades):,} baris) disimpan ke {args.out}")

    if args.pdf:
        meta = {
            "date_range": f"{pd.Timestamp(t[0]).strftime('%Y-%m-%d')} s/d {pd.Timestamp(t[-1]).strftime('%Y-%m-%d')}",
            "total_ticks": len(t),
            "tzname": tzname,
            "sess_col": sess_col,
            "ranges_info": ", ".join(f"R{r}" for r in sorted(ranges)),
        }
        generate_pdf_report(args.pdf, meta, trades, summary_df, grid_sorted, df_matrix, best_info, cap_sim, args, ranges, sess_label)


if __name__ == "__main__":
    main()
