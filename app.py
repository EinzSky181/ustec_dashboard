"""
USTEC Breakout Strategy Hub - Web Dashboard (Streamlit)
Platform analisis visual intraday range breakout USTEC / US Tech 100 CFD.
"""

import os
import sys
import io
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tambahkan direktori kerja ke sys.path agar modul ustec_range_mae dapat diimpor
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from ustec_range_mae import (
    load_ticks,
    simulate_range,
    summarize,
    simulate_capital,
    compute_streaks,
    run_grid_optimization,
    generate_pdf_report,
    DEFAULT_RANGES,
    SESSION_NAMES_EN,
    fmt_idr,
    fmt_duration,
    hhmm_to_min,
    fmt_time_min,
    parse_capital,
)

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
