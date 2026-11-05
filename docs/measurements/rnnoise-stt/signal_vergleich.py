"""Signalvergleich Dot (Echo Küche) vs. Handy/Laptop-Mikro – 40 Satzpaare (★-Liste).
Misst pro Datei: Rauschabstand (50-ms-Fenster), Pegel, Frequenzumfang, Nachhall (grob),
Spitzen/Kompression, Vorlauf/Nachlauf. Ausgabe: CSV + Markdown-Tabelle + Spektrogramm-PNGs."""
import csv, sys, numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import stft, welch, butter, sosfiltfilt

BASE = Path("./aufnahmen")   # folder with the WAV pairs (sternchen/, _sternchen_handy_*/ + CSVs)
OUT = Path("./out"); OUT.mkdir(exist_ok=True, parents=True)
SR = 16000
WIN = int(0.050 * SR)  # 50 ms

def lade(p):
    sr, x = wavfile.read(p)
    assert sr == SR, (p, sr)
    x = x.astype(np.float32) / 32768.0
    if x.ndim > 1: x = x[:, 0]
    return x

def nullvorlauf(x):
    """Dot-Dateien beginnen mit exakt 0,8 s digitalen Nullen – die sind kein Mikrofonsignal und werden vor der Messung entfernt."""
    nz = np.where(x != 0)[0]
    return (nz[0] / SR, x[nz[0]:]) if len(nz) else (0.0, x)

def db(v): return 20 * np.log10(np.maximum(v, 1e-9))

def frames_rms(x):
    n = len(x) // WIN
    f = x[:n * WIN].reshape(n, WIN)
    return db(np.sqrt(np.mean(f ** 2, axis=1) + 1e-12))

def sprach_maske(lvl, boden, schwelle_db=8.0):
    """Frames > Boden + schwelle; Löcher < 150 ms schließen, Inseln < 100 ms löschen."""
    m = lvl > boden + schwelle_db
    # Löcher schließen (bis 3 Frames)
    idx = np.where(m)[0]
    if len(idx) == 0: return m
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= 3: m[a:b] = True
    # kurze Inseln löschen
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]: j += 1
            if j - i < 2: m[i:j] = False
            i = j
        else: i += 1
    return m

def analyse(x):
    r = {}
    r["nullen_vorn_s"], x = nullvorlauf(x)
    lvl = frames_rms(x)
    srt = np.sort(lvl)
    boden = float(np.median(srt[: max(3, len(srt) // 5)]))  # leisestes Fünftel
    sprache = float(np.percentile(lvl, 90))
    r["boden_dbfs"] = boden
    r["sprache_dbfs"] = sprache
    r["snr_db"] = sprache - boden
    m = sprach_maske(lvl, boden)
    r["sprachframes"] = int(m.sum())
    if m.sum() == 0:
        m = lvl > np.percentile(lvl, 60)
    idx = np.where(m)[0]
    a, b = idx[0], idx[-1] + 1
    r["vorlauf_s"] = a * WIN / SR
    r["nachlauf_s"] = (len(lvl) - b) * WIN / SR
    r["dauer_s"] = len(x) / SR
    r["sprachdauer_s"] = (b - a) * WIN / SR
    # Spitzen / Kompression
    xs = x[a * WIN: b * WIN]
    peak = float(np.max(np.abs(x)))
    r["peak_dbfs"] = db(peak)
    r["clip_anteil_pct"] = 100 * float(np.mean(np.abs(x) > 0.985))
    rms_s = float(np.sqrt(np.mean(xs ** 2) + 1e-12))
    r["crest_db"] = db(peak) - db(rms_s)
    lv_s = lvl[a:b][m[a:b]]
    r["dynamik_db"] = float(np.percentile(lv_s, 95) - np.percentile(lv_s, 10)) if len(lv_s) > 2 else np.nan
    # Frequenzumfang: Welch-PSD Sprache vs. Boden
    ruhe = np.concatenate([x[: a * WIN], x[b * WIN:]]) if (a > 2 or len(lvl) - b > 2) else x[: max(WIN * 2, int(0.2 * SR))]
    f, Ps = welch(xs, SR, nperseg=512)
    f, Pn = welch(ruhe, SR, nperseg=512) if len(ruhe) > 512 else (f, np.full_like(Ps, 1e-12))
    Ps_db, Pn_db = 10 * np.log10(Ps + 1e-14), 10 * np.log10(Pn + 1e-14)
    diff = Ps_db - Pn_db
    # oberste Frequenz, bei der Sprache noch >= 6 dB über Boden liegt (ab 300 Hz, Lücken erlaubt: letzte 250-Hz-Band-Mittelung)
    bands = np.arange(0, 8001, 250)
    ob = 0
    for lo, hi in zip(bands[:-1], bands[1:]):
        sel = (f >= lo) & (f < hi)
        if sel.any() and np.mean(diff[sel]) >= 6: ob = hi
    r["sprache_bis_hz"] = ob
    # Rolloff 95 % der Sprachenergie
    cs = np.cumsum(Ps) / np.sum(Ps)
    r["rolloff95_hz"] = float(f[np.searchsorted(cs, 0.95)])
    def band(lo, hi, P): sel = (f >= lo) & (f < hi); return 10 * np.log10(np.sum(P[sel]) + 1e-14)
    r["hoehen_4_8k_minus_mitten_0.3_4k_db"] = band(4000, 8000, Ps) - band(300, 4000, Ps)
    r["band_6_8k_minus_boden_db"] = float(np.mean(diff[(f >= 6000) & (f < 8000)]))
    r["band_4_6k_minus_boden_db"] = float(np.mean(diff[(f >= 4000) & (f < 6000)]))
    r["boden_tief_0_300_dbhz"] = band(0, 300, Pn)
    r["boden_hoch_4_8k_dbhz"] = band(4000, 8000, Pn)
    # Nachhall grob 1: Abklingzeit nach Sprachende (letzter Sprachframe) bis Boden+3 dB, aus 10-ms-Hüllkurve
    env_win = int(0.010 * SR)
    n = len(x) // env_win
    env = db(np.sqrt(np.mean(x[: n * env_win].reshape(n, env_win) ** 2, axis=1) + 1e-12))
    ende = b * WIN // env_win
    start_lvl = np.max(env[max(0, ende - 10): ende]) if ende > 0 else env[0]
    t_ab = np.nan
    for k in range(ende, len(env)):
        if env[k] <= boden + 3: t_ab = (k - ende) * 0.010; break
    r["abkling_ende_s"] = t_ab
    # Nachhall grob 2: Modulationstiefe – Std der 10-ms-Pegel innerhalb der Sprache (Hall & Kompression glätten)
    r["modulation_std_db"] = float(np.std(env[a * WIN // env_win: b * WIN // env_win]))
    # Nachhall grob 3: Anteil der Pausen innerhalb der Sprache, die unter Boden+6 dB fallen (Hall füllt Pausen)
    inner = env[a * WIN // env_win: b * WIN // env_win]
    r["pausen_tief_pct"] = 100 * float(np.mean(inner < boden + 6))
    r["dc_offset"] = float(np.mean(x))
    return r, (f, Ps_db, Pn_db)

def lies_csv(p):
    with open(p, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh, delimiter=";"))

paare = []
st = lies_csv(BASE / "_sternchen.csv")
for sp, ordner in (("A", "_sternchen_handy_A"), ("C", "_sternchen_handy_C")):
    hd = {r["wahrheit"]: r["datei"] for r in lies_csv(BASE / f"{ordner}.csv")}
    for r in st:
        if r["sprecher"] != sp: continue
        paare.append((sp, int(r["listen_nr"]), r["wahrheit"], BASE / "sternchen" / r["datei"], BASE / ordner / hd[r["wahrheit"]]))

felder = ["snr_db", "boden_dbfs", "sprache_dbfs", "peak_dbfs", "crest_db", "dynamik_db", "clip_anteil_pct",
          "sprache_bis_hz", "rolloff95_hz", "hoehen_4_8k_minus_mitten_0.3_4k_db", "band_4_6k_minus_boden_db", "band_6_8k_minus_boden_db",
          "abkling_ende_s", "modulation_std_db", "pausen_tief_pct", "vorlauf_s", "nachlauf_s", "sprachdauer_s", "dauer_s", "dc_offset",
          "sprachframes", "boden_tief_0_300_dbhz", "boden_hoch_4_8k_dbhz", "nullen_vorn_s"]
rows = []
spektren = {}
for sp, nr, text, pd_, ph in paare:
    for quelle, p in (("dot", pd_), ("handy", ph)):
        x = lade(p)
        r, spk = analyse(x)
        r.update(sprecher=sp, nr=nr, quelle=quelle, datei=p.name, text=text)
        rows.append(r)
        spektren[(sp, nr, quelle)] = spk
with open(OUT / "signal_vergleich.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["sprecher", "nr", "quelle", "datei", "text"] + felder, delimiter=";")
    w.writeheader()
    for r in rows: w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})

# Zusammenfassung
import statistics as S
def med(q, sp, k):
    v = [r[k] for r in rows if r["quelle"] == q and (sp is None or r["sprecher"] == sp) and not (isinstance(r[k], float) and np.isnan(r[k]))]
    return S.median(v) if v else np.nan
print("Median je Quelle (alle 40 / Anna 20 / Christoph 20):")
print(f"{'Kennzahl':42s} {'Dot':>8s} {'Handy':>8s} | {'Dot A':>7s} {'Handy A':>7s} | {'Dot C':>7s} {'Handy C':>7s}")
for k in felder:
    print(f"{k:42s} {med('dot',None,k):8.2f} {med('handy',None,k):8.2f} | {med('dot','A',k):7.2f} {med('handy','A',k):7.2f} | {med('dot','C',k):7.2f} {med('handy','C',k):7.2f}")

# Paartabelle Markdown
with open(OUT / "paare.md", "w", encoding="utf-8") as fh:
    fh.write("| Spr. | Nr | Satz | SNR Dot/Handy (dB) | Boden Dot/Handy (dBFS) | Sprache bis (Hz) Dot/Handy | 6–8 kHz über Boden Dot/Handy (dB) | Crest Dot/Handy (dB) | Dynamik Dot/Handy (dB) | Abkling. Dot/Handy (s) | Vorlauf Dot/Handy (s) | Nachlauf Dot/Handy (s) |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for sp, nr, text, _, _ in paare:
        d = next(r for r in rows if r["sprecher"] == sp and r["nr"] == nr and r["quelle"] == "dot")
        h = next(r for r in rows if r["sprecher"] == sp and r["nr"] == nr and r["quelle"] == "handy")
        def pair(k, f="{:.1f}"): return f"{f.format(d[k])} / {f.format(h[k])}"
        fh.write(f"| {sp[0].upper()} | {nr} | {text.replace('Alexa, ','')} | {pair('snr_db')} | {pair('boden_dbfs','{:.0f}')} | {pair('sprache_bis_hz','{:.0f}')} | {pair('band_6_8k_minus_boden_db')} | {pair('crest_db')} | {pair('dynamik_db')} | {pair('abkling_ende_s','{:.2f}')} | {pair('vorlauf_s','{:.2f}')} | {pair('nachlauf_s','{:.2f}')} |\n")

# Mittleres Sprachspektrum Dot vs Handy (Differenz = Übertragung Dot-Kette inkl. Raum), je Sprecher
np.save(OUT / "spektren.npy", {k: v for k, v in spektren.items()}, allow_pickle=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, sp in zip(axs, ("A", "C")):
    f = spektren[(sp, 1, "dot")][0]
    Sd = np.mean([spektren[(sp, n, "dot")][1] for n in range(1, 21)], axis=0)
    Nd = np.mean([spektren[(sp, n, "dot")][2] for n in range(1, 21)], axis=0)
    Sh = np.mean([spektren[(sp, n, "handy")][1] for n in range(1, 21)], axis=0)
    Nh = np.mean([spektren[(sp, n, "handy")][2] for n in range(1, 21)], axis=0)
    ax.plot(f, Sd, label="Dot – Sprache", color="#c0392b"); ax.plot(f, Nd, "--", label="Dot – Boden", color="#c0392b", alpha=.6)
    ax.plot(f, Sh, label="Handy – Sprache", color="#2471a3"); ax.plot(f, Nh, "--", label="Handy – Boden", color="#2471a3", alpha=.6)
    ax.set_title(f"Mittleres Spektrum, {sp.capitalize()} (20 Sätze)"); ax.set_xlabel("Hz"); ax.grid(alpha=.3); ax.set_xlim(0, 8000)
    ax.set_ylim(-120, -40)
axs[0].set_ylabel("dB/Hz"); axs[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "spektrum_mittel.png", dpi=130)

# Spektrogramme 3 typische Paare
def spektrogramm(ax, x, titel):
    f, t, Z = stft(x, SR, nperseg=512, noverlap=384)
    ax.pcolormesh(t, f, 20 * np.log10(np.abs(Z) + 1e-6), vmin=-100, vmax=-20, cmap="magma", shading="auto")
    ax.set_title(titel, fontsize=9); ax.set_ylim(0, 8000)
auswahl = [("A", 16), ("C", 16), ("A", 3)]  # Hauptstadt von Portugal (A, C), Rollladen im Bad (A)
if len(sys.argv) > 1: auswahl = [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[1:]]
for sp, nr in auswahl:
    sp_, nr_, text, pd_, ph = next(p for p in paare if p[0] == sp and p[1] == nr)
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.5), sharex=False)
    for ax, (q, p) in zip(axs, (("Dot (Echo Küche)", pd_), ("Handy/Laptop-Mikro", ph))):
        x = lade(p); r = next(rr for rr in rows if rr["datei"] == p.name and rr["sprecher"] == sp and rr["nr"] == nr)
        x = nullvorlauf(x)[1]
        spektrogramm(ax, x, f"{q} – {text} – SNR {r['snr_db']:.0f} dB, Boden {r['boden_dbfs']:.0f} dBFS, Sprache bis {r['sprache_bis_hz']:.0f} Hz")
        ax.set_ylabel("Hz")
    axs[1].set_xlabel("s")
    plt.tight_layout(); plt.savefig(OUT / f"spektrogramm_{sp}_{nr:02d}.png", dpi=130); plt.close(fig)
print("fertig:", sorted(p.name for p in OUT.iterdir()))
