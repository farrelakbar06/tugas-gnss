#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TUGAS INDIVIDU GNSS
Perhitungan koordinat receiver (Single Point Positioning) dari Navigation Message GPS
- Data   : 42621400.23n (nav GPS), 42621400.23o (observasi), RINEX 2.11
- Epoch  : 2 epoch saja (default 02:00:00 dan 02:00:15 GPS Time)
- Metode : posisi satelit dari broadcast ephemeris -> pseudorange C1 -> least squares

Cara pakai:
    pip install numpy matplotlib
    python gnss_spp.py
(taruh file .23n dan .23o di folder yang sama dengan script; hasil muncul di folder "hasil")
"""
import os
import math
import datetime as dt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ PENGATURAN
HERE = os.path.dirname(os.path.abspath(__file__))
# Otomatis: baca file data dari folder yang sama dengan script ini
DATA_DIR = HERE if os.path.exists(os.path.join(HERE, "42621400.23n")) else "/mnt/user-data/uploads"
NAV_FILE = os.path.join(DATA_DIR, "42621400.23n")
OBS_FILE = os.path.join(DATA_DIR, "42621400.23o")
OUT_DIR = os.path.join(HERE, "hasil")        # hasil (txt + gambar) disimpan di sub-folder "hasil"
EPOCHS = [(2023, 5, 20, 2, 0, 0.0),          # epoch 1
          (2023, 5, 20, 2, 0, 15.0)]         # epoch 2
ELEV_MASK = 10.0                             # derajat
DEMO_PRN = 3                                 # satelit yang dijabarkan langkah demi langkah

# ------------------------------------------------------------------ KONSTANTA
MU = 3.986005e14            # parameter gravitasi bumi (m^3/s^2)
OMEGA_E = 7.2921151467e-5   # kecepatan rotasi bumi (rad/s)
C = 299792458.0             # kecepatan cahaya (m/s)
F_REL = -4.442807633e-10    # konstanta koreksi relativistik (s/sqrt(m))
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)
os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LOG = []
def log(s=""):
    print(s)
    LOG.append(s)

# ------------------------------------------------------------------ WAKTU GPS
def gps_week_tow(y, mo, d, h, mi, s):
    t = dt.datetime(y, mo, d, h, mi, int(s)) + dt.timedelta(seconds=s - int(s))
    delta = t - dt.datetime(1980, 1, 6)
    sec = delta.total_seconds()
    week = int(sec // 604800)
    tow = sec - week * 604800
    return week, tow

# ------------------------------------------------------------------ BACA NAV RINEX 2.11
def D(s):
    s = s.strip().replace("D", "E").replace("d", "e")
    return float(s) if s else 0.0

def read_nav(fname):
    with open(fname) as f:
        lines = [l.rstrip("\r\n") for l in f]
    hdr = {}
    i = 0
    while "END OF HEADER" not in lines[i]:
        l = lines[i]
        if "ION ALPHA" in l:
            hdr["alpha"] = [D(l[2 + 12 * k: 14 + 12 * k]) for k in range(4)]
        if "ION BETA" in l:
            hdr["beta"] = [D(l[2 + 12 * k: 14 + 12 * k]) for k in range(4)]
        i += 1
    i += 1
    eph = []
    while i + 7 < len(lines) + 0 and lines[i].strip():
        l0 = lines[i]
        prn = int(l0[0:2])
        yy = int(l0[3:5]); mo = int(l0[6:8]); dd = int(l0[9:11])
        hh = int(l0[12:14]); mi = int(l0[15:17]); ss = float(l0[17:22])
        yy += 2000 if yy < 80 else 1900
        v = [D(l0[22 + 19 * k: 41 + 19 * k]) for k in range(3)]
        for k in range(1, 8):
            lk = lines[i + k].ljust(79)
            v += [D(lk[3 + 19 * j: 22 + 19 * j]) for j in range(4)]
        e = dict(prn=prn, toc_ymdhms=(yy, mo, dd, hh, mi, ss),
                 af0=v[0], af1=v[1], af2=v[2],
                 IODE=v[3], Crs=v[4], dn=v[5], M0=v[6],
                 Cuc=v[7], ecc=v[8], Cus=v[9], sqrtA=v[10],
                 toe=v[11], Cic=v[12], Omega0=v[13], Cis=v[14],
                 i0=v[15], Crc=v[16], omega=v[17], Omegadot=v[18],
                 IDOT=v[19], codesL2=v[20], week=v[21], L2P=v[22],
                 acc=v[23], health=v[24], TGD=v[25], IODC=v[26],
                 ttx=v[27], fit=v[28])
        _, e["toc"] = gps_week_tow(*e["toc_ymdhms"])
        eph.append(e)
        i += 8
    return hdr, eph

# ------------------------------------------------------------------ BACA OBS RINEX 2.11
def read_obs(fname, wanted_epochs):
    with open(fname) as f:
        lines = [l.rstrip("\r\n") for l in f]
    types, approx = [], None
    i = 0
    while "END OF HEADER" not in lines[i]:
        l = lines[i]
        if "# / TYPES OF OBSERV" in l:
            n = int(l[0:6])
            types = l[6:60].split()[:n]
        if "APPROX POSITION XYZ" in l:
            approx = np.array([float(l[0:14]), float(l[14:28]), float(l[28:42])])
        i += 1
    i += 1
    want = {tuple(round(x, 3) for x in w): w for w in wanted_epochs}
    nobs = len(types)
    nlines = math.ceil(nobs / 5)
    data = {}
    while i < len(lines):
        l = lines[i]
        if not l.strip():
            i += 1; continue
        yy = int(l[1:3]); mo = int(l[4:6]); dd = int(l[7:9])
        hh = int(l[10:12]); mi = int(l[13:15]); ss = float(l[15:26])
        flag = int(l[28:29]); nsat = int(l[29:32])
        sats = l[32:68]
        i += 1
        while len(sats) < 3 * nsat:          # baris lanjutan daftar satelit
            sats += lines[i][32:68]; i += 1
        sats = [sats[3 * k: 3 * k + 3] for k in range(nsat)]
        key = (2000 + yy, mo, dd, hh, mi, round(ss, 3))
        keep = key in want and flag == 0
        for s in sats:
            rec = {}
            vals = []
            for k in range(nlines):
                lk = lines[i + k].ljust(80)
                for j in range(5):
                    seg = lk[16 * j: 16 * j + 14].strip()
                    vals.append(float(seg) if seg else None)
            i += nlines
            if keep:
                for t, v in zip(types, vals):
                    rec[t] = v
                data.setdefault(key, {})[s.replace(" ", "0")] = rec
    return types, approx, data

# ------------------------------------------------------------------ PILIH EPHEMERIS
def pick_eph(eph_list, prn, tow):
    cands = [e for e in eph_list if e["prn"] == prn and e["health"] == 0]
    if not cands:
        return None
    return min(cands, key=lambda e: abs(wrap(tow - e["toe"])))

def wrap(t):
    if t > 302400: t -= 604800
    elif t < -302400: t += 604800
    return t

# ------------------------------------------------------------------ POSISI SATELIT (12 LANGKAH)
def sat_position(e, t_tx, verbose=False):
    """Posisi ECEF satelit pada waktu pancar t_tx (detik GPS dalam minggu)."""
    steps = {}
    A = e["sqrtA"] ** 2
    n0 = math.sqrt(MU / A ** 3)                 # 1. gerakan rata-rata terhitung
    tk = wrap(t_tx - e["toe"])                  # 2. waktu sejak toe
    n = n0 + e["dn"]                            # 3. gerakan rata-rata terkoreksi
    M = e["M0"] + n * tk                        # 4. anomali rata-rata
    E = M                                       # 5. persamaan Kepler (iterasi)
    for _ in range(30):
        E_new = M + e["ecc"] * math.sin(E)
        if abs(E_new - E) < 1e-13:
            E = E_new; break
        E = E_new
    nu = math.atan2(math.sqrt(1 - e["ecc"] ** 2) * math.sin(E),
                    math.cos(E) - e["ecc"])     # 6. anomali sejati
    phi = nu + e["omega"]                       # 7. argumen lintang
    s2p, c2p = math.sin(2 * phi), math.cos(2 * phi)
    du = e["Cus"] * s2p + e["Cuc"] * c2p        # 8. koreksi harmonik orde-2
    dr = e["Crs"] * s2p + e["Crc"] * c2p
    di = e["Cis"] * s2p + e["Cic"] * c2p
    u = phi + du                                # 9. argumen lintang, radius, inklinasi terkoreksi
    r = A * (1 - e["ecc"] * math.cos(E)) + dr
    i = e["i0"] + di + e["IDOT"] * tk
    xp, yp = r * math.cos(u), r * math.sin(u)   # 10. posisi pada bidang orbit
    Om = e["Omega0"] + (e["Omegadot"] - OMEGA_E) * tk - OMEGA_E * e["toe"]   # 11. bujur node naik
    X = xp * math.cos(Om) - yp * math.cos(i) * math.sin(Om)                  # 12. koordinat ECEF
    Y = xp * math.sin(Om) + yp * math.cos(i) * math.cos(Om)
    Z = yp * math.sin(i)
    steps = dict(A=A, n0=n0, tk=tk, n=n, M=M, E=E, nu=nu, phi=phi, du=du, dr=dr, di=di,
                 u=u, r=r, i=i, xp=xp, yp=yp, Omega=Om, X=X, Y=Y, Z=Z)
    return np.array([X, Y, Z]), E, steps

def sat_clock(e, t_tx, E):
    dtc = wrap(t_tx - e["toc"])
    poly = e["af0"] + e["af1"] * dtc + e["af2"] * dtc ** 2
    rel = F_REL * e["ecc"] * e["sqrtA"] * math.sin(E)
    return poly + rel - e["TGD"], poly, rel

# ------------------------------------------------------------------ GEODESI
def ecef2llh(x, y, z):
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - WGS84_E2))
    for _ in range(10):
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - N
        lat = math.atan2(z, p * (1 - WGS84_E2 * N / (N + h)))
    return math.degrees(lat), math.degrees(lon), h

def llh2ecef(lat, lon, h):
    lat, lon = math.radians(lat), math.radians(lon)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
    return np.array([(N + h) * math.cos(lat) * math.cos(lon),
                     (N + h) * math.cos(lat) * math.sin(lon),
                     (N * (1 - WGS84_E2) + h) * math.sin(lat)])

def az_el(rec, sat):
    lat, lon, _ = ecef2llh(*rec)
    la, lo = math.radians(lat), math.radians(lon)
    d = sat - rec
    e = -math.sin(lo) * d[0] + math.cos(lo) * d[1]
    n = -math.sin(la) * math.cos(lo) * d[0] - math.sin(la) * math.sin(lo) * d[1] + math.cos(la) * d[2]
    u = math.cos(la) * math.cos(lo) * d[0] + math.cos(la) * math.sin(lo) * d[1] + math.sin(la) * d[2]
    az = math.degrees(math.atan2(e, n)) % 360
    el = math.degrees(math.atan2(u, math.hypot(e, n)))
    return az, el

# ------------------------------------------------------------------ KOREKSI ATMOSFER
def klobuchar(alpha, beta, lat, lon, az, el, tow):
    """Model ionosfer Klobuchar (IS-GPS-200). Hasil: tunda ionosfer (meter), L1."""
    if alpha is None:
        return 0.0
    phi_u, lam_u = lat / 180.0, lon / 180.0            # semicircle
    E = el / 180.0
    psi = 0.0137 / (E + 0.11) - 0.022
    phi_i = phi_u + psi * math.cos(math.radians(az))
    phi_i = max(-0.416, min(0.416, phi_i))
    lam_i = lam_u + psi * math.sin(math.radians(az)) / math.cos(phi_i * math.pi)
    phi_m = phi_i + 0.064 * math.cos((lam_i - 1.617) * math.pi)
    t = (43200 * lam_i + tow) % 86400
    AMP = sum(alpha[k] * phi_m ** k for k in range(4))
    PER = sum(beta[k] * phi_m ** k for k in range(4))
    AMP = max(AMP, 0.0)
    PER = max(PER, 72000.0)
    x = 2 * math.pi * (t - 50400) / PER
    F = 1.0 + 16.0 * (0.53 - E) ** 3
    if abs(x) < 1.57:
        dI = F * (5e-9 + AMP * (1 - x ** 2 / 2 + x ** 4 / 24))
    else:
        dI = F * 5e-9
    return dI * C

def saastamoinen(h, el):
    """Tunda troposfer (meter) model Saastamoinen sederhana, atmosfer standar."""
    h = max(h, 0.0)
    P = 1013.25 * (1 - 2.2557e-5 * h) ** 5.2568
    T = 288.15 - 6.5e-3 * h
    e = 0.5 * 6.108 * math.exp(17.15 * (T - 273.15) / (T - 38.45)) * 0.5   # RH 50% (kasar)
    z = math.radians(90 - el)
    return 0.002277 / math.cos(z) * (P + (1255 / T + 0.05) * e - math.tan(z) ** 2)

# ------------------------------------------------------------------ SPP (LEAST SQUARES)
def spp(obs, eph, hdr, week, tow_rx, verbose=True):
    sats = []
    for s, rec in sorted(obs.items()):
        if not s.startswith("G"):
            continue
        P = rec.get("C1")
        if P is None:
            continue
        prn = int(s[1:])
        e = pick_eph(eph, prn, tow_rx)
        if e is None:
            continue
        sats.append((prn, P, e))

    x = np.zeros(4)           # [X, Y, Z, c*dt_rx]  mulai dari pusat bumi
    used = []
    for it in range(8):
        rows, res, used, geo = [], [], [], []
        have_pos = np.linalg.norm(x[:3]) > 1e6
        if have_pos:
            lat, lon, h = ecef2llh(*x[:3])
        for prn, P, e in sats:
            tau = P / C
            t_tx = tow_rx - tau - x[3] / C
            for _ in range(3):                       # iterasi waktu pancar
                pos, E, st = sat_position(e, t_tx)
                dts, _, _ = sat_clock(e, t_tx, E)
                t_tx = tow_rx - x[3] / C - P / C + dts
            pos, E, st = sat_position(e, t_tx)
            dts, poly, rel = sat_clock(e, t_tx, E)
            # koreksi rotasi bumi (Sagnac)
            rho0 = np.linalg.norm(pos - x[:3]) if have_pos else 2.2e7
            th = OMEGA_E * rho0 / C
            R = np.array([[math.cos(th), math.sin(th), 0],
                          [-math.sin(th), math.cos(th), 0],
                          [0, 0, 1]])
            posr = R @ pos
            dtrop = dion = 0.0
            if have_pos:
                az, el = az_el(x[:3], posr)
                if el < ELEV_MASK:
                    continue
                dion = klobuchar(hdr.get("alpha"), hdr.get("beta"), lat, lon, az, el, tow_rx) if it >= 2 else 0
                dtrop = saastamoinen(h, el) if it >= 2 else 0
            rho = np.linalg.norm(posr - x[:3])
            pred = rho + x[3] - C * dts + dion + dtrop
            los = (x[:3] - posr) / rho if rho > 0 else np.zeros(3)
            rows.append(np.r_[los, 1.0])
            res.append(P - pred)
            used.append(prn)
            geo.append((prn, posr, P, dts, dion, dtrop, rho))
        Hm, y = np.array(rows), np.array(res)
        dx, *_ = np.linalg.lstsq(Hm, y, rcond=None)
        x = x + dx
        if np.linalg.norm(dx[:3]) < 1e-4 and it > 2:
            break
    # DOP & sigma
    Q = np.linalg.inv(Hm.T @ Hm)
    lat, lon, h = ecef2llh(*x[:3])
    la, lo = math.radians(lat), math.radians(lon)
    Rg = np.array([[-math.sin(lo), math.cos(lo), 0],
                   [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                   [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])
    Qenu = Rg @ Q[:3, :3] @ Rg.T
    dop = dict(GDOP=math.sqrt(np.trace(Q)), PDOP=math.sqrt(np.trace(Q[:3, :3])),
               HDOP=math.sqrt(Qenu[0, 0] + Qenu[1, 1]), VDOP=math.sqrt(Qenu[2, 2]))
    return x, (lat, lon, h), geo, y, dop

# ------------------------------------------------------------------ GAMBAR
def fig_nav_message():
    """Gambar 1: struktur navigation message & komponen ephemeris."""
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
    ax.set_title("Struktur Navigation Message GPS (LNAV) dan komponen parameter ephemeris", fontsize=12, weight="bold")
    # frame
    ax.add_patch(plt.Rectangle((2, 50), 96, 6, fc="#dde7f5", ec="k"))
    ax.text(50, 53, "1 frame = 5 subframe = 30 detik (1 subframe = 6 detik = 300 bit);  25 frame = 1 superframe = 12,5 menit", ha="center", va="center", fontsize=9)
    cols = ["#f9d5d3", "#d6efd4", "#fff0c2", "#e6dcf5", "#d9f0f4"]
    names = ["Subframe 1", "Subframe 2", "Subframe 3", "Subframe 4", "Subframe 5"]
    contents = [
        "Koreksi jam satelit\nWN, URA, health\nTGD, IODC\ntoc, af0, af1, af2",
        "Ephemeris (1)\nIODE, Crs, Δn, M0\nCuc, e, Cus, √A, toe",
        "Ephemeris (2)\nCic, Ω0, Cis, i0\nCrc, ω, dΩ/dt, IDOT, IODE",
        "Almanak sat 25-32,\nparameter ionosfer\n(α, β), UTC, health",
        "Almanak sat 1-24,\nhealth, WNa, toa",
    ]
    for k in range(5):
        x0 = 2 + k * 19.4
        ax.add_patch(plt.Rectangle((x0, 30), 18.6, 17, fc=cols[k], ec="k"))
        ax.text(x0 + 9.3, 44.5, names[k], ha="center", weight="bold", fontsize=9)
        ax.text(x0 + 9.3, 37, contents[k], ha="center", va="center", fontsize=7.6)
    # kelompok
    ax.annotate("", xy=(2, 28), xytext=(58, 28), arrowprops=dict(arrowstyle="<->"))
    ax.text(30, 25.5, "Dipakai untuk perhitungan posisi satelit & jam", ha="center", fontsize=9, weight="bold", color="#a40000")
    # tabel komponen
    comp = [
        ("Parameter Keplerian", "√A, e, i0, Ω0, ω, M0", "bentuk, ukuran & orientasi orbit"),
        ("Parameter koreksi orbit", "Δn, dΩ/dt, IDOT, Cuc, Cus,\nCrc, Crs, Cic, Cis", "gangguan (perturbasi) orbit"),
        ("Referensi waktu", "toe, toc, WN", "waktu referensi ephemeris/jam"),
        ("Koreksi jam satelit", "af0, af1, af2, TGD", "kesalahan jam satelit"),
        ("Koreksi ionosfer", "α0-α3, β0-β3 (Klobuchar)", "tunda ionosfer sinyal L1"),
    ]
    ax.text(2, 21, "Kelompok komponen", weight="bold", fontsize=9)
    ax.text(30, 21, "Simbol", weight="bold", fontsize=9)
    ax.text(68, 21, "Fungsi", weight="bold", fontsize=9)
    for k, (a, b, c) in enumerate(comp):
        yy = 16.5 - k * 4.3
        ax.plot([2, 98], [yy + 2.6, yy + 2.6], color="#999", lw=0.5)
        ax.text(2, yy, a, fontsize=8.5, va="center")
        ax.text(30, yy, b, fontsize=8.5, va="center")
        ax.text(68, yy, c, fontsize=8.5, va="center")
    fig.savefig(os.path.join(FIG_DIR, "gambar1_struktur_navmessage.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

def fig_kepler(steps, prn):
    """Gambar 2: elemen orbit Kepler (bidang orbit dan orientasi 3D)."""
    fig = plt.figure(figsize=(12, 5.6))
    # kiri: bidang orbit
    ax = fig.add_subplot(1, 2, 1)
    a = 1.0; e = 0.45     # eksentrisitas dibesar-besarkan agar terlihat jelas
    th = np.linspace(0, 2 * np.pi, 400)
    b = a * math.sqrt(1 - e ** 2)
    ax.plot(a * np.cos(th) - a * e, b * np.sin(th), "b", lw=1.5)
    ax.add_patch(plt.Circle((-a * e, 0), a, fill=False, ls="--", color="gray"))
    ax.plot(0, 0, "o", color="g", ms=11); ax.text(0.04, -0.12, "Bumi (fokus)", color="g")
    Ed = 1.15
    xs, ys = a * math.cos(Ed) - a * e, b * math.sin(Ed)
    ax.plot(xs, ys, "ro"); ax.text(xs + 0.03, ys + 0.03, "Satelit", color="r")
    ax.plot([0, xs], [0, ys], "r-", lw=1)
    ax.plot([-a * e, a * math.cos(Ed) - a * e], [0, a * math.sin(Ed)], "k:", lw=1)
    ax.plot([xs, xs], [ys, a * math.sin(Ed)], "k:", lw=0.8)
    ax.plot([-a * e, xs], [0, ys], "k:", lw=0.6)
    nu = math.atan2(ys, xs)
    ax.annotate("", xy=(1.2, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="k"))
    ax.text(1.22, -0.05, "perigee", fontsize=9)
    ax.text(0.28, 0.07, "ν (anomali sejati)", fontsize=9)
    ax.text(-1.55, 0.55, "E (anomali eksentrik,\nsudut pada lingkaran bantu)", fontsize=8)
    ax.text(0.2, 0.95, "r = a(1 − e cos E) + δr", fontsize=9, color="r")
    ax.text(-1.5, -1.05, "a = (√A)², e = eksentrisitas,  M = M0 + (n0+Δn)·tk", fontsize=9)
    ax.set_aspect("equal"); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.15, 1.15)
    ax.set_title("(a) Geometri pada bidang orbit", fontsize=10)
    ax.axis("off")
    # kanan: orientasi 3D
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    inc = math.radians(55); Om = math.radians(50); w = math.radians(60)
    t = np.linspace(0, 2 * np.pi, 300)
    R0 = 2.0
    Rz = lambda x: np.array([[math.cos(x), -math.sin(x), 0], [math.sin(x), math.cos(x), 0], [0, 0, 1]])
    Rx = lambda x: np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    Q = Rz(Om) @ Rx(inc)
    orb = Q @ np.vstack([R0 * np.cos(t), R0 * np.sin(t), 0 * t])
    ax3.plot(*orb, "b", lw=1.6)
    ax3.plot(R0 * np.cos(t), R0 * np.sin(t), 0 * t, color="gray", lw=0.9)          # ekuator
    ax3.text(-2.3, 1.2, 0, "bidang ekuator", color="gray", fontsize=8)
    ax3.text(-1.2, -1.8, 1.4, "bidang orbit", color="b", fontsize=8)
    # sumbu
    ax3.plot([0, 2.8], [0, 0], [0, 0], "k-", lw=1.2); ax3.text(2.9, 0, 0, "X (ke titik Aries γ)", fontsize=8)
    ax3.plot([0, 0], [0, 0], [0, 2.6], "k-", lw=1.2); ax3.text(0, 0, 2.7, "Z (kutub utara)", fontsize=8)
    # garis node naik
    node = Rz(Om) @ np.array([R0, 0, 0])
    ax3.plot([0, node[0]], [0, node[1]], [0, 0], "g-", lw=2.5)
    ax3.text(node[0] + .1, node[1] + .1, 0, "node naik", color="g", fontsize=8)
    # busur Omega (di ekuator)
    a_ = np.linspace(0, Om, 30)
    ax3.plot(1.0 * np.cos(a_), 1.0 * np.sin(a_), 0 * a_, "g-", lw=1.5)
    ax3.text(1.15, 0.45, 0, "Ω", color="g", fontsize=12)
    # busur inklinasi i (di bidang tegak lurus garis node)
    ex = np.array([-math.sin(Om), math.cos(Om), 0])    # arah di ekuator tegak lurus node
    ez = np.array([0, 0, 1.0])
    b_ = np.linspace(0, inc, 30)
    base = node * 0.55
    arc = np.array([base + 0.8 * (math.cos(x) * ex + math.sin(x) * ez) for x in b_]).T
    ax3.plot(*arc, "r-", lw=1.5)
    mid = base + 0.8 * (math.cos(inc / 2) * ex + math.sin(inc / 2) * ez)
    ax3.text(mid[0] + .1, mid[1] + .1, mid[2], "i", color="r", fontsize=12)
    # perigee dan sudut omega (di bidang orbit)
    perigee = Q @ np.array([R0 * math.cos(w), R0 * math.sin(w), 0])
    ax3.plot([0, perigee[0]], [0, perigee[1]], [0, perigee[2]], "m-", lw=2)
    ax3.text(perigee[0], perigee[1], perigee[2] + .15, "perigee", color="m", fontsize=8)
    c_ = np.linspace(0, w, 30)
    arc2 = Q @ np.vstack([0.7 * np.cos(c_), 0.7 * np.sin(c_), 0 * c_])
    ax3.plot(*arc2, "m-", lw=1.5)
    mid2 = Q @ np.array([0.9 * math.cos(w / 2), 0.9 * math.sin(w / 2), 0])
    ax3.text(mid2[0], mid2[1], mid2[2], "ω", color="m", fontsize=12)
    ax3.plot([0], [0], [0], "go", ms=8)
    ax3.set_title("(b) Orientasi orbit: Ω (hijau), i (merah), ω (ungu)", fontsize=10)
    ax3.set_axis_off(); ax3.view_init(24, 25)
    ax3.set_box_aspect((1, 1, 0.9))
    fig.suptitle("Gambar 2. Komponen Keplerian dari navigation message", weight="bold")
    fig.savefig(os.path.join(FIG_DIR, "gambar2_elemen_kepler.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

def fig_flow():
    """Gambar 3: diagram alir perhitungan."""
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.set_xlim(0, 10); ax.set_ylim(0, 28); ax.axis("off")
    boxes = [
        "Baca RINEX nav (.23n): ephemeris + α/β",
        "Baca RINEX obs (.23o): pseudorange C1 pada 2 epoch",
        "Waktu pancar: t_tx = t_rx − P/c − koreksi jam",
        "tk = t_tx − toe ;  n = √(μ/a³) + Δn",
        "M = M0 + n·tk ;  Kepler: E = M + e·sin E (iterasi)",
        "ν = atan2(√(1−e²)·sinE , cosE − e)",
        "Φ = ν + ω ;  δu, δr, δi (Cuc,Cus,Crc,Crs,Cic,Cis)",
        "u = Φ+δu ; r = a(1−e·cosE)+δr ; i = i0+δi+IDOT·tk",
        "x' = r·cos u ;  y' = r·sin u   (bidang orbit)",
        "Ω = Ω0 + (dΩ/dt − ωe)·tk − ωe·toe",
        "X, Y, Z satelit (ECEF) + rotasi bumi (Sagnac)",
        "Koreksi: jam satelit, relativistik, TGD, ionosfer, troposfer",
        "Least squares: [X, Y, Z, c·δt] dari ≥ 4 satelit",
        "Hasil: X, Y, Z → φ, λ, h (WGS-84) + DOP",
    ]
    n = len(boxes)
    for k, t in enumerate(boxes):
        y = 27 - k * 1.95
        ax.add_patch(plt.Rectangle((0.8, y - 0.7), 8.4, 1.4, fc="#e8f0fb" if k not in (0, 1, n - 1) else "#d6efd4", ec="k"))
        ax.text(5, y, f"{k + 1}. {t}", ha="center", va="center", fontsize=9)
        if k < n - 1:
            ax.annotate("", xy=(5, y - 1.25), xytext=(5, y - 0.7), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Gambar 3. Diagram alir perhitungan koordinat dari Navigation Message", weight="bold")
    fig.savefig(os.path.join(FIG_DIR, "gambar3_diagram_alir.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

def fig_skyplot(results):
    fig = plt.figure(figsize=(11, 5.4))
    for k, (label, x, geo) in enumerate(results):
        ax = fig.add_subplot(1, 2, k + 1, projection="polar")
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_rlim(90, 0); ax.set_rticks([0, 30, 60, 90])
        for prn, pos, *_ in geo:
            az, el = az_el(x[:3], pos)
            ax.plot(math.radians(az), el, "o", ms=14, color="#2a6fdb")
            ax.text(math.radians(az), el, f"G{prn:02d}", color="w", ha="center", va="center", fontsize=7, weight="bold")
        ax.set_title(label, fontsize=10, pad=14)
    fig.suptitle("Gambar 4. Skyplot satelit GPS yang dipakai", weight="bold", y=1.06)
    fig.savefig(os.path.join(FIG_DIR, "gambar4_skyplot.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

def fig_geometry(x, geo, label):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    R = 6371e3
    ax.plot_wireframe(R * np.cos(u) * np.sin(v), R * np.sin(u) * np.sin(v), R * np.cos(v), color="#9fb8d9", lw=0.3)
    for prn, pos, *_ in geo:
        ax.plot([x[0], pos[0]], [x[1], pos[1]], [x[2], pos[2]], "-", color="#bbb", lw=0.7)
        ax.plot(*pos, "ro"); ax.text(*pos, f" G{prn:02d}", fontsize=8)
    ax.plot(*x[:3], "g^", ms=10)
    ax.set_title(f"Gambar 5. Posisi satelit (ECEF) dan receiver – {label}", fontsize=10, weight="bold")
    ax.set_box_aspect((1, 1, 1))
    fig.savefig(os.path.join(FIG_DIR, "gambar5_geometri_satelit.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------------ UTAMA
def main():
    hdr, eph = read_nav(NAV_FILE)
    types, approx, obsdata = read_obs(OBS_FILE, EPOCHS)
    log("=" * 78)
    log("PERHITUNGAN KOORDINAT DARI NAVIGATION MESSAGE GPS (SPP)")
    log("=" * 78)
    log(f"Jumlah record ephemeris GPS : {len(eph)}")
    log(f"Jenis observasi di file obs : {types}")
    log(f"Koordinat pendekatan header : X={approx[0]:.4f}  Y={approx[1]:.4f}  Z={approx[2]:.4f}")
    la0, lo0, h0 = ecef2llh(*approx)
    log(f"                              lat={la0:.6f}  lon={lo0:.6f}  h={h0:.3f}")
    log(f"Parameter Klobuchar alpha   : {hdr['alpha']}")
    log(f"Parameter Klobuchar beta    : {hdr['beta']}")

    fig_nav_message(); fig_flow()

    results = []
    summary = []
    demo_done = False
    for ep in EPOCHS:
        key = tuple(round(v, 3) for v in ep)
        obs = obsdata[key]
        week, tow = gps_week_tow(*ep)
        label = "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:04.1f} GPS (minggu {}, TOW {:.1f} s)".format(*ep, week, tow)
        log("\n" + "-" * 78)
        log(f"EPOCH: {label}")
        log("-" * 78)

        # --- tabel langkah demi langkah satu satelit
        if not demo_done:
            e = pick_eph(eph, DEMO_PRN, tow)
            P = obs[f"G{DEMO_PRN:02d}"]["C1"]
            t_tx = tow - P / C
            log(f"\n>>> CONTOH PERHITUNGAN LENGKAP SATELIT G{DEMO_PRN:02d} (epoch pertama)")
            log(f"Pseudorange C1 = {P:.3f} m ; waktu rambat ≈ {P / C * 1000:.4f} ms ; t_tx ≈ {t_tx:.6f} s")
            log("Data ephemeris yang dipilih (dari .23n):")
            for k in ["IODE", "toe", "toc", "sqrtA", "ecc", "i0", "Omega0", "omega", "M0", "dn", "Omegadot", "IDOT",
                      "Cuc", "Cus", "Crc", "Crs", "Cic", "Cis", "af0", "af1", "af2", "TGD"]:
                log(f"   {k:9s} = {e[k]: .12e}")
            pos, E, st = sat_position(e, t_tx)
            dts, poly, rel = sat_clock(e, t_tx, E)
            log("Langkah perhitungan posisi:")
            names = [("A", "a = (√A)²  [m]"), ("n0", "n0 = √(μ/a³)  [rad/s]"), ("tk", "tk = t − toe  [s]"),
                     ("n", "n = n0 + Δn  [rad/s]"), ("M", "M = M0 + n·tk  [rad]"), ("E", "E (Kepler)  [rad]"),
                     ("nu", "ν anomali sejati  [rad]"), ("phi", "Φ = ν + ω  [rad]"), ("du", "δu  [rad]"),
                     ("dr", "δr  [m]"), ("di", "δi  [rad]"), ("u", "u  [rad]"), ("r", "r  [m]"), ("i", "i  [rad]"),
                     ("xp", "x' [m]"), ("yp", "y' [m]"), ("Omega", "Ω terkoreksi [rad]"),
                     ("X", "X ECEF [m]"), ("Y", "Y ECEF [m]"), ("Z", "Z ECEF [m]")]
            for k, d in names:
                log(f"   {d:28s} = {st[k]: .9f}")
            log(f"Koreksi jam satelit: polinomial = {poly:.9e} s ; relativistik = {rel:.9e} s ; TGD = {e['TGD']:.3e} s")
            log(f"   δt_sat = {dts:.9e} s  -> {dts * C:.3f} m")
            fig_kepler(st, DEMO_PRN)
            demo_done = True

        x, llh, geo, resid, dop = spp(obs, eph, hdr, week, tow)
        d = x[:3] - approx
        log(f"\nSatelit dipakai ({len(geo)}): " + " ".join(f"G{g[0]:02d}" for g in geo))
        log("\nPRN   X_sat (m)        Y_sat (m)        Z_sat (m)     P_C1 (m)       dt_sat(µs)  Iono(m) Trop(m)")
        for prn, pos, P, dts, dion, dtrop, rho in geo:
            log(f"G{prn:02d} {pos[0]:15.3f} {pos[1]:15.3f} {pos[2]:15.3f} {P:14.3f} {dts * 1e6:11.4f} {dion:7.2f} {dtrop:7.2f}")
        log(f"\nHASIL KOORDINAT RECEIVER (ECEF WGS-84):")
        log(f"   X = {x[0]:.3f} m   Y = {x[1]:.3f} m   Z = {x[2]:.3f} m")
        log(f"   Lintang  = {llh[0]:.8f}°   Bujur = {llh[1]:.8f}°   Tinggi elipsoid = {llh[2]:.3f} m")
        log(f"   Kesalahan jam receiver c·dt = {x[3]:.3f} m ({x[3] / C * 1e3:.6f} ms)")
        log(f"   Selisih terhadap koordinat header : dX={d[0]:.3f}  dY={d[1]:.3f}  dZ={d[2]:.3f}  (3D = {np.linalg.norm(d):.3f} m)")
        log(f"   RMS residual = {math.sqrt(np.mean(resid ** 2)):.3f} m ;  DOP: " + "  ".join(f"{k}={v:.2f}" for k, v in dop.items()))
        results.append((label, x, geo))
        summary.append((ep, x, llh, d))

    fig_skyplot(results)
    fig_geometry(results[0][1], results[0][2], "epoch 1")

    log("\n" + "=" * 78)
    log("RINGKASAN")
    log("=" * 78)
    log("Epoch              X (m)           Y (m)           Z (m)        lat (°)      lon (°)     h (m)")
    for ep, x, llh, d in summary:
        log(f"{ep[3]:02d}:{ep[4]:02d}:{ep[5]:04.1f} {x[0]:15.3f} {x[1]:15.3f} {x[2]:15.3f} {llh[0]:12.7f} {llh[1]:12.7f} {llh[2]:9.3f}")
    with open(os.path.join(OUT_DIR, "hasil_perhitungan.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))

if __name__ == "__main__":
    main()
