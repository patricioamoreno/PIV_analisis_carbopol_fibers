"""
parche_gradientes.py
====================
Recalcula gamma_dot y vort DESDE EL CACHE EXISTENTE, sin volver a leer el PIV
crudo ni rehacer la asignacion de zonas.

POR QUE EXISTE
--------------
cache_zonas/*.npz ya guarda x, y, u, v por punto. Los gradientes son funcion
exclusiva de esos cuatro arrays, asi que no hace falta reconstruir el cache
completo para aplicar el filtro de condicionamiento: basta recalcular las dos
columnas derivadas y reescribirlas.

Ademas el bucle original recorre punto a punto en Python con un lstsq 3x3 por
punto. Aqui se vectoriza dentro de cada frame (una sola query al KDTree y un
solo np.linalg.solve por lotes), lo que reduce el costo en 1-2 ordenes de
magnitud.

QUE CAMBIA RESPECTO AL CALCULO ORIGINAL
---------------------------------------
  1. MIN_VECINOS 3 -> 4. Con exactamente 3 vecinos el sistema [1, dx, dy] queda
     exactamente determinado: el ajuste interpola, no promedia, y un solo vector
     espurio pasa integro al gradiente.
  2. Rechazo por cond(A) > COND_MAX. En bordes de material los vecinos se
     alinean a lo largo del frente; la matriz de diseno es casi singular y el
     gradiente transversal queda indeterminado. lstsq devuelve valores enormes
     en vez de NaN, y esa es la fuente probable de los gamma_dot ~ 1e6.

USO
---
    python parche_gradientes.py              # solo diagnostica, no escribe
    python parche_gradientes.py --aplicar    # reescribe los .npz (hace backup)

El modo diagnostico informa cuanto se moverian las medianas por zona, que es
el estadistico que realmente entra a la tabla. Si se mueven poco, tienes un
argumento para reportar que el resultado es robusto al filtro y no hace falta
aplicar nada.
"""

import os
import sys
import glob
import shutil
import time

import numpy as np
from scipy.spatial import cKDTree

CACHE_DIR   = "cache_zonas"
K_VECINOS   = 6
DIST_MAX_MM = 5.0
MIN_VECINOS = 4
COND_MAX    = 1.0e4


# ----------------------------------------------------------------------
def gradientes_vectorizado(x, y, u, v, frame_idx,
                           k=K_VECINOS, dist_max=DIST_MAX_MM,
                           min_vecinos=MIN_VECINOS, cond_max=COND_MAX):
    """
    gamma_dot y vort por ajuste lineal local, vectorizado por frame.

    Identico en formulacion al calculo original:
        gamma_dot = sqrt(2*dudx^2 + 2*dvdy^2 + (dudy + dvdx)^2)   [= sqrt(2 D:D)]
        vort      = dvdx - dudy
    La diferencia esta en min_vecinos y en el rechazo por condicionamiento.

    Devuelve (gamma_dot, vort, n_descartados_por_cond).
    """
    n_tot     = len(x)
    gamma_dot = np.full(n_tot, np.nan, dtype=np.float32)
    vort      = np.full(n_tot, np.nan, dtype=np.float32)
    n_cond    = 0

    orden = np.argsort(frame_idx, kind="stable")
    bordes = np.searchsorted(frame_idx[orden], np.unique(frame_idx))
    bordes = np.append(bordes, n_tot)

    for b0, b1 in zip(bordes[:-1], bordes[1:]):
        idx_g = orden[b0:b1]
        n = len(idx_g)
        if n < min_vecinos + 1:
            continue

        xf, yf = x[idx_g].astype(float), y[idx_g].astype(float)
        uf, vf = u[idx_g].astype(float), v[idx_g].astype(float)

        tree = cKDTree(np.column_stack([xf, yf]))
        kq = min(k + 1, n)
        dd, ii = tree.query(np.column_stack([xf, yf]), k=kq,
                            distance_upper_bound=dist_max, workers=-1)

        # La primera columna es el propio punto (distancia 0): se descarta.
        dd, ii = dd[:, 1:], ii[:, 1:]
        valido = np.isfinite(dd) & (ii < n)
        n_vec  = valido.sum(axis=1)

        activo = np.where(n_vec >= min_vecinos)[0]
        if len(activo) == 0:
            continue

        ii_a, val_a = ii[activo], valido[activo]
        ii_s = np.where(val_a, ii_a, 0)          # indice seguro para fancy-index
        w    = val_a.astype(float)               # peso 0 anula al vecino invalido

        dx = (xf[ii_s] - xf[activo][:, None]) * w
        dy = (yf[ii_s] - yf[activo][:, None]) * w
        uu = uf[ii_s] * w
        vv = vf[ii_s] * w
        w1 = w

        # Ecuaciones normales A^T A c = A^T b, con A = [1, dx, dy] por punto.
        S1  = w1.sum(1)
        Sx  = dx.sum(1);      Sy  = dy.sum(1)
        Sxx = (dx * dx).sum(1); Syy = (dy * dy).sum(1); Sxy = (dx * dy).sum(1)

        AtA = np.empty((len(activo), 3, 3))
        AtA[:, 0, 0] = S1;  AtA[:, 0, 1] = Sx;  AtA[:, 0, 2] = Sy
        AtA[:, 1, 0] = Sx;  AtA[:, 1, 1] = Sxx; AtA[:, 1, 2] = Sxy
        AtA[:, 2, 0] = Sy;  AtA[:, 2, 1] = Sxy; AtA[:, 2, 2] = Syy

        Atu = np.stack([uu.sum(1), (dx * uu).sum(1), (dy * uu).sum(1)], axis=1)
        Atv = np.stack([vv.sum(1), (dx * vv).sum(1), (dy * vv).sum(1)], axis=1)

        # cond(A) = sqrt(cond(A^T A)) para A de rango completo.
        with np.errstate(all="ignore"):
            cond = np.sqrt(np.linalg.cond(AtA))
        ok = np.isfinite(cond) & (cond <= cond_max)
        n_cond += int((~ok).sum())
        if not ok.any():
            continue

        sel = activo[ok]
        try:
            # NumPy >= 2.0: un b 2-D se interpreta como matriz, no como pila
            # de vectores. Hay que anadir el eje explicitamente.
            cu = np.linalg.solve(AtA[ok], Atu[ok][..., None])[..., 0]
            cv = np.linalg.solve(AtA[ok], Atv[ok][..., None])[..., 0]
        except np.linalg.LinAlgError:
            continue

        dudx, dudy = cu[:, 1], cu[:, 2]
        dvdx, dvdy = cv[:, 1], cv[:, 2]

        g = np.sqrt(2.0 * dudx**2 + 2.0 * dvdy**2 + (dudy + dvdx)**2)
        gamma_dot[idx_g[sel]] = g.astype(np.float32)
        vort[idx_g[sel]]      = (dvdx - dudy).astype(np.float32)

    return gamma_dot, vort, n_cond


# ----------------------------------------------------------------------
def resumen_por_zona(zona, valores, etiqueta):
    """Mediana por zona, que es el estadistico que entra a la tabla."""
    out = {}
    for z in np.unique(zona):
        if z == "fuera":
            continue
        m = (zona == z) & np.isfinite(valores)
        if m.sum() >= 10:
            out[str(z)] = float(np.median(valores[m]))
    return out


def main(aplicar=False):
    archivos = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    if not archivos:
        print(f"[ERROR] No hay .npz en '{CACHE_DIR}/'")
        return 1

    print(f"{len(archivos)} caches encontrados en '{CACHE_DIR}/'")
    print(f"Modo: {'APLICAR (reescribe)' if aplicar else 'DIAGNOSTICO (no escribe)'}\n")

    filas = []
    t_ini = time.time()

    for i, path in enumerate(archivos, 1):
        d = np.load(path, allow_pickle=True)
        x, y, u, v = d["x"], d["y"], d["u"], d["v"]
        fidx, zona = d["frame_idx"], d["zona"]
        g_old, w_old = d["gamma_dot"], d["vort"]

        t0 = time.time()
        g_new, w_new, n_cond = gradientes_vectorizado(x, y, u, v, fidx)
        dt = time.time() - t0

        val_old = np.isfinite(g_old).sum()
        val_new = np.isfinite(g_new).sum()

        med_old = resumen_por_zona(zona, g_old, "old")
        med_new = resumen_por_zona(zona, g_new, "new")
        cambios = [abs(med_new[z] - med_old[z]) / abs(med_old[z]) * 100
                   for z in med_old if z in med_new and med_old[z] != 0]

        nombre = os.path.basename(path).replace("_zonas.npz", "")
        print(f"[{i}/{len(archivos)}] {nombre[:42]:42s} "
              f"{dt:5.1f}s  puntos {val_old}->{val_new} "
              f"({100*(val_old-val_new)/max(val_old,1):+.1f}%)  "
              f"cambio mediana gamma_dot: "
              f"{np.median(cambios) if cambios else 0:.1f}% mediano")

        filas.append(dict(toma=nombre, val_old=val_old, val_new=val_new,
                          n_cond=n_cond,
                          cambio_med=np.median(cambios) if cambios else 0.0,
                          cambio_max=max(cambios) if cambios else 0.0))

        if aplicar:
            if not os.path.exists(path + ".bak"):
                shutil.copy2(path, path + ".bak")
            campos = {k: d[k] for k in d.files}
            campos["gamma_dot"] = g_new
            campos["vort"]      = w_new
            np.savez_compressed(path, **campos)

    print(f"\nTiempo total: {time.time()-t_ini:.1f} s")

    cm = [f["cambio_med"] for f in filas]
    cx = [f["cambio_max"] for f in filas]
    pf = [100*(f["val_old"]-f["val_new"])/max(f["val_old"], 1) for f in filas]
    print("\n" + "=" * 66)
    print("RESUMEN")
    print("=" * 66)
    print(f"  Puntos descartados de mas : {np.mean(pf):.2f} % (media por toma)")
    print(f"  Cambio en la MEDIANA de gamma_dot por zona:")
    print(f"      mediano : {np.median(cm):.2f} %")
    print(f"      maximo  : {max(cx):.2f} %")
    print()
    if np.median(cm) < 2.0:
        print("  -> Las medianas por zona practicamente no se mueven.")
        print("     El resultado es ROBUSTO al filtro de condicionamiento.")
        print("     Puedes reportarlo asi y NO aplicar el parche.")
    else:
        print("  -> Las medianas se mueven de forma apreciable.")
        print("     Conviene aplicar el parche y rehacer la tabla.")
    if not aplicar:
        print("\n  (no se escribio nada; usa --aplicar para reescribir)")
    return 0


if __name__ == "__main__":
    sys.exit(main(aplicar="--aplicar" in sys.argv))
