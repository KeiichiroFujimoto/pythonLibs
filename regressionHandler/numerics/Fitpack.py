"""Dierckx FITPACK smoothing spline (curfit / fpcurf) and B-spline evaluation (splev).

A line-by-line port of the Fortran routines fpcurf, fpbspl, fpgivs, fprota,
fpback, fpdisc, fpknot and fprati as shipped with scipy, so that a smoothing
spline (splrep / UnivariateSpline with s > 0) is reproduced to round-off:
the same knot insertion, the same rational interpolation for the smoothing
parameter p, the same tolerances (tol = 0.001, maxit = 20).

Arrays keep the Fortran 1-based layout (index 0 unused) to keep the port
checkable against the original. ``CurfitState`` carries everything fpcurf
needs to continue with iopt = 1 (a new smoothing factor on the knots found so
far, or a larger ``nest``), which is what UnivariateSpline.set_smoothing_factor
does.

References:
    P. Dierckx, "Curve and Surface Fitting with Splines", Oxford (1993).
    P. Dierckx, "An algorithm for smoothing, differentiation and integration of
    experimental data using spline functions", J. Comp. Appl. Maths 1 (1975).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TOL = 0.001
MAXIT = 20

CURFIT_MESSAGES = {
    1: "The required storage space exceeds the available storage space (nest too small): "
       "the result is a least-squares spline on the knots found so far.",
    2: "A theoretically impossible result was found during the iteration process for finding "
       "a smoothing spline with fp = s: s too small.",
    3: "The maximal number of iterations (20) allowed for finding a smoothing spline "
       "with fp = s has been reached: s too small.",
    10: "Error on entry, no approximation returned.",
}


@dataclass
class CurfitState:
    """In/out arrays of fpcurf (1-based, length nest + 1) plus the scalars."""
    nest: int
    n: int = 0
    fp: float = 0.0
    ier: int = 0
    t: list = field(default_factory=list)
    c: list = field(default_factory=list)
    fpint: list = field(default_factory=list)
    nrdata: list = field(default_factory=list)

    @classmethod
    def empty(cls, nest: int) -> "CurfitState":
        return cls(nest=nest, t=[0.0] * (nest + 1), c=[0.0] * (nest + 1), fpint=[0.0] * (nest + 1),
                   nrdata=[0] * (nest + 1))

    def resized(self, nest: int) -> "CurfitState":
        """Same state with room for ``nest`` knots (UnivariateSpline._reset_nest)."""
        if nest < self.n:
            raise ValueError("nest can only be increased")
        pad = nest - self.nest
        return CurfitState(nest=nest, n=self.n, fp=self.fp, ier=self.ier, t=self.t + [0.0] * pad,
                           c=self.c + [0.0] * pad, fpint=self.fpint + [0.0] * pad,
                           nrdata=self.nrdata + [0] * pad)

    def tck(self, k: int):
        """Knots t (n,) and coefficients c (n,) as 0-based arrays (c padded with zeros like FITPACK)."""
        n = self.n
        c = np.zeros(n)
        c[:n - k - 1] = self.c[1:n - k]
        return np.array(self.t[1:n + 1], dtype=float), c


# ---------------------------------------------------------------- FITPACK kernels (1-based)
def fpbspl(t, k, x, l):
    """Values h[1..k+1] of the k+1 nonzero B-splines at t[l] <= x < t[l+1]."""
    h = [0.0] * (k + 2)
    hh = [0.0] * (k + 1)
    h[1] = 1.0
    for j in range(1, k + 1):
        for i in range(1, j + 1):
            hh[i] = h[i]
        h[1] = 0.0
        for i in range(1, j + 1):
            li = l + i
            lj = li - j
            if t[li] != t[lj]:
                f = hh[i] / (t[li] - t[lj])
                h[i] = h[i] + f * (t[li] - x)
                h[i + 1] = f * (x - t[lj])
            else:
                h[i + 1] = 0.0
    return h


def fpgivs(piv, ww):
    """Givens rotation eliminating piv against ww; returns (cos, sin, new ww)."""
    store = abs(piv)
    if store >= ww:
        dd = store * math.sqrt(1.0 + (ww / piv) ** 2)
    else:
        dd = ww * math.sqrt(1.0 + (piv / ww) ** 2)
    return ww / dd, piv / dd, dd


def fprota(cos, sin, a, b):
    """Apply a Givens rotation to (a, b); returns (new a, new b)."""
    return cos * a - sin * b, cos * b + sin * a


def fpback(a, z, n, k, c):
    """Back substitution for the upper triangular band matrix a (bandwidth k); c may alias z."""
    k1 = k - 1
    c[n] = z[n] / a[n][1]
    i = n - 1
    if i == 0:
        return
    for j in range(2, n + 1):
        store = z[i]
        i1 = k1
        if j <= k1:
            i1 = j - 1
        mm = i
        for l in range(1, i1 + 1):
            mm += 1
            store = store - c[mm] * a[i][l + 1]
        c[i] = store / a[i][1]
        i -= 1


def fpdisc(t, n, k2, b):
    """Discontinuity jumps of the k-th derivative of the B-splines at the interior knots."""
    k1 = k2 - 1
    k = k1 - 1
    nk1 = n - k1
    nrint = nk1 - k
    an = float(nrint)
    fac = an / (t[nk1 + 1] - t[k1])
    h = [0.0] * (2 * k2 + 1)
    for l in range(k2, nk1 + 1):
        lmk = l - k1
        for j in range(1, k1 + 1):
            ik = j + k1
            lj = l + j
            lk = lj - k2
            h[j] = t[l] - t[lk]
            h[ik] = t[l] - t[lj]
        lp = lmk
        for j in range(1, k2 + 1):
            jk = j
            prod = h[j]
            for _ in range(1, k + 1):
                jk += 1
                prod = prod * h[jk] * fac
            lk = lp + k1
            b[lmk][j] = (t[lk] - t[lp]) / prod
            lp += 1


def fpknot(x, t, n, fpint, nrdata, nrint, istart):
    """Add one knot in the interval with the largest residual sum; returns (n, nrint)."""
    k = (n - nrint - 1) // 2
    fpmax = 0.0
    jbegin = istart
    iserr = True
    number = maxpt = maxbeg = 0
    for j in range(1, nrint + 1):
        jpoint = nrdata[j]
        if not (fpmax >= fpint[j] or jpoint == 0):
            iserr = False
            fpmax = fpint[j]
            number = j
            maxpt = jpoint
            maxbeg = jbegin
        jbegin = jbegin + jpoint + 1
    if not iserr:
        ihalf = maxpt // 2 + 1
        nrx = maxbeg + ihalf
        nxt = number + 1
        if nxt <= nrint:
            for j in range(nxt, nrint + 1):
                jj = nxt + nrint - j
                fpint[jj + 1] = fpint[jj]
                nrdata[jj + 1] = nrdata[jj]
                jk = jj + k
                t[jk + 1] = t[jk]
        nrdata[number] = ihalf - 1
        nrdata[nxt] = maxpt - ihalf
        am = float(maxpt)
        an = float(nrdata[number])
        fpint[number] = fpmax * an / am
        an = float(nrdata[nxt])
        fpint[nxt] = fpmax * an / am
        jk = nxt + k
        t[jk] = x[nrx]
    return n + 1, nrint + 1


def fprati(p1, f1, p2, f2, p3, f3):
    """Rational interpolation for the root of f(p) = 0; returns (p, p1, f1, p3, f3)."""
    if p3 > 0.0:
        h1 = f1 * (f2 - f3)
        h2 = f2 * (f3 - f1)
        h3 = f3 * (f1 - f2)
        p = -(p1 * p2 * h3 + p2 * p3 * h1 + p3 * p1 * h2) / (p1 * h1 + p2 * h2 + p3 * h3)
    else:
        p = (p1 * (f1 - f3) * f2 - p2 * (f2 - f3) * f1) / ((f1 - f2) * f3)
    if f2 < 0.0:
        p3, f3 = p2, f2
    else:
        p1, f1 = p2, f2
    return p, p1, f1, p3, f3


# ---------------------------------------------------------------- fpcurf
def fpcurf(iopt, x, y, w, xb, xe, k, s, st: CurfitState, tol=TOL, maxit=MAXIT) -> CurfitState:
    """Smoothing spline of degree k with sum((w (y - s(x)))^2) <= s (FITPACK fpcurf, iopt 0 or 1).

    x, y, w are 0-based sequences (x non-decreasing); ``st`` is updated in place and returned.
    """
    if iopt not in (0, 1):
        raise ValueError("iopt must be 0 or 1")
    m = len(x)
    x = [0.0] + [float(v) for v in x]
    y = [0.0] + [float(v) for v in y]
    w = [0.0] + [float(v) for v in w]
    nest = st.nest
    t, c, fpint, nrdata = st.t, st.c, st.fpint, st.nrdata
    n, ier, fp = st.n, st.ier, st.fp
    k1, k2 = k + 1, k + 2
    sqrt = math.sqrt
    z = [0.0] * (nest + 1)
    a = [[0.0] * (k1 + 1) for _ in range(nest + 1)]
    b = [[0.0] * (k2 + 1) for _ in range(nest + 1)]
    g = [[0.0] * (k2 + 1) for _ in range(nest + 1)]
    q = [[0.0] * (k1 + 1) for _ in range(m + 1)]
    con1, con9, con4, half = 0.1, 0.9, 0.04, 0.5
    nmin = 2 * k1
    acc = tol * s
    nmax = m + k1
    fp0 = fpold = 0.0
    nplus = 0

    def done():
        st.n, st.ier, st.fp = n, ier, fp
        return st

    def interpolationKnots():                                   # label 10
        mk1 = m - k1
        if mk1 == 0:
            return
        k3 = k // 2
        i, j = k2, k3 + 2
        for _ in range(mk1):
            t[i] = (x[j] + x[j - 1]) * half if k3 * 2 == k else x[j]
            i += 1
            j += 1

    if s > 0.0:                                                  # label 45
        fresh = True
        if iopt != 0 and n != nmin:
            fp0, fpold, nplus = fpint[n], fpint[n - 1], nrdata[n]
            fresh = not fp0 > s
        if fresh:                                                # label 50
            n = nmin
            fpold = 0.0
            nplus = 0
            nrdata[1] = m - 2
    else:
        n = nmax
        if nmax > nest:
            ier = 1
            return done()
        interpolationKnots()

    fpms = 0.0
    while True:                                                  # label 60 (restarted by "go to 10")
        restart = False
        for _ in range(m):
            if n == nmin:
                ier = -2
            nrint = n - nmin + 1
            nk1 = n - k1
            i = n
            for j in range(1, k1 + 1):
                t[j] = xb
                t[i] = xe
                i -= 1
            fp = 0.0
            for i in range(1, nk1 + 1):
                z[i] = 0.0
                row = a[i]
                for j in range(1, k1 + 1):
                    row[j] = 0.0
            l = k1
            for it in range(1, m + 1):
                xi = x[it]
                wi = w[it]
                yi = y[it] * wi
                while not (xi < t[l + 1] or l == nk1):
                    l += 1
                h = fpbspl(t, k, xi, l)
                qit = q[it]
                for i in range(1, k1 + 1):
                    qit[i] = h[i]
                    h[i] = h[i] * wi
                j = l - k1
                for i in range(1, k1 + 1):
                    j += 1
                    piv = h[i]
                    if piv == 0.0:
                        continue
                    aj = a[j]
                    ww = aj[1]
                    store = abs(piv)                      # fpgivs
                    if store >= ww:
                        dd = store * sqrt(1.0 + (ww / piv) ** 2)
                    else:
                        dd = ww * sqrt(1.0 + (piv / ww) ** 2)
                    cos = ww / dd
                    sin = piv / dd
                    aj[1] = dd
                    zj = z[j]                             # fprota(yi, z[j])
                    z[j] = cos * zj + sin * yi
                    yi = cos * yi - sin * zj
                    if i == k1:
                        break
                    i2 = 1
                    for i1 in range(i + 1, k1 + 1):
                        i2 += 1
                        hv = h[i1]
                        av = aj[i2]
                        aj[i2] = cos * av + sin * hv
                        h[i1] = cos * hv - sin * av
                fp = fp + yi * yi
            if ier == -2:
                fp0 = fp
            fpint[n] = fp0
            fpint[n - 1] = fpold
            nrdata[n] = nplus
            fpback(a, z, nk1, k1, c)
            fpms = fp - s
            if abs(fpms) < acc:
                return done()
            if fpms < 0.0:
                break
            if n == nmax:
                ier = -1
                return done()
            if n == nest:
                ier = 1
                return done()
            if ier == 0:                                         # label 140
                npl1 = nplus * 2
                rn = float(nplus)
                if fpold - fp > acc:
                    npl1 = int(rn * fpms / (fpold - fp))
                nplus = min(nplus * 2, max(npl1, nplus // 2, 1))
            else:
                nplus = 1
                ier = 0
            fpold = fp                                           # label 150
            fpart = 0.0
            i = 1
            l = k2
            new = 0
            for it in range(1, m + 1):
                if not (x[it] < t[l] or l > nk1):
                    new = 1
                    l += 1
                term = 0.0
                l0 = l - k2
                qit = q[it]
                for j in range(1, k1 + 1):
                    l0 += 1
                    term = term + c[l0] * qit[j]
                term = (w[it] * (term - y[it])) ** 2
                fpart = fpart + term
                if new == 0:
                    continue
                store = term * half
                fpint[i] = fpart - store
                i += 1
                fpart = store
                new = 0
            fpint[nrint] = fpart
            for _ in range(nplus):
                n, nrint = fpknot(x, t, n, fpint, nrdata, nrint, 1)
                if n == nmax:
                    interpolationKnots()
                    restart = True
                    break
                if n == nest:
                    break
            if restart:
                break
        if not restart:
            break
    # label 250: smoothing spline with fp = s on the current knots
    if ier == -2:
        return done()
    nk1 = n - k1
    fpdisc(t, n, k2, b)
    p1 = 0.0
    f1 = fp0 - s
    p3 = -1.0
    f3 = fpms
    p = 0.0
    for i in range(1, nk1 + 1):
        p = p + a[i][1]
    rn = float(nk1)
    p = rn / p
    ich1 = ich3 = 0
    n8 = n - nmin
    h = [0.0] * (k2 + 1)
    for itr in range(1, maxit + 1):
        pinv = 1.0 / p
        for i in range(1, nk1 + 1):
            c[i] = z[i]
            gi, ai = g[i], a[i]
            gi[k2] = 0.0
            for j in range(1, k1 + 1):
                gi[j] = ai[j]
        for it in range(1, n8 + 1):
            bit = b[it]
            for i in range(1, k2 + 1):
                h[i] = bit[i] * pinv
            yi = 0.0
            for j in range(it, nk1 + 1):
                piv = h[1]
                gj = g[j]
                ww = gj[1]
                store = abs(piv)                          # fpgivs
                if store >= ww:
                    dd = store * sqrt(1.0 + (ww / piv) ** 2)
                else:
                    dd = ww * sqrt(1.0 + (piv / ww) ** 2)
                cos = ww / dd
                sin = piv / dd
                gj[1] = dd
                cj = c[j]                                 # fprota(yi, c[j])
                c[j] = cos * cj + sin * yi
                yi = cos * yi - sin * cj
                if j == nk1:
                    break
                i2 = k1
                if j > n8:
                    i2 = nk1 - j
                for i in range(1, i2 + 1):
                    i1 = i + 1
                    hv = h[i1]
                    gv = gj[i1]
                    gj[i1] = cos * gv + sin * hv
                    h[i] = cos * hv - sin * gv
                h[i2 + 1] = 0.0
        fpback(g, c, nk1, k2, c)
        fp = 0.0
        l = k2
        for it in range(1, m + 1):
            if not (x[it] < t[l] or l > nk1):
                l += 1
            l0 = l - k2
            term = 0.0
            qit = q[it]
            for j in range(1, k1 + 1):
                l0 += 1
                term = term + c[l0] * qit[j]
            fp = fp + (w[it] * (term - y[it])) ** 2
        fpms = fp - s
        if abs(fpms) < acc:
            return done()
        if itr == maxit:
            ier = 3
            return done()
        p2 = p
        f2 = fpms
        if ich3 == 0:
            if not (f2 - f3 > acc):
                p3 = p2
                f3 = f2
                p = p * con4
                if p <= p1:
                    p = p1 * con9 + p2 * con1
                continue
            if f2 < 0.0:
                ich3 = 1
        if ich1 == 0:
            if not (f1 - f2 > acc):
                p1 = p2
                f1 = f2
                p = p / con4
                if p3 < 0.0:
                    continue
                if p >= p3:
                    p = p2 * con1 + p3 * con9
                continue
            if f2 > 0.0:
                ich1 = 1
        if f2 >= f1 or f2 <= f3:
            ier = 2
            return done()
        p, p1, f1, p3, f3 = fprati(p1, f1, p2, f2, p3, f3)
    return done()


# ---------------------------------------------------------------- evaluation
def splev(t: np.ndarray, c: np.ndarray, k: int, x: np.ndarray, ext: int = 0) -> np.ndarray:
    """Spline values at x (FITPACK splev); ext 0 extrapolate, 1 zero, 2 raise, 3 clamp. t, c 0-based."""
    x = np.asarray(x, dtype=float)
    n = t.size
    k1 = k + 1
    nk1 = n - k1
    tb, te = t[k1 - 1], t[nk1]
    arg = x.ravel().copy()
    outside = (arg < tb) | (arg > te)
    if ext == 2 and outside.any():
        raise ValueError("x value out of the spline's domain")
    if ext == 3:
        arg = np.clip(arg, tb, te)
    # 1-based l with t(l) <= arg < t(l+1), clamped to [k1, nk1] (splev's interval search).
    l = np.clip(np.searchsorted(t, arg, side="right"), k1, nk1)
    tt = np.concatenate([[0.0], t])
    h = np.zeros((arg.size, k + 2))
    hh = np.zeros((arg.size, k + 1))
    h[:, 1] = 1.0
    for j in range(1, k + 1):
        hh[:, 1:j + 1] = h[:, 1:j + 1]
        h[:, 1] = 0.0
        for i in range(1, j + 1):
            tli = tt[l + i]
            tlj = tt[l + i - j]
            same = tli == tlj
            with np.errstate(divide="ignore", invalid="ignore"):
                f = hh[:, i] / (tli - tlj)
            h[:, i] = np.where(same, h[:, i], h[:, i] + f * (tli - arg))
            h[:, i + 1] = np.where(same, 0.0, f * (arg - tlj))
    cc = np.concatenate([[0.0], c])
    sp = np.zeros(arg.size)
    for j in range(1, k1 + 1):
        sp = sp + cc[l - k1 + j] * h[:, j]
    if ext == 1:
        sp[outside] = 0.0
    return sp.reshape(x.shape)


def splder(t: np.ndarray, c: np.ndarray, k: int):
    """Knots, coefficients and degree of the derivative spline (FITPACK splder, nu = 1)."""
    if k < 1:
        raise ValueError("a spline of degree 0 has no derivative")
    n = t.size
    nk1 = n - k - 1
    dt = t[k + 1:k + nk1] - t[1:nk1]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(dt > 0, k * (c[1:nk1] - c[:nk1 - 1]) / np.where(dt > 0, dt, 1.0), 0.0)
    cd = np.zeros(n - 2)
    cd[:nk1 - 1] = d
    return t[1:-1].copy(), cd, k - 1
