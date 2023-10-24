import matplotlib.pyplot as plt
from math import *

plt.rcParams.update({'figure.max_open_warning': 0})

def draw_latex(latex):
    fig, axes = plt.subplots()
    axes.set_axis_off()
    plt.text(0.5, 0.5, latex, fontsize=16, horizontalalignment='center', verticalalignment='center', color='black')
    fig.set_size_inches(0.1,0.1)
    # plt.show()
    
def complement(raw_str, **subs):
    parts = raw_str.split('`')
    for i in range(1, len(parts), 2):
        if parts[i] in subs:
            parts[i] = str(subs[parts[i]])
    return "".join(parts)

def fts(value, d_count = 3):
    return str(round(value, d_count))

def lfts(value):
    return "$" + fts(value) + "$"

def afts(array):
    result = []
    for el in array:
        result += [lfts(el)]
    return result

def pfts(value, p, d_count=3):
    if p == 0:
        return fts(value, d_count)
    value *= 10**(-p)
    return fts(value, d_count) + r"\cdot 10^{" + str(p) + "}"

def brand(array, start=0):
    return range(start, len(array))

sqr = lambda x: x*x
cpf = lambda x, y: x*y

def mnk(xs, ys, showTable=False):
    count = len(xs)
    mx = sum(xs)/count
    x2 = list(map(sqr, xs))
    mx2 = sum(x2)/count
    my = sum(ys)/count
    y2 = list(map(sqr, ys))
    my2 = sum(y2)/count
    xy = list(map(cpf, xs, ys))
    mxy = sum(xy)/count
    k = (mxy - mx*my)/(mx2 - mx*mx)
    b = my - k*mx
    sgm_k = 1/sqrt(count)*sqrt((my2-my*my)/(mx2-mx*mx)-k*k)
    sgm_b = sgm_k*sqrt(mx2-mx*mx)
    return k, b, sgm_k, sgm_b

def MNK(xs, ys):
    return __MNK__(xs, ys, '', '', '', '', ' ', ' ')


def __MNK__(xs, ys, x_tag, y_tag, x_dim, y_dim, k_dim, b_dim, p = 0, ap = 0):
    rows = []
    for i in brand(xs):
        rows += [[xs[i], ys[i], xs[i]**2, ys[i]**2, xs[i]*ys[i]]]
    aves = [0, 0, 0, 0, 0]
    for i in brand(xs):
        for j in range(5):
            aves[j] += rows[i][j]
        rows[i] = afts(rows[i])
    for j in range(5):
        aves[j] /= len(xs)
    fig, axes = plt.subplots()
    axes.set_axis_off()
    fig.set_size_inches(7,0.1)
    tab = plt.table(cellLoc="center", 
                    colLabels=["$" + x_tag + ((", " + x_dim) if x_dim is not None else "") + "$",
                    "$" + y_tag + ((", " + y_dim) if y_dim is not None else "") + "$",
                    "$" + x_tag + "^2$", "$" + y_tag + "^2$", "${" + x_tag + r"}\cdot{" + y_tag + "}$"],
                    rowLabels=[str(x+1) for x in brand(xs)] + ["<>"],
                    cellText=rows + [afts(aves)])
    tab.set_fontsize(18)
    tab.scale(1, 4)
    # plt.show()
    k_formula = r"\frac{{`xy`} - {`x`} \cdot {`y`}}{`x2` - `x_2`}"
    k = (aves[4] - aves[0]*aves[1]) / (aves[2] - aves[0]**2)
    draw_latex(complement("$x = `x_tag` \; `x_dim`$", x_tag=x_tag, x_dim=("[" + x_dim + "]" if x_dim is not None else "")))
    draw_latex(complement("$y = `y_tag` \; `y_dim`$", y_tag=y_tag, y_dim=("[" + y_dim + "]" if y_dim is not None else "")))
    if k_dim is not None:
        draw_latex("$k = " + complement(k_formula, xy="<xy>", x="<x>", y="<y>", x2="<x^2>", x_2="<x>^2") + 
                "=" + complement(k_formula, xy=fts(aves[4]), x=fts(aves[0]), y=fts(aves[1]), x2=fts(aves[2]), x_2=fts(aves[0]*aves[0])) + 
                " = " + pfts(k*10**ap, p) + r"\;" + k_dim + "$")
    b_formula = r"`y` - `k` \cdot `x`"
    b = aves[1] - k*aves[0]
    if b_dim is not None:
        draw_latex("$b = " + complement(b_formula, x="<x>", y="<y>") + 
                "=" + complement(b_formula, x=fts(aves[0]), y=fts(aves[1]), k=fts(k)) + 
                " = " + pfts(b*10**ap, p) + r"\;" + b_dim + "$")
    sgm_k_formula = r"\frac{1}{\sqrt{`n`}} \sqrt{ \frac{`y2` - `y_2`}{`x2` - `x_2`} - `k2`}"
    sgm_k = 1/sqrt(len(xs))*sqrt( (aves[3] - aves[1]**2)/(aves[2] - aves[0]**2) - k*k )
    if k_dim is not None:
        draw_latex("$\sigma_k = " + 
                complement(sgm_k_formula, x2="<x^2>", x_2="<x>^2", y2="<y^2>", y_2="<y>^2", k2="k^2") + "=" +
                complement(sgm_k_formula, n=len(xs), x2=fts(aves[2]), x_2=fts(aves[0]*aves[0]), y2=fts(aves[3]), y_2=fts(aves[1]*aves[1]), k2=fts(k*k)) + 
                "=" + pfts(sgm_k*10**ap, p) + "\;" + k_dim + "$")
    sgm_b_formula = r"`s` \cdot \sqrt{`x2` - `x_2`}"
    sgm_b = sgm_k * sqrt(aves[2] - aves[0]**2)
    if b_dim is not None:
        draw_latex("$\sigma_b = " + 
                complement(sgm_b_formula, x2="<x^2>", x_2="<x>^2", s="\sigma_k") + "=" +
                complement(sgm_b_formula, x2=fts(aves[2]), x_2=fts(aves[0]*aves[0]), s=fts(sgm_k)) + 
                "=" + pfts(sgm_b*10**ap, p) + "\;" + b_dim + "$")
    return k, b, sgm_k, sgm_b