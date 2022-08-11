import numpy as np
import math
import cv2
from imresize_matlab import imresize


def SAM(GT, Fused):
    """
    get the SAM metric value
    the image shape is [C, H, W]
    :param GT: ground truth image
    :param Fused: fused image
    :return: SAM value
    """
    GT = np.float64(GT)
    Fused = np.float64(Fused)
    H, W = GT.shape[1], GT.shape[2]
    norm_orig = np.linalg.norm(GT, axis=0)
    norm_fusa = np.linalg.norm(Fused, axis=0)
    prod_scal = np.sum(GT * Fused, axis=0)
    prod_norm = np.maximum(norm_fusa * norm_orig, np.finfo(np.float64).eps)
    prod_scal = np.reshape(prod_scal, W * H)
    prod_norm = np.reshape(prod_norm, W * H)
    prod_scal = prod_scal[prod_norm != 0]
    prod_norm = prod_norm[prod_norm != 0]
    angolo = np.mean(np.arccos(np.clip(prod_scal/prod_norm, -1, 1)))
    SAM_index = np.real(angolo) * 180 / math.pi
    return SAM_index


def ERGAS(GT, Fused, ratio=4):
    """
    get the ERGAS metric value
    the image shape is [C, H, W]
    :param GT: ground truth image
    :param Fused: fused image
    :param ratio: the spatial resolution ratio between PAN and MS
    :return:ERGAS value
    """
    GT = np.float64(GT)
    Fused = np.float64(Fused)
    Err = np.array(GT - Fused)
    ERGAS_index = np.sum(np.mean(Err ** 2, axis=(1, 2)) / np.mean(GT, axis=(1, 2)) ** 2)
    ERGAS_index = (100 / ratio) * np.sqrt((1 / Err.shape[0]) * ERGAS_index)
    return ERGAS_index


def SCC(GT, Fused):
    """
    get the SCC metric value
    the image shape is [C, H, W]
    :param GT: ground truth image
    :param Fused: fused image
    :return: SCC value
    """
    GT = np.float64(GT)
    Fused = np.float64(Fused)
    GT = np.transpose(GT, (1, 2, 0))
    Fused = np.transpose(Fused, (1, 2, 0))

    Im_Lap_F_y = cv2.Sobel(Fused, -1, dx=0, dy=1, ksize=3, borderType=cv2.BORDER_REFLECT)
    Im_Lap_F_x = cv2.Sobel(Fused, -1, dx=1, dy=0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Im_Lap_F = np.sqrt(Im_Lap_F_y ** 2 + Im_Lap_F_x ** 2)

    Im_Lap_GT_y = cv2.Sobel(GT, -1, dx=0, dy=1, ksize=3, borderType=cv2.BORDER_REFLECT)
    Im_Lap_GT_x = cv2.Sobel(GT, -1, dx=1, dy=0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Im_Lap_GT = np.sqrt(Im_Lap_GT_y ** 2 + Im_Lap_GT_x ** 2)

    sCC = np.sum(Im_Lap_F * Im_Lap_GT)
    sCC = sCC/np.sqrt(np.sum(Im_Lap_F ** 2))
    sCC = sCC/np.sqrt(np.sum(Im_Lap_GT ** 2))
    return sCC


def norm_blocco(x):
    a = np.mean(x)
    c = np.std(x, ddof=1)
    if c == 0:
        c = np.finfo(np.float64).eps
    y = (x - a) / c + 1
    return y, a, c


def onion_mult(onion1, onion2):
    onion1 = onion1.copy()
    onion2 = onion2.copy()
    N = len(onion1)
    if N > 1:
        L = N // 2
        a = onion1[0:L]
        b = onion1[L:]
        b[1:] = -b[1:]
        c = onion2[0:L]
        d = onion2[L:]
        d[1:] = -d[1:]
        if N == 2:
            ris = np.array([a * c - d * b, a * d + c * b])
            return ris
        else:
            ris1 = onion_mult(a, c)
            ris2 = onion_mult(d, np.append(b[0], -b[1:]))
            ris3 = onion_mult(np.append(a[0], -a[1:]), d)
            ris4 = onion_mult(c, b)
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = np.append(aux1, aux2)
            return ris
    else:
        ris = onion1 * onion2
        return ris


def onion_mult2D(onion1, onion2):
    onion1 = onion1.copy()
    onion2 = onion2.copy()
    N3 = onion1.shape[2]
    if N3 > 1:
        L = N3 // 2
        a = onion1[:, :, 0:L]
        b = onion1[:, :, L:]
        b = np.append(b[:, :, 0, None], -b[:, :, 1:], axis=2)
        c = onion2[:, :, 0:L]
        d = onion2[:, :, L:]
        d = np.append(d[:, :, 0, None], -d[:, :, 1:], axis=2)

        if N3 == 2:
            ris = np.append(a * c - d * b, a * d + c * b, axis=2)
            return ris
        else:
            ris1 = onion_mult2D(a, c)
            ris2 = onion_mult2D(d, np.append(b[:, :, 0, None], -b[:, :, 1:], axis=2))
            ris3 = onion_mult2D(np.append(a[:, :, 0, None], -a[:, :, 1:], axis=2), d)
            ris4 = onion_mult2D(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = np.append(aux1, aux2, axis=2)
            return ris
    else:
        ris = onion1 * onion2
        return ris


def onions_quality(dat1, dat2, size1):
    dat1 = dat1.copy()
    dat2 = dat2.copy()
    dat2 = np.append(dat2[:, :, 0, None], -dat2[:, :, 1:], axis=2)
    C = dat2.shape[2]
    for i in range(C):
        a1, s, t = norm_blocco(dat1[:, :, i])
        dat1[:, :, i] = a1
        if s == 0:
            if i == 0:
                dat2[:, :, i] = dat2[:, :, i] - s + 1
            else:
                dat2[:, :, i] = -(-dat2[:, :, i] - s + 1)
        else:
            if i == 0:
                dat2[:, :, i] = ((dat2[:, :, i] - s) / t) + 1
            else:
                dat2[:, :, i] = -(((-dat2[:, :, i] - s) / t) + 1)

    mod_q1 = np.zeros((size1, size1))
    mod_q2 = np.zeros((size1, size1))

    m1 = np.mean(dat1, axis=(0, 1))
    m2 = np.mean(dat2, axis=(0, 1))
    mod_q1m = np.sum(m1 ** 2)
    mod_q2m = np.sum(m2 ** 2)
    mod_q1 = mod_q1 + np.sum(dat1 ** 2, axis=2)
    mod_q2 = mod_q2 + np.sum(dat2 ** 2, axis=2)

    mod_q1m = np.sqrt(mod_q1m)
    mod_q2m = np.sqrt(mod_q2m)
    mod_q1 = np.sqrt(mod_q1)
    mod_q2 = np.sqrt(mod_q2)

    termine2 = (mod_q1m * mod_q2m)
    termine4 = (mod_q1m ** 2) + (mod_q2m ** 2)
    int1 = (size1 * size1) / ((size1 * size1) - 1) * np.mean(mod_q1 ** 2)
    int2 = (size1 * size1) / ((size1 * size1) - 1) * np.mean(mod_q2 ** 2)
    termine3 = int1 + int2 - (size1 * size1) / ((size1 * size1) - 1) * ((mod_q1m ** 2) + (mod_q2m ** 2))
    mean_bias = 2 * termine2 / termine4
    if termine3 == 0:
        q = np.zeros((1, 1, C))
        q[:, :, C-1] = mean_bias
    else:
        cbm = 2 / termine3
        qu = onion_mult2D(dat1, dat2)
        qm = onion_mult(m1, m2)
        qv = (size1 * size1) / ((size1 * size1) - 1) * np.mean(qu, axis=(0, 1))
        q = qv - (size1 * size1) / ((size1 * size1) - 1) * qm
        q = q * mean_bias * cbm
    return q


def Q2n(GT, Fused, Q_blocks_size=32, Q_shift=32):
    """
    get the Q4 or Q8 metric value
    the image shape is [C, H, W]
    :param GT: ground truth image
    :param Fused: fused image
    :param Q_blocks_size: Block size of the Q-index locally applied
    :param Q_shift: Block shift of the Q-index locally applied
    :return: Q4 or Q8 value
    """
    GT = np.float64(GT)
    Fused = np.float64(Fused)
    C, H, W = GT.shape
    GT = np.transpose(GT, (1, 2, 0))
    Fused = np.transpose(Fused, (1, 2, 0))
    stepx = math.ceil(H / Q_shift)
    stepy = math.ceil(W / Q_shift)
    est1 = (stepx - 1) * Q_shift + Q_blocks_size - H
    est2 = (stepy - 1) * Q_shift + Q_blocks_size - W
    if est1 != 0 or est2 != 0:
        refref = []
        fusfus = []

        for i in range(C):
            a1 = GT[:, :, i]
            ia1 = np.zeros((H + est1, W + est2))
            ia1[0: H, 0: W] = a1
            ia1[:, W:W + est2] = ia1[:, W - 1:W - est2 - 1:-1]
            ia1[H:H + est1, :] = ia1[H - 1:H - est1 - 1:-1, :]
            np.append(refref, ia1, axis=2)
        GT = refref
        for i in range(C):
            a2 = Fused[:, :, i]
            ia2 = np.zeros(H + est1, W + est2)
            ia2[0: H, 0: W] = a2
            ia2[:, W:W + est2] = ia2[:, W - 1:W - est2 - 1:-1]
            ia2[H:H + est1, :] = ia2[H - 1:H - est1 - 1:-1, :]
            np.append(fusfus, ia2, axis=2)
        Fused = fusfus

    valori = np.zeros((stepx, stepy, C))

    for j in range(stepx):
        for i in range(stepy):
            valori[j, i, :] = onions_quality(GT[j * Q_shift:j * Q_shift + Q_blocks_size, i * Q_shift:i * Q_shift + Q_blocks_size, :],
                                             Fused[j * Q_shift:j * Q_shift + Q_blocks_size, i * Q_shift:i * Q_shift + Q_blocks_size, :],
                                             Q_blocks_size)
    Q2n_index_map = np.sqrt(np.sum((valori ** 2), axis=2))
    Q2n_index = np.mean(Q2n_index_map)
    return Q2n_index


def uqi(x, y):
    x = x.reshape((1, -1))
    y = y.reshape((1, -1))
    mx = np.mean(x)
    my = np.mean(y)
    C = np.cov(x, y)
    Q = 4 * C[0, 1] * mx * my / (C[0, 0] + C[1, 1]) / (mx ** 2 + my ** 2)
    return Q


def D_s(Fused, MS, PAN, ratio=4, S=32):
    """
    get the D_s metric value
    the image shape is [C, H, W]
    :param Fused: fused image
    :param MS: low resolution multispectral image
    :param PAN: panchromatic image
    :param ratio: the spatial resolution ratio between PAN and MS
    :param S: block size
    :return: D_s value
    """
    C, H, W = Fused.shape
    PAN = np.float64(PAN)
    Fused = np.float64(Fused)
    Fused = np.transpose(Fused, (1, 2, 0))
    MS = np.float64(MS)
    MS = np.transpose(MS, (1, 2, 0))
    PANlr = imresize(PAN, 1 / ratio)
    D_s_index = 0
    for i in range(C):
        band1 = Fused[:, :, i]
        band2 = PAN
        Qmap_high = np.zeros((H//S, W//S))
        for step1 in range(H//S):
            for step2 in range(W//S):
                Qmap_high[step1, step2] = uqi(band1[step1 * S:(step1 + 1) * S, step2 * S:(step2 + 1) * S],
                                              band2[step1 * S:(step1 + 1) * S, step2 * S:(step2 + 1) * S])
        Qhigh = np.mean(Qmap_high)

        band1 = MS[:, :, i]
        band2 = PANlr
        Qmap_low = np.zeros((H // S, W // S))
        for step1 in range(H // S):
            for step2 in range(W // S):
                Qmap_low[step1, step2] = uqi(band1[step1 * S // ratio:(step1 + 1) * S // ratio, step2 * S // ratio:(step2 + 1) * S // ratio],
                                             band2[step1 * S // ratio:(step1 + 1) * S // ratio, step2 * S // ratio:(step2 + 1) * S // ratio])
        Qlow = np.mean(Qmap_low)
        D_s_index = D_s_index + abs(Qhigh - Qlow)
    D_s_index = D_s_index / C
    return D_s_index


def D_lambda(Fused, MS, ratio=4, S=32):
    """
    get the D_lambda metric value
    the image shape is [C, H, W]
    :param Fused: fused image
    :param MS: low resolution multispectral image
    :param ratio: the spatial resolution ratio between PAN and MS
    :param S: block size
    :return: D_lambda value
    """
    C, H, W = Fused.shape
    Fused = np.float64(Fused)
    Fused = np.transpose(Fused, (1, 2, 0))
    MS = np.float64(MS)
    MS = np.transpose(MS, (1, 2, 0))

    D_lambda_index = 0
    for i in range(C - 1):
        for j in range(i + 1, C):
            band1 = MS[:, :, i]
            band2 = MS[:, :, j]
            Qmap_exp = np.zeros((H // S, W // S))
            for step1 in range(H // S):
                for step2 in range(W // S):
                    Qmap_exp[step1, step2] = uqi(band1[step1 * S // ratio:(step1 + 1) * S // ratio, step2 * S // ratio:(step2 + 1) * S // ratio],
                                                 band2[step1 * S // ratio:(step1 + 1) * S // ratio, step2 * S // ratio:(step2 + 1) * S // ratio])
            Q_exp = np.mean(Qmap_exp)
            band1 = Fused[:, :, i]
            band2 = Fused[:, :, j]
            Qmap_fused = np.zeros((H // S, W // S))
            for step1 in range(H // S):
                for step2 in range(W // S):
                    Qmap_fused[step1, step2] = uqi(band1[step1 * S:(step1 + 1) * S, step2 * S:(step2 + 1) * S],
                                                   band2[step1 * S:(step1 + 1) * S, step2 * S:(step2 + 1) * S])
            Q_fused = np.mean(Qmap_fused)
            D_lambda_index = D_lambda_index + abs(Q_fused - Q_exp)
    D_lambda_index = D_lambda_index / (C * (C - 1) / 2)
    return D_lambda_index


def QNR(Fused, MS, PAN, ratio=4, S=32, alpha=1, beta=1):
    """
    get the QNR metric value
    the image shape is [C, H, W]
    :param Fused: fused image
    :param MS: low resolution multispectral image
    :param PAN: panchromatic image
    :param ratio: the spatial resolution ratio between PAN and MS
    :param S: block size
    :param alpha: default 1
    :param beta: default 1
    :return: QNR value, D_s value, D_lambda value
    """
    D_s_index = D_s(Fused, MS, PAN, ratio, S)
    D_lambda_index = D_lambda(Fused, MS, ratio, S)
    QNR_index = (1 - D_lambda_index) ** alpha * (1 - D_s_index) ** beta
    return QNR_index, D_s_index, D_lambda_index
