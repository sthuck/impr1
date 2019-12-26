import numpy as np
from numpy.matlib import repmat

def imGradSobel(img):
    r, c = img.shape
    imgPad = np.zeros((r + 2, c + 2), dtype=(np.float32))
    imgPad[1:r + 1, 1:c + 1] = img
    imgPad[0, 1:c + 1] = img[0, :]
    imgPad[-1, 1:c + 1] = img[-1, :]
    imgPad[1:r + 1, 0] = img[:, 0]
    imgPad[1:r + 1, -1] = img[:, -1]
    imgPad[(0, 0)] = img[(0, 0)]
    imgPad[(r + 1, c + 1)] = img[(r - 1, c - 1)]
    imgPad[(0, c + 1)] = img[(0, c - 1)]
    imgPad[(r + 1, 0)] = img[(0, c - 1)]
    Gx = imgPad[:, 2:] - imgPad[:, :-2]
    Gx = Gx[:-2, :] + Gx[2:, :] + 2 * Gx[1:-1, :]
    Gy = imgPad[2:, :] - imgPad[:-2, :]
    Gy = Gy[:, :-2] + Gy[:, 2:] + 2 * Gy[:, 1:-1]
    Gmag = np.sqrt(Gx ** 2 + Gy ** 2)
    return (
     Gx, Gy, Gmag)


def interp2Bilinear(img, coords):
    r, c = img.shape
    x = coords[:, 0]
    y = coords[:, 1]
    indx = np.where(np.logical_or(x < 0, x > c - 1))
    indy = np.where(np.logical_or(y < 0, y > r - 1))
    x0 = np.int32(np.maximum(x, np.zeros(x.shape[0])))
    x0 = np.int32(np.minimum(x, (c - 1) * np.ones(x.shape[0])))
    y0 = np.int32(np.maximum(y, np.zeros(y.shape[0])))
    y0 = np.int32(np.minimum(y, (r - 1) * np.ones(y.shape[0])))
    x1 = np.int32(np.maximum(x + 1, np.zeros(x.shape[0])))
    x1 = np.int32(np.minimum(x + 1, (c - 1) * np.ones(x.shape[0])))
    y1 = np.int32(np.maximum(y + 1, np.zeros(y.shape[0])))
    y1 = np.int32(np.minimum(y + 1, (r - 1) * np.ones(y.shape[0])))
    A00 = img[(y0, x0)]
    A01 = img[(y0, x1)]
    A10 = img[(y1, x0)]
    A11 = img[(y1, x1)]
    A00[indx] = 0
    A01[indx] = 0
    A10[indx] = 0
    A11[indx] = 0
    A00[indy] = 0
    A01[indy] = 0
    A10[indy] = 0
    A11[indy] = 0
    wx = x - np.float32(x0)
    wy = y - np.float32(y0)
    Y0 = A00 * (1 - wx) + A01 * wx
    Y1 = A10 * (1 - wx) + A11 * wx
    val = Y0 * (1 - wy) + Y1 * wy
    return val


def getAffineTransformation(pts1, pts2):
    numOfRows = pts1.shape[0] * 2
    row_even_indices = np.arange(0, numOfRows, 2)
    row_odd_indices = np.arange(1, numOfRows, 2)
    A = np.zeros((numOfRows, 6), np.float32)
    A[(row_even_indices, 0)] = pts1[:, 0]
    A[(row_even_indices, 1)] = pts1[:, 1]
    A[(row_even_indices, 4)] = 1
    A[(row_odd_indices, 2)] = pts1[:, 0]
    A[(row_odd_indices, 3)] = pts1[:, 1]
    A[(row_odd_indices, 5)] = 1
    b = pts2.flatten()
    T = np.linalg.lstsq(A, b, rcond=None)[0]
    affineT = np.array([[T[0], T[1], T[4]], [T[2], T[3], T[5]], [0, 0, 1]])
    return affineT


def applyAffineTransToImage(src, affineT):
    r, c = src.shape
    X, Y = np.meshgrid(np.linspace(0, c - 1, c), np.linspace(0, r - 1, r))
    invT = np.linalg.inv(affineT)
    coords = np.column_stack((X.flatten(),
     Y.flatten(), np.ones(X.flatten().shape))).T
    sourceCoords = np.dot(invT, coords)
    imgT = interp2Bilinear(src, sourceCoords[0:2].T)
    imgT = np.reshape(imgT, (r, c))
    return imgT


def deformationBetweenLines(img, Q1, P1, Q2, P2, a=1, b=1):
    epsilon = 0.01
    r = img.shape[0]
    c = img.shape[1]
    u1 = (Q1 - P1) / np.linalg.norm(Q1 - P1)
    v1 = np.asarray([u1[1], -u1[0]], np.float32)
    u2 = (Q2 - P2) / np.linalg.norm(Q2 - P2)
    v2 = np.asarray([u2[1], -u2[0]], np.float32)
    X, Y = np.meshgrid(np.linspace(0, c - 1, c), np.linspace(0, r - 1, r))
    R2 = np.vstack((X.flatten(), Y.flatten())).T
    alpha = np.dot(R2 - repmat(P2, R2.shape[0], 1), u2) / np.linalg.norm(Q2 - P2)
    beta = np.dot(R2 - repmat(P2, R2.shape[0], 1), v2)
    normP1Q1 = np.linalg.norm(Q1 - P1)
    R1 = P1 + np.multiply((alpha * normP1Q1)[:, np.newaxis], repmat(u1, alpha.shape[0], 1)) + np.multiply(beta[:, np.newaxis], repmat(v1, beta.shape[0], 1))
    weight = (normP1Q1 ** a / (epsilon + np.abs(beta))) ** b
    return (
     R1, weight)


def multipleSegmentDefromation(img, Qs, Ps, Qt, Pt, a, b):
    numOfSegments = Qs.shape[0]
    Rsrc = np.zeros((img.shape[0] * img.shape[1], 2), dtype=(np.float32))
    weightSum = 0
    for segIdx in np.arange(0, numOfSegments):
        Q11 = Qs[segIdx, :]
        P11 = Ps[segIdx, :]
        Q12 = Qt[segIdx, :]
        P12 = Pt[segIdx, :]
        R1, weight1 = deformationBetweenLines(img, Q11, P11, Q12, P12, a, b)
        Rsrc = Rsrc + R1 * weight1[:, np.newaxis]
        weightSum = weightSum + weight1

    Rsrc = np.divide(Rsrc, weightSum[:, np.newaxis])
    imgT = interp2Bilinear(img, Rsrc)
    imgT = np.reshape(imgT, (img.shape[0], img.shape[1]))
    return imgT
# okay decompiling ex2.pyc
