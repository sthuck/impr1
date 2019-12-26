import numpy as np
from numpy.linalg import norm
from numpy.matlib import repmat


def getAffineTransformation(pts1: np.ndarray, pts2: np.ndarray):
    b = pts2.reshape(6)
    M = np.array([
        [pts1[0][0], pts1[0][1], 0, 0, 1, 0],
        [0, 0, pts1[0][0], pts1[0][1], 0, 1],
        [pts1[1][0], pts1[1][1], 0, 0, 1, 0],
        [0, 0, pts1[1][0], pts1[1][1], 0, 1],
        [pts1[2][0], pts1[2][1], 0, 0, 1, 0],
        [0, 0, pts1[2][0], pts1[2][1], 0, 1],
    ])
    p = np.linalg.lstsq(M, b)[0]
    transformMatrix = np.array([
        [p[0], p[1], p[4]],
        [p[2], p[3], p[5]],
        [0, 0, 1],
    ])
    return transformMatrix


def applyAffineTransToImage(img: np.ndarray, T: np.ndarray):
    r, c = img.shape
    X, Y = np.meshgrid(np.linspace(0, c - 1, c),
                       np.linspace(0, r - 1, r))
    # calculate source coordinates
    invT = np.linalg.inv(T)
    coords = np.column_stack((X.flatten(),
                              Y.flatten(),
                              np.ones(Y.shape).flatten())).T
    source_coor = np.dot(invT, coords)
    source_coor_no_homogeneous = source_coor[0:2].T

    result = do_bilinear(source_coor_no_homogeneous, img)
    result = result.reshape((r, c))
    return result


# mapping_to_orig is of shape (r*c)x2, first row X, second Y
def do_bilinear(mapping_to_orig: np.ndarray, orig: np.ndarray):
    r, c = orig.shape
    X = mapping_to_orig[:, 0]
    Y = mapping_to_orig[:, 1]
    invalid_x_lower_0 = np.where(X < 0)
    invalid_x_larger_max = np.where(X > c - 1)
    invalid_y_lower_0 = np.where(Y < 0)
    invalid_y_larger_max = np.where(Y > r - 1)
    X[invalid_x_lower_0] = 0
    X[invalid_x_larger_max] = c - 1
    Y[invalid_y_lower_0] = 0
    Y[invalid_y_larger_max] = r - 1

    # y2 Q12-----Q22
    #    |       |
    #    y   p   |
    #    |       |
    #    |       |
    #   Q11--x---Q21
    # x1,y1        x2
    #
    # f(x, y1) ~= (x2-x)*f(Q11) + (x-x1)*f(Q21)
    # f(x, y2) ~= (x2-x)*f(Q12) + (x-x1)*f(Q22)
    # f(x, y) ~= (y2-y)*f(x,y1) + (y-y1)*f(x, y2)
    #
    # u = S - SW (x axis)
    # v = V - S (y axis)
    # S(u) = SE * u + SW * (1 -u)
    # N(u) = NE * u + NW * (1-u)
    # V(u, v) = N(u) * v + S(u) * (1-v)

    X1 = np.floor(X).astype(np.uint8)
    X2 = np.ceil(X).astype(np.uint8)
    Y1 = np.floor(Y).astype(np.uint8)
    Y2 = np.ceil(Y).astype(np.uint8)

    SW = orig[(Y1, X1)]
    NW = orig[(Y2, X1)]
    SE = orig[(Y1, X2)]
    NE = orig[(Y2, X2)]

    u = X - X1
    v = Y - Y1
    S = SE * u + SW * (1 - u)
    N = NE * u + NW * (1 - u)
    V = N * v + S * (1-v)

    invalids = np.concatenate(
        (invalid_x_larger_max[0], invalid_x_lower_0[0], invalid_y_larger_max[0], invalid_y_lower_0[0]))
    V[invalids] = 0

    return V


def get_perpendicular(point):
    return np.array([point[1], point[0] * -1])


def element_wise_mult_with_extend(columnV, rowV):
    return np.vstack((columnV, columnV)).T * np.tile(rowV, (columnV.shape[0], 1))


def compute_mapping(start, end, start_T, end_T, r, c):
    # Q - Start point, P = end point
    # alpha = ((R' - P') * u') / ||Q' - P'||
    # beta = (R' -P') * v'
    # u = Q - P / ||Q - P||
    # v = (u_y, -u_x)
    # u' = Q' -P' / ||Q-P||
    # v' ...
    # R = P + alpah * ||Q - P|| * u + beta * v
    # W_i(a, p, b) = ((|Q - P|^p)/a + beta) ^ b
    X, Y = np.meshgrid(np.linspace(0, c - 1, c),
                       np.linspace(0, r - 1, r))
    base_mapping = np.array([X.flatten(), Y.flatten()]).T

    u = (start - end) / norm(start - end)
    u_T = (start_T - end_T) / norm(start_T - end_T)
    v = get_perpendicular(u)
    v_T = get_perpendicular(u_T)
    alpha = np.matmul((base_mapping - end_T), u_T) / norm(start_T - end_T)
    beta = np.matmul((base_mapping - end_T), v_T)
    mapping = end + element_wise_mult_with_extend(alpha*norm(start - end), u) + element_wise_mult_with_extend(beta, v)
    return mapping, beta


def compute_weight(start, end, p, b, beta, a = 0.001):
    # W_i(a, p, b) = ((||Q - P||^p)/a + beta) ^ b
    return np.power((np.power(norm(start - end), p)/(a + beta)), b)


def multipleSegmentDefromation(img: np.ndarray, Qs, Ps, Qt, Pt, p, b):
    r, c = img.shape
    how_many_lines = Qs.shape[0]
    weights = []
    mappings = []

    for i in range(how_many_lines):
        start, end, start_T, end_T = (Qs[i], Ps[i], Qt[i], Pt[i])
        mapping, beta = compute_mapping(start, end, start_T, end_T, r, c)
        weight = compute_weight(start, end, p, b, beta)
        mappings.append(mapping)
        weight_as_column_vector = weight.reshape(-1, 1)
        weights.append(np.hstack((weight_as_column_vector, weight_as_column_vector)))

    final_mapping = np.sum(np.array(weights) * np.array(mappings), axis=0)/np.sum(np.array(weights), axis=0)
    result = do_bilinear(final_mapping, img)
    result = np.reshape(result, (r, c))
    return result
