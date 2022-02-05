import numpy as np

def arnoldi(A, b, m):
    n = A.shape[0]
    eps = 1e-12

    V = np.zeros((n,m))
    H = np.zeros((m,m))

    V[:,0] = b / np.linalg.norm(b, 2)

    for j in range(1, m+1):
        w = np.dot(A, V[:, j-1])

        for i in range(1,j+1):
            H[i-1, j-1] = np.dot(V[:, i-1].conj(), w)
            w = w - H[i-1, j-1]*V[:, i-1]
        
        if (j != m):
            H[j, j-1] = np.linalg.norm(w,2)
            
            if H[j, j-1] > eps:
                V[:, j] = w / H[j, j-1]
            else:
                return V, H

    return V, H