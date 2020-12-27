import numpy as np
import networkx as nx
from scipy.sparse import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import random
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse import eye as sparse_eye
from matplotlib import gridspec
import matplotlib
from sklearn.datasets import load_digits
import matplotlib.colors as mcolors
import os
import itertools


"""
=============networkx graphs=============
======product graph and random walk======
"""

def matrices(G):
    """
    return matrices using L_g = D_g - M_g
    self note: we can calc the D_g by sum each row of M_g and put the values in the cross
    :param G: nx graph
    :return: M_g is adjacency_matrix
            D_g is degree matrix
            L_g is laplacian matrix
    """
    M_g = nx.adjacency_matrix(G)
    L_g = nx.laplacian_matrix(G)
    D_g = M_g + L_g
    return (M_g,D_g,L_g)

def EVD(G,normed=False):
    """
    calculate the g Laplacian matrix eigen values and right eigen vectors
    :param G: gpath
    :return:sorted lists of eigen values and vectors
    """
    if normed:
        w,v = np.linalg.eig(normalize(matrices(G)[2].A))
    else:
        w, v = np.linalg.eig(matrices(G)[2].A)
    s_w= sorted(w)
    s_v = [v[np.where(w==i)] for i in s_w]
    # vals,vecs
    return s_w,s_v

def calc_analiticals(n,kind="connected"):
    if kind=="ring":
        evals = [2*(1-np.cos(2*np.pi * k / n)) for k in range(n)]
        x=[]
        y=[]
        for k in range(1+ n//2):
            if not n==0 and not(n%2 ==0 and k== n//2):
                y_i = [np.sin(2 * np.pi * k * a) for a in range(n)]
            x_i = [np.cos(2 * np.pi * k * a) for a in range(n)]
            x.append(x_i)
            y.append(y_i)
        evecs = y+x
    else:
        evals =[2*(1-np.cos(np.pi * k / n)) for k in range(n)]
        evecs =  [[np.cos(((2*a-1)*np.pi * k)/(2*n)) for a in range(n)] for k in range(n)]
    return evals,evecs

def plot_3D(G,title = "G in 3D"):
    # 3d spring layout
    pos = nx.spring_layout(G, dim=3, seed=779)
    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Plot the nodes - alpha is scaled by "depth" automatically

    ax.scatter(*node_xyz.T, s=100, ec="w")#,c=node_color)
    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    def _format_axes(ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    _format_axes(ax)
    fig.tight_layout()
    plt.title(title)
    plt.savefig(title+'.png')
    plt.show()

def plot_3D_evec_colors(G,title="graph",indexes = [0]):
    # 3d spring layout
    pos = nx.spring_layout(G, dim=3, seed=779)
    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    # Create the 3D figure
    fig = plt.figure()
    cols = 1 + len(indexes) // 3
    rows = 1 + len(indexes) // cols
    _, evecs = EVD(G)
    for i in range(len(indexes)):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        ax.scatter(*node_xyz.T, s=100, ec="w", c=evecs[indexes[i]][0])
        print(evecs[indexes[i]][0])
        # Plot the edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")

        def _format_axes(ax):
            """Visualization options for the 3D axes."""
            # Turn gridlines off
            ax.grid(False)
            # Suppress tick labels
            for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                dim.set_ticks([])
            # Set axes labels
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

        _format_axes(ax)
        fig.tight_layout()
        ax.set_title("colored by {} eigenvec".format(str(indexes[i]+1)))
    plt.subplots_adjust(wspace=0.5)
    plt.savefig(title + '.png')
    plt.show()

def noisy_permute_G(n_x, n_y):
    R = 10
    r = 4
    x = np.linspace(0, 1, num=n_x, endpoint=False)
    y = np.linspace(0, 1, num=n_y, endpoint=False)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    s_x = (R + r * np.cos(2 * np.pi * y)) * np.cos(2 * np.pi * x)
    s_y = (R + r * np.cos(2 * np.pi * y)) * np.sin(2 * np.pi * x)
    s_z = r * np.sin(2 * np.pi * y)
    vertexes = np.column_stack((s_x, s_y, s_z))
    n = vertexes.shape[0]
    permut = np.random.permutation(n)
    new_vertexes =vertexes[permut, :]
    noise = np.random.normal(scale=np.sqrt(0.01), size=(n,3))
    return new_vertexes + noise

def noisy_adj(nodes,n):
    #n = nodes.shape[0]
    M_g = np.zeros((n,n))
    for i in range(n):
        M_g[i, :] = np.linalg.norm(nodes[i] - nodes, axis=1)
    treshold = sorted(M_g[0])[8] - 1e-4 #all nodes are reletevly the same
    M_g[M_g > treshold] = 0 #masking high value
    return M_g

def plot_2D(G,title="2d graph"):
    plt.figure()
    ax = plt.gca()
    ax.set_title(title)
    nx.draw(G)#,node_size = 25)
    plt.savefig(title + '.png')
    plt.show()

def eigenvals_plot(vals,title = "eigen values"):
    x = np.linspace(0, 1, len(vals))
    plt.scatter(x, vals)
    plt.title(title)
    plt.savefig(title + '.png')
    plt.show()

def graphs():
    #======================= section 1 =======================#
    n = 201
    Pn = nx.path_graph(n)
    Rn = nx.cycle_graph(n)
    ##a
    M_p, D_p, L_p = matrices(Pn)
    M_r, D_r, L_r = matrices(Rn)
    ##b
    Pn_sorted_evals, Pn_sorted_evec = EVD(Pn)
    Rn_sorted_evals, Rn_sorted_evec = EVD(Rn)
    ##c-i
    plot_2D(Pn, "Path graph")
    plot_2D(Rn, "Ring graph")

    ##c-ii
    eigenvals_plot(Pn_sorted_evals, "Pn sorted eigenvalues")
    eigenvals_plot(Rn_sorted_evals, "Rn sorted eigenvalues")

    # c-iii
    plot_3D_evec_colors(Pn, "Pn colored by eigenvectors", [0, 1, 4, 9])
    plot_3D_evec_colors(Rn, "Rn colored by eigenvectors", [0, 1, 4, 9])

    # d
    Pn_analitical_evals, Pn_analitical_evec = calc_analiticals(n, "connected")
    Rn_analitical_evals, Rn_analitical_evec = calc_analiticals(n, "ring")
    print("connected graph eigenvalues: {}\nanalitic eigenvalues: {}".format(Pn_sorted_evals,Pn_analitical_evals ))
    print("ring graph eigenvalues: {}\nanalitic eigenvalues: {}".format(Rn_sorted_evals, Rn_analitical_evals))
    # TODO: make sure vecs allined
    # print("ring graph second eigen vec: {}\nanalitic eigenvalues: {}".format(Rn_sorted_evec[1], Rn_analitical_evec[1]))
    # ==================== end of section 1 ====================#

    # ======================= section 2 =======================#
    n_x, n_y = 71, 31
    #n_x, n_y = 9,9

    Rnx = nx.cycle_graph(n_x)
    Rny = nx.cycle_graph(n_y)
    G = nx.cartesian_product(Rnx, Rny)
    M_g, D_g, L_g = matrices(G)
    G_sorted_evals, G_sorted_evec = EVD(G)
    plot_3D(G)
    eigenvals_plot(G_sorted_evals, "G_sorted_evals")
    plot_3D_evec_colors(G, "G colored by eigenvectors", indexes=[0, 1, 4, 6, 9])
    # e - analitic expression
    Rnx_evals, Rnx_evecs = EVD(Rnx)
    Rny_evals, Rny_evecs = EVD(Rny)
    G_analitical_evals = [x_eval + y_eval for x_eval in Rnx_evals for y_eval in Rny_evals]
    print("G eigen values directly computed: ",G_sorted_evals)
    print("G eigen values analitical computed: ",G_analitical_evals)
    G_analitical_evec = [np.concatenate((x_vec, y_vec)) for x_vec in Rnx_evecs[0] for y_vec in Rny_evecs[0]]
    print("G eigenvectors directly computed: ",G_sorted_evec)
    print("G eigenvectors analitical computed ",G_analitical_evec)
    # ==================== end of section 2 ====================#

    # ======================= section 3 =======================#
    noisy_G = noisy_permute_G(n_x, n_y)
    M_g = noisy_adj(noisy_G, n_x * n_y)
    G2 = nx.from_numpy_matrix(M_g)
    G2.edges(data=True)
    plot_3D(G2, "Noisy G in 3D")
    G2_sorted_evals, G2_sorted_evec = EVD(G2)
    eigenvals_plot(G2_sorted_evals, "Noisy permuted G_sorted_evals")
    plot_3D_evec_colors(G2, "Noisy G colored by eigenvectors", indexes=[0, 1, 4, 6, 9])
    # ==================== end of section 3 ====================#

"""
=============Lazy random walk=============
=============and power iteration =========
"""

def gen_graph(n=30,type="Dubmle"):
    a = nx.complete_graph(n)
    b = nx.complete_graph(n)
    if type=="Dubmle":
        Dubmle = nx.union(a, b, rename=('a-', 'b-'))
        Dubmle.add_edge('a-0', 'b-0')
        return Dubmle
    elif type=="Bolas":
        Bolas = nx.union(a, b, rename=('a-', 'b-'))
        for i in range(n - 1):
            Bolas.add_node('c-' + str(i))
            if i == 0:
                Bolas.add_edge('a-0', 'c-0')
            else:
                Bolas.add_edge('c-' + str(i - 1), 'c-' + str(i))
            if i == n - 2:
                Bolas.add_edge('c-' + str(i), 'b-0')
        return Bolas
    else:
        print("please choose Dubmle or Bolas")
        return

def walk_eigens_from_norm_laplace(NL_evals,NL_evecs,D,W):
    W_vals, W_vecs = np.linalg.eig(W)
    W_vals = sorted(W_vals)
    W_vecs = [W_vecs[np.where(W_vals == i)] for i in W_vals]
    """
    The theorem is that the NL matrix vals V with vectors Vecs,
    corispond with the W matrix vals W= 1-0.5v with vectors D^-1* Vec
    """
    calc_W_vals = 1 - (0.5 * np.array(NL_evals))
    calc_W_vecs = [(D.toarray() **.5) @ np.array(NL_evec).T for NL_evec in np.array(NL_evecs)]
    print("Walk matrix evals: {}\nWalk matrix evals from normed laplacian matrix: {}".format(W_vals, calc_W_vals))
    print("Walk matrix evecs: {}\nWalk matrix evecs from normed laplacian matrix: {}".format(W_vecs, calc_W_vecs))

def random_walk(G,n,M,W,alpa=8):
    #init p as delta
    p = np.zeros(n)
    a=random.randint(0,n - 1)
    p[a] = 1
    p_prev = np.zeros(n)

    #printing params
    steps = 1
    po = 0

    #vectors for aboundty
    d = np.sum(M, axis=1)
    pi = d/np.sum(d)
    w2 = sorted(np.linalg.eig(W)[0])[-2]

    fig, axes = plt.subplots(nrows=3, ncols=2)
    ax = axes.flatten()

    gap = []
    bound = []
    i=0
    while (p != p_prev).any():
        gap.append(np.sum(np.abs(p-pi)))
        bound.append(np.sum(w2**steps * np.sqrt(d / d[a])))
        if steps == alpa ** po:
            nx.draw(G, node_color=(p * (1/np.max(p))**2),vmin=0,vmax=1,ax=ax[i])  # , with_labels=True)
            i += 1
            po += 1
        p_prev = p
        p = W @ p_prev
        # todo: update convergence in log scale
        steps += 1
        if(steps>70000):
            break

    nx.draw(G, node_color=(p * (1 / np.max(p))), vmin=0, vmax=1, ax=ax[i])
    #plt.title("Lazy random walk")
    plt.savefig("Lazy random walk" + '.png')
    plt.show()

    x = np.linspace(0, 1, steps-1)
    plt.scatter(x, gap,marker='o')
    plt.scatter(x, bound, color='red', marker='.')
    plt.title("Lazy random walk bounded steps")
    plt.savefig("Lazy random walk bounded steps" + '.png')
    plt.show()

def ex4_3():
    n=30
    # presection
    Dubmle = gen_graph(n=n,type="Dubmle")
    Bolas = gen_graph(n=n, type="Bolas")
    plot_2D(Dubmle,"Dubmle")
    plot_2D(Bolas, "Bolas")

    ##a
    M_Dubmle, D_Dubmle, L_Dubmle = matrices(Dubmle)
    M_Bolas, D_Bolas, L_Bolas = matrices(Bolas)
    W_Dubmle = 0.5 * (np.eye(n*2) + M_Dubmle.toarray() @ np.linalg.inv(D_Dubmle.toarray()))
    W_Bolas = 0.5 * (np.eye(n*2+ n - 1) + M_Bolas.toarray() @ np.linalg.inv(D_Bolas.toarray()))
    print(W_Dubmle)
    print(W_Bolas)

    ##b
    normed_L_Dubmle = normalize(L_Dubmle.toarray())
    normed_L_Bolas = normalize(L_Bolas.toarray())
    NLD_evals,NLD_evecs= EVD(Dubmle,normed=True)
    NLB_evals, NLB_evecs = EVD(Bolas,normed=True)
    
    print("the sorted eigenvalues of the normed Laplacian matrix of the Dumble graph: ",NLD_evals)
    print("the sorted eigenvalues of the normed Laplacian matrix of the Bolas graph: ",NLB_evals)
    eigenvals_plot(NLD_evals, "Duble Normelized Laplacian matrix evals")
    eigenvals_plot(NLB_evals, "Bolas Normelized Laplacian matrix evals")
    
    ##c
    walk_eigens_from_norm_laplace(NLD_evals,NLD_evecs,D_Dubmle,W_Dubmle)
    walk_eigens_from_norm_laplace(NLB_evals,NLB_evecs,D_Bolas,W_Bolas)

    ##d
    random_walk(Dubmle,2*n,M_Dubmle,W_Dubmle,alpa= 8)
    random_walk(Bolas,3*n-1,M_Bolas,W_Bolas,alpa= 10)

def PowerMethod(B,epsilon = 1e-5):
    """
    this function using power iteration method
    to find the hightes eigenvalue with the corispond
    eigenvector
    :param B: similar to simetric matrix
    :return: v_t - largest val, u_t- corrispond vec
    """
    n = B.shape[1]
    u_t = np.random.rand(n)
    v_t = 0
    #while(np.abs(v_t- 1) > epsilon):
    for _ in range(n**3):
        u_t1 = B @ u_t
        u_t1_norm = np.linalg.norm(u_t1)
        u_t = u_t1 / u_t1_norm
        v_t = u_t.T @ B @ u_t
    return v_t, u_t

def PowerMethod2(B):
    v_1,u_1 = PowerMethod(B)
    norm = np.linalg.norm(u_1)
    x = u_1 / (norm**2)
    C = B - (v_1 * np.outer(u_1.T, x))
    v_2, w_2 = PowerMethod(C)
    u_2 = ((v_2 - v_1) * w_2) + (v_1 * np.dot(x.T, w_2) * u_1)
    u_2 = u_2 / np.linalg.norm(u_2)
    return v_2,u_2

#this method is just for tesing only
def create_w(n=10):
    b = np.random.random_integers(-2000, 2000, size=(n, n))
    return (b + b.T) / 2

def ex4_2():
    B_list =[create_w(i) for i in [2,4,6]]
    for B in B_list:
        print("ground truth:")
        w, v = np.linalg.eig(B)
        s_w = sorted(np.abs(w))
        s_v = [v[np.where(np.abs(w) == i)] for i in s_w]
        print(s_w[-1], s_v[-1])
        print(s_w[-2], s_v[-2])
        print("power iteration:")
        print(PowerMethod(B))
        print(PowerMethod2(B))

def Lazy_random_walk():
    ex4_2()
    ex4_3()

"""
==============================
Manifold Learning and Dimension Reduction
==============================
"""
def AffinityMat(Z, n_neighbor, epsilon, kernel_method= "Gaussian"):
    """
    Creates an Affinity matrix from the data of Z.
     In order to get a symmetric matrix, two points will consider adjacent if one of them is in the KNN of the other.
    """

    #Get Knn according to n_neighbor
    nbrs = NearestNeighbors(n_neighbors=n_neighbor, algorithm='auto').fit(Z)
    distances, indices = nbrs.kneighbors(Z)

    row_idx = []; col_idx = []; data = []

    if kernel_method == "Gaussian":
        for i in range(Z.shape[0]):
            dist = distances[i, 1:]
            col_idx.extend(indices[i, 1:])
            row_idx.extend([i] * len(dist))
            data.extend(np.exp(-(dist ** 2) / epsilon))
        A = csr_matrix((data, (row_idx, col_idx)), shape=(Z.shape[0], Z.shape[0]))
        A = A.maximum(A.transpose())

    elif kernel_method =='L2':
        for i in range(Z.shape[0]):
            dist = distances[i, 1:]
            col_idx.extend(indices[i, 1:])
            row_idx.extend([i] * len(dist))
            data.extend(dist)
        A = csr_matrix((data, (row_idx, col_idx)), shape=(Z.shape[0], Z.shape[0]))
        A = A.maximum(A.transpose())
    else:
        raise NameError('No such Kernel: {}'.format(kernel_method))

    return A

def CreateTorus(N,R,r,visualize=False):
    """
    Create a Torus surface with N points with radius R,r
    """
    X = np.random.uniform(size=(N, 2))
    S = np.array([(R + r * np.cos(2 * np.pi * X[:, 1])) * np.cos(2 * np.pi * X[:, 0]),
                  (R + r * np.cos(2 * np.pi * X[:, 1])) * np.sin(2 * np.pi * X[:, 0]),
                  R + r * np.sin(2 * np.pi * X[:, 1])]).T
    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(S[:, 0], S[:, 1], S[:, 2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
    return S

def LLE(X,n_nei,n_dim):
    """
    Use LLE on the data X. Embed to lower d dimension space. Create weight matrix with n_neighbor KNN
    :param X: rows are features columns are observations (DxN)
    :param n_dim:
    :param n_nei:
    :return: Embedded data Y [ rows are embedded observations, cols are lower dimension]
    """
    #Params
    D,N = X.shape
    #Get KNN of each point
    nbrs = NearestNeighbors(n_neighbors=n_nei+1, algorithm='auto').fit(X.T)
    distances, indices = nbrs.kneighbors(X.T)
    #Create W matrix
    row_idx = []; col_idx = []; csr_vals = []
    Wf = np.zeros((N,N))
    for i in range(N):
        C_i = np.matmul(np.transpose(X[:,i].reshape((D,1)) - X[:,indices[i,1:]]),(X[:,i].reshape((D,1))- X[:,indices[i,1:]]))
        onesHot_vect = np.ones((n_nei,1))
        C_i_Pinv = np.linalg.pinv(C_i)
        w_tilde_i = np.matmul(C_i_Pinv,onesHot_vect) / (np.dot(onesHot_vect.T,np.matmul(C_i_Pinv,onesHot_vect)))
        #Add to Sparse structure
        row_idx.extend(indices[i,1:])
        col_idx.extend([i]*len(indices[i,1:]))
        csr_vals.extend(w_tilde_i.flatten())

        Wf[indices[i,1:],i] = w_tilde_i.flatten()

    #Create Sparse W
    W = csr_matrix((csr_vals, (row_idx, col_idx)), shape=(N,N))

    #Compute Embedding by EVD
    S = sparse_eye(N,N) - W
    M = np.dot(S,S.T)


    #Use linlang EVD
    eig_values, eig_vectors = np.linalg.eig(M.toarray())
    idx_sorted = eig_values.argsort()
    eig_values = eig_values[idx_sorted]
    eig_vectors = eig_vectors[:, idx_sorted]

    Y = np.real(eig_vectors[:,1:n_dim+1])

    return Y

def DiffusionMapsEmbedding(X,n_dim,n_nei,epsilon,t=1,kernel_method="Gaussian"):
    """
    Applay Diffusion Maps Embedding on the data X. Creates adjacency matrix from n_nei.
     Do EVD to the random walk matrix on t iteration and taking n_dim smallest eignvectoes (except the constant one)
    :param X:
    :param n_dim:
    :param n_nei:
    :param t:
    :return: Embedded data Y [ rows are embedded observations, cols are lower dimension]
    """

    #Create Random Walk Matrix
    M = AffinityMat(X,n_neighbor=n_nei,epsilon=epsilon,kernel_method= kernel_method).toarray()
    D = np.diag(np.sum(M,1))
    W = np.matmul(np.linalg.inv(D),M)


    #Create t step random walk matrix
    W_t = np.linalg.matrix_power(W,t)

    #EVD
    eig_values, eig_vectors = np.linalg.eig(W_t)
    idx_sorted = eig_values.argsort()[::-1]
    eig_values = eig_values[idx_sorted]
    eig_vectors = eig_vectors[:, idx_sorted]

    Y = np.matmul(np.diag(eig_values[1:n_dim+1]),np.transpose(eig_vectors[:,1:n_dim+1]))
    Y = np.real(Y)
    return Y.T

def Plotter(data,method, n_nei_arr, n_dim=2,saveFigName= None):
    """
    Plot data for different embedding methods for different n_nei values
    :param data:
    :param method:
    :param n_nei_arr:
    :param n_dim_arr:
    :param saveFigName: string for save path, if None does not save figure
    :return:
    """

    #Create figure
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    ncol = 4
    nrow = len(n_nei_arr)
    scaleFigX = 4
    scaleFixY = 4
    fig = plt.figure(figsize=(scaleFigX*(ncol + 1),scaleFixY*( nrow + 1)))
    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.1, hspace=0.2,
                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
    if method == "LLE":
        title = "Local Linear Embedding"
    elif method == "DiffMaps":
        title = "Diffusion Maps Embbeding"
    fig.suptitle(title, fontsize="36")


    for i,n_nei in enumerate(n_nei_arr):
        if method == "LLE":
            Y = LLE(data.T,n_nei=n_nei,n_dim=n_dim)
        elif method == "DiffMaps":
            Y = DiffusionMapsEmbedding(data,n_dim,n_nei,epsilon=2,t=1)

        ax = plt.subplot(gs[i, 0], projection='3d')
        sct1 = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=Y[:, 0], s=5)
        fig.colorbar(sct1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # ax.set_zlabel(r"n_nei={}".format(n_nei), fontsize=14)
        ax.text(-25, -30, 10, "n_nei={}".format(n_nei), 'z', fontsize=20)
        if i == 0:
            ax.set_title(r"Data Colored By $\psi_1$", fontsize=20)
        ax = plt.subplot(gs[i, 1], projection='3d')
        sct2 = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=Y[:, 1], s=5)
        fig.colorbar(sct2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        if i == 0:
            ax.set_title(r"Data Colored By $\psi_2$", fontsize=20)
        ax = plt.subplot(gs[i, 2], projection='3d')
        sct3 = ax.scatter(Y[:, 0], Y[:, 1], c=data[:, 0], s=5)
        fig.colorbar(sct3)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        if i == 0:
            ax.set_title(r"Embedded Colored By Data $x$", fontsize=20)
        ax = plt.subplot(gs[i, 3], projection='3d')
        sct4 = ax.scatter(Y[:, 0], Y[:, 1], c=data[:, 1], s=5)
        fig.colorbar(sct4)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        if i == 0:
            ax.set_title(r"Embedded Colored By Data $y$", fontsize=20)

    plt.show()
    if saveFigName:
        fig.savefig("{}.png".format(saveFigName))

def ClassifierPlotter(dataStruct,method,n_nei_arr,n_dim=2,saveFigName= None,titleadd= "",kernel_method = "Gaussian"):
    """
    Plot the data painted by classes
    :param data:
    :param method:
    :param n_nei_arr:
    :param n_dim:
    :param saveFigName:
    :return:
    """

    # Create figure
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    ncol = len(n_nei_arr)
    nrow = 1
    scaleFigX = 4
    scaleFixY = 4
    fig = plt.figure(figsize=(scaleFigX * (ncol + 1), scaleFixY * (nrow + 1)))
    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.1, hspace=0.2,
                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
    if method == "LLE":
        title = "Local Linear Embedding"
    elif method == "DiffMaps":
        title = "Diffusion Maps Embbeding"
    fig.suptitle(title + "({})".format(titleadd), fontsize="36")

    lables = dataStruct.target
    cmap = mcolors.ListedColormap(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    n = len(np.unique(lables))
    norm = mcolors.BoundaryNorm(np.arange(n + 1) - 0.5, n)

    for i,n_nei in enumerate(n_nei_arr):
        if method == "LLE":
            Y = LLE(dataStruct.data.T,n_nei=n_nei,n_dim=n_dim)
        elif method == "DiffMaps":
            Y = DiffusionMapsEmbedding(dataStruct.data,n_dim,n_nei,epsilon=2,t=1,kernel_method=kernel_method)

        ax = plt.subplot(gs[0, i])
        sct1 = ax.scatter(Y[:, 0], Y[:, 1], c=lables,cmap=cmap,norm=norm, s=5)
        fig.colorbar(sct1,ticks=np.unique(lables))
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        #ax.set_zticklabels([])
        # ax.set_zlabel(r"n_nei={}".format(n_nei), fontsize=14)
        #ax.text(-25, -30, 10, "n_nei={}".format(n_nei), 'z', fontsize=20)

        ax.set_title("n_nei={}".format(n_nei), fontsize=20)


    plt.show()
    if saveFigName:
        fig.savefig("{}.png".format(saveFigName))

def Dimension_Reduction():
    ###### 3.a - Torus
    n_nei_arr = [10, 50, 100]
    # Create Torus
    S_tourus = CreateTorus(N=2000, R=10, r=4, visualize=False)

    # LLE
    Plotter(S_tourus, method='LLE', n_nei_arr=n_nei_arr, n_dim=2, saveFigName="LLE")

    # DiffMaps
    Plotter(S_tourus, method='DiffMaps', n_nei_arr=n_nei_arr, n_dim=2, saveFigName="DiffMaps")

    ###### 3.b - Digits
    n_nei_arr = [10,50,100]
    numOfClassList = [3,5,7]
    for numOfClass in numOfClassList:
        digits = load_digits(n_class=numOfClass)
        name = "Digits {}".format(numOfClass)
        ClassifierPlotter(digits, method='LLE', n_nei_arr=n_nei_arr, n_dim=2, titleadd=name,
                          saveFigName=name+"_LLE")
        ClassifierPlotter(digits, method='DiffMaps', n_nei_arr=n_nei_arr, n_dim=2, titleadd=name+ " (Kernel: L2 Distance)",
                          saveFigName=name + "_DiffMaps_L2",kernel_method="L2")


if __name__ == "__main__":
    graphs()
    Lazy_random_walk()
    Dimension_Reduction()
