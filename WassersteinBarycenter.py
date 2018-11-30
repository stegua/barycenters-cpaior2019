# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 20:23:07 2018

@author: gualandi, stefano.gualandi ( at ) gmail.com
"""

import time
import csv
import numpy as np
from math import sqrt, pow

from numpy import genfromtxt
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum, tuplelist


import logging

def DrawDigit(A, label=''):
    """ Draw single digit as a greyscale matrix"""
    plt.figure(figsize=(6,6))
    # Uso la colormap 'gray' per avere la schacchiera in bianco&nero
    plt.imshow(A, cmap='gray_r')
    plt.xlabel(label)
    plt.show()


def EuclideanBarycenter(images):
    """ Compute Arithmetic mean of all the images """
    n = len(images[0])
    N = range(n)
    avg = [0 for _ in N]
    for row in images:
        for i in N:
            avg[i] += row[i] 
    
    k = len(images)
    for i in N:
        avg[i] = avg[i]/k
        
    return avg


def BuildBipartiteGraph(N, ground='L2', plot=False):
    """ Build a bipartite graph, as support to compute KW distances """
    def ID(x,y):
        return x*N+y

    G = nx.DiGraph()

    n = N*N
    for i in range(N):
        for j in range(N):
            G.add_node(ID(i,j), pos=(2,ID(i,j)))
            G.add_node(n+ID(i,j), pos=(6,ID(i,j)))
            
    for i in range(N):
        for j in range(N):
            for v in range(N):
                for w in range(N):
                    if ground == 'L1':
                        G.add_edge(ID(i,j), n+ID(v, w), weight=abs(i-v) + abs(j-w))
                    if ground == 'L2':
                        G.add_edge(ID(i,j), n+ID(v, w), weight=sqrt(pow(i-v, 2) + pow(j-w, 2)))
                    if ground == 'Linf':
                        G.add_edge(ID(i,j), n+ID(v, w), weight=max(abs(i-v), abs(j-w)))
    
    if plot:
        plt.figure(3,figsize=(12,12))
        plt.axis('equal')
        pos=nx.get_node_attributes(G,'pos')
        nx.draw(G, pos, font_weight='bold', node_color='blue', 
                arrows=True, arrowstyle='->',  arrowsize=15, width=1, node_size=200)
        plt.savefig("bipartite_{}.png".format(N), format="PNG")

    return G

def BarycenterBipartite(images, G):
    """ Compute the Kantorovich Wasserstein Barycenter of any order using bipartite subgraphs"""
    K = len(images)
    n = len(images[0])
        
    # Build model
    m = Model()
    m.setParam(GRB.Param.Method, 2)
    m.setParam(GRB.Param.NumericFocus, 1)
    m.setParam(GRB.Param.OutputFlag, 0)
    m.setParam(GRB.Param.Crossover, 0)
    m.setAttr(GRB.Attr.ModelSense, 1)         
    
    # Create variables
    Y = {}
    for v in range(n):
        Y[v] = m.addVar(obj=0)
    
    X = {}
    for e in G.edges():
        i, j = e
        for k in range(K):
            X[i,j,k] = m.addVar(obj=G.edges[i,j]['weight'])
                 
    m.update()
    
    # In and out flow variable on the bipartite graph
    for v in range(n):
        Fs = [w for w in G.out_edges(v)] 
        for k in range(K):
            m.addConstr(quicksum(X[i,j,k] for i,j in Fs) == images[k][v])
    
    for v in range(n):
        Bs = [w for w in G.in_edges(n+v)] 
        for k in range(K):
            m.addConstr(quicksum(X[i,j,k] for i,j in Bs) == Y[v])

    m.addConstr(quicksum(Y[v] for v in Y) == 1.0)

    # Solve the model
    m.optimize()

    return m.getAttr(GRB.Attr.ObjVal), [Y[i].X for i in Y]
    

def TestPaperBipartite():
    """ Run complete test for digits with Bipartite graphs """
    # create file handler which logs even debug messages
    logger = logging.getLogger('BarycenterBipartite')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('Bipartite.log')
    fh.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    
    # Build bipartite graphs only onces
    GD = ['L1', 'Linf', 'L2']
    G = {}
    for gd in GD:
        G[gd] = BuildBipartiteGraph(28, gd)

    DIR = '..\\data\\barycenter_'
    SFX = '.csv'
    NUM = [str(i) for i in range(10)]
    
    for K in [50]:
        for gd in GD:        
            for n in NUM:
                FILEIN = DIR+n+SFX
                start = time.time()
                my_data = genfromtxt(FILEIN, delimiter=',', skip_header=1)
                read_time = time.time()-start
                start = time.time()
                
                # Normalize pixels
                images = []
                for row in my_data[:K]:
                    A = np.array(row[1:])
                    A = A/sum(A)
                    images.append(A)
        
                obj, A = BarycenterBipartite(images, G[gd])
                logger.info('Test: {} Images {} GD: {} Readtime: {:f} Runtime: {:f} Obj: {:f}'.format(n, K, gd, read_time, time.time()-start, obj))
                fh.flush()
                # Dump barycenter
                FILEOUT = DIR+'Bipartite.{}.{}.{}.barycenter.csv'.format(gd, n, K)
                with open(FILEOUT, "a") as output:
                    writer = csv.writer(output, lineterminator='\n')
                    writer.writerow(A)    
    
    
def BarycenterL1(images):
    """ Compute the Kantorovich Wasserstein Barycenter of order 1 with L1 as ground distance """
    K = len(images)
    n = len(images[0])
    s = int(np.sqrt(n))
    
    def ID(x,y):
        return x*s+y
    
    # Build model
    m = Model()
    m.setParam(GRB.Param.Method, 2)
    m.setParam(GRB.Param.Crossover, 0)
    m.setParam(GRB.Param.NumericFocus, 1)
    m.setParam(GRB.Param.OutputFlag, 0)
    m.setAttr(GRB.Attr.ModelSense, 1)         
    
    # Create variables
    Z = {}
    for i in range(n):
        Z[i] = m.addVar(obj=0)
    
    X = {}
    for k in range(K):
        for i in range(s):
            for j in range(s):
                X[k, ID(i,j)] = {}
    X[k, n] = {}
        
    A = []
    for k in range(K):
        for i in range(s):
            for j in range(s-1):
                X[k,ID(i,j)][k,ID(i,j+1)] = m.addVar(obj=1)
                X[k,ID(i,j+1)][k,ID(i,j)] = m.addVar(obj=1)
    
                # Keep adding arcs to A
                if k == 0:
                    A.append((ID(i,j), ID(i,j+1)))
                    A.append((ID(i,j+1), ID(i,j)))

        for i in range(s-1):
            for j in range(s):
                X[k,ID(i,j)][k,ID(i+1,j)] = m.addVar(obj=1)
                X[k,ID(i+1,j)][k,ID(i,j)] = m.addVar(obj=1)

                # Keep adding arcs to A           
                if k == 0:
                    A.append((ID(i,j), ID(i+1,j)))
                    A.append((ID(i+1,j), ID(i,j)))

    A = tuplelist(A)      

    m.update()
    
    # Flow variables
    for i in range(n):
        Fs = A.select(i,'*')
        Bs = A.select('*',i)
        for k in range(K):
            m.addConstr(quicksum(X[k,i][k,j] for _,j in Fs) - quicksum(X[k,j][k,i] for j,_ in Bs) == images[k][i] - Z[i])

    m.addConstr(quicksum(Z[i] for i in range(n)) == 1.0)
    
    # Solve the model
    m.optimize()
    
    # Return the final image
    return m.getAttr(GRB.Attr.ObjVal), [Z[i].X for i in range(n)]

def TestPaperL1():
    """ Run complete test for digits with L1 norm """
    # create file handler which logs even debug messages
    logger = logging.getLogger('BarycenterL1')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('L1.log')
    fh.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)

    
    DIR = '..\\data\\barycenter_'
    SFX = '.csv'
    NUM = [str(i) for i in range(10)]
    
    for K in [50, 100, 200, 400, 800, 1600, 3200]:
        for n in NUM:
            FILEIN = DIR+n+SFX
            start = time.time()
            my_data = genfromtxt(FILEIN, delimiter=',', skip_header=1)
            read_time = time.time()-start
            start = time.time()
            
            # Normalize pixels
            images = []
            for row in my_data[:K]:
                A = np.array(row[1:])
                A = A/sum(A)
                images.append(A)
    
            obj, A = BarycenterL1(images)
            logger.info('Test: {} Images {} Readtime: {:f} Runtime: {:f} Obj: {:f}'.format(n, K, read_time, time.time()-start, obj))
            fh.flush()
            # Dump barycenter
            FILEOUT = DIR+'L1.{}.{}.barycenter.csv'.format(n, K)
            with open(FILEOUT, "a") as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerow(A)


def BarycenterLinf(images):
    """ Compute the Kantorovich Wasserstein Barycenter of order 1 with Linf as ground distance """
    K = len(images)
    n = len(images[0])
    s = int(np.sqrt(n))
    
    def ID(x,y):
        return x*s+y
    
    # Build model
    m = Model()
    m.setParam(GRB.Param.Method, 2)
    m.setParam(GRB.Param.Crossover, 0)
    m.setParam(GRB.Param.NumericFocus, 1)
    m.setParam(GRB.Param.OutputFlag, 0)
    m.setAttr(GRB.Attr.ModelSense, 1)         
    
    # Create variables
    Z = {}
    for i in range(n):
        Z[i] = m.addVar(obj=0)
    
    X = {}
    for k in range(K):
        for i in range(s):
            for j in range(s):
                X[k, ID(i,j)] = {}
    X[k, n] = {}
        
    A = []
    for k in range(K):
        for i in range(s):
            for j in range(s-1):
                X[k,ID(i,j)][k,ID(i,j+1)] = m.addVar(obj=1)
                X[k,ID(i,j+1)][k,ID(i,j)] = m.addVar(obj=1)
    
                if k == 0:
                    A.append((ID(i,j), ID(i,j+1)))
                    A.append((ID(i,j+1), ID(i,j)))

        for i in range(s-1):
            for j in range(s):
                X[k,ID(i,j)][k,ID(i+1,j)] = m.addVar(obj=1)
                X[k,ID(i+1,j)][k,ID(i,j)] = m.addVar(obj=1)
           
                if k == 0:
                    A.append((ID(i,j), ID(i+1,j)))
                    A.append((ID(i+1,j), ID(i,j)))


        for i in range(s-1):
            for j in range(s-1):
                X[k,ID(i,j)][k,ID(i+1,j+1)] = m.addVar(obj=1)
                X[k,ID(i+1,j+1)][k,ID(i,j)] = m.addVar(obj=1)
                X[k,ID(i,j+1)][k,ID(i+1,j)] = m.addVar(obj=1)
                X[k,ID(i+1,j)][k,ID(i,j+1)] = m.addVar(obj=1)

                if k == 0:
                    A.append((ID(i,j), ID(i+1,j+1)))
                    A.append((ID(i+1,j+1), ID(i,j)))
                    A.append((ID(i,j+1), ID(i+1,j)))
                    A.append((ID(i+1,j), ID(i,j+1)))
                 
    A = tuplelist(A)      

    m.update()
    
    # Flow variables
    for i in range(n):
        Fs = A.select(i,'*')
        Bs = A.select('*',i)
        for k in range(K):
            m.addConstr(quicksum(X[k,i][k,j] for _,j in Fs) - quicksum(X[k,j][k,i] for j,_ in Bs) == images[k][i] - Z[i])

    m.addConstr(quicksum(Z[i] for i in range(n)) == 1.0)
    # Solve the model
    m.optimize()
    
    return m.getAttr(GRB.Attr.ObjVal), [Z[i].X for i in range(n)]

def TestPaperLinf():
    """ Run complete test for digits with Linf norm """
    # create file handler which logs even debug messages
    logger = logging.getLogger('BarycenterLinf')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('Linf.log')
    fh.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)

    DIR = '..\\data\\barycenter_'
    SFX = '.csv'
    NUM = [str(i) for i in range(10)]
    
    for K in [50, 100, 200, 400, 800, 1600, 3200]:
        for n in NUM:
            FILEIN = DIR+n+SFX
            start = time.time()
            my_data = genfromtxt(FILEIN, delimiter=',', skip_header=1)
            read_time = time.time()-start
            start = time.time()
            
            # Normalize pixels
            images = []
            for row in my_data[:K]:
                A = np.array(row[1:])
                A = A/sum(A)
                images.append(A)
    
            obj, A = BarycenterLinf(images)
            logger.info('Test: {} Images {} Readtime: {:f} Runtime: {:f} Obj: {:f}'.format(n, K, read_time, time.time()-start, obj))
            fh.flush()
            # Dump barycenter
            FILEOUT = DIR+'Linf.{}.{}.barycenter.csv'.format(n, K)
            with open(FILEOUT, "a") as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerow(A)
                

def BarycenterL2(images, G):
    """ Compute the Kantorovich Wasserstein Barycenter of order 1 with L2 as ground distance """
    K = len(images)
             
    # Build model
    m = Model()
    m.setParam(GRB.Param.Method, 2)
    m.setParam(GRB.Param.NumericFocus, 1)
    m.setParam(GRB.Param.OutputFlag, 0)
    m.setParam(GRB.Param.Crossover, 0)
    m.setAttr(GRB.Attr.ModelSense, 1)         
    
    # Create variables
    Y = {}
    for v in G.nodes():
        Y[v] = m.addVar(obj=0)
    
    X = {}
    for e in G.edges():
        i, j = e
        for k in range(K):
            X[i,j,k] = m.addVar(obj=G.edges[i,j]['weight'])
                 
    m.update()
    
    # Flow variables
    for v in G.nodes():
        Fs = [w for w in G.out_edges(v)] 
        Bs = [w for w in G.in_edges(v)] 
        for k in range(K):
            m.addConstr(quicksum(X[i,j,k] for i,j in Fs) - quicksum(X[i,j,k] for i,j in Bs) == images[k][v] - Y[v])

    m.addConstr(quicksum(Y[v] for v in G.nodes()) == 1.0)

    m.update()
            
    # Solve the model
    m.optimize()

    return m.getAttr(GRB.Attr.ObjVal), [Y[i].X for i in Y]


def GCD(a, b):
    a = max(a, -a)
    b = max(b, -b)
    while b != 0:
        t = b
        b = a % b
        a = t
    return a

def CoprimesSet(L):
    Cs = []
    for v in range(-L, L+1):
        for w in range(-L, L+1):
            if (not (v == 0 and w == 0)) and GCD(v, w) == 1:
                Cs.append((v, w))
    return Cs   
    
import networkx as nx
def BuildGraph(N, L, plot=False):
    """ Build a flow graph with ell_2 norms costs, and parameter L """
    def ID(x,y):
        return x*N+y

    G = nx.DiGraph()

    for i in range(N):
        for j in range(N):
            G.add_node(ID(i,j), pos=(i,j))
            
    Cs = CoprimesSet(L)
    for i in range(N):
        for j in range(N):
            for (v, w) in Cs:
                if i + v >= 0 and i + v < N and j + w >= 0 and j + w < N:
                    G.add_edge(ID(i,j), ID(i+v, j+w), weight=sqrt(pow(v, 2) + pow(w, 2)))
    
    if plot:
        plt.figure(3,figsize=(12,12))
        plt.axis('equal')
        pos=nx.get_node_attributes(G,'pos')
        nx.draw(G, pos, font_weight='bold', node_color='blue', 
                arrows=True, arrowstyle='->',  arrowsize=15, width=1, node_size=500)
        plt.savefig("grid_{}_{}.png".format(N,L), format="PNG")

    return G

def BuildGraphL1(N, plot=False):
    """ Support for plotting graph with only four neighbours (used in the paper) """
    def ID(x,y):
        return x*N+y

    G = nx.DiGraph()

    for i in range(N):
        for j in range(N):
            G.add_node(ID(i,j), pos=(i,j))
            
    for i in range(N):
        for j in range(N):
            if i+1<N:
                G.add_edge(ID(i,j), ID(i+1, j), weight=1)
            if j+1<N:
                G.add_edge(ID(i,j), ID(i, j+1), weight=1)    
            if i-1>=0:
                G.add_edge(ID(i,j), ID(i-1, j), weight=1)
            if j-1>=0:
                G.add_edge(ID(i,j), ID(i, j-1), weight=1)    
    
    if plot:
        plt.figure(3,figsize=(12,12))
        plt.axis('equal')
        pos=nx.get_node_attributes(G,'pos')
        nx.draw(G, pos, font_weight='bold', node_color='blue', 
                arrows=True, arrowstyle='->',  arrowsize=15, width=1, node_size=500)
        plt.savefig("gridL1_{}.png".format(N), format="PNG")

    return G

def TestPaperL2():
    """ Run complete test for digits with Linf norm """
    # create file handler which logs even debug messages
    logger = logging.getLogger('BarycenterL2')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('L2.log')
    fh.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)

    DIR = '..\\data\\barycenter_'
    SFX = '.csv'
    NUM = [str(i) for i in range(10)]
    
    # Build graphs only onces
    Ls = [1, 2, 3, 5]
    Gs = {}
    for L in Ls:
        Gs[L] = BuildGraph(28, L)
    
    # WARNING: with K > 800, andL > 3 it might run out of memory, even with 32GB of RAM
    for K in [50, 100, 200, 400, 800, 1600, 3200]:
        for n in NUM:
            for L in Ls:                
                FILEIN = DIR+n+SFX
                start = time.time()
                my_data = genfromtxt(FILEIN, delimiter=',', skip_header=1)
                read_time = time.time()-start
                start = time.time()
                
                # Normalize pixels
                images = []
                for row in my_data[:K]:
                    A = np.array(row[1:])
                    A = A/sum(A)
                    images.append(A)
        
                obj, A = BarycenterL2(images, Gs[L])
                logger.info('L: {} Test: {} Images {} Readtime: {:f} Runtime: {:f} Obj: {:f}'.format(L, n, K, read_time, time.time()-start, obj))
                fh.flush()
                # Dump barycenter
                FILEOUT = DIR+'L2.{}.{}.{}.barycenter.csv'.format(L, n, K)
                with open(FILEOUT, "a") as output:
                    writer = csv.writer(output, lineterminator='\n')
                    writer.writerow(A)
                    

#------------------------------------------
#              MAIN ENTRY POINT
#------------------------------------------
if __name__ == "__main__":
    # Uncomment the following to run the test described in the paper
    #TestPaperL1()
    #TestPaperLinf()
    #TestPaperL2()    
    #TestPaperBipartite()
    
    # Uncomment the following for plotting a gew graph
    #BuildGraph(5, 1, plot=True)
    #BuildGraphL1(5, plot=True)
    pass