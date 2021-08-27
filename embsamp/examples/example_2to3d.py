import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from embsamp.src.projAndLifting import *
from embsamp.src.functionDomain import *
from embsamp.src.embeddingSamplers import *


def main():

    D = 3; d = 2

    M = int( input("Enter <mc_samples> : ") )

    amb     = BoundedDomain( D , -np.ones( D ) , np.ones( D ) , 'gaussian' , k_sigma=0.1 )
    emb     = Domain( d , -np.ones( d ) , np.ones( d ) , 'gaussian' )
    trans   = RandomProjectionCompression( D , d , 'hypersphere' )
    sampler = EmbeddingSampler( trans , amb , emb , 'MCMC_ambientConstrained' )

    X     = sampler.sample( M )
    Xlift = trans.get_lifted_coordinates_list( X )

    print( str( len(X) ) + ' samples generated' )

    plt.scatter( [ xi[0] for xi in X ] , [ xi[1] for xi in X ] , s=2 )
    plt.gca().set_aspect( 'equal' )
    plt.title( 'embedded space coordinates' )

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter( [ xi[0] for xi in Xlift ] , [ xi[1] for xi in Xlift ] , [ xi[2] for xi in Xlift ] , s=2 )
    ax.set_xlim( list(zip( amb.D_lower_bounds , amb.D_upper_bounds ))[0] )
    ax.set_ylim( list(zip( amb.D_lower_bounds , amb.D_upper_bounds ))[1] )
    ax.set_zlim( list(zip( amb.D_lower_bounds , amb.D_upper_bounds ))[2] )
    plt.title( 'ambient space coordinates' )

    plt.show()


if __name__ == '__main__':

    main()