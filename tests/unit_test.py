import numpy as np
import unittest

from embsamp.src.embeddingSamplers import EmbeddingSampler
from embsamp.src.functionDomain import BoundedDomain, Domain
from embsamp.src.projAndLifting import *


class Test_projAndLifting( unittest.TestCase ):
    
    def setUp( self ):
        
        self.d = 2
        self.D = 100
        self.M = 128
    
    def test_randomProjection_JL( self ):

        model = RandomProjectionCompression( self.D , self.d , 'JL' )
        X     = np.random.uniform( 0 , 1 , [ self.D , self.M ] )

        self.assertEqual( model.get_lowdimn_coordinates( X ).shape , ( self.d , self.M ) )
        self.assertEqual( model.get_lifted_coordinates( model.get_lowdimn_coordinates( X ) ).shape , X.shape )

    def test_randomProjection_hypersphere( self ):

        model = RandomProjectionCompression( self.D , self.d , 'hypersphere' )
        X     = np.random.uniform( 0 , 1 , [ self.D , self.M ] )

        self.assertEqual( model.get_lowdimn_coordinates( X ).shape , ( self.d , self.M ) )
        self.assertEqual( model.get_lifted_coordinates( model.get_lowdimn_coordinates( X ) ).shape , X.shape )
        
        for vi in model.get_list_of_projection_vectors():
            self.assertAlmostEqual( np.linalg.norm( vi ) , 1.0 )


class Test_functionDomain( unittest.TestCase ):

    def setUp( self ):

        self.D = 100
        self.D_lower = np.zeros( self.D )
        self.D_upper = np.ones( self.D )
        self.M = 128

    def test_isInDomain( self ):

        dom = BoundedDomain( self.D , self.D_lower , self.D_upper , 'uniform' )

        X_in  = [ np.random.uniform( 0.5 , 0.7 , self.D ) for i in range(self.M) ]
        X_out = [ np.random.uniform( 1.5 , 1.7 , self.D ) for i in range(self.M) ]

        for xi in X_in:
            self.assertTrue( dom.is_in_domain( xi ) )
        for xi in X_out:
            self.assertFalse( dom.is_in_domain( xi ) )


class Test_embeddingSamplers( unittest.TestCase ):

    def setUp( self ):

        self.D = 5
        self.D_lower = -np.ones( self.D )
        self.D_upper =  np.ones( self.D )

        self.d = 2
        self.d_lower = -np.ones( self.d )
        self.d_upper =  np.ones( self.d )
        
        self.M = 10

    def test_ambientConstrained( self ):

        amb   = BoundedDomain( self.D , self.D_lower , self.D_upper , 'uniform' )
        emb   = Domain( self.d , self.d_lower , self.d_upper , 'uniform' )
        trans = RandomProjectionCompression( self.D , self.d , 'hypersphere' )
        
        sampler = EmbeddingSampler( trans , amb , emb , 'MCMC_ambientConstrained' ) #'MC_ambientConstrained' )

        print( 'samples : ' )
        for xi in sampler.sample( self.M ):
            print( xi , ' -> ' , trans.get_lifted_coordinates( xi ) )
            self.assertTrue( amb.is_in_domain( trans.get_lifted_coordinates( xi ) ) )







if __name__ == '__main__':

    unittest.main()
