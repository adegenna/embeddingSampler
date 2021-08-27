from typing import Callable, List
import numpy as np
from abc import ABC , abstractmethod
from dataclasses import dataclass


@dataclass
class ProjectionAndLiftingOperators( ABC ):

    """
    interface for managing projection and lifting mappings 
    for functions with a low-dimensional structure
    """

    @abstractmethod
    def get_lowdimn_coordinates( self , X_high : np.array ) -> np.array :
        """
        map X_high from high dimn ambient space -> low dimn embedded space
        """
        pass

    @abstractmethod
    def get_lifted_coordinates( self , X_low : np.array ) -> np.array :
        """
        map X_low from low dimn embedded space -> high dimn ambient space
        """
        pass


@dataclass
class ProjectionAndLiftingOperators_Linear( ProjectionAndLiftingOperators ):

    """
    interface for low dimensional models of functions
    where the low-dimensional manifold is linear
    """

    @abstractmethod
    def get_list_of_projection_vectors( self ) -> List :
        
        """
        return list[ <np.array> ] , each list element is D-dimn, and there are d of them
        """
        pass

    @abstractmethod
    def get_list_of_lifting_vectors( self ) -> List :

        """
        return list[ <np.array> ] , each list element is d-dimn, and there are D of them
        """


@dataclass
class RandomProjectionCompression( ProjectionAndLiftingOperators_Linear ):

    """
    implementation for function compression with random gaussian matrices
    based on johnnson-lindenstrauss lemma
    
    D : ambient space dimension
    d : embedded space dimension
    """

    D : int
    d : int 
    type_random_proj : str

    def __post_init__( self ):

        self.P = self.__factory_get_random_proj_matrix( self.type_random_proj )
        self.P_inv = np.linalg.pinv( self.P )

    # Public interface

    def get_lowdimn_coordinates( self , X : np.array ) -> np.array :

        return self.P.dot( X )

    def get_lifted_coordinates( self , X : np.array ) -> np.array :

        return self.P_inv.dot( X )

    def get_list_of_projection_vectors( self ) -> List :

        return list( self.P )

    def get_list_of_lifting_vectors( self ) -> List :

        return list( self.P_inv )

    def get_lifted_coordinates_list(self, X_low: List[ np.array ] ) -> List[ np.array ]:

        """
        use when you have a list of low-dimn coordinates rather than an array
        maps to a list of high-dimn coordinates
        """
        
        return list( self.get_lifted_coordinates( np.array(X_low).T ).T )

    # Private interface

    def __factory_get_random_proj_matrix( self ,  type_random_proj : str ) -> np.array :

        factories = {
            'JL'          : self.__get_random_proj_matrix_JL ,
            'hypersphere' : self.__get_random_proj_matrix_hypersphere
        }

        return factories[ type_random_proj ]( self.D , self.d )

    def __get_random_proj_matrix_JL( self , 
                                     D : int , 
                                     d : int ) -> np.array :

        return np.random.normal( 0 , 1, [ d , D ] )
    
    def __get_random_proj_matrix_hypersphere( self , 
                                              D : int , 
                                              d : int ) -> np.array :
        
        P = self.__get_random_proj_matrix_JL( self.D , self.d )

        return P / np.tile( np.linalg.norm( P , axis=1 ) , (self.D,1) ).T
