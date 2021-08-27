from typing import Callable, List, Tuple , Dict
import numpy as np
from abc import ABC , abstractmethod
from dataclasses import dataclass


@dataclass
class Domain:

    """
    interface for defining a domain of a function
    domain is equipped with a standardized pdf

    inputs : 
        dimn : domain dimn
        mu : domain distribution mean
        sigma : domain distribution length scale
        type_rho : pdf type
    """

    dimn : int
    mu : float
    sigma : float
    type_rho : str

    def __post_init__( self ):

        self.f_sample_from_rho = self.__factory_make_rho_sampler( self.type_rho )

    # Private/protected interface

    def __factory_make_rho_sampler( self , type_rho : str ) -> Dict[ str , Callable[ [ None ] , float ] ]:

        factories = {
            'uniform'  : lambda : np.random.uniform( self.mu - self.sigma , self.mu + self.sigma ) ,
            'gaussian' : lambda : np.random.normal( self.mu , self.sigma )
        }

        return factories[ type_rho ]



@dataclass
class BoundedDomain:

    """
    interface for defining a bounded domain of a function

    inputs : 
        dimn : domain dimn
        D_lower_bounds : list of lower bounds, each element is of size dimn
        D_upper_bounds : list of upper bounds, each element is of size dimn
        type_rho : pdf type
        k_sigma : coefficient of length scale of distribution
    """

    dimn : int
    D_lower_bounds : List[ Tuple[ float ] ]
    D_upper_bounds : List[ Tuple[ float ] ]
    type_rho : str
    k_sigma : float = 1.0

    def __post_init__(self):

        self.mu    = 0.5 * ( self.D_lower_bounds + self.D_upper_bounds )
        self.sigma = self.k_sigma * self.get_average_box_length()
        self.f_sample_from_rho , self.f_rho = self.__factory_make_rho( self.type_rho )

    def is_in_domain( self , X : np.array ) -> bool :

        """
        is query point X (of size self.dimn) inside the ambient domain?
        """

        lower = np.all( [ xi >= li for (xi,li) in zip( X , self.D_lower_bounds ) ] )
        upper = np.all( [ xi <= ui for (xi,ui) in zip( X , self.D_upper_bounds ) ] )

        return lower & upper

    def evaluate_pdf_centered_gaussian( self , 
                                        X : np.array , 
                                        k_sigma : float = 1 ) -> float :

        """
        return value proportional to prob( X ~ N( mu , sigma ) )
        """

        return np.exp( -0.5 * np.linalg.norm( X - self.mu )**2 / ( k_sigma * self.sigma )**2 )

    def get_average_box_length( self ) -> float :

        return np.mean( self.D_upper_bounds - self.D_lower_bounds )

    # Private/protected interface

    def __factory_make_rho( self , type_rho : str ) -> Tuple[ Dict[ str , Callable ] , Dict[ str , Callable ] ]:

        def f_gen_trunc_gaus() -> float:
            while True:
                xcand = np.random.normal( self.mu , self.sigma )
                if self.is_in_domain( xcand ):
                    return xcand

        def f_gen_uniform() -> float:
            return np.random.uniform( self.k_sigma * self.D_lower_bounds , self.k_sigma * self.D_upper_bounds )

        def f_uniform( x : float ) -> float:
            return 1 if self.is_in_domain( x / self.k_sigma ) else 0 # division needed for length scale of rho

        def f_trunc_gaus( x : float ) -> float:
            return np.exp( -0.5 * np.linalg.norm( x - self.mu )**2 / ( self.sigma )**2 ) if self.is_in_domain( x ) else 0

        factories = {
            'uniform'  : ( f_gen_uniform , f_uniform ) ,
            'gaussian' : ( f_gen_trunc_gaus , f_trunc_gaus )
        }

        return factories[ type_rho ]