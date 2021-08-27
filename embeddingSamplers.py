from projAndLifting import ProjectionAndLiftingOperators
from functionDomain import BoundedDomain, Domain
from typing import Callable, List
import numpy as np
from abc import ABC , abstractmethod
from dataclasses import dataclass


@dataclass
class EmbeddingSampler:

    """
    interface for various methods for sampling from the embedded space
    """

    proj_and_lift  : ProjectionAndLiftingOperators
    ambient_space  : BoundedDomain
    embedded_space : Domain
    sampling_type  : str

    def __post_init__( self ):

        self.__f_sample = self.__factory_make_sampler( self.sampling_type )

    # Public interface

    def sample( self , npts : int ) -> List[ np.array ] :
        
        """
        sample npts from embedded space
        """
        
        return self.__f_sample( npts )

    # Private interface

    def __factory_make_sampler( self , type_sampler : str ) -> Callable[ [ int ] , List[ np.array ] ] :

        """
        factory to make a sampler
        sampler accepts an int npts and outputs a list of npts-points drawn from embedded space 
        """
        
        factories = {
            'MC_unconstrained'        : self.__make_sampler_mc_unconstrained ,
            'MC_ambientConstrained'   : self.__make_sampler_mc_ambientConstrained , 
            'MCMC_ambientConstrained' : self.__make_sampler_mcmc_ambientConstrained
        }

        return factories[ type_sampler ]()

    def __make_sampler_mc_unconstrained( self ) -> Callable[ [ int ] , List[ np.array ] ]:

        return lambda npts : [ self.embedded_space.f_sample_from_rho() for i in range(npts) ]

    def __make_sampler_mc_ambientConstrained( self , limit : int = 10000 ) -> Callable[ [ int ] , List[ np.array ] ]:
        
        f_unconstrained = self.__make_sampler_mc_unconstrained()

        def f_rejectionSampling( npts ):

            valid_samples = []

            count = 0

            while ( (len(valid_samples) < npts ) & (count < limit) ):

                X_candidate = f_unconstrained( 1 )[0]

                if self.ambient_space.is_in_domain( self.proj_and_lift.get_lifted_coordinates( X_candidate ) ) :

                    valid_samples.append( X_candidate )

                count += 1

            return valid_samples

        return f_rejectionSampling

    def __make_sampler_mcmc_ambientConstrained( self , 
                                                limit : int = 100000 , 
                                                k_sigma_jump : float = 0.5 , 
                                                burn_in : int = 1000 ) -> Callable[ [ int ] , List[ np.array ] ]:

        """
        limit : limit on number of MCMC iterations
        k_sigma_jump : coefficient in front of stddev used in the jump distribution
        burn_in : number of burn-in samples
        """

        f_unconstrained = self.__make_sampler_mc_unconstrained()
        
        f_rho = self.ambient_space.f_rho

        f_jump = lambda x : np.random.normal( x , k_sigma_jump * self.ambient_space.get_average_box_length() )

        def f_rejectionSampling( npts ):

            valid_samples = []
            
            count = 0

            while ( ( True ) & ( count < limit ) ):
                
                x0    = f_unconstrained( 1 )[0]
                rho_0 = f_rho( self.proj_and_lift.get_lifted_coordinates( x0 ) )

                if rho_0 > 1e-4:
                    break
                
                count += 1
            
            count = 0

            while ( (len(valid_samples) < npts ) & (count < limit) ):
                
                X_candidate   = f_jump( x0 )
                rho_candidate = f_rho( self.proj_and_lift.get_lifted_coordinates( X_candidate ) )
                alpha         = rho_candidate / rho_0

                if ( ( alpha > np.random.uniform(0,1) ) & 
                     ( self.ambient_space.is_in_domain( self.proj_and_lift.get_lifted_coordinates( X_candidate ) ) ) & 
                     ( count > burn_in ) ):

                    valid_samples.append( X_candidate )
                    x0 = X_candidate.copy()
                    rho_0 = f_rho( self.proj_and_lift.get_lifted_coordinates( x0 ) )
                    
                count += 1

            return valid_samples

        return f_rejectionSampling