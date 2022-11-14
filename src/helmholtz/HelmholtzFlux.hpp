#ifndef HELMHOLTZ_FLUX_HPP
#define HELMHOLTZ_FLUX_HPP

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Helmholtz
{

/******************************************************************************/
/*! Helhomltz flux functor.
  
    given a filtered density gradient, scale by length scale squared
*/
/******************************************************************************/
template<typename ElementType>
class HelmholtzFlux
{
  private:
    Plato::Array<ElementType::mNumSpatialDims> mLengthScale;

  public:

    HelmholtzFlux(const Plato::Array<ElementType::mNumSpatialDims> & aLengthScale) :
      mLengthScale(aLengthScale) {}

    template<typename HGradScalarType, typename HFluxScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
            Plato::OrdinalType                                            aCellOrdinal,
            Plato::Array<ElementType::mNumSpatialDims, HFluxScalarType> & aFlux,
      const Plato::Array<ElementType::mNumSpatialDims, HGradScalarType> & aGrad
    ) const
    {
      // scale filtered density gradient
      //
      for( Plato::OrdinalType iDim=0; iDim<ElementType::mNumSpatialDims; iDim++){
        aFlux(iDim) = mLengthScale(iDim)*mLengthScale(iDim)*aGrad(iDim);
      }
    }
};
// class HelmholtzFlux

} // namespace Helmholtz

} // namespace Plato
#endif
