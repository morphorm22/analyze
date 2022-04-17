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
    Plato::Scalar mLengthScale;

  public:

    HelmholtzFlux(const Plato::Scalar aLengthScale) {
      mLengthScale = aLengthScale;
    }

    template<typename HGradScalarType, typename HFluxScalarType>
    DEVICE_TYPE inline void
    operator()(
            Plato::OrdinalType                                            aCellOrdinal,
            Plato::Array<ElementType::mNumSpatialDims, HFluxScalarType> & aFlux,
      const Plato::Array<ElementType::mNumSpatialDims, HGradScalarType> & aGrad
    ) const
    {
      // scale filtered density gradient
      //
      Plato::Scalar tLengthScaleSquared = mLengthScale*mLengthScale;

      for( Plato::OrdinalType iDim=0; iDim<ElementType::mNumSpatialDims; iDim++){
        aFlux(iDim) = tLengthScaleSquared*aGrad(iDim);
      }
    }
};
// class HelmholtzFlux

} // namespace Helmholtz

} // namespace Plato
#endif
