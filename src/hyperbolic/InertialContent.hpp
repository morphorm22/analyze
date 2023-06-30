#pragma once

#include "PlatoStaticsTypes.hpp"
#include "LinearElasticMaterial.hpp"

namespace Plato
{

/******************************************************************************/
/*! Inertial content functor.
  
    given an acceleration vector, compute the inertial content, \rho \bm{a}
*/
/******************************************************************************/
template<typename ElementType>
class InertialContent : public ElementType
{
  private:
    using ElementType::mNumSpatialDims;

    const Plato::Scalar mCellDensity;
    const Plato::Scalar mRayleighA;

  public:
    InertialContent(const Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> aMaterialModel ) :
            mCellDensity (aMaterialModel->getMassDensity()),
            mRayleighA   (aMaterialModel->getRayleighA()) {}

    template<typename TScalarType, typename TContentScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()( Plato::Array<mNumSpatialDims, TContentScalarType> & aContent,
                const Plato::Array<mNumSpatialDims, TScalarType>  & aAcceleration) const {

      // compute inertial content
      //
      for(Plato::OrdinalType tDimIndex = 0; tDimIndex < mNumSpatialDims; tDimIndex++)
      {
          aContent(tDimIndex) = aAcceleration(tDimIndex)*mCellDensity;
      }
    }

    template<typename TVelocityType, typename TAccelerationType, typename TContentScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()( Plato::Array<mNumSpatialDims, TContentScalarType>      & aContent,
                const Plato::Array<mNumSpatialDims, TVelocityType>     & aVelocity,
                const Plato::Array<mNumSpatialDims, TAccelerationType> & aAcceleration) const {

      // compute inertial content
      //
      for(Plato::OrdinalType tDimIndex = 0; tDimIndex < mNumSpatialDims; tDimIndex++)
      {
          aContent(tDimIndex) = aAcceleration(tDimIndex)*mCellDensity
                                            + aVelocity(tDimIndex)*mCellDensity*mRayleighA;
      }
    }
};
// class InertialContent

} // namespace Plato

