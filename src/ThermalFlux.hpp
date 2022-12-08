#ifndef THERMAL_FLUX_HPP
#define THERMAL_FLUX_HPP

#include "PlatoStaticsTypes.hpp"
#include "MaterialModel.hpp"

namespace Plato
{

/******************************************************************************/
/*! Thermal flux functor.
  
    given a temperature gradient, compute the thermal flux
*/
/******************************************************************************/
template<typename ElementType>
class ThermalFlux
{
  private:
    Plato::MaterialModelType mModelType;

    Plato::TensorFunctor<ElementType::mNumSpatialDims> mConductivityFunctor;

    Plato::TensorConstant<ElementType::mNumSpatialDims> mConductivityConstant;

  public:

    ThermalFlux(const Plato::MaterialModel<ElementType::mNumSpatialDims>& aMaterialModel)
    {
        mModelType = aMaterialModel.type();
        if (mModelType == Plato::MaterialModelType::Nonlinear)
        {
            mConductivityFunctor = aMaterialModel.getTensorFunctor("Thermal Conductivity");
        } else
        if (mModelType == Plato::MaterialModelType::Linear)
        {
            mConductivityConstant = aMaterialModel.getTensorConstant("Thermal Conductivity");
        }
    }

    template<typename TScalarType, typename TGradScalarType, typename TFluxScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
            Plato::Array<ElementType::mNumSpatialDims, TFluxScalarType> & tflux,
      const Plato::Array<ElementType::mNumSpatialDims, TGradScalarType> & tgrad,
      const TScalarType                                                 & temperature
    ) const
    {

      // compute thermal flux
      //
      if (mModelType == Plato::MaterialModelType::Linear)
      {
        for( Plato::OrdinalType iDim=0; iDim<ElementType::mNumSpatialDims; iDim++){
          tflux(iDim) = 0.0;
          for( Plato::OrdinalType jDim=0; jDim<ElementType::mNumSpatialDims; jDim++){
            tflux(iDim) -= tgrad(jDim)*mConductivityConstant(iDim, jDim);
          }
        }
      } else
      if (mModelType == Plato::MaterialModelType::Nonlinear)
      {
        TScalarType cellT = temperature;
        for( Plato::OrdinalType iDim=0; iDim<ElementType::mNumSpatialDims; iDim++){
          tflux(iDim) = 0.0;
          for( Plato::OrdinalType jDim=0; jDim<ElementType::mNumSpatialDims; jDim++){
            tflux(iDim) -= tgrad(jDim)*mConductivityFunctor(cellT, iDim, jDim);
          }
        }
      }
    }

    template<typename TScalarType, typename TGradScalarType, typename TFluxScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarMultiVectorT<TFluxScalarType> tflux,
                Plato::ScalarMultiVectorT<TGradScalarType> tgrad,
                Plato::ScalarVectorT     <TScalarType>     temperature) const {

      // compute thermal flux
      //
      if (mModelType == Plato::MaterialModelType::Linear)
      {
        for( Plato::OrdinalType iDim=0; iDim<ElementType::mNumSpatialDims; iDim++){
          tflux(cellOrdinal,iDim) = 0.0;
          for( Plato::OrdinalType jDim=0; jDim<ElementType::mNumSpatialDims; jDim++){
            tflux(cellOrdinal,iDim) -= tgrad(cellOrdinal,jDim)*mConductivityConstant(iDim, jDim);
          }
        }
      } else
      if (mModelType == Plato::MaterialModelType::Nonlinear)
      {
        TScalarType cellT = temperature(cellOrdinal);
        for( Plato::OrdinalType iDim=0; iDim<ElementType::mNumSpatialDims; iDim++){
          tflux(cellOrdinal,iDim) = 0.0;
          for( Plato::OrdinalType jDim=0; jDim<ElementType::mNumSpatialDims; jDim++){
            tflux(cellOrdinal,iDim) -= tgrad(cellOrdinal,jDim)*mConductivityFunctor(cellT, iDim, jDim);
          }
        }
      }
    }

    template<typename TGradScalarType, typename TFluxScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarMultiVectorT<TFluxScalarType> tflux,
                Plato::ScalarMultiVectorT<TGradScalarType> tgrad) const {

      // compute thermal flux
      //
      for( Plato::OrdinalType iDim=0; iDim<ElementType::mNumSpatialDims; iDim++){
        tflux(cellOrdinal,iDim) = 0.0;
        for( Plato::OrdinalType jDim=0; jDim<ElementType::mNumSpatialDims; jDim++){
          tflux(cellOrdinal,iDim) -= tgrad(cellOrdinal,jDim)*mConductivityConstant(iDim, jDim);
        }
      }
    }
};
// class ThermalFlux

} // namespace Plato
#endif
