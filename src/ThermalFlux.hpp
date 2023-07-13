#ifndef THERMAL_FLUX_HPP
#define THERMAL_FLUX_HPP

#include "PlatoStaticsTypes.hpp"
#include "materials/MaterialModel.hpp"

namespace Plato
{

/******************************************************************************/
/*! Thermal flux functor.
  
    given a temperature gradient, compute the thermal flux
*/
/******************************************************************************/
template<typename EvaluationType>
class ThermalFlux
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of spatial dimensions
  static constexpr int SpatialDim = ElementType::mNumSpatialDims;
  /// @brief physics response type: linear or nonlinear
  Plato::MaterialModelType mModelType;
  /// @brief expression evaluators for material conductivity tensor
  Plato::TensorFunctor<SpatialDim> mConductivityFunctor;
  /// @brief constant material conductivity tensor
  Plato::TensorConstant<SpatialDim> mConductivityConstant;

public:
  /// @brief class constructor
  /// @param [in] aMaterialModel material constitutive model
  ThermalFlux(
    const std::shared_ptr<Plato::MaterialModel<EvaluationType>> & aMaterialModel
  )
  {
    mModelType = aMaterialModel->type();
    if (mModelType == Plato::MaterialModelType::Nonlinear)
    {
      mConductivityFunctor = aMaterialModel->getTensorFunctor("Thermal Conductivity");
    } else
    if (mModelType == Plato::MaterialModelType::Linear)
    {
      mConductivityConstant = aMaterialModel->getTensorConstant("Thermal Conductivity");
    }
  }

  template<typename TScalarType, typename TGradScalarType, typename TFluxScalarType>
  KOKKOS_INLINE_FUNCTION void
  operator()(
          Plato::Array<SpatialDim, TFluxScalarType> & tflux,
    const Plato::Array<SpatialDim, TGradScalarType> & tgrad,
    const TScalarType                               & temperature
  ) const
  {
    // compute thermal flux
    //
    if (mModelType == Plato::MaterialModelType::Linear)
    {
      for( Plato::OrdinalType iDim=0; iDim<SpatialDim; iDim++){
        tflux(iDim) = 0.0;
        for( Plato::OrdinalType jDim=0; jDim<SpatialDim; jDim++){
          tflux(iDim) -= tgrad(jDim)*mConductivityConstant(iDim, jDim);
        }
      }
    } else
    if (mModelType == Plato::MaterialModelType::Nonlinear)
    {
      TScalarType cellT = temperature;
      for( Plato::OrdinalType iDim=0; iDim<SpatialDim; iDim++){
        tflux(iDim) = 0.0;
        for( Plato::OrdinalType jDim=0; jDim<SpatialDim; jDim++){
          tflux(iDim) -= tgrad(jDim)*mConductivityFunctor(cellT, iDim, jDim);
        }
      }
    }
  }

  template<typename TScalarType, typename TGradScalarType, typename TFluxScalarType>
  KOKKOS_INLINE_FUNCTION void
  operator()( 
    Plato::OrdinalType cellOrdinal,
    Plato::ScalarMultiVectorT<TFluxScalarType> tflux,
    Plato::ScalarMultiVectorT<TGradScalarType> tgrad,
    Plato::ScalarVectorT     <TScalarType>     temperature) 
  const 
  {
    // compute thermal flux
    //
    if (mModelType == Plato::MaterialModelType::Linear)
    {
      for( Plato::OrdinalType iDim=0; iDim<SpatialDim; iDim++){
        tflux(cellOrdinal,iDim) = 0.0;
        for( Plato::OrdinalType jDim=0; jDim<SpatialDim; jDim++){
          tflux(cellOrdinal,iDim) -= tgrad(cellOrdinal,jDim)*mConductivityConstant(iDim, jDim);
        }
      }
    } else
    if (mModelType == Plato::MaterialModelType::Nonlinear)
    {
      TScalarType cellT = temperature(cellOrdinal);
      for( Plato::OrdinalType iDim=0; iDim<SpatialDim; iDim++){
        tflux(cellOrdinal,iDim) = 0.0;
        for( Plato::OrdinalType jDim=0; jDim<SpatialDim; jDim++){
          tflux(cellOrdinal,iDim) -= tgrad(cellOrdinal,jDim)*mConductivityFunctor(cellT, iDim, jDim);
        }
      }
    }
  }

  template<typename TGradScalarType, typename TFluxScalarType>
  KOKKOS_INLINE_FUNCTION void
  operator()( 
    Plato::OrdinalType cellOrdinal,
    Plato::ScalarMultiVectorT<TFluxScalarType> tflux,
    Plato::ScalarMultiVectorT<TGradScalarType> tgrad
  ) const 
  {
    // compute thermal flux
    //
    for( Plato::OrdinalType iDim=0; iDim<SpatialDim; iDim++){
      tflux(cellOrdinal,iDim) = 0.0;
      for( Plato::OrdinalType jDim=0; jDim<SpatialDim; jDim++){
        tflux(cellOrdinal,iDim) -= tgrad(cellOrdinal,jDim)*mConductivityConstant(iDim, jDim);
      }
    }
  }
};
// class ThermalFlux

} // namespace Plato
#endif