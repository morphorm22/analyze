#ifndef SCALAR_PRODUCT_HPP
#define SCALAR_PRODUCT_HPP

#include "PlatoMathTypes.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Scalar product functor.
  
    Given two arrays (either vectors or second rank voigt tensors), compute
    the scalar product.
*/
/******************************************************************************/
template<Plato::OrdinalType NumTerms>
class ScalarProduct
{
  public:

    template<typename ProductScalarType, 
             typename Array1ScalarType, 
             typename Array2ScalarType,
             typename VolumeScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType                          aCellOrdinal,
                Plato::ScalarVectorT<ProductScalarType>     aScalarProduct,
                Plato::ScalarMultiVectorT<Array1ScalarType> aArray1,
                Plato::ScalarMultiVectorT<Array2ScalarType> aArray2,
                Plato::ScalarVectorT<VolumeScalarType>      aCellVolume,
                Plato::Scalar                               aScale = 1.0 ) const {

      // compute scalar product
      //
      ProductScalarType tInc(0.0);
      for( Plato::OrdinalType iTerm=0; iTerm<NumTerms; iTerm++){
        tInc += aArray1(aCellOrdinal,iTerm)*aArray2(aCellOrdinal,iTerm);
      }
      aScalarProduct(aCellOrdinal) += aScale*tInc*aCellVolume(aCellOrdinal);
    }

    template<typename ProductScalarType, 
             typename Array1ScalarType, 
             typename Array2ScalarType,
             typename VolumeScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType                         aCellOrdinal,
                Plato::ScalarVectorT<ProductScalarType>    aScalarProduct,
          const Plato::Array<NumTerms, Array1ScalarType> & aArray1,
          const Plato::Array<NumTerms, Array2ScalarType> & aArray2,
                VolumeScalarType                           aCellVolume,
                Plato::Scalar                              aScale = 1.0
    ) const
    {
      // compute scalar product
      //
      ProductScalarType tInc(0.0);
      for( Plato::OrdinalType iTerm=0; iTerm<NumTerms; iTerm++){
        tInc += aArray1(iTerm)*aArray2(iTerm);
      }
      Kokkos::atomic_add(&aScalarProduct(aCellOrdinal), aScale*tInc*aCellVolume);
    }

    template<typename ProductScalarType, 
             typename Array1ScalarType, 
             typename VolumeScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType aCellOrdinal,
                Plato::ScalarVectorT<ProductScalarType>     aScalarProduct,
                Plato::ScalarMultiVectorT<Array1ScalarType> aArray1,
                Plato::Array<NumTerms>                      aArray2,
                Plato::ScalarVectorT<VolumeScalarType>      aCellVolume,
                Plato::Scalar                               aScale = 1.0 ) const {

      // compute scalar product
      //
      ProductScalarType tInc(0.0);
      for( Plato::OrdinalType iTerm=0; iTerm<NumTerms; iTerm++){
        tInc += aArray1(aCellOrdinal,iTerm)*aArray2[iTerm];
      }
      aScalarProduct(aCellOrdinal) += aScale*tInc*aCellVolume(aCellOrdinal);
    }
};

} // namespace Plato

#endif
