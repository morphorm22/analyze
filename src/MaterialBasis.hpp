#pragma once

#include "VoigtUtils.hpp"
#include "SpatialModel.hpp"
#include "MaterialModel.hpp"

namespace Plato {

  template<int SpatialDim>
  class UniformMaterialBasis
  {
    const Plato::Matrix<SpatialDim,SpatialDim> mBasis;
    public:
      UniformMaterialBasis(
        Plato::Matrix<SpatialDim,SpatialDim> const & aBasis
    ) : mBasis(aBasis) { }

    /******************************************************************************
     * \brief Compute voigt tensor component in the material basis
     * \param [in/out] aVoigtTensor On input, voigt components in global basis. On
                       output, voigt components in material basis.

           [ix, jx, kx] 
       B = |iy, jy, ky| 
           [iz, jz, kz]

     where i={ix,iy,yz}, j={jx,jy,jz}, and k={kx,ky,kz} are the user provided
     material basis vectors.  The components, v, of a vector in the global frame are
     computed from the components, v', of the vector in the material frame by:

     v = B v'

     B is required to be orthonormal, so B^(-1) == B^T and:

     v' = B^T v

     Assume the vector, v, is operated on by a tensor, S, to yield a vector, w:

     w = S v.
     
     Define S' to be the tensor that produces the same operation in the material 
     basis:

     w' = S' v'.

     Substuting the change of basis:

     B^T w = S' B^T v
     w = B S' B^T v

     and by comparison,

     S = B S' B^T and S' = B^T S B.
     ******************************************************************************/
    template<typename T>
    void
    VoigtTensorToMaterialBasis(Plato::ScalarArray3DT<T> aVoigtTensor)
    {
      auto tNumCells = aVoigtTensor.extent(0);
      auto tNumPoints = aVoigtTensor.extent(1);
      auto& tBasis = mBasis;
      Kokkos::parallel_for("to material basis", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        Plato::Matrix<SpatialDim,SpatialDim,T> tTensor = Plato::FromVoigt<SpatialDim,T>(aVoigtTensor, iCellOrdinal, iGpOrdinal);
        Plato::Matrix<SpatialDim,SpatialDim,T> tToTensor(0.0);
        for(int i=0; i<SpatialDim; i++) {
          for(int j=0; j<SpatialDim; j++) {
            for(int k=0; k<SpatialDim; k++) {
              for(int l=0; l<SpatialDim; l++) {
                tToTensor(i,l) += tBasis(j,i) * tTensor(j,k) * tBasis(k,l); // S' = B^T S B
              }
            }
          }
        }
        Plato::ToVoigt<SpatialDim,T>(tToTensor, aVoigtTensor, iCellOrdinal, iGpOrdinal);
      });
    }
    template<typename T>
    void
    VoigtTensorFromMaterialBasis(Plato::ScalarArray3DT<T> aVoigtTensor)
    {
      auto tNumCells = aVoigtTensor.extent(0);
      auto tNumPoints = aVoigtTensor.extent(1);
      auto& tBasis = mBasis;
      Kokkos::parallel_for("from material basis", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        Plato::Matrix<SpatialDim,SpatialDim,T> tTensor = Plato::FromVoigt<SpatialDim,T>(aVoigtTensor, iCellOrdinal, iGpOrdinal);
        Plato::Matrix<SpatialDim,SpatialDim,T> tFromTensor(0.0);
        for(int i=0; i<SpatialDim; i++) {
          for(int j=0; j<SpatialDim; j++) {
            for(int k=0; k<SpatialDim; k++) {
              for(int l=0; l<SpatialDim; l++) {
                tFromTensor(i,l) += tBasis(i,j) * tTensor(j,k) * tBasis(l,k); // S = B S' B^T
              }
            }
          }
        }
        Plato::ToVoigt<SpatialDim,T>(tFromTensor, aVoigtTensor, iCellOrdinal, iGpOrdinal);
      });
    }

    /******************************************************************************
     * \brief Compute vector component in the material basis
     * \param [in/out] aVector On input, cartesian components in global basis. On
                       output, components in material basis.

           [ix, jx, kx] 
       B = |iy, jy, ky| 
           [iz, jz, kz]

     where i={ix,iy,yz}, j={jx,jy,jz}, and k={kx,ky,kz} are the user provided
     material basis vectors.  The components, v, of a vector in the global frame are
     computed from the components, v', of the vector in the material frame by:

     v = B v'

     B is required to be orthonormal, so B^(-1) == B^T and:

     v' = B^T v
     ******************************************************************************/
    template<typename T>
    void
    VectorToMaterialBasis(Plato::ScalarArray3DT<T> aVector)
    {
      auto tNumCells = aVector.extent(0);
      auto tNumPoints = aVector.extent(1);
      auto& tBasis = mBasis;
      Kokkos::parallel_for("to material basis", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        Plato::Array<SpatialDim,T> tInMatBasis(0.0);
        for(int i=0; i<SpatialDim; i++)
        {
          for(int j=0; j<SpatialDim; j++)
          {
            tInMatBasis(i) += tBasis(j,i) * aVector(iCellOrdinal, iGpOrdinal, j); // v' = B^T v
          }
        }
        for(int i=0; i<SpatialDim; i++)
        {
          aVector(iCellOrdinal, iGpOrdinal, i) = tInMatBasis(i);
        }
      });
    }
    template<typename T>
    void
    VectorFromMaterialBasis(Plato::ScalarArray3DT<T> aVector)
    {
      auto tNumCells = aVector.extent(0);
      auto tNumPoints = aVector.extent(1);
      auto& tBasis = mBasis;
      Kokkos::parallel_for("to material basis", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        Plato::Array<SpatialDim,T> tInGlobalBasis(0.0);
        for(int i=0; i<SpatialDim; i++)
        {
          for(int j=0; j<SpatialDim; j++)
          {
            tInGlobalBasis(i) += tBasis(i,j) * aVector(iCellOrdinal, iGpOrdinal, j); // v = B v'
          }
        }
        for(int i=0; i<SpatialDim; i++)
        {
          aVector(iCellOrdinal, iGpOrdinal, i) = tInGlobalBasis(i);
        }
      });
    }
  };

  template<int SpatialDim>
  class VaryingMaterialBasis
  {
    const Plato::ScalarArray3D mBasis;
    public:
      VaryingMaterialBasis(
        Plato::ScalarArray3D aBasis
    ) : mBasis(aBasis) { }

    /******************************************************************************
     * \brief Compute voigt tensor component in the material basis
     * \param [in/out] aVoigtTensor On input, voigt components in global basis. On
                       output, voigt components in material basis.

           [ix, jx, kx] 
       B = |iy, jy, ky| 
           [iz, jz, kz]

     where i={ix,iy,yz}, j={jx,jy,jz}, and k={kx,ky,kz} are the user provided
     material basis vectors.  The components, v, of a vector in the global frame are
     computed from the components, v', of the vector in the material frame by:

     v = B v'

     B is required to be orthonormal, so B^(-1) == B^T and:

     v' = B^T v

     Assume the vector, v, is operated on by a tensor, S, to yield a vector, w:

     w = S v.
     
     Define S' to be the tensor that produces the same operation in the material 
     basis:

     w' = S' v'.

     Substuting the change of basis:

     B^T w = S' B^T v
     w = B S' B^T v

     and by comparison,

     S = B S' B^T and S' = B^T S B.
     ******************************************************************************/
    template<typename T>
    void
    VoigtTensorToMaterialBasis(Plato::ScalarArray3DT<T> aVoigtTensor)
    {
      auto tNumCells = aVoigtTensor.extent(0);
      auto tNumPoints = aVoigtTensor.extent(1);
      auto& tBasis = mBasis;
      Kokkos::parallel_for("to material basis", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        Plato::Matrix<SpatialDim,SpatialDim,T> tTensor = Plato::FromVoigt<SpatialDim,T>(aVoigtTensor, iCellOrdinal, iGpOrdinal);
        Plato::Matrix<SpatialDim,SpatialDim,T> tToTensor(0.0);
        for(int i=0; i<SpatialDim; i++) {
          for(int j=0; j<SpatialDim; j++) {
            for(int k=0; k<SpatialDim; k++) {
              for(int l=0; l<SpatialDim; l++) {
                tToTensor(i,l) += tBasis(iCellOrdinal,j,i) * tTensor(j,k) * tBasis(iCellOrdinal,k,l); // S' = B^T S B
              }
            }
          }
        }
        Plato::ToVoigt<SpatialDim,T>(tToTensor, aVoigtTensor, iCellOrdinal, iGpOrdinal);
      });
    }
    template<typename T>
    void
    VoigtTensorFromMaterialBasis(Plato::ScalarArray3DT<T> aVoigtTensor)
    {
      auto tNumCells = aVoigtTensor.extent(0);
      auto tNumPoints = aVoigtTensor.extent(1);
      auto& tBasis = mBasis;
      Kokkos::parallel_for("from material basis", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        Plato::Matrix<SpatialDim,SpatialDim,T> tTensor = Plato::FromVoigt<SpatialDim,T>(aVoigtTensor, iCellOrdinal, iGpOrdinal);
        Plato::Matrix<SpatialDim,SpatialDim,T> tFromTensor(0.0);
        for(int i=0; i<SpatialDim; i++) {
          for(int j=0; j<SpatialDim; j++) {
            for(int k=0; k<SpatialDim; k++) {
              for(int l=0; l<SpatialDim; l++) {
                tFromTensor(i,l) += tBasis(iCellOrdinal,i,j) * tTensor(j,k) * tBasis(iCellOrdinal,l,k); // S = B S' B^T
              }
            }
          }
        }
        Plato::ToVoigt<SpatialDim,T>(tFromTensor, aVoigtTensor, iCellOrdinal, iGpOrdinal);
      });
    }

    /******************************************************************************
     * \brief Compute vector component in the material basis
     * \param [in/out] aVector On input, cartesian components in global basis. On
                       output, components in material basis.

           [ix, jx, kx] 
       B = |iy, jy, ky| 
           [iz, jz, kz]

     where i={ix,iy,yz}, j={jx,jy,jz}, and k={kx,ky,kz} are the user provided
     material basis vectors.  The components, v, of a vector in the global frame are
     computed from the components, v', of the vector in the material frame by:

     v = B v'

     B is required to be orthonormal, so B^(-1) == B^T and:

     v' = B^T v
     ******************************************************************************/
    template<typename T>
    void
    VectorToMaterialBasis(Plato::ScalarArray3DT<T> aVector)
    {
      auto tNumCells = aVector.extent(0);
      auto tNumPoints = aVector.extent(1);
      auto& tBasis = mBasis;
      Kokkos::parallel_for("to material basis", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        Plato::Array<SpatialDim,T> tInMatBasis(0.0);
        for(int i=0; i<SpatialDim; i++)
        {
          for(int j=0; j<SpatialDim; j++)
          {
            tInMatBasis(i) += tBasis(iCellOrdinal,j,i) * aVector(iCellOrdinal, iGpOrdinal, j); // v' = B^T v
          }
        }
        for(int i=0; i<SpatialDim; i++)
        {
          aVector(iCellOrdinal, iGpOrdinal, i) = tInMatBasis(i);
        }
      });
    }
    template<typename T>
    void
    VectorFromMaterialBasis(Plato::ScalarArray3DT<T> aVector)
    {
      auto tNumCells = aVector.extent(0);
      auto tNumPoints = aVector.extent(1);
      auto& tBasis = mBasis;
      Kokkos::parallel_for("to material basis", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        Plato::Array<SpatialDim,T> tInGlobalBasis(0.0);
        for(int i=0; i<SpatialDim; i++)
        {
          for(int j=0; j<SpatialDim; j++)
          {
            tInGlobalBasis(i) += tBasis(iCellOrdinal,i,j) * aVector(iCellOrdinal, iGpOrdinal, j); // v = B v'
          }
        }
        for(int i=0; i<SpatialDim; i++)
        {
          aVector(iCellOrdinal, iGpOrdinal, i) = tInGlobalBasis(i);
        }
      });
    }
  };

  class UniformMaterialBasisFactory
  {
    public:

    template <int SpatialDim>
    std::shared_ptr<Plato::UniformMaterialBasis<SpatialDim>>
    create(
      Teuchos::RCP<Plato::MaterialModel<SpatialDim>> const   aMaterialModel,
      Plato::SpatialDomain                           const & aSpatialDomain
    )
    {
      bool tHasMaterialBasis = aMaterialModel->hasCartesianBasis();
      bool tHasBlockBasis = aSpatialDomain.hasUniformCartesianBasis();

      if( !tHasBlockBasis && !tHasMaterialBasis )
      {
        return nullptr;
      }

      Plato::Matrix<SpatialDim,SpatialDim> tUniformBasis;
      if( tHasMaterialBasis && tHasBlockBasis )
      {
        auto tMaterialBasis = aMaterialModel->getCartesianBasis();

        Plato::Matrix<SpatialDim,SpatialDim> tBlockBasis;
        aSpatialDomain.getUniformCartesianBasis(tBlockBasis);

        for(int i=0; i<SpatialDim; i++){
          for(int j=0; j<SpatialDim; j++){
            tUniformBasis(i,j) = 0.0;
            for(int k=0; k<SpatialDim; k++){
              tUniformBasis(i,j) += tBlockBasis(i,k)*tMaterialBasis(k,j);
            }
          }
        }
      }
      else 
      if( tHasMaterialBasis )
      {
        tUniformBasis = aMaterialModel->getCartesianBasis();
      }
      else 
      if( tHasBlockBasis )
      {
        aSpatialDomain.getUniformCartesianBasis(tUniformBasis);
      }

      return std::make_shared<UniformMaterialBasis<SpatialDim>>(tUniformBasis);
    }
  };

  class VaryingMaterialBasisFactory
  {
    public:

    template <int SpatialDim>
    std::shared_ptr<Plato::VaryingMaterialBasis<SpatialDim>>
    create(
      Plato::DataMap       const & aDataMap,
      Plato::SpatialDomain const & aSpatialDomain
    )
    {
      bool tHasBasis = aSpatialDomain.hasVaryingCartesianBasis();
      if(!tHasBasis)
      {
        return nullptr;
      }

      auto tBasis = aSpatialDomain.getVaryingCartesianBasis();
      return std::make_shared<VaryingMaterialBasis<SpatialDim>>(tBasis);
    }
  };

} // namespace Plato
