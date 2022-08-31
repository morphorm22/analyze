#ifndef PLATOMATHHELPERS_HPP_
#define PLATOMATHHELPERS_HPP_

#include <vector>

#include <assert.h>

#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "AnalyzeMacros.hpp"
#include "ApplyProjection.hpp"
#include "BLAS1.hpp"
#include "HyperbolicTangentProjection.hpp"
#include "Mechanics.hpp"
#include "PlatoMathFunctors.hpp"
#include "PlatoMathHelpers.hpp"
#include "Solutions.hpp"
#include "StabilizedMechanics.hpp"
#include "VectorFunctionVMS.hpp"
#include "alg/CrsMatrix.hpp"
#include "alg/PlatoSolverFactory.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
#include "elliptic/VectorFunction.hpp"

#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_LU_Serial_Impl.hpp"
#include "KokkosBatched_Trsm_Decl.hpp"
#include "KokkosBatched_Trsm_Serial_Impl.hpp"

#include "KokkosKernels_SparseUtils.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spadd.hpp"
#include "KokkosSparse_spgemm.hpp"
#include <KokkosKernels_IOUtils.hpp>
#include <Kokkos_Concepts.hpp>

namespace Plato{
namespace TestHelpers {

template <typename DataType>
void setViewFromVector(Plato::ScalarVectorT<DataType> aView,
                       const std::vector<DataType> &aVector) {
  Kokkos::View<const DataType *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      tHostView(aVector.data(), aVector.size());
  Kokkos::deep_copy(aView, tHostView);
}

void setMatrixData(Teuchos::RCP<Plato::CrsMatrixType> aMatrix,
                   const std::vector<Plato::OrdinalType> &aRowMap,
                   const std::vector<Plato::OrdinalType> &aColMap,
                   const std::vector<Plato::Scalar> &aValues);

void sortColumnEntries(
    const Plato::ScalarVectorT<Plato::OrdinalType> &aMatrixRowMap,
    Plato::ScalarVectorT<Plato::OrdinalType> &aMatrixColMap,
    Plato::ScalarVectorT<Plato::Scalar> &aMatrixValues);

void fromFull(Teuchos::RCP<Plato::CrsMatrixType> aOutMatrix,
              const std::vector<std::vector<Plato::Scalar>>& aInMatrix);

void RowSum(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrix,
            Plato::ScalarVector &aOutRowSum);

void InverseMultiply(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrix,
                     Plato::ScalarVector &aInDiagonal);

void SlowDumbRowSummedInverseMultiply(
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
    Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo);

void MatrixMinusEqualsMatrix(
    Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo);

void MatrixMinusEqualsMatrix(
    Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo,
    Plato::OrdinalType aOffset);

void SlowDumbMatrixMinusMatrix(
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo, int aOffset = -1);

void SlowDumbMatrixMatrixMultiply(
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo,
    Teuchos::RCP<Plato::CrsMatrixType> &aOutMatrix);

template <typename DataType>
bool is_same(const Plato::ScalarVectorT<DataType> &aView,
             const std::vector<DataType> &aVec) {
  auto tView_host = Kokkos::create_mirror(aView);
  Kokkos::deep_copy(tView_host, aView);
  for (unsigned int i = 0; i < aVec.size(); ++i) {
    if (tView_host(i) != aVec[i]) {
      return false;
    }
  }
  return true;
}

template <typename DataType>
bool is_same(const Plato::ScalarVectorT<DataType> &aViewA,
             const Plato::ScalarVectorT<DataType> &aViewB) {
  if (aViewA.extent(0) != aViewB.extent(0))
    return false;

  auto tViewA_host = Kokkos::create_mirror(aViewA);
  Kokkos::deep_copy(tViewA_host, aViewA);
  auto tViewB_host = Kokkos::create_mirror(aViewB);
  Kokkos::deep_copy(tViewB_host, aViewB);
  for (unsigned int i = 0; i < aViewA.extent(0); ++i) {
    if (tViewA_host(i) != tViewB_host(i))
      return false;
  }
  return true;
}

bool is_sequential(const Plato::ScalarVectorT<Plato::OrdinalType> &aRowMap,
                   const Plato::ScalarVectorT<Plato::OrdinalType> &aColMap);

bool is_equivalent(const Plato::ScalarVectorT<Plato::OrdinalType> &aRowMap,
                   const Plato::ScalarVectorT<Plato::OrdinalType> &aColMapA,
                   const Plato::ScalarVectorT<Plato::Scalar> &aValuesA,
                   const Plato::ScalarVectorT<Plato::OrdinalType> &aColMapB,
                   const Plato::ScalarVectorT<Plato::Scalar> &aValuesB,
                   Plato::Scalar tolerance = 1.0e-14);

bool is_zero(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrix);

bool is_same(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixA,
             const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixB);

Teuchos::RCP<Plato::CrsMatrixType> createSquareMatrix();

const Teuchos::ParameterList& test_elastostatics_params();

template <typename PhysicsT>
Teuchos::RCP<Plato::VectorFunctionVMS<PhysicsT>>
createStabilizedResidual(const Plato::SpatialModel &aSpatialModel) {
  Plato::DataMap tDataMap;
  return Teuchos::rcp(new Plato::VectorFunctionVMS<PhysicsT>(
      aSpatialModel, tDataMap, test_elastostatics_params(),
      test_elastostatics_params().get<std::string>("PDE Constraint")));
}

template <typename PhysicsT>
Teuchos::RCP<Plato::VectorFunctionVMS<typename PhysicsT::ProjectorT>>
createStabilizedProjector(const Plato::SpatialModel &aSpatialModel) {
  Plato::DataMap tDataMap;
  return Teuchos::rcp(
      new Plato::VectorFunctionVMS<typename PhysicsT::ProjectorT>(
          aSpatialModel, tDataMap, test_elastostatics_params(),
          std::string("State Gradient Projection")));
}

template <typename DataType>
void print_view(const Plato::ScalarVectorT<DataType> &aView) {
  auto tView_host = Kokkos::create_mirror(aView);
  Kokkos::deep_copy(tView_host, aView);
  std::cout << '\n';
  for (unsigned int i = 0; i < aView.extent(0); ++i) {
    std::cout << tView_host(i) << '\n';
  }
}

} // namespace TestHelpers
} // namespace Plato

#endif