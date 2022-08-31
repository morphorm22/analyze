#ifndef PLATOMATHTESTHELPERS_HPP_
#define PLATOMATHTESTHELPERS_HPP_

#include "PlatoStaticsTypes.hpp"

#include <Teuchos_RCPDecl.hpp>

#include <Kokkos_View.hpp>
#include <Kokkos_CopyViews.hpp>

#include <vector>

namespace Plato{
namespace TestHelpers {

template <typename DataType>
void set_view_from_vector(Plato::ScalarVectorT<DataType> aView,
                          const std::vector<DataType> &aVector);

void set_matrix_data(Teuchos::RCP<Plato::CrsMatrixType> aMatrix,
                     const std::vector<Plato::OrdinalType> &aRowMap,
                     const std::vector<Plato::OrdinalType> &aColMap,
                     const std::vector<Plato::Scalar> &aValues);

void from_full(Teuchos::RCP<Plato::CrsMatrixType> aOutMatrix,
               const std::vector<std::vector<Plato::Scalar>>& aInMatrix);

void row_sum(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrix,
             Plato::ScalarVector &aOutRowSum);

void inverse_multiply(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrix,
                      Plato::ScalarVector &aInDiagonal);

void slow_dumb_row_summed_inverse_multiply(
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
    Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo);

void matrix_minus_equals_matrix(
    Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo);

void matrix_minus_equals_matrix(
    Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo,
    Plato::OrdinalType aOffset);

void slow_dumb_matrix_minus_matrix(
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo, int aOffset = -1);

void slow_dumb_matrix_matrix_multiply(
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo,
    Teuchos::RCP<Plato::CrsMatrixType> &aOutMatrix);

template <typename DataType>
bool is_same(const Plato::ScalarVectorT<DataType> &aView,
             const std::vector<DataType> &aVec);

template <typename DataType>
bool is_same(const Plato::ScalarVectorT<DataType> &aViewA,
             const Plato::ScalarVectorT<DataType> &aViewB);

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

template <typename DataType>
void print_view(const Plato::ScalarVectorT<DataType> &aView);

// Implementations
template <typename DataType>
void set_view_from_vector(Plato::ScalarVectorT<DataType> aView,
                       const std::vector<DataType> &aVector) {
  Kokkos::View<const DataType *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      tHostView(aVector.data(), aVector.size());
  Kokkos::deep_copy(aView, tHostView);
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