#include "PlatoMathTestHelpers.hpp"
#include "PlatoTestHelpers.hpp"

#include "PlatoMathFunctors.hpp"

#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_LU_Serial_Impl.hpp>
#include <KokkosBatched_Trsm_Decl.hpp>
#include <KokkosBatched_Trsm_Serial_Impl.hpp>

#include <KokkosKernels_SparseUtils.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spadd.hpp>
#include <KokkosSparse_spgemm.hpp>
#include <KokkosKernels_IOUtils.hpp>
#include <Kokkos_Concepts.hpp>

#include <assert.h>
#include <vector>

namespace Plato {
namespace TestHelpers {

using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

void set_matrix_data(Teuchos::RCP<Plato::CrsMatrixType> aMatrix,
                   const std::vector<Plato::OrdinalType>& aRowMap,
                   const std::vector<Plato::OrdinalType>& aColMap,
                   const std::vector<Plato::Scalar>& aValues) {
  Plato::ScalarVectorT<Plato::OrdinalType> tRowMap("row map", aRowMap.size());
  set_view_from_vector(tRowMap, aRowMap);
  aMatrix->setRowMap(tRowMap);

  Plato::ScalarVectorT<Plato::OrdinalType> tColMap("col map", aColMap.size());
  set_view_from_vector(tColMap, aColMap);
  aMatrix->setColumnIndices(tColMap);

  Plato::ScalarVectorT<Plato::Scalar> tValues("values", aValues.size());
  set_view_from_vector(tValues, aValues);
  aMatrix->setEntries(tValues);
}

void from_full(Teuchos::RCP<Plato::CrsMatrixType> aOutMatrix,
              const std::vector<std::vector<Plato::Scalar>>& aInMatrix) {
  using Plato::OrdinalType;
  using Plato::Scalar;

  if (aOutMatrix->numRows() != aInMatrix.size()) {
    ANALYZE_THROWERR("matrices have incompatible shapes");
  }
  if (aOutMatrix->numCols() != aInMatrix[0].size()) {
    ANALYZE_THROWERR("matrices have incompatible shapes");
  }

  const auto tNumRowsPerBlock = aOutMatrix->numRowsPerBlock();
  const auto tNumColsPerBlock = aOutMatrix->numColsPerBlock();
  const auto tNumBlockRows = aOutMatrix->numRows() / tNumRowsPerBlock;
  const auto tNumBlockCols = aOutMatrix->numCols() / tNumColsPerBlock;

  std::vector<OrdinalType> tBlockRowMap(tNumBlockRows + 1);

  tBlockRowMap[0] = 0;
  std::vector<OrdinalType> tColumnIndices;
  std::vector<Scalar> tBlockEntries;
  for (OrdinalType iBlockRowIndex = 0; iBlockRowIndex < tNumBlockRows;
       iBlockRowIndex++) {
    for (OrdinalType iBlockColIndex = 0; iBlockColIndex < tNumBlockCols;
         iBlockColIndex++) {
      bool blockIsNonZero = false;
      std::vector<Scalar> tLocalEntries;
      for (OrdinalType iLocalBlockRowIndex = 0;
           iLocalBlockRowIndex < tNumRowsPerBlock; iLocalBlockRowIndex++) {
        for (OrdinalType iLocalBlockColIndex = 0;
             iLocalBlockColIndex < tNumColsPerBlock; iLocalBlockColIndex++) {
          auto tMatrixRow =
              iBlockRowIndex * tNumRowsPerBlock + iLocalBlockRowIndex;
          auto tMatrixCol =
              iBlockColIndex * tNumColsPerBlock + iLocalBlockColIndex;
          tLocalEntries.push_back(aInMatrix[tMatrixRow][tMatrixCol]);
          if (aInMatrix[tMatrixRow][tMatrixCol] != 0.0)
            blockIsNonZero = true;
        }
      }
      if (blockIsNonZero) {
        tColumnIndices.push_back(iBlockColIndex);
        tBlockEntries.insert(tBlockEntries.end(), tLocalEntries.begin(),
                             tLocalEntries.end());
      }
    }
    tBlockRowMap[iBlockRowIndex + 1] = tColumnIndices.size();
  }

  set_matrix_data(aOutMatrix, tBlockRowMap, tColumnIndices, tBlockEntries);
}

void row_sum(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrix,
             Plato::ScalarVector &aOutRowSum) {
  auto tNumBlockRows = aInMatrix->rowMap().size() - 1;
  auto tNumRows = aInMatrix->numRows();
  Plato::RowSum tRowSumFunctor(aInMatrix);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, tNumBlockRows),
      KOKKOS_LAMBDA(const Plato::OrdinalType &tBlockRowOrdinal) {
        tRowSumFunctor(tBlockRowOrdinal, aOutRowSum);
      });
}

void inverse_multiply(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrix,
                     Plato::ScalarVector &aInDiagonal) {
  auto tNumBlockRows = aInMatrix->rowMap().size() - 1;
  Plato::DiagonalInverseMultiply tDiagInverseMultiplyFunctor(aInMatrix);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, tNumBlockRows),
      KOKKOS_LAMBDA(const Plato::OrdinalType &tBlockRowOrdinal) {
        tDiagInverseMultiplyFunctor(tBlockRowOrdinal, aInDiagonal);
      });
}

void slow_dumb_row_summed_inverse_multiply(
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
    Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo) {
  auto F1 = to_full(aInMatrixOne);
  auto F2 = to_full(aInMatrixTwo);

  auto tNumM1Rows = aInMatrixOne->numRows();
  auto tNumM1Cols = aInMatrixOne->numCols();
  if (tNumM1Rows != tNumM1Cols) {
    ANALYZE_THROWERR("matrix one must be square");
  }

  auto tNumM2Rows = aInMatrixTwo->numRows();
  auto tNumM2Cols = aInMatrixTwo->numCols();
  if (tNumM1Cols != tNumM2Rows) {
    ANALYZE_THROWERR("matrices have incompatible shapes");
  }

  for (auto iRow = 0; iRow < tNumM1Rows; iRow++) {
    Plato::Scalar tRowSum = 0.0;
    for (auto iCol = 0; iCol < tNumM1Cols; iCol++) {
      tRowSum += F1[iRow][iCol];
    }
    for (auto iCol = 0; iCol < tNumM2Cols; iCol++) {
      F2[iRow][iCol] /= tRowSum;
    }
  }
  from_full(aInMatrixTwo, F2);
}

void matrix_minus_equals_matrix(
    Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo) {
  auto tEntriesOne = aInMatrixOne->entries();
  auto tEntriesTwo = aInMatrixTwo->entries();
  auto tNumEntries = tEntriesOne.extent(0);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, tNumEntries),
      KOKKOS_LAMBDA(const Plato::OrdinalType &tEntryOrdinal) {
        tEntriesOne(tEntryOrdinal) -= tEntriesTwo(tEntryOrdinal);
      });
}

void matrix_minus_equals_matrix(
    Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo,
    Plato::OrdinalType aOffset) {
  auto tRowMap = aInMatrixOne->rowMap();
  auto tNumBlockRows = tRowMap.size() - 1;

  auto tFromNumRowsPerBlock = aInMatrixOne->numRowsPerBlock();
  auto tFromNumColsPerBlock = aInMatrixOne->numColsPerBlock();
  auto tToNumRowsPerBlock = aInMatrixTwo->numRowsPerBlock();
  auto tToNumColsPerBlock = aInMatrixTwo->numColsPerBlock();

  assert(tToNumColsPerBlock == tFromNumColsPerBlock);

  auto tEntriesOne = aInMatrixOne->entries();
  auto tEntriesTwo = aInMatrixTwo->entries();
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, tNumBlockRows),
      KOKKOS_LAMBDA(const Plato::OrdinalType &tBlockRowOrdinal) {
        auto tFrom = tRowMap(tBlockRowOrdinal);
        auto tTo = tRowMap(tBlockRowOrdinal + 1);
        for (auto tColMapIndex = tFrom; tColMapIndex < tTo; ++tColMapIndex) {
          auto tFromEntryOffset =
              tColMapIndex * tFromNumRowsPerBlock * tFromNumColsPerBlock;
          auto tToEntryOffset =
              tColMapIndex * tToNumRowsPerBlock * tToNumColsPerBlock +
              aOffset * tToNumColsPerBlock;
          ;
          for (Plato::OrdinalType tBlockColOrdinal = 0;
               tBlockColOrdinal < tToNumColsPerBlock; ++tBlockColOrdinal) {
            auto tToEntryIndex = tToEntryOffset + tBlockColOrdinal;
            auto tFromEntryIndex = tFromEntryOffset + tBlockColOrdinal;
            tEntriesOne[tToEntryIndex] -= tEntriesTwo[tFromEntryIndex];
          }
        }
      });
}

void slow_dumb_matrix_minus_matrix(
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo, const int aOffset) {
  const auto tNumM1Rows = aInMatrixOne->numRows();
  const auto tNumM1Cols = aInMatrixOne->numCols();
  const auto tNumM2Rows = aInMatrixTwo->numRows();
  const auto tNumM2Cols = aInMatrixTwo->numCols();
  const auto tNumM1RowsPerBlock = aInMatrixOne->numRowsPerBlock();
  const auto tNumM2RowsPerBlock = aInMatrixTwo->numRowsPerBlock();

  if (aOffset == -1) {
    if (tNumM1Rows != tNumM2Rows || tNumM1Cols != tNumM2Cols) {
      ANALYZE_THROWERR("input matrices have incompatible shapes");
    }
  } else {
    if (tNumM2RowsPerBlock != 1 || tNumM1Cols != tNumM2Cols) {
      ANALYZE_THROWERR("input matrices have incompatible shapes");
    }
  }

  using Plato::Scalar;
  std::vector<std::vector<Scalar>> tFullMatrix(
      tNumM1Rows, std::vector<Scalar>(tNumM1Cols, 0.0));

  auto F1 = to_full(aInMatrixOne);
  auto F2 = to_full(aInMatrixTwo);

  if (aOffset == -1) {
    for (auto iRow = 0; iRow < tNumM1Rows; iRow++) {
      for (auto iCol = 0; iCol < tNumM1Cols; iCol++) {
        tFullMatrix[iRow][iCol] = F1[iRow][iCol] - F2[iRow][iCol];
      }
    }
  } else {
    for (auto iRow = 0; iRow < tNumM1Rows; iRow++) {
      for (auto iCol = 0; iCol < tNumM1Cols; iCol++) {
        tFullMatrix[iRow][iCol] = F1[iRow][iCol];
      }
    }
    for (auto iRow = 0; iRow < tNumM2Rows; iRow++) {
      for (auto iCol = 0; iCol < tNumM2Cols; iCol++) {
        tFullMatrix[tNumM1RowsPerBlock * iRow + aOffset][iCol] -=
            F2[iRow][iCol];
      }
    }
  }

  from_full(aInMatrixOne, tFullMatrix);
}

void slow_dumb_matrix_matrix_multiply(
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
    const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo,
    Teuchos::RCP<Plato::CrsMatrixType> &aOutMatrix) {
  auto tNumOutMatrixRows = aInMatrixOne->numRows();
  auto tNumOutMatrixCols = aInMatrixTwo->numCols();

  if (aInMatrixOne->numCols() != aInMatrixTwo->numRows()) {
    ANALYZE_THROWERR("input matrices have incompatible shapes");
  }

  auto tNumInner = aInMatrixOne->numCols();

  using Plato::Scalar;
  std::vector<std::vector<Scalar>> tFullMatrix(
      tNumOutMatrixRows, std::vector<Scalar>(tNumOutMatrixCols, 0.0));

  auto F1 = to_full(aInMatrixOne);
  auto F2 = to_full(aInMatrixTwo);

  for (auto iRow = 0; iRow < tNumOutMatrixRows; iRow++) {
    for (auto iCol = 0; iCol < tNumOutMatrixCols; iCol++) {
      tFullMatrix[iRow][iCol] = 0.0;
      for (auto iK = 0; iK < tNumInner; iK++) {
        tFullMatrix[iRow][iCol] += F1[iRow][iK] * F2[iK][iCol];
      }
    }
  }

  from_full(aOutMatrix, tFullMatrix);
}

bool is_sequential(const Plato::ScalarVectorT<Plato::OrdinalType> &aRowMap,
                   const Plato::ScalarVectorT<Plato::OrdinalType> &aColMap) {
  auto tRowMap = get(aRowMap);
  auto tColMap = get(aColMap);

  auto tNumRows = tRowMap.extent(0) - 1;

  for (unsigned int i = 0; i < tNumRows; ++i) {
    auto tFrom = tRowMap(i);
    auto tTo = tRowMap(i + 1);
    for (auto iColMapEntry = tFrom; iColMapEntry < tTo - 1; iColMapEntry++) {
      if (tColMap(iColMapEntry) >= tColMap(iColMapEntry + 1)) {
        return false;
      }
    }
  }
  return true;
}
bool is_equivalent(const Plato::ScalarVectorT<Plato::OrdinalType> &aRowMap,
                   const Plato::ScalarVectorT<Plato::OrdinalType> &aColMapA,
                   const Plato::ScalarVectorT<Plato::Scalar> &aValuesA,
                   const Plato::ScalarVectorT<Plato::OrdinalType> &aColMapB,
                   const Plato::ScalarVectorT<Plato::Scalar> &aValuesB,
                   Plato::Scalar tolerance) {
  if (aColMapA.extent(0) != aColMapB.extent(0))
    return false;
  if (aValuesA.extent(0) != aValuesB.extent(0))
    return false;

  auto tRowMap = get(aRowMap);
  auto tColMapA = get(aColMapA);
  auto tValuesA = get(aValuesA);
  auto tColMapB = get(aColMapB);
  auto tValuesB = get(aValuesB);

  Plato::OrdinalType tANumEntriesPerBlock =
      aValuesA.extent(0) / aColMapA.extent(0);
  Plato::OrdinalType tBNumEntriesPerBlock =
      aValuesB.extent(0) / aColMapB.extent(0);
  if (tANumEntriesPerBlock != tBNumEntriesPerBlock)
    return false;

  auto tNumRows = tRowMap.extent(0) - 1;
  for (unsigned int i = 0; i < tNumRows; ++i) {
    auto tFrom = tRowMap(i);
    auto tTo = tRowMap(i + 1);
    for (auto iColMapEntryA = tFrom; iColMapEntryA < tTo; iColMapEntryA++) {
      auto tColumnIndexA = tColMapA(iColMapEntryA);
      for (auto iColMapEntryB = tFrom; iColMapEntryB < tTo; iColMapEntryB++) {
        if (tColumnIndexA == tColMapB(iColMapEntryB)) {
          for (auto iBlockEntry = 0; iBlockEntry < tANumEntriesPerBlock;
               iBlockEntry++) {
            auto tBlockEntryIndexA =
                iColMapEntryA * tANumEntriesPerBlock + iBlockEntry;
            auto tBlockEntryIndexB =
                iColMapEntryB * tBNumEntriesPerBlock + iBlockEntry;
            Plato::Scalar tSum = fabs(tValuesA(tBlockEntryIndexA)) +
                                 fabs(tValuesB(tBlockEntryIndexB));
            Plato::Scalar tDif =
                fabs(tValuesA(tBlockEntryIndexA) - tValuesB(tBlockEntryIndexB));
            Plato::Scalar tRelVal = (tSum != 0.0) ? 2.0 * tDif / tSum : 0.0;
            if (tRelVal > tolerance) {
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

bool is_zero(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrix) {
  auto tEntries = aInMatrix->entries();
  auto tEntries_host = Kokkos::create_mirror(tEntries);
  Kokkos::deep_copy(tEntries_host, tEntries);
  for (unsigned int i = 0; i < tEntries_host.extent(0); ++i) {
    if (tEntries_host(i) != 0.0)
      return false;
  }
  return true;
}

bool is_same(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixA,
             const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixB) {
  if (!is_same(aInMatrixA->rowMap(), aInMatrixB->rowMap())) {
    return false;
  }
  if (!is_same(aInMatrixA->columnIndices(), aInMatrixB->columnIndices())) {
    return false;
  }
  if (!is_same(aInMatrixA->entries(), aInMatrixB->entries())) {
    return false;
  }
  return true;
}

} // namespace TestHelpers
} // namespace Plato
