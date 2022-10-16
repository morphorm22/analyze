#pragma once

#include "PlatoStaticsTypes.hpp"

#include <Teuchos_RCP.hpp>

#include <KokkosSparse_spgemm.hpp>

namespace Plato
{

/******************************************************************************//**
* \brief Determine the max of the arguments aA and aB
* \return maximum value
**********************************************************************************/
template <typename T>
KOKKOS_INLINE_FUNCTION T max2(T aA, T aB)
{
  return (aA < aB) ? (aB) : (aA);
}

/******************************************************************************//**
* \brief Determine the min of the arguments aA and aB
* \return minimum value
**********************************************************************************/
template <typename T>
KOKKOS_INLINE_FUNCTION T min2(T aA, T aB)
{
  return (aA > aB) ? (aB) : (aA);
}

/******************************************************************************//**
 * \brief Device only function used to compare two values (conditional values)
 * between themselves and return the decision (consequent value). The conditional
 * expression evaluated in this function is defined as if(X > Y) A = B.
 * \param [in] aConditionalValOne conditional value given by X
 * \param [in] aConditionalValTwo conditional value given by Y
 * \param [in] aConsequentValOne consequent value given by A
 * \param [in] aConsequentValTwo consequent value given by B
 * \return result/decision
**********************************************************************************/
KOKKOS_INLINE_FUNCTION Plato::Scalar
conditional_expression(const Plato::Scalar & aX,
                       const Plato::Scalar & aY,
                       const Plato::Scalar & aA,
                       const Plato::Scalar & aB)
{
    auto tConditionalExpression = aX - aY - static_cast<Plato::Scalar>(1.0);
    tConditionalExpression = exp(tConditionalExpression);
    Plato::OrdinalType tCoeff = fmin(static_cast<Plato::Scalar>(1.0), tConditionalExpression);
    Plato::Scalar tScalarCoeff = tCoeff;
    auto tOutput = tScalarCoeff * aB + (static_cast<Plato::Scalar>(1.0) - tScalarCoeff) * aA;
    return (tOutput);
}

Plato::Scalar diagonalAveAbs(
    Plato::CrsMatrixType const & aMatrix);

void shiftDiagonal(
    Plato::CrsMatrixType const & aMatrix,
    Plato::Scalar                aScale);

/******************************************************************************//**
 * \brief Matrix times vector plus vector
 * \param [in] aMatrix multiplier of 1D container A
 * \param [in] aInput input 1D container
 * \param [out] aOutput output 1D container
**********************************************************************************/
template<typename ScalarT>
void MatrixTimesVectorPlusVector(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
                                 const Plato::ScalarVectorT<ScalarT> & aInput,
                                 const Plato::ScalarVectorT<ScalarT> & aOutput);

/******************************************************************************//**
 * \brief Vector times Matrix plus vector
 * \param [in] aInput input 1D container
 * \param [in] aMatrix multiplier of 1D container A
 * \param [out] aOutput output 1D container
**********************************************************************************/
template<typename ScalarT>
void VectorTimesMatrixPlusVector(
    const Plato::ScalarVectorT<ScalarT>      & aInput,
    const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
    const Plato::ScalarVectorT<ScalarT>      & aOutput);

/******************************************************************************//**
 * \brief Compute row sum inverse product
 * \param [in] aA
 * \param [in/out] aB
 *
 * aB = RowSum(aA)^{-1} . aB
 *
**********************************************************************************/
void
RowSummedInverseMultiply( const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
                                Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo);


/******************************************************************************/
/*! 
  \brief Extract block matrix graph and values into a non-block matrix format
  \param [in] aMatrix Block matrix from which graph and values will be extracted
  \param [out] aMatrixRowMap Non-block row map: (rowIndex) => {offsetIndex}
  \param [out] aMatrixColMap Non-block column map: (offsetIndex) => {columnIndex}
  \param [out] aMatrixValues Non-block matrix entries: (offsetIndex) => {matrix value}
  \param [in] aRowStride If greater than one, the input matrix is considered a sub-block matrix.
  \param [in] aRowOffset Offset of the submatrix.  Ignored if aRowStride equals one.

  The column map for the block matrix provides the node column index.  The matrix 
  values are indexed by offsetIndex*numColsPerBlock*numRowsPerBlock + blockRowIndex*numColsPerBlock + blockColIndex.
  The output matrix values are indexed by offsetIndex.

  If aRowStride is greater than one, the input matrix is assumed to be a sub-block matrix, i.e., 
  the block data provided for each node is only one row of the full block matrix.  The value of
  aRowStride is the number of rows in the full block matrix, and aRowOffset is the index of the
  sub-matrix row within the full block matrix.  The resulting non-block maps are for the full
  block matrix.
*/
/******************************************************************************/
void
getDataAsNonBlock( const Teuchos::RCP<Plato::CrsMatrixType>       & aMatrix,
                         Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixRowMap,
                         Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixColMap,
                         Plato::ScalarVectorT<Plato::Scalar>      & aMatrixValues,
                         int aRowStride=1, int aRowOffset=0);

/******************************************************************************//**
 * \brief Sort the column indices within each row to ascending order and apply
          the same map to the matrix entries array.
 * \param [in] aMatrixRowMap matrix row map
 * \param [in/out] aMatrixColMap matrix column map to be sorted
 * \param [in/out] aMatrixValues matrix values
 *
**********************************************************************************/
void
sortColumnEntries( const Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixRowMap,
                         Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixColMap,
                         Plato::ScalarVectorT<Plato::Scalar>      & aMatrixValues);

/******************************************************************************//**
 * \brief Set block matrix data from non-block data
 * \param [in/out] aMatrix
 * \param [in] aMatrixRowMap Non-block matrix row map
 * \param [in] aMatrixColMap Non-block matrix col map
 * \param [in] aMatrixValues Non-block matrix values
 *
**********************************************************************************/
void
setDataFromNonBlock(      Teuchos::RCP<Plato::CrsMatrixType>       & aMatrix,
                    const Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixRowMap,
                          Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixColMap,
                          Plato::ScalarVectorT<Plato::Scalar>      & aMatrixValues);

/******************************************************************************//**
 * \brief Compute the matrix product
 * \param [in] aM1
 * \param [in] aM2
 * \param [out] aProduct
 *
 * aProduct = aM1 . aM2
 *
**********************************************************************************/
void
MatrixMatrixMultiply( const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
                      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo,
                            Teuchos::RCP<Plato::CrsMatrixType> & aOutMatrix,
                            KokkosSparse::SPGEMMAlgorithm aAlgorithm = KokkosSparse::SPGEMM_KK_SPEED);

/******************************************************************************//**
 * \brief matrix minus matrix
 * \param [in/out] aM1
 * \param [in] aM2
 *
 * aM1 = aM1 - aM2
 *
**********************************************************************************/
void
MatrixMinusMatrix(      Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
                  const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo,
                        Plato::OrdinalType aOffset);

/******************************************************************************//**
 * \brief Condense into reduced matrix
 * \param [in/out] aA
 * \param [in] aB
 * \param [in] aC
 * \param [in] aD
 *
 * aA = aA - aB . RowSum(aC)^{-1} . aD
 *
**********************************************************************************/
void Condense(       Teuchos::RCP<Plato::CrsMatrixType> & aA,
                      const Teuchos::RCP<Plato::CrsMatrixType> & aB,
                      const Teuchos::RCP<Plato::CrsMatrixType> & aC,
                            Teuchos::RCP<Plato::CrsMatrixType> & aD,
                            Plato::OrdinalType aOffset );

/******************************************************************************//**
 * \brief Compute matrix transpose
 * \param [in] aMatrix
 * \param [out] aMatrixTranspose
 *
**********************************************************************************/
void
MatrixTranspose( const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
                       Teuchos::RCP<Plato::CrsMatrixType> & aMatrixTranspose);

} 
