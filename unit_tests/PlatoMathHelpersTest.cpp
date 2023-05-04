#include <vector>
#include <array>

//#define COMPUTE_GOLD_
#ifdef COMPUTE_GOLD_
  #include <iostream>
  #include <fstream>
#endif

#include <assert.h>

#include "util/PlatoTestHelpers.hpp"
#include "util/PlatoMathTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "BLAS1.hpp"
#include "Solutions.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoMathFunctors.hpp"
#include "Mechanics.hpp"
#include "stabilized/Mechanics.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
#include "elliptic/VectorFunction.hpp"
#include "stabilized/VectorFunction.hpp"
#include "ApplyProjection.hpp"
#include "AnalyzeMacros.hpp"
#include "HyperbolicTangentProjection.hpp"
#include "solver/CrsMatrix.hpp"
#include "solver/PlatoSolverFactory.hpp"

#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_LU_Serial_Impl.hpp"
#include "KokkosBatched_Trsm_Decl.hpp"
#include "KokkosBatched_Trsm_Serial_Impl.hpp"

#include <Kokkos_Concepts.hpp>
#include "KokkosKernels_SparseUtils.hpp"
#include "KokkosSparse_spgemm.hpp"
#include "KokkosSparse_spadd.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include <KokkosKernels_IOUtils.hpp>

namespace PlatoUnitTests
{
namespace pth = Plato::TestHelpers;

using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

namespace{
constexpr char kElastostaticParamsXML[] = 
        "<ParameterList name='Plato Problem'>                                  "
        "      \n"
        "  <ParameterList name='Spatial Model'>                                "
        "      \n"
        "    <ParameterList name='Domains'>                                    "
        "      \n"
        "      <ParameterList name='Design Volume'>                            "
        "      \n"
        "        <Parameter name='Element Block' type='string' value='body'/>  "
        "      \n"
        "        <Parameter name='Material Model' type='string' "
        "value='Unobtainium'/>\n"
        "      </ParameterList>                                                "
        "      \n"
        "    </ParameterList>                                                  "
        "      \n"
        "  </ParameterList>                                                    "
        "      \n"
        "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>   "
        "      \n"
        "  <Parameter name='Self-Adjoint' type='bool' value='false'/>          "
        "      \n"
        "  <ParameterList name='Material Models'>                              "
        "      \n"
        "    <ParameterList name='Unobtainium'>                                "
        "      \n"
        "      <ParameterList name='Isotropic Linear Elastic'>                 "
        "      \n"
        "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>  "
        "      \n"
        "        <Parameter name='Youngs Modulus' type='double' "
        "value='1.0e6'/>      \n"
        "      </ParameterList>                                                "
        "      \n"
        "    </ParameterList>                                                  "
        "      \n"
        "  </ParameterList>                                                    "
        "      \n"
        "  <ParameterList name='Elliptic'>                                     "
        "      \n"
        "    <ParameterList name='Penalty Function'>                           "
        "      \n"
        "      <Parameter name='Exponent' type='double' value='1.0'/>          "
        "      \n"
        "      <Parameter name='Type' type='string' value='SIMP'/>             "
        "      \n"
        "    </ParameterList>                                                  "
        "      \n"
        "  </ParameterList>                                                    "
        "      \n"
        "</ParameterList>                                                      "
        "      \n";

Teuchos::RCP<Teuchos::ParameterList> test_elastostatics_params() {
  return Teuchos::getParametersFromXmlString(kElastostaticParamsXML);
}

Teuchos::RCP<Plato::CrsMatrixType> createSquareMatrix()
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = pth::get_box_mesh("TET4", meshWidth);
  auto nverts = tMesh->NumNodes();

  // create vector data
  //
  Plato::ScalarVector u("state", spaceDim*nverts);
  Plato::ScalarVector z("control", nverts);
  Plato::blas1::fill(1.0, z);

  // create residual function
  //
  Plato::DataMap tDataMap;

  const Teuchos::RCP<Teuchos::ParameterList> elastostaticsParams = test_elastostatics_params();
  Plato::SpatialModel tSpatialModel(tMesh, *elastostaticsParams, tDataMap);

  Plato::Elliptic::VectorFunction<::Plato::Mechanics<Plato::Tet4>>
    tVectorFunction(tSpatialModel, tDataMap, *elastostaticsParams, 
        elastostaticsParams->get<std::string>("PDE Constraint"));

  // compute and test gradient_u
  //
  return tVectorFunction.gradient_u(u,z);
}

template <typename PhysicsType>
Teuchos::RCP<Plato::Stabilized::VectorFunction<PhysicsType>>
createStabilizedResidual(const Plato::SpatialModel & aSpatialModel)
{
  const Teuchos::RCP<Teuchos::ParameterList> elastostaticsParams = test_elastostatics_params();
  Plato::DataMap tDataMap;
  return Teuchos::rcp( new Plato::Stabilized::VectorFunction<PhysicsType>
         (aSpatialModel, tDataMap, *elastostaticsParams, elastostaticsParams->get<std::string>("PDE Constraint")));
}

template <typename PhysicsType>
Teuchos::RCP<Plato::Stabilized::VectorFunction<typename PhysicsType::ProjectorType>>
createStabilizedProjector(const Plato::SpatialModel & aSpatialModel)
{
  const Teuchos::RCP<Teuchos::ParameterList> elastostaticsParams = test_elastostatics_params();
  Plato::DataMap tDataMap;
  return Teuchos::rcp( new Plato::Stabilized::VectorFunction<typename PhysicsType::ProjectorType>
         (aSpatialModel, tDataMap, *elastostaticsParams, std::string("State Gradient Projection")));
}
}

/******************************************************************************/
/*! 
  \brief Transform a block matrix to a non-block matrix and back then verify
 that the starting and final matrices are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_FromToBlockMatrix)
{
  const auto tMatrixA = createSquareMatrix();

  const auto tNumRows = tMatrixA->numRows();
  const auto tNumCols = tMatrixA->numCols();
  const auto tNumRowsPerBlock = tMatrixA->numRowsPerBlock();
  const auto tNumColsPerBlock = tMatrixA->numColsPerBlock();
  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock) );

  Plato::ScalarVectorT<Plato::Scalar> tMatrixEntries;
  Plato::ScalarVectorT<Plato::OrdinalType> tMatrixRowMap, tMatrixColMap;
  Plato::getDataAsNonBlock  (tMatrixA, tMatrixRowMap, tMatrixColMap, tMatrixEntries);
  Plato::setDataFromNonBlock(tMatrixB, tMatrixRowMap, tMatrixColMap, tMatrixEntries);

  TEST_ASSERT(pth::is_same(tMatrixA->rowMap(), tMatrixB->rowMap()));

  TEST_ASSERT(pth::is_equivalent(tMatrixA->rowMap(),
                                 tMatrixA->columnIndices(), tMatrixA->entries(),
                                 tMatrixB->columnIndices(), tMatrixB->entries()));
}

/******************************************************************************/
/*! 
  \brief Transform a rectangular block matrix to a non-block matrix and back then verify
 that the starting and final matrices are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_FromToBlockMatrix_Rect)
{
  const auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(12, 9, 4, 3) );
  const std::vector<Plato::OrdinalType> tRowMapA = { 0, 2, 3, 4 };
  const std::vector<Plato::OrdinalType> tColMapA = { 0, 2, 0, 2 };
  const std::vector<Plato::Scalar>      tValuesA = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
  pth::set_matrix_data(tMatrixA, tRowMapA, tColMapA, tValuesA);

  const auto tNumRows = tMatrixA->numRows();
  const auto tNumCols = tMatrixA->numCols();
  const auto tNumRowsPerBlock = tMatrixA->numRowsPerBlock();
  const auto tNumColsPerBlock = tMatrixA->numColsPerBlock();
  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock) );

  Plato::ScalarVectorT<Plato::Scalar> tMatrixEntries;
  Plato::ScalarVectorT<Plato::OrdinalType> tMatrixRowMap, tMatrixColMap;
  Plato::getDataAsNonBlock  (tMatrixA, tMatrixRowMap, tMatrixColMap, tMatrixEntries);
  Plato::setDataFromNonBlock(tMatrixB, tMatrixRowMap, tMatrixColMap, tMatrixEntries);

  TEST_ASSERT(pth::is_same(tMatrixA->rowMap(), tMatrixB->rowMap()));

  TEST_ASSERT(pth::is_equivalent(tMatrixA->rowMap(),
                                 tMatrixA->columnIndices(), tMatrixA->entries(),
                                 tMatrixB->columnIndices(), tMatrixB->entries()));
}

/******************************************************************************/
/*! 
  \brief Transform a rectangular block matrix to a non-block matrix and check with known
  non-block matrix.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_FromToBlockMatrix_Rect2)
{
  const auto tMatrixA1 = Teuchos::rcp( new Plato::CrsMatrixType(2, 4, 2, 2) );
  const std::vector<Plato::OrdinalType> tRowMapA1 = { 0, 2 };
  const std::vector<Plato::OrdinalType> tColMapA1 = { 0, 1 };
  const std::vector<Plato::Scalar>      tValuesA1 = { 0, 1, 2, 0, 0, 3, 1, 0 };
  pth::set_matrix_data(tMatrixA1, tRowMapA1, tColMapA1, tValuesA1);

  const auto tMatrixA2 = Teuchos::rcp( new Plato::CrsMatrixType(4, 2, 2, 2) );
  const std::vector<Plato::OrdinalType> tRowMapA2 = { 0, 1, 2 };
  const std::vector<Plato::OrdinalType> tColMapA2 = { 0, 0 };
  const std::vector<Plato::Scalar>      tValuesA2 = { 0, 2, 1, 0, 0, 1, 3, 0 };
  pth::set_matrix_data(tMatrixA2, tRowMapA2, tColMapA2, tValuesA2);

  const auto tGoldMatrixA1 = Teuchos::rcp( new Plato::CrsMatrixType(2, 4, 1, 1) );
  const std::vector<Plato::OrdinalType> tGoldRowMapA1 = { 0, 4, 8 };
  const std::vector<Plato::OrdinalType> tGoldColMapA1 = { 0, 1, 2, 3, 0, 1, 2, 3 };
  const std::vector<Plato::Scalar>      tGoldValuesA1 = { 0, 1, 0, 3, 2, 0, 1, 0 };
  pth::set_matrix_data(tGoldMatrixA1, tGoldRowMapA1, tGoldColMapA1, tGoldValuesA1);

  const auto tGoldMatrixA2 = Teuchos::rcp( new Plato::CrsMatrixType(4, 2, 1, 1) );
  const std::vector<Plato::OrdinalType> tGoldRowMapA2 = { 0, 2, 4, 6, 8 };
  const std::vector<Plato::OrdinalType> tGoldColMapA2 = { 0, 1, 0, 1, 0, 1, 0, 1 };
  const std::vector<Plato::Scalar>      tGoldValuesA2 = { 0, 2, 1, 0, 0, 1, 3, 0 };
  pth::set_matrix_data(tGoldMatrixA2, tGoldRowMapA2, tGoldColMapA2, tGoldValuesA2);

  auto tMatrixA1NonBlock = Teuchos::rcp( new Plato::CrsMatrixType( 2, 4, 1, 1) );
  Plato::ScalarVectorT<Plato::Scalar> tMatrixA1NonBlockEntries;
  Plato::ScalarVectorT<Plato::OrdinalType> tMatrixA1NonBlockRowMap, tMatrixA1NonBlockColMap;
  Plato::getDataAsNonBlock  (tMatrixA1, tMatrixA1NonBlockRowMap, tMatrixA1NonBlockColMap, tMatrixA1NonBlockEntries);
  tMatrixA1NonBlock->setRowMap(tMatrixA1NonBlockRowMap);
  tMatrixA1NonBlock->setColumnIndices(tMatrixA1NonBlockColMap);
  tMatrixA1NonBlock->setEntries(tMatrixA1NonBlockEntries);

  auto tMatrixA2NonBlock = Teuchos::rcp( new Plato::CrsMatrixType( 4, 2, 1, 1) );
  Plato::ScalarVectorT<Plato::Scalar> tMatrixA2NonBlockEntries;
  Plato::ScalarVectorT<Plato::OrdinalType> tMatrixA2NonBlockRowMap, tMatrixA2NonBlockColMap;
  Plato::getDataAsNonBlock  (tMatrixA2, tMatrixA2NonBlockRowMap, tMatrixA2NonBlockColMap, tMatrixA2NonBlockEntries);
  tMatrixA2NonBlock->setRowMap(tMatrixA2NonBlockRowMap);
  tMatrixA2NonBlock->setColumnIndices(tMatrixA2NonBlockColMap);
  tMatrixA2NonBlock->setEntries(tMatrixA2NonBlockEntries);

  TEST_ASSERT(pth::is_same(tMatrixA1NonBlock->rowMap(), tGoldMatrixA1->rowMap()));
  TEST_ASSERT(pth::is_equivalent(tMatrixA1NonBlock->rowMap(),
                                 tMatrixA1NonBlock->columnIndices(), tMatrixA1NonBlock->entries(),
                                 tGoldMatrixA1->columnIndices(), tGoldMatrixA1->entries()));

  TEST_ASSERT(pth::is_same(tMatrixA2NonBlock->rowMap(), tGoldMatrixA2->rowMap()));
  TEST_ASSERT(pth::is_equivalent(tMatrixA2NonBlock->rowMap(),
                                 tMatrixA2NonBlock->columnIndices(), tMatrixA2NonBlock->entries(),
                                 tGoldMatrixA2->columnIndices(), tGoldMatrixA2->entries()));
}


/******************************************************************************/
/*! 
  \brief Make sure is_same(A, A) == true, and is_same(A, B) == false when A != B
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_is_same)
{
  const auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(12, 6, 4, 2) );
  const std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
  const std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
  const std::vector<Plato::Scalar>      tValuesA = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
  pth::set_matrix_data(tMatrixA, tRowMap, tColMap, tValuesA);

  const auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType(6, 12, 2, 4) );
  const std::vector<Plato::Scalar>      tValuesB = 
    { 1, 5, 2, 6, 3, 7, 4, 8, 1, 5, 2, 6, 3, 7, 4, 8,
      1, 5, 2, 6, 3, 7, 4, 8, 1, 5, 2, 6, 3, 7, 4, 8 };
  pth::set_matrix_data(tMatrixB, tRowMap, tColMap, tValuesB);

  TEST_ASSERT(pth::is_same(tMatrixA, tMatrixA));
  TEST_ASSERT(!pth::is_same(tMatrixA, tMatrixB));
}

/******************************************************************************/
/*! 
  \brief Create rectangular matrix, A and B=Tranpose(A), then compute A.B and 
         compare against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_MatrixMatrixMultiply_Rect)
{
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(12, 6, 4, 2) );
  std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesA = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
  pth::set_matrix_data(tMatrixA, tRowMap, tColMap, tValuesA);

  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType(6, 12, 2, 4) );
  std::vector<Plato::Scalar>      tValuesB = 
    { 1, 5, 2, 6, 3, 7, 4, 8, 1, 5, 2, 6, 3, 7, 4, 8,
      1, 5, 2, 6, 3, 7, 4, 8, 1, 5, 2, 6, 3, 7, 4, 8 };
  pth::set_matrix_data(tMatrixB, tRowMap, tColMap, tValuesB);

  auto tMatrixAB         = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  auto tSlowDumbMatrixAB = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );

  Plato::MatrixMatrixMultiply              ( tMatrixA, tMatrixB, tMatrixAB);
  pth::slow_dumb_matrix_matrix_multiply ( tMatrixA, tMatrixB, tSlowDumbMatrixAB);

  TEST_ASSERT(pth::is_same(tMatrixAB, tSlowDumbMatrixAB));
}

/******************************************************************************/
/*! 
  \brief Create a square matrix, A, then compute A.A and compare against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_MatrixMatrixMultiply_1)
{
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  std::vector<Plato::OrdinalType> tRowMapA = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapA = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesA = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  pth::set_matrix_data(tMatrixA, tRowMapA, tColMapA, tValuesA);

  auto tMatrixAA         = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  auto tSlowDumbMatrixAA = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );

  Plato::MatrixMatrixMultiply             ( tMatrixA, tMatrixA, tMatrixAA);
  pth::slow_dumb_matrix_matrix_multiply( tMatrixA, tMatrixA, tSlowDumbMatrixAA);

  std::vector<Plato::Scalar> tGoldMatrixEntries = {
    90.0, 100.0, 110.0, 120.0, 202.0, 228.0, 254.0, 280.0, 314.0, 356.0, 398.0, 440.0, 426.0, 484.0,  542.0,  600.0,
   180.0, 200.0, 220.0, 240.0, 404.0, 456.0, 508.0, 560.0, 628.0, 712.0, 796.0, 880.0, 852.0, 968.0, 1084.0, 1200.0,
    90.0, 100.0, 110.0, 120.0, 202.0, 228.0, 254.0, 280.0, 314.0, 356.0, 398.0, 440.0, 426.0, 484.0,  542.0,  600.0,
    90.0, 100.0, 110.0, 120.0, 202.0, 228.0, 254.0, 280.0, 314.0, 356.0, 398.0, 440.0, 426.0, 484.0,  542.0,  600.0,
    90.0, 100.0, 110.0, 120.0, 202.0, 228.0, 254.0, 280.0, 314.0, 356.0, 398.0, 440.0, 426.0, 484.0,  542.0,  600.0
  };
  TEST_ASSERT(pth::is_same(tMatrixAA->entries(), tGoldMatrixEntries));

  std::vector<Plato::OrdinalType> tGoldMatrixRowMap = { 0, 2, 4, 5 };
  TEST_ASSERT(pth::is_same(tMatrixAA->rowMap(), tGoldMatrixRowMap));

  std::vector<Plato::OrdinalType> tGoldMatrixColMap = { 0, 2, 0, 2, 2 };
  TEST_ASSERT(pth::is_same(tMatrixAA->columnIndices(), tGoldMatrixColMap));

  TEST_ASSERT(pth::is_same(tMatrixAA, tSlowDumbMatrixAA));
}

/******************************************************************************/
/*! 
  \brief Create a square matrix, A, then sort the column entries
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_SortColumnEntries)
{
  auto tMatrix = createSquareMatrix();

  Plato::ScalarVectorT<Plato::Scalar> tMatrixEntries;
  Plato::ScalarVectorT<Plato::OrdinalType> tMatrixRowMap, tMatrixColMap;
  Plato::getDataAsNonBlock (tMatrix, tMatrixRowMap, tMatrixColMap, tMatrixEntries);

  Plato::ScalarVectorT<Plato::Scalar> tMatrixEntriesSorted("values", tMatrixEntries.extent(0));
  Kokkos::deep_copy(tMatrixEntriesSorted, tMatrixEntries);

  Plato::ScalarVectorT<Plato::OrdinalType> tMatrixColMapSorted("col map", tMatrixColMap.extent(0));
  Kokkos::deep_copy(tMatrixColMapSorted, tMatrixColMap);

  Plato::sortColumnEntries(tMatrixRowMap, tMatrixColMapSorted, tMatrixEntriesSorted);

  TEST_ASSERT(pth::is_sequential(tMatrixRowMap, tMatrixColMapSorted) );
  TEST_ASSERT(pth::is_equivalent(tMatrixRowMap,
                            tMatrixColMap, tMatrixEntries,
                            tMatrixColMapSorted, tMatrixEntriesSorted) );
}

/******************************************************************************/
/*! 
  \brief Create a square matrix, A, then convert to and from full (non-sparse)
         matrix and verify that A doesn't change.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_ToFromFull)
{
  auto tMatrix = createSquareMatrix();
  auto tFullMatrix = pth::to_full(tMatrix);
  auto tSparseMatrix = Teuchos::rcp( new Plato::CrsMatrixType( tMatrix->numRows(), tMatrix->numCols(), 
                                     tMatrix->numRowsPerBlock(), tMatrix->numRowsPerBlock()));
  pth::from_full(tSparseMatrix, tFullMatrix);
  
  TEST_ASSERT(pth::is_equivalent(tMatrix->rowMap(),
                            tMatrix->columnIndices(), tMatrix->entries(),
                            tSparseMatrix->columnIndices(), tSparseMatrix->entries()));
}

/******************************************************************************/
/*! 
  \brief Create a square matrix, A, then compute A.A and compare.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_MatrixMatrixMultiply_2)
{
  auto tMatrixA = createSquareMatrix();
  auto tMatrixB = createSquareMatrix();

  auto tNumRows = tMatrixA->numRows();
  auto tNumCols = tMatrixB->numCols();
  auto tNumRowsPerBlock = tMatrixA->numRowsPerBlock();
  auto tNumColsPerBlock = tMatrixB->numColsPerBlock();
  auto tMatrixAB         = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock) );
  auto tSlowDumbMatrixAB = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock) );

  Plato::MatrixMatrixMultiply              ( tMatrixA, tMatrixB, tMatrixAB);
  pth::slow_dumb_matrix_matrix_multiply ( tMatrixA, tMatrixB, tSlowDumbMatrixAB);

  TEST_ASSERT(pth::is_same(tMatrixAB->rowMap(), tSlowDumbMatrixAB->rowMap()));
  TEST_ASSERT(pth::is_equivalent(tMatrixAB->rowMap(),
                            tMatrixAB->columnIndices(), tMatrixAB->entries(),
                            tSlowDumbMatrixAB->columnIndices(), tSlowDumbMatrixAB->entries()));
}

/******************************************************************************/
/*! 
  \brief Create a square matrix, A, then verify that A - A = 0.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_MatrixMinusEqualsMatrix)
{
  auto tMatrixA = createSquareMatrix();
  pth::matrix_minus_equals_matrix( tMatrixA, tMatrixA );
  TEST_ASSERT(pth::is_zero(tMatrixA));
}


/******************************************************************************/
/*! 
  \brief Create a sub-block matrix, B, and convert to full non-block.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_GetDataAsNonBlock)
{
  Plato::OrdinalType tTargetBlockColSize = 4;
  Plato::OrdinalType tTargetBlockColOffset = 3;
  auto tMatrix = Teuchos::rcp( new Plato::CrsMatrixType(  3, 12, 1, 4) );
  std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValues = 
    { 13, 14, 15, 16,
      13, 14, 15, 16,
      13, 14, 15, 16,
      13, 14, 15, 16 };
  pth::set_matrix_data(tMatrix, tRowMap, tColMap, tValues);

  Plato::ScalarVectorT<Plato::Scalar> tMatrixEntries;
  Plato::ScalarVectorT<Plato::OrdinalType> tMatrixRowMap, tMatrixColMap;
  Plato::getDataAsNonBlock (tMatrix, tMatrixRowMap, tMatrixColMap, tMatrixEntries,
                                 tTargetBlockColSize, tTargetBlockColOffset);

  std::vector<Plato::OrdinalType> tMatrixRowMap_Gold = {
    0, 8, 16, 24, 32, 36, 40, 44, 48, 52, 56, 60, 64
  };
  TEST_ASSERT(pth::is_same(tMatrixRowMap, tMatrixRowMap_Gold));

  std::vector<Plato::OrdinalType> tMatrixColMap_Gold = {
    0, 1, 2,  3,  8, 9, 10, 11,
    0, 1, 2,  3,  8, 9, 10, 11,
    0, 1, 2,  3,  8, 9, 10, 11,
    0, 1, 2,  3,  8, 9, 10, 11,
    0, 1, 2,  3,  0, 1, 2,  3,
    0, 1, 2,  3,  0, 1, 2,  3,
    8, 9, 10, 11, 8, 9, 10, 11,
    8, 9, 10, 11, 8, 9, 10, 11
  };
  TEST_ASSERT(pth::is_same(tMatrixColMap, tMatrixColMap_Gold));

  std::vector<Plato::Scalar> tMatrixEntries_Gold = {
     0.0,  0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,
     0.0,  0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,
     0.0,  0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,
    13.0, 14.0, 15.0, 16.0, 13.0, 14.0, 15.0, 16.0,
     0.0,  0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,
     0.0,  0.0,  0.0,  0.0, 13.0, 14.0, 15.0, 16.0,
     0.0,  0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,
     0.0,  0.0,  0.0,  0.0, 13.0, 14.0, 15.0, 16.0
  };
  TEST_ASSERT(pth::is_same(tMatrixEntries, tMatrixEntries_Gold));
}

/******************************************************************************/
/*! 
  \brief Create a block matrix, compute the row sum, and compare against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_RowSum)
{
  auto tMatrix = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  const std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
  const std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
  const std::vector<Plato::Scalar>      tValues = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  pth::set_matrix_data(tMatrix, tRowMap, tColMap, tValues);

  Plato::ScalarVector tRowSum("row sum", 12);

  pth::row_sum( tMatrix, tRowSum );
  const std::vector<Plato::Scalar> tRowSum_Gold = {
    20.0, 52.0, 84.0, 116.0, 10.0, 26.0, 42.0, 58.0, 10.0, 26.0, 42.0, 58.0
  };
  TEST_ASSERT(pth::is_same(tRowSum, tRowSum_Gold));
}

/******************************************************************************/
/*! 
  \brief Create a block matrix, A, and a vector of diagonal weights, d={1.0}, and 
         compute the inverse weighted A: A[I][J]/d[I], and compare against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_InverseMultiply_1)
{
  auto tMatrix = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValues = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  pth::set_matrix_data(tMatrix, tRowMap, tColMap, tValues);

  Plato::ScalarVector tDiagonals("diagonals", 12);
  Plato::blas1::fill( 1.0, tDiagonals );
  
  pth::inverse_multiply( tMatrix, tDiagonals );
  std::vector<Plato::Scalar> tGoldMatrixEntries =
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  auto tMatrixEntries = tMatrix->entries();
  TEST_ASSERT(pth::is_same(tMatrixEntries, tGoldMatrixEntries));
}

/******************************************************************************/
/*! 
  \brief Create a block matrix, A, and a vector of diagonal weights, d={2.0}, and 
         compute the inverse weighted A: A[I][J]/d[I], and compare against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_InverseMultiply_2)
{
  auto tMatrix = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValues = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  pth::set_matrix_data(tMatrix, tRowMap, tColMap, tValues);

  Plato::ScalarVector tDiagonals("diagonals", 12);
  Plato::blas1::fill( 2.0, tDiagonals );
  
  pth::inverse_multiply( tMatrix, tDiagonals );
  std::vector<Plato::Scalar> tGoldMatrixEntries =
    { 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
      0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
      0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
      0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8 };
  auto tMatrixEntries = tMatrix->entries();
  TEST_ASSERT(pth::is_same(tMatrixEntries, tGoldMatrixEntries));
}

/******************************************************************************/
/*! 
  \brief Create block matrices, A and B, and apply the row summed inverse of A 
         to B: B[I][J]/d[I], where d[I] = Sum(A[I][J] in J). Compare against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_SlowDumbRowSummedInverseMultiply)
{
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  {
    std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
    std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
    std::vector<Plato::Scalar>      tValues = 
      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    pth::set_matrix_data(tMatrixA, tRowMap, tColMap, tValues);
  }

  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  {
    std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
    std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
    std::vector<Plato::Scalar>      tValues = 
      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    pth::set_matrix_data(tMatrixB, tRowMap, tColMap, tValues);
  }

  pth::slow_dumb_row_summed_inverse_multiply( tMatrixA, tMatrixB );
  std::vector<Plato::Scalar> tGoldMatrixEntries = {
   1.0/20.0,  2.0/20.0,  3.0/20.0,  4.0/20.0,  5.0/ 52.0,  6.0/ 52.0,  7.0/ 52.0,  8.0/ 52.0,
   9.0/84.0, 10.0/84.0, 11.0/84.0, 12.0/84.0, 13.0/116.0, 14.0/116.0, 15.0/116.0, 16.0/116.0,
   1.0/20.0,  2.0/20.0,  3.0/20.0,  4.0/20.0,  5.0/ 52.0,  6.0/ 52.0,  7.0/ 52.0,  8.0/ 52.0,
   9.0/84.0, 10.0/84.0, 11.0/84.0, 12.0/84.0, 13.0/116.0, 14.0/116.0, 15.0/116.0, 16.0/116.0,
   1.0/10.0,  2.0/10.0,  3.0/10.0,  4.0/10.0,  5.0/ 26.0,  6.0/ 26.0,  7.0/ 26.0,  8.0/ 26.0,
   9.0/42.0, 10.0/42.0, 11.0/42.0, 12.0/42.0, 13.0/ 58.0, 14.0/ 58.0, 15.0/ 58.0, 16.0/ 58.0,
   1.0/10.0,  2.0/10.0,  3.0/10.0,  4.0/10.0,  5.0/ 26.0,  6.0/ 26.0,  7.0/ 26.0,  8.0/ 26.0,
   9.0/42.0, 10.0/42.0, 11.0/42.0, 12.0/42.0, 13.0/ 58.0, 14.0/ 58.0, 15.0/ 58.0, 16.0/ 58.0
  };
  auto tMatrixEntries = tMatrixB->entries();
  TEST_ASSERT(pth::is_same(tMatrixEntries, tGoldMatrixEntries));
}
/******************************************************************************/
/*! 
  \brief Create block matrices, A and B, and apply the row summed inverse of A 
         to B: B[I][J]/d[I], where d[I] = Sum(A[I][J] in J). Compare against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_RowSummedInverseMultiply)
{
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  {
    std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
    std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
    std::vector<Plato::Scalar>      tValues = 
      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    pth::set_matrix_data(tMatrixA, tRowMap, tColMap, tValues);
  }

  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  {
    std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
    std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
    std::vector<Plato::Scalar>      tValues = 
      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    pth::set_matrix_data(tMatrixB, tRowMap, tColMap, tValues);
  }

  Plato::RowSummedInverseMultiply( tMatrixA, tMatrixB );
  std::vector<Plato::Scalar> tGoldMatrixEntries = {
   1.0/20.0,  2.0/20.0,  3.0/20.0,  4.0/20.0,  5.0/ 52.0,  6.0/ 52.0,  7.0/ 52.0,  8.0/ 52.0,
   9.0/84.0, 10.0/84.0, 11.0/84.0, 12.0/84.0, 13.0/116.0, 14.0/116.0, 15.0/116.0, 16.0/116.0,
   1.0/20.0,  2.0/20.0,  3.0/20.0,  4.0/20.0,  5.0/ 52.0,  6.0/ 52.0,  7.0/ 52.0,  8.0/ 52.0,
   9.0/84.0, 10.0/84.0, 11.0/84.0, 12.0/84.0, 13.0/116.0, 14.0/116.0, 15.0/116.0, 16.0/116.0,
   1.0/10.0,  2.0/10.0,  3.0/10.0,  4.0/10.0,  5.0/ 26.0,  6.0/ 26.0,  7.0/ 26.0,  8.0/ 26.0,
   9.0/42.0, 10.0/42.0, 11.0/42.0, 12.0/42.0, 13.0/ 58.0, 14.0/ 58.0, 15.0/ 58.0, 16.0/ 58.0,
   1.0/10.0,  2.0/10.0,  3.0/10.0,  4.0/10.0,  5.0/ 26.0,  6.0/ 26.0,  7.0/ 26.0,  8.0/ 26.0,
   9.0/42.0, 10.0/42.0, 11.0/42.0, 12.0/42.0, 13.0/ 58.0, 14.0/ 58.0, 15.0/ 58.0, 16.0/ 58.0
  };
  auto tMatrixEntries = tMatrixB->entries();
  TEST_ASSERT(pth::is_same(tMatrixEntries, tGoldMatrixEntries));
}

/******************************************************************************/
/*! 
  \brief Create a condensed matrix: R = A - B RowSum(C)^-1 D

  Dimensions:
    A: Nf x Nf
    B: Nn x Nm
    C: Nm x Nm
    D: Nm x Nf
    R: Nf x Nf <= (Nf x Nf) - O(Nf,3)((Nn x Nm) . (Nm x Nm) . (Nm x Nf))

    where O(n,p)(M) expand a sub-block matrix, M, with row index, p, into
    a full-block matrix with number block rows, n.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_CondenseMatrix_1)
{
  const int Nf = 4;
  const int Nn = 1;
  const int Nm = 3;

  const int tNumNodes = 3;

  auto tA = Teuchos::rcp( new Plato::CrsMatrixType(tNumNodes*Nf, tNumNodes*Nf, Nf, Nf) );
  auto tA_SlowDumb = Teuchos::rcp( new Plato::CrsMatrixType(tNumNodes*Nf, tNumNodes*Nf, Nf, Nf) );
  {
    std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
    std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
    std::vector<Plato::Scalar>      tValues = 
      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    pth::set_matrix_data(tA, tRowMap, tColMap, tValues);
    pth::set_matrix_data(tA_SlowDumb, tRowMap, tColMap, tValues);
  }

  auto tB = Teuchos::rcp( new Plato::CrsMatrixType(tNumNodes*Nn, tNumNodes*Nm, Nn, Nm) );
  auto tB_SlowDumb = Teuchos::rcp( new Plato::CrsMatrixType(tNumNodes*Nn, tNumNodes*Nm, Nn, Nm) );
  {
    std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
    std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
    std::vector<Plato::Scalar>      tValues = 
      { 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3 };
    pth::set_matrix_data(tB, tRowMap, tColMap, tValues);
    pth::set_matrix_data(tB_SlowDumb, tRowMap, tColMap, tValues);
  }

  auto tC = Teuchos::rcp( new Plato::CrsMatrixType(tNumNodes*Nm, tNumNodes*Nm, Nm, Nm) );
  auto tC_SlowDumb = Teuchos::rcp( new Plato::CrsMatrixType(tNumNodes*Nm, tNumNodes*Nm, Nm, Nm) );
  {
    std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
    std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
    std::vector<Plato::Scalar>      tValues = 
      { 1, 2, 3, 4, 5, 6, 7, 8, 9,
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        1, 2, 3, 4, 5, 6, 7, 8, 9 };
    pth::set_matrix_data(tC, tRowMap, tColMap, tValues);
    pth::set_matrix_data(tC_SlowDumb, tRowMap, tColMap, tValues);
  }

  auto tD = Teuchos::rcp( new Plato::CrsMatrixType(tNumNodes*Nm, tNumNodes*Nf, Nm, Nf) );
  auto tD_SlowDumb = Teuchos::rcp( new Plato::CrsMatrixType(tNumNodes*Nm, tNumNodes*Nf, Nm, Nf) );
  {
    std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
    std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
    std::vector<Plato::Scalar>      tValues = 
      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    pth::set_matrix_data(tD, tRowMap, tColMap, tValues);
    pth::set_matrix_data(tD_SlowDumb, tRowMap, tColMap, tValues);
  }

  // Nn x Nf
  auto tMatrixProduct = Teuchos::rcp( new Plato::CrsMatrixType( tNumNodes*Nn, tNumNodes*Nf, Nn, Nf ) );
  auto tMatrixProduct_SlowDumb = Teuchos::rcp( new Plato::CrsMatrixType( tNumNodes*Nn, tNumNodes*Nf, Nn, Nf ) );

  // Nm x Nm . Nm x Nf => Nm x Nf
  Plato::RowSummedInverseMultiply              ( tC, tD );
  pth::slow_dumb_row_summed_inverse_multiply ( tC_SlowDumb, tD_SlowDumb );

  // Nn x Nm . Nm x Nf => Nn x Nf
  Plato::MatrixMatrixMultiply              ( tB, tD, tMatrixProduct );
  pth::slow_dumb_matrix_matrix_multiply ( tB_SlowDumb, tD_SlowDumb, tMatrixProduct_SlowDumb );

  const int tOffset = 3;
  // Nf x Nf - O(Nf,3)(Nn x Nf)
  Plato::MatrixMinusMatrix              ( tA, tMatrixProduct, tOffset );
  pth::slow_dumb_matrix_minus_matrix ( tA_SlowDumb, tMatrixProduct_SlowDumb, tOffset );

  TEST_ASSERT(pth::is_equivalent(tA->rowMap(),
                            tA->columnIndices(), tA->entries(),
                            tA_SlowDumb->columnIndices(), tA_SlowDumb->entries()));
}
/******************************************************************************/
/*! 
  \brief Create a stabilized residual, g, and a projector, P, then compute
 derivatives dg/du^T, dg/dn^T, dP/du^T, and dP/dn and the condensed matrix:

 A = dg/du^T - dP/du^T . RowSum(dP/dn)^{-1} . dg/dn^T

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_CondenseMatrix_2)
{
  constexpr int cMeshWidth = 2;
  auto tMesh = pth::get_box_mesh("TET4", cMeshWidth);
  
  Plato::DataMap tDataMap;
  const Teuchos::RCP<Teuchos::ParameterList> elastostaticsParams = test_elastostatics_params();
  Plato::SpatialModel tSpatialModel(tMesh, *elastostaticsParams, tDataMap);

  using PhysicsType = Plato::Stabilized::Mechanics<Plato::Tet4>;
  auto tResidual = createStabilizedResidual<PhysicsType>(tSpatialModel);
  auto tProjector = createStabilizedProjector<PhysicsType>(tSpatialModel);

  auto tNverts = tMesh->NumNodes();
  Plato::ScalarVector U("state",          tResidual->size());
  Plato::ScalarVector N("project p grad", tProjector->size());
  Plato::ScalarVector z("control",        tNverts);
  Plato::ScalarVector p("nodal pressure", tNverts);
  Plato::blas1::fill(1.0, z);

  //                                        u, n, z
  auto t_dg_du_T = tResidual->gradient_u_T (U, N, z);
  auto t_dg_dn_T = tResidual->gradient_n_T (U, N, z);
  auto t_dP_dn_T = tProjector->gradient_n_T(N, p, z);
  auto t_dP_du   = tProjector->gradient_u  (N, p, z);

  auto t_dg_du_T_sd = tResidual->gradient_u_T (U, N, z);
  auto t_dg_dn_T_sd = tResidual->gradient_n_T (U, N, z);
  auto t_dP_dn_T_sd = tProjector->gradient_n_T(N, p, z);
  auto t_dP_du_sd   = tProjector->gradient_u  (N, p, z);
  
  auto tNumRows = t_dP_dn_T->numRows();
  auto tNumCols = t_dg_dn_T->numCols();
  auto tNumRowsPerBlock = t_dP_dn_T->numRowsPerBlock();
  auto tNumColsPerBlock = t_dg_dn_T->numColsPerBlock();

  // Nn x Nf
  auto tMatrixProduct    = Teuchos::rcp( new Plato::CrsMatrixType(tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock) );
  auto tMatrixProduct_sd = Teuchos::rcp( new Plato::CrsMatrixType(tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock) );

  // Nm x Nm . Nm x Nf => Nm x Nf
  Plato::RowSummedInverseMultiply              ( t_dP_du,    t_dg_dn_T    );
  pth::slow_dumb_row_summed_inverse_multiply ( t_dP_du_sd, t_dg_dn_T_sd );

  // Nn x Nm . Nm x Nf => Nn x Nf
  Plato::MatrixMatrixMultiply              ( t_dP_dn_T,    t_dg_dn_T,    tMatrixProduct    );
  pth::slow_dumb_matrix_matrix_multiply ( t_dP_dn_T_sd, t_dg_dn_T_sd, tMatrixProduct_sd );

  auto tOffset = PhysicsType::ProjectorType::ElementType::mProjectionDof;
  // Nf x Nf - O( Nn x Nf ) => Nf x Nf
  Plato::MatrixMinusMatrix              ( t_dg_du_T,    tMatrixProduct,    tOffset );
  pth::slow_dumb_matrix_minus_matrix ( t_dg_du_T_sd, tMatrixProduct_sd, tOffset );

  TEST_ASSERT(pth::is_equivalent(t_dg_du_T->rowMap(),
                            t_dg_du_T->columnIndices(), t_dg_du_T->entries(),
                            t_dg_du_T_sd->columnIndices(), t_dg_du_T_sd->entries()));
}

/******************************************************************************/
/*! 
  \brief Create a full block matrix, A, and a sub-block matrix, B, and
         compute C=A-B, then verify C against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_SlowDumbMatrixMinusMatrix_1)
{
  Plato::OrdinalType tOffset = 3;

  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType( 12, 12, 4, 4) );
  std::vector<Plato::OrdinalType> tRowMapA = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapA = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesA = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  pth::set_matrix_data(tMatrixA, tRowMapA, tColMapA, tValuesA);

  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType(  3, 12, 1, 4) );
  std::vector<Plato::OrdinalType> tRowMapB = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapB = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesB = 
    { 13, 14, 15, 16,
      13, 14, 15, 16,
      13, 14, 15, 16,
      13, 14, 15, 16 };
  pth::set_matrix_data(tMatrixB, tRowMapB, tColMapB, tValuesB);

  pth::slow_dumb_matrix_minus_matrix( tMatrixA, tMatrixB, tOffset );

  auto tMatrixC_Gold = Teuchos::rcp( new Plato::CrsMatrixType( 12, 12, 4, 4) );
  std::vector<Plato::OrdinalType> tRowMapC = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapC = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesC = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  0,  0,  0,  0,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  0,  0,  0,  0,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  0,  0,  0,  0,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  0,  0,  0,  0 };
  pth::set_matrix_data(tMatrixC_Gold, tRowMapC, tColMapC, tValuesC);

  TEST_ASSERT(pth::is_same(tMatrixA, tMatrixC_Gold));
}

/******************************************************************************/
/*! 
  \brief Create a full block matrix, A, and a full block matrix, B, and
         compute C=A-B, then verify C against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_SlowDumbMatrixMinusMatrix_2)
{
  Plato::OrdinalType tOffset = 0;

  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType( 4, 4, 1, 1) );
  std::vector<Plato::OrdinalType> tRowMapA = { 0, 2, 4, 6, 8 };
  std::vector<Plato::OrdinalType> tColMapA = { 0, 1, 0, 1, 2, 3, 2, 3 };
  std::vector<Plato::Scalar>      tValuesA = 
    { 1, 2, 3, 4, 1, 2, 3, 4};
  pth::set_matrix_data(tMatrixA, tRowMapA, tColMapA, tValuesA);

  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType( 4, 4, 1, 1) );
  std::vector<Plato::OrdinalType> tRowMapB = { 0, 2, 2, 2, 4 };
  std::vector<Plato::OrdinalType> tColMapB = { 0, 1, 2, 3 };
  std::vector<Plato::Scalar>      tValuesB = 
    { 1, 2, 3, 4};
  pth::set_matrix_data(tMatrixB, tRowMapB, tColMapB, tValuesB);

  pth::slow_dumb_matrix_minus_matrix( tMatrixA, tMatrixB, tOffset );

  auto tMatrixC_Gold = Teuchos::rcp( new Plato::CrsMatrixType( 4, 4, 1, 1) );
  std::vector<Plato::OrdinalType> tRowMapC = { 0, 0, 2, 4, 4 };
  std::vector<Plato::OrdinalType> tColMapC = { 0, 1, 2, 3 };
  std::vector<Plato::Scalar>      tValuesC = 
    { 3, 4, 1, 2};
  pth::set_matrix_data(tMatrixC_Gold, tRowMapC, tColMapC, tValuesC);

  TEST_ASSERT(pth::is_same(tMatrixA, tMatrixC_Gold));
}

/******************************************************************************/
/*! 
  \brief Create a full non-square block matrix, A, with size nrows X ncols and
         a vector, a, with length nrows and a vector, b, with length ncols.
         compute c = a*A + b, then verify c against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_VectorTimesMatrixPlusVector)
{
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType( 3, 12, 1, 4) );
  std::vector<Plato::OrdinalType> tRowMapA = { 0, 3, 6, 9 };
  std::vector<Plato::OrdinalType> tColMapA = { 0, 1, 2, 0, 1, 2, 0, 1, 2 };
  std::vector<Plato::Scalar>      tValuesA = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
  pth::set_matrix_data(tMatrixA, tRowMapA, tColMapA, tValuesA);

  auto tMatrixA_full = pth::to_full(tMatrixA);

  Plato::ScalarVector tVector_a("a", 3);
  std::vector<Plato::Scalar> tVector_a_full({12,11,10});
  pth::set_view_from_vector<Plato::Scalar>(tVector_a, tVector_a_full);

  Plato::ScalarVector tVector_b("b", 12);
  std::vector<Plato::Scalar> tVector_b_full({12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  pth::set_view_from_vector<Plato::Scalar>(tVector_b, tVector_b_full);

  std::vector<Plato::Scalar> tVector_c_gold(12);
  for(int j=0; j<12; j++)
  {
    tVector_c_gold[j] = tVector_b_full[j];
    for(int i=0; i<3; i++)
    {
      tVector_c_gold[j] += tVector_a_full[i] * tMatrixA_full[i][j];
    }
  }

  Plato::VectorTimesMatrixPlusVector( tVector_a, tMatrixA, tVector_b );

  TEST_ASSERT(pth::is_same(tVector_b, tVector_c_gold));
}
/******************************************************************************/
/*! 
  \brief Create a full block matrix, A, and a sub-block matrix, B, and
         compute C=A-B, then verify C against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_MatrixMinusMatrix_1)
{
  Plato::OrdinalType tOffset = 3;

  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType( 12, 12, 4, 4) );
  std::vector<Plato::OrdinalType> tRowMapA = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapA = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesA = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  pth::set_matrix_data(tMatrixA, tRowMapA, tColMapA, tValuesA);

  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType(  3, 12, 1, 4) );
  std::vector<Plato::OrdinalType> tRowMapB = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapB = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesB = 
    { 13, 14, 15, 16,
      13, 14, 15, 16,
      13, 14, 15, 16,
      13, 14, 15, 16 };
  pth::set_matrix_data(tMatrixB, tRowMapB, tColMapB, tValuesB);

  Plato::MatrixMinusMatrix( tMatrixA, tMatrixB, tOffset );

  auto tMatrixC_Gold = Teuchos::rcp( new Plato::CrsMatrixType( 12, 12, 4, 4) );
  std::vector<Plato::OrdinalType> tRowMapC = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapC = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesC = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  0,  0,  0,  0,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  0,  0,  0,  0,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  0,  0,  0,  0,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  0,  0,  0,  0 };
  pth::set_matrix_data(tMatrixC_Gold, tRowMapC, tColMapC, tValuesC);

  TEST_ASSERT(pth::is_same(tMatrixA, tMatrixC_Gold));
}

/******************************************************************************/
/*! 
  \brief Invert local matrices
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_InvertLocalMatrices)
{
    const int N = 3; // Number of matrices to invert
    Plato::ScalarArray3D tMatrix("Matrix A", N, 2, 2);
    auto tHostMatrix = Kokkos::create_mirror(tMatrix);
    for (unsigned int i = 0; i < N; ++i)
    {
      const Plato::Scalar tScaleFactor = 1.0 / (1.0 + i);
      tHostMatrix(i,0,0) = -2.0 * tScaleFactor;
      tHostMatrix(i,1,0) =  1.0 * tScaleFactor;
      tHostMatrix(i,0,1) =  1.5 * tScaleFactor;
      tHostMatrix(i,1,1) = -0.5 * tScaleFactor;
    }
    Kokkos::deep_copy(tMatrix, tHostMatrix);

    Plato::ScalarArray3D tAInverse("A Inverse", N, 2, 2);
    auto tHostAInverse = Kokkos::create_mirror(tAInverse);
    for (unsigned int i = 0; i < N; ++i)
    {
      tHostAInverse(i,0,0) = 1.0;
      tHostAInverse(i,1,0) = 0.0;
      tHostAInverse(i,0,1) = 0.0;
      tHostAInverse(i,1,1) = 1.0;
    }
    Kokkos::deep_copy(tAInverse, tHostAInverse);

    using namespace KokkosBatched;

    /// [template]AlgoType: Unblocked, Blocked, CompatMKL
    /// [in/out]A: 2d view
    /// [in]tiny: a magnitude scalar value compatible to the value type of A
    /// int SerialLU<Algo::LU::Unblocked>::invoke(const AViewType &A, const ScalarType tiny = 0)

    /// [template]SideType: Side::Left or Side::Right
    /// [template]UploType: Uplo::Upper or Uplo::Lower
    /// [template]TransType: Trans::NoTranspose or Trans::Transpose
    /// [template]DiagType: Diag::Unit or Diag::NonUnit
    /// [template]AlgoType: Unblocked, Blocked, CompatMKL
    /// [in]alpha: a scalar value
    /// [in]A: 2d view
    /// [in/out]B: 2d view
    /// int SerialTrsm<SideType,UploType,TransType,DiagType,AlgoType>
    ///    ::invoke(const ScalarType alpha, const AViewType &A, const BViewType &B);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,N), KOKKOS_LAMBDA(const Plato::OrdinalType & n) {
      auto A    = Kokkos::subview(tMatrix  , n, Kokkos::ALL(), Kokkos::ALL());
      auto Ainv = Kokkos::subview(tAInverse, n, Kokkos::ALL(), Kokkos::ALL());

      SerialLU<Algo::LU::Blocked>::invoke(A);
      SerialTrsm<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::Unit   ,Algo::Trsm::Blocked>::invoke(1.0, A, Ainv);
      SerialTrsm<Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,Algo::Trsm::Blocked>::invoke(1.0, A, Ainv);
    });

    const Plato::Scalar tTolerance = 1e-6;
    std::vector<std::vector<Plato::Scalar> > tGoldMatrixInverse = { {1.0, 3.0}, {2.0, 4.0} };

    Kokkos::deep_copy(tHostAInverse, tAInverse);
    for (unsigned int n = 0; n < N; ++n)
      for (unsigned int i = 0; i < 2; ++i)
        for (unsigned int j = 0; j < 2; ++j)
          {
            //printf("Matrix %d Inverse (%d,%d) = %f\n", n, i, j, tHostAInverse(n, i, j));
            const Plato::Scalar tScaleFactor = (1.0 + n);
            TEST_FLOATING_EQUALITY(tHostAInverse(n, i, j), tScaleFactor * tGoldMatrixInverse[i][j], tTolerance);
          }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, HyperbolicTangentProjection)
{
    const Plato::OrdinalType tNumNodesPerCell = 2;
    typedef Sacado::Fad::SFad<Plato::Scalar, tNumNodesPerCell> FadType;

    // SET EVALUATION TYPES FOR UNIT TEST
    const Plato::OrdinalType tNumCells = 1;
    Plato::ScalarVectorT<Plato::Scalar> tOutputVal("OutputVal", tNumCells);
    Plato::ScalarVectorT<Plato::Scalar> tOutputGrad("OutputGrad", tNumNodesPerCell);
    Plato::ScalarMultiVectorT<FadType> tControl("Control", tNumCells, tNumNodesPerCell);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
    {
        tControl(aCellOrdinal, 0) = FadType(tNumNodesPerCell, 0, 1.0);
        tControl(aCellOrdinal, 1) = FadType(tNumNodesPerCell, 1, 1.0);
    }, "Set Controls");

    // SET EVALUATION TYPES FOR UNIT TEST
    Plato::HyperbolicTangentProjection tProjection;
    Plato::ApplyProjection<Plato::HyperbolicTangentProjection> tApplyProjection(tProjection);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
    {
        FadType tValue = tApplyProjection(aCellOrdinal, tControl);
        tOutputVal(aCellOrdinal) = tValue.val();
        tOutputGrad(0) = tValue.dx(0);
        tOutputGrad(1) = tValue.dx(1);
    }, "UnitTest: HyperbolicTangentProjection_GradZ");

    // TEST OUTPUT
    auto tHostVal = Kokkos::create_mirror(tOutputVal);
    Kokkos::deep_copy(tHostVal, tOutputVal);
    auto tHostGrad = Kokkos::create_mirror(tOutputGrad);
    Kokkos::deep_copy(tHostGrad, tOutputGrad);

    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGoldVal = { 1.0 };
    std::vector<Plato::Scalar> tGoldGrad = { 4.539992985607449e-4, 4.539992985607449e-4 };
    TEST_FLOATING_EQUALITY(tHostVal(0), tGoldVal[0], tTolerance);
    TEST_FLOATING_EQUALITY(tHostGrad(0), tGoldGrad[0], tTolerance);
    TEST_FLOATING_EQUALITY(tHostGrad(1), tGoldGrad[1], tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_ConditionalExpression)
{
    const Plato::OrdinalType tRange = 1;
    Plato::ScalarVector tOuput("Output", 2 /* number of outputs */);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tRange), KOKKOS_LAMBDA(Plato::OrdinalType tOrdinal)
    {
        Plato::Scalar tConditionalValOne = 5;
        Plato::Scalar tConditionalValTwo = 4;
        Plato::Scalar tConsequentValOne = 2;
        Plato::Scalar tConsequentValTwo = 3;
        tOuput(tOrdinal) = Plato::conditional_expression(tConditionalValOne, tConditionalValTwo, tConsequentValOne, tConsequentValTwo);

        tConditionalValOne = 3;
        tOuput(tOrdinal + 1) = Plato::conditional_expression(tConditionalValOne, tConditionalValTwo, tConsequentValOne, tConsequentValTwo);
    }, "Test inline conditional_expression function");

    auto tHostOuput = Kokkos::create_mirror(tOuput);
    Kokkos::deep_copy(tHostOuput, tOuput);
    const Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tHostOuput(0), 3.0, tTolerance);
    TEST_FLOATING_EQUALITY(tHostOuput(1), 2.0, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_dot)
{
  constexpr Plato::OrdinalType tNumElems = 10;
  Plato::ScalarVector tVecA("Vec A", tNumElems);
  Plato::blas1::fill(1.0, tVecA);
  Plato::ScalarVector tVecB("Vec B", tNumElems);
  Plato::blas1::fill(2.0, tVecB);

  const Plato::Scalar tOutput = Plato::blas1::dot(tVecA, tVecB);

  constexpr Plato::Scalar tTolerance = 1e-4;
  TEST_FLOATING_EQUALITY(20., tOutput, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_norm)
{
  constexpr Plato::OrdinalType tNumElems = 10;
  Plato::ScalarVector tVecA("Vec A", tNumElems);
  Plato::blas1::fill(1.0, tVecA);

  const Plato::Scalar tOutput = Plato::blas1::norm(tVecA);
  constexpr Plato::Scalar tTolerance = 1e-6;
  TEST_FLOATING_EQUALITY(3.16227766016838, tOutput, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_sum)
{
  constexpr Plato::OrdinalType tNumElems = 10;
  Plato::ScalarVector tVecA("Vec", tNumElems);
  Plato::blas1::fill(1.0, tVecA);

  Plato::Scalar tOutput = 0.0;
  Plato::blas1::local_sum(tVecA, tOutput);

  constexpr Plato::Scalar tTolerance = 1e-4;
  TEST_FLOATING_EQUALITY(10., tOutput, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_fill)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = pth::get_box_mesh("TRI3", meshWidth);

  int numVerts = tMesh->NumNodes();
  
  Plato::ScalarVector tSomeVector("some vector", numVerts);
  Plato::blas1::fill(2.0, tSomeVector);

  auto tSomeVectorHost = Kokkos::create_mirror_view(tSomeVector);
  Kokkos::deep_copy(tSomeVectorHost, tSomeVector);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(0), 2.0, 1e-17);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(numVerts-1), 2.0, 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_copy)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = pth::get_box_mesh("TRI3", meshWidth);

  int numVerts = tMesh->NumNodes();
  
  Plato::ScalarVector tSomeVector("some vector", numVerts);
  Plato::blas1::fill(2.0, tSomeVector);

  Plato::ScalarVector tSomeOtherVector("some other vector", numVerts);
  Plato::blas1::copy(tSomeVector, tSomeOtherVector);

  auto tSomeVectorHost = Kokkos::create_mirror_view(tSomeVector);
  Kokkos::deep_copy(tSomeVectorHost, tSomeVector);
  auto tSomeOtherVectorHost = Kokkos::create_mirror_view(tSomeOtherVector);
  Kokkos::deep_copy(tSomeOtherVectorHost, tSomeOtherVector);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(0), tSomeOtherVectorHost(0), 1e-17);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(numVerts-1), tSomeOtherVectorHost(numVerts-1), 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_scale)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = pth::get_box_mesh("TRI3", meshWidth);

  int numVerts = tMesh->NumNodes();
  
  Plato::ScalarVector tSomeVector("some vector", numVerts);
  Plato::blas1::fill(1.0, tSomeVector);
  Plato::blas1::scale(2.0, tSomeVector);

  auto tSomeVectorHost = Kokkos::create_mirror_view(tSomeVector);
  Kokkos::deep_copy(tSomeVectorHost, tSomeVector);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(0), 2.0, 1e-17);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(numVerts-1), 2.0, 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_update)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = pth::get_box_mesh("TRI3", meshWidth);

  int numVerts = tMesh->NumNodes();
  
  Plato::ScalarVector tVector_A("vector a", numVerts);
  Plato::ScalarVector tVector_B("vector b", numVerts);
  Plato::blas1::fill(1.0, tVector_A);
  Plato::blas1::fill(2.0, tVector_B);
  Plato::blas1::update(2.0, tVector_A, 3.0, tVector_B);

  auto tVector_B_Host = Kokkos::create_mirror_view(tVector_B);
  Kokkos::deep_copy(tVector_B_Host, tVector_B);
  TEST_FLOATING_EQUALITY(tVector_B_Host(0), 8.0, 1e-17);
  TEST_FLOATING_EQUALITY(tVector_B_Host(numVerts-1), 8.0, 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_MatrixTimesVectorPlusVector)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = pth::get_box_mesh("TET4", meshWidth);

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( tMesh->NumNodes(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);

  // create mesh based displacement from host data
  //
  auto stateSize = spaceDim*tMesh->NumNodes();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, stateSize);
  auto u = Kokkos::subview(U, 0, Kokkos::ALL());
  auto u_host = Kokkos::create_mirror_view(u);
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for( int i = 0; i<stateSize; i++) u_host(i) = (disp += dval);
  Kokkos::deep_copy(u, u_host);

  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Internal Elastic Energy'>                            \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
    "      <ParameterList name='Penalty Function'>                                 \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Elliptic'>                                             \n"
    "    <ParameterList name='Penalty Function'>                                   \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                  \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create criterion
  //
  Plato::DataMap tDataMap;
  std::string tMyFunction("Internal Elastic Energy");

  Plato::SpatialModel tSpatialModel(tMesh, *tParams, tDataMap);

  Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<Plato::Tet4>>
    eeScalarFunction(tSpatialModel, tDataMap, *tParams, tMyFunction);

  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto dfdx = eeScalarFunction.gradient_x(tSolution, z);

  // create PDE constraint
  //
  Plato::Elliptic::VectorFunction<::Plato::Mechanics<Plato::Tet4>>
    esVectorFunction(tSpatialModel, tDataMap, *tParams, tParams->get<std::string>("PDE Constraint"));

  auto dgdx = esVectorFunction.gradient_x(u,z);

#ifdef COMPUTE_GOLD_
  {
    auto dfdxHost = Kokkos::create_mirror_view(dfdx); 
    Kokkos::deep_copy(dfdxHost, dfdx);
    std::ofstream ofile;
    ofile.open("dfdx_before.dat");
    for(int i=0; i<dfdxHost.size(); i++) 
      ofile << std::setprecision(18) << dfdxHost(i) << std::endl;
    ofile.close();
  }
#endif

  Plato::MatrixTimesVectorPlusVector(dgdx, (Plato::ScalarVector)u, dfdx);

  auto dfdx_host = Kokkos::create_mirror_view(dfdx);
  Kokkos::deep_copy(dfdx_host, dfdx);

#ifdef COMPUTE_GOLD_
  {
    auto dfdxHost = Kokkos::create_mirror_view(dfdx); 
    Kokkos::deep_copy(dfdxHost, dfdx);
    std::ofstream ofile;
    ofile.open("dfdx_after.dat");
    for(int i=0; i<dfdxHost.size(); i++) 
      ofile << std::setprecision(18) << dfdxHost(i) << std::endl;
    ofile.close();
  }

  {
    std::ofstream ofile;
    ofile.open("u.dat");
    for(int i=0; i<u_host.size(); i++) 
      ofile << u_host(i) << std::endl;
    ofile.close();
  }

  {
    auto rowMapHost = Kokkos::create_mirror_view(dgdx->rowMap()); 
    Kokkos::deep_copy(rowMapHost, dgdx->rowMap());
    std::ofstream ofile;
    ofile.open("rowMap.dat");
    for(int i=0; i<rowMapHost.size(); i++) 
      ofile << rowMapHost(i) << std::endl;
    ofile.close();
  }

  {
    auto columnIndicesHost = Kokkos::create_mirror_view(dgdx->columnIndices());
    Kokkos::deep_copy(columnIndicesHost, dgdx->columnIndices());
    std::ofstream ofile;
    ofile.open("columnIndices.dat");
    for(int i=0; i<columnIndicesHost.size(); i++) 
      ofile << columnIndicesHost(i) << std::endl;
    ofile.close();
  }

  {
    auto entriesHost = Kokkos::create_mirror_view(dgdx->entries());
    Kokkos::deep_copy(entriesHost, dgdx->entries());
    std::ofstream ofile;
    ofile.open("entries.dat");
    for(int i=0; i<entriesHost.size(); i++) 
      ofile << std::setprecision(18) << entriesHost(i) << std::endl;
    ofile.close();
  }
#endif

  std::vector<Plato::Scalar> dfdx_gold = {
30.4874999999999758, 2.13750000000000107, -7.31249999999998934, 
29.1418269230768985, -2.32355769230767883, 5.24423076923076792, 
-1.34567307692307758, -4.46105769230768701, 12.5567307692307573, 
26.3379807692307537, 14.7980769230769234, -13.1235576923076778, 
19.4971153846153626, 18.5365384615384485, 6.17884615384615188, 
-6.84086538461538929, 3.73846153846153229, 19.3024038461538368, 
-4.14951923076923279, 12.6605769230769187, -5.81105769230769109, 
-9.64471153846153406, 20.8600961538461505, 0.934615384615383737, 
-5.49519230769230482, 8.19951923076923173, 6.74567307692307860, 
35.9826923076923606, -6.06201923076923421, -14.0581730769230902, 
38.7865384615385267, -23.1836538461538417, 4.30961538461539018, 
2.80384615384615321, -17.1216346153846146, 18.3677884615384883, 
33.1788461538462585, 11.0596153846154177, -32.4259615384615643, 
1.37667655053519411e-13, 5.32907051820075139e-14, 
1.83186799063150829e-14, -33.1788461538461377, 
-11.0596153846153840, 32.4259615384615927, -2.80384615384616342, 
17.1216346153846537, -18.3677884615384777, -38.7865384615384770, 
23.1836538461538986, -4.30961538461538662, -35.9826923076923180, 
6.06201923076924398, 14.0581730769230955, 5.49519230769232259, 
-8.19951923076924594, -6.74567307692308837, 9.64471153846156781, 
-20.8600961538461966, -0.934615384615391509, 4.14951923076924256, 
-12.6605769230769525, 5.81105769230770530, 6.84086538461540439, 
-3.73846153846155671, -19.3024038461538936, -19.4971153846154337, 
-18.5365384615385125, -6.17884615384617764, -26.3379807692308461, 
-14.7980769230769660, 13.1235576923077346, 1.34567307692308225, 
4.46105769230770743, -12.5567307692308106, -29.1418269230770193, 
2.32355769230769837, -5.24423076923078302, -30.4875000000000966, 
-2.13750000000000284, 7.31250000000002665};

  for(int iNode=0; iNode<int(dfdx_gold.size()); iNode++)
  {
      if(fabs(dfdx_gold[iNode]) < 1e-12)
      {
          TEST_ASSERT(dfdx_host[iNode] < 1e-12);
      }
      else
      {
          TEST_FLOATING_EQUALITY(dfdx_host[iNode], dfdx_gold[iNode], 1e-12);
      }
  }
}

/******************************************************************************/
/*! 
  \brief Check multiplication of block with non-block matrices.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_MatrixMatrixMultiply_3)
{
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(4, 4, 1, 1) );
  std::vector<Plato::OrdinalType> tRowMapA = { 0, 1, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapA = { 3, 1, 3, 3 };
  std::vector<Plato::Scalar>      tValuesA = { 2, 1, 3, 4 };
  pth::set_matrix_data(tMatrixA, tRowMapA, tColMapA, tValuesA);

  auto tMatrixAb = Teuchos::rcp( new Plato::CrsMatrixType(4, 4, 2, 2) );
  std::vector<Plato::OrdinalType> tRowMapAb = { 0, 2, 3 };
  std::vector<Plato::OrdinalType> tColMapAb = { 0, 1, 1 };
  std::vector<Plato::Scalar>      tValuesAb = { 0, 0, 0, 1, 0, 2, 0, 0, 0, 3, 0, 4 };
  pth::set_matrix_data(tMatrixAb, tRowMapAb, tColMapAb, tValuesAb);

  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType(4, 4, 2, 2) );
  std::vector<Plato::OrdinalType> tRowMapB = { 0, 2, 4 };
  std::vector<Plato::OrdinalType> tColMapB = { 0, 1, 0, 1 };
  std::vector<Plato::Scalar>      tValuesB = 
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
  pth::set_matrix_data(tMatrixB, tRowMapB, tColMapB, tValuesB);

  auto tMatrixBANonBlock      = Teuchos::rcp( new Plato::CrsMatrixType(4, 4, 1, 1) );
  auto tMatrixBABlock         = Teuchos::rcp( new Plato::CrsMatrixType(4, 4, 2, 2) );
  auto tMatrixBAbBlock        = Teuchos::rcp( new Plato::CrsMatrixType(4, 4, 2, 2) );

  Plato::MatrixMatrixMultiply             ( tMatrixB, tMatrixA, tMatrixBANonBlock);
  Plato::MatrixMatrixMultiply             ( tMatrixB, tMatrixA, tMatrixBABlock);
  Plato::MatrixMatrixMultiply             ( tMatrixB, tMatrixAb, tMatrixBAbBlock);

  auto tGoldMatrix = Teuchos::rcp( new Plato::CrsMatrixType(4, 4, 1, 1) );
  std::vector<Plato::OrdinalType> tGoldMatrixRowMap = { 0, 2, 4, 6, 8 };
  std::vector<Plato::OrdinalType> tGoldMatrixColMap = { 1, 3, 1, 3, 1, 3, 1, 3 };
  std::vector<Plato::Scalar> tGoldMatrixEntries = { 1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 1.0, 9.0 };
  pth::set_matrix_data(tGoldMatrix, tGoldMatrixRowMap, tGoldMatrixColMap, tGoldMatrixEntries);

  auto tGoldMatrixBlock = Teuchos::rcp( new Plato::CrsMatrixType(4, 4, 2, 2) );
  std::vector<Plato::OrdinalType> tGoldMatrixBlockRowMap = { 0, 2, 4 };
  std::vector<Plato::OrdinalType> tGoldMatrixBlockColMap = { 0, 1, 0, 1 };
  std::vector<Plato::Scalar> tGoldMatrixBlockEntries = { 
      0.0, 1.0, 0.0, 1.0, 0.0, 9.0, 0.0, 9.0,
      0.0, 1.0, 0.0, 1.0, 0.0, 9.0, 0.0, 9.0
  };
  pth::set_matrix_data(tGoldMatrixBlock, tGoldMatrixBlockRowMap, tGoldMatrixBlockColMap, tGoldMatrixBlockEntries);

  auto tGoldMatrixBlockAlg = Teuchos::rcp( new Plato::CrsMatrixType(4, 4, 2, 2) );
  Plato::ScalarVectorT<Plato::OrdinalType> tGoldRowMap = tGoldMatrix->rowMap();
  Plato::ScalarVectorT<Plato::OrdinalType> tGoldColMap = tGoldMatrix->columnIndices();
  Plato::ScalarVectorT<Plato::Scalar> tGoldEntries = tGoldMatrix->entries();
  Plato::setDataFromNonBlock(tGoldMatrixBlockAlg, tGoldRowMap, tGoldColMap, tGoldEntries);

  TEST_ASSERT(pth::is_same(tMatrixBANonBlock->rowMap(), tGoldMatrix->rowMap()));
  TEST_ASSERT(pth::is_equivalent(tMatrixBANonBlock->rowMap(),
                            tMatrixBANonBlock->columnIndices(), tMatrixBANonBlock->entries(),
                            tGoldMatrix->columnIndices(), tGoldMatrix->entries()));

  TEST_ASSERT(pth::is_same(tMatrixBABlock->rowMap(), tGoldMatrixBlockAlg->rowMap()));
  TEST_ASSERT(pth::is_equivalent(tMatrixBABlock->rowMap(),
                            tMatrixBABlock->columnIndices(), tMatrixBABlock->entries(),
                            tGoldMatrixBlockAlg->columnIndices(), tGoldMatrixBlockAlg->entries()));

  TEST_ASSERT(pth::is_same(tMatrixBAbBlock->rowMap(), tGoldMatrixBlock->rowMap()));
  TEST_ASSERT(pth::is_equivalent(tMatrixBAbBlock->rowMap(),
                            tMatrixBAbBlock->columnIndices(), tMatrixBAbBlock->entries(),
                            tGoldMatrixBlock->columnIndices(), tGoldMatrixBlock->entries()));

}

/******************************************************************************/
/*! 
  \brief Check multiplication of block rectangular matrices with gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_MatrixMatrixMultiply_Rect2)
{
  auto tMatrixA1 = Teuchos::rcp( new Plato::CrsMatrixType(2, 4, 2, 2) );
  std::vector<Plato::OrdinalType> tRowMapA1 = { 0, 2 };
  std::vector<Plato::OrdinalType> tColMapA1 = { 0, 1 };
  std::vector<Plato::Scalar>      tValuesA1 = { 0, 1, 2, 0, 0, 3, 1, 0 };
  pth::set_matrix_data(tMatrixA1, tRowMapA1, tColMapA1, tValuesA1);

  auto tMatrixA2 = Teuchos::rcp( new Plato::CrsMatrixType(4, 4, 2, 2) );
  std::vector<Plato::OrdinalType> tRowMapA2 = { 0, 2, 4 };
  std::vector<Plato::OrdinalType> tColMapA2 = { 0, 1, 0, 1 };
  std::vector<Plato::Scalar>      tValuesA2 = 
    { 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4 };
  pth::set_matrix_data(tMatrixA2, tRowMapA2, tColMapA2, tValuesA2);

  auto tMatrixB1 = Teuchos::rcp( new Plato::CrsMatrixType(2, 2, 2, 2) );
  std::vector<Plato::OrdinalType> tRowMapB1 = { 0, 1 };
  std::vector<Plato::OrdinalType> tColMapB1 = { 0 };
  std::vector<Plato::Scalar>      tValuesB1 = { 1, 1, 1, 1 };
  pth::set_matrix_data(tMatrixB1, tRowMapB1, tColMapB1, tValuesB1);

  auto tMatrixB2 = Teuchos::rcp( new Plato::CrsMatrixType(4, 4, 2, 2) );
  std::vector<Plato::OrdinalType> tRowMapB2 = { 0, 2, 4 };
  std::vector<Plato::OrdinalType> tColMapB2 = { 0, 1, 0, 1 };
  std::vector<Plato::Scalar>      tValuesB2 = 
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
  pth::set_matrix_data(tMatrixB2, tRowMapB2, tColMapB2, tValuesB2);

  auto tMatrixB3 = Teuchos::rcp( new Plato::CrsMatrixType(4, 2, 2, 2) );
  std::vector<Plato::OrdinalType> tRowMapB3 = { 0, 1, 2 };
  std::vector<Plato::OrdinalType> tColMapB3 = { 0, 0 };
  std::vector<Plato::Scalar>      tValuesB3 = { 1, 1, 1, 1, 1, 1, 1, 1 };
  pth::set_matrix_data(tMatrixB3, tRowMapB3, tColMapB3, tValuesB3);

  auto tMatrixB1A1      = Teuchos::rcp( new Plato::CrsMatrixType(2, 4, 2, 2) );
  auto tMatrixA1B2      = Teuchos::rcp( new Plato::CrsMatrixType(2, 4, 2, 2) );
  auto tMatrixA1B3      = Teuchos::rcp( new Plato::CrsMatrixType(2, 2, 2, 2) );
  auto tMatrixA2B3      = Teuchos::rcp( new Plato::CrsMatrixType(4, 2, 2, 2) );

  Plato::MatrixMatrixMultiply             ( tMatrixB1, tMatrixA1, tMatrixB1A1);
  Plato::MatrixMatrixMultiply             ( tMatrixA1, tMatrixB2, tMatrixA1B2);
  Plato::MatrixMatrixMultiply             ( tMatrixA1, tMatrixB3, tMatrixA1B3);
  Plato::MatrixMatrixMultiply             ( tMatrixA2, tMatrixB3, tMatrixA2B3);

  auto tGoldMatrix1 = Teuchos::rcp( new Plato::CrsMatrixType(2, 4, 2, 2) );
  std::vector<Plato::OrdinalType> tGoldMatrixRowMap1 = { 0, 2 };
  std::vector<Plato::OrdinalType> tGoldMatrixColMap1 = { 0, 1 };
  std::vector<Plato::Scalar> tGoldMatrixEntries1 = { 2.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 3.0 };
  pth::set_matrix_data(tGoldMatrix1, tGoldMatrixRowMap1, tGoldMatrixColMap1, tGoldMatrixEntries1);

  auto tGoldMatrix2 = Teuchos::rcp( new Plato::CrsMatrixType(2, 4, 2, 2) );
  std::vector<Plato::OrdinalType> tGoldMatrixRowMap2 = { 0, 2 };
  std::vector<Plato::OrdinalType> tGoldMatrixColMap2 = { 0, 1 };
  std::vector<Plato::Scalar> tGoldMatrixEntries2 = { 4.0, 4.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0 };
  pth::set_matrix_data(tGoldMatrix2, tGoldMatrixRowMap2, tGoldMatrixColMap2, tGoldMatrixEntries2);

  auto tGoldMatrix3 = Teuchos::rcp( new Plato::CrsMatrixType(2, 2, 2, 2) );
  std::vector<Plato::OrdinalType> tGoldMatrixRowMap3 = { 0, 1 };
  std::vector<Plato::OrdinalType> tGoldMatrixColMap3 = { 0 };
  std::vector<Plato::Scalar> tGoldMatrixEntries3 = { 4.0, 4.0, 3.0, 3.0 };
  pth::set_matrix_data(tGoldMatrix3, tGoldMatrixRowMap3, tGoldMatrixColMap3, tGoldMatrixEntries3);

  auto tGoldMatrix4 = Teuchos::rcp( new Plato::CrsMatrixType(4, 2, 2, 2) );
  std::vector<Plato::OrdinalType> tGoldMatrixRowMap4 = { 0, 1, 2 };
  std::vector<Plato::OrdinalType> tGoldMatrixColMap4 = { 0, 0 };
  std::vector<Plato::Scalar> tGoldMatrixEntries4 = { 2.0, 2.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0 };
  pth::set_matrix_data(tGoldMatrix4, tGoldMatrixRowMap4, tGoldMatrixColMap4, tGoldMatrixEntries4);


  TEST_ASSERT(pth::is_same(tMatrixB1A1->rowMap(), tGoldMatrix1->rowMap()));
  TEST_ASSERT(pth::is_equivalent(tMatrixB1A1->rowMap(),
                            tMatrixB1A1->columnIndices(), tMatrixB1A1->entries(),
                            tGoldMatrix1->columnIndices(), tGoldMatrix1->entries()));

  TEST_ASSERT(pth::is_same(tMatrixA1B2->rowMap(), tGoldMatrix2->rowMap()));
  TEST_ASSERT(pth::is_equivalent(tMatrixA1B2->rowMap(),
                            tMatrixA1B2->columnIndices(), tMatrixA1B2->entries(),
                            tGoldMatrix2->columnIndices(), tGoldMatrix2->entries()));

  TEST_ASSERT(pth::is_same(tMatrixA1B3->rowMap(), tGoldMatrix3->rowMap()));
  TEST_ASSERT(pth::is_equivalent(tMatrixA1B3->rowMap(),
                            tMatrixA1B3->columnIndices(), tMatrixA1B3->entries(),
                            tGoldMatrix3->columnIndices(), tGoldMatrix3->entries()));

  TEST_ASSERT(pth::is_same(tMatrixA2B3->rowMap(), tGoldMatrix4->rowMap()));
  TEST_ASSERT(pth::is_equivalent(tMatrixA2B3->rowMap(),
                            tMatrixA2B3->columnIndices(), tMatrixA2B3->entries(),
                            tGoldMatrix4->columnIndices(), tGoldMatrix4->entries()));

}

/******************************************************************************/
/*! 
  \brief Check multiplication of non-block rectangular matrices using Kokkos SPGEMM
  with slow dumb.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_MatrixMatrixMultiply_Rect3)
{
  auto tMatrixA1 = Teuchos::rcp( new Plato::CrsMatrixType(2, 4, 1, 1) );
  std::vector<Plato::OrdinalType> tRowMapA1 = { 0, 4, 8 };
  std::vector<Plato::OrdinalType> tColMapA1 = { 0, 1, 2, 3, 0, 1, 2, 3 };
  std::vector<Plato::Scalar>      tValuesA1 = { 0, 1, 0, 3, 2, 0, 1, 0 };
  pth::set_matrix_data(tMatrixA1, tRowMapA1, tColMapA1, tValuesA1);

  auto tMatrixA2 = Teuchos::rcp( new Plato::CrsMatrixType(4, 4, 1, 1) );
  std::vector<Plato::OrdinalType> tRowMapA2 = { 0, 4, 8, 12, 16 };
  std::vector<Plato::OrdinalType> tColMapA2 = 
    { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 };
  std::vector<Plato::Scalar>      tValuesA2 = 
    { 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4 };
  pth::set_matrix_data(tMatrixA2, tRowMapA2, tColMapA2, tValuesA2);

  auto tMatrixB1 = Teuchos::rcp( new Plato::CrsMatrixType(2, 2, 1, 1) );
  std::vector<Plato::OrdinalType> tRowMapB1 = { 0, 2, 4 };
  std::vector<Plato::OrdinalType> tColMapB1 = { 0, 1, 0, 1 };
  std::vector<Plato::Scalar>      tValuesB1 = { 1, 1, 1, 1 };
  pth::set_matrix_data(tMatrixB1, tRowMapB1, tColMapB1, tValuesB1);

  /* auto tMatrixB2 = Teuchos::rcp( new Plato::CrsMatrixType(4, 4, 2, 2) ); */
  /* std::vector<Plato::OrdinalType> tRowMapB2 = { 0, 2, 4 }; */
  /* std::vector<Plato::OrdinalType> tColMapB2 = { 0, 1, 0, 1 }; */
  /* std::vector<Plato::Scalar>      tValuesB2 = */ 
  /*   { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }; */
  /* pth::set_matrix_data(tMatrixB2, tRowMapB2, tColMapB2, tValuesB2); */

  auto tMatrixB3 = Teuchos::rcp( new Plato::CrsMatrixType(4, 2, 1, 1) );
  std::vector<Plato::OrdinalType> tRowMapB3 = { 0, 2, 4, 6, 8 };
  std::vector<Plato::OrdinalType> tColMapB3 = { 0, 1, 0, 1, 0, 1, 0, 1 };
  std::vector<Plato::Scalar>      tValuesB3 = { 1, 1, 1, 1, 1, 1, 1, 1 };
  pth::set_matrix_data(tMatrixB3, tRowMapB3, tColMapB3, tValuesB3);

  auto tMatrixB1A1      = Teuchos::rcp( new Plato::CrsMatrixType(2, 4, 1, 1) );
  /* auto tMatrixA1B2      = Teuchos::rcp( new Plato::CrsMatrixType(2, 4, 2, 2) ); */
  auto tMatrixA1B3      = Teuchos::rcp( new Plato::CrsMatrixType(2, 2, 1, 1) );
  auto tMatrixA2B3      = Teuchos::rcp( new Plato::CrsMatrixType(4, 2, 1, 1) );

  // Set up Kokkos SPGEMM 
  typedef Plato::ScalarVectorT<Plato::OrdinalType> OrdinalView;
  typedef Plato::ScalarVectorT<Plato::Scalar>  ScalarView;

  typedef KokkosKernels::Experimental::KokkosKernelsHandle
      <Plato::OrdinalType, Plato::OrdinalType, Plato::Scalar,
      typename Plato::ExecSpace, 
      typename Plato::MemSpace,
      typename Plato::MemSpace > KernelHandle;

  KernelHandle tKernel;
  tKernel.set_team_work_size(1);
  tKernel.set_dynamic_scheduling(false);
  SPGEMMAlgorithm aAlgorithm = SPGEMM_KK_SPEED;
  tKernel.create_spgemm_handle(aAlgorithm);

  // B1A1
  const Plato::OrdinalType tNumRowsOneB1A1 = tMatrixB1->numRows();
  const Plato::OrdinalType tNumColsOneB1A1 = tMatrixB1->numCols();
  const Plato::OrdinalType tNumRowsTwoB1A1 = tMatrixA1->numRows();
  const Plato::OrdinalType tNumColsTwoB1A1 = tMatrixA1->numCols();
  const Plato::OrdinalType tNumRowsOutB1A1 = tMatrixB1A1->numRows();
  const Plato::OrdinalType tNumColsOutB1A1 = tMatrixB1A1->numCols();

  ScalarView  tMatOneValuesB1A1 = tMatrixB1->entries();
  OrdinalView tMatOneRowMapB1A1 = tMatrixB1->rowMap();
  OrdinalView tMatOneColMapB1A1 = tMatrixB1->columnIndices();

  ScalarView  tMatTwoValuesB1A1 = tMatrixA1->entries();
  OrdinalView tMatTwoRowMapB1A1 = tMatrixA1->rowMap();
  OrdinalView tMatTwoColMapB1A1 = tMatrixA1->columnIndices();

  OrdinalView tOutRowMapB1A1 ("output row map", tNumRowsOneB1A1 + 1);
  spgemm_symbolic ( &tKernel, tNumRowsOneB1A1, tNumRowsTwoB1A1, tNumColsTwoB1A1,
      tMatOneRowMapB1A1, tMatOneColMapB1A1, /*transpose=*/false,
      tMatTwoRowMapB1A1, tMatTwoColMapB1A1, /*transpose=*/false,
      tOutRowMapB1A1
  );

  OrdinalView tOutColMapB1A1;
  ScalarView  tOutValuesB1A1;
  size_t tNumOutValuesB1A1 = tKernel.get_spgemm_handle()->get_c_nnz();
  if (tNumOutValuesB1A1){
    tOutColMapB1A1 = OrdinalView(Kokkos::ViewAllocateWithoutInitializing("out column map"), tNumOutValuesB1A1);
    tOutValuesB1A1 = ScalarView (Kokkos::ViewAllocateWithoutInitializing("out values"),  tNumOutValuesB1A1);
  }
  spgemm_numeric( &tKernel, tNumRowsOneB1A1, tNumRowsTwoB1A1, tNumColsTwoB1A1,
      tMatOneRowMapB1A1, tMatOneColMapB1A1, tMatOneValuesB1A1, /*transpose=*/false,
      tMatTwoRowMapB1A1, tMatTwoColMapB1A1, tMatTwoValuesB1A1, /*transpose=*/false,
      tOutRowMapB1A1, tOutColMapB1A1, tOutValuesB1A1
  );

  tMatrixB1A1->setRowMap(tOutRowMapB1A1);
  tMatrixB1A1->setColumnIndices(tOutColMapB1A1);
  tMatrixB1A1->setEntries(tOutValuesB1A1);
  
  // A1B3
  const Plato::OrdinalType tNumRowsOneA1B3 = tMatrixA1->numRows();
  const Plato::OrdinalType tNumColsOneA1B3 = tMatrixA1->numCols();
  const Plato::OrdinalType tNumRowsTwoA1B3 = tMatrixB3->numRows();
  const Plato::OrdinalType tNumColsTwoA1B3 = tMatrixB3->numCols();
  const Plato::OrdinalType tNumRowsOutA1B3 = tMatrixA1B3->numRows();
  const Plato::OrdinalType tNumColsOutA1B3 = tMatrixA1B3->numCols();

  ScalarView  tMatOneValuesA1B3 = tMatrixA1->entries();
  OrdinalView tMatOneRowMapA1B3 = tMatrixA1->rowMap();
  OrdinalView tMatOneColMapA1B3 = tMatrixA1->columnIndices();

  ScalarView  tMatTwoValuesA1B3 = tMatrixB3->entries();
  OrdinalView tMatTwoRowMapA1B3 = tMatrixB3->rowMap();
  OrdinalView tMatTwoColMapA1B3 = tMatrixB3->columnIndices();

  OrdinalView tOutRowMapA1B3 ("output row map", tNumRowsOneA1B3 + 1);
  spgemm_symbolic ( &tKernel, tNumRowsOneA1B3, tNumRowsTwoA1B3, tNumColsTwoA1B3,
      tMatOneRowMapA1B3, tMatOneColMapA1B3, /*transpose=*/false,
      tMatTwoRowMapA1B3, tMatTwoColMapA1B3, /*transpose=*/false,
      tOutRowMapA1B3
  );

  OrdinalView tOutColMapA1B3;
  ScalarView  tOutValuesA1B3;
  size_t tNumOutValuesA1B3 = tKernel.get_spgemm_handle()->get_c_nnz();
  if (tNumOutValuesA1B3){
    tOutColMapA1B3 = OrdinalView(Kokkos::ViewAllocateWithoutInitializing("out column map"), tNumOutValuesA1B3);
    tOutValuesA1B3 = ScalarView (Kokkos::ViewAllocateWithoutInitializing("out values"),  tNumOutValuesA1B3);
  }
  spgemm_numeric( &tKernel, tNumRowsOneA1B3, tNumRowsTwoA1B3, tNumColsTwoA1B3,
      tMatOneRowMapA1B3, tMatOneColMapA1B3, tMatOneValuesA1B3, /*transpose=*/false,
      tMatTwoRowMapA1B3, tMatTwoColMapA1B3, tMatTwoValuesA1B3, /*transpose=*/false,
      tOutRowMapA1B3, tOutColMapA1B3, tOutValuesA1B3
  );

  tMatrixA1B3->setRowMap(tOutRowMapA1B3);
  tMatrixA1B3->setColumnIndices(tOutColMapA1B3);
  tMatrixA1B3->setEntries(tOutValuesA1B3);
  
  // A2B3
  const Plato::OrdinalType tNumRowsOneA2B3 = tMatrixA2->numRows();
  const Plato::OrdinalType tNumColsOneA2B3 = tMatrixA2->numCols();
  const Plato::OrdinalType tNumRowsTwoA2B3 = tMatrixB3->numRows();
  const Plato::OrdinalType tNumColsTwoA2B3 = tMatrixB3->numCols();
  const Plato::OrdinalType tNumRowsOutA2B3 = tMatrixA2B3->numRows();
  const Plato::OrdinalType tNumColsOutA2B3 = tMatrixA2B3->numCols();

  ScalarView  tMatOneValuesA2B3 = tMatrixA2->entries();
  OrdinalView tMatOneRowMapA2B3 = tMatrixA2->rowMap();
  OrdinalView tMatOneColMapA2B3 = tMatrixA2->columnIndices();

  ScalarView  tMatTwoValuesA2B3 = tMatrixB3->entries();
  OrdinalView tMatTwoRowMapA2B3 = tMatrixB3->rowMap();
  OrdinalView tMatTwoColMapA2B3 = tMatrixB3->columnIndices();

  OrdinalView tOutRowMapA2B3 ("output row map", tNumRowsOneA2B3 + 1);
  spgemm_symbolic ( &tKernel, tNumRowsOneA2B3, tNumRowsTwoA2B3, tNumColsTwoA2B3,
      tMatOneRowMapA2B3, tMatOneColMapA2B3, /*transpose=*/false,
      tMatTwoRowMapA2B3, tMatTwoColMapA2B3, /*transpose=*/false,
      tOutRowMapA2B3
  );

  OrdinalView tOutColMapA2B3;
  ScalarView  tOutValuesA2B3;
  size_t tNumOutValuesA2B3 = tKernel.get_spgemm_handle()->get_c_nnz();
  if (tNumOutValuesA2B3){
    tOutColMapA2B3 = OrdinalView(Kokkos::ViewAllocateWithoutInitializing("out column map"), tNumOutValuesA2B3);
    tOutValuesA2B3 = ScalarView (Kokkos::ViewAllocateWithoutInitializing("out values"),  tNumOutValuesA2B3);
  }
  spgemm_numeric( &tKernel, tNumRowsOneA2B3, tNumRowsTwoA2B3, tNumColsTwoA2B3,
      tMatOneRowMapA2B3, tMatOneColMapA2B3, tMatOneValuesA2B3, /*transpose=*/false,
      tMatTwoRowMapA2B3, tMatTwoColMapA2B3, tMatTwoValuesA2B3, /*transpose=*/false,
      tOutRowMapA2B3, tOutColMapA2B3, tOutValuesA2B3
  );

  tMatrixA2B3->setRowMap(tOutRowMapA2B3);
  tMatrixA2B3->setColumnIndices(tOutColMapA2B3);
  tMatrixA2B3->setEntries(tOutValuesA2B3);

  tKernel.destroy_spgemm_handle();

  // Slow Dumb MatrixMatrixMultiply
  auto tSlowDumbMatrixB1A1      = Teuchos::rcp( new Plato::CrsMatrixType(2, 4, 1, 1) );
  /* auto tSlowDumbMatrixA1B2      = Teuchos::rcp( new Plato::CrsMatrixType(2, 4, 2, 2) ); */
  auto tSlowDumbMatrixA1B3      = Teuchos::rcp( new Plato::CrsMatrixType(2, 2, 1, 1) );
  auto tSlowDumbMatrixA2B3      = Teuchos::rcp( new Plato::CrsMatrixType(4, 2, 1, 1) );

  pth::slow_dumb_matrix_matrix_multiply( tMatrixB1, tMatrixA1, tSlowDumbMatrixB1A1);
  /* pth::slow_dumb_matrix_matrix_multiply( tMatrixA1, tMatrixB2, tSlowDumbMatrixA1B2); */
  pth::slow_dumb_matrix_matrix_multiply( tMatrixA1, tMatrixB3, tSlowDumbMatrixA1B3);
  pth::slow_dumb_matrix_matrix_multiply( tMatrixA2, tMatrixB3, tSlowDumbMatrixA2B3);

  TEST_ASSERT(pth::is_same(tMatrixB1A1->rowMap(), tSlowDumbMatrixB1A1->rowMap()));
  TEST_ASSERT(pth::is_equivalent(tMatrixB1A1->rowMap(),
                            tMatrixB1A1->columnIndices(), tMatrixB1A1->entries(),
                            tSlowDumbMatrixB1A1->columnIndices(), tSlowDumbMatrixB1A1->entries()));

  /* TEST_ASSERT(pth::is_same(tMatrixA1B2->rowMap(), tSlowDumbMatrixA1B2->rowMap())); */
  /* TEST_ASSERT(pth::is_equivalent(tMatrixA1B2->rowMap(), */
  /*                           tMatrixA1B2->columnIndices(), tMatrixA1B2->entries(), */
  /*                           tSlowDumbMatrixA1B2->columnIndices(), tSlowDumbMatrixA1B2->entries())); */

  TEST_ASSERT(pth::is_same(tMatrixA1B3->rowMap(), tSlowDumbMatrixA1B3->rowMap()));
  TEST_ASSERT(pth::is_equivalent(tMatrixA1B3->rowMap(),
                                 tMatrixA1B3->columnIndices(), tMatrixA1B3->entries(),
                                 tSlowDumbMatrixA1B3->columnIndices(), tSlowDumbMatrixA1B3->entries()));

  TEST_ASSERT(pth::is_same(tMatrixA2B3->rowMap(), tSlowDumbMatrixA2B3->rowMap()));
  TEST_ASSERT(pth::is_equivalent(tMatrixA2B3->rowMap(),
                                 tMatrixA2B3->columnIndices(), tMatrixA2B3->entries(),
                                 tSlowDumbMatrixA2B3->columnIndices(), tSlowDumbMatrixA2B3->entries()));

}

/******************************************************************************/
/*! 
 \brief create rectangular block matrices A and B = Transpose(A) and check 
 Transpose(A) = B as well as Transpose(Transpose(A)) = A.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_TransposeBlockMatrix)
{
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(12, 6, 4, 2) );
  std::vector<Plato::OrdinalType> tRowMapA = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapA = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesA = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
  pth::set_matrix_data(tMatrixA, tRowMapA, tColMapA, tValuesA);

  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType(6, 12, 2, 4) );
  std::vector<Plato::OrdinalType> tRowMapB = { 0, 2, 2, 4 };
  std::vector<Plato::OrdinalType> tColMapB = { 0, 1, 0, 2 };
  std::vector<Plato::Scalar>      tValuesB = 
    { 1, 3, 5, 7, 2, 4, 6, 8, 1, 3, 5, 7, 2, 4, 6, 8,
      1, 3, 5, 7, 2, 4, 6, 8, 1, 3, 5, 7, 2, 4, 6, 8 };
  pth::set_matrix_data(tMatrixB, tRowMapB, tColMapB, tValuesB);

  auto tNumRows = tMatrixA->numRows();
  auto tNumCols = tMatrixA->numCols();
  auto tNumRowsPerBlock = tMatrixA->numRowsPerBlock();
  auto tNumColsPerBlock = tMatrixA->numColsPerBlock();
  auto tMatrixAT = Teuchos::rcp( new Plato::CrsMatrixType( tNumCols, tNumRows, tNumColsPerBlock, tNumRowsPerBlock) );
  Plato::MatrixTranspose(tMatrixA, tMatrixAT);

  auto tMatrixATT = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock) );
  Plato::MatrixTranspose(tMatrixAT, tMatrixATT);

  TEST_ASSERT(pth::is_same(tMatrixAT, tMatrixB));

  TEST_ASSERT(pth::is_same(tMatrixA, tMatrixATT));
}

/******************************************************************************/
/*! 
 \brief create rectangular non block matrices A and B = Transpose(A) and check 
 Transpose(A) = B as well as Transpose(Transpose(A)) = A.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_TransposeNonBlockMatrix)
{
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(4, 7, 1, 1) );
  std::vector<Plato::OrdinalType> tRowMapA = { 0, 2, 3, 6, 8 };
  std::vector<Plato::OrdinalType> tColMapA = { 0, 5, 2, 1, 4, 5, 3, 6 };
  std::vector<Plato::Scalar>      tValuesA = { 1, 2, 3, 4, 5, 6, 7, 8 };
  pth::set_matrix_data(tMatrixA, tRowMapA, tColMapA, tValuesA);

  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType(7, 4, 1, 1) );
  std::vector<Plato::OrdinalType> tRowMapB = { 0, 1, 2, 3, 4, 5, 7, 8 };
  std::vector<Plato::OrdinalType> tColMapB = { 0, 2, 1, 3, 2, 0, 2, 3 };
  std::vector<Plato::Scalar>      tValuesB = { 1, 4, 3, 7, 5, 2, 6, 8 };
  pth::set_matrix_data(tMatrixB, tRowMapB, tColMapB, tValuesB);

  auto tNumRows = tMatrixA->numRows();
  auto tNumCols = tMatrixA->numCols();
  auto tNumRowsPerBlock = tMatrixA->numRowsPerBlock();
  auto tNumColsPerBlock = tMatrixA->numColsPerBlock();
  auto tMatrixAT = Teuchos::rcp( new Plato::CrsMatrixType( tNumCols, tNumRows, tNumColsPerBlock, tNumRowsPerBlock) );
  Plato::MatrixTranspose(tMatrixA, tMatrixAT);

  auto tMatrixATT = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock) );
  Plato::MatrixTranspose(tMatrixAT, tMatrixATT);

  TEST_ASSERT(pth::is_same(tMatrixAT, tMatrixB));
}

/******************************************************************************/
/*! 
 \brief create symmetric block matrices A and B, A==B.  Scale the diagonal of
 B by 2.0 and compare against A.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_diagonalAveAbs)
{
  auto tSparseMatrix = Teuchos::rcp( new Plato::CrsMatrixType( /*tNumRows=*/6, /*tNumCols=*/6, /*tNumRowsPerBlock=*/3, /*tNumColsPerBlock=*/3) );

  {
    std::vector<std::vector<Plato::Scalar>> tFullMatrix = {{1, 0, 0, 0, 0, 0},
                                                           {0, 1, 0, 0, 0, 0},
                                                           {0, 0, 1, 0, 0, 0},
                                                           {0, 0, 0, 1, 0, 0},
                                                           {0, 0, 0, 0, 1, 0},
                                                           {0, 0, 0, 0, 0, 1}};
    pth::from_full(tSparseMatrix, tFullMatrix);

    auto tDiagonalAveAbs = diagonalAveAbs(*tSparseMatrix);

    decltype(tDiagonalAveAbs) tGold(1.0);
    TEST_FLOATING_EQUALITY(tDiagonalAveAbs, tGold, DBL_EPSILON);
  }

  {
    std::vector<std::vector<Plato::Scalar>> tFullMatrix = {{1, 0, 0, 0, 0, 0},
                                                           {0,-1, 0, 0, 0, 0},
                                                           {0, 0, 1, 0, 0, 0},
                                                           {0, 0, 0,-1, 0, 0},
                                                           {0, 0, 0, 0,-1, 0},
                                                           {0, 0, 0, 0, 0, 1}};
    pth::from_full(tSparseMatrix, tFullMatrix);

    auto tDiagonalAveAbs = diagonalAveAbs(*tSparseMatrix);

    decltype(tDiagonalAveAbs) tGold(1.0);
    TEST_FLOATING_EQUALITY(tDiagonalAveAbs, tGold, DBL_EPSILON);
  }

  {
    std::vector<std::vector<Plato::Scalar>> tFullMatrix = {{1, 0, 2, 0, 2, 0},
                                                           {0,-1, 0, 0, 0, 0},
                                                           {0, 0, 1, 0, 2, 0},
                                                           {0, 0, 0,-1, 0, 0},
                                                           {0, 0, 0, 0,-1, 0},
                                                           {0, 0, 0, 0, 0, 1}};
    pth::from_full(tSparseMatrix, tFullMatrix);

    auto tDiagonalAveAbs = diagonalAveAbs(*tSparseMatrix);

    decltype(tDiagonalAveAbs) tGold(1.0);
    TEST_FLOATING_EQUALITY(tDiagonalAveAbs, tGold, DBL_EPSILON);
  }
};

/******************************************************************************/
/*! 
 \brief create symmetric block matrices A and B, A==B.  Scale the diagonal of
 B by 2.0 and compare against A.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_shiftDiagonal)
{
  auto tMatrixA = createSquareMatrix();
  auto tMatrixB = createSquareMatrix();

  shiftDiagonal(*tMatrixB, 2.0);

  auto tMatrixA_full = pth::to_full(tMatrixA);
  auto tMatrixB_full = pth::to_full(tMatrixB);

  for(int i=0; i<tMatrixA_full.size(); i++)
  {
    for(int j=0; j<tMatrixA_full[i].size(); j++)
    {
      if( i==j )
      {
        TEST_FLOATING_EQUALITY(tMatrixA_full[i][j]+2.0, tMatrixB_full[i][j], DBL_EPSILON);
      }
      else
      {
        TEST_FLOATING_EQUALITY(tMatrixA_full[i][j], tMatrixB_full[i][j], DBL_EPSILON);
      }
    }
  }
};


/******************************************************************************/
/*! 
 \brief create symmetric block matrix A and check Transpose(A) = A and
 Transpose(Transpose(A)) = A.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_TransposeSymmetricMatrix)
{
  const auto tMatrixA = createSquareMatrix();

  const auto tNumRows = tMatrixA->numRows();
  const auto tNumCols = tMatrixA->numCols();
  const auto tNumRowsPerBlock = tMatrixA->numRowsPerBlock();
  const auto tNumColsPerBlock = tMatrixA->numColsPerBlock();
  auto tMatrixAT = Teuchos::rcp( new Plato::CrsMatrixType( tNumCols, tNumRows, tNumColsPerBlock, tNumRowsPerBlock) );
  Plato::MatrixTranspose(tMatrixA, tMatrixAT);

  auto tMatrixATT = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock) );
  Plato::MatrixTranspose(tMatrixAT, tMatrixATT);

  TEST_ASSERT(pth::is_same(tMatrixA->rowMap(), tMatrixAT->rowMap()));
  TEST_ASSERT(pth::is_equivalent(tMatrixA->rowMap(),
                                 tMatrixA->columnIndices(), tMatrixA->entries(),
                                 tMatrixAT->columnIndices(), tMatrixAT->entries()));

  TEST_ASSERT(pth::is_same(tMatrixA->rowMap(), tMatrixATT->rowMap()));
  TEST_ASSERT(pth::is_equivalent(tMatrixA->rowMap(),
                                 tMatrixA->columnIndices(), tMatrixA->entries(),
                                 tMatrixATT->columnIndices(), tMatrixATT->entries()));
}

} // namespace PlatoUnitTests
