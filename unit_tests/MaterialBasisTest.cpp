#include "util/PlatoTestHelpers.hpp"
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "ParseTools.hpp"
#include "MaterialBasis.hpp"
#include "PlatoMathTypes.hpp"

namespace PlatoUnitTests
{

/******************************************************************************/
/*!
  \brief Unit tests for Plato::ParseTools::getBasis

  The cartesian basis vectors are stored as the columns of the returned
  basis, B, so:

  v = B v'

  where v' are the vector coefficients in the basis, B, and v are the vector
  coefficients in the global basis.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MaterialBasis_Parse_3D)
{
  Teuchos::RCP<Teuchos::ParameterList> tParams =
  Teuchos::getParametersFromXmlString(
      "<ParameterList name='Model'>                                           \n"
      "  <ParameterList name='Basis'>                                         \n"
      "    <Parameter name='X' type='Array(double)' value='{0.0,-1.0, 0.0}'/> \n"
      "    <Parameter name='Y' type='Array(double)' value='{1.0, 0.0, 0.0}'/> \n"
      "    <Parameter name='Z' type='Array(double)' value='{0.0, 0.0, 1.0}'/> \n"
      "  </ParameterList>                                                     \n"
      "</ParameterList>                                                       \n"
  );

  Plato::Matrix<3,3> tBasis;
  Plato::ParseTools::getBasis(*tParams, tBasis);

  TEST_ASSERT(tBasis(0,0) == 0.0);
  TEST_ASSERT(tBasis(0,1) == 1.0);
  TEST_ASSERT(tBasis(0,2) == 0.0);

  TEST_ASSERT(tBasis(1,0) ==-1.0);
  TEST_ASSERT(tBasis(1,1) == 0.0);
  TEST_ASSERT(tBasis(1,2) == 0.0);

  TEST_ASSERT(tBasis(2,0) == 0.0);
  TEST_ASSERT(tBasis(2,1) == 0.0);
  TEST_ASSERT(tBasis(2,2) == 1.0);

  // Only the first two vectors are required 
  //
  Teuchos::RCP<Teuchos::ParameterList> tParams_noZ =
  Teuchos::getParametersFromXmlString(
      "<ParameterList name='Model'>                                           \n"
      "  <ParameterList name='Basis'>                                         \n"
      "    <Parameter name='X' type='Array(double)' value='{0.0,-1.0, 0.0}'/> \n"
      "    <Parameter name='Y' type='Array(double)' value='{1.0, 0.0, 0.0}'/> \n"
      "  </ParameterList>                                                     \n"
      "</ParameterList>                                                       \n"
  );

  Plato::Matrix<3,3> tBasis_noZ;
  Plato::ParseTools::getBasis(*tParams_noZ, tBasis_noZ);

  TEST_ASSERT(tBasis_noZ(0,0) == 0.0);
  TEST_ASSERT(tBasis_noZ(0,1) == 1.0);
  TEST_ASSERT(tBasis_noZ(0,2) == 0.0);

  TEST_ASSERT(tBasis_noZ(1,0) ==-1.0);
  TEST_ASSERT(tBasis_noZ(1,1) == 0.0);
  TEST_ASSERT(tBasis_noZ(1,2) == 0.0);

  TEST_ASSERT(tBasis_noZ(2,0) == 0.0);
  TEST_ASSERT(tBasis_noZ(2,1) == 0.0);
  TEST_ASSERT(tBasis_noZ(2,2) == 1.0);

  // Vectors that are not unit length are normalized and a warning is 
  // written to the console.
  //
  Teuchos::RCP<Teuchos::ParameterList> tParams_notUnit =
  Teuchos::getParametersFromXmlString(
      "<ParameterList name='Model'>                                           \n"
      "  <ParameterList name='Basis'>                                         \n"
      "    <Parameter name='X' type='Array(double)' value='{0.0,-2.0, 0.0}'/> \n"
      "    <Parameter name='Y' type='Array(double)' value='{2.0, 0.0, 0.0}'/> \n"
      "  </ParameterList>                                                     \n"
      "</ParameterList>                                                       \n"
  );

  Plato::Matrix<3,3> tBasis_notUnit;
  Plato::ParseTools::getBasis(*tParams_notUnit, tBasis_notUnit);

  TEST_ASSERT(tBasis_notUnit(0,0) == 0.0);
  TEST_ASSERT(tBasis_notUnit(0,1) == 1.0);
  TEST_ASSERT(tBasis_notUnit(0,2) == 0.0);

  TEST_ASSERT(tBasis_notUnit(1,0) ==-1.0);
  TEST_ASSERT(tBasis_notUnit(1,1) == 0.0);
  TEST_ASSERT(tBasis_notUnit(1,2) == 0.0);

  TEST_ASSERT(tBasis_notUnit(2,0) == 0.0);
  TEST_ASSERT(tBasis_notUnit(2,1) == 0.0);
  TEST_ASSERT(tBasis_notUnit(2,2) == 1.0);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MaterialBasis_Parse_2D)
{
  Teuchos::RCP<Teuchos::ParameterList> tParams =
  Teuchos::getParametersFromXmlString(
      "<ParameterList name='Model'>                                      \n"
      "  <ParameterList name='Basis'>                                    \n"
      "    <Parameter name='X' type='Array(double)' value='{0.0,-1.0}'/> \n"
      "    <Parameter name='Y' type='Array(double)' value='{1.0, 0.0}'/> \n"
      "  </ParameterList>                                                \n"
      "</ParameterList>                                                  \n"
  );

  Plato::Matrix<2,2> tBasis;
  Plato::ParseTools::getBasis(*tParams, tBasis);

  TEST_ASSERT(tBasis(0,0) == 0.0);
  TEST_ASSERT(tBasis(0,1) == 1.0);

  TEST_ASSERT(tBasis(1,0) ==-1.0);
  TEST_ASSERT(tBasis(1,1) == 0.0);

  // Only the first vector is required 
  //
  Teuchos::RCP<Teuchos::ParameterList> tParams_noY =
  Teuchos::getParametersFromXmlString(
      "<ParameterList name='Model'>                                      \n"
      "  <ParameterList name='Basis'>                                    \n"
      "    <Parameter name='X' type='Array(double)' value='{0.0,-1.0}'/> \n"
      "  </ParameterList>                                                \n"
      "</ParameterList>                                                  \n"
  );

  Plato::Matrix<2,2> tBasis_noY;
  Plato::ParseTools::getBasis(*tParams_noY, tBasis_noY);

  TEST_ASSERT(tBasis_noY(0,0) == 0.0);
  TEST_ASSERT(tBasis_noY(0,1) == 1.0);

  TEST_ASSERT(tBasis_noY(1,0) ==-1.0);
  TEST_ASSERT(tBasis_noY(1,1) == 0.0);

  // Vectors that are not unit length are normalized and a warning is 
  // written to the console.
  //
  Teuchos::RCP<Teuchos::ParameterList> tParams_notUnit =
  Teuchos::getParametersFromXmlString(
      "<ParameterList name='Model'>                                      \n"
      "  <ParameterList name='Basis'>                                    \n"
      "    <Parameter name='X' type='Array(double)' value='{0.0,-2.0}'/> \n"
      "  </ParameterList>                                                \n"
      "</ParameterList>                                                  \n"
  );

  Plato::Matrix<2,2> tBasis_notUnit;
  Plato::ParseTools::getBasis(*tParams_notUnit, tBasis_notUnit);

  TEST_ASSERT(tBasis_notUnit(0,0) == 0.0);
  TEST_ASSERT(tBasis_notUnit(0,1) == 1.0);

  TEST_ASSERT(tBasis_notUnit(1,0) ==-1.0);
  TEST_ASSERT(tBasis_notUnit(1,1) == 0.0);
}

/******************************************************************************/
/*!
  \brief Unit tests for Plato::UniformMaterialBasis::VectorFromMaterialBasis
  and Plato::UniformMaterialBasis::VectorToMaterialBasis

  The cartesian basis vectors are stored as the columns of the returned
  basis, B, so:

  v = B v'

  where v' are the vector coefficients in the basis, B, and v are the vector
  coefficients in the global basis.  VectorFromMaterialBasis takes v' and
  returns v.

      [ 0 1 0 ]
  B = |-1 0 0 |
      [ 0 0 1 ]

  v' = {1 0 0}

  v = {0 -1 0}
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MaterialBasis_TransformVector_3D)
{
  Teuchos::RCP<Teuchos::ParameterList> tParams =
  Teuchos::getParametersFromXmlString(
      "<ParameterList name='Model'>                                           \n"
      "  <ParameterList name='Basis'>                                         \n"
      "    <Parameter name='X' type='Array(double)' value='{0.0,-1.0, 0.0}'/> \n"
      "    <Parameter name='Y' type='Array(double)' value='{1.0, 0.0, 0.0}'/> \n"
      "    <Parameter name='Z' type='Array(double)' value='{0.0, 0.0, 1.0}'/> \n"
      "  </ParameterList>                                                     \n"
      "</ParameterList>                                                       \n"
  );

  Plato::Matrix<3,3> tBasis;
  Plato::ParseTools::getBasis(*tParams, tBasis);

  Plato::UniformMaterialBasis<3> tMaterialBasis(tBasis);

  int tNumCells = 5;
  int tNumPoints = 2;
  int tNumDims = 3;
  Plato::ScalarArray3D tVectors("vectors", tNumCells, tNumPoints, tNumDims);

  Kokkos::parallel_for("to material basis", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    tVectors(iCellOrdinal, iGpOrdinal, 0) = 1.0;
    tVectors(iCellOrdinal, iGpOrdinal, 1) = 0.0;
    tVectors(iCellOrdinal, iGpOrdinal, 2) = 0.0;
  });

  tMaterialBasis.VectorFromMaterialBasis(tVectors);

  auto tVectors_Host = Kokkos::create_mirror_view(tVectors);
  Kokkos::deep_copy(tVectors_Host, tVectors);

  for(int iCell=0; iCell<tNumCells; iCell++){
    for(int iPoint=0; iPoint<tNumPoints; iPoint++){
      TEST_ASSERT(tVectors_Host(iCell, iPoint, 0) == 0.0);
      TEST_ASSERT(tVectors_Host(iCell, iPoint, 1) ==-1.0);
      TEST_ASSERT(tVectors_Host(iCell, iPoint, 2) == 0.0);
    }
  }

  tMaterialBasis.VectorToMaterialBasis(tVectors);

  Kokkos::deep_copy(tVectors_Host, tVectors);

  for(int iCell=0; iCell<tNumCells; iCell++){
    for(int iPoint=0; iPoint<tNumPoints; iPoint++){
      TEST_ASSERT(tVectors_Host(iCell, iPoint, 0) == 1.0);
      TEST_ASSERT(tVectors_Host(iCell, iPoint, 1) == 0.0);
      TEST_ASSERT(tVectors_Host(iCell, iPoint, 2) == 0.0);
    }
  }
}

/******************************************************************************/
/*!
  \brief Unit tests for Plato::UniformMaterialBasis::VoigtTensorFromMaterialBasis
         and Plato::UniformMaterialBasis::VoigtTensorToMaterialBasis

  The cartesian basis vectors are stored as the columns of the returned
  basis, B, so:

      [ 0 1 0 ]
  B = |-1 0 0 |
      [ 0 0 1 ]

  v' = {1 0 0}

  v = {0 -1 0}

  v = B v'

  where v' are the vector coefficients in the basis, B, and v are the vector
  coefficients in the global basis. If S is a tensor that transforms v into
  w in the global coordinates, we want to find S' that transforms v' into w':

  w = S v
  w = B w'
  
  B w' = S B v'
  w' = B^T S B v'
  S' = B^T S B

  TensorFromMaterialBasis takes S' and returns S.  TensorToMaterialBasis takes
  S and returns S'.  Note that S and S' are in Voigt form, {xx, yy, zz, yz, xz, xy}.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MaterialBasis_TransformTensorFromBasis_3D)
{
  Teuchos::RCP<Teuchos::ParameterList> tParams =
  Teuchos::getParametersFromXmlString(
      "<ParameterList name='Model'>                                           \n"
      "  <ParameterList name='Basis'>                                         \n"
      "    <Parameter name='X' type='Array(double)' value='{0.0,-1.0, 0.0}'/> \n"
      "    <Parameter name='Y' type='Array(double)' value='{1.0, 0.0, 0.0}'/> \n"
      "    <Parameter name='Z' type='Array(double)' value='{0.0, 0.0, 1.0}'/> \n"
      "  </ParameterList>                                                     \n"
      "</ParameterList>                                                       \n"
  );

  Plato::Matrix<3,3> tBasis;
  Plato::ParseTools::getBasis(*tParams, tBasis);

  Plato::UniformMaterialBasis<3> tMaterialBasis(tBasis);

  int tNumCells = 5;
  int tNumPoints = 2;
  int tNumVoigt = 6;
  Plato::ScalarArray3D tTensors("Tensors", tNumCells, tNumPoints, tNumVoigt);

  Kokkos::parallel_for("to material basis", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    tTensors(iCellOrdinal, iGpOrdinal, 0) = 1.0;
    tTensors(iCellOrdinal, iGpOrdinal, 1) = 2.0;
    tTensors(iCellOrdinal, iGpOrdinal, 2) = 2.0;
    tTensors(iCellOrdinal, iGpOrdinal, 3) = 0.0;
    tTensors(iCellOrdinal, iGpOrdinal, 4) = 0.0;
    tTensors(iCellOrdinal, iGpOrdinal, 5) = 0.0;
  });

  tMaterialBasis.VoigtTensorFromMaterialBasis(tTensors);

  auto tTensors_Host = Kokkos::create_mirror_view(tTensors);
  Kokkos::deep_copy(tTensors_Host, tTensors);

  for(int iCell=0; iCell<tNumCells; iCell++){
    for(int iPoint=0; iPoint<tNumPoints; iPoint++){
      TEST_ASSERT(tTensors_Host(iCell, iPoint, 0) == 2.0);
      TEST_ASSERT(tTensors_Host(iCell, iPoint, 1) == 1.0);
      TEST_ASSERT(tTensors_Host(iCell, iPoint, 2) == 2.0);
      TEST_ASSERT(tTensors_Host(iCell, iPoint, 3) == 0.0);
      TEST_ASSERT(tTensors_Host(iCell, iPoint, 4) == 0.0);
      TEST_ASSERT(tTensors_Host(iCell, iPoint, 5) == 0.0);
    }
  }

  tMaterialBasis.VoigtTensorToMaterialBasis(tTensors);

  Kokkos::deep_copy(tTensors_Host, tTensors);

  for(int iCell=0; iCell<tNumCells; iCell++){
    for(int iPoint=0; iPoint<tNumPoints; iPoint++){
      TEST_ASSERT(tTensors_Host(iCell, iPoint, 0) == 1.0);
      TEST_ASSERT(tTensors_Host(iCell, iPoint, 1) == 2.0);
      TEST_ASSERT(tTensors_Host(iCell, iPoint, 2) == 2.0);
      TEST_ASSERT(tTensors_Host(iCell, iPoint, 3) == 0.0);
      TEST_ASSERT(tTensors_Host(iCell, iPoint, 4) == 0.0);
      TEST_ASSERT(tTensors_Host(iCell, iPoint, 5) == 0.0);
    }
  }
}

}
