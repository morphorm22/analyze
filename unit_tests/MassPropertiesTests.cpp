/*
 * PlatoAugLagStressTest.cpp
 *
 *  Created on: Feb 3, 2019
 */

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "util/PlatoTestHelpers.hpp"

#include "Tri3.hpp"
#include "Tet4.hpp"
#include "BLAS1.hpp"
#include "Geometrical.hpp"
#include "Analyze_Diagnostics.hpp"
#include "geometric/MassMoment.hpp"
#include "geometric/GeometricalElement.hpp"
#include "geometric/WeightedSumFunction.hpp"
#include "geometric/GeometryScalarFunction.hpp"
#include "geometric/MassPropertiesFunction.hpp"


namespace MassPropertiesTest
{

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassInsteadOfVolume2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);

    using ElementType = typename Plato::GeometricalElement<Plato::Tri3>;

    Teuchos::RCP<Teuchos::ParameterList> params =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                           \n"
      "  <ParameterList name='Spatial Model'>                                         \n"
      "    <ParameterList name='Domains'>                                             \n"
      "      <ParameterList name='Design Volume'>                                     \n"
      "        <Parameter name='Element Block' type='string' value='body'/>           \n"
      "        <Parameter name='Material Model' type='string' value='Beef Jerky'/>    \n"
      "      </ParameterList>                                                         \n"
      "    </ParameterList>                                                           \n"
      "  </ParameterList>                                                             \n"
      "  <ParameterList name='Material Models'>                                       \n"
      "    <ParameterList name='Beef Jerky'>                                          \n"
      "      <ParameterList name='Thermoelastic'>                                     \n"
      "        <ParameterList name='Elastic Stiffness'>                               \n"
      "          <Parameter  name='Poissons Ratio' type='double' value='0.3'/>        \n"
      "          <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>     \n"
      "        </ParameterList>                                                       \n"
      "        <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/>  \n"
      "        <Parameter  name='Thermal Conductivity' type='double' value='910.0'/>  \n"
      "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>   \n"
      "      </ParameterList>                                                         \n"
      "    </ParameterList>                                                           \n"
      "  </ParameterList>                                                             \n"
      "</ParameterList>                                                               \n"
    );

    using Residual = typename Plato::Geometric::Evaluation<ElementType>::Residual;
    using ConfigT  = typename Residual::ConfigScalarType;
    using ResultT  = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->NumElements();

    // Create control workset
    const Plato::Scalar tPseudoDensity = 0.8;
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(tPseudoDensity, tControl);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *params, tDataMap);
    Plato::Geometric::WeightedSumFunction<Plato::Geometrical<Plato::Tri3>> tWeightedSum(tSpatialModel, tDataMap);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    const Plato::Scalar tMaterialDensity = 0.5;
    const auto tCriterion = std::make_shared<Plato::Geometric::MassMoment<Residual>>(tOnlyDomain, tDataMap);
    tCriterion->setMaterialDensity(tMaterialDensity);
    tCriterion->setCalculationType("Mass");

    const auto tGeometryScalarFunc =
          std::make_shared<Plato::Geometric::GeometryScalarFunction<Plato::Geometrical<Plato::Tri3>>>(tSpatialModel, tDataMap);

    tGeometryScalarFunc->setEvaluator(tCriterion, tOnlyDomain.getDomainName());

    const Plato::Scalar tFunctionWeight = 0.75;
    tWeightedSum.allocateScalarFunctionBase(tGeometryScalarFunc);
    tWeightedSum.appendFunctionWeight(tFunctionWeight);

    auto tObjFuncVal = tWeightedSum.value(tControl);

    Plato::Scalar tGoldValue = pow(static_cast<Plato::Scalar>(tMeshWidth), tSpaceDim)
                               * tPseudoDensity * tFunctionWeight * tMaterialDensity;

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(tGoldValue, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassInsteadOfVolume3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    using ElementType = typename Plato::GeometricalElement<Plato::Tet4>;

    Teuchos::RCP<Teuchos::ParameterList> params =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                           \n"
      "  <ParameterList name='Spatial Model'>                                         \n"
      "    <ParameterList name='Domains'>                                             \n"
      "      <ParameterList name='Design Volume'>                                     \n"
      "        <Parameter name='Element Block' type='string' value='body'/>           \n"
      "        <Parameter name='Material Model' type='string' value='Snapple'/>       \n"
      "      </ParameterList>                                                         \n"
      "    </ParameterList>                                                           \n"
      "  </ParameterList>                                                             \n"
      "  <ParameterList name='Material Models'>                                       \n"
      "    <ParameterList name='Snapple'>                                             \n"
      "      <ParameterList name='Thermoelastic'>                                     \n"
      "        <ParameterList name='Elastic Stiffness'>                               \n"
      "          <Parameter  name='Poissons Ratio' type='double' value='0.3'/>        \n"
      "          <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>     \n"
      "        </ParameterList>                                                       \n"
      "        <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/>  \n"
      "        <Parameter  name='Thermal Conductivity' type='double' value='910.0'/>  \n"
      "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>   \n"
      "      </ParameterList>                                                         \n"
      "    </ParameterList>                                                           \n"
      "  </ParameterList>                                                             \n"
      "</ParameterList>                                                               \n"
    );

    using Residual = typename Plato::Geometric::Evaluation<ElementType>::Residual;
    using ConfigT  = typename Residual::ConfigScalarType;
    using ResultT  = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->NumElements();

    // Create control workset
    const Plato::Scalar tPseudoDensity = 0.8;
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(tPseudoDensity, tControl);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *params, tDataMap);
    Plato::Geometric::WeightedSumFunction<Plato::Geometrical<Plato::Tet4>> tWeightedSum(tSpatialModel, tDataMap);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    const Plato::Scalar tMaterialDensity = 0.5;
    const auto tCriterion = std::make_shared<Plato::Geometric::MassMoment<Residual>>(tOnlyDomain, tDataMap);
    tCriterion->setMaterialDensity(tMaterialDensity);
    tCriterion->setCalculationType("Mass");

    const auto tGeometryScalarFunc =
          std::make_shared<Plato::Geometric::GeometryScalarFunction<Plato::Geometrical<Plato::Tet4>>>(tSpatialModel, tDataMap);

    tGeometryScalarFunc->setEvaluator(tCriterion, tOnlyDomain.getDomainName());

    const Plato::Scalar tFunctionWeight = 0.75;
    tWeightedSum.allocateScalarFunctionBase(tGeometryScalarFunc);
    tWeightedSum.appendFunctionWeight(tFunctionWeight);

    auto tObjFuncVal = tWeightedSum.value(tControl);

    Plato::Scalar tGoldValue = pow(static_cast<Plato::Scalar>(tMeshWidth), tSpaceDim)
                               * tPseudoDensity * tFunctionWeight * tMaterialDensity;

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(tGoldValue, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPropertiesValue3D)
{
    constexpr Plato::OrdinalType tMeshWidth = 15; // Need high mesh density in order to get correct inertias
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    using ElementType = typename Plato::GeometricalElement<Plato::Tet4>;

    //const Plato::OrdinalType tNumCells = tMesh->NumElements();

    // Create control workset
    const Plato::Scalar tPseudoDensity = 0.8;
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(tPseudoDensity, tControl);

    Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(

    "<ParameterList name='Plato Problem'>                                           \n"
    "  <ParameterList name='Criteria'>                                                        \n"
    "    <ParameterList name='Mass Properties'>                                               \n"
    "        <Parameter name='Type' type='string' value='Mass Properties'/>                   \n"
    "        <Parameter name='Properties' type='Array(string)' value='{Mass,CGx,CGy,CGz,Ixx,Iyy,Izz,Ixy,Iyz}'/>     \n"
    "        <Parameter name='Weights' type='Array(double)' value='{2.0,0.1,2.0,3.0,4.0,5.0,6.0,7.0,8.0}'/>         \n"
    "        <Parameter name='Gold Values' type='Array(double)' value='{0.2,0.05,0.55,0.75,0.5,0.5,0.5,0.3,0.3}'/>  \n"
    "    </ParameterList>                                                                     \n"
    "  </ParameterList>                                                                       \n"
    "  <ParameterList name='Material Models'>                                       \n"
    "    <ParameterList name='Goop'>                                                \n"
    "      <Parameter  name='Density' type='double' value='0.5'/>                   \n"
    "    </ParameterList>                                                           \n"
    "  </ParameterList>                                                             \n"
    "  <ParameterList name='Spatial Model'>                                         \n"
    "    <ParameterList name='Domains'>                                             \n"
    "      <ParameterList name='Design Volume'>                                     \n"
    "        <Parameter name='Element Block' type='string' value='body'/>           \n"
    "        <Parameter name='Material Model' type='string' value='Goop'/>          \n"
    "      </ParameterList>                                                         \n"
    "    </ParameterList>                                                           \n"
    "  </ParameterList>                                                             \n"
    "</ParameterList>                                                               \n"
    );

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tParams, tDataMap);
    std::string tFuncName = "Mass Properties";
    Plato::Geometric::MassPropertiesFunction<Plato::Geometrical<Plato::Tet4>>
          tMassProperties(tSpatialModel, tDataMap, *tParams, tFuncName);

    auto tObjFuncVal = tMassProperties.value(tControl);

    Plato::Scalar tGoldValue = 2.0*pow((0.4-0.2)/0.2, 2) + 0.1*pow((0.5-0.05),2)
                             + 2.0*pow((0.5-0.55),2) + 3.0*pow((0.5-0.75),2)
                             + 4.0*pow((0.2666666-0.5)/0.5,2)
                             + 5.0*pow((0.2666666-0.5)/0.5,2)
                             + 6.0*pow((0.2666666-0.5)/0.5,2)
                             + 7.0*pow((-0.1-0.3)/0.3,2)
                             + 8.0*pow((-0.1-0.3)/0.3,2);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(tGoldValue, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPropertiesValue3DNormalized)
{
    constexpr Plato::OrdinalType tMeshWidth = 15; // Need high mesh density in order to get correct inertias
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    // Create control workset
    const Plato::Scalar tPseudoDensity = 0.8;
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(tPseudoDensity, tControl);

    Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                           \n"
    "  <Parameter name='Objective' type='string' value='My Mass Properties'/>       \n"
    "  <ParameterList name='Criteria'>                                                        \n"
    "    <ParameterList name='Mass Properties'>                                               \n"
    "        <Parameter name='Type' type='string' value='Mass Properties'/>                   \n"
    "        <Parameter name='Properties' type='Array(string)' value='{Mass,CGx,CGy,CGz,Ixx,Iyy,Izz,Ixy,Ixz,Iyz}'/>     \n"
    "        <Parameter name='Weights' type='Array(double)' value='{2.0,0.1,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0}'/>         \n"
    "        <Parameter name='Gold Values' type='Array(double)' value='{0.2,0.05,0.55,0.75,5.4,5.5,5.4,-0.1,-0.1,-0.15}'/>  \n"
    "    </ParameterList>                                                                     \n"
    "  </ParameterList>                                                                       \n"
    "  <ParameterList name='Material Models'>                                       \n"
    "    <ParameterList name='Goop'>                                                \n"
    "      <Parameter  name='Density' type='double' value='0.5'/>                   \n"
    "    </ParameterList>                                                           \n"
    "  </ParameterList>                                                             \n"
    "  <ParameterList name='Spatial Model'>                                         \n"
    "    <ParameterList name='Domains'>                                             \n"
    "      <ParameterList name='Design Volume'>                                     \n"
    "        <Parameter name='Element Block' type='string' value='body'/>           \n"
    "        <Parameter name='Material Model' type='string' value='Goop'/>          \n"
    "      </ParameterList>                                                         \n"
    "    </ParameterList>                                                           \n"
    "  </ParameterList>                                                             \n"
    "</ParameterList>                                                               \n"
  );

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tParams, tDataMap);
    std::string tFuncName = "Mass Properties";
    Plato::Geometric::MassPropertiesFunction<Plato::Geometrical<Plato::Tet4>>
          tMassProperties(tSpatialModel, tDataMap, *tParams, tFuncName);

    auto tObjFuncVal = tMassProperties.value(tControl);

    Plato::Scalar tGoldValue = 2.0*pow((0.4-0.2)/0.2, 2) + 0.1*pow((0.5-0.05),2)
                             + 2.0*pow((0.5-0.55),2) + 3.0*pow((0.5-0.75),2)
                             + 4.0*pow((-1.0589e-01-5.1241) /  5.1241,2)
                             + 5.0*pow((2.6130e-02-5.4403)  /  5.4403,2)
                             + 6.0*pow((1.8531e-01-5.3886)  /  5.3886,2)
                             + 7.0*pow((1.9408e-04-0.0000)  /  5.1241,2)
                             + 8.0*pow((9.5366e-02-0.0000)  /  5.1241,2)
                             + 9.0*pow((3.9663e-02-0.0000)  /  5.1241,2);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(tGoldValue, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPropertiesGradZ_3D)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    using ElementType = typename Plato::GeometricalElement<Plato::Tet4>;

    using GradientZ = typename Plato::Geometric::Evaluation<ElementType>::GradientZ;

    Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <ParameterList name='Criteria'>                                         \n"
    "    <ParameterList name='Mass Properties'>                                \n"
    "        <Parameter name='Type' type='string' value='Mass Properties'/>    \n"
    "        <Parameter name='Properties' type='Array(string)' value='{Mass,CGx,CGy,CGz,Ixx,Iyy,Izz,Ixy,Iyz}'/>  \n"
    "        <Parameter name='Weights' type='Array(double)' value='{2.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0}'/>      \n"
    "        <Parameter name='Gold Values' type='Array(double)' value='{0.2,0.45,0.55,0.75,0.5,0.5,0.5,0.3,0.3}'/>  \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Goop'>                                           \n"
    "      <Parameter  name='Density' type='double' value='0.5'/>              \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Goop'/>     \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                          \n"
  );

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tParams, tDataMap);
    std::string tFuncName = "Mass Properties";
    Plato::Geometric::MassPropertiesFunction<Plato::Geometrical<Plato::Tet4>>
          tMassProperties(tSpatialModel, tDataMap, *tParams, tFuncName);

    Plato::test_partial_control<GradientZ, ElementType>(tMesh, tMassProperties);
}

} // namespace MassPropertiesTest
