/*
 * MicromorphicLinearMaterialModelTests.cpp
 *
 *  Created on: Oct 18, 2021
 */

#include "PlatoTestHelpers.hpp"
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "ImplicitFunctors.hpp"
#include "WorksetBase.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "InterpolateFromNodal.hpp"
#include "ProjectToNode.hpp"

#include "PlatoProblemFactory.hpp"
#include "PlatoAbstractProblem.hpp"
#include "hyperbolic/InertialContent.hpp"

#include "hyperbolic/micromorphic/MicromorphicElasticModelFactory.hpp"
#include "hyperbolic/micromorphic/MicromorphicInertiaModelFactory.hpp"
#include "hyperbolic/micromorphic/MicromorphicKinematics.hpp"
#include "hyperbolic/micromorphic/MicromorphicKinetics.hpp"
#include "hyperbolic/micromorphic/FullStressDivergence.hpp"
#include "hyperbolic/micromorphic/ProjectStressToNode.hpp"

namespace RelaxedMicromorphicTest
{

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, ElasticCubic1D)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                    \n"
      "  <ParameterList name='Material Models'>                           \n"
      "    <ParameterList name='material_1'>                           \n"
      "      <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "        <Parameter  name='Lambda_e' type='double' value='-120.74'/>   \n"
      "        <Parameter  name='Mu_e' type='double' value='557.11'/>   \n"
      "        <Parameter  name='Mu_c' type='double' value='1.8e-4'/>   \n"
      "        <Parameter  name='Lambda_m' type='double' value='180.63'/>   \n"
      "        <Parameter  name='Mu_m' type='double' value='255.71'/>   \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 1;
    Plato::MicromorphicElasticModelFactory<tSpaceDim> tMaterialModelFactory(*tParams);
    auto tMaterialModel = tMaterialModelFactory.create("material_1");

    auto tStiffnessMatrixCe = tMaterialModel->getStiffnessMatrixCe();
    const Plato::Scalar tTolerance = 1e-12;
    TEST_FLOATING_EQUALITY(993.48, tStiffnessMatrixCe(0,0), tTolerance);

    auto tStiffnessMatrixCc = tMaterialModel->getStiffnessMatrixCc();
    TEST_FLOATING_EQUALITY(1.8e-4, tStiffnessMatrixCc(0,0), tTolerance);

    auto tStiffnessMatrixCm = tMaterialModel->getStiffnessMatrixCm();
    TEST_FLOATING_EQUALITY(692.05, tStiffnessMatrixCm(0,0), tTolerance);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, ElasticCubic1D_Lambda_e_Keyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "      <Parameter  name='Lamda_e' type='double' value='-120.74'/>   \n"
      "      <Parameter  name='Mu_e' type='double' value='557.11'/>   \n"
      "      <Parameter  name='Mu_c' type='double' value='1.8e-4'/>   \n"
      "      <Parameter  name='Lambda_m' type='double' value='180.63'/>   \n"
      "      <Parameter  name='Mu_m' type='double' value='255.71'/>   \n"
      "    </ParameterList>                                                  \n"
    );
    constexpr Plato::OrdinalType tSpaceDim = 1;
    TEST_THROW(Plato::CubicMicromorphicLinearElasticMaterial<tSpaceDim> tMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, ElasticCubic2D)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                    \n"
      "  <ParameterList name='Material Models'>                           \n"
      "    <ParameterList name='material_1'>                           \n"
      "      <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "        <Parameter  name='Lambda_e' type='double' value='-120.74'/>   \n"
      "        <Parameter  name='Mu_e' type='double' value='557.11'/>   \n"
      "        <Parameter  name='Mu_star_e' type='double' value='8.37'/>   \n"
      "        <Parameter  name='Mu_c' type='double' value='1.8e-4'/>   \n"
      "        <Parameter  name='Lambda_m' type='double' value='180.63'/>   \n"
      "        <Parameter  name='Mu_m' type='double' value='255.71'/>   \n"
      "        <Parameter  name='Mu_star_m' type='double' value='181.28'/>   \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::MicromorphicElasticModelFactory<tSpaceDim> tMaterialModelFactory(*tParams);
    auto tMaterialModel = tMaterialModelFactory.create("material_1");

    auto tStiffnessMatrixCe = tMaterialModel->getStiffnessMatrixCe();
    const Plato::Scalar tTolerance = 1e-12;
    TEST_FLOATING_EQUALITY(993.48,  tStiffnessMatrixCe(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(-120.74, tStiffnessMatrixCe(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(-120.74, tStiffnessMatrixCe(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(993.48,  tStiffnessMatrixCe(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(8.37,    tStiffnessMatrixCe(2,2), tTolerance);

    auto tStiffnessMatrixCc = tMaterialModel->getStiffnessMatrixCc();
    TEST_FLOATING_EQUALITY(1.8e-4, tStiffnessMatrixCc(0,0), tTolerance);

    auto tStiffnessMatrixCm = tMaterialModel->getStiffnessMatrixCm();
    TEST_FLOATING_EQUALITY(692.05, tStiffnessMatrixCm(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(180.63, tStiffnessMatrixCm(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCm(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(180.63, tStiffnessMatrixCm(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(692.05, tStiffnessMatrixCm(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCm(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCm(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCm(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(181.28, tStiffnessMatrixCm(2,2), tTolerance);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, ElasticCubic2D_Mu_star_m_Keyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "      <Parameter  name='Lambda_e' type='double' value='-120.74'/>   \n"
      "      <Parameter  name='Mu_e' type='double' value='557.11'/>   \n"
      "      <Parameter  name='Mu_star_e' type='double' value='8.37'/>   \n"
      "      <Parameter  name='Mu_c' type='double' value='1.8e-4'/>   \n"
      "      <Parameter  name='Lambda_m' type='double' value='180.63'/>   \n"
      "      <Parameter  name='Mu_m' type='double' value='255.71'/>   \n"
      "      <Parameter  name='Mu_m_star' type='double' value='181.28'/>   \n"
      "    </ParameterList>                                                  \n"
    );
    constexpr Plato::OrdinalType tSpaceDim = 2;
    TEST_THROW(Plato::CubicMicromorphicLinearElasticMaterial<tSpaceDim> tMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, ElasticCubic3D)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                    \n"
      "  <ParameterList name='Material Models'>                           \n"
      "    <ParameterList name='material_1'>                           \n"
      "      <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "        <Parameter  name='Lambda_e' type='double' value='-120.74'/>   \n"
      "        <Parameter  name='Mu_e' type='double' value='557.11'/>   \n"
      "        <Parameter  name='Mu_star_e' type='double' value='8.37'/>   \n"
      "        <Parameter  name='Mu_c' type='double' value='1.8e-4'/>   \n"
      "        <Parameter  name='Lambda_m' type='double' value='180.63'/>   \n"
      "        <Parameter  name='Mu_m' type='double' value='255.71'/>   \n"
      "        <Parameter  name='Mu_star_m' type='double' value='181.28'/>   \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::MicromorphicElasticModelFactory<tSpaceDim> tMaterialModelFactory(*tParams);
    auto tMaterialModel = tMaterialModelFactory.create("material_1");

    auto tStiffnessMatrixCe = tMaterialModel->getStiffnessMatrixCe();
    const Plato::Scalar tTolerance = 1e-12;
    TEST_FLOATING_EQUALITY(993.48,  tStiffnessMatrixCe(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(-120.74, tStiffnessMatrixCe(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(-120.74, tStiffnessMatrixCe(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(0,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(0,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(0,5), tTolerance);
    TEST_FLOATING_EQUALITY(-120.74, tStiffnessMatrixCe(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(993.48,  tStiffnessMatrixCe(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(-120.74, tStiffnessMatrixCe(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(1,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(1,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(1,5), tTolerance);
    TEST_FLOATING_EQUALITY(-120.74, tStiffnessMatrixCe(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(-120.74, tStiffnessMatrixCe(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(993.48,  tStiffnessMatrixCe(2,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(2,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(2,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(2,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(3,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(3,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(3,2), tTolerance);
    TEST_FLOATING_EQUALITY(8.37,    tStiffnessMatrixCe(3,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(3,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(3,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(4,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(4,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(4,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(4,3), tTolerance);
    TEST_FLOATING_EQUALITY(8.37,    tStiffnessMatrixCe(4,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(4,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(5,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(5,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(5,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(5,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(5,4), tTolerance);
    TEST_FLOATING_EQUALITY(8.37,    tStiffnessMatrixCe(5,5), tTolerance);

    auto tStiffnessMatrixCc = tMaterialModel->getStiffnessMatrixCc();
    TEST_FLOATING_EQUALITY(1.8e-4, tStiffnessMatrixCc(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCc(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCc(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCc(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(1.8e-4, tStiffnessMatrixCc(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCc(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCc(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCc(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(1.8e-4, tStiffnessMatrixCc(2,2), tTolerance);

    auto tStiffnessMatrixCm = tMaterialModel->getStiffnessMatrixCm();
    TEST_FLOATING_EQUALITY(692.05,  tStiffnessMatrixCm(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(180.63,  tStiffnessMatrixCm(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(180.63,  tStiffnessMatrixCm(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(0,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(0,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(0,5), tTolerance);
    TEST_FLOATING_EQUALITY(180.63,  tStiffnessMatrixCm(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(692.05,  tStiffnessMatrixCm(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(180.63,  tStiffnessMatrixCm(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(1,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(1,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(1,5), tTolerance);
    TEST_FLOATING_EQUALITY(180.63,  tStiffnessMatrixCm(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(180.63,  tStiffnessMatrixCm(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(692.05,  tStiffnessMatrixCm(2,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(2,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(2,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(2,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(3,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(3,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(3,2), tTolerance);
    TEST_FLOATING_EQUALITY(181.28,  tStiffnessMatrixCm(3,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(3,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(3,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(4,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(4,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(4,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(4,3), tTolerance);
    TEST_FLOATING_EQUALITY(181.28,  tStiffnessMatrixCm(4,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(4,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(5,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(5,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(5,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(5,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(5,4), tTolerance);
    TEST_FLOATING_EQUALITY(181.28,  tStiffnessMatrixCm(5,5), tTolerance);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, ElasticCubic3D_Mu_c_Keyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "      <Parameter  name='Lambda_e' type='double' value='-120.74'/>   \n"
      "      <Parameter  name='Mu_e' type='double' value='557.11'/>   \n"
      "      <Parameter  name='Mu_star_e' type='double' value='8.37'/>   \n"
      "      <Parameter  name='Moo_c' type='double' value='1.8e-4'/>   \n"
      "      <Parameter  name='Lambda_m' type='double' value='180.63'/>   \n"
      "      <Parameter  name='Mu_m' type='double' value='255.71'/>   \n"
      "      <Parameter  name='Mu_m_star' type='double' value='181.28'/>   \n"
      "    </ParameterList>                                                  \n"
    );
    constexpr Plato::OrdinalType tSpaceDim = 3;
    TEST_THROW(Plato::CubicMicromorphicLinearElasticMaterial<tSpaceDim> tMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, InertiaCubic1D)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                    \n"
      "  <ParameterList name='Material Models'>                           \n"
      "    <ParameterList name='material_1'>                           \n"
      "      <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "        <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "        <Parameter  name='Eta_bar_1' type='double' value='0.6'/>   \n"
      "        <Parameter  name='Eta_bar_3' type='double' value='2.0'/>   \n"
      "        <Parameter  name='Eta_bar_2' type='double' value='1.0e-4'/>   \n"
      "        <Parameter  name='Eta_1' type='double' value='2300.0'/>   \n"
      "        <Parameter  name='Eta_3' type='double' value='-1800.0'/>   \n"
      "        <Parameter  name='Eta_2' type='double' value='7.0e-4'/>   \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 1;
    Plato::MicromorphicInertiaModelFactory<tSpaceDim> tInertiaModelFactory(*tParams);
    auto tInertiaModel = tInertiaModelFactory.create("material_1");

    auto tRho = tInertiaModel->getMacroscopicMassDensity();
    const Plato::Scalar tTolerance = 1e-12;
    TEST_FLOATING_EQUALITY(1451.8, tRho, tTolerance);

    auto tInertiaMatrixTe = tInertiaModel->getInertiaMatrixTe();
    TEST_FLOATING_EQUALITY(3.2, tInertiaMatrixTe(0,0), tTolerance);

    auto tInertiaMatrixTc = tInertiaModel->getInertiaMatrixTc();
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixTc(0,0), tTolerance);

    auto tInertiaMatrixJm = tInertiaModel->getInertiaMatrixJm();
    TEST_FLOATING_EQUALITY(2800.0, tInertiaMatrixJm(0,0), tTolerance);

    auto tInertiaMatrixJc = tInertiaModel->getInertiaMatrixJc();
    TEST_FLOATING_EQUALITY(7.0e-4, tInertiaMatrixJc(0,0), tTolerance);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, InertiaCubic1D_Eta_bar_2_Keyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "      <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "      <Parameter  name='Eta_bar_1' type='double' value='0.6'/>   \n"
      "      <Parameter  name='Eta_bar_3' type='double' value='2.0'/>   \n"
      "      <Parameter  name='Eta_bat_2' type='double' value='1.0e-4'/>   \n"
      "      <Parameter  name='Eta_1' type='double' value='2300.0'/>   \n"
      "      <Parameter  name='Eta_3' type='double' value='-1800.0'/>   \n"
      "      <Parameter  name='Eta_2' type='double' value='7.0e-4'/>   \n"
      "    </ParameterList>                                                  \n"
    );
    constexpr Plato::OrdinalType tSpaceDim = 1;
    TEST_THROW(Plato::CubicMicromorphicInertiaMaterial<tSpaceDim> tInertiaModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, InertiaCubic2D)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                    \n"
      "  <ParameterList name='Material Models'>                           \n"
      "    <ParameterList name='material_1'>                           \n"
      "      <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "        <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "        <Parameter  name='Eta_bar_1' type='double' value='0.6'/>   \n"
      "        <Parameter  name='Eta_bar_3' type='double' value='2.0'/>   \n"
      "        <Parameter  name='Eta_bar_2' type='double' value='1.0e-4'/>   \n"
      "        <Parameter  name='Eta_bar_star_1' type='double' value='0.2'/>   \n"
      "        <Parameter  name='Eta_1' type='double' value='2300.0'/>   \n"
      "        <Parameter  name='Eta_3' type='double' value='-1800.0'/>   \n"
      "        <Parameter  name='Eta_2' type='double' value='1.0e-4'/>   \n"
      "        <Parameter  name='Eta_star_1' type='double' value='4500.0'/>   \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::MicromorphicInertiaModelFactory<tSpaceDim> tInertiaModelFactory(*tParams);
    auto tInertiaModel = tInertiaModelFactory.create("material_1");

    auto tRho = tInertiaModel->getMacroscopicMassDensity();
    const Plato::Scalar tTolerance = 1e-12;
    TEST_FLOATING_EQUALITY(1451.8, tRho, tTolerance);

    auto tInertiaMatrixTe = tInertiaModel->getInertiaMatrixTe();
    TEST_FLOATING_EQUALITY(3.2, tInertiaMatrixTe(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(2.0, tInertiaMatrixTe(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(2.0, tInertiaMatrixTe(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(3.2, tInertiaMatrixTe(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.2, tInertiaMatrixTe(2,2), tTolerance);

    auto tInertiaMatrixTc = tInertiaModel->getInertiaMatrixTc();
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixTc(0,0), tTolerance);

    auto tInertiaMatrixJm = tInertiaModel->getInertiaMatrixJm();
    TEST_FLOATING_EQUALITY(2800.0,  tInertiaMatrixJm(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(-1800.0, tInertiaMatrixJm(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(-1800.0, tInertiaMatrixJm(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(2800.0,  tInertiaMatrixJm(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(4500.0,  tInertiaMatrixJm(2,2), tTolerance);

    auto tInertiaMatrixJc = tInertiaModel->getInertiaMatrixJc();
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixJc(0,0), tTolerance);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, InertiaCubic2D_Eta_bar_star_1_Keyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "      <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "      <Parameter  name='Eta_bar_1' type='double' value='0.6'/>   \n"
      "      <Parameter  name='Eta_bar_3' type='double' value='2.0'/>   \n"
      "      <Parameter  name='Eta_bar_2' type='double' value='1.0e-4'/>   \n"
      "      <Parameter  name='Eta_bar_1_star' type='double' value='0.2'/>   \n"
      "      <Parameter  name='Eta_1' type='double' value='2300.0'/>   \n"
      "      <Parameter  name='Eta_3' type='double' value='-1800.0'/>   \n"
      "      <Parameter  name='Eta_2' type='double' value='1.0e-4'/>   \n"
      "      <Parameter  name='Eta_star_1' type='double' value='4500.0'/>   \n"
      "    </ParameterList>                                                  \n"
    );
    constexpr Plato::OrdinalType tSpaceDim = 2;
    TEST_THROW(Plato::CubicMicromorphicInertiaMaterial<tSpaceDim> tInertiaModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, InertiaCubic3D)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                    \n"
      "  <ParameterList name='Material Models'>                           \n"
      "    <ParameterList name='material_1'>                           \n"
      "      <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "        <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "        <Parameter  name='Eta_bar_1' type='double' value='0.6'/>   \n"
      "        <Parameter  name='Eta_bar_3' type='double' value='2.0'/>   \n"
      "        <Parameter  name='Eta_bar_2' type='double' value='1.0e-4'/>   \n"
      "        <Parameter  name='Eta_bar_star_1' type='double' value='0.2'/>   \n"
      "        <Parameter  name='Eta_1' type='double' value='2300.0'/>   \n"
      "        <Parameter  name='Eta_3' type='double' value='-1800.0'/>   \n"
      "        <Parameter  name='Eta_2' type='double' value='1.0e-4'/>   \n"
      "        <Parameter  name='Eta_star_1' type='double' value='4500.0'/>   \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::MicromorphicInertiaModelFactory<tSpaceDim> tInertiaModelFactory(*tParams);
    auto tInertiaModel = tInertiaModelFactory.create("material_1");

    auto tRho = tInertiaModel->getMacroscopicMassDensity();
    const Plato::Scalar tTolerance = 1e-12;
    TEST_FLOATING_EQUALITY(1451.8, tRho, tTolerance);

    auto tInertiaMatrixTe = tInertiaModel->getInertiaMatrixTe();
    TEST_FLOATING_EQUALITY(3.2, tInertiaMatrixTe(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(2.0, tInertiaMatrixTe(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(2.0, tInertiaMatrixTe(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(0,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(0,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(0,5), tTolerance);
    TEST_FLOATING_EQUALITY(2.0, tInertiaMatrixTe(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(3.2, tInertiaMatrixTe(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(2.0, tInertiaMatrixTe(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(1,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(1,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(1,5), tTolerance);
    TEST_FLOATING_EQUALITY(2.0, tInertiaMatrixTe(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(2.0, tInertiaMatrixTe(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(3.2, tInertiaMatrixTe(2,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(2,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(2,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(2,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(3,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(3,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(3,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.2, tInertiaMatrixTe(3,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(3,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(3,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(4,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(4,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(4,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(4,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.2, tInertiaMatrixTe(4,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(4,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(5,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(5,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(5,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(5,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(5,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.2, tInertiaMatrixTe(5,5), tTolerance);

    auto tInertiaMatrixTc = tInertiaModel->getInertiaMatrixTc();
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixTc(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixTc(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixTc(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixTc(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixTc(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixTc(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixTc(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixTc(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixTc(2,2), tTolerance);

    auto tInertiaMatrixJm = tInertiaModel->getInertiaMatrixJm();
    TEST_FLOATING_EQUALITY(2800.0,  tInertiaMatrixJm(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(-1800.0, tInertiaMatrixJm(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(-1800.0, tInertiaMatrixJm(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(0,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(0,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(0,5), tTolerance);
    TEST_FLOATING_EQUALITY(-1800.0, tInertiaMatrixJm(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(2800.0,  tInertiaMatrixJm(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(-1800.0, tInertiaMatrixJm(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(1,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(1,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(1,5), tTolerance);
    TEST_FLOATING_EQUALITY(-1800.0, tInertiaMatrixJm(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(-1800.0, tInertiaMatrixJm(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(2800.0,  tInertiaMatrixJm(2,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(2,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(2,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(2,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(3,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(3,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(3,2), tTolerance);
    TEST_FLOATING_EQUALITY(4500.0,  tInertiaMatrixJm(3,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(3,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(3,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(4,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(4,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(4,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(4,3), tTolerance);
    TEST_FLOATING_EQUALITY(4500.0,  tInertiaMatrixJm(4,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(4,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(5,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(5,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(5,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(5,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(5,4), tTolerance);
    TEST_FLOATING_EQUALITY(4500.0,  tInertiaMatrixJm(5,5), tTolerance);

    auto tInertiaMatrixJc = tInertiaModel->getInertiaMatrixJc();
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixJc(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixJc(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixJc(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixJc(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixJc(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixJc(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixJc(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixJc(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixJc(2,2), tTolerance);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, InertiaCubic3D_Eta_3_Keyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "      <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "      <Parameter  name='Eta_bar_1' type='double' value='0.6'/>   \n"
      "      <Parameter  name='Eta_bar_3' type='double' value='2.0'/>   \n"
      "      <Parameter  name='Eta_bar_2' type='double' value='1.0e-4'/>   \n"
      "      <Parameter  name='Eta_bar_1_star' type='double' value='0.2'/>   \n"
      "      <Parameter  name='Eta_1' type='double' value='2300.0'/>   \n"
      "      <Parameter  name='Ets_3' type='double' value='-1800.0'/>   \n"
      "      <Parameter  name='Eta_2' type='double' value='1.0e-4'/>   \n"
      "      <Parameter  name='Eta_star_1' type='double' value='4500.0'/>   \n"
      "    </ParameterList>                                                  \n"
    );
    constexpr Plato::OrdinalType tSpaceDim = 3;
    TEST_THROW(Plato::CubicMicromorphicInertiaMaterial<tSpaceDim> tInertiaModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, Residual3D)
{
    // create test mesh
    //
    constexpr int tMeshWidth=2;
    constexpr int tSpaceDim=3;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);
    int tNumCells = tMesh->NumElements();

    // set material model
    //
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                    \n"
      "  <ParameterList name='Material Models'>                           \n"
      "    <ParameterList name='material_1'>                           \n"
      "      <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "        <Parameter  name='Lambda_e' type='double' value='-120.74'/>   \n"
      "        <Parameter  name='Mu_e' type='double' value='557.11'/>   \n"
      "        <Parameter  name='Mu_star_e' type='double' value='8.37'/>   \n"
      "        <Parameter  name='Mu_c' type='double' value='1.8e-4'/>   \n"
      "        <Parameter  name='Lambda_m' type='double' value='180.63'/>   \n"
      "        <Parameter  name='Mu_m' type='double' value='255.71'/>   \n"
      "        <Parameter  name='Mu_star_m' type='double' value='181.28'/>   \n"
      "      </ParameterList>                                                  \n"
      "      <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "        <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "        <Parameter  name='Eta_bar_1' type='double' value='0.6'/>   \n"
      "        <Parameter  name='Eta_bar_3' type='double' value='2.0'/>   \n"
      "        <Parameter  name='Eta_bar_2' type='double' value='1.0e-4'/>   \n"
      "        <Parameter  name='Eta_bar_star_1' type='double' value='0.2'/>   \n"
      "        <Parameter  name='Eta_1' type='double' value='2300.0'/>   \n"
      "        <Parameter  name='Eta_3' type='double' value='-1800.0'/>   \n"
      "        <Parameter  name='Eta_2' type='double' value='1.0e-4'/>   \n"
      "        <Parameter  name='Eta_star_1' type='double' value='4500.0'/>   \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );
    Plato::MicromorphicElasticModelFactory<tSpaceDim> tMaterialModelFactory(*tParams);
    auto tMaterialModel = tMaterialModelFactory.create("material_1");

    Plato::MicromorphicInertiaModelFactory<tSpaceDim> tInertiaModelFactory(*tParams);
    auto tInertiaModel = tInertiaModelFactory.create("material_1");

    // get problem data
    //
    constexpr int tNumVoigtTerms   = Plato::SimplexMicromorphicMechanics<tSpaceDim>::mNumVoigtTerms;
    constexpr int tNumSkwTerms   = Plato::SimplexMicromorphicMechanics<tSpaceDim>::mNumSkwTerms;
    constexpr int tNumDofsPerCell  = Plato::SimplexMicromorphicMechanics<tSpaceDim>::mNumDofsPerCell;
    constexpr int tNumDofsPerNode  = Plato::SimplexMicromorphicMechanics<tSpaceDim>::mNumDofsPerNode;
    constexpr int tNumNodesPerCell = Plato::SimplexMicromorphicMechanics<tSpaceDim>::mNumNodesPerCell;
    
    // create mesh based state
    //
    int tNumNodes = tMesh->NumNodes();
    int tNumDofs = tNumNodes*tNumDofsPerNode;
    Plato::ScalarVector tState("state", tNumDofs);
    Plato::ScalarVector tStateDotDot("state dot dot", tNumDofs);
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
      for (int tDofOrdinal=0; tDofOrdinal<tNumDofsPerNode; tDofOrdinal++)
      {
          tState(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-7)*(tDofOrdinal + 1)*aNodeOrdinal;
          tStateDotDot(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-5)*(tDofOrdinal + 1)*aNodeOrdinal;
      }
  }, "state");

    // initialize data storage
    //
    Plato::ScalarArray3DT<Plato::Scalar>     tGradient("gradient",tNumCells,tNumNodesPerCell,tSpaceDim);
    Plato::ScalarVectorT<Plato::Scalar>      tCellVolume("cell volume",tNumCells);

    Plato::ScalarMultiVectorT<Plato::Scalar> tSymDisplacementGradient ("strain", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<Plato::Scalar> tSkwDisplacementGradient ("strain", tNumCells, tNumSkwTerms);
    Plato::ScalarMultiVectorT<Plato::Scalar> tSymMicroDistortionTensor("strain", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<Plato::Scalar> tSkwMicroDistortionTensor("strain", tNumCells, tNumSkwTerms);

    Plato::ScalarMultiVectorT<Plato::Scalar> tSymGradientMicroInertia ("inertia strain", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<Plato::Scalar> tSkwGradientMicroInertia ("inertia strain", tNumCells, tNumSkwTerms);
    Plato::ScalarMultiVectorT<Plato::Scalar> tSymFreeMicroInertia     ("inertia strain", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<Plato::Scalar> tSkwFreeMicroInertia     ("inertia strain", tNumCells, tNumSkwTerms);

    Plato::ScalarMultiVectorT<Plato::Scalar> tSymCauchyStress("stress", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<Plato::Scalar> tSkwCauchyStress("stress", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<Plato::Scalar> tSymMicroStress ("stress", tNumCells, tNumVoigtTerms);

    Plato::ScalarMultiVectorT<Plato::Scalar> tSymGradientInertiaStress("inertia stress", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<Plato::Scalar> tSkwGradientInertiaStress("inertia stress", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<Plato::Scalar> tSymFreeInertiaStress    ("inertia stress", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<Plato::Scalar> tSkwFreeInertiaStress    ("inertia stress", tNumCells, tNumVoigtTerms);

    Plato::ScalarMultiVectorT<Plato::Scalar> tAccelerationGP("acceleration at Gauss point", tNumCells, tSpaceDim);
    Plato::ScalarMultiVectorT<Plato::Scalar> tInertialContentGP("density-scaled acceleration at Gauss point", tNumCells, tSpaceDim);

    Plato::ScalarMultiVectorT<Plato::Scalar> tResidual("residual", tNumCells, tNumDofsPerCell);
    Plato::ScalarMultiVectorT<Plato::Scalar> tInertiaResidual("inertia residual", tNumCells, tNumDofsPerCell);

    // cubature data
    //
    Plato::LinearTetCubRuleDegreeOne<tSpaceDim> tCubatureRule;
    auto tBasisFunctions = tCubatureRule.getBasisFunctions();
    Plato::Scalar tQuadratureWeight = tCubatureRule.getCubWeight();

    // workset operations
    //
    Plato::ScalarArray3DT<Plato::Scalar>     tConfigWS("config workset",tNumCells,tNumNodesPerCell,tSpaceDim);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset",tNumCells, tNumDofsPerCell);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateDotDotWS("state dot dot workset",tNumCells, tNumDofsPerCell);
    Plato::WorksetBase<Plato::SimplexMicromorphicMechanics<tSpaceDim>> tWorksetBase(tMesh);
    tWorksetBase.worksetConfig(tConfigWS);
    tWorksetBase.worksetState(tState, tStateWS);
    tWorksetBase.worksetState(tStateDotDot, tStateDotDotWS);
    
    // initialize functors 
    //
    Plato::ComputeGradientWorkset<tSpaceDim>        tComputeGradient;
    Plato::MicromorphicKinematics<tSpaceDim>        tKinematics;
    Plato::MicromorphicKinetics<tSpaceDim>          tKinetics(tMaterialModel);
    Plato::MicromorphicKinetics<tSpaceDim>          tInertiaKinetics(tInertiaModel);
    Plato::FullStressDivergence<tSpaceDim>          tComputeStressDivergence;
    Plato::ProjectStressToNode<tSpaceDim,tSpaceDim> tComputeStressForMicromorphicResidual;
    Plato::InterpolateFromNodal<tSpaceDim, tNumDofsPerNode, /*offset=*/0, tSpaceDim> tInterpolateFromNodal;
    Plato::InertialContent<tSpaceDim>               tInertialContent(tInertiaModel);
    Plato::ProjectToNode<tSpaceDim, tNumDofsPerNode> tProjectInertialContent;

    // compute on device
    //
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(int aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;

        tKinematics(aCellOrdinal, tSymDisplacementGradient, tSkwDisplacementGradient, tSymMicroDistortionTensor, tSkwMicroDistortionTensor, tStateWS, tBasisFunctions, tGradient);
        tKinematics(aCellOrdinal, tSymGradientMicroInertia, tSkwGradientMicroInertia, tSymFreeMicroInertia, tSkwFreeMicroInertia, tStateDotDotWS, tBasisFunctions, tGradient);
        tKinetics(aCellOrdinal, tSymCauchyStress, tSkwCauchyStress, tSymMicroStress, tSymDisplacementGradient, tSkwDisplacementGradient, tSymMicroDistortionTensor, tSkwMicroDistortionTensor);
        tInertiaKinetics(aCellOrdinal, tSymGradientInertiaStress, tSkwGradientInertiaStress, tSymFreeInertiaStress, tSkwFreeInertiaStress, tSymGradientMicroInertia, tSkwGradientMicroInertia, tSymFreeMicroInertia, tSkwFreeMicroInertia);
        tComputeStressDivergence(aCellOrdinal, tResidual, tSymCauchyStress, tSkwCauchyStress, tGradient, tCellVolume);
        tComputeStressForMicromorphicResidual(aCellOrdinal, tResidual, tSymCauchyStress, tSkwCauchyStress, tSymMicroStress, tBasisFunctions, tCellVolume);

        tComputeStressDivergence(aCellOrdinal, tInertiaResidual, tSymGradientInertiaStress, tSkwGradientInertiaStress, tGradient, tCellVolume);
        tComputeStressForMicromorphicResidual(aCellOrdinal, tInertiaResidual, tSymFreeInertiaStress, tSkwFreeInertiaStress, tBasisFunctions, tCellVolume);
        
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, tStateDotDotWS, tAccelerationGP);
        tInertialContent(aCellOrdinal, tInertialContentGP, tAccelerationGP);
        tProjectInertialContent(aCellOrdinal, tCellVolume, tBasisFunctions, tInertialContentGP, tInertiaResidual);

    }, "device computations");

    // test shape functions
    //
    auto tHostBasisFunctions = Kokkos::create_mirror(tBasisFunctions);
    Kokkos::deep_copy(tHostBasisFunctions, tBasisFunctions);
    double tGoldScalar = 1.0/4.0;
    TEST_FLOATING_EQUALITY(tHostBasisFunctions(0), tGoldScalar, 1e-12);
    TEST_FLOATING_EQUALITY(tHostBasisFunctions(1), tGoldScalar, 1e-12);
    TEST_FLOATING_EQUALITY(tHostBasisFunctions(2), tGoldScalar, 1e-12);
    TEST_FLOATING_EQUALITY(tHostBasisFunctions(3), tGoldScalar, 1e-12);
    
    // test cell volume
    //
    auto tCellVolume_Host = Kokkos::create_mirror_view( tCellVolume );
    Kokkos::deep_copy( tCellVolume_Host, tCellVolume );

    std::vector<Plato::Scalar> tCellVolume_gold = { 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333
    };

    for(int iCell=0; iCell<tCellVolume_gold.size(); iCell++){
      if(tCellVolume_gold[iCell] == 0.0){
        TEST_ASSERT(fabs(tCellVolume_Host(iCell)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tCellVolume_Host(iCell), tCellVolume_gold[iCell], 1e-13);
      }
    }
    
    // test shape function gradients
    //
    auto tGradient_host = Kokkos::create_mirror_view( tGradient );
    Kokkos::deep_copy( tGradient_host, tGradient );

    std::vector<std::vector<std::vector<Plato::Scalar>>> tGradient_gold = { 
      {{0.0,-2.0, 0.0},{ 2.0, 0.0, -2.0},{-2.0, 2.0, 0.0},{ 0.0, 0.0, 2.0}},
      {{0.0,-2.0, 0.0},{ 0.0, 2.0, -2.0},{-2.0, 0.0, 2.0},{ 2.0, 0.0, 0.0}},
      {{0.0, 0.0, -2.0},{-2.0, 2.0, 0.0},{ 0.0, -2.0, 2.0},{ 2.0, 0.0, 0.0}},
      {{0.0, 0.0, -2.0},{ -2.0, 0.0, 2.0},{ 2.0, -2.0, 0.0},{0.0, 2.0, 0.0}},
      {{-2.0,0.0, 0.0},{ 0.0, -2.0, 2.0},{2.0, 0.0, -2.0},{ 0.0, 2.0, 0.0}},
      {{-2.0, 0.0, 0.0},{ 2.0, -2.0, 0.0},{0.0, 2.0, -2.0},{ 0.0, 0.0, 2.0}}
    };

    int tNumGoldCells = tGradient_gold.size();
    for(int iCell=0; iCell<tNumGoldCells; iCell++){
      for(int iNode=0; iNode<tSpaceDim+1; iNode++){
        for(int iDim=0; iDim<tSpaceDim; iDim++){
          if(tGradient_gold[iCell][iNode][iDim] == 0.0){
            TEST_ASSERT(fabs(tGradient_host(iCell,iNode,iDim)) < 1e-12);
          } else {
            TEST_FLOATING_EQUALITY(tGradient_host(iCell,iNode,iDim), tGradient_gold[iCell][iNode][iDim], 1e-13);
          }
        }
      }
    }

  // test symmetric displacement gradient
  //
  auto tSymDisplacementGradient_Host = Kokkos::create_mirror_view( tSymDisplacementGradient );
  Kokkos::deep_copy( tSymDisplacementGradient_Host, tSymDisplacementGradient );

  std::vector<std::vector<Plato::Scalar>> tSymDisplacementGradient_Gold = { 
    {1.8e-06,  1.2e-06,  6.0e-07, 2.2e-06, 5.6e-06, 4.2e-06}, 
    {1.8e-06,  1.2e-06,  6.0e-07, 2.2e-06, 5.6e-06, 4.2e-06}, 
    {1.8e-06,  1.2e-06,  6.0e-07, 2.2e-06, 5.6e-06, 4.2e-06}, 
    {1.8e-06,  1.2e-06,  6.0e-07, 2.2e-06, 5.6e-06, 4.2e-06}, 
  };

  for(int iCell=0; iCell<int(tSymDisplacementGradient_Gold.size()); iCell++){
    for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
      if(tSymDisplacementGradient_Gold[iCell][iVoigt] == 0.0){
        TEST_ASSERT(fabs(tSymDisplacementGradient_Host(iCell,iVoigt)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tSymDisplacementGradient_Host(iCell,iVoigt), tSymDisplacementGradient_Gold[iCell][iVoigt], 1e-13);
      }
    }
  }
  
  // test skew displacement gradient
  //
  auto tSkwDisplacementGradient_Host = Kokkos::create_mirror_view( tSkwDisplacementGradient );
  Kokkos::deep_copy( tSkwDisplacementGradient_Host, tSkwDisplacementGradient );

  std::vector<std::vector<Plato::Scalar>> tSkwDisplacementGradient_Gold = { 
    {-1.4e-06, -5.2e-06, -3.0e-06}, 
    {-1.4e-06, -5.2e-06, -3.0e-06}, 
    {-1.4e-06, -5.2e-06, -3.0e-06}, 
    {-1.4e-06, -5.2e-06, -3.0e-06}, 
  };

  for(int iCell=0; iCell<int(tSkwDisplacementGradient_Gold.size()); iCell++){
    for(int iSkw=0; iSkw<tNumSkwTerms; iSkw++){
      if(tSkwDisplacementGradient_Gold[iCell][iSkw] == 0.0){
        TEST_ASSERT(fabs(tSkwDisplacementGradient_Host(iCell,iSkw)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tSkwDisplacementGradient_Host(iCell,iSkw), tSkwDisplacementGradient_Gold[iCell][iSkw], 1e-13);
      }
    }
  }
  
  // test symmetric micro distortion tensor
  //
  auto tSymMicroDistortionTensor_Host = Kokkos::create_mirror_view( tSymMicroDistortionTensor );
  Kokkos::deep_copy( tSymMicroDistortionTensor_Host, tSymMicroDistortionTensor );

  std::vector<std::vector<Plato::Scalar>> tSymMicroDistortionTensor_Gold = { 
    {2.8e-06, 3.5e-06,  4.2e-06, 1.19e-05, 1.33e-05, 1.47e-05}, 
    {2.0e-06, 2.5e-06,  3.0e-06, 8.5e-06,  9.5e-06,  1.05e-05},
    {1.8e-06, 2.25e-06, 2.7e-06, 7.65e-06, 8.55e-06, 9.45e-06},
    {2.4e-06, 3.0e-06,  3.6e-06, 1.02e-05, 1.14e-05, 1.26e-05}
  };

  for(int iCell=0; iCell<int(tSymMicroDistortionTensor_Gold.size()); iCell++){
    for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
      if(tSymMicroDistortionTensor_Gold[iCell][iVoigt] == 0.0){
        TEST_ASSERT(fabs(tSymMicroDistortionTensor_Host(iCell,iVoigt)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tSymMicroDistortionTensor_Host(iCell,iVoigt), tSymMicroDistortionTensor_Gold[iCell][iVoigt], 1e-13);
      }
    }
  }
  
  // test skew micro distortion tensor
  //
  auto tSkwMicroDistortionTensor_Host = Kokkos::create_mirror_view( tSkwMicroDistortionTensor );
  Kokkos::deep_copy( tSkwMicroDistortionTensor_Host, tSkwMicroDistortionTensor );

  std::vector<std::vector<Plato::Scalar>> tSkwMicroDistortionTensor_Gold = { 
    {-2.1e-06,  -2.1e-06,  -2.1e-06}, 
    {-1.5e-06,  -1.5e-06,  -1.5e-06},
    {-1.35e-06, -1.35e-06, -1.35e-06},
    {-1.8e-06,  -1.8e-06,  -1.8e-06}
  };

  for(int iCell=0; iCell<int(tSkwMicroDistortionTensor_Gold.size()); iCell++){
    for(int iSkw=0; iSkw<tNumSkwTerms; iSkw++){
      if(tSkwMicroDistortionTensor_Gold[iCell][iSkw] == 0.0){
        TEST_ASSERT(fabs(tSkwMicroDistortionTensor_Host(iCell,iSkw)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tSkwMicroDistortionTensor_Host(iCell,iSkw), tSkwMicroDistortionTensor_Gold[iCell][iSkw], 1e-13);
      }
    }
  }
  
  // test symmetric cauchy stress
  //
  auto tSymCauchyStress_Host = Kokkos::create_mirror_view( tSymCauchyStress );
  Kokkos::deep_copy( tSymCauchyStress_Host, tSymCauchyStress );

  std::vector<std::vector<Plato::Scalar>> tSymCauchyStress_Gold = { 
    {-2.811140000000000e-04, -1.729600000000000e-03, -3.178086000000000e-03, -8.118899999999998e-05, -6.444900000000001e-05, -8.788499999999998e-05}, 
    {2.480420000000000e-04, -9.775999999999999e-04, -2.203242000000000e-03, -5.273099999999999e-05, -3.264300000000001e-05, -5.273100000000001e-05},
    {3.803310000000000e-04, -7.896000000000001e-04, -1.959531000000000e-03, -4.561649999999999e-05, -2.469150000000000e-05, -4.394249999999999e-05}
  };

  for(int iCell=0; iCell<int(tSymCauchyStress_Gold.size()); iCell++){
    for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
      if(tSymCauchyStress_Gold[iCell][iVoigt] == 0.0){
        TEST_ASSERT(fabs(tSymCauchyStress_Host(iCell,iVoigt)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tSymCauchyStress_Host(iCell,iVoigt), tSymCauchyStress_Gold[iCell][iVoigt], 1e-13);
      }
    }
  }
  
  // test skew cauchy stress
  //
  auto tSkwCauchyStress_Host = Kokkos::create_mirror_view( tSkwCauchyStress );
  Kokkos::deep_copy( tSkwCauchyStress_Host, tSkwCauchyStress );

  std::vector<std::vector<Plato::Scalar>> tSkwCauchyStress_Gold = { 
    {0.0, 0.0, 0.0, 1.260000000000001e-10,  -5.579999999999998e-10,  -1.620000000000001e-10}, 
    {0.0, 0.0, 0.0, 1.799999999999994e-11, -6.660000000000001e-10, -2.700000000000000e-10},
    {0.0, 0.0, 0.0, -8.999999999999953e-12, -6.930000000000001e-10, -2.970000000000001e-10}
  };

  for(int iCell=0; iCell<int(tSkwCauchyStress_Gold.size()); iCell++){
    for(int iSkw=0; iSkw<tNumVoigtTerms; iSkw++){
      if(tSkwCauchyStress_Gold[iCell][iSkw] == 0.0){
        TEST_ASSERT(fabs(tSkwCauchyStress_Host(iCell,iSkw)) < 1e-14);
      } else {
        TEST_FLOATING_EQUALITY(tSkwCauchyStress_Host(iCell,iSkw), tSkwCauchyStress_Gold[iCell][iSkw], 1e-13);
      }
    }
  }
  
  // test symmetric micro stress
  //
  auto tSymMicroStress_Host = Kokkos::create_mirror_view( tSymMicroStress );
  Kokkos::deep_copy( tSymMicroStress_Host, tSymMicroStress );

  std::vector<std::vector<Plato::Scalar>> tSymMicroStress_Gold = { 
    {3.328591000000000e-03, 3.686585000000000e-03, 4.044579000000000e-03, 2.157232000000000e-03, 2.411024000000000e-03, 2.664816000000000e-03}, 
    {2.377565000000000e-03, 2.633275000000000e-03, 2.888985000000000e-03, 1.540880000000000e-03, 1.722160000000000e-03, 1.903440000000000e-03},
    {2.139808500000000e-03, 2.369947500000000e-03, 2.600086500000000e-03, 1.386792000000000e-03, 1.549944000000000e-03, 1.713096000000000e-03}
  };

  for(int iCell=0; iCell<int(tSymMicroStress_Gold.size()); iCell++){
    for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
      if(tSymMicroStress_Gold[iCell][iVoigt] == 0.0){
        TEST_ASSERT(fabs(tSymMicroStress_Host(iCell,iVoigt)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tSymMicroStress_Host(iCell,iVoigt), tSymMicroStress_Gold[iCell][iVoigt], 1e-13);
      }
    }
  }
  
  // test residual
  //
  auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
  Kokkos::deep_copy( tResidual_Host, tResidual );

  std::vector<std::vector<Plato::Scalar>> tResidual_gold = { 
   {3.66188174999999e-06, 7.20666666666666e-05, 3.38288024999999e-06, 1.8800546875e-05, 2.8209296875e-05, 3.76180468749999e-05, 1.16584420520833e-05, 1.28930914479166e-05, 1.433698521875e-05, 1.16584433645833e-05, 1.28930856354166e-05, 1.433698353125e-05,
    -9.02768508333332e-06, -2.78998499999999e-07, 0.00012973489825, 1.8800546875e-05, 2.8209296875e-05, 3.76180468749999e-05, 1.16584420520833e-05, 1.28930914479166e-05, 1.433698521875e-05, 1.16584433645833e-05, 1.28930856354166e-05, 1.433698353125e-05,
    8.05120158333332e-06, -6.84047984166666e-05, -6.97528499999998e-07, 1.8800546875e-05, 2.8209296875e-05, 3.76180468749999e-05, 1.16584420520833e-05, 1.28930914479166e-05, 1.433698521875e-05, 1.16584433645833e-05, 1.28930856354166e-05, 1.433698353125e-05,
    -2.68539825e-06, -3.38286974999999e-06, -0.00013242025, 1.8800546875e-05, 2.8209296875e-05, 3.76180468749999e-05, 1.16584420520833e-05, 1.28930914479166e-05, 1.433698521875e-05, 1.16584433645833e-05, 1.28930856354166e-05, 1.433698353125e-05},
   {2.19713625e-06, 4.07333333333333e-05, 2.19712575e-06, 1.1091265625e-05, 1.8806640625e-05, 2.6522015625e-05, 8.30005719791665e-06, 9.13960242708332e-06, 1.018839203125e-05, 8.30005738541665e-06, 9.13959548958332e-06, 1.018838921875e-05,
    -8.36983499999999e-07, -3.85362090833333e-05, 8.96046242499999e-05, 1.1091265625e-05, 1.8806640625e-05, 2.6522015625e-05, 8.30005719791665e-06, 9.13960242708332e-06, 1.018839203125e-05, 8.30005738541665e-06, 9.13959548958332e-06, 1.018838921875e-05,
    -1.16952360833333e-05, -1.04999999994353e-11, -9.04416527499998e-05, 1.1091265625e-05, 1.8806640625e-05, 2.6522015625e-05, 8.30005719791665e-06, 9.13960242708332e-06, 1.018839203125e-05, 8.30005738541665e-06, 9.13959548958332e-06, 1.018838921875e-05,
    1.03350833333333e-05, -2.19711375e-06, -1.36009725e-06, 1.1091265625e-05, 1.8806640625e-05, 2.6522015625e-05, 8.30005719791665e-06, 9.13960242708332e-06, 1.018839203125e-05, 8.30005738541665e-06, 9.13959548958332e-06, 1.018838921875e-05}
  };

  for(int iCell=0; iCell<int(tResidual_gold.size()); iCell++){
    for(int iDof=0; iDof<tNumDofsPerCell; iDof++){
      if(tResidual_gold[iCell][iDof] == 0.0){
        TEST_ASSERT(fabs(tResidual_Host(iCell,iDof)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tResidual_Host(iCell,iDof), tResidual_gold[iCell][iDof], 1e-10);
      }
    }
  }
  
  // test symmetric gradient micro inertia
  //
  auto tSymGradientMicroInertia_Host = Kokkos::create_mirror_view( tSymGradientMicroInertia );
  Kokkos::deep_copy( tSymGradientMicroInertia_Host, tSymGradientMicroInertia );

  std::vector<std::vector<Plato::Scalar>> tSymGradientMicroInertia_Gold = { 
    {1.8e-04,  1.2e-04,  6.0e-05, 2.2e-04, 5.6e-04, 4.2e-04}, 
    {1.8e-04,  1.2e-04,  6.0e-05, 2.2e-04, 5.6e-04, 4.2e-04}, 
    {1.8e-04,  1.2e-04,  6.0e-05, 2.2e-04, 5.6e-04, 4.2e-04}, 
    {1.8e-04,  1.2e-04,  6.0e-05, 2.2e-04, 5.6e-04, 4.2e-04}, 
  };

  for(int iCell=0; iCell<int(tSymGradientMicroInertia_Gold.size()); iCell++){
    for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
      if(tSymGradientMicroInertia_Gold[iCell][iVoigt] == 0.0){
        TEST_ASSERT(fabs(tSymGradientMicroInertia_Host(iCell,iVoigt)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tSymGradientMicroInertia_Host(iCell,iVoigt), tSymGradientMicroInertia_Gold[iCell][iVoigt], 1e-13);
      }
    }
  }
  
  // test skew gradient micro inertia
  //
  auto tSkwGradientMicroInertia_Host = Kokkos::create_mirror_view( tSkwGradientMicroInertia );
  Kokkos::deep_copy( tSkwGradientMicroInertia_Host, tSkwGradientMicroInertia );

  std::vector<std::vector<Plato::Scalar>> tSkwGradientMicroInertia_Gold = { 
    {-1.4e-04, -5.2e-04, -3.0e-04}, 
    {-1.4e-04, -5.2e-04, -3.0e-04}, 
    {-1.4e-04, -5.2e-04, -3.0e-04}, 
    {-1.4e-04, -5.2e-04, -3.0e-04}, 
  };

  for(int iCell=0; iCell<int(tSkwGradientMicroInertia_Gold.size()); iCell++){
    for(int iSkw=0; iSkw<tNumSkwTerms; iSkw++){
      if(tSkwGradientMicroInertia_Gold[iCell][iSkw] == 0.0){
        TEST_ASSERT(fabs(tSkwGradientMicroInertia_Host(iCell,iSkw)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tSkwGradientMicroInertia_Host(iCell,iSkw), tSkwGradientMicroInertia_Gold[iCell][iSkw], 1e-13);
      }
    }
  }
  
  // test symmetric free micro inertia
  //
  auto tSymFreeMicroInertia_Host = Kokkos::create_mirror_view( tSymFreeMicroInertia );
  Kokkos::deep_copy( tSymFreeMicroInertia_Host, tSymFreeMicroInertia );

  std::vector<std::vector<Plato::Scalar>> tSymFreeMicroInertia_Gold = { 
    {2.8e-04, 3.5e-04,  4.2e-04, 1.19e-03, 1.33e-03, 1.47e-03}, 
    {2.0e-04, 2.5e-04,  3.0e-04, 8.5e-04,  9.5e-04,  1.05e-03},
    {1.8e-04, 2.25e-04, 2.7e-04, 7.65e-04, 8.55e-04, 9.45e-04},
    {2.4e-04, 3.0e-04,  3.6e-04, 1.02e-03, 1.14e-03, 1.26e-03}
  };

  for(int iCell=0; iCell<int(tSymFreeMicroInertia_Gold.size()); iCell++){
    for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
      if(tSymFreeMicroInertia_Gold[iCell][iVoigt] == 0.0){
        TEST_ASSERT(fabs(tSymFreeMicroInertia_Host(iCell,iVoigt)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tSymFreeMicroInertia_Host(iCell,iVoigt), tSymFreeMicroInertia_Gold[iCell][iVoigt], 1e-13);
      }
    }
  }
  
  // test skew free micro inertia
  //
  auto tSkwFreeMicroInertia_Host = Kokkos::create_mirror_view( tSkwFreeMicroInertia );
  Kokkos::deep_copy( tSkwFreeMicroInertia_Host, tSkwFreeMicroInertia );

  std::vector<std::vector<Plato::Scalar>> tSkwFreeMicroInertia_Gold = { 
    {-2.1e-04,  -2.1e-04,  -2.1e-04}, 
    {-1.5e-04,  -1.5e-04,  -1.5e-04},
    {-1.35e-04, -1.35e-04, -1.35e-04},
    {-1.8e-04,  -1.8e-04,  -1.8e-04}
  };

  for(int iCell=0; iCell<int(tSkwFreeMicroInertia_Gold.size()); iCell++){
    for(int iSkw=0; iSkw<tNumSkwTerms; iSkw++){
      if(tSkwFreeMicroInertia_Gold[iCell][iSkw] == 0.0){
        TEST_ASSERT(fabs(tSkwFreeMicroInertia_Host(iCell,iSkw)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tSkwFreeMicroInertia_Host(iCell,iSkw), tSkwFreeMicroInertia_Gold[iCell][iSkw], 1e-13);
      }
    }
  }
  
  // test symmetric gradient inertia stress
  //
  auto tSymGradientInertiaStress_Host = Kokkos::create_mirror_view( tSymGradientInertiaStress );
  Kokkos::deep_copy( tSymGradientInertiaStress_Host, tSymGradientInertiaStress );

  std::vector<std::vector<Plato::Scalar>> tSymGradientInertiaStress_Gold = { 
    {0.000936, 0.000864, 0.000792, 4.4e-05, 0.000112, 8.4e-05}, 
    {0.000936, 0.000864, 0.000792, 4.4e-05, 0.000112, 8.4e-05}, 
    {0.000936, 0.000864, 0.000792, 4.4e-05, 0.000112, 8.4e-05}, 
  };

  for(int iCell=0; iCell<int(tSymGradientInertiaStress_Gold.size()); iCell++){
    for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
      if(tSymGradientInertiaStress_Gold[iCell][iVoigt] == 0.0){
        TEST_ASSERT(fabs(tSymGradientInertiaStress_Host(iCell,iVoigt)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tSymGradientInertiaStress_Host(iCell,iVoigt), tSymGradientInertiaStress_Gold[iCell][iVoigt], 1e-13);
      }
    }
  }
  
  // test skew gradient inertia stress
  //
  auto tSkwGradientInertiaStress_Host = Kokkos::create_mirror_view( tSkwGradientInertiaStress );
  Kokkos::deep_copy( tSkwGradientInertiaStress_Host, tSkwGradientInertiaStress );

  std::vector<std::vector<Plato::Scalar>> tSkwGradientInertiaStress_Gold = { 
    {0.0, 0.0, 0.0, -1.4e-08, -5.2e-08, -3e-08}, 
    {0.0, 0.0, 0.0, -1.4e-08, -5.2e-08, -3e-08}, 
    {0.0, 0.0, 0.0, -1.4e-08, -5.2e-08, -3e-08}, 
  };

  for(int iCell=0; iCell<int(tSkwGradientInertiaStress_Gold.size()); iCell++){
    for(int iSkw=0; iSkw<tNumVoigtTerms; iSkw++){
      if(tSkwGradientInertiaStress_Gold[iCell][iSkw] == 0.0){
        TEST_ASSERT(fabs(tSkwGradientInertiaStress_Host(iCell,iSkw)) < 1e-14);
      } else {
        TEST_FLOATING_EQUALITY(tSkwGradientInertiaStress_Host(iCell,iSkw), tSkwGradientInertiaStress_Gold[iCell][iSkw], 1e-13);
      }
    }
  }
  
  // test symmetric free inertia stress
  //
  auto tSymFreeInertiaStress_Host = Kokkos::create_mirror_view( tSymFreeInertiaStress );
  Kokkos::deep_copy( tSymFreeInertiaStress_Host, tSymFreeInertiaStress );

  std::vector<std::vector<Plato::Scalar>> tSymFreeInertiaStress_Gold = { 
    {-0.602, -0.28, 0.0420000000000003, 5.355, 5.985, 6.615}, 
    {-0.43, -0.2, 0.03, 3.825, 4.275, 4.725},
    {-0.387, -0.18, 0.0270000000000001, 3.4425, 3.8475, 4.2525}
  };

  for(int iCell=0; iCell<int(tSymFreeInertiaStress_Gold.size()); iCell++){
    for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
      if(tSymFreeInertiaStress_Gold[iCell][iVoigt] == 0.0){
        TEST_ASSERT(fabs(tSymFreeInertiaStress_Host(iCell,iVoigt)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tSymFreeInertiaStress_Host(iCell,iVoigt), tSymFreeInertiaStress_Gold[iCell][iVoigt], 1e-13);
      }
    }
  }
  
  // test skew free inertia stress
  //
  auto tSkwFreeInertiaStress_Host = Kokkos::create_mirror_view( tSkwFreeInertiaStress );
  Kokkos::deep_copy( tSkwFreeInertiaStress_Host, tSkwFreeInertiaStress );

  std::vector<std::vector<Plato::Scalar>> tSkwFreeInertiaStress_Gold = { 
    {0.0, 0.0, 0.0, -2.1e-08, -2.1e-08, -2.1e-08}, 
    {0.0, 0.0, 0.0, -1.5e-08, -1.5e-08, -1.5e-08},
    {0.0, 0.0, 0.0, -1.35e-08, -1.35e-08, -1.35e-08}
  };

  for(int iCell=0; iCell<int(tSkwFreeInertiaStress_Gold.size()); iCell++){
    for(int iSkw=0; iSkw<tNumVoigtTerms; iSkw++){
      if(tSkwFreeInertiaStress_Gold[iCell][iSkw] == 0.0){
        TEST_ASSERT(fabs(tSkwFreeInertiaStress_Host(iCell,iSkw)) < 1e-14);
      } else {
        TEST_FLOATING_EQUALITY(tSkwFreeInertiaStress_Host(iCell,iSkw), tSkwFreeInertiaStress_Gold[iCell][iSkw], 1e-13);
      }
    }
  }
  
  // test inertia residual
  //
  auto tInertiaResidual_Host = Kokkos::create_mirror_view( tInertiaResidual );
  Kokkos::deep_copy( tInertiaResidual_Host, tInertiaResidual );

  std::vector<std::vector<Plato::Scalar>> tInertiaResidual_gold = { 
   {0.000525803333333333, 0.00102260416666667, 0.00158607233333333, -0.00313541666666666, -0.00145833333333333, 0.000218750000000001, 0.027890624890625, 0.031171874890625, 0.034453124890625, 0.027890625109375, 0.031171875109375, 0.034453125109375,
    0.000563637583333333, 0.00106027266666667, 0.00155957508333333, -0.00313541666666666, -0.00145833333333333, 0.000218750000000001, 0.027890624890625, 0.031171874890625, 0.034453124890625, 0.027890625109375, 0.031171875109375, 0.034453125109375,
    0.000493800833333333, 0.00109110291666667, 0.00158507133333333, -0.00313541666666666, -0.00145833333333333, 0.000218750000000001, 0.027890624890625, 0.031171874890625, 0.034453124890625, 0.027890625109375, 0.031171875109375, 0.034453125109375,
    0.000533966583333333, 0.00106043691666667, 0.00162090625, -0.00313541666666666, -0.00145833333333333, 0.000218750000000001, 0.027890624890625, 0.031171874890625, 0.034453124890625, 0.027890625109375, 0.031171875109375, 0.034453125109375},
   {0.000374574166666666, 0.000720145833333332, 0.00113238483333333, -0.00223958333333333, -0.00104166666666667, 0.00015625, 0.019921874921875, 0.022265624921875, 0.024609374921875, 0.019921875078125, 0.022265625078125, 0.024609375078125,
    0.000376907166666666, 0.000790313083333332, 0.00110305266666666, -0.00223958333333333, -0.00104166666666667, 0.00015625, 0.019921874921875, 0.022265624921875, 0.024609374921875, 0.019921875078125, 0.022265625078125, 0.024609375078125,
    0.000343737416666666, 0.000754477333333332, 0.00116254991666667, -0.00223958333333333, -0.00104166666666667, 0.00015625, 0.019921874921875, 0.022265624921875, 0.024609374921875, 0.019921875078125, 0.022265625078125, 0.024609375078125,
    0.000417072916666666, 0.000759647083333332, 0.00113888758333333, -0.00223958333333333, -0.00104166666666667, 0.00015625, 0.019921874921875, 0.022265624921875, 0.024609374921875, 0.019921875078125, 0.022265625078125, 0.024609375078125}
  };

  for(int iCell=0; iCell<int(tInertiaResidual_gold.size()); iCell++){
    for(int iDof=0; iDof<tNumDofsPerCell; iDof++){
      if(tInertiaResidual_gold[iCell][iDof] == 0.0){
        TEST_ASSERT(fabs(tInertiaResidual_Host(iCell,iDof)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tInertiaResidual_Host(iCell,iDof), tInertiaResidual_gold[iCell][iDof], 1e-13);
      }
    }
  }

}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, ErrorAFormNotSpecified)
{
    // create test mesh
    //
    constexpr int tMeshWidth=2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);
    int tNumCells = tMesh->NumElements();

    // set parameters
    //
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Problem'>                                    \n"
      "  <Parameter name='Physics' type='string' value='Plato Driver' />  \n"
      "  <ParameterList name='Plato Problem'>                                    \n"
      "    <Parameter name='Physics' type='string' value='Micromorphic Mechanical' />  \n"
      "    <Parameter name='PDE Constraint' type='string' value='Hyperbolic' /> \n"
      "    <ParameterList name='Material Models'>                           \n"
      "      <ParameterList name='material_1'>                           \n"
      "        <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "          <Parameter  name='Lambda_e' type='double' value='-120.74'/>   \n"
      "          <Parameter  name='Mu_e' type='double' value='557.11'/>   \n"
      "          <Parameter  name='Mu_star_e' type='double' value='8.37'/>   \n"
      "          <Parameter  name='Mu_c' type='double' value='1.8e-4'/>   \n"
      "          <Parameter  name='Lambda_m' type='double' value='180.63'/>   \n"
      "          <Parameter  name='Mu_m' type='double' value='255.71'/>   \n"
      "          <Parameter  name='Mu_star_m' type='double' value='181.28'/>   \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "          <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "          <Parameter  name='Eta_bar_1' type='double' value='0.6'/>   \n"
      "          <Parameter  name='Eta_bar_3' type='double' value='2.0'/>   \n"
      "          <Parameter  name='Eta_bar_2' type='double' value='1.0e-4'/>   \n"
      "          <Parameter  name='Eta_bar_star_1' type='double' value='0.2'/>   \n"
      "          <Parameter  name='Eta_1' type='double' value='2300.0'/>   \n"
      "          <Parameter  name='Eta_3' type='double' value='-1800.0'/>   \n"
      "          <Parameter  name='Eta_2' type='double' value='1.0e-4'/>   \n"
      "          <Parameter  name='Eta_star_1' type='double' value='4500.0'/>   \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "    <ParameterList name='Spatial Model'>                                    \n"
      "      <ParameterList name='Domains'>                                        \n"
      "        <ParameterList name='Design Volume'>                                \n"
      "          <Parameter name='Element Block' type='string' value='body'/>      \n"
      "          <Parameter name='Material Model' type='string' value='material_1'/> \n"
      "        </ParameterList>                                                    \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Hyperbolic'>                                    \n"
      "      <ParameterList name='Penalty Function'>                                    \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>      \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>      \n"
      "        <Parameter name='Minimum Value' type='double' value='1e-9'/>      \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Time Integration'>                                    \n"
      "      <Parameter name='Termination Time' type='double' value='20.0e-6'/>      \n"
      "      <Parameter name='Newmark Gamma' type='double' value='0.5'/>      \n"
      "      <Parameter name='Newmark Beta' type='double' value='0.0'/>      \n"
      "    </ParameterList>                                                        \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);
    
    Plato::ProblemFactory<3> tProblemFactory;
    TEST_THROW(tProblemFactory.create(tMesh, *tParams, tMachine), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, ErrorAFormFalse)
{
    // create test mesh
    //
    constexpr int tMeshWidth=2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);
    int tNumCells = tMesh->NumElements();

    // set parameters
    //
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Problem'>                                    \n"
      "  <Parameter name='Physics' type='string' value='Plato Driver' />  \n"
      "  <ParameterList name='Plato Problem'>                                    \n"
      "    <Parameter name='Physics' type='string' value='Micromorphic Mechanical' />  \n"
      "    <Parameter name='PDE Constraint' type='string' value='Hyperbolic' /> \n"
      "    <ParameterList name='Material Models'>                           \n"
      "      <ParameterList name='material_1'>                           \n"
      "        <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "          <Parameter  name='Lambda_e' type='double' value='-120.74'/>   \n"
      "          <Parameter  name='Mu_e' type='double' value='557.11'/>   \n"
      "          <Parameter  name='Mu_star_e' type='double' value='8.37'/>   \n"
      "          <Parameter  name='Mu_c' type='double' value='1.8e-4'/>   \n"
      "          <Parameter  name='Lambda_m' type='double' value='180.63'/>   \n"
      "          <Parameter  name='Mu_m' type='double' value='255.71'/>   \n"
      "          <Parameter  name='Mu_star_m' type='double' value='181.28'/>   \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "          <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "          <Parameter  name='Eta_bar_1' type='double' value='0.6'/>   \n"
      "          <Parameter  name='Eta_bar_3' type='double' value='2.0'/>   \n"
      "          <Parameter  name='Eta_bar_2' type='double' value='1.0e-4'/>   \n"
      "          <Parameter  name='Eta_bar_star_1' type='double' value='0.2'/>   \n"
      "          <Parameter  name='Eta_1' type='double' value='2300.0'/>   \n"
      "          <Parameter  name='Eta_3' type='double' value='-1800.0'/>   \n"
      "          <Parameter  name='Eta_2' type='double' value='1.0e-4'/>   \n"
      "          <Parameter  name='Eta_star_1' type='double' value='4500.0'/>   \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "    <ParameterList name='Spatial Model'>                                    \n"
      "      <ParameterList name='Domains'>                                        \n"
      "        <ParameterList name='Design Volume'>                                \n"
      "          <Parameter name='Element Block' type='string' value='body'/>      \n"
      "          <Parameter name='Material Model' type='string' value='material_1'/> \n"
      "        </ParameterList>                                                    \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Hyperbolic'>                                    \n"
      "      <ParameterList name='Penalty Function'>                                    \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>      \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>      \n"
      "        <Parameter name='Minimum Value' type='double' value='1e-9'/>      \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Time Integration'>                                    \n"
      "      <Parameter name='Termination Time' type='double' value='20.0e-6'/>      \n"
      "      <Parameter name='A-Form' type='bool' value='false'/>      \n"
      "      <Parameter name='Newmark Gamma' type='double' value='0.5'/>      \n"
      "      <Parameter name='Newmark Beta' type='double' value='0.0'/>      \n"
      "    </ParameterList>                                                        \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);
    
    Plato::ProblemFactory<3> tProblemFactory;
    TEST_THROW(tProblemFactory.create(tMesh, *tParams, tMachine), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, ErrorExplicitNotSpecified)
{
    // create test mesh
    //
    constexpr int tMeshWidth=2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);
    int tNumCells = tMesh->NumElements();

    // set parameters
    //
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Problem'>                                    \n"
      "  <Parameter name='Physics' type='string' value='Plato Driver' />  \n"
      "  <ParameterList name='Plato Problem'>                                    \n"
      "    <Parameter name='Physics' type='string' value='Micromorphic Mechanical' />  \n"
      "    <Parameter name='PDE Constraint' type='string' value='Hyperbolic' /> \n"
      "    <ParameterList name='Material Models'>                           \n"
      "      <ParameterList name='material_1'>                           \n"
      "        <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "          <Parameter  name='Lambda_e' type='double' value='-120.74'/>   \n"
      "          <Parameter  name='Mu_e' type='double' value='557.11'/>   \n"
      "          <Parameter  name='Mu_star_e' type='double' value='8.37'/>   \n"
      "          <Parameter  name='Mu_c' type='double' value='1.8e-4'/>   \n"
      "          <Parameter  name='Lambda_m' type='double' value='180.63'/>   \n"
      "          <Parameter  name='Mu_m' type='double' value='255.71'/>   \n"
      "          <Parameter  name='Mu_star_m' type='double' value='181.28'/>   \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "          <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "          <Parameter  name='Eta_bar_1' type='double' value='0.6'/>   \n"
      "          <Parameter  name='Eta_bar_3' type='double' value='2.0'/>   \n"
      "          <Parameter  name='Eta_bar_2' type='double' value='1.0e-4'/>   \n"
      "          <Parameter  name='Eta_bar_star_1' type='double' value='0.2'/>   \n"
      "          <Parameter  name='Eta_1' type='double' value='2300.0'/>   \n"
      "          <Parameter  name='Eta_3' type='double' value='-1800.0'/>   \n"
      "          <Parameter  name='Eta_2' type='double' value='1.0e-4'/>   \n"
      "          <Parameter  name='Eta_star_1' type='double' value='4500.0'/>   \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "    <ParameterList name='Spatial Model'>                                    \n"
      "      <ParameterList name='Domains'>                                        \n"
      "        <ParameterList name='Design Volume'>                                \n"
      "          <Parameter name='Element Block' type='string' value='body'/>      \n"
      "          <Parameter name='Material Model' type='string' value='material_1'/> \n"
      "        </ParameterList>                                                    \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Hyperbolic'>                                    \n"
      "      <ParameterList name='Penalty Function'>                                    \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>      \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>      \n"
      "        <Parameter name='Minimum Value' type='double' value='1e-9'/>      \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Time Integration'>                                    \n"
      "      <Parameter name='Termination Time' type='double' value='20.0e-6'/>      \n"
      "      <Parameter name='A-Form' type='bool' value='true'/>      \n"
      "      <Parameter name='Newmark Gamma' type='double' value='0.5'/>      \n"
      "      <Parameter name='Newmark Beta' type='double' value='0.25'/>      \n"
      "    </ParameterList>                                                        \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);
    
    Plato::ProblemFactory<3> tProblemFactory;
    TEST_THROW(tProblemFactory.create(tMesh, *tParams, tMachine), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, CreateProblem)
{
    // create test mesh
    //
    constexpr int tMeshWidth=2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);
    int tNumCells = tMesh->NumElements();

    // set parameters
    //
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Problem'>                                    \n"
      "  <Parameter name='Physics' type='string' value='Plato Driver' />  \n"
      "  <ParameterList name='Plato Problem'>                                    \n"
      "    <Parameter name='Physics' type='string' value='Micromorphic Mechanical' />  \n"
      "    <Parameter name='PDE Constraint' type='string' value='Hyperbolic' /> \n"
      "    <ParameterList name='Material Models'>                           \n"
      "      <ParameterList name='material_1'>                           \n"
      "        <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "          <Parameter  name='Lambda_e' type='double' value='-120.74'/>   \n"
      "          <Parameter  name='Mu_e' type='double' value='557.11'/>   \n"
      "          <Parameter  name='Mu_star_e' type='double' value='8.37'/>   \n"
      "          <Parameter  name='Mu_c' type='double' value='1.8e-4'/>   \n"
      "          <Parameter  name='Lambda_m' type='double' value='180.63'/>   \n"
      "          <Parameter  name='Mu_m' type='double' value='255.71'/>   \n"
      "          <Parameter  name='Mu_star_m' type='double' value='181.28'/>   \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "          <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "          <Parameter  name='Eta_bar_1' type='double' value='0.6'/>   \n"
      "          <Parameter  name='Eta_bar_3' type='double' value='2.0'/>   \n"
      "          <Parameter  name='Eta_bar_2' type='double' value='1.0e-4'/>   \n"
      "          <Parameter  name='Eta_bar_star_1' type='double' value='0.2'/>   \n"
      "          <Parameter  name='Eta_1' type='double' value='2300.0'/>   \n"
      "          <Parameter  name='Eta_3' type='double' value='-1800.0'/>   \n"
      "          <Parameter  name='Eta_2' type='double' value='1.0e-4'/>   \n"
      "          <Parameter  name='Eta_star_1' type='double' value='4500.0'/>   \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "    <ParameterList name='Spatial Model'>                                    \n"
      "      <ParameterList name='Domains'>                                        \n"
      "        <ParameterList name='Design Volume'>                                \n"
      "          <Parameter name='Element Block' type='string' value='body'/>      \n"
      "          <Parameter name='Material Model' type='string' value='material_1'/> \n"
      "        </ParameterList>                                                    \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Hyperbolic'>                                    \n"
      "      <ParameterList name='Penalty Function'>                                    \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>      \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>      \n"
      "        <Parameter name='Minimum Value' type='double' value='1e-9'/>      \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Time Integration'>                                    \n"
      "      <Parameter name='Termination Time' type='double' value='20.0e-6'/>      \n"
      "      <Parameter name='A-Form' type='bool' value='true'/>      \n"
      "      <Parameter name='Newmark Gamma' type='double' value='0.5'/>      \n"
      "      <Parameter name='Newmark Beta' type='double' value='0.0'/>      \n"
      "    </ParameterList>                                                        \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    Plato::ProblemFactory<3> tProblemFactory;
    TEST_NOTHROW(tProblemFactory.create(tMesh, *tParams, tMachine));
}

}
// namespace RelaxedMicromorphicTest


