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

#include "hyperbolic/MicromorphicElasticModelFactory.hpp"
#include "hyperbolic/MicromorphicInertiaModelFactory.hpp"
#include "hyperbolic/MicromorphicKinematics.hpp"
#include "hyperbolic/MicromorphicKinetics.hpp"
#include "hyperbolic/FullStressDivergence.hpp"
#include "hyperbolic/ProjectStressToNode.hpp"
#include "hyperbolic/InertialContent.hpp"

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
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    int tNumCells = tMesh->nelems();

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
    int tNumNodes = tMesh->nverts();
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
    Plato::WorksetBase<Plato::SimplexMicromorphicMechanics<tSpaceDim>> tWorksetBase(*tMesh);
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
      {{0.0,-2.0, 0.0},{ 2.0, 0.0,-2.0},{-2.0, 2.0, 0.0},{ 0.0, 0.0, 2.0}},
      {{0.0,-2.0, 0.0},{ 0.0, 2.0,-2.0},{-2.0, 0.0, 2.0},{ 2.0, 0.0, 0.0}},
      {{0.0, 0.0,-2.0},{-2.0, 2.0, 0.0},{ 0.0,-2.0, 2.0},{ 2.0, 0.0, 0.0}},
      {{0.0, 0.0,-2.0},{ 2.0,-2.0, 0.0},{ 0.0, 2.0, 0.0},{-2.0, 0.0, 2.0}},
      {{0.0,-2.0, 0.0},{ 2.0, 0.0,-2.0},{-2.0, 2.0, 0.0},{ 0.0, 0.0, 2.0}},
      {{0.0,-2.0, 0.0},{ 0.0, 2.0,-2.0},{-2.0, 0.0, 2.0},{ 2.0, 0.0, 0.0}}
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
    {2.0e-07,  3.2e-06,  2.4e-06, 6.4e-06, 1.4e-06, 2.0e-06}, 
    {2.0e-06,  3.2e-06, -3.0e-06, 2.8e-06, 5.0e-06, 5.6e-06},
    {2.0e-06,  8.0e-07,  6.0e-07, 1.6e-06, 6.2e-06, 4.4e-06},
    {4.0e-06, -3.2e-06,  6.0e-07, -4.4e-06, 1.22e-05, 6.4e-06}
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
    {-3.2e-06,  2.0e-07,  1.2e-06}, 
    {-6.8e-06,  -7.0e-06, -2.4e-06},
    {-8.0e-07,  -5.8e-06,  -3.6e-06},
    {5.2e-06, -1.18e-05, -9.6e-06}
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
    {3.0e-06,  3.75e-06,  4.5e-06, 1.275e-05, 1.425e-05, 1.575e-05}, 
    {2.4e-06,  3.0e-06, 3.6e-06, 1.02e-05, 1.14e-05, 1.26e-05},
    {1.7e-06,  2.125e-06, 2.55e-06, 7.225e-06, 8.075e-06, 8.925e-06},
    {3.5e-06, 4.375e-06,  5.25e-06, 1.4875e-05, 1.6625e-05, 1.8375e-05}
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
    {-2.25e-06,  -2.25e-06,  -2.25e-06}, 
    {-1.8e-06,  -1.8e-06, -1.8e-06},
    {-1.275e-06,  -1.275e-06, -1.275e-06},
    {-2.625e-06, -2.625e-06, -2.625e-06}
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
    {-2.461783e-03, 45.2119999999998e-06, -1.681829e-03, -53.1495e-06, -107.5545e-06, -115.0875e-06}, 
    {375.344e-06, 1.043876e-03, -6.53282e-03, -61.938e-06, -53.568e-06, -58.59e-06},
    {693.4675e-06, -1.11714e-03, -1.8135275e-03, -47.08125e-06, -15.69375e-06, -37.87425e-06}
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
    {0.0, 0.0, 0.0, -171.0e-12,  441.0e-12,  621.0e-12}, 
    {0.0, 0.0, 0.0, -900.0e-12, -936.0e-012, -108.0e-012},
    {0.0, 0.0, 0.0, 85.5e-012,  -814.5e-012, -418.5e-012}
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
    {3.5663475e-03, 3.9499125e-03, 4.3334775e-03, 2.31132e-03, 2.58324e-03, 2.85516e-03}, 
    {2.853078e-03, 3.15993e-03, 3.466782e-03, 1.849056e-03, 2.066592e-03, 2.284128e-03},
    {2.02093025e-03, 2.23828375e-03, 2.45563725e-03, 1.309748e-03, 1.463836e-03, 1.617924e-03}
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
   {4.79528662499999e-06, -1.88383333333332e-06, 2.214555375e-06, 31.3965130208333e-06, 20.3369817708333e-06, 31.3297213541666e-06, 12.3149462031250e-06, 14.0145523906250e-06, 15.470035828125e-06, 12.314944421875e-06, 14.014556984375e-06, 15.470042296875e-06,
    -98.0928725416665e-06, -2.58076875e-06, 65.5947524583332e-06, 31.3965130208333e-06, 20.3369817708333e-06, 31.3297213541666e-06, 12.314946203125e-06, 14.014552390625e-06, 15.470035828125e-06, 12.314944421875e-06, 14.014556984375e-06, 15.470042296875e-06,
    97.7790050416665e-06, 6.67917170833331e-06, 2.2669005e-06, 31.3965130208333e-06, 20.3369817708333e-06, 31.3297213541666e-06, 12.314946203125e-06, 14.014552390625e-06, 15.470035828125e-06, 12.314944421875e-06, 14.014556984375e-06, 15.470042296875e-06,
    -4.48141912499999e-06, -2.214569625e-06, -70.0762083333332e-06, 31.3965130208333e-06, 20.3369817708333e-06, 31.3297213541666e-06, 12.3149462031250e-06, 14.0145523906250e-06, 15.470035828125e-06, 12.314944421875e-06, 14.014556984375e-06, 15.470042296875e-06},
   {2.4412545e-06, -43.4948333333332e-06, 2.5807125e-06, 12.9048645833333e-06, 11.0211145833333e-06, 52.0812604166666e-06, 9.95309843749999e-06, 11.042504875e-06, 12.2016568125e-06, 9.95308906249998e-06, 11.042495125e-06, 12.2016556875000e-06,
    -209.2155e-09, 46.0756208333332e-06, 269.620120833333e-06, 12.9048645833333e-06, 11.0211145833333e-06, 52.0812604166666e-06, 9.95309843749999e-06, 11.042504875e-06, 12.2016568125e-06, 9.95308906249998e-06, 11.042495125e-06, 12.2016556875000e-06,
    -17.8713723333333e-06, -139.542e-09, -269.968872333333e-06, 12.9048645833333e-06, 11.0211145833333e-06, 52.0812604166666e-06, 9.95309843749999e-06, 11.042504875e-06, 12.2016568125e-06, 9.95308906249998e-06, 11.042495125e-06, 12.2016556875000e-06,
    15.6393333333333e-06, -2.4412455e-06, -2.231961e-06, 12.9048645833333e-06, 11.0211145833333e-06, 52.0812604166666e-06, 9.95309843749999e-06, 11.042504875e-06, 12.2016568125e-06, 9.95308906249998e-06, 11.042495125e-06, 12.2016556875000e-06}
  };

  for(int iCell=0; iCell<int(tResidual_gold.size()); iCell++){
    for(int iDof=0; iDof<tNumDofsPerCell; iDof++){
      if(tResidual_gold[iCell][iDof] == 0.0){
        TEST_ASSERT(fabs(tResidual_Host(iCell,iDof)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tResidual_Host(iCell,iDof), tResidual_gold[iCell][iDof], 1e-13);
      }
    }
  }
  
  // test symmetric gradient micro inertia
  //
  auto tSymGradientMicroInertia_Host = Kokkos::create_mirror_view( tSymGradientMicroInertia );
  Kokkos::deep_copy( tSymGradientMicroInertia_Host, tSymGradientMicroInertia );

  std::vector<std::vector<Plato::Scalar>> tSymGradientMicroInertia_Gold = { 
    {2.0e-05,  3.2e-04,  2.4e-04, 6.4e-04, 1.4e-04, 2.0e-04}, 
    {2.0e-04,  3.2e-04, -3.0e-04, 2.8e-04, 5.0e-04, 5.6e-04},
    {2.0e-04,  8.0e-05,  6.0e-05, 1.6e-04, 6.2e-04, 4.4e-04},
    {4.0e-04, -3.2e-04,  6.0e-05, -4.4e-04, 1.22e-03, 6.4e-04}
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
    {-3.2e-04,  2.0e-05,  1.2e-04}, 
    {-6.8e-04,  -7.0e-04, -2.4e-04},
    {-8.0e-05,  -5.8e-04,  -3.6e-04},
    {5.2e-04, -1.18e-03, -9.6e-04}
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
    {3.0e-04,  3.75e-04,  4.5e-04, 1.275e-03, 1.425e-03, 1.575e-03}, 
    {2.4e-04,  3.0e-04, 3.6e-04, 1.02e-03, 1.14e-03, 1.26e-03},
    {1.7e-04,  2.125e-04, 2.55e-04, 7.225e-04, 8.075e-04, 8.925e-04},
    {3.5e-04, 4.375e-04,  5.25e-04, 1.4875e-03, 1.6625e-03, 1.8375e-03}
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
    {-2.25e-04,  -2.25e-04,  -2.25e-04}, 
    {-1.8e-04,  -1.8e-04, -1.8e-04},
    {-1.275e-04,  -1.275e-04, -1.275e-04},
    {-2.625e-04, -2.625e-04, -2.625e-04}
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
    {1.184e-03, 1.544e-03, 1.448e-03, 128.0e-06, 28.0e-06, 40.0e-06}, 
    {680.0e-06, 824.0e-06, 80.0000000000002e-06, 56.0e-06, 100.0e-06, 112.0e-06},
    {920.0e-06, 776.0e-06, 752.0e-06, 32.0e-06, 124.0e-06, 88.0e-06}
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
    {0.0, 0.0, 0.0, -32.0e-09,  2.0e-09, 12.0e-09}, 
    {0.0, 0.0, 0.0, -68.0e-09, -70.0e-09, -24.0e-09},
    {0.0, 0.0, 0.0, -8.0e-09,  -58.0e-09, -36.0e-09}
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
    {-645.0e-03, -300.0e-03, 45.0000000000001e-03, 5.7375e+00, 6.4125e+00, 7.0875e+00}, 
    {-516.0e-03, -240.0e-03, 36.0000000000001e-03, 4.59e+00, 5.13e+00, 5.67e+00},
    {-365.5e-03, -170.0e-03, 25.5e-03, 3.25125e+00, 3.63375e+00, 4.01625e+00}
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
    {0.0, 0.0, 0.0, -22.5e-09, -22.5e-09, -22.5e-09}, 
    {0.0, 0.0, 0.0, -18.0e-09, -18.0e-09, -18.0e-09},
    {0.0, 0.0, 0.0, -12.75e-09, -12.75e-09, -12.75e-09}
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
   {565.442208333332e-06, 1.06988541666667e-03, 1.69599345833333e-03, -3.35937499999999e-03, -1.5625e-03, 234.375e-06, 29.8828123828125e-03, 33.3984373828125e-03, 36.9140623828124e-03, 29.8828126171875e-03, 33.3984376171875e-03, 36.9140626171874e-03,
    615.275958333332e-06, 1.13055291666667e-03, 1.642161375e-03, -3.35937499999999e-03, -1.5625e-03, 234.375e-06, 29.8828123828125e-03, 33.3984373828125e-03, 36.9140623828124e-03, 29.8828126171875e-03, 33.3984376171875e-03, 36.9140626171874e-03,
    519.443208333333e-06, 1.19688591666666e-03, 1.70549620833333e-03, -3.35937499999999e-03, -1.5625e-03, 234.375e-06, 29.8828123828125e-03, 33.3984373828125e-03, 36.9140623828124e-03, 29.8828126171875e-03, 33.3984376171875e-03, 36.9140626171874e-03,
    568.276124999999e-06, 1.13955075e-03, 1.76166145833333e-03, -3.35937499999999e-03, -1.5625e-03, 234.375e-06, 29.8828123828125e-03, 33.3984373828125e-03, 36.9140623828124e-03, 29.8828126171875e-03, 33.3984376171875e-03, 36.9140626171874e-03},
   {449.021833333333e-06, 873.041666666665e-06, 1.35872633333333e-03, -2.6875e-03, -1.25e-03, 187.5e-06, 23.90624990625e-03, 26.71874990625e-03, 29.53124990625e-03, 23.90625009375e-03, 26.71875009375e-03, 29.53125009375e-03,
    454.189416666666e-06, 939.377833333332e-06, 1.36006533333333e-03, -2.6875e-03, -1.25e-03, 187.5e-06, 23.90624990625e-03, 26.71874990625e-03, 29.53124990625e-03, 23.90625009375e-03, 26.71875009375e-03, 29.53125009375e-03,
    429.517916666666e-06, 905.037833333332e-06, 1.36022625e-03, -2.6875e-03, -1.25e-03, 187.5e-06, 23.90624990625e-03, 26.71874990625e-03, 29.53124990625e-03, 23.90625009375e-03, 26.71875009375e-03, 29.53125009375e-03,
    482.020833333333e-06, 912.042666666665e-06, 1.36523208333333e-03, -2.6875e-03, -1.25e-03, 187.5e-06, 23.90624990625e-03, 26.71874990625e-03, 29.53124990625e-03, 23.90625009375e-03, 26.71875009375e-03, 29.53125009375e-03}
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
    constexpr int tSpaceDim=3;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    int tNumCells = tMesh->nelems();

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
    
    // get mesh sets
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Plato::ProblemFactory<3> tProblemFactory;
    TEST_THROW(tProblemFactory.create(*tMesh, tMeshSets, *tParams, tMachine), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, ErrorAFormFalse)
{
    // create test mesh
    //
    constexpr int tMeshWidth=2;
    constexpr int tSpaceDim=3;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    int tNumCells = tMesh->nelems();

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
    
    // get mesh sets
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Plato::ProblemFactory<3> tProblemFactory;
    TEST_THROW(tProblemFactory.create(*tMesh, tMeshSets, *tParams, tMachine), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, ErrorExplicitNotSpecified)
{
    // create test mesh
    //
    constexpr int tMeshWidth=2;
    constexpr int tSpaceDim=3;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    int tNumCells = tMesh->nelems();

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
    
    // get mesh sets
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Plato::ProblemFactory<3> tProblemFactory;
    TEST_THROW(tProblemFactory.create(*tMesh, tMeshSets, *tParams, tMachine), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicTest, CreateProblem)
{
    // create test mesh
    //
    constexpr int tMeshWidth=2;
    constexpr int tSpaceDim=3;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    int tNumCells = tMesh->nelems();

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
    
    // get mesh sets
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Plato::ProblemFactory<3> tProblemFactory;
    TEST_NOTHROW(tProblemFactory.create(*tMesh, tMeshSets, *tParams, tMachine));
}

}
// namespace RelaxedMicromorphicTest


