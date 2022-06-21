/*
 * PlatoAugLagStressTest.cpp
 *
 *  Created on: Feb 3, 2019
 */

#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoTestHelpers.hpp"

#include "Tet4.hpp"
#include "Tri3.hpp"
#include "MechanicsElement.hpp"
#include "ThermomechanicsElement.hpp"
#include "Solutions.hpp"
#include "Analyze_Diagnostics.hpp"
#include "IsotropicLinearElasticMaterial.hpp"
#include "ThermalMassMaterial.hpp"
#include "Eigenvalues.hpp"
#include "Plato_AugLagStressCriterion.hpp"
#include "Plato_AugLagStressCriterionGeneral.hpp"
#include "Plato_AugLagStressCriterionQuadratic.hpp"
#include "VonMisesLocalMeasure.hpp"
#include "TensileEnergyDensityLocalMeasure.hpp"
#include "ThermalVonMisesLocalMeasure.hpp"
#include "elliptic/WeightedSumFunction.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
#include "elliptic/MassMoment.hpp"

namespace AugLagStressTest
{
    Teuchos::RCP<Teuchos::ParameterList> tGenericParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                               \n"
        "<ParameterList name='Spatial Model'>                                             \n"
          "<ParameterList name='Domains'>                                                 \n"
            "<ParameterList name='Design Volume'>                                         \n"
              "<Parameter name='Element Block' type='string' value='body'/>               \n"
              "<Parameter name='Material Model' type='string' value='Playdoh'/>           \n"
            "</ParameterList>                                                             \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
        "<ParameterList name='Material Models'>                                           \n"
          "<ParameterList name='Playdoh'>                                                 \n"
            "<ParameterList name='Isotropic Linear Elastic'>                              \n"
              "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>             \n"
              "<Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>           \n"
            "</ParameterList>                                                             \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
    "</ParameterList>                                                                     \n"
    );


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_Eigenvalue1D)
{
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 1;
    constexpr Plato::OrdinalType tNumVoigtTerms = 1;
    Plato::Eigenvalues<tSpaceDim, tNumVoigtTerms> tComputeEigenvalues;
    Plato::ScalarMultiVector tCauchyStrain("strain", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStrain = Kokkos::create_mirror(tCauchyStrain);
    tHostCauchyStrain(0, 0) = 3.0;
    tHostCauchyStrain(1, 0) = 0.5;
    tHostCauchyStrain(2, 0) = -1.0;
    Kokkos::deep_copy(tCauchyStrain, tHostCauchyStrain);

    Plato::ScalarMultiVector tPrincipalStrains("principal strains", tNumCells, tSpaceDim);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        Plato::Array<tNumVoigtTerms> tStrain;
        tStrain(0) = tCauchyStrain(tCellOrdinal, 0);

        Plato::Array<tSpaceDim> tPrincipalStrain;

        tComputeEigenvalues(tStrain, tPrincipalStrain, true);

        tPrincipalStrains(tCellOrdinal, 0) = tPrincipalStrain(0);

    }, "Test Computing Eigenvalues");

    constexpr Plato::Scalar tTolerance = 1e-8;
    std::vector<Plato::Scalar> tGold1 = {3.0};
    std::vector<Plato::Scalar> tGold2 = {0.5};
    std::vector<Plato::Scalar> tGold3 = {-1.0};
    auto tHostPrincipalStrains = Kokkos::create_mirror(tPrincipalStrains);
    Kokkos::deep_copy(tHostPrincipalStrains, tPrincipalStrains);

    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(0, tIndex), tGold1[tIndex], tTolerance);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(1, tIndex), tGold2[tIndex], tTolerance);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(2, tIndex), tGold3[tIndex], tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_Eigenvalue2D)
{
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;
    Plato::Eigenvalues<tSpaceDim, tNumVoigtTerms> tComputeEigenvalues;
    Plato::ScalarMultiVector tCauchyStrain("strain", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStrain = Kokkos::create_mirror(tCauchyStrain);
    tHostCauchyStrain(0, 0) = 3.0; tHostCauchyStrain(0, 1) = 2.0; tHostCauchyStrain(0, 2) = 0.0;
    tHostCauchyStrain(1, 0) = 0.5; tHostCauchyStrain(1, 1) = 0.2; tHostCauchyStrain(1, 2) = 1.6;
    tHostCauchyStrain(2, 0) = 0.0; tHostCauchyStrain(2, 1) = 0.0; tHostCauchyStrain(2, 2) = 1.6;
    Kokkos::deep_copy(tCauchyStrain, tHostCauchyStrain);

    Plato::ScalarMultiVector tPrincipalStrains("principal strains", tNumCells, tSpaceDim);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        Plato::Array<tNumVoigtTerms> tStrain;
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
        {
            tStrain(iVoigt) = tCauchyStrain(tCellOrdinal, iVoigt);
        }

        Plato::Array<tSpaceDim> tPrincipalStrain;

        tComputeEigenvalues(tStrain, tPrincipalStrain, true);

        for(int iDim=0; iDim<tSpaceDim; iDim++)
        {
            tPrincipalStrains(tCellOrdinal, iDim) = tPrincipalStrain(iDim);
        }
    }, "Test Computing Eigenvalues");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold1 = {3.0, 2.0};
    std::vector<Plato::Scalar> tGold2 = {1.16394103, -0.46394103};
    std::vector<Plato::Scalar> tGold3 = {-0.8, 0.8};
    auto tHostPrincipalStrains = Kokkos::create_mirror(tPrincipalStrains);
    Kokkos::deep_copy(tHostPrincipalStrains, tPrincipalStrains);

    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(0, tIndex), tGold1[tIndex], tTolerance);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(1, tIndex), tGold2[tIndex], tTolerance);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(2, tIndex), tGold3[tIndex], tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_Eigenvalue3D)
{
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    Plato::Eigenvalues<tSpaceDim, tNumVoigtTerms> tComputeEigenvalues;
    Plato::ScalarMultiVector tCauchyStrain("strain", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStrain = Kokkos::create_mirror(tCauchyStrain);
    tHostCauchyStrain(0, 0) = 3.0; tHostCauchyStrain(0, 1) = 2.0; tHostCauchyStrain(0, 2) = 1.0;
    tHostCauchyStrain(0, 3) = 0.0; tHostCauchyStrain(0, 4) = 0.0; tHostCauchyStrain(0, 5) = 0.0;
    tHostCauchyStrain(1, 0) = 0.5; tHostCauchyStrain(1, 1) = 0.2; tHostCauchyStrain(1, 2) = 0.8;
    tHostCauchyStrain(1, 3) = 1.1; tHostCauchyStrain(1, 4) = 1.5; tHostCauchyStrain(1, 5) = 0.3;
    tHostCauchyStrain(2, 0) = 1.64913808; tHostCauchyStrain(2, 1) = 0.61759347; tHostCauchyStrain(2, 2) = 0.33326845;
    tHostCauchyStrain(2, 3) = 0.65938917; tHostCauchyStrain(2, 4) = -0.1840644; tHostCauchyStrain(2, 5) = 1.55789418;
    Kokkos::deep_copy(tCauchyStrain, tHostCauchyStrain);

    Plato::ScalarMultiVector tPrincipalStrains("principal strains", tNumCells, tSpaceDim);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        Plato::Array<tNumVoigtTerms> tStrain;
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
        {
            tStrain(iVoigt) = tCauchyStrain(tCellOrdinal, iVoigt);
        }

        Plato::Array<tSpaceDim> tPrincipalStrain;

        tComputeEigenvalues(tStrain, tPrincipalStrain, true);

        for(int iDim=0; iDim<tSpaceDim; iDim++)
        {
            tPrincipalStrains(tCellOrdinal, iDim) = tPrincipalStrain(iDim);
        }
    }, "Test Computing Eigenvalues");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold1 = {3.0, 2.0, 1.0};
    std::vector<Plato::Scalar> tGold2 = {-0.28251642,  0.17136166,  1.61115476};
    std::vector<Plato::Scalar> tGold3 = {2.07094021, -0.07551018, 0.60456996};
    auto tHostPrincipalStrains = Kokkos::create_mirror(tPrincipalStrains);
    Kokkos::deep_copy(tHostPrincipalStrains, tPrincipalStrains);

    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(0, tIndex), tGold1[tIndex], tTolerance);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(1, tIndex), tGold2[tIndex], tTolerance);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(2, tIndex), tGold3[tIndex], tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagQuadratic_EvaluateVonMises)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tet4>>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::blas1::fill(0.1, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            {   tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal);}, "fill state");
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result/output workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                               \n"
        "<ParameterList name='Spatial Model'>                                             \n"
          "<ParameterList name='Domains'>                                                 \n"
            "<ParameterList name='Design Volume'>                                         \n"
              "<Parameter name='Element Block' type='string' value='body'/>               \n"
              "<Parameter name='Material Model' type='string' value='Playdoh'/>           \n"
            "</ParameterList>                                                             \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
        "<ParameterList name='Material Models'>                                           \n"
          "<ParameterList name='Playdoh'>                                                 \n"
            "<ParameterList name='Isotropic Linear Elastic'>                              \n"
              "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>             \n"
              "<Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>           \n"
            "</ParameterList>                                                             \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
    "</ParameterList>                                                                     \n"
    );

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::AugLagStressCriterionQuadratic<Residual> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    Plato::ScalarVector tLagrangeMultipliers("Lagrange Multiplier", tNumCells);
    Plato::blas1::fill(0.1, tLagrangeMultipliers);
    tCriterion.setLagrangeMultipliers(tLagrangeMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    auto tCellStiffMatrix = tMatModel.getStiffnessMatrix();

    const auto tLocalMeasure = std::make_shared<Plato::VonMisesLocalMeasure<Residual>>(tOnlyDomain, tCellStiffMatrix, "VonMises");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure);

    tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.01322740, 0.01322740, 0.01322740, 0.01322740, 0.01322740, 0.01322740};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }

    // ****** TEST GLOBAL SUM ******
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(0.079364, tObjFuncVal, tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagQuadratic_EvaluateTensileEnergyDensity2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TRI3", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

    using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::blas1::fill(0.1, tState);
    //Plato::fill(0.0, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal) * 2; }, "fill state");
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result/output workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                               \n"
        "<ParameterList name='Spatial Model'>                                             \n"
          "<ParameterList name='Domains'>                                                 \n"
            "<ParameterList name='Design Volume'>                                         \n"
              "<Parameter name='Element Block' type='string' value='body'/>               \n"
              "<Parameter name='Material Model' type='string' value='Playdoh'/>           \n"
            "</ParameterList>                                                             \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
        "<ParameterList name='Material Models'>                                           \n"
          "<ParameterList name='Playdoh'>                                                 \n"
            "<ParameterList name='Isotropic Linear Elastic'>                              \n"
              "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>             \n"
              "<Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>           \n"
            "</ParameterList>                                                             \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
    "</ParameterList>                                                                     \n"
    );

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::AugLagStressCriterionQuadratic<Residual> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    Plato::ScalarVector tLagrangeMultipliers("lagrange multipliers", tNumCells);
    Plato::blas1::fill(0.1, tLagrangeMultipliers);
    tCriterion.setLagrangeMultipliers(tLagrangeMultipliers);
    tCriterion.setAugLagPenalty(1.5);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;

    const auto tLocalMeasure = std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                 (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure);
    tCriterion.setLocalMeasureValueLimit(0.15);

    tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {387.53864555, 387.53864555};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }

    // ****** TEST GLOBAL SUM ******
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(775.0772911, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagQuadratic_EvalTensileEnergyScalarFuncBase2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TRI3", tMeshWidth);

    using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->NumElements();

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControl);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarMultiVector tStates("States", /*numStates=*/ 1, tNumDofs);
    Kokkos::deep_copy(tStates, 0.1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            { tStates(0, aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal) * 2; }, "fill state");

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                               \n"
        "<ParameterList name='Spatial Model'>                                             \n"
          "<ParameterList name='Domains'>                                                 \n"
            "<ParameterList name='Design Volume'>                                         \n"
              "<Parameter name='Element Block' type='string' value='body'/>               \n"
              "<Parameter name='Material Model' type='string' value='Playdoh'/>           \n"
            "</ParameterList>                                                             \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
        "<ParameterList name='Material Models'>                                           \n"
          "<ParameterList name='Playdoh'>                                                 \n"
            "<ParameterList name='Isotropic Linear Elastic'>                              \n"
              "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>             \n"
              "<Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>           \n"
            "</ParameterList>                                                             \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
    "</ParameterList>                                                                     \n"
    );

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::Elliptic::WeightedSumFunction<Plato::Mechanics<Plato::Tri3>> tWeightedSum(tSpatialModel, tDataMap);

    const auto tCriterion = std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    Plato::ScalarVector tLagrangeMultipliers("lagrange multipliers", tNumCells);
    Plato::blas1::fill(0.1, tLagrangeMultipliers);
    tCriterion->setLagrangeMultipliers(tLagrangeMultipliers);
    tCriterion->setAugLagPenalty(1.5);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;

    const auto tLocalMeasure = std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                 (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion->setLocalMeasure(tLocalMeasure, tLocalMeasure);
    tCriterion->setLocalMeasureValueLimit(0.15);

    const auto tPhysicsScalarFunc =
          std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tri3>>>(tSpatialModel, tDataMap);

    tPhysicsScalarFunc->setEvaluator(tCriterion, tOnlyDomain.getDomainName());
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFunc);
    tWeightedSum.appendFunctionWeight(1.0);

    Plato::Solutions tSolution;
    tSolution.set("State", tStates);
    auto tObjFuncVal = tWeightedSum.value(tSolution, tControl, 0.0);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(775.0772911, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagQuadratic_CheckThermalVonMises3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::ThermomechanicsElement<Plato::Tet4>;
    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;

    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;

    constexpr Plato::OrdinalType tDofsPerNode = ElementType::mNumDofsPerNode;
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;

    const Plato::OrdinalType tNumCells = tMesh->NumElements();

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create state workset
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    const Plato::OrdinalType tNumDofs  = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::blas1::fill(0.0, tState);
    Plato::OrdinalType tDofStride = tDofsPerNode;
    Plato::OrdinalType tDofToSet = ElementType::mTDofOffset;
    Plato::Scalar tTemperature = 11.0;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs / tDofStride),
                         LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeIndex)
    {
        Plato::OrdinalType tIndex = tDofStride * aNodeIndex + tDofToSet;
        tState(tIndex) = tTemperature;
    }
    , "fill specific vector entry globally");
    PlatoUtestHelpers::set_dof_value_in_vector_on_boundary_3D(tMesh, "x+", tState, tDofStride, 0, 0.1 - 0.05);
    PlatoUtestHelpers::set_dof_value_in_vector_on_boundary_3D(tMesh, "y+", tState, tDofStride, 1, 0.1 - 0.10);
    PlatoUtestHelpers::set_dof_value_in_vector_on_boundary_3D(tMesh, "z+", tState, tDofStride, 2, 0.1 + 0.05);
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Playdoh'/>      \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Playdoh'>                                            \n"
    "      <ParameterList name='Thermoelastic'>                                    \n"
    "        <ParameterList name='Elastic Stiffness'>                              \n"
    "          <Parameter  name='Poissons Ratio' type='double' value='0.0'/>       \n"
    "          <Parameter  name='Youngs Modulus' type='double' value='100.0'/>     \n"
    "        </ParameterList>                                                      \n"
    "        <Parameter  name='Thermal Expansivity' type='double' value='1.0e-1'/> \n"
    "        <Parameter  name='Thermal Conductivity' type='double' value='910.0'/> \n"
    "        <Parameter  name='Reference Temperature' type='double' value='1.0'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
    );

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    const std::string tName = "ThermalVonMises";
    Plato::ThermalVonMisesLocalMeasure<Residual> tLocalMeasure(tOnlyDomain, *tParamList, tName);

    Plato::ScalarVector tResult("ThermalVonMises", tNumCells);
    tLocalMeasure(tStateWS, tConfigWS, tResult);

    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    constexpr Plato::Scalar tTolerance = 1e-6;
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        //printf("Cell %d , TVM %f\n", tIndex, tHostResult(tIndex));
        TEST_FLOATING_EQUALITY(13.2287565, tHostResult(tIndex), tTolerance);
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagQuadratic_FiniteDiff_TensileEnergyScalarFuncBaseGradZ_3D)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    Plato::Elliptic::WeightedSumFunction<Plato::Mechanics<Plato::Tet4>> tWeightedSum(tSpatialModel, tDataMap);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using Residual  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;
    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<Residual>> tCriterionResidual =
          std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(tOnlyDomain, tDataMap);
    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<GradientZ>> tCriterionGradZ =
          std::make_shared<Plato::AugLagStressCriterionQuadratic<GradientZ>>(tOnlyDomain, tDataMap);
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const auto tLocalMeasureGradZ =
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<GradientZ>>
                                              (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const auto tLocalMeasurePODType =
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    tCriterionGradZ->setLocalMeasure(tLocalMeasureGradZ, tLocalMeasurePODType);

    const auto tPhysicsScalarFunc =
          std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tet4>>>(tSpatialModel, tDataMap);

    tPhysicsScalarFunc->setEvaluator(tCriterionResidual, tOnlyDomain.getDomainName());
    tPhysicsScalarFunc->setEvaluator(tCriterionGradZ, tOnlyDomain.getDomainName());

    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFunc);
    tWeightedSum.appendFunctionWeight(1.0);

    Plato::test_partial_control<GradientZ, ElementType>(tMesh, tWeightedSum);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagQuadratic_FiniteDiff_TensileEnergyScalarFuncBaseGradU_2D)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TRI3", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);
    Plato::Elliptic::WeightedSumFunction<Plato::Mechanics<Plato::Tri3>> tWeightedSum(tSpatialModel, tDataMap);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using Jacobian  = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    using Residual  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<Residual>> tCriterionResidual =
          std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(tOnlyDomain, tDataMap);
    const std::shared_ptr<Plato::AugLagStressCriterionQuadratic<Jacobian>> tCriterionGradU =
          std::make_shared<Plato::AugLagStressCriterionQuadratic<Jacobian>>(tOnlyDomain, tDataMap);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const auto tLocalMeasureEvaluationType =
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Jacobian>>
                                              (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const auto tLocalMeasurePODType =
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    tCriterionGradU->setLocalMeasure(tLocalMeasureEvaluationType, tLocalMeasurePODType);

    const auto tPhysicsScalarFunc =
          std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tri3>>>(tSpatialModel, tDataMap);

    tPhysicsScalarFunc->setEvaluator(tCriterionResidual, tOnlyDomain.getDomainName());
    tPhysicsScalarFunc->setEvaluator(tCriterionGradU, tOnlyDomain.getDomainName());

    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFunc);
    tWeightedSum.appendFunctionWeight(1.0);

    Plato::test_partial_state<Jacobian, ElementType>(tMesh, tWeightedSum);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagQuadratic_FiniteDiff_TensileEnergyScalarFuncBaseGradU_3D)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);
    Plato::Elliptic::WeightedSumFunction<Plato::Mechanics<Plato::Tet4>> tWeightedSum(tSpatialModel, tDataMap);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using Jacobian = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    const auto tCriterionResidual = std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(tOnlyDomain, tDataMap);
    const auto tCriterionGradU = std::make_shared<Plato::AugLagStressCriterionQuadratic<Jacobian>>(tOnlyDomain, tDataMap);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const auto tLocalMeasureEvaluationType =
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Jacobian>>
                                              (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const auto tLocalMeasurePODType =
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    tCriterionGradU->setLocalMeasure(tLocalMeasureEvaluationType, tLocalMeasurePODType);

    const auto tPhysicsScalarFunc =
          std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tet4>>>(tSpatialModel, tDataMap);

    tPhysicsScalarFunc->setEvaluator(tCriterionResidual, tOnlyDomain.getDomainName());
    tPhysicsScalarFunc->setEvaluator(tCriterionGradU, tOnlyDomain.getDomainName());

    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFunc);
    tWeightedSum.appendFunctionWeight(1.0);

    Plato::test_partial_state<Jacobian, ElementType>(tMesh, tWeightedSum);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagQuadratic_EvaluateTensileEnergyDensity3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::blas1::fill(0.1, tState);
    //Plato::fill(0.0, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); }, "fill state");
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result/output workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::AugLagStressCriterionQuadratic<Residual> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    Plato::ScalarVector tLagrangeMultipliers("lagrange multipliers", tNumCells);
    Plato::blas1::fill(0.1, tLagrangeMultipliers);
    tCriterion.setLagrangeMultipliers(tLagrangeMultipliers);
    tCriterion.setAugLagPenalty(1.5);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;

    const auto tLocalMeasure =
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure);

    tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {
      3.024956014559, 3.024956014559, 3.024956014559,
      3.024956014559, 3.024956014559, 3.024956014559};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }

    // ****** TEST GLOBAL SUM ******
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(18.14973609, tObjFuncVal, tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagQuadratic_FiniteDiff_TensileEnergyCriterionGradZ_2D)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TRI3", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;
    Plato::AugLagStressCriterionQuadratic<GradientZ> tCriterion(tOnlyDomain, tDataMap);
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const auto tLocalMeasureEvaluationType =
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<GradientZ>>
                                              (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const auto tLocalMeasurePODType =
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasureEvaluationType, tLocalMeasurePODType);

    Plato::test_partial_control<GradientZ, ElementType>(tMesh, tCriterion);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagQuadratic_FiniteDiff_TensileEnergyCriterionGradZ_3D)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;
    Plato::AugLagStressCriterionQuadratic<GradientZ> tCriterion(tOnlyDomain, tDataMap);
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const auto  tLocalMeasureEvaluationType =
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<GradientZ>>
                                              (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const auto tLocalMeasurePODType =
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasureEvaluationType, tLocalMeasurePODType);

    Plato::test_partial_control<GradientZ, ElementType>(tMesh, tCriterion);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagQuadratic_FiniteDiff_TensileEnergyCriterionGradU_2D)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TRI3", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using Jacobian = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    Plato::AugLagStressCriterionQuadratic<Jacobian> tCriterion(tOnlyDomain, tDataMap);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const auto tLocalMeasureEvaluationType =
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Jacobian>>
                                              (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const auto tLocalMeasurePODType =
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasureEvaluationType, tLocalMeasurePODType);

    Plato::test_partial_state<Jacobian, ElementType>(tMesh, tCriterion);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagQuadratic_FiniteDiff_TensileEnergyCriterionGradU_3D)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using Jacobian = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    Plato::AugLagStressCriterionQuadratic<Jacobian> tCriterion(tOnlyDomain, tDataMap);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const auto tLocalMeasureEvaluationType =
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Jacobian>>
                                              (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const auto tLocalMeasurePODType =
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasureEvaluationType, tLocalMeasurePODType);

    Plato::test_partial_state<Jacobian, ElementType>(tMesh, tCriterion);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagQuadratic_UpdateMultipliers1)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    Plato::AugLagStressCriterionQuadratic<Residual> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->NumElements();

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();

    const auto tLocalMeasure = std::make_shared<Plato::VonMisesLocalMeasure<Residual>>(tOnlyDomain, tCellStiffMatrix, "VonMises");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = ElementType::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("controls", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("State", tNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0) = 0.4360751; tHostState(1) = 0.2577532;  tHostState(2) = 0.4132397;
    tHostState(3) = 0.4193760; tHostState(4) = 0.4646589;  tHostState(5) = 0.1790205;
    tHostState(6) = 0.2340891; tHostState(7) = 0.4072918;  tHostState(8) = 0.2111099;
    tHostState(9) = 0.3215880; tHostState(10) = 0.2909588;  tHostState(11) = 0.3515484;
    tHostState(12) = 0.2459138; tHostState(13) = 0.3053604;  tHostState(14) = 0.4808919;
    tHostState(15) = 0.4664780; tHostState(16) = 0.3542847;  tHostState(17) = 0.3869188;
    tHostState(18) = 0.1566410; tHostState(19) = 0.3427876;  tHostState(20) = 0.1065202;
    tHostState(21) = 0.1971547; tHostState(22) = 0.1548926;  tHostState(23) = 0.4216707;
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // TEST UPDATE PROBLEM FUNCTION
    tCriterion.updateProblem(tStateWS, tControlWS, tConfigWS);
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers(tNumCells, 0.01);
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagQuadratic_UpdateMultipliers2)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    Plato::AugLagStressCriterionQuadratic<Residual> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->NumElements();

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();

    const auto tLocalMeasure = std::make_shared<Plato::VonMisesLocalMeasure<Residual>>(tOnlyDomain, tCellStiffMatrix, "VonMises");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = ElementType::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("controls", tNumVerts);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("State", tNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0) = 4.360751; tHostState(1) = 2.577532;  tHostState(2) = 4.132397;
    tHostState(3) = 4.193760; tHostState(4) = 4.646589;  tHostState(5) = 1.790205;
    tHostState(6) = 2.340891; tHostState(7) = 4.072918;  tHostState(8) = 2.111099;
    tHostState(9) = 3.215880; tHostState(10) = 2.909588;  tHostState(11) = 3.515484;
    tHostState(12) = 2.459138; tHostState(13) = 3.053604;  tHostState(14) = 4.808919;
    tHostState(15) = 4.664780; tHostState(16) = 3.542847;  tHostState(17) = 3.869188;
    tHostState(18) = 1.566410; tHostState(19) = 3.427876;  tHostState(20) = 1.065202;
    tHostState(21) = 1.971547; tHostState(22) = 1.548926;  tHostState(23) = 4.216707;
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // TEST UPDATE PROBLEM FUNCTION
    tCriterion.updateProblem(tStateWS, tControlWS, tConfigWS);
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers =
      {0.131739476, 0.112932370, 0.064022544, 0.111203340, 0.042183347, 0.219975864};

    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_VonMises3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;

    Plato::ScalarMultiVector tCellCauchyStress("Cauchy Stress", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStress = Kokkos::create_mirror(tCellCauchyStress);
    tHostCauchyStress(0, 0) = 1.096154;
    tHostCauchyStress(1, 0) = 1.557692;
    tHostCauchyStress(0, 1) = 1.557692;
    tHostCauchyStress(1, 1) = 1.557692;
    tHostCauchyStress(0, 2) = 1.096154;
    tHostCauchyStress(1, 2) = 0.634615;
    tHostCauchyStress(0, 3) = 0.461538;
    tHostCauchyStress(1, 3) = 0.230769;
    tHostCauchyStress(0, 4) = 0.230769;
    tHostCauchyStress(1, 4) = 0.230769;
    tHostCauchyStress(0, 5) = 0.461538;
    tHostCauchyStress(1, 5) = 0.692308;
    Kokkos::deep_copy(tCellCauchyStress, tHostCauchyStress);

    Plato::ScalarVector tCellVonMises("Von Mises Stress", tNumCells);

    Plato::VonMisesYieldFunction<tSpaceDim, tNumVoigtTerms> tComputeVonMises;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
    {
        Plato::Array<tNumVoigtTerms> tStress;
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
        {
            tStress(iVoigt) = tCellCauchyStress(tCellOrdinal, iVoigt);
        }

        Plato::Scalar tVonMises(0.0);
               
        tComputeVonMises(tCellOrdinal, tStress, tVonMises);

        tCellVonMises(tCellOrdinal) = tVonMises;
    }, "Test Von Mises Yield Stress Calculation");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {1.284867, 1.615385};
    auto tHostCellVonMises = Kokkos::create_mirror(tCellVonMises);
    Kokkos::deep_copy(tHostCellVonMises, tCellVonMises);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostCellVonMises(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_VonMises2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;

    Plato::ScalarMultiVector tCellCauchyStress("Cauchy Stress", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStress = Kokkos::create_mirror(tCellCauchyStress);
    tHostCauchyStress(0, 0) = 1.096154;
    tHostCauchyStress(1, 0) = 1.457692;
    tHostCauchyStress(0, 1) = 1.557692;
    tHostCauchyStress(1, 1) = 1.557692;
    tHostCauchyStress(0, 2) = 1.096154;
    tHostCauchyStress(1, 2) = 0.634615;
    Kokkos::deep_copy(tCellCauchyStress, tHostCauchyStress);

    Plato::ScalarVector tCellVonMises("Von Mises Stress", tNumCells);

    Plato::VonMisesYieldFunction<tSpaceDim, tNumVoigtTerms> tComputeVonMises;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
    {
        Plato::Array<tNumVoigtTerms> tStress;
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
        {
            tStress(iVoigt) = tCellCauchyStress(tCellOrdinal, iVoigt);
        }

        Plato::Scalar tVonMises(0.0);
               
        tComputeVonMises(tCellOrdinal, tStress, tVonMises);

        tCellVonMises(tCellOrdinal) = tVonMises;
    }, "Test Von Mises Yield Stress Calculation");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {2.350563425, 1.867844683};
    auto tHostCellVonMises = Kokkos::create_mirror(tCellVonMises);
    Kokkos::deep_copy(tHostCellVonMises, tCellVonMises);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostCellVonMises(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_VonMises1D)
{
    constexpr Plato::OrdinalType tSpaceDim = 1;
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 1;

    Plato::ScalarMultiVector tCellCauchyStress("Cauchy Stress", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStress = Kokkos::create_mirror(tCellCauchyStress);
    tHostCauchyStress(0, 0) = 1.096154;
    tHostCauchyStress(1, 0) = 1.457692;
    Kokkos::deep_copy(tCellCauchyStress, tHostCauchyStress);

    Plato::ScalarVector tCellVonMises("Von Mises Stress", tNumCells);

    Plato::VonMisesYieldFunction<tSpaceDim, tNumVoigtTerms> tComputeVonMises;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
    {
        Plato::Array<tNumVoigtTerms> tStress;
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
        {
            tStress(iVoigt) = tCellCauchyStress(tCellOrdinal, iVoigt);
        }

        Plato::Scalar tVonMises(0.0);
               
        tComputeVonMises(tCellOrdinal, tStress, tVonMises);

        tCellVonMises(tCellOrdinal) = tVonMises;
    }, "Test Von Mises Yield Stress Calculation");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {1.096154, 1.457692};
    auto tHostCellVonMises = Kokkos::create_mirror(tCellVonMises);
    Kokkos::deep_copy(tHostCellVonMises, tCellVonMises);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostCellVonMises(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_ComputeStructuralMass_3D)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tet4>>::Residual;
    Plato::AugLagStressCriterion<Residual> tCriterion(tOnlyDomain, tDataMap);
    tCriterion.setCellMaterialDensity(0.5);
    tCriterion.computeStructuralMass();

    // TEST STRUCTURAL MASS CALCULATION
    auto tStructMass = tCriterion.getMassNormalizationMultiplier();
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(0.5, tStructMass, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_CriterionEval_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::blas1::fill(0.1, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            {   tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal);}, "fill state");
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result/output workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::AugLagStressCriterion<Residual> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::blas1::fill(0.01, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);
    Plato::ScalarVector tLagrangeMultipliers("Lagrange Multiplier", tNumCells);
    Plato::blas1::fill(0.1, tLagrangeMultipliers);
    tCriterion.setLagrangeMultipliers(tLagrangeMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.01489406, 0.01489406, 0.01489406, 0.01489406, 0.01489406, 0.01489406};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }

    // ****** TEST GLOBAL SUM ******
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(0.0893644, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_FiniteDiff_CriterionGradZ_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TRI3", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;
    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;
    Plato::AugLagStressCriterion<GradientZ> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::blas1::fill(0.1, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_control<GradientZ, ElementType>(tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_FiniteDiff_CriterionGradU_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TRI3", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using Jacobian = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    Plato::AugLagStressCriterion<Jacobian> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::blas1::fill(0.1, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_state<Jacobian, ElementType>(tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_FiniteDiff_CriterionGradZ_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;
    using StateT = typename GradientZ::StateScalarType;
    using ConfigT = typename GradientZ::ConfigScalarType;
    using ResultT = typename GradientZ::ResultScalarType;
    using ControlT = typename GradientZ::ControlScalarType;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::AugLagStressCriterion<GradientZ> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::blas1::fill(0.1, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_control<GradientZ, ElementType>(tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_FiniteDiff_CriterionGradU_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    using Jacobian = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    using StateT = Jacobian::StateScalarType;
    using ConfigT = Jacobian::ConfigScalarType;
    using ResultT = Jacobian::ResultScalarType;
    using ControlT = Jacobian::ControlScalarType;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::AugLagStressCriterion<Jacobian> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::blas1::fill(0.1, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_state<Jacobian, ElementType>(tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_UpdateMultipliers1)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    Plato::AugLagStressCriterion<Residual> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::blas1::fill(0.01, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = ElementType::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("controls", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("State", tNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0) = 0.4360751; tHostState(1) = 0.2577532;  tHostState(2) = 0.4132397;
    tHostState(3) = 0.4193760; tHostState(4) = 0.4646589;  tHostState(5) = 0.1790205;
    tHostState(6) = 0.2340891; tHostState(7) = 0.4072918;  tHostState(8) = 0.2111099;
    tHostState(9) = 0.3215880; tHostState(10) = 0.2909588;  tHostState(11) = 0.3515484;
    tHostState(12) = 0.2459138; tHostState(13) = 0.3053604;  tHostState(14) = 0.4808919;
    tHostState(15) = 0.4664780; tHostState(16) = 0.3542847;  tHostState(17) = 0.3869188;
    tHostState(18) = 0.1566410; tHostState(19) = 0.3427876;  tHostState(20) = 0.1065202;
    tHostState(21) = 0.1971547; tHostState(22) = 0.1548926;  tHostState(23) = 0.4216707;
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // TEST UPDATE PROBLEM FUNCTION
    tCriterion.updateProblem(tStateWS, tControlWS, tConfigWS);
    tMassMultipliers = tCriterion.getMassMultipliers();
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldMassMultipliers(tNumCells, 0.525);
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers(tNumCells, 0.01);
    auto tHostMassMultipliers = Kokkos::create_mirror(tMassMultipliers);
    Kokkos::deep_copy(tHostMassMultipliers, tMassMultipliers);
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldMassMultipliers[tIndex], tHostMassMultipliers(tIndex), tTolerance);
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_UpdateMultipliers2)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    Plato::AugLagStressCriterion<Residual> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::blas1::fill(0.01, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = ElementType::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("controls", tNumVerts);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("State", tNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0) = 4.360751; tHostState(1) = 2.577532;  tHostState(2) = 4.132397;
    tHostState(3) = 4.193760; tHostState(4) = 4.646589;  tHostState(5) = 1.790205;
    tHostState(6) = 2.340891; tHostState(7) = 4.072918;  tHostState(8) = 2.111099;
    tHostState(9) = 3.215880; tHostState(10) = 2.909588;  tHostState(11) = 3.515484;
    tHostState(12) = 2.459138; tHostState(13) = 3.053604;  tHostState(14) = 4.808919;
    tHostState(15) = 4.664780; tHostState(16) = 3.542847;  tHostState(17) = 3.869188;
    tHostState(18) = 1.566410; tHostState(19) = 3.427876;  tHostState(20) = 1.065202;
    tHostState(21) = 1.971547; tHostState(22) = 1.548926;  tHostState(23) = 4.216707;
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // TEST UPDATE PROBLEM FUNCTION
    tCriterion.updateProblem(tStateWS, tControlWS, tConfigWS);
    tMassMultipliers = tCriterion.getMassMultipliers();
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldMassMultipliers(tNumCells, 0.);
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers =
      {0.131739476, 0.112932370, 0.064022544, 0.111203340, 0.042183347, 0.219975864};
    auto tHostMassMultipliers = Kokkos::create_mirror(tMassMultipliers);
    Kokkos::deep_copy(tHostMassMultipliers, tMassMultipliers);
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldMassMultipliers[tIndex], tHostMassMultipliers(tIndex), tTolerance);
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagGeneral_FiniteDiff_CriterionGradZ_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TRI3", tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

    using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;
    Plato::AugLagStressCriterionGeneral<GradientZ> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_control<GradientZ, ElementType>(tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagGeneral_FiniteDiff_CriterionGradU_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TRI3", tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

    using Jacobian = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    Plato::AugLagStressCriterionGeneral<Jacobian> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_state<Jacobian, ElementType>(tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagGeneral_FiniteDiff_CriterionGradZ_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;
    using StateT = typename GradientZ::StateScalarType;
    using ConfigT = typename GradientZ::ConfigScalarType;
    using ResultT = typename GradientZ::ResultScalarType;
    using ControlT = typename GradientZ::ControlScalarType;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::AugLagStressCriterionGeneral<GradientZ> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_control<GradientZ, ElementType>(tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagGeneral_FiniteDiff_CriterionGradU_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    using Jacobian = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    using StateT = Jacobian::StateScalarType;
    using ConfigT = Jacobian::ConfigScalarType;
    using ResultT = Jacobian::ResultScalarType;
    using ControlT = Jacobian::ControlScalarType;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::AugLagStressCriterionGeneral<Jacobian> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_state<Jacobian, ElementType>(tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLagGeneral_computeStructuralMass)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tet4>>::Residual;
    Plato::AugLagStressCriterionGeneral<Residual> tCriterion(tOnlyDomain, tDataMap);

    // TEST FUNCTION
    tCriterion.computeStructuralMass();
    auto tStructuralMass = tCriterion.getMassNormalizationMultiplier();
    constexpr Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(1.0, tStructuralMass, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_UpdateProbelm1)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    Plato::AugLagStressCriterion<Residual> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = ElementType::mNumNodesPerCell;

    // Create configuration workset
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("controls", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("State", tNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0) = 0.4360751; tHostState(1) = 0.2577532;  tHostState(2) = 0.4132397;
    tHostState(3) = 0.4193760; tHostState(4) = 0.4646589;  tHostState(5) = 0.1790205;
    tHostState(6) = 0.2340891; tHostState(7) = 0.4072918;  tHostState(8) = 0.2111099;
    tHostState(9) = 0.3215880; tHostState(10) = 0.2909588;  tHostState(11) = 0.3515484;
    tHostState(12) = 0.2459138; tHostState(13) = 0.3053604;  tHostState(14) = 0.4808919;
    tHostState(15) = 0.4664780; tHostState(16) = 0.3542847;  tHostState(17) = 0.3869188;
    tHostState(18) = 0.1566410; tHostState(19) = 0.3427876;  tHostState(20) = 0.1065202;
    tHostState(21) = 0.1971547; tHostState(22) = 0.1548926;  tHostState(23) = 0.4216707;
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // TEST UPDATE PROBLEM FUNCTION
    tCriterion.updateProblem(tStateWS, tControlWS, tConfigWS);

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers(tNumCells, 0.01);
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_UpdateProblem2)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    Plato::AugLagStressCriterion<Residual> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = ElementType::mNumNodesPerCell;

    // Create configuration workset
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("controls", tNumVerts);
    Plato::blas1::fill(0.5, tControl);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("State", tNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0) = 4.360751; tHostState(1) = 2.577532;  tHostState(2) = 4.132397;
    tHostState(3) = 4.193760; tHostState(4) = 4.646589;  tHostState(5) = 1.790205;
    tHostState(6) = 2.340891; tHostState(7) = 4.072918;  tHostState(8) = 2.111099;
    tHostState(9) = 3.215880; tHostState(10) = 2.909588;  tHostState(11) = 3.515484;
    tHostState(12) = 2.459138; tHostState(13) = 3.053604;  tHostState(14) = 4.808919;
    tHostState(15) = 4.664780; tHostState(16) = 3.542847;  tHostState(17) = 3.869188;
    tHostState(18) = 1.566410; tHostState(19) = 3.427876;  tHostState(20) = 1.065202;
    tHostState(21) = 1.971547; tHostState(22) = 1.548926;  tHostState(23) = 4.216707;
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // TEST UPDATE PROBLEM FUNCTION
    tCriterion.updateProblem(tStateWS, tControlWS, tConfigWS);

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers =
      {0.131739476, 0.112932370, 0.064022544, 0.111203340, 0.042183347, 0.219975864};
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }

    auto tPenaltyMultiplier = tCriterion.getAugLagPenalty();
    TEST_FLOATING_EQUALITY(0.105, tPenaltyMultiplier, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, AugLag_CellDensity)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumNodesPerCell = 4;
    Plato::ScalarMultiVector tCellControls("Control Workset", tNumCells, tNumNodesPerCell);
    auto tHostCellControls = Kokkos::create_mirror(tCellControls);
    tHostCellControls(0, 0) = 1.00;
    tHostCellControls(1, 0) = 0.93;
    tHostCellControls(0, 1) = 0.90;
    tHostCellControls(1, 1) = 1.00;
    tHostCellControls(0, 2) = 0.95;
    tHostCellControls(1, 2) = 0.89;
    tHostCellControls(0, 3) = 0.89;
    tHostCellControls(1, 3) = 0.91;
    Kokkos::deep_copy(tCellControls, tHostCellControls);

    Plato::ScalarVector tCellDensity("Cell Density", tNumCells);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
    { tCellDensity(tCellOrdinal) = Plato::cell_density<tNumNodesPerCell>(tCellOrdinal, tCellControls); }, "Test cell density inline function");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.935, 0.9325};
    auto tHostCellDensity = Kokkos::create_mirror(tCellDensity);
    Kokkos::deep_copy(tHostCellDensity, tCellDensity);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostCellDensity(tIndex), tGold[tIndex], tTolerance);
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPlusTensileEnergy2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TRI3", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->NumElements();

    // Create control workset
    const Plato::Scalar tPseudoDensity = 1.0;
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(tPseudoDensity, tControl);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarMultiVector tStates("States", /*numStates=*/ 1, tNumDofs);
    Kokkos::deep_copy(tStates, 0.1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            { tStates(0, aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal) * 2; }, "fill state");

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::Elliptic::WeightedSumFunction<Plato::Mechanics<Plato::Tri3>> tWeightedSum(tSpatialModel, tDataMap);

    const Plato::Scalar tMaterialDensity = 0.5;
    const auto tMassCriterion = std::make_shared<Plato::Elliptic::MassMoment<Residual>>(tOnlyDomain, tDataMap);
    tMassCriterion->setMaterialDensity(tMaterialDensity);
    tMassCriterion->setCalculationType("Mass");

    const auto tTensileEnergyCriterion = std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(tOnlyDomain, tDataMap);
    Plato::ScalarVector tLagrangeMultipliers("lagrange multipliers", tNumCells);
    Plato::blas1::fill(0.1, tLagrangeMultipliers);
    tTensileEnergyCriterion->setLagrangeMultipliers(tLagrangeMultipliers);
    tTensileEnergyCriterion->setAugLagPenalty(1.5);
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const auto tLocalMeasure =
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tOnlyDomain, tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tTensileEnergyCriterion->setLocalMeasure(tLocalMeasure, tLocalMeasure);
    tTensileEnergyCriterion->setLocalMeasureValueLimit(0.15);

    // Append function one
    const auto tPhysicsScalarFuncMass =
          std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tri3>>>(tSpatialModel, tDataMap);

    tPhysicsScalarFuncMass->setEvaluator(tMassCriterion, tOnlyDomain.getDomainName());

    const Plato::Scalar tMassFunctionWeight = 0.75;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncMass);
    tWeightedSum.appendFunctionWeight(tMassFunctionWeight);

    // Append function two
    const auto tPhysicsScalarFuncTensileEnergy =
          std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tri3>>>(tSpatialModel, tDataMap);

    const Plato::Scalar tTensileEnergyFunctionWeight = 0.5;
    tPhysicsScalarFuncTensileEnergy->setEvaluator(tTensileEnergyCriterion, tOnlyDomain.getDomainName());
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncTensileEnergy);
    tWeightedSum.appendFunctionWeight(tTensileEnergyFunctionWeight);

    Plato::Solutions tSolution;
    tSolution.set("State", tStates);
    auto tObjFuncVal = tWeightedSum.value(tSolution, tControl, 0.0);

    Plato::Scalar tMassGoldValue = pow(static_cast<Plato::Scalar>(tMeshWidth), tSpaceDim)
                                   * tPseudoDensity * tMassFunctionWeight * tMaterialDensity;

    Plato::Scalar tTensileEnergyGoldValue = tTensileEnergyFunctionWeight * 775.0772911;

    Plato::Scalar tGoldWeightedSum = tMassGoldValue + tTensileEnergyGoldValue;

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(tGoldWeightedSum, tObjFuncVal, tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPlusVonMises_GradZ_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TRI3", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::Elliptic::WeightedSumFunction<Plato::Mechanics<Plato::Tri3>> tWeightedSum(tSpatialModel, tDataMap);

    using Residual  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;

    const auto tCriterionResidual =
          std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(tOnlyDomain, tDataMap);

    const auto tCriterionGradZ =
          std::make_shared<Plato::AugLagStressCriterionQuadratic<GradientZ>>(tOnlyDomain, tDataMap);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    const auto tLocalMeasureGradZ = std::make_shared<Plato::VonMisesLocalMeasure<GradientZ>> (tOnlyDomain, tCellStiffMatrix, "VonMises");
    const auto tLocalMeasurePODType = std::make_shared<Plato::VonMisesLocalMeasure<Residual>> (tOnlyDomain, tCellStiffMatrix, "VonMises");

    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    tCriterionGradZ->setLocalMeasure(tLocalMeasureGradZ, tLocalMeasurePODType);

    // Append function one
    const auto tPhysicsScalarFuncVonMises =
          std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tri3>>>(tSpatialModel, tDataMap);

    tPhysicsScalarFuncVonMises->setEvaluator(tCriterionResidual, tOnlyDomain.getDomainName());
    tPhysicsScalarFuncVonMises->setEvaluator(tCriterionGradZ, tOnlyDomain.getDomainName());

    const Plato::Scalar tVonMisesFunctionWeight = 1.0;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncVonMises);
    tWeightedSum.appendFunctionWeight(tVonMisesFunctionWeight);

    const Plato::Scalar tMaterialDensity = 0.5;
    const std::shared_ptr<Plato::Elliptic::MassMoment<GradientZ>> tMassCriterionGradZ =
          std::make_shared<Plato::Elliptic::MassMoment<GradientZ>>(tOnlyDomain, tDataMap);
    tMassCriterionGradZ->setMaterialDensity(tMaterialDensity);
    tMassCriterionGradZ->setCalculationType("Mass");

    const std::shared_ptr<Plato::Elliptic::MassMoment<Residual>> tMassCriterion =
          std::make_shared<Plato::Elliptic::MassMoment<Residual>>(tOnlyDomain, tDataMap);
    tMassCriterion->setMaterialDensity(tMaterialDensity);
    tMassCriterion->setCalculationType("Mass");

    // Append function two
    const auto tPhysicsScalarFuncMass =
          std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tri3>>>(tSpatialModel, tDataMap);

    tPhysicsScalarFuncMass->setEvaluator(tMassCriterion, tOnlyDomain.getDomainName());
    tPhysicsScalarFuncMass->setEvaluator(tMassCriterionGradZ, tOnlyDomain.getDomainName());

    const Plato::Scalar tMassFunctionWeight = 0.75;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncMass);
    tWeightedSum.appendFunctionWeight(tMassFunctionWeight);

    Plato::test_partial_control<GradientZ, ElementType>(tMesh, tWeightedSum);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPlusVonMises_GradZ_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::Elliptic::WeightedSumFunction<Plato::Mechanics<Plato::Tet4>> tWeightedSum(tSpatialModel, tDataMap);

    using Residual  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;

    const auto tCriterionResidual = std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(tOnlyDomain, tDataMap);

    const auto tCriterionGradZ = std::make_shared<Plato::AugLagStressCriterionQuadratic<GradientZ>>(tOnlyDomain, tDataMap);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    const auto tLocalMeasureGradZ = std::make_shared<Plato::VonMisesLocalMeasure<GradientZ>> (tOnlyDomain, tCellStiffMatrix, "VonMises");
    const auto tLocalMeasurePODType = std::make_shared<Plato::VonMisesLocalMeasure<Residual>> (tOnlyDomain, tCellStiffMatrix, "VonMises");

    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    tCriterionGradZ->setLocalMeasure(tLocalMeasureGradZ, tLocalMeasurePODType);

    // Append function one
    const auto tPhysicsScalarFuncVonMises =
          std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tet4>>>(tSpatialModel, tDataMap);

    tPhysicsScalarFuncVonMises->setEvaluator(tCriterionResidual, tOnlyDomain.getDomainName());
    tPhysicsScalarFuncVonMises->setEvaluator(tCriterionGradZ, tOnlyDomain.getDomainName());

    const Plato::Scalar tVonMisesFunctionWeight = 1.0;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncVonMises);
    tWeightedSum.appendFunctionWeight(tVonMisesFunctionWeight);

    const Plato::Scalar tMaterialDensity = 0.5;
    const auto tMassCriterionGradZ = std::make_shared<Plato::Elliptic::MassMoment<GradientZ>>(tOnlyDomain, tDataMap);
    tMassCriterionGradZ->setMaterialDensity(tMaterialDensity);
    tMassCriterionGradZ->setCalculationType("Mass");

    const auto tMassCriterion = std::make_shared<Plato::Elliptic::MassMoment<Residual>>(tOnlyDomain, tDataMap);
    tMassCriterion->setMaterialDensity(tMaterialDensity);
    tMassCriterion->setCalculationType("Mass");

    // Append function two
    const auto tPhysicsScalarFuncMass =
          std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tet4>>>(tSpatialModel, tDataMap);

    tPhysicsScalarFuncMass->setEvaluator(tMassCriterion, tOnlyDomain.getDomainName());
    tPhysicsScalarFuncMass->setEvaluator(tMassCriterionGradZ, tOnlyDomain.getDomainName());

    const Plato::Scalar tMassFunctionWeight = 0.75;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncMass);
    tWeightedSum.appendFunctionWeight(tMassFunctionWeight);

    Plato::test_partial_control<GradientZ, ElementType>(tMesh, tWeightedSum);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPlusVonMises_GradU_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TRI3", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::Elliptic::WeightedSumFunction<Plato::Mechanics<Plato::Tri3>> tWeightedSum(tSpatialModel, tDataMap);

    using Residual  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using Jacobian = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;

    const auto tCriterionResidual = std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(tOnlyDomain, tDataMap);

    const auto tCriterionGradU = std::make_shared<Plato::AugLagStressCriterionQuadratic<Jacobian>>(tOnlyDomain, tDataMap);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    const auto tLocalMeasureGradU =
        std::make_shared<Plato::VonMisesLocalMeasure<Jacobian>> (tOnlyDomain, tCellStiffMatrix, "VonMises");
    const auto tLocalMeasurePODType =
        std::make_shared<Plato::VonMisesLocalMeasure<Residual>> (tOnlyDomain, tCellStiffMatrix, "VonMises");

    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    tCriterionGradU->setLocalMeasure(tLocalMeasureGradU, tLocalMeasurePODType);

    // Append function one
    const auto tPhysicsScalarFuncVonMises =
          std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tri3>>>(tSpatialModel, tDataMap);

    tPhysicsScalarFuncVonMises->setEvaluator(tCriterionResidual, tOnlyDomain.getDomainName());
    tPhysicsScalarFuncVonMises->setEvaluator(tCriterionGradU, tOnlyDomain.getDomainName());

    const Plato::Scalar tVonMisesFunctionWeight = 1.0;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncVonMises);
    tWeightedSum.appendFunctionWeight(tVonMisesFunctionWeight);

    const Plato::Scalar tMaterialDensity = 0.5;
    const auto tMassCriterionGradU = std::make_shared<Plato::Elliptic::MassMoment<Jacobian>>(tOnlyDomain, tDataMap);
    tMassCriterionGradU->setMaterialDensity(tMaterialDensity);
    tMassCriterionGradU->setCalculationType("Mass");

    const auto tMassCriterion = std::make_shared<Plato::Elliptic::MassMoment<Residual>>(tOnlyDomain, tDataMap);
    tMassCriterion->setMaterialDensity(tMaterialDensity);
    tMassCriterion->setCalculationType("Mass");

    // Append function two
    const auto tPhysicsScalarFuncMass =
          std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tri3>>>(tSpatialModel, tDataMap);

    tPhysicsScalarFuncMass->setEvaluator(tMassCriterion, tOnlyDomain.getDomainName());
    tPhysicsScalarFuncMass->setEvaluator(tMassCriterionGradU, tOnlyDomain.getDomainName());

    const Plato::Scalar tMassFunctionWeight = 0.75;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncMass);
    tWeightedSum.appendFunctionWeight(tMassFunctionWeight);

    Plato::test_partial_state<Jacobian, ElementType>(tMesh, tWeightedSum);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPlusVonMises_GradU_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    constexpr Plato::OrdinalType tNumVoigtTerms = ElementType::mNumVoigtTerms;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::Elliptic::WeightedSumFunction<Plato::Mechanics<Plato::Tet4>> tWeightedSum(tSpatialModel, tDataMap);

    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using Jacobian = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;

    const auto tCriterionResidual = std::make_shared<Plato::AugLagStressCriterionQuadratic<Residual>>(tOnlyDomain, tDataMap);

    const auto tCriterionGradU = std::make_shared<Plato::AugLagStressCriterionQuadratic<Jacobian>>(tOnlyDomain, tDataMap);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Plato::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    const auto tLocalMeasureGradU = std::make_shared<Plato::VonMisesLocalMeasure<Jacobian>>(tOnlyDomain, tCellStiffMatrix, "VonMises");
    const auto tLocalMeasurePODType = std::make_shared<Plato::VonMisesLocalMeasure<Residual>>(tOnlyDomain, tCellStiffMatrix, "VonMises");

    tCriterionResidual->setLocalMeasure(tLocalMeasurePODType, tLocalMeasurePODType);
    tCriterionGradU->setLocalMeasure(tLocalMeasureGradU, tLocalMeasurePODType);

    // Append function one
    const auto tPhysicsScalarFuncVonMises =
          std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tet4>>>(tSpatialModel, tDataMap);

    tPhysicsScalarFuncVonMises->setEvaluator(tCriterionResidual, tOnlyDomain.getDomainName());
    tPhysicsScalarFuncVonMises->setEvaluator(tCriterionGradU, tOnlyDomain.getDomainName());

    const Plato::Scalar tVonMisesFunctionWeight = 1.0;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncVonMises);
    tWeightedSum.appendFunctionWeight(tVonMisesFunctionWeight);

    const Plato::Scalar tMaterialDensity = 0.5;
    const auto tMassCriterionGradU = std::make_shared<Plato::Elliptic::MassMoment<Jacobian>>(tOnlyDomain, tDataMap);
    tMassCriterionGradU->setMaterialDensity(tMaterialDensity);
    tMassCriterionGradU->setCalculationType("Mass");

    const auto tMassCriterion = std::make_shared<Plato::Elliptic::MassMoment<Residual>>(tOnlyDomain, tDataMap);
    tMassCriterion->setMaterialDensity(tMaterialDensity);
    tMassCriterion->setCalculationType("Mass");

    // Append function two
    const auto tPhysicsScalarFuncMass =
          std::make_shared<Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<Plato::Tet4>>>(tSpatialModel, tDataMap);

    tPhysicsScalarFuncMass->setEvaluator(tMassCriterion, tOnlyDomain.getDomainName());
    tPhysicsScalarFuncMass->setEvaluator(tMassCriterionGradU, tOnlyDomain.getDomainName());

    const Plato::Scalar tMassFunctionWeight = 0.75;
    tWeightedSum.allocateScalarFunctionBase(tPhysicsScalarFuncMass);
    tWeightedSum.appendFunctionWeight(tMassFunctionWeight);

    Plato::test_partial_state<Jacobian, ElementType>(tMesh, tWeightedSum);
}

} // namespace AugLagStressTest
