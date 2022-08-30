/*
 * StabilizedMechanicsTests.cpp
 *
 *  Created on: Mar 26, 2020
 */

#include <Teuchos_XMLParameterListCoreHelpers.hpp>

#include "Teuchos_UnitTestHarness.hpp"

#include "Tet4.hpp"
#include "FadTypes.hpp"
#include "GradientMatrix.hpp"
#include "stabilized/Kinematics.hpp"
#include "stabilized/Mechanics.hpp"
#include "stabilized/MechanicsElement.hpp"
#include "PlatoUtilities.hpp"
#include "PlatoTestHelpers.hpp"
#include "stabilized/Problem.hpp"


namespace StabilizedMechanicsTests
{


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StabilizedMechanics_Kinematics3D)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = Plato::Stabilized::MechanicsElement<Plato::Tet4>;

    constexpr int tSpaceDim        = ElementType::mNumSpatialDims;
    constexpr int tNumVoigtTerms   = ElementType::mNumVoigtTerms;
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    constexpr int tNumDofsPerCell  = ElementType::mNumDofsPerCell;

    // Set configuration workset
    auto tNumCells = tMesh->NumElements();
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarArray3D tConfig("configuration", tNumCells, ElementType::mNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfig);

    // Set state workset
    auto tNumNodes = tMesh->NumNodes();
    Plato::ScalarVector tState("state", tNumDofsPerNode * tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), KOKKOS_LAMBDA(const Plato::OrdinalType & aNodeOrdinal)
    {
        tState(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal; // disp_x
        tState(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal; // disp_y
        tState(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal; // disp_z
        tState(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal; // press
    }, "set global state");
    Plato::ScalarMultiVector tStateWS("current state", tNumCells, tNumDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    auto tCubPoints  = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    auto tNumPoints  = tCubWeights.size();

    Plato::ScalarArray3D tStrains("strains", tNumCells, tNumPoints, tNumVoigtTerms);

    Plato::Stabilized::Kinematics <ElementType> tKinematics;
    Plato::ComputeGradientMatrix  <ElementType> tComputeGradient;
    Kokkos::parallel_for("gradients", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int cellOrdinal, const int gpOrdinal)
    {
        Plato::Scalar tVolume(0.0);

        Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, Plato::Scalar> tGradient;

        Plato::Array<ElementType::mNumVoigtTerms, Plato::Scalar> tStrain(0.0);
        Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tPressGrad(0.0);

        auto tCubPoint = tCubPoints(gpOrdinal);

        tComputeGradient(cellOrdinal, tCubPoint, tConfig, tGradient, tVolume);
        tKinematics(cellOrdinal, tStrain, tPressGrad, tStateWS, tGradient);

        for(Plato::OrdinalType iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
        {
            tStrains(cellOrdinal, gpOrdinal, iVoigt) = tStrain(iVoigt);
        }
    });

    std::vector<std::vector<std::vector<Plato::Scalar>>> tGold =
        { {{4e-7, 4e-7, 3e-7, 8.0e-7, 13e-7, 10e-7}},
          {{4e-7, 4e-7, 3e-7, 8.0e-7, 13e-7, 10e-7}},
          {{4e-7, 4e-7, 3e-7, 8.0e-7, 13e-7, 10e-7}},
          {{4e-7, 4e-7, 3e-7, 8.0e-7, 13e-7, 10e-7}},
          {{4e-7, 4e-7, 3e-7, 8.0e-7, 13e-7, 10e-7}},
          {{4e-7, 4e-7, 3e-7, 8.0e-7, 13e-7, 10e-7}} };
    auto tHostStrains = Kokkos::create_mirror(tStrains);
    Kokkos::deep_copy(tHostStrains, tStrains);

    const Plato::Scalar tTolerance = 1e-4;
    const Plato::OrdinalType tDim0 = tStrains.extent(0);
    const Plato::OrdinalType tDim1 = tStrains.extent(1);
    const Plato::OrdinalType tDim2 = tStrains.extent(2);
    for (Plato::OrdinalType tIndexI = 0; tIndexI < tDim0; tIndexI++)
    {
        for (Plato::OrdinalType tIndexJ = 0; tIndexJ < tDim1; tIndexJ++)
        {
            for (Plato::OrdinalType tIndexK = 0; tIndexK < tDim2; tIndexK++)
            {
                TEST_FLOATING_EQUALITY(tHostStrains(tIndexI, tIndexJ, tIndexK), tGold[tIndexI][tIndexJ][tIndexK], tTolerance);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StabilizedMechanics_Solution3D)
{
    // 1. DEFINE PROBLEM
    const bool tOutputData = false; // for debugging purpose, set true to enable the Paraview output file
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = Plato::Stabilized::MechanicsElement<Plato::Tet4>;

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                               \n"
        "<Parameter name='Physics'         type='string'  value='Stabilized Mechanical'/> \n"
        "<Parameter name='PDE Constraint'  type='string'  value='Elliptic'/>              \n"
        "<ParameterList name='Elliptic'>                                                  \n"
          "<ParameterList name='Penalty Function'>                                        \n"
            "<Parameter name='Type' type='string' value='SIMP'/>                          \n"
            "<Parameter name='Exponent' type='double' value='3.0'/>                       \n"
            "<Parameter name='Minimum Value' type='double' value='1.0e-9'/>               \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
        "<ParameterList name='Time Stepping'>                                             \n"
          "<Parameter name='Number Time Steps' type='int' value='2'/>                     \n"
          "<Parameter name='Time Step' type='double' value='1.0'/>                        \n"
        "</ParameterList>                                                                 \n"
        "<ParameterList name='Newton Iteration'>                                          \n"
          "<Parameter name='Number Iterations' type='int' value='3'/>                     \n"
        "</ParameterList>                                                                 \n"
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
        "<ParameterList name='Essential Boundary Conditions'>                             \n"
        "</ParameterList>                                                                 \n"
    "</ParameterList>                                                                     \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    Plato::Stabilized::Problem<Plato::Stabilized::Mechanics<Plato::Tet4>> tEllipticVMSProblem(tMesh, *tParamList, tMachine);

    // 2. Get Dirichlet Boundary Conditions
    Plato::OrdinalType tDispDofX = 0;
    Plato::OrdinalType tDispDofY = 1;
    Plato::OrdinalType tDispDofZ = 2;
    constexpr Plato::OrdinalType tNumDofsPerNode = ElementType::mNumDofsPerNode;
    auto tDirichletIndicesBoundaryX0_Xdof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(tMesh, "x-", tNumDofsPerNode, tDispDofX);
    auto tDirichletIndicesBoundaryX0_Ydof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(tMesh, "x-", tNumDofsPerNode, tDispDofY);
    auto tDirichletIndicesBoundaryX0_Zdof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(tMesh, "x-", tNumDofsPerNode, tDispDofZ);
    auto tDirichletIndicesBoundaryX1_Ydof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(tMesh, "x+", tNumDofsPerNode, tDispDofY);

    // 3. Set Dirichlet Boundary Conditions
    Plato::Scalar tValueToSet = 0;
    auto tNumDirichletDofs = tDirichletIndicesBoundaryX0_Xdof.size() + tDirichletIndicesBoundaryX0_Ydof.size() +
            tDirichletIndicesBoundaryX0_Zdof.size() + tDirichletIndicesBoundaryX1_Ydof.size();
    Plato::ScalarVector tDirichletValues("Dirichlet Values", tNumDirichletDofs);
    Plato::OrdinalVector tDirichletDofs("Dirichlet Dofs", tNumDirichletDofs);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_Xdof.size()), KOKKOS_LAMBDA(const Plato::OrdinalType & aIndex)
    {
        tDirichletValues(aIndex) = tValueToSet;
        tDirichletDofs(aIndex) = tDirichletIndicesBoundaryX0_Xdof(aIndex);
    }, "set dirichlet values and indices");

    auto tOffset = tDirichletIndicesBoundaryX0_Xdof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_Ydof.size()), KOKKOS_LAMBDA(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryX0_Ydof(aIndex);
    }, "set dirichlet values and indices");

    tOffset += tDirichletIndicesBoundaryX0_Ydof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_Zdof.size()), KOKKOS_LAMBDA(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryX0_Zdof(aIndex);
    }, "set dirichlet values and indices");

    tValueToSet = -1e-3;
    tOffset += tDirichletIndicesBoundaryX0_Zdof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX1_Ydof.size()), KOKKOS_LAMBDA(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryX1_Ydof(aIndex);
    }, "set dirichlet values and indices");
    tEllipticVMSProblem.setEssentialBoundaryConditions(tDirichletDofs, tDirichletValues);

    // 4. Solve problem
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tEllipticVMSProblem.solution(tControls);
    auto tState = tSolution.get("State");

    // 5. Test Results
    std::vector<std::vector<Plato::Scalar>> tGold = {
   { 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00},
   { 0.00000000e+00,  0.00000000e+00, 0.00000000e+00, -3.76599539e-06, 
     0.00000000e+00,  0.00000000e+00, 0.00000000e+00, -2.75665794e-05, 
     0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  8.62653419e-05, 
     0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  7.08165405e-05, 
    -1.80390643e-04, -1.00000000e-03, 9.08131637e-05, -6.99967486e-05, 
    -3.92749639e-04, -1.00000000e-03, 5.10044731e-05, -9.98603044e-05, 
     3.11823346e-04, -1.00000000e-03, 4.81515303e-05,  1.77457777e-05, 
     2.34034774e-04, -1.00000000e-03, 4.35769056e-05, -3.76599539e-06 }
  };
    auto tHostState = Kokkos::create_mirror(tState);
    Kokkos::deep_copy(tHostState, tState);

    const Plato::Scalar tTolerance = 1e-4;
    const Plato::OrdinalType tDim0 = tState.extent(0);
    const Plato::OrdinalType tDim1 = tState.extent(1);
    for (Plato::OrdinalType tIndexI = 0; tIndexI < tDim0; tIndexI++)
    {
        for (Plato::OrdinalType tIndexJ = 0; tIndexJ < tDim1; tIndexJ++)
        {
            // printf("X(%d,%d) = %12.8e\n", tIndexI, tIndexJ, tHostState(tIndexI, tIndexJ));
            TEST_FLOATING_EQUALITY(tHostState(tIndexI, tIndexJ), tGold[tIndexI][tIndexJ], tTolerance);
        }
    }

    // 6. Output Data
    if(tOutputData)
    {
        tEllipticVMSProblem.output("Output");
    }

}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StabilizedMechanics_Residual3D)
{
    // 1. PREPARE PROBLEM INPUS FOR TEST
    Plato::DataMap tDataMap;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    Teuchos::RCP<Teuchos::ParameterList> tPDEInputs =
        Teuchos::getParametersFromXmlString(
        "<ParameterList name='Plato Problem'>                                           \n"
        "  <ParameterList name='Spatial Model'>                                         \n"
        "    <ParameterList name='Domains'>                                             \n"
        "      <ParameterList name='Design Volume'>                                     \n"
        "        <Parameter name='Element Block' type='string' value='body'/>           \n"
        "        <Parameter name='Material Model' type='string' value='Fancy Feast'/>   \n"
        "      </ParameterList>                                                         \n"
        "    </ParameterList>                                                           \n"
        "  </ParameterList>                                                             \n"
        "  <ParameterList name='Material Models'>                                       \n"
        "    <ParameterList name='Fancy Feast'>                                         \n"
        "      <ParameterList name='Isotropic Linear Elastic'>                          \n"
        "        <Parameter  name='Poissons Ratio' type='double' value='0.35'/>         \n"
        "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>       \n"
        "      </ParameterList>                                                         \n"
        "    </ParameterList>                                                           \n"
        "  </ParameterList>                                                             \n"
        "  <ParameterList name='Elliptic'>                                              \n"
        "    <ParameterList name='Penalty Function'>                                    \n"
        "      <Parameter name='Type' type='string' value='SIMP'/>                      \n"
        "      <Parameter name='Exponent' type='double' value='3.0'/>                   \n"
        "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>           \n"
        "    </ParameterList>                                                           \n"
        "  </ParameterList>                                                             \n"
        "</ParameterList>                                                               \n"
      );

    Plato::SpatialModel tSpatialModel(tMesh, *tPDEInputs);

    // 2. PREPARE FUNCTION INPUTS FOR TEST
    const Plato::OrdinalType tNumNodes = tMesh->NumNodes();
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    using ElementType = Plato::Stabilized::MechanicsElement<Plato::Tet4>;
    using EvalType = typename Plato::Stabilized::Evaluation<ElementType>::Residual;
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);

    // 2.1 SET CONFIGURATION
    Plato::ScalarArray3DT<EvalType::ConfigScalarType> tConfigWS("configuration", tNumCells, ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims);
    tWorksetBase.worksetConfig(tConfigWS);

    // 2.2 SET DESIGN VARIABLES
    Plato::ScalarMultiVectorT<EvalType::ControlScalarType> tControlWS("design variables", tNumCells, ElementType::mNumNodesPerCell);
    Kokkos::deep_copy(tControlWS, 1.0);

    // 2.3 SET GLOBAL STATE
    auto tNumDofsPerNode = ElementType::mNumDofsPerNode;
    Plato::ScalarVector tState("state", tNumDofsPerNode * tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), KOKKOS_LAMBDA(const Plato::OrdinalType & aNodeOrdinal)
    {
        tState(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal; // disp_x
        tState(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal; // disp_y
        tState(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal; // disp_z
        tState(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal; // press
    }, "set global state");
    Plato::ScalarMultiVectorT<EvalType::StateScalarType> tStateWS("current global state", tNumCells, ElementType::mNumDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // 2.4 SET PROJECTED PRESSURE GRADIENT
    auto tNumNodesPerCell = ElementType::mNumNodesPerCell;
    auto tNumSpatialDims  = ElementType::mNumSpatialDims;
    Plato::ScalarMultiVectorT<EvalType::NodeStateScalarType> tProjPressGradWS("projected pressure grad", tNumCells, ElementType::mNumNodeStatePerCell);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNodeIndex=0; tNodeIndex< tNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDimIndex=0; tDimIndex< tNumSpatialDims; tDimIndex++)
            {
                tProjPressGradWS(aCellOrdinal, tNodeIndex*tNumSpatialDims+tDimIndex) = (4e-7)*(tNodeIndex+1)*(tDimIndex+1)*(aCellOrdinal+1);
            }
        }
    }, "set projected pressure grad");

    auto tOnlyDomain = tSpatialModel.Domains.front();

    // 3. CALL FUNCTION
    auto tPenaltyParams = tPDEInputs->sublist("Elliptic").sublist("Penalty Function");
    Plato::Stabilized::ElastostaticResidual<EvalType, Plato::MSIMP> tComputeResidual(tOnlyDomain, tDataMap, *tPDEInputs, tPenaltyParams);
    Plato::ScalarMultiVectorT<EvalType::ResultScalarType> tResidualWS("residual", tNumCells, ElementType::mNumDofsPerCell);
    tComputeResidual.evaluate(tStateWS, tProjPressGradWS, tControlWS, tConfigWS, tResidualWS);

    // 5. TEST RESULTS
    std::vector<std::vector<Plato::Scalar>> tGold =
    {
     {-6172.8395061728, -28189.3004115226, -4938.2716049383, 4878.2244177454,
       20164.6090534979, 1234.5679012346, -18930.0411522634, -11946.9662566263,
      -22016.4609053498, 22016.4609053498, -3086.4197530864, 4878.2244179725,
       8024.6913580247, 4938.2716049383, 26954.7325102881, -5216.8899864990 },
     {-6172.8395061728, -22633.7448559671, -4938.2716049383, 6267.1133064828,
      -1851.8518518519, 17695.4732510288, -16460.9053497942, -3828.0010979887,
      -14609.0534979424, -1234.5679012346, 13374.4855967078, 9632.1514419629,
       22633.7448559671, 6172.8395061728, 8024.6913580247, -13923.1155023089 },
     {-8024.6913580247, -4938.2716049383, -19547.3251028807, 3365.0381341929,
      -14609.0534979424, 14609.0534979424, -3086.4197530864, 6730.0762699758,
       1851.8518518519, -15843.6213991770, 14609.0534979424, 3365.0381351015,
       20781.8930041152, 6172.8395061728, 8024.6913580247, -13460.1525392702 },
     {-8024.6913580247, -4938.2716049383, -23251.0288065844, 2439.1122080398,
      -16460.9053497942, -1234.5679012346, 15226.3374485597, 9169.1884793028,
       18312.7572016461, -18312.7572016461, 3086.4197530864, -7656.0021959774,
       6172.8395061728, 24485.5967078189, 4938.2716049383, -7656.0021950689 },
     {-30041.1522633745, -6172.8395061728, -8024.6913580247, 11145.3377243039,
       1851.8518518519, -25102.8806584362, 23868.3127572016, 1050.2233204381,
       22016.4609053498, 1234.5679012346, -20781.8930041152, -12409.9292201949,
       6172.8395061728, 30041.1522633745, 4938.2716049383, -9044.8910838063 },
     {-31893.0041152263, -6172.8395061728, -8024.6913580247, 10682.3747612653,
       25720.1646090535, -25720.1646090535, 3086.4197530864, -9507.8540479807,
      -1851.8518518519, 26954.7325102881, -25720.1646090535, -6142.8159131064,
       8024.6913580247, 4938.2716049383, 30658.4362139918, -6142.8159112893 }
    };

    auto tHostResidualWS = Kokkos::create_mirror(tResidualWS);
    Kokkos::deep_copy(tHostResidualWS, tResidualWS);
    constexpr Plato::Scalar tTolerance = 1e-4;
    for(Plato::OrdinalType tCellIndex=0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tDofIndex=0; tDofIndex< ElementType::mNumDofsPerCell; tDofIndex++)
        {
            //printf("residual(%d,%d) = %.10f\n", tCellIndex, tDofIndex, tHostResidualWS(tCellIndex, tDofIndex));
            TEST_FLOATING_EQUALITY(tHostResidualWS(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
        }
    }
}
}
// namespace StabilizedMechanicsTests
