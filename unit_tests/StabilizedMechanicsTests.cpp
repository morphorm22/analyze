/*
 * StabilizedMechanicsTests.cpp
 *
 *  Created on: Mar 26, 2020
 */

#include <Teuchos_XMLParameterListCoreHelpers.hpp>

#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoUtilities.hpp"
#include "OmegaHUtilities.hpp"
#include "PlatoTestHelpers.hpp"
#include "EllipticVMSProblem.hpp"


namespace StabilizedMechanicsTests
{


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StabilizedMechanics_Kinematics3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // Set configuration workset
    auto tNumCells = tMesh->nelems();
    using PhysicsT = Plato::StabilizedMechanics<tSpaceDim>;
    Plato::WorksetBase<PhysicsT> tWorksetBase(*tMesh);
    Plato::ScalarArray3D tConfig("configuration", tNumCells, PhysicsT::mNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfig);

    // Set state workset
    auto tNumNodes = tMesh->nverts();
    auto tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    Plato::ScalarVector tState("state", tSpaceDim * tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tState(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal; // disp_x
        tState(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal; // disp_y
        tState(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal; // disp_z
        tState(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal; // press
    }, "set global state");
    Plato::ScalarMultiVector tStateWS("current state", tNumCells, PhysicsT::mNumDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    Plato::ScalarVector tCellVolume("cell volume", tNumCells);
    Plato::ScalarMultiVector tStrains("strains", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVector tPressGrad("pressure grad", tNumCells, tSpaceDim);
    Plato::ScalarArray3D tGradient("gradient", tNumCells, PhysicsT::mNumNodesPerCell, tSpaceDim);

    Plato::StabilizedKinematics <tSpaceDim> tKinematics;
    Plato::ComputeGradientWorkset <tSpaceDim> tComputeGradient;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfig, tCellVolume);
        tKinematics(aCellOrdinal, tStrains, tPressGrad, tStateWS, tGradient);
    }, "kinematics test");

    std::vector<std::vector<Plato::Scalar>> tGold =
        { {1e-7,  6e-7,  3e-7, 1.1e-6,   4e-7,   5e-7},
          {3e-7,  6e-7, -3e-7,   7e-7,   8e-7,   9e-7},
          {3e-7,  2e-7,  3e-7,   5e-7,   1e-6,   7e-7},
          {5e-7, -2e-7,  3e-7,  -1e-7, 1.6e-6,   9e-7},
          {7e-7, -2e-7, -3e-7,  -5e-7,   2e-6, 1.3e-6},
          {7e-7, -6e-7,  3e-7,  -7e-7, 2.2e-6, 1.1e-6} };
    auto tHostStrains = Kokkos::create_mirror(tStrains);
    Kokkos::deep_copy(tHostStrains, tStrains);

    const Plato::Scalar tTolerance = 1e-4;
    const Plato::OrdinalType tDim0 = tStrains.dimension_0();
    const Plato::OrdinalType tDim1 = tStrains.dimension_1();
    for (Plato::OrdinalType tIndexI = 0; tIndexI < tDim0; tIndexI++)
    {
        for (Plato::OrdinalType tIndexJ = 0; tIndexJ < tDim1; tIndexJ++)
        {
            //printf("X(%d,%d) = %f\n", tIndexI, tIndexJ, tHostInput(tIndexI, tIndexJ));
            TEST_FLOATING_EQUALITY(tHostStrains(tIndexI, tIndexJ), tGold[tIndexI][tIndexJ], tTolerance);
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StabilizedMechanics_Solution3D)
{
    // 1. DEFINE PROBLEM
    const bool tOutputData = false; // for debugging purpose, set true to enable the Paraview output file
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::MeshSets tMeshSets;

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
          "<Parameter name='Number Iterations' type='int' value='3'/>                      \n"
        "</ParameterList>                                                                 \n"
        "<ParameterList name='Material Model'>                                            \n"
          "<ParameterList name='Isotropic Linear Elastic'>                                 \n"
            "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>               \n"
            "<Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>             \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
    "</ParameterList>                                                                     \n"
    );

    using PhysicsT = Plato::StabilizedMechanics<tSpaceDim>;
    Plato::EllipticVMSProblem<PhysicsT> tEllipticVMSProblem(*tMesh, tMeshSets, *tParamList);

    // 2. Get Dirichlet Boundary Conditions
    Plato::OrdinalType tDispDofX = 0;
    Plato::OrdinalType tDispDofY = 1;
    Plato::OrdinalType tDispDofZ = 2;
    constexpr Plato::OrdinalType tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    auto tDirichletIndicesBoundaryX0_Xdof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(*tMesh, "x0", tNumDofsPerNode, tDispDofX);
    auto tDirichletIndicesBoundaryX0_Ydof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(*tMesh, "x0", tNumDofsPerNode, tDispDofY);
    auto tDirichletIndicesBoundaryX0_Zdof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(*tMesh, "x0", tNumDofsPerNode, tDispDofZ);
    auto tDirichletIndicesBoundaryX1_Ydof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(*tMesh, "x1", tNumDofsPerNode, tDispDofY);

    // 3. Set Dirichlet Boundary Conditions
    Plato::Scalar tValueToSet = 0;
    auto tNumDirichletDofs = tDirichletIndicesBoundaryX0_Xdof.size() + tDirichletIndicesBoundaryX0_Ydof.size() +
            tDirichletIndicesBoundaryX0_Zdof.size() + tDirichletIndicesBoundaryX1_Ydof.size();
    Plato::ScalarVector tDirichletValues("Dirichlet Values", tNumDirichletDofs);
    Plato::LocalOrdinalVector tDirichletDofs("Dirichlet Dofs", tNumDirichletDofs);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_Xdof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        tDirichletValues(aIndex) = tValueToSet;
        tDirichletDofs(aIndex) = tDirichletIndicesBoundaryX0_Xdof(aIndex);
    }, "set dirichlet values and indices");

    auto tOffset = tDirichletIndicesBoundaryX0_Xdof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_Ydof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryX0_Ydof(aIndex);
    }, "set dirichlet values and indices");

    tOffset += tDirichletIndicesBoundaryX0_Ydof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_Zdof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryX0_Zdof(aIndex);
    }, "set dirichlet values and indices");

    tValueToSet = -1e-3;
    tOffset += tDirichletIndicesBoundaryX0_Zdof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX1_Ydof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryX1_Ydof(aIndex);
    }, "set dirichlet values and indices");
    tEllipticVMSProblem.setEssentialBoundaryConditions(tDirichletDofs, tDirichletValues);

    // 4. Solve problem
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::fill(1.0, tControls);
    auto tSolution = tEllipticVMSProblem.solution(tControls);

    // 5. Test Results
    std::vector<std::vector<Plato::Scalar>> tGold =
        { {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {0, 0, 0, -3.765995e-6, 0, 0, 0, -2.756658e-5, 0, 0, 0, 7.081654e-5, 0, 0, 0, 8.626534e-05,
           3.118233e-4, -1.0e-3, 4.815153e-5, 1.774578e-5, 2.340348e-4, -1.0e-3, 4.357691e-5, -3.765995e-6,
           -3.927496e-4, -1.0e-3, 5.100447e-5, -9.986030e-5, -1.803906e-4, -1.0e-3, 9.081316e-5, -6.999675e-5}};
    auto tHostSolution = Kokkos::create_mirror(tSolution);
    Kokkos::deep_copy(tHostSolution, tSolution);

    const Plato::Scalar tTolerance = 1e-4;
    const Plato::OrdinalType tDim0 = tSolution.dimension_0();
    const Plato::OrdinalType tDim1 = tSolution.dimension_1();
    for (Plato::OrdinalType tIndexI = 0; tIndexI < tDim0; tIndexI++)
    {
        for (Plato::OrdinalType tIndexJ = 0; tIndexJ < tDim1; tIndexJ++)
        {
            //printf("X(%d,%d) = %f\n", tIndexI, tIndexJ, tHostInput(tIndexI, tIndexJ));
            TEST_FLOATING_EQUALITY(tHostSolution(tIndexI, tIndexJ), tGold[tIndexI][tIndexJ], tTolerance);
        }
    }

    // 6. Output Data
    if(tOutputData)
    {
        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer("SolutionMesh", tMesh.getRawPtr(), tSpaceDim);
        for(Plato::OrdinalType tTime = 0; tTime < tSolution.dimension_0(); tTime++)
        {
            auto tSubView = Kokkos::subview(tSolution, tTime, Kokkos::ALL());
            Plato::output_vtk_node_field<tSpaceDim, tNumDofsPerNode>(tTime, tSubView, "State", *tMesh, tWriter);
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StabilizedMechanics_Residual3D)
{
    // 1. PREPARE PROBLEM INPUS FOR TEST
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    Teuchos::RCP<Teuchos::ParameterList> tPDEInputs =
        Teuchos::getParametersFromXmlString(
        "<ParameterList name='Plato Problem'>                                   \n"
        "<ParameterList name='Material Model'>                                  \n"
        "  <ParameterList name='Isotropic Linear Elastic'>                      \n"
        "    <Parameter  name='Poissons Ratio' type='double' value='0.35'/>     \n"
        "    <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>   \n"
        "  </ParameterList>                                                     \n"
        "</ParameterList>                                                       \n"
        "<ParameterList name='Elliptic'>                                        \n"
        "  <ParameterList name='Penalty Function'>                              \n"
        "    <Parameter name='Type' type='string' value='SIMP'/>                \n"
        "    <Parameter name='Exponent' type='double' value='3.0'/>             \n"
        "    <Parameter name='Minimum Value' type='double' value='1.0e-9'/>     \n"
        "  </ParameterList>                                                     \n"
        "</ParameterList>                                                       \n"
        "</ParameterList>                                                       \n"
      );

    // 2. PREPARE FUNCTION INPUTS FOR TEST
    const Plato::OrdinalType tNumNodes = tMesh->nverts();
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    using PhysicsT = Plato::StabilizedMechanics<tSpaceDim>;
    using EvalType = typename Plato::Evaluation<PhysicsT>::Residual;
    Plato::WorksetBase<PhysicsT> tWorksetBase(*tMesh);

    // 2.1 SET CONFIGURATION
    Plato::ScalarArray3DT<EvalType::ConfigScalarType> tConfigWS("configuration", tNumCells, PhysicsT::mNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // 2.2 SET DESIGN VARIABLES
    Plato::ScalarMultiVectorT<EvalType::ControlScalarType> tControlWS("design variables", tNumCells, PhysicsT::mNumNodesPerCell);
    Kokkos::deep_copy(tControlWS, 1.0);

    // 2.3 SET GLOBAL STATE
    auto tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    Plato::ScalarVector tState("state", tSpaceDim * tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tState(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal; // disp_x
        tState(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal; // disp_y
        tState(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal; // press
    }, "set global state");
    Plato::ScalarMultiVectorT<EvalType::StateScalarType> tStateWS("current global state", tNumCells, PhysicsT::mNumDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // 2.4 SET PROJECTED PRESSURE GRADIENT
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    Plato::ScalarMultiVectorT<EvalType::NodeStateScalarType> tProjPressGradWS("projected pressure grad", tNumCells, PhysicsT::mNumNodeStatePerCell);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNodeIndex=0; tNodeIndex< tNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDimIndex=0; tDimIndex< tSpaceDim; tDimIndex++)
            {
                tProjPressGradWS(aCellOrdinal, tNodeIndex*tSpaceDim+tDimIndex) = (4e-7)*(tNodeIndex+1)*(tDimIndex+1)*(aCellOrdinal+1);
            }
        }
    }, "set projected pressure grad");

    // 3. CALL FUNCTION
    auto tPenaltyParams = tPDEInputs->sublist("Elliptic").sublist("Penalty Function");
    Plato::StabilizedElastostaticResidual<EvalType, Plato::MSIMP> tComputeResidual(*tMesh, tMeshSets, tDataMap, *tPDEInputs, tPenaltyParams);
    Plato::ScalarMultiVectorT<EvalType::ResultScalarType> tResidualWS("residual", tNumCells, PhysicsT::mNumDofsPerCell);
    tComputeResidual.evaluate(tStateWS, tProjPressGradWS, tControlWS, tConfigWS, tResidualWS);

    // 5. TEST RESULTS
    Plato::print_array_2D(tResidualWS, "residual");
}


}
// namespace StabilizedMechanicsTests
