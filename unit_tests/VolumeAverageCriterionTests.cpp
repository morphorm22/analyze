#include "util/PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>
#include "Analyze_Diagnostics.hpp"
#include "elliptic/Problem.hpp"

#include "Mechanics.hpp"

namespace VolumeAverageCriterionTests
{

TEUCHOS_UNIT_TEST(VolumeAverageCriterionTests, VolumeAverageVonMisesStressAxial_3D)
{
    const bool tOutputData = false;
    const Plato::Scalar tBoxWidth = 2.0;
    const Plato::OrdinalType tNumElemX = 1;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tBoxWidth, tNumElemX, tBoxWidth, tNumElemY, tBoxWidth, tNumElemZ);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Mechanical'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "      <Parameter name='Iterations' type='int' value='500'/>                              \n"
      "      <Parameter name='Tolerance' type='double' value='1.0e-10'/>                        \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                    \n"
      "        <Parameter  name='Density' type='double' value='1'/>                             \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.2'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e4'/>                  \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <Parameter name='Plottable' type='Array(string)' value='{Vonmises}'/>                \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='VolAvgMisesStress'>                                             \n"
      "      <Parameter name='Type' type='string' value='Volume Average Criterion'/>            \n"
      "      <Parameter name='Local Measure' type='string' value='VonMises'/>                   \n"
      "      <Parameter name='Function' type='string' value='2.0*x'/>                           \n"
      "      <ParameterList name='Penalty Function'>                                            \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>                              \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
      "        <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                   \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Natural Boundary Conditions'>                                   \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='y-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='z-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied X Displacement Boundary Condition'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                   \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x+'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='0.2'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    using PhysicsT = Plato::Mechanics<Plato::Tet4>;

    Plato::Elliptic::Problem<PhysicsT> tProblem(tMesh, *tParamList, tMachine);
    tProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->NumNodes();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);

    // 5. Test results
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::string tCriterionName("VolAvgMisesStress");
    auto tCriterionValue = tProblem.criterionValue(tControls, tCriterionName);
    TEST_FLOATING_EQUALITY(tCriterionValue, 1000.0, tTolerance);

    auto tCriterionGrad = tProblem.criterionGradient(tControls, tSolution, tCriterionName);
    std::vector<Plato::Scalar> tGold = {
      7.50000e+02, 1.87500e+02, 1.87500e+02, 1.25000e+02,
      3.75000e+02, 3.12500e+02, 3.12500e+02, 7.50000e+02};
    auto tHostGrad = Kokkos::create_mirror(tCriterionGrad);
    Kokkos::deep_copy(tHostGrad, tCriterionGrad);
    TEST_ASSERT( tHostGrad.size() == static_cast<Plato::OrdinalType>(tGold.size() ));
    for(Plato::OrdinalType tIndex = 0; tIndex < tHostGrad.size(); tIndex++)
    {
        //printf("%12.5e\n", tHostGrad(tIndex));
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }

    // 6. Output Data
    if (tOutputData)
    {
        tProblem.output("VolumeAverageVonMisesStressAxial_3D");
    }
}

TEUCHOS_UNIT_TEST(VolumeAverageCriterionTests, VolumeAverageVonMisesStressShear_3D)
{
    const bool tOutputData = false;
    const Plato::Scalar tBoxWidth = 2.0;
    const Plato::OrdinalType tNumElemX = 1;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tBoxWidth, tNumElemX, tBoxWidth, tNumElemY, tBoxWidth, tNumElemZ);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Mechanical'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "      <Parameter name='Iterations' type='int' value='500'/>                              \n"
      "      <Parameter name='Tolerance' type='double' value='1.0e-10'/>                        \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                    \n"
      "        <Parameter  name='Density' type='double' value='1'/>                             \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.2'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e4'/>                  \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='VolAvgMisesStress'>                                             \n"
      "      <Parameter name='Type' type='string' value='Volume Average Criterion'/>            \n"
      "      <Parameter name='Local Measure' type='string' value='VonMises'/>                   \n"
      "      <Parameter name='Function' type='string' value='2.0*x'/>                           \n"
      "      <ParameterList name='Penalty Function'>                                            \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>                              \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
      "        <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                   \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Natural Boundary Conditions'>                                   \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X0 Fixed Y'>                                                  \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y0 Fixed X'>                                                  \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='y-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='X1 Applied Y'>                                                \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                   \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x+'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='0.2'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y1 Applied X'>                                                \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                   \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='y+'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='0.2'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='z-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    using PhysicsT = Plato::Mechanics<Plato::Tet4>;

    Plato::Elliptic::Problem<PhysicsT> tProblem(tMesh, *tParamList, tMachine);
    tProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    Plato::Scalar tDensity = 0.9;
    auto tNumVertices = tMesh->NumNodes();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(tDensity, tControls);
    auto tSolution = tProblem.solution(tControls);


    // 5. Test results
    Plato::Scalar tSimpPenalty = 1.0e-8 + (1.0 - 1.0e-8) * std::pow(tDensity, 3);
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::string tCriterionName("VolAvgMisesStress");
    auto tCriterionValue = tProblem.criterionValue(tControls, tCriterionName);
    TEST_FLOATING_EQUALITY(tCriterionValue, tSimpPenalty*1443.3756727, tTolerance);

    auto tCriterionGrad = tProblem.criterionGradient(tControls, tSolution, tCriterionName);
    std::vector<Plato::Scalar> tGold = {
      8.76851e+02, 2.19213e+02, 2.19213e+02, 1.46142e+02,
      4.38425e+02, 3.65354e+02, 3.65354e+02, 8.76851e+02};
    auto tHostGrad = Kokkos::create_mirror(tCriterionGrad);
    Kokkos::deep_copy(tHostGrad, tCriterionGrad);
    TEST_ASSERT( tHostGrad.size() == static_cast<Plato::OrdinalType>(tGold.size() ));
    for(Plato::OrdinalType tIndex = 0; tIndex < tHostGrad.size(); tIndex++)
    {
        //printf("%12.5e\n", tHostGrad(tIndex));
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }

    // 6. Output Data
    if (tOutputData)
    {
        tProblem.output("VolumeAverageVonMisesStressShear_3D");
    }
}

TEUCHOS_UNIT_TEST(VolumeAverageCriterionTests, VolumeAverageVonMisesStressGradientZ_3D)
{
    const Plato::Scalar tBoxWidth = 2.0;
    const Plato::OrdinalType tNumElemX = 5;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tBoxWidth, tNumElemX, tBoxWidth, tNumElemY, tBoxWidth, tNumElemZ);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Mechanical'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "      <Parameter name='Iterations' type='int' value='500'/>                              \n"
      "      <Parameter name='Tolerance' type='double' value='1.0e-10'/>                        \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                    \n"
      "        <Parameter  name='Density' type='double' value='1'/>                             \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.2'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e4'/>                  \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='VolAvgMisesStress'>                                             \n"
      "      <Parameter name='Type' type='string' value='Volume Average Criterion'/>            \n"
      "      <Parameter name='Local Measure' type='string' value='VonMises'/>                   \n"
      "      <Parameter name='Function' type='string' value='exp(10.0*y)'/>                     \n"
      "      <ParameterList name='Penalty Function'>                                            \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>                              \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
      "        <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                   \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Natural Boundary Conditions'>                                   \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='z-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Y Displacement Boundary Condition'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                   \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x+'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='0.1'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    using PhysicsT = Plato::Mechanics<Plato::Tet4>;

    Plato::Elliptic::Problem<PhysicsT> tProblem(tMesh, *tParamList, tMachine);
    tProblem.readEssentialBoundaryConditions(*tParamList);

    std::string tCriterionName("VolAvgMisesStress");
    auto tApproxError = Plato::test_criterion_grad_wrt_control(tProblem, tMesh, tCriterionName);
    const Plato::Scalar tUpperBound = 1e-6;
    TEST_ASSERT(tApproxError < tUpperBound);
}

TEUCHOS_UNIT_TEST(VolumeAverageCriterionTests, VolumeAverageTensileEnergyAxial_3D)
{
    const bool tOutputData = false;
    const Plato::Scalar tBoxWidth = 2.0;
    const Plato::OrdinalType tNumElemX = 1;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tBoxWidth, tNumElemX, tBoxWidth, tNumElemY, tBoxWidth, tNumElemZ);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Mechanical'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "      <Parameter name='Iterations' type='int' value='500'/>                              \n"
      "      <Parameter name='Tolerance' type='double' value='1.0e-10'/>                        \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                    \n"
      "        <Parameter  name='Density' type='double' value='1'/>                             \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.2'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e4'/>                  \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='VolAvgTensileEnergy'>                                           \n"
      "      <Parameter name='Type' type='string' value='Volume Average Criterion'/>            \n"
      "      <Parameter name='Local Measure' type='string' value='TensileEnergyDensity'/>       \n"
      "      <Parameter name='Function' type='string' value='2.0*x'/>                           \n"
      "      <ParameterList name='Penalty Function'>                                            \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>                              \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
      "        <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                   \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Natural Boundary Conditions'>                                   \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='y-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='z-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied X Displacement Boundary Condition'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                   \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x+'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='0.2'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    using PhysicsT = Plato::Mechanics<Plato::Tet4>;

    Plato::Elliptic::Problem<PhysicsT> tProblem(tMesh, *tParamList, tMachine);
    tProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->NumNodes();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);

    // 5. Test results
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::string tCriterionName("VolAvgTensileEnergy");
    auto tCriterionValue = tProblem.criterionValue(tControls, tCriterionName);
    TEST_FLOATING_EQUALITY(tCriterionValue, 46.666666666666, tTolerance);

    auto tCriterionGrad = tProblem.criterionGradient(tControls, tSolution, tCriterionName);
    std::vector<Plato::Scalar> tGold = {
      3.50000e+01, 8.75000e+00, 8.75000e+00, 5.83333e+00,
      1.75000e+01, 1.45833e+01, 1.45833e+01, 3.50000e+01};
    auto tHostGrad = Kokkos::create_mirror(tCriterionGrad);
    Kokkos::deep_copy(tHostGrad, tCriterionGrad);
    TEST_ASSERT( tHostGrad.size() == static_cast<Plato::OrdinalType>(tGold.size() ));
    for(Plato::OrdinalType tIndex = 0; tIndex < tHostGrad.size(); tIndex++)
    {
        //printf("%12.5e\n", tHostGrad(tIndex));
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }

    // 6. Output Data
    if (tOutputData)
    {
        tProblem.output("VolumeAverageTensileEnergyAxial_3D");
    }
}

TEUCHOS_UNIT_TEST(VolumeAverageCriterionTests, VolumeAverageTensileEnergyShear_3D)
{
    const bool tOutputData = false;
    const Plato::Scalar tBoxWidth = 2.0;
    const Plato::OrdinalType tNumElemX = 1;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tBoxWidth, tNumElemX, tBoxWidth, tNumElemY, tBoxWidth, tNumElemZ);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Mechanical'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "      <Parameter name='Iterations' type='int' value='500'/>                              \n"
      "      <Parameter name='Tolerance' type='double' value='1.0e-10'/>                        \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                    \n"
      "        <Parameter  name='Density' type='double' value='1'/>                             \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.2'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e4'/>                  \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='VolAvgTensileEnergy'>                                           \n"
      "      <Parameter name='Type' type='string' value='Volume Average Criterion'/>            \n"
      "      <Parameter name='Local Measure' type='string' value='TensileEnergyDensity'/>       \n"
      "      <Parameter name='Function' type='string' value='2.0*x'/>                           \n"
      "      <ParameterList name='Penalty Function'>                                            \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>                              \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
      "        <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                   \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Natural Boundary Conditions'>                                   \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X0 Fixed Y'>                                                  \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y0 Fixed X'>                                                  \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='y-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='X1 Applied Y'>                                                \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                   \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x+'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='0.2'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y1 Applied X'>                                                \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                   \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='y+'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='0.2'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='z-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    using PhysicsT = Plato::Mechanics<Plato::Tet4>;

    Plato::Elliptic::Problem<PhysicsT> tProblem(tMesh, *tParamList, tMachine);
    tProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    Plato::Scalar tDensity = 0.9;
    auto tNumVertices = tMesh->NumNodes();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(tDensity, tControls);
    auto tSolution = tProblem.solution(tControls);


    // 5. Test results
    Plato::Scalar tSimpPenalty = 1.0e-8 + (1.0 - 1.0e-8) * std::pow(tDensity, 3);
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::string tCriterionName("VolAvgTensileEnergy");
    auto tCriterionValue = tProblem.criterionValue(tControls, tCriterionName);
    TEST_FLOATING_EQUALITY(tCriterionValue, tSimpPenalty*41.6666666666, tTolerance);

    auto tCriterionGrad = tProblem.criterionGradient(tControls, tSolution, tCriterionName);
    std::vector<Plato::Scalar> tGold = {
      2.53125e+01, 6.32812e+00, 6.32812e+00, 4.21875e+00,
      1.26562e+01, 1.05469e+01, 1.05469e+01, 2.53125e+01};
    auto tHostGrad = Kokkos::create_mirror(tCriterionGrad);
    Kokkos::deep_copy(tHostGrad, tCriterionGrad);
    TEST_ASSERT( tHostGrad.size() == static_cast<Plato::OrdinalType>(tGold.size() ));
    for(Plato::OrdinalType tIndex = 0; tIndex < tHostGrad.size(); tIndex++)
    {
        //printf("%12.5e\n", tHostGrad(tIndex));
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }

    // 6. Output Data
    if (tOutputData)
    {
        tProblem.output("VolumeAverageTensileEnergyShear_3D");
    }
}

TEUCHOS_UNIT_TEST(VolumeAverageCriterionTests, VolumeAverageTensileEnergyGradientZ_3D)
{
    const Plato::Scalar tBoxWidth = 2.0;
    const Plato::OrdinalType tNumElemX = 5;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tBoxWidth, tNumElemX, tBoxWidth, tNumElemY, tBoxWidth, tNumElemZ);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Mechanical'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "      <Parameter name='Iterations' type='int' value='500'/>                              \n"
      "      <Parameter name='Tolerance' type='double' value='1.0e-10'/>                        \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                    \n"
      "        <Parameter  name='Density' type='double' value='1'/>                             \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.2'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e4'/>                  \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='VolAvgTensileEnergy'>                                           \n"
      "      <Parameter name='Type' type='string' value='Volume Average Criterion'/>            \n"
      "      <Parameter name='Local Measure' type='string' value='TensileEnergyDensity'/>       \n"
      "      <Parameter name='Function' type='string' value='exp(10.0*y)'/>                     \n"
      "      <ParameterList name='Penalty Function'>                                            \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>                              \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
      "        <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                   \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Natural Boundary Conditions'>                                   \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='z-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Y Displacement Boundary Condition'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                   \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x+'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='1.0'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    using PhysicsT = Plato::Mechanics<Plato::Tet4>;

    Plato::Elliptic::Problem<PhysicsT> tProblem(tMesh, *tParamList, tMachine);
    tProblem.readEssentialBoundaryConditions(*tParamList);

    std::string tCriterionName("VolAvgTensileEnergy");
    auto tApproxError = Plato::test_criterion_grad_wrt_control(tProblem, tMesh, tCriterionName);
    const Plato::Scalar tUpperBound = 1e-6;
    TEST_ASSERT(tApproxError < tUpperBound);
}

} // namespace VolumeAverageCriterionTests
