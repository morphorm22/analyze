#include "util/PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "Tri3.hpp"
#include "solver/ParallelComm.hpp"
#include "elliptic/mechanical/linear/Mechanics.hpp"
#include "elliptic/criterioneval/CriterionEvaluatorSolutionFunction.hpp"


/******************************************************************************/
/*!
  \brief Compute value and gradients (wrt state, control, and configuration) of
         Solution criterion in 2D
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, Solution2D )
{
  // create test mesh
  //
  constexpr int meshWidth=4;
  constexpr int spaceDim=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", 1.0, meshWidth, 1.0, meshWidth);


  // create mesh based density from host data
  //
  Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector z("density", tNumVerts);
  Kokkos::deep_copy(z, 1.0);


  // create displacement field, u = x
  //
  auto tCoords = tMesh->Coordinates();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, tNumVerts*spaceDim);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, tNumVerts), KOKKOS_LAMBDA(int aNodeOrdinal)
  {
    U(0, aNodeOrdinal*spaceDim + 0) = tCoords[aNodeOrdinal*spaceDim + 0];
    U(0, aNodeOrdinal*spaceDim + 1) = tCoords[aNodeOrdinal*spaceDim + 1];
  }, "set disp");


  // setup the problem
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>         \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Fancy Material'/> \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                     \n"
    "    <ParameterList name='Fancy Material'>                                     \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Criteria'>                                           \n"
    "    <ParameterList name='Displacement'>                                     \n"
    "      <Parameter name='Type' type='string' value='Solution'/>               \n"
    "      <Parameter name='Normal' type='Array(double)' value='{1.0,1.0}'/>     \n"
    "      <Parameter name='Domain' type='string' value='y-'/>                   \n"
    "      <Parameter name='Magnitude' type='bool' value='false'/>               \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  // create objective
  //
  std::string tMyFunction("Displacement");
  Plato::Elliptic::CriterionEvaluatorSolutionFunction<::Plato::Elliptic::Linear::Mechanics<Plato::Tri3>>
    scalarFunction(tSpatialModel, tDataMap, *tParamList, tMyFunction);


  // compute and test objective value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto value = scalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 0.5;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);

  // compute and test objective gradient wrt state, u
  //
  auto grad_u = scalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  TEST_FLOATING_EQUALITY(grad_u_Host(0),   1.0 / 5, 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(1),   1.0 / 5, 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(10),  1.0 / 5, 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(11),  1.0 / 5, 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(20),  1.0 / 5, 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(21),  1.0 / 5, 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(30),  1.0 / 5, 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(31),  1.0 / 5, 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(40),  1.0 / 5, 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(41),  1.0 / 5, 1e-15);


  // compute and test objective gradient wrt control, z
  //
  auto grad_z = scalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  for(int iNode=0; iNode<int(grad_z_Host.size()); iNode++){
    TEST_ASSERT(grad_z_Host[iNode] == 0.0);
  }

  // compute and test objective gradient wrt node position, x
  //
  auto grad_x = scalarFunction.gradient_x(tSolution, z);

  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  for(int iNode=0; iNode<int(grad_x_Host.size()); iNode++){
    TEST_ASSERT(grad_x_Host[iNode] == 0.0);
  }
}
/******************************************************************************/
/*!
  \brief Compute value and gradients (wrt state, control, and configuration) of
         Solution magnitude criterion in 3D
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, Solution2D_Mag )
{
  // create test mesh
  //
  constexpr int meshWidth=4;
  constexpr int spaceDim=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", 1.0, meshWidth, 1.0, meshWidth);


  // create mesh based density from host data
  //
  Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector z("density", tNumVerts);
  Kokkos::deep_copy(z, 1.0);


  // create displacement field, u = x
  //
  auto tCoords = tMesh->Coordinates();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, tNumVerts*spaceDim);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, tNumVerts), KOKKOS_LAMBDA(int aNodeOrdinal)
  {
    U(0, aNodeOrdinal*spaceDim + 0) = tCoords[aNodeOrdinal*spaceDim + 0];
    U(0, aNodeOrdinal*spaceDim + 1) = tCoords[aNodeOrdinal*spaceDim + 1];
  }, "set disp");


  // setup the problem
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>         \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Fancy Material'/> \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Fancy Material'>                                     \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Displacement'>                                       \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Solution'/> \n"
    "      <Parameter name='Normal' type='Array(double)' value='{1.0,1.0}'/>       \n"
    "      <Parameter name='Domain' type='string' value='y-'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  // create objective
  //
  std::string tMyFunction("Displacement");
  Plato::Elliptic::CriterionEvaluatorSolutionFunction<::Plato::Elliptic::Linear::Mechanics<Plato::Tri3>>
    scalarFunction(tSpatialModel, tDataMap, *tParamList, tMyFunction);


  // compute and test objective value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto value = scalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 0.5;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);

  // compute and test objective gradient wrt state, u
  //
  auto grad_u = scalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  TEST_FLOATING_EQUALITY(grad_u_Host(0),   1.0 * (1.0*0.25) / (0.25 * 5), 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(1),   1.0 * (1.0*0.25) / (0.25 * 5), 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(10),  1.0 * (1.0*0.50) / (0.50 * 5), 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(11),  1.0 * (1.0*0.50) / (0.50 * 5), 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(20),  1.0 * (1.0*0.75) / (0.75 * 5), 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(21),  1.0 * (1.0*0.75) / (0.75 * 5), 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(30),  1.0 * (1.0*1.00) / (1.00 * 5), 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(31),  1.0 * (1.0*1.00) / (1.00 * 5), 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(40),  1.0 * (1.0*1.00) / (1.00 * 5), 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(41),  1.0 * (1.0*1.00) / (1.00 * 5), 1e-15);


  // compute and test objective gradient wrt control, z
  //
  auto grad_z = scalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  for(int iNode=0; iNode<int(grad_z_Host.size()); iNode++){
    TEST_ASSERT(grad_z_Host[iNode] == 0.0);
  }

  // compute and test objective gradient wrt node position, x
  //
  auto grad_x = scalarFunction.gradient_x(tSolution, z);

  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  for(int iNode=0; iNode<int(grad_x_Host.size()); iNode++){
    TEST_ASSERT(grad_x_Host[iNode] == 0.0);
  }
}

