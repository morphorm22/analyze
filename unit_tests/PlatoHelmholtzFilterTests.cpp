#include "util/PlatoTestHelpers.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "helmholtz/Helmholtz.hpp"
#include "helmholtz/VectorFunction.hpp"
#include "helmholtz/HelmholtzElement.hpp"
#include "helmholtz/Problem.hpp"

#include "Tri3.hpp"
#include "Tet4.hpp"
#include "Tet10.hpp"
#include "Hex8.hpp"
#include "Hex27.hpp"
#include "BLAS1.hpp"
#include "PlatoMathHelpers.hpp"
#include "alg/PlatoSolverFactory.hpp"

#ifdef HAVE_AMGX
#include <alg/AmgXSparseLinearProblem.hpp>
#endif

#include <memory>

/******************************************************************************/
/*!
  \brief test parsing of length scale parameter

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(HelmholtzFilterTests, LengthScaleKeywordError)
{
  // create test mesh
  //
  constexpr int meshWidth=20;
  auto tMesh = Plato::TestHelpers::get_box_mesh("tri3", meshWidth);

  using PhysicsType = ::Plato::HelmholtzFilter<Plato::Tri3>;

  // set parameters
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Helmholtz Filter'/> \n"
    "  <ParameterList name='FakeParameters'>                                    \n"
    "    <Parameter name='LengthScale' type='double' value='0.10'/>              \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                        \n"
  );

  // create PDE
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  TEST_THROW(Plato::Helmholtz::VectorFunction<PhysicsType> vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint")), std::runtime_error);
}

/******************************************************************************/
/*!
  \brief test parsing Helmholtz problem

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(HelmholtzFilterTests, HelmholtzProblemError)
{
  // create test mesh
  //
  constexpr int meshWidth=4;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);
  
  // create mesh based density
  //
  using PhysicsType = ::Plato::HelmholtzFilter<Plato::Tet4>;
  using ElementType = typename PhysicsType::ElementType;
  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);
  Plato::ScalarVector testControl("test density", tNumDofs);
  Kokkos::deep_copy(testControl, 1.0);

  // set parameters
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Helmholtz Filter'/> \n"
    "  <Parameter name='Physics' type='string' value='Helmholtz Filter'/> \n"
    "  <ParameterList name='Parameters'>                                    \n"
    "    <Parameter name='Length Scale' type='double' value='0.10'/>              \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                        \n"
  );

  // get machine
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  // construct problem
  auto tProblem = Plato::Helmholtz::Problem<PhysicsType>(tMesh, *tParamList, tMachine);

  // perform necessary operations
  auto tSolution = tProblem.solution(control);
  Plato::ScalarVector tFilteredControl = Kokkos::subview(tSolution.get("State"), 0, Kokkos::ALL());
  Kokkos::deep_copy(testControl, tFilteredControl);

  std::string tDummyString = "Helmholtz gradient";
  Plato::ScalarVector tGradient = tProblem.criterionGradient(control,tDummyString);

}

/******************************************************************************/
/*!
  \brief homogeneous Helmholtz problem

  Construct a 2D Helmholtz filter problem with uniform unfiltered density 
  and solve. Test passes if filtered density values match unfiltered.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HelmholtzFilterTests, Helmholtz2DUniformFieldTest )
{
  // create test mesh
  //
  constexpr int meshWidth=8;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using PhysicsType = ::Plato::HelmholtzFilter<Plato::Tri3>;
  using ElementType = typename PhysicsType::ElementType;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);

  // create PDE
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                         \n"
    "  <ParameterList name='Spatial Model'>                                       \n"
    "    <ParameterList name='Domains'>                                           \n"
    "      <ParameterList name='Design Volume'>                                   \n"
    "        <Parameter name='Element Block' type='string' value='body'/>         \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                       \n"
    "    </ParameterList>                                                         \n"
    "  </ParameterList>                                                           \n"
    "  <Parameter name='PDE Constraint' type='string' value='Helmholtz Filter'/>  \n"
    "  <ParameterList name='Parameters'>                                          \n"
    "    <Parameter name='Length Scale' type='double' value='0.10'/>              \n"
    "  </ParameterList>                                                           \n"
    "  <ParameterList name='Linear Solver'>                                       \n"
    "    <Parameter name='Solver Stack' type='string' value='Epetra'/>            \n"
    "    <Parameter name='Display Iterations' type='int' value='1'/>              \n"
    "    <Parameter name='Iterations' type='int' value='50'/>                     \n"
    "    <Parameter name='Tolerance' type='double' value='1e-14'/>                \n"
    "  </ParameterList>                                                           \n"
    "</ParameterList>                                                             \n"
  );

  // create PDE
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);
  Plato::Helmholtz::VectorFunction<PhysicsType>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);

  // create solver
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                              \n"
    "  <Parameter name='Solver Stack' type='string' value='Epetra'/>   \n"
    "  <Parameter name='Display Iterations' type='int' value='1'/>     \n"
    "  <Parameter name='Iterations' type='int' value='50'/>            \n"
    "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "</ParameterList>                                                  \n"
  );
  Plato::SolverFactory tSolverFactory(*tSolverParams);

  auto tSolver = tSolverFactory.create(tMesh->NumNodes(), tMachine, tNumDofsPerNode);
  
  // solve linear system
  //
  tSolver->solve(*jacobian, state, residual);

  // create mirror view of displacement solution
  //
  Plato::ScalarVector statesView("State",tNumDofs);
  Kokkos::deep_copy(statesView, state);

  auto stateView_host = Kokkos::create_mirror_view(statesView);
  Kokkos::deep_copy(stateView_host, statesView);

  // test that filtered density field is still 1
  //
  for(int iDof=0; iDof<tNumDofs; iDof++){
    TEST_FLOATING_EQUALITY(stateView_host(iDof), 1.0, 1.0e-14);
  }
}

/******************************************************************************/
/*!
  \brief homogeneous Helmholtz problem

  Construct a Tet4 Helmholtz filter problem with uniform unfiltered density 
  and solve. Test passes if filtered density values match unfiltered.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HelmholtzFilterTests, HelmholtzUniformFieldTest_Tet4 )
{
  // create test mesh
  //
  constexpr int meshWidth=4;
  auto tMesh = Plato::TestHelpers::get_box_mesh("Tet4", meshWidth);

  using PhysicsType = ::Plato::HelmholtzFilter<Plato::Tet4>;
  using ElementType = typename PhysicsType::ElementType;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);

  // create PDE
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                         \n"
    "  <ParameterList name='Spatial Model'>                                       \n"
    "    <ParameterList name='Domains'>                                           \n"
    "      <ParameterList name='Design Volume'>                                   \n"
    "        <Parameter name='Element Block' type='string' value='body'/>         \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                       \n"
    "    </ParameterList>                                                         \n"
    "  </ParameterList>                                                           \n"
    "  <Parameter name='PDE Constraint' type='string' value='Helmholtz Filter'/>  \n"
    "  <ParameterList name='Parameters'>                                          \n"
    "    <Parameter name='Length Scale' type='double' value='0.10'/>              \n"
    "  </ParameterList>                                                           \n"
    "  <ParameterList name='Linear Solver'>                                       \n"
    "    <Parameter name='Solver Stack' type='string' value='Epetra'/>            \n"
    "    <Parameter name='Display Iterations' type='int' value='1'/>              \n"
    "    <Parameter name='Iterations' type='int' value='50'/>                     \n"
    "    <Parameter name='Tolerance' type='double' value='1e-14'/>                \n"
    "  </ParameterList>                                                           \n"
    "</ParameterList>                                                             \n"
  );

  // create PDE
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);
  Plato::Helmholtz::VectorFunction<PhysicsType>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);

  // create solver
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                              \n"
    "  <Parameter name='Solver Stack' type='string' value='Epetra'/>   \n"
    "  <Parameter name='Display Iterations' type='int' value='1'/>     \n"
    "  <Parameter name='Iterations' type='int' value='50'/>            \n"
    "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "</ParameterList>                                                  \n"
  );
  Plato::SolverFactory tSolverFactory(*tSolverParams);

  auto tSolver = tSolverFactory.create(tMesh->NumNodes(), tMachine, tNumDofsPerNode);
  
  // solve linear system
  //
  tSolver->solve(*jacobian, state, residual);

  // create mirror view of displacement solution
  //
  Plato::ScalarVector statesView("State",tNumDofs);
  Kokkos::deep_copy(statesView, state);

  auto stateView_host = Kokkos::create_mirror_view(statesView);
  Kokkos::deep_copy(stateView_host, statesView);

  // test that filtered density field is still 1
  //
  for(int iDof=0; iDof<tNumDofs; iDof++){
    TEST_FLOATING_EQUALITY(stateView_host(iDof), 1.0, 1.0e-14);
  }
}

/******************************************************************************/
/*!
  \brief homogeneous Helmholtz problem

  Construct a Hex8 Helmholtz filter problem with uniform unfiltered density 
  and solve. Test passes if filtered density values match unfiltered.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HelmholtzFilterTests, HelmholtzUniformFieldTest_Hex8 )
{
  // create test mesh
  //
  constexpr int meshWidth=4;
  auto tMesh = Plato::TestHelpers::get_box_mesh("Hex8", meshWidth);

  using PhysicsType = ::Plato::HelmholtzFilter<Plato::Hex8>;
  using ElementType = typename PhysicsType::ElementType;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);

  // create PDE
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                         \n"
    "  <ParameterList name='Spatial Model'>                                       \n"
    "    <ParameterList name='Domains'>                                           \n"
    "      <ParameterList name='Design Volume'>                                   \n"
    "        <Parameter name='Element Block' type='string' value='body'/>         \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                       \n"
    "    </ParameterList>                                                         \n"
    "  </ParameterList>                                                           \n"
    "  <Parameter name='PDE Constraint' type='string' value='Helmholtz Filter'/>  \n"
    "  <ParameterList name='Parameters'>                                          \n"
    "    <Parameter name='Length Scale' type='double' value='0.10'/>              \n"
    "  </ParameterList>                                                           \n"
    "  <ParameterList name='Linear Solver'>                                       \n"
    "    <Parameter name='Solver Stack' type='string' value='Epetra'/>            \n"
    "    <Parameter name='Display Iterations' type='int' value='1'/>              \n"
    "    <Parameter name='Iterations' type='int' value='50'/>                     \n"
    "    <Parameter name='Tolerance' type='double' value='1e-14'/>                \n"
    "  </ParameterList>                                                           \n"
    "</ParameterList>                                                             \n"
  );

  // create PDE
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);
  Plato::Helmholtz::VectorFunction<PhysicsType>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);

  // create solver
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                              \n"
    "  <Parameter name='Solver Stack' type='string' value='Epetra'/>   \n"
    "  <Parameter name='Display Iterations' type='int' value='1'/>     \n"
    "  <Parameter name='Iterations' type='int' value='50'/>            \n"
    "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "</ParameterList>                                                  \n"
  );
  Plato::SolverFactory tSolverFactory(*tSolverParams);

  auto tSolver = tSolverFactory.create(tMesh->NumNodes(), tMachine, tNumDofsPerNode);
  
  // solve linear system
  //
  tSolver->solve(*jacobian, state, residual);

  // create mirror view of displacement solution
  //
  Plato::ScalarVector statesView("State",tNumDofs);
  Kokkos::deep_copy(statesView, state);

  auto stateView_host = Kokkos::create_mirror_view(statesView);
  Kokkos::deep_copy(stateView_host, statesView);

  // test that filtered density field is still 1
  //
  for(int iDof=0; iDof<tNumDofs; iDof++){
    TEST_FLOATING_EQUALITY(stateView_host(iDof), 1.0, 1.0e-14);
  }
}

/******************************************************************************/
/*!
  \brief homogeneous Helmholtz problem

  Construct a Tet10 Helmholtz filter problem with uniform unfiltered density 
  and solve. Test passes if filtered density values match unfiltered.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HelmholtzFilterTests, HelmholtzUniformFieldTest_Tet10 )
{
  // create test mesh
  //
  constexpr int meshWidth=4;
  auto tMesh = Plato::TestHelpers::get_box_mesh("Tet10", meshWidth);

  using PhysicsType = ::Plato::HelmholtzFilter<Plato::Tet10>;
  using ElementType = typename PhysicsType::ElementType;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);

  // create PDE
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                         \n"
    "  <ParameterList name='Spatial Model'>                                       \n"
    "    <ParameterList name='Domains'>                                           \n"
    "      <ParameterList name='Design Volume'>                                   \n"
    "        <Parameter name='Element Block' type='string' value='body'/>         \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                       \n"
    "    </ParameterList>                                                         \n"
    "  </ParameterList>                                                           \n"
    "  <Parameter name='PDE Constraint' type='string' value='Helmholtz Filter'/>  \n"
    "  <ParameterList name='Parameters'>                                          \n"
    "    <Parameter name='Length Scale' type='double' value='0.10'/>              \n"
    "  </ParameterList>                                                           \n"
    "  <ParameterList name='Linear Solver'>                                       \n"
    "    <Parameter name='Solver Stack' type='string' value='Epetra'/>            \n"
    "    <Parameter name='Display Iterations' type='int' value='1'/>              \n"
    "    <Parameter name='Iterations' type='int' value='50'/>                     \n"
    "    <Parameter name='Tolerance' type='double' value='1e-14'/>                \n"
    "  </ParameterList>                                                           \n"
    "</ParameterList>                                                             \n"
  );

  // create PDE
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);
  Plato::Helmholtz::VectorFunction<PhysicsType>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);

  // create solver
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                              \n"
    "  <Parameter name='Solver Stack' type='string' value='Epetra'/>   \n"
    "  <Parameter name='Display Iterations' type='int' value='1'/>     \n"
    "  <Parameter name='Iterations' type='int' value='50'/>            \n"
    "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "</ParameterList>                                                  \n"
  );
  Plato::SolverFactory tSolverFactory(*tSolverParams);

  auto tSolver = tSolverFactory.create(tMesh->NumNodes(), tMachine, tNumDofsPerNode);
  
  // solve linear system
  //
  tSolver->solve(*jacobian, state, residual);

  // create mirror view of displacement solution
  //
  Plato::ScalarVector statesView("State",tNumDofs);
  Kokkos::deep_copy(statesView, state);

  auto stateView_host = Kokkos::create_mirror_view(statesView);
  Kokkos::deep_copy(stateView_host, statesView);

  // test that filtered density field is still 1
  //
  for(int iDof=0; iDof<tNumDofs; iDof++){
    TEST_FLOATING_EQUALITY(stateView_host(iDof), 1.0, 1.0e-13);
  }
}

/******************************************************************************/
/*!
  \brief homogeneous Helmholtz problem

  Construct a Hex27 Helmholtz filter problem with uniform unfiltered density 
  and solve. Test passes if filtered density values match unfiltered.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HelmholtzFilterTests, HelmholtzUniformFieldTest_Hex27 )
{
  // create test mesh
  //
  constexpr int meshWidth=4;
  auto tMesh = Plato::TestHelpers::get_box_mesh("Hex27", meshWidth);

  using PhysicsType = ::Plato::HelmholtzFilter<Plato::Hex27>;
  using ElementType = typename PhysicsType::ElementType;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);

  // create PDE
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                         \n"
    "  <ParameterList name='Spatial Model'>                                       \n"
    "    <ParameterList name='Domains'>                                           \n"
    "      <ParameterList name='Design Volume'>                                   \n"
    "        <Parameter name='Element Block' type='string' value='body'/>         \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                       \n"
    "    </ParameterList>                                                         \n"
    "  </ParameterList>                                                           \n"
    "  <Parameter name='PDE Constraint' type='string' value='Helmholtz Filter'/>  \n"
    "  <ParameterList name='Parameters'>                                          \n"
    "    <Parameter name='Length Scale' type='double' value='0.10'/>              \n"
    "  </ParameterList>                                                           \n"
    "  <ParameterList name='Linear Solver'>                                       \n"
    "    <Parameter name='Solver Stack' type='string' value='Epetra'/>            \n"
    "    <Parameter name='Display Iterations' type='int' value='1'/>              \n"
    "    <Parameter name='Iterations' type='int' value='50'/>                     \n"
    "    <Parameter name='Tolerance' type='double' value='1e-14'/>                \n"
    "  </ParameterList>                                                           \n"
    "</ParameterList>                                                             \n"
  );

  // create PDE
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);
  Plato::Helmholtz::VectorFunction<PhysicsType>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);

  // create solver
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                              \n"
    "  <Parameter name='Solver Stack' type='string' value='Epetra'/>   \n"
    "  <Parameter name='Display Iterations' type='int' value='1'/>     \n"
    "  <Parameter name='Iterations' type='int' value='50'/>            \n"
    "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "</ParameterList>                                                  \n"
  );
  Plato::SolverFactory tSolverFactory(*tSolverParams);

  auto tSolver = tSolverFactory.create(tMesh->NumNodes(), tMachine, tNumDofsPerNode);
  
  // solve linear system
  //
  tSolver->solve(*jacobian, state, residual);

  // create mirror view of displacement solution
  //
  Plato::ScalarVector statesView("State",tNumDofs);
  Kokkos::deep_copy(statesView, state);

  auto stateView_host = Kokkos::create_mirror_view(statesView);
  Kokkos::deep_copy(stateView_host, statesView);

  // test that filtered density field is still 1
  //
  for(int iDof=0; iDof<tNumDofs; iDof++){
    TEST_FLOATING_EQUALITY(stateView_host(iDof), 1.0, 1.0e-13);
  }
}
