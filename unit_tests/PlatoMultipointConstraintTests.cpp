#include "util/PlatoTestHelpers.hpp"
#include "util/PlatoMathTestHelpers.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "Mechanics.hpp"
#include "EssentialBCs.hpp"
#include "elliptic/VectorFunction.hpp"
#include "ApplyConstraints.hpp"
#include "LinearElasticMaterial.hpp"
#include "alg/PlatoSolverFactory.hpp"
#include "MultipointConstraints.hpp"

#include "PlatoStaticsTypes.hpp"
#include "PlatoMathHelpers.hpp"

#include "SpatialModel.hpp"

#include "Tri3.hpp"
#include "Tet4.hpp"

#ifdef HAVE_AMGX
#include <alg/AmgXSparseLinearProblem.hpp>
#endif

#include <memory>
#include <typeinfo>
#include <vector>

/******************************************************************************/
/*!
  \brief test operations that form condensed systen

  Construct a linear system with tie multipoint constraints.
  Test passes if transformed Jacobian and residual have correct sizes
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( MultipointConstraintTests, BuildCondensedSystem )
{
  // specify parameter input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                    \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>     \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>             \n"
    "  <ParameterList name='Elliptic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>            \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                   \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>          \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>   \n"
    "      <Parameter name='Values' type='Array(double)' value='{1e3, 0}'/>  \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>        \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Essential Boundary Conditions'>                 \n"
    "    <ParameterList  name='X Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='0'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='x-'/>            \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='1'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='x-'/>            \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Multipoint Constraints'>                        \n"
    "    <ParameterList  name='Node Tie Constraint 1'>                       \n"
    "      <Parameter  name='Type'     type='string'    value='Tie'/>        \n"
    "      <Parameter  name='Child'    type='string'    value='y+'/>         \n"
    "      <Parameter  name='Parent'   type='string'    value='y-'/>         \n"
    "      <Parameter  name='Value'    type='double'    value='4.2'/>        \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "</ParameterList>                                                        \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using PhysicsType = ::Plato::Mechanics<Plato::Tri3>;
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

  // parse essential BCs
  //
  Plato::OrdinalVector mBcDofs;
  Plato::ScalarVector mBcValues;
  Plato::EssentialBCs<ElementType>
      tEssentialBoundaryConditions(params->sublist("Essential Boundary Conditions",false), tMesh);
  tEssentialBoundaryConditions.get(mBcDofs, mBcValues);

  // create vector function
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *params, tDataMap);
  Plato::Elliptic::VectorFunction<PhysicsType>
    vectorFunction(tSpatialModel, tDataMap, *params, params->get<std::string>("PDE Constraint"));

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);
  
  // parse multipoint constraints
  //
  std::shared_ptr<Plato::MultipointConstraints> tMPCs = std::make_shared<Plato::MultipointConstraints>(tSpatialModel, tNumDofsPerNode, params->sublist("Multipoint Constraints", false));
  tMPCs->setupTransform();

  // apply essential BCs
  //
  Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(jacobian, residual, mBcDofs, mBcValues);

  // setup transformation
  //
  //Teuchos::RCP<Plato::CrsMatrixType> aA(&jacobian, /*hasOwnership=*/ false);
  Teuchos::RCP<Plato::CrsMatrixType> aA = jacobian;
  const Plato::OrdinalType tNumCondensedNodes = tMPCs->getNumCondensedNodes();
  auto tNumCondensedDofs = tNumCondensedNodes*tNumDofsPerNode;
  
  // get MPC condensation matrices and RHS
  Teuchos::RCP<Plato::CrsMatrixType> tTransformMatrix = tMPCs->getTransformMatrix();
  Teuchos::RCP<Plato::CrsMatrixType> tTransformMatrixTranspose = tMPCs->getTransformMatrixTranspose();
  Plato::ScalarVector tMpcRhs = tMPCs->getRhsVector();
  
  // build condensed matrix
  auto tCondensedALeft = Teuchos::rcp( new Plato::CrsMatrixType(tNumDofs, tNumCondensedDofs, tNumDofsPerNode, tNumDofsPerNode) );
  auto tCondensedA     = Teuchos::rcp( new Plato::CrsMatrixType(tNumCondensedDofs, tNumCondensedDofs, tNumDofsPerNode, tNumDofsPerNode) );
      
  Plato::MatrixMatrixMultiply(aA, tTransformMatrix, tCondensedALeft);
  Plato::MatrixMatrixMultiply(tTransformMatrixTranspose, tCondensedALeft, tCondensedA);

  // build condensed vector
  Plato::ScalarVector tInnerB = residual;
  Plato::blas1::scale(-1.0, tMpcRhs);
  Plato::MatrixTimesVectorPlusVector(aA, tMpcRhs, tInnerB);
  
  Plato::ScalarVector tCondensedB("Condensed RHS Vector", tNumCondensedDofs);
  Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tCondensedB);
  
  Plato::MatrixTimesVectorPlusVector(tTransformMatrixTranspose, tInnerB, tCondensedB);

  // Compute condensed jacobian with slow dumb
  auto tSlowDumbCondensedALeft = Teuchos::rcp( new Plato::CrsMatrixType(tNumDofs, tNumCondensedDofs, tNumDofsPerNode, tNumDofsPerNode) );
  auto tSlowDumbCondensedA     = Teuchos::rcp( new Plato::CrsMatrixType(tNumCondensedDofs, tNumCondensedDofs, tNumDofsPerNode, tNumDofsPerNode) );
  
  Plato::TestHelpers::slow_dumb_matrix_matrix_multiply( aA, tTransformMatrix, tSlowDumbCondensedALeft);
  Plato::TestHelpers::slow_dumb_matrix_matrix_multiply( tTransformMatrixTranspose, tSlowDumbCondensedALeft, tSlowDumbCondensedA);

  // test lengths
  TEST_EQUALITY(tCondensedA->rowMap().size(), tSlowDumbCondensedA->rowMap().size());
}

/******************************************************************************/
/*!
  \brief 2D Elastic problem with Tie multipoint constraints

  Construct a linear system with tie multipoint constraints.
  Test passes if nodal displacements are offset by specified amount in MPC
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( MultipointConstraintTests, Elastic2DTieMPC )
{
  // specify parameter input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                    \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>     \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>             \n"
    "  <ParameterList name='Elliptic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>            \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                   \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>          \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>   \n"
    "      <Parameter name='Values' type='Array(double)' value='{1e3, 0}'/>  \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>        \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Essential Boundary Conditions'>                 \n"
    "    <ParameterList  name='X Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='0'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='x-'/>            \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='1'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='x-'/>            \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Multipoint Constraints'>                        \n"
    "    <ParameterList  name='Node Tie Constraint 1'>                       \n"
    "      <Parameter  name='Type'     type='string'    value='Tie'/>        \n"
    "      <Parameter  name='Child'    type='string'    value='y+'/>         \n"
    "      <Parameter  name='Parent'   type='string'    value='y-'/>         \n"
    "      <Parameter  name='Value'    type='double'    value='4.2'/>        \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "</ParameterList>                                                        \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using PhysicsType = ::Plato::Mechanics<Plato::Tri3>;
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

  // parse essential BCs
  //
  Plato::OrdinalVector mBcDofs;
  Plato::ScalarVector mBcValues;
  Plato::EssentialBCs<ElementType>
      tEssentialBoundaryConditions(params->sublist("Essential Boundary Conditions",false), tMesh);
  tEssentialBoundaryConditions.get(mBcDofs, mBcValues);

  // create vector function
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *params, tDataMap);
  Plato::Elliptic::VectorFunction<PhysicsType>
    vectorFunction(tSpatialModel, tDataMap, *params, params->get<std::string>("PDE Constraint"));

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);
  
  // parse multipoint constraints
  //
  std::shared_ptr<Plato::MultipointConstraints> tMPCs = std::make_shared<Plato::MultipointConstraints>(tSpatialModel, tNumDofsPerNode, params->sublist("Multipoint Constraints", false));
  tMPCs->setupTransform();
  
  // create solver
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                              \n"
    "  <Parameter name='Solver' type='string' value='gmres'/>          \n"
    "  <Parameter name='Display Iterations' type='int' value='0'/>     \n"
    "  <Parameter name='Iterations' type='int' value='200'/>           \n"
    "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "</ParameterList>                                                  \n"
  );
  Plato::SolverFactory tSolverFactory(*tSolverParams);

  auto tSolver = tSolverFactory.create(tMesh->NumNodes(), tMachine, tNumDofsPerNode, tMPCs);

  // apply essential BCs
  //
  Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(jacobian, residual, mBcDofs, mBcValues);

  // solve linear system
  //
  tSolver->solve(*jacobian, state, residual);

  // create mirror view of displacement solution
  //
  Plato::ScalarVector statesView("State",tNumDofs);
  Kokkos::deep_copy(statesView, state);

  auto stateView_host = Kokkos::create_mirror_view(statesView);
  Kokkos::deep_copy(stateView_host, statesView);

  // test difference between constrained nodes
  //
  Plato::OrdinalType checkChildNode = 5;
  Plato::OrdinalType checkParentNode = 3;
  Plato::Scalar      checkValue = 4.2;

  Plato::OrdinalType checkChildDof0 = checkChildNode*tNumDofsPerNode;
  Plato::OrdinalType checkChildDof1 = checkChildNode*tNumDofsPerNode + 1;

  Plato::OrdinalType checkParentDof0 = checkParentNode*tNumDofsPerNode;
  Plato::OrdinalType checkParentDof1 = checkParentNode*tNumDofsPerNode + 1;

  Plato::Scalar checkDifferenceDof0 = stateView_host(checkChildDof0) - stateView_host(checkParentDof0);
  Plato::Scalar checkDifferenceDof1 = stateView_host(checkChildDof1) - stateView_host(checkParentDof1);

  TEST_FLOATING_EQUALITY(checkDifferenceDof0, checkValue, 1.0e-12);
  TEST_FLOATING_EQUALITY(checkDifferenceDof1, checkValue, 1.0e-12);

}

/******************************************************************************/
/*!
  \brief 2D Elastic problem with PBC multipoint constraints

  Construct a linear system with PBC multipoint constraints.
  Test passes if nodal displacements are offset by specified amount in MPC
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( MultipointConstraintTests, Elastic3DPbcMPC )
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  using PhysicsType = ::Plato::Mechanics<Plato::Tet4>;
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
  
  // specify parameter input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                    \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>     \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>             \n"
    "  <ParameterList name='Elliptic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>            \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                   \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>          \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>   \n"
    "      <Parameter name='Values' type='Array(double)' value='{1e3, 0, 0}'/>  \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>        \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Essential Boundary Conditions'>                 \n"
    "    <ParameterList  name='X Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='0'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='x-'/>            \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='1'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='x-'/>            \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Z Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='2'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='x-'/>            \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Multipoint Constraints'>                        \n"
    "    <ParameterList  name='PBC Constraint 1'>                            \n"
    "      <Parameter  name='Type'     type='string'    value='PBC'/>        \n"
    "      <Parameter  name='Child'    type='string'    value='y-'/>         \n"
    "      <Parameter  name='Parent'   type='string'    value='Design Volume'/>  \n"
    "      <Parameter  name='Vector'  type='Array(double)' value='{0, 1, 0}'/>  \n"
    "      <Parameter  name='Value'    type='double'    value='0.0'/>        \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "</ParameterList>                                                        \n"
  );

  // parse essential BCs
  //
  Plato::OrdinalVector mBcDofs;
  Plato::ScalarVector mBcValues;
  Plato::EssentialBCs<ElementType>
      tEssentialBoundaryConditions(params->sublist("Essential Boundary Conditions",false), tMesh);
  tEssentialBoundaryConditions.get(mBcDofs, mBcValues);

  // create vector function
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *params, tDataMap);
  Plato::Elliptic::VectorFunction<PhysicsType>
    vectorFunction(tSpatialModel, tDataMap, *params, params->get<std::string>("PDE Constraint"));

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);
  
  // parse multipoint constraints
  //
  std::shared_ptr<Plato::MultipointConstraints> tMPCs = std::make_shared<Plato::MultipointConstraints>(tSpatialModel, tNumDofsPerNode, params->sublist("Multipoint Constraints", false));
  tMPCs->setupTransform();
  
  // create solver
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                              \n"
    "  <Parameter name='Solver' type='string' value='gmres'/>          \n"
    "  <Parameter name='Display Iterations' type='int' value='0'/>     \n"
    "  <Parameter name='Iterations' type='int' value='200'/>           \n"
    "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "</ParameterList>                                                  \n"
  );
  Plato::SolverFactory tSolverFactory(*tSolverParams);

  auto tSolver = tSolverFactory.create(tMesh->NumNodes(), tMachine, tNumDofsPerNode, tMPCs);

  // apply essential BCs
  //
  Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(jacobian, residual, mBcDofs, mBcValues);

  // solve linear system
  //
  tSolver->solve(*jacobian, state, residual);

  // create mirror view of displacement solution
  //
  Plato::ScalarVector statesView("State",tNumDofs);
  Kokkos::deep_copy(statesView, state);

  auto stateView_host = Kokkos::create_mirror_view(statesView);
  Kokkos::deep_copy(stateView_host, statesView);

  // test difference between constrained nodes
  //
  Plato::OrdinalType checkChildNode = 0;
  Plato::OrdinalType checkParentNode = 3;
  Plato::Scalar      checkValue = 0.0;

  Plato::OrdinalType checkChildDof0 = checkChildNode*tNumDofsPerNode;
  Plato::OrdinalType checkChildDof1 = checkChildNode*tNumDofsPerNode + 1;
  Plato::OrdinalType checkChildDof2 = checkChildNode*tNumDofsPerNode + 2;

  Plato::OrdinalType checkParentDof0 = checkParentNode*tNumDofsPerNode;
  Plato::OrdinalType checkParentDof1 = checkParentNode*tNumDofsPerNode + 1;
  Plato::OrdinalType checkParentDof2 = checkParentNode*tNumDofsPerNode + 2;

  Plato::Scalar checkDifferenceDof0 = stateView_host(checkChildDof0) - stateView_host(checkParentDof0);
  Plato::Scalar checkDifferenceDof1 = stateView_host(checkChildDof1) - stateView_host(checkParentDof1);
  Plato::Scalar checkDifferenceDof2 = stateView_host(checkChildDof2) - stateView_host(checkParentDof2);

  TEST_FLOATING_EQUALITY(checkDifferenceDof0, checkValue, 1.0e-8);
  TEST_FLOATING_EQUALITY(checkDifferenceDof1, checkValue, 1.0e-8);
  TEST_FLOATING_EQUALITY(checkDifferenceDof2, checkValue, 1.0e-8);
}
