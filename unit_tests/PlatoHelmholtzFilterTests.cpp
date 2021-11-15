#include "PlatoTestHelpers.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "helmholtz/Helmholtz.hpp"
#include "helmholtz/VectorFunction.hpp"
#include "helmholtz/SimplexHelmholtz.hpp"
#include "helmholtz/Problem.hpp"
#include "helmholtz/FixedDomainDofs.hpp"

#include "BLAS1.hpp"
#include "UtilsOmegaH.hpp"
#include "OmegaHUtilities.hpp"
#include "PlatoMathHelpers.hpp"
#include "alg/PlatoSolverFactory.hpp"

#ifdef HAVE_AMGX
#include <alg/AmgXSparseLinearProblem.hpp>
#endif

#include <fenv.h>
#include <memory>

template <typename DataType>
void print_view(const Plato::ScalarVectorT<DataType> & aView)
{
    auto tView_host = Kokkos::create_mirror(aView);
    Kokkos::deep_copy(tView_host, aView);
    std::cout << '\n';
    for (unsigned int i = 0; i < aView.extent(0); ++i)
    {
        std::cout << tView_host(i) << '\n';
    }
}

// print full matrix entries
void PrintFullMatrix(const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrix)
{
    auto tNumRows = aInMatrix->numRows();
    auto tNumCols = aInMatrix->numCols();

    auto tFullMat = ::PlatoUtestHelpers::toFull(aInMatrix);

    printf("\n Full matrix entries: \n");
    for (auto iRow = 0; iRow < tNumRows; iRow++)
    {
        for (auto iCol = 0; iCol < tNumCols; iCol++)
        {
            printf("%f ",tFullMat[iRow][iCol]);
        }
        printf("\n");
    
    }
}

TEUCHOS_UNIT_TEST(HelmholtzFilterTests, TestOmegaH)
{
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
  //Plato::write_exodus_file("mesh.exo", *tMesh);

  auto tBoundaryEntitiesIDs = Plato::omega_h::get_boundary_entities<Omega_h::EDGE>(*tMesh);
  //Plato::omega_h::print<Omega_h::LOs>(tBoundaryEntitiesIDs, "IDs");
  
  auto tCopy = Plato::omega_h::copy<Plato::OrdinalType>(tBoundaryEntitiesIDs);
  auto tHostCopy = Kokkos::create_mirror_view(tCopy);
  Kokkos::deep_copy(tHostCopy, tCopy);

  std::vector<Plato::OrdinalType> tGold = {0, 2, 3, 4};
  for (auto &tValue : tGold)
  {
    auto tIndex = &tValue - &tGold[0];
    TEST_EQUALITY(tValue, tHostCopy(tIndex));
  }
}

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
  constexpr int spaceDim=1;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::HelmholtzFilter<spaceDim>;

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

  // get mesh sets
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  // create PDE
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  TEST_THROW(Plato::Helmholtz::VectorFunction<SimplexPhysics> vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint")), std::runtime_error);
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
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);
  
  // create mesh based density
  //
  using SimplexPhysics = ::Plato::HelmholtzFilter<spaceDim>;
  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = tMesh->nverts();
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

  // get mesh sets
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  // get machine
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  // construct problem
  auto tProblem = Plato::Helmholtz::Problem<::Plato::HelmholtzFilter<spaceDim>>(*tMesh, tMeshSets, *tParamList, tMachine);

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
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::HelmholtzFilter<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = tMesh->nverts();
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
    "  <ParameterList name='Parameters'>                                    \n"
    "    <Parameter name='Length Scale' type='double' value='0.10'/>              \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Linear Solver'>                              \n"
    "    <Parameter name='Solver Stack' type='string' value='Epetra'/>   \n"
    "    <Parameter name='Display Iterations' type='int' value='1'/>     \n"
    "    <Parameter name='Iterations' type='int' value='50'/>            \n"
    "    <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "  </ParameterList>                                                  \n"
    "</ParameterList>                                                        \n"
  );

  // get mesh sets
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  // create PDE
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);
  Plato::Helmholtz::VectorFunction<SimplexPhysics>
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

  auto tSolver = tSolverFactory.create(tMesh->nverts(), tMachine, tNumDofsPerNode);
  
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
  \brief parse fixed blocks

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HelmholtzFilterTests, ParseFixedBlocks )
{
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='block_1'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "      <ParameterList name='Fixed Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='block_2'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Fixed Domains'>                                    \n"
    "    <ParameterList name='block_2'>                                     \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                        \n"
  );

  std::vector<std::string> tFixedDomainNames;

  if (tParamList->isSublist("Fixed Domains"))
  {
      auto tFixedDomains = tParamList->sublist("Fixed Domains");
      for (auto tIndex = tFixedDomains.begin(); tIndex != tFixedDomains.end(); ++tIndex)
      {
          tFixedDomainNames.push_back(tFixedDomains.name(tIndex));
      }
  }

  TEST_EQUALITY(tFixedDomainNames.size(), 1);
  TEST_EQUALITY(tFixedDomainNames[0], "block_2");
}

/******************************************************************************/
/*!
  \brief check fixed block in list

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HelmholtzFilterTests, FindFixedBlock )
{
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Fixed Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Fixed Domains'>                                    \n"
    "    <ParameterList name='body'>                                     \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                        \n"
  );

  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::HelmholtzFilter<spaceDim>;
  auto tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  auto tNumNodesPerCell = SimplexPhysics::mNumNodesPerCell;

  Plato::FixedDomainDofs 
      tSetFixedDomainEssentialBcDofs(*tMesh,tParamList->sublist("Fixed Domains"),tNumDofsPerNode,tNumNodesPerCell);

  TEST_EQUALITY(tSetFixedDomainEssentialBcDofs.isFixedDomain("Fixed Volume"), false);
  TEST_EQUALITY(tSetFixedDomainEssentialBcDofs.isFixedDomain("body"), true);
}

/******************************************************************************/
/*!
  \brief throw error if number of DOFs per node is greater than 1

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HelmholtzFilterTests, ThrowIfNumDofsPerNodeGreaterThanOne )
{
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Fixed Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Fixed Domains'>                                    \n"
    "    <ParameterList name='body'>                                     \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                        \n"
  );

  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::HelmholtzFilter<spaceDim>;
  auto tNumNodesPerCell = SimplexPhysics::mNumNodesPerCell;
  auto tNumDofsPerNode = 2;

  TEST_THROW(Plato::FixedDomainDofs tSetFixedDomainEssentialBcDofs(*tMesh,tParamList->sublist("Fixed Domains"),tNumDofsPerNode,tNumNodesPerCell), std::runtime_error);
}

/******************************************************************************/
/*!
  \brief build EBC DOF array for non-fixed block 

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HelmholtzFilterTests, BuildEssentialBCArrayForNonFixedBlock )
{
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Fixed Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                        \n"
  );

  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::HelmholtzFilter<spaceDim>;
  auto tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  auto tNumNodesPerCell = SimplexPhysics::mNumNodesPerCell;

  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList); 

  Plato::LocalOrdinalVector tBcDofs;

  if(tParamList->isSublist("Fixed Domains"))
  {
      Plato::FixedDomainDofs 
          tSetFixedDomainEssentialBcDofs(*tMesh,tParamList->sublist("Fixed Domains"),tNumDofsPerNode,tNumNodesPerCell);
      tSetFixedDomainEssentialBcDofs(tSpatialModel,tBcDofs);
  }

  TEST_EQUALITY(tBcDofs.size(), 0);
}

/******************************************************************************/
/*!
  \brief build EBC DOF array for fixed block 

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HelmholtzFilterTests, BuildEssentialBCArrayForFixedBlock )
{
  // set parameters
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Fixed Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Fixed Domains'>                                    \n"
    "    <ParameterList name='body'>                                     \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                        \n"
  );

  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::HelmholtzFilter<spaceDim>;
  auto tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  auto tNumNodesPerCell = SimplexPhysics::mNumNodesPerCell;

  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList); 

  Plato::LocalOrdinalVector tBcDofs;

  if(tParamList->isSublist("Fixed Domains"))
  {
      Plato::FixedDomainDofs 
          tSetFixedDomainEssentialBcDofs(*tMesh,tParamList->sublist("Fixed Domains"),tNumDofsPerNode,tNumNodesPerCell);
      tSetFixedDomainEssentialBcDofs(tSpatialModel,tBcDofs);
  }

  auto tNumMeshNodes = tMesh->nverts();
  TEST_EQUALITY(tBcDofs.size(), tNumMeshNodes);

  auto tBcDofs_host = Kokkos::create_mirror_view(tBcDofs);
  Kokkos::deep_copy(tBcDofs_host, tBcDofs);
  for (auto iNodeOrdinal = 0; iNodeOrdinal < tNumMeshNodes; iNodeOrdinal++)
  {
      TEST_EQUALITY(iNodeOrdinal, tBcDofs_host(iNodeOrdinal));
  }
}

