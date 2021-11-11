#include "PlatoTestHelpers.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "helmholtz/Helmholtz.hpp"
#include "helmholtz/VectorFunction.hpp"
#include "helmholtz/SimplexHelmholtz.hpp"
#include "helmholtz/Problem.hpp"
#include "alg/PlatoSolverFactory.hpp"
#include "BLAS1.hpp"
#include "PlatoMathHelpers.hpp"

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
bool isFixedDomain(const std::string & aDomainName, const std::vector<std::string> & aFixedDomains) 
{
    for (auto iNameOrdinal(0); iNameOrdinal < aFixedDomains.size(); iNameOrdinal++)
    {
        if(aFixedDomains[iNameOrdinal] == aDomainName)
            return true;
    }
    return false;
}

void markBlockNodes(Omega_h::Mesh & aMesh, 
                    const Plato::SpatialDomain & aDomain, 
                    const Plato::OrdinalType aNumNodesPerCell, 
                    Plato::LocalOrdinalVector aMarkedNodes)
{
    auto tCells2Nodes = aMesh.ask_elem_verts();
    auto tDomainCells = aDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tDomainCells.size()), LAMBDA_EXPRESSION(Plato::OrdinalType iElemOrdinal)
    {
        Plato::OrdinalType tElement = tDomainCells(iElemOrdinal); 
        for(Plato::OrdinalType iVertOrdinal=0; iVertOrdinal < aNumNodesPerCell; ++iVertOrdinal)
        {
            Plato::OrdinalType tVertIndex = tCells2Nodes[tElement*aNumNodesPerCell + iVertOrdinal];
            aMarkedNodes(tVertIndex) = 1;
        }
    }, "nodes in domain element set");
}

Plato::OrdinalType getNumberOfUniqueNodes(const Plato::LocalOrdinalVector & aNodeVector)
{
    Plato::OrdinalType tSum(0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0,aNodeVector.size()),
    LAMBDA_EXPRESSION(const Plato::OrdinalType& aOrdinal, Plato::OrdinalType & aUpdate)
    {
        aUpdate += aNodeVector(aOrdinal);
    }, tSum);
    return tSum;
}

void storeUniqueNodes(const Plato::LocalOrdinalVector & aMarkedNodes, 
                      Plato::LocalOrdinalVector & aUniqueNodes)
{
    Plato::OrdinalType tOffset(0);
    Kokkos::parallel_scan (Kokkos::RangePolicy<Plato::OrdinalType>(0,aMarkedNodes.size()),
    KOKKOS_LAMBDA (const Plato::OrdinalType& iOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
    {
        const Plato::OrdinalType tVal = aMarkedNodes(iOrdinal);
        if( tIsFinal && tVal ) 
            aUniqueNodes(aUpdate) = iOrdinal; 
        aUpdate += tVal;
    }, tOffset);
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
    "  <ParameterList name='Length Scale'>                                    \n"
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
    "  <ParameterList name='Length Scale'>                                    \n"
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
    "  <ParameterList name='Length Scale'>                                    \n"
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
  // set parameters
  //
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
    "    <ParameterList name='Fixed Volume'>                                     \n"
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
  TEST_EQUALITY(tFixedDomainNames[0], "Fixed Volume");
}

/******************************************************************************/
/*!
  \brief check fixed block in list

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HelmholtzFilterTests, FindFixedBlock )
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
    "    <ParameterList name='Fixed Volume'>                                     \n"
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

  TEST_EQUALITY(isFixedDomain("Fixed Vol",tFixedDomainNames), false);
  TEST_EQUALITY(isFixedDomain("Fixed Volume",tFixedDomainNames), true);
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
    "    <ParameterList name='Fixed Volume'>                                     \n"
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

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);
  auto tNumNodes = tMesh->nverts();

  // get mesh sets
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  using SimplexPhysics = ::Plato::HelmholtzFilter<spaceDim>;
  auto tNumNodesPerCell = SimplexPhysics::mNumNodesPerCell;

  Plato::LocalOrdinalVector tFixedBlockNodes("Nodes in fixed blocks", tNumNodes);
  Plato::blas1::fill(static_cast<Plato::OrdinalType>(0), tFixedBlockNodes);

  for(const auto& tDomain : tSpatialModel.Domains)
  {
      auto tDomainName = tDomain.getDomainName();
      if (isFixedDomain(tDomainName,tFixedDomainNames))
          markBlockNodes(*tMesh, tDomain, tNumNodesPerCell, tFixedBlockNodes);
  }

  auto tNumUniqueNodes = getNumberOfUniqueNodes(tFixedBlockNodes);
  Plato::LocalOrdinalVector tUniqueFixedBlockNodes("Unique nodes in fixed blocks", tNumUniqueNodes);
  storeUniqueNodes(tFixedBlockNodes,tUniqueFixedBlockNodes);

  TEST_EQUALITY(tNumUniqueNodes, tNumNodes);

  auto tUniqueFixedBlockNodes_host = Kokkos::create_mirror_view(tUniqueFixedBlockNodes);
  Kokkos::deep_copy(tUniqueFixedBlockNodes_host, tUniqueFixedBlockNodes);
  for (auto iNodeOrdinal = 0; iNodeOrdinal < tNumNodes; iNodeOrdinal++)
  {
      TEST_EQUALITY(iNodeOrdinal, tUniqueFixedBlockNodes_host(iNodeOrdinal));
  }
}

