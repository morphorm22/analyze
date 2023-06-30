#include "util/PlatoTestHelpers.hpp"
#include "util/PlatoMathTestHelpers.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "base/Database.hpp"
#include "elliptic/mechanical/linear/Mechanics.hpp"
#include "bcs/dirichlet/DirichletBCs.hpp"
#include "elliptic/base/VectorFunction.hpp"
#include "ApplyConstraints.hpp"
#include "LinearElasticMaterial.hpp"
#include "solver/PlatoSolverFactory.hpp"
#include "solver/EpetraLinearSolver.hpp"

#ifdef PLATO_TPETRA
#include "solver/TpetraLinearSolver.hpp"
#endif

#ifdef HAVE_AMGX
#include <solver/AmgXSparseLinearProblem.hpp>
#endif

#include "Tri3.hpp"

#include <memory>

namespace
{
template <typename ClassT>
using rcp = std::shared_ptr<ClassT>;

Plato::rcp<Plato::AbstractSolver> solver(
  const std::string& aSolverParams,
  const unsigned int aNumRows)
{
  namespace pth = Plato::TestHelpers;

  Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(aSolverParams);
  Plato::SolverFactory tSolverFactory(*tParamList);
  
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  return tSolverFactory.create(aNumRows, tMachine, 1);
}

std::vector<std::vector<Plato::Scalar>>
to_full(rcp<Epetra_VbrMatrix> aInMatrix)
{
    int tNumMatrixRows = aInMatrix->NumGlobalRows();

    std::vector<std::vector<Plato::Scalar>>
        tRetMatrix(tNumMatrixRows, std::vector<Plato::Scalar>(tNumMatrixRows, 0.0));

    for(int iMatrixRow=0; iMatrixRow<tNumMatrixRows; iMatrixRow++)
    {
        int tNumEntriesThisRow = 0;
        aInMatrix->NumMyRowEntries(iMatrixRow, tNumEntriesThisRow);
        int tNumEntriesFound = 0;
        std::vector<Plato::Scalar> tVals(tNumEntriesThisRow,0);
        std::vector<int> tInds(tNumEntriesThisRow,0);
        aInMatrix->ExtractMyRowCopy(iMatrixRow, tNumEntriesThisRow, tNumEntriesFound, tVals.data(), tInds.data());
        for(int iEntry=0; iEntry<tNumEntriesFound; iEntry++)
        {
            tRetMatrix[iMatrixRow][tInds[iEntry]] = tVals[iEntry];
        }
    }
    return tRetMatrix;
}

Teuchos::RCP<Teuchos::ParameterList> elastic_2d_xml_parameters()
{
    return Teuchos::getParametersFromXmlString(
        "<ParameterList name='Plato Problem'>                                      \n"
        "  <ParameterList name='Spatial Model'>                                    \n"
        "    <ParameterList name='Domains'>                                        \n"
        "      <ParameterList name='Design Volume'>                                \n"
        "        <Parameter name='Element Block' type='string' value='body'/>      \n"
        "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
        "      </ParameterList>                                                    \n"
        "    </ParameterList>                                                      \n"
        "  </ParameterList>                                                        \n"
        "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>       \n"
        "  <Parameter name='Self-Adjoint' type='bool' value='true'/>               \n"
        "  <ParameterList name='Elliptic'>                                         \n"
        "    <ParameterList name='Penalty Function'>                               \n"
        "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
        "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
        "    </ParameterList>                                                      \n"
        "  </ParameterList>                                                        \n"
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
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
        "      <Parameter  name='Index'    type='long long'    value='0'/>       \n"
#else
        "      <Parameter  name='Index'    type='int'    value='0'/>             \n"
#endif
        "      <Parameter  name='Sides'    type='string' value='x-'/>            \n"
        "    </ParameterList>                                                    \n"
        "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>     \n"
        "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
        "      <Parameter  name='Index'    type='long long'    value='1'/>       \n"
#else
        "      <Parameter  name='Index'    type='int'    value='1'/>             \n"
#endif
        "      <Parameter  name='Sides'    type='string' value='y-'/>            \n"
        "    </ParameterList>                                                    \n"
        "  </ParameterList>                                                      \n"
        "</ParameterList>                                                        \n"
    );
}

struct ElasticProblemParameters
{
    Plato::Scalar mTraction = 0.0;
    Plato::Scalar mYoungsModulus = 0.0;
    Plato::Scalar mPoissonsRatio = 0.0;
    Plato::Scalar mMeshWidth = 0.0;
};

ElasticProblemParameters elastic_2d_parameters(const Plato::Scalar aMeshPhysicalWidth)
{
    const Teuchos::RCP<const Teuchos::ParameterList> tParams = elastic_2d_xml_parameters();
    const Teuchos::ParameterList tNaturalBCs = 
        tParams->sublist("Natural Boundary Conditions").sublist("Traction Vector Boundary Condition");
    const Teuchos::ParameterList tMaterial = 
        tParams->sublist("Material Models").sublist("Unobtainium").sublist("Isotropic Linear Elastic");
    // Use designated initializers in C++20:
    return {/*.mTraction =*/ tNaturalBCs.get<Teuchos::Array<double>>("Values")[0], 
        /*.mYoungsModulus =*/ tMaterial.get<double>("Youngs Modulus"),
        /*.mPoissonsRatio =*/ tMaterial.get<double>("Poissons Ratio"),
        /*.mMeshWidth = */ aMeshPhysicalWidth};
}

Plato::ScalarVector::HostMirror
test_elastic_problem_solution(const Plato::Mesh& aMesh, const std::string& aSolverParameters)
{
    using PhysicsType = ::Plato::Elliptic::Linear::Mechanics<Plato::Tri3>;
    using ElementType = typename PhysicsType::ElementType;

    const int tNumDofsPerNode = ElementType::mNumDofsPerNode;
    const int tNumNodes = aMesh->NumNodes();
    const int tNumDofs = tNumNodes * tNumDofsPerNode;

    // create mesh based density
    //
    Plato::Database tDatabase;
    Plato::ScalarVector tControl("density", tNumDofs);
    Kokkos::deep_copy(tControl, 1.0);
    tDatabase.vector("controls",tControl);

    // create mesh based state
    //
    Plato::ScalarVector tState("state", tNumDofs);
    Kokkos::deep_copy(tState, 0.0);
    tDatabase.vector("states",tState);

    // create material model
    //
    Teuchos::RCP<Teuchos::ParameterList> tParamList = elastic_2d_xml_parameters();

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(aMesh, *tParamList, tDataMap);
    Plato::Elliptic::VectorFunction<PhysicsType>
      vectorFunction(tParamList->get<std::string>("PDE Constraint"), tSpatialModel, tDataMap, *tParamList);

    // compute and test constraint value
    //
    auto tResidual = vectorFunction.value(tDatabase,/*cycle=*/0.);

    // compute and test constraint value
    //
    auto tJacobian = vectorFunction.jacobianState(tDatabase,/*cycle=*/0.);

    // parse constraints
    //
    Plato::OrdinalVector tBcDofs;
    Plato::ScalarVector tBcValues;
    Plato::DirichletBCs<ElementType>
        tEssentialBoundaryConditions(tParamList->sublist("Essential Boundary Conditions",false), aMesh);
    tEssentialBoundaryConditions.get(tBcDofs, tBcValues);
    Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(tJacobian, tResidual, tBcDofs, tBcValues);

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    Kokkos::deep_copy(tState, 0.0);
    {
        Teuchos::RCP<Teuchos::ParameterList> tSolverParams = Teuchos::getParametersFromXmlString(aSolverParameters);
        Plato::SolverFactory tSolverFactory(*tSolverParams);
        auto tSolver = tSolverFactory.create(aMesh->NumNodes(), tMachine, tNumDofsPerNode);
        tSolver->solve(*tJacobian, tState, tResidual);
    }
    Plato::ScalarVector tStateSolution("state", tNumDofs);
    Kokkos::deep_copy(tStateSolution, tState);

    Plato::ScalarVector::HostMirror tStateSolutionHost = Kokkos::create_mirror_view(tStateSolution);
    Kokkos::deep_copy(tStateSolutionHost, tStateSolution);
    return tStateSolutionHost;
}

/// Exact solution for the problem defined in test_elastic_problem_solution.
/// The solution is given by:
/// \f$u = \frac{(\nu^2 - 1)}{E} P x \f$
/// \f$v = \frac{\nu (1 + \nu)}{E} P y \f$
Plato::Scalar analytic_elastic_2d_solution(const ElasticProblemParameters& aElasticParams, 
    const int aMeshWidth, const int aIndex)
{
    const Plato::Scalar tDx = aElasticParams.mMeshWidth / aMeshWidth;
    constexpr int tNumComponents = 2;
    const int tNodeIndex = aIndex / tNumComponents;
    const int tComponentIndex = aIndex % tNumComponents;
    const Plato::Scalar tNu = aElasticParams.mPoissonsRatio;
    const Plato::Scalar tE = aElasticParams.mYoungsModulus;
    if(tComponentIndex == 0)
    {
        const int tNodeXIndex = tNodeIndex / (aMeshWidth + 1);
        const Plato::Scalar tX = tDx * tNodeXIndex;
        const Plato::Scalar tCoeff = (tNu * tNu - 1.0) / tE;
        return tX * tCoeff * aElasticParams.mTraction;
    }
    else 
    {
        const int tNodeYIndex = tNodeIndex % (aMeshWidth + 1);
        const Plato::Scalar tY = tDx * tNodeYIndex;
        const Plato::Scalar tCoeff = tNu * (1.0 + tNu) / tE;
        return tY * tCoeff * aElasticParams.mTraction;
    }
}

void test_vs_analytic_2d_solution(
    const std::string& aSolverParameters,
    const int aMeshWidth, 
    const double aRelativeTol, 
    const double aSmallTol,
    Teuchos::FancyOStream &aOut, 
    bool &aSuccess)
{
    // Use structured binding in C++17:
    Plato::Mesh tMesh;
    BamG::MeshSpec tMeshSpec;
    std::tie(tMesh, tMeshSpec) = Plato::TestHelpers::get_box_mesh_with_spec("TRI3", aMeshWidth);
    const auto tSolution = test_elastic_problem_solution(tMesh, aSolverParameters);
    const ElasticProblemParameters tElasticParams = elastic_2d_parameters(tMeshSpec.dimX);
    for(int i = 0; i < tSolution.size(); i++)
    {
        if(std::abs(tSolution(i)) > aSmallTol)
        {
            TEUCHOS_TEST_FLOATING_EQUALITY(tSolution(i), 
                analytic_elastic_2d_solution(tElasticParams, aMeshWidth, i), aRelativeTol, aOut, aSuccess);
        }
        else
        {
            TEUCHOS_TEST_EQUALITY(analytic_elastic_2d_solution(tElasticParams, aMeshWidth, i), 0.0, aOut, aSuccess);
        }
    }
}
}

#ifdef PLATO_EPETRA
/******************************************************************************/
/*!
  \brief Test matrix conversion

  Create an EpetraSystem then convert a 2D elasticity jacobian from a
  Plato::CrsMatrix<int> to an Epetra_VbrMatrix.  Then, convert both to a full
  matrix and compare entries.  Test passes if entries are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, MatrixConversionEpetra )
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using PhysicsType = ::Plato::Elliptic::Linear::Mechanics<Plato::Tri3>;
  using ElementType = typename PhysicsType::ElementType;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::Database tDatabase;
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);
  tDatabase.vector("controls",control);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);
  tDatabase.vector("states",state);

  // create material model
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
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>       \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>               \n"
    "  <ParameterList name='Elliptic'>                                         \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                          \n"
  );

  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  Plato::Elliptic::VectorFunction<PhysicsType>
    vectorFunction(tParamList->get<std::string>("PDE Constraint"), tSpatialModel, tDataMap, *tParamList);

  // compute and test constraint value
  //
  auto jacobian = vectorFunction.jacobianState(tDatabase,/*cycle=*/0.);

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::EpetraSystem tSystem(tMesh->NumNodes(), tMachine, tNumDofsPerNode);

  auto tEpetra_VbrMatrix = tSystem.fromMatrix(*jacobian);

  auto tFullEpetra = to_full(tEpetra_VbrMatrix);
  auto tFullPlato  = Plato::TestHelpers::to_full(jacobian);

  for(int iRow=0; iRow<tFullEpetra.size(); iRow++)
  {
      for(int iCol=0; iCol<tFullEpetra[iRow].size(); iCol++)
      {
          TEST_FLOATING_EQUALITY(tFullEpetra[iRow][iCol], tFullPlato[iRow][iCol], 1.0e-15);
      }
  }
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Create an EpetraSystem then convert a Plato::ScalarVector to an Epetra_Vector.
  Test passes if entries of both vectors are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionToEpetraVector )
{
  // create test mesh
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using PhysicsType = ::Plato::Elliptic::Linear::Mechanics<Plato::Tri3>;
  using ElementType = typename PhysicsType::ElementType;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::EpetraSystem tSystem(tMesh->NumNodes(), tMachine, tNumDofsPerNode);

  Plato::ScalarVector tTestVector("test vector", tNumDofs);

  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumDofs), KOKKOS_LAMBDA(int vectorIndex)
  {
    tTestVector(vectorIndex) = (double) vectorIndex;
  }, "fill vector");

  auto tConvertedVector = tSystem.fromVector(tTestVector);

  auto tTestVectorHostMirror = Kokkos::create_mirror_view(tTestVector);

  Kokkos::deep_copy(tTestVectorHostMirror,tTestVector);

  for(int i = 0; i < tNumDofs; ++i)
  {
    TEST_FLOATING_EQUALITY(tTestVectorHostMirror(i), (*tConvertedVector)[i], 1.0e-15);
  }
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Provide input of incorrect size. Test passes if std::domain_error is thrown.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionToEpetraVector_invalidInput )
{
  // create test mesh
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using PhysicsType = ::Plato::Elliptic::Linear::Mechanics<Plato::Tri3>;
  using ElementType = typename PhysicsType::ElementType;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::EpetraSystem tSystem(tMesh->NumNodes(), tMachine, tNumDofsPerNode);

  Plato::ScalarVector tTestVector("test vector", tNumDofs+1);

  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumDofs), KOKKOS_LAMBDA(int vectorIndex)
  {
    tTestVector(vectorIndex) = (double) vectorIndex;
  }, "fill vector");

  TEST_THROW(tSystem.fromVector(tTestVector),std::domain_error);
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Create an EpetraSystem then convert an Epetra_Vector to a Plato::ScalarVector.
  Test passes if entries of both vectors are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionFromEpetraVector )
{
  // create test mesh
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using PhysicsType = ::Plato::Elliptic::Linear::Mechanics<Plato::Tri3>;
  using ElementType = typename PhysicsType::ElementType;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::EpetraSystem tSystem(tMesh->NumNodes(), tMachine, tNumDofsPerNode);
  
  auto tTestVector = std::make_shared<Epetra_Vector>(*(tSystem.getMap()));

  tTestVector->Random();

  Plato::ScalarVector tConvertedVector("converted vector", tNumDofs);

  tSystem.toVector(tConvertedVector, tTestVector);

  auto tConvertedVectorHostMirror = Kokkos::create_mirror_view(tConvertedVector);

  Kokkos::deep_copy(tConvertedVectorHostMirror,tConvertedVector);

  for(int i = 0; i < tNumDofs; ++i)
  {
    TEST_FLOATING_EQUALITY((*tTestVector)[i], tConvertedVectorHostMirror(i), 1.0e-15);
  }
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Provide input of incorrect size. Test passes if std::domain_error is thrown.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionFromEpetraVector_invalidInput )
{
  // create test mesh
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using PhysicsType = ::Plato::Elliptic::Linear::Mechanics<Plato::Tri3>;
  using ElementType = typename PhysicsType::ElementType;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::EpetraSystem tSystem(tMesh->NumNodes(), tMachine, tNumDofsPerNode);

  auto tBogusMap = std::make_shared<Epetra_BlockMap>(tNumNodes+1, tNumDofsPerNode, 0, *(tMachine.epetraComm));
  auto tTestVector = std::make_shared<Epetra_Vector>(*tBogusMap);

  Plato::ScalarVector tConvertedVector("converted vector", tNumDofs+tNumDofsPerNode);

  TEST_THROW(tSystem.toVector(tConvertedVector,tTestVector),std::domain_error);
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Provide output ScalarVector of incorrect size. Test passes if std::range_error is thrown.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionFromEpetraVector_invalidOutputContainerProvided )
{
  // create test mesh
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using PhysicsType = ::Plato::Elliptic::Linear::Mechanics<Plato::Tri3>;
  using ElementType = typename PhysicsType::ElementType;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::EpetraSystem tSystem(tMesh->NumNodes(), tMachine, tNumDofsPerNode);

  auto tTestVector = std::make_shared<Epetra_Vector>(*(tSystem.getMap()));

  Plato::ScalarVector tConvertedVector("converted vector", tNumDofs+tNumDofsPerNode);

  TEST_THROW(tSystem.toVector(tConvertedVector,tTestVector),std::range_error);
}

/******************************************************************************/
/*!
  \brief 2D Elastic problem

  Construct a linear system and solve it with the Epetra interface.  
  Test compares the numerical solution with an analytic solution.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, EpetraElastic2D )
{
    constexpr int tMeshWidth = 8;
    // *** use Epetra solver interface *** //
    //
    constexpr auto tEpetraParameters = 
      "<ParameterList name='Linear Solver'>                              \n"
      "  <Parameter name='Solver Stack' type='string' value='Epetra'/>   \n"
      "  <Parameter name='Display Iterations' type='int' value='0'/>     \n"
      "  <Parameter name='Iterations' type='int' value='50'/>            \n"
      "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
      "</ParameterList>                                                  \n";
    constexpr double tRelativeTol = 1e-12;
    constexpr double tSmallTol = 1e-18;
    test_vs_analytic_2d_solution(tEpetraParameters, tMeshWidth, tRelativeTol, tSmallTol, out, success);
}
#endif

#ifdef PLATO_TPETRA
/******************************************************************************/
/*!
  \brief Test matrix conversion

  Create an TpetraSystem then convert a 2D elasticity jacobian from a
  Plato::CrsMatrix<int> to an Tpetra_Matrix.  Then, convert both to a full
  matrix and compare entries.  Test passes if entries are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, MatrixConversionTpetra )
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::Database tDatabase;
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);
  tDatabase.vector("controls",control);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);
  tDatabase.vector("states",state);

  // create material model
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
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>       \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>               \n"
    "  <ParameterList name='Elliptic'>                                         \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                          \n"
  );

  Plato::DataMap tDataMap;

  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  Plato::Elliptic::VectorFunction<::Plato::Elliptic::Linear::Mechanics<Plato::Tri3>>
    vectorFunction(tParamList->get<std::string>("PDE Constraint"), tSpatialModel, tDataMap, *tParamList);

  // compute and test constraint value
  //
  auto jacobian = vectorFunction.jacobianState(tDatabase,/*cycle=*/0.);

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::TpetraSystem tSystem(tMesh->NumNodes(), tMachine, tNumDofsPerNode);

  auto tTpetra_Matrix = tSystem.fromMatrix(*jacobian);

  auto tFullPlato  = Plato::TestHelpers::to_full(jacobian);

  using indices_view_type = Tpetra::CrsMatrix<Plato::Scalar, int, Plato::OrdinalType>::nonconst_global_inds_host_view_type;
  using values_view_type = Tpetra::CrsMatrix<Plato::Scalar, int, Plato::OrdinalType>::nonconst_values_host_view_type;

  for(int iRow=0; iRow<tFullPlato.size(); iRow++)
  {
    size_t tNumEntriesInRow = tTpetra_Matrix->getNumEntriesInGlobalRow(iRow);
    values_view_type tRowValues("values", tNumEntriesInRow);
    indices_view_type tColumnIndices("indices", tNumEntriesInRow);
    tTpetra_Matrix->getGlobalRowCopy(iRow, tColumnIndices, tRowValues, tNumEntriesInRow);

    std::vector<Plato::Scalar> tTpetraRowValues(tFullPlato[iRow].size(), 0.0);
    for(size_t i = 0; i < tNumEntriesInRow; ++i)
    {
      tTpetraRowValues[tColumnIndices[i]] = tRowValues[i];
    }

    for(int iCol=0; iCol<tFullPlato[iRow].size(); iCol++)
    {
        TEST_FLOATING_EQUALITY(tTpetraRowValues[iCol], tFullPlato[iRow][iCol], 1.0e-15);
    }
  }
}


/******************************************************************************/
/*!
  \brief Test matrix conversion mismatch

  Create an TpetraSystem and a 2D elasticity jacobian Plato::CrsMatrix<int>
  with different sizes. Try to conver the jacobian to a Tpetra_Matrix.
  Test passes if a std::domain_error is thrown
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, MatrixConversionTpetra_wrongSize )
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::Database tDatabase;
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);
  tDatabase.vector("controls",control);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);
  tDatabase.vector("states",state);

  // create material model
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
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>       \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>               \n"
    "  <ParameterList name='Elliptic'>                                         \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Material Models'>                                   \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                          \n"
  );

  Plato::DataMap tDataMap;

  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  Plato::Elliptic::VectorFunction<::Plato::Elliptic::Linear::Mechanics<Plato::Tri3>>
    vectorFunction(tParamList->get<std::string>("PDE Constraint"), tSpatialModel, tDataMap, *tParamList);

  // compute and test constraint value
  //
  auto jacobian = vectorFunction.jacobianState(tDatabase,/*cycle=*/0.);

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  constexpr int tBogusMeshWidth=3;
  auto tBogusMesh = Plato::TestHelpers::get_box_mesh("TRI3", tBogusMeshWidth, "BogusMesh.exo");

  Plato::TpetraSystem tSystem(tBogusMesh->NumNodes(), tMachine, tNumDofsPerNode);

  TEST_THROW(tSystem.fromMatrix(*jacobian),std::domain_error);
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Create a TpetraSystem then convert a Plato::ScalarVector to a Tpetra_MultiVector.
  Test passes if entries of both vectors are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionToTpetraVector )
{
  // create test mesh
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::TpetraSystem tSystem(tMesh->NumNodes(), tMachine, tNumDofsPerNode);

  Plato::ScalarVector tTestVector("test vector", tNumDofs);

  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumDofs), KOKKOS_LAMBDA(int vectorIndex)
  {
    tTestVector(vectorIndex) = (double) vectorIndex;
  }, "fill vector");

  auto tConvertedVector = tSystem.fromVector(tTestVector);

  auto tTestVectorHostMirror = Kokkos::create_mirror_view(tTestVector);
  Kokkos::deep_copy(tTestVectorHostMirror,tTestVector);

  auto tConvertedVectorDeviceView2D = tConvertedVector->getLocalView<Plato::DeviceType>(Tpetra::Access::ReadWrite);
  auto tConvertedVectorDeviceView1D = Kokkos::subview(tConvertedVectorDeviceView2D,Kokkos::ALL(), 0);
  auto tConvertedVectorHostMirror = Kokkos::create_mirror_view(tConvertedVectorDeviceView1D);
  Kokkos::deep_copy(tConvertedVectorHostMirror,tConvertedVectorDeviceView1D);

  for(int i = 0; i < tNumDofs; ++i)
  {
    TEST_FLOATING_EQUALITY(tTestVectorHostMirror(i), tConvertedVectorHostMirror(i), 1.0e-15);
  }
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Provide input of incorrect size. Test passes if std::domain_error is thrown.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionToTpetraVector_invalidInput )
{
  // create test mesh
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::TpetraSystem tSystem(tMesh->NumNodes(), tMachine, tNumDofsPerNode);

  Plato::ScalarVector tTestVector("test vector", tNumDofs+1);

  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumDofs), KOKKOS_LAMBDA(int vectorIndex)
  {
    tTestVector(vectorIndex) = (double) vectorIndex;
  }, "fill vector");

  TEST_THROW(tSystem.fromVector(tTestVector),std::domain_error);
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Create an TpetraSystem then convert a Tpetra_MultiVector to a Plato::ScalarVector.
  Test passes if entries of both vectors are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionFromTpetraVector )
{
  // create test mesh
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::TpetraSystem tSystem(tMesh->NumNodes(), tMachine, tNumDofsPerNode);

  auto tTestVector = Teuchos::rcp(new Plato::Tpetra_MultiVector(tSystem.getMap(),1));

  tTestVector->randomize();

  Plato::ScalarVector tConvertedVector("converted vector", tNumDofs);

  tSystem.toVector(tConvertedVector,tTestVector);

  auto tConvertedVectorHostMirror = Kokkos::create_mirror_view(tConvertedVector);
  Kokkos::deep_copy(tConvertedVectorHostMirror,tConvertedVector);

  auto tTestVectorDeviceView2D = tTestVector->getLocalView<Plato::DeviceType>(Tpetra::Access::ReadWrite);
  auto tTestVectorDeviceView1D = Kokkos::subview(tTestVectorDeviceView2D, Kokkos::ALL(), 0);
  auto tTestVectorHostMirror = Kokkos::create_mirror_view(tTestVectorDeviceView1D); 
  Kokkos::deep_copy(tTestVectorHostMirror,tTestVectorDeviceView1D);

  for(int i = 0; i < tNumDofs; ++i)
  {
    TEST_FLOATING_EQUALITY(tTestVectorHostMirror(i), tConvertedVectorHostMirror(i), 1.0e-15);
  }
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Provide input of incorrect size. Test passes if std::domain_error is thrown.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionFromTpetraVector_invalidInput )
{
  // create test mesh
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::TpetraSystem tSystem(tMesh->NumNodes(), tMachine, tNumDofsPerNode);

  auto tBogusMap = Teuchos::rcp(new Plato::Tpetra_Map(tNumDofs+1, 0, tMachine.teuchosComm));

  auto tTestVector = Teuchos::rcp(new Plato::Tpetra_MultiVector(tBogusMap,1));

  tTestVector->randomize();

  Plato::ScalarVector tConvertedVector("converted vector", tNumDofs+1);

  TEST_THROW(tSystem.toVector(tConvertedVector,tTestVector), std::domain_error);
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Provide output ScalarVector of incorrect size. Test passes if std::range_error is thrown.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionFromTpetraVector_invalidOutputContainerProvided )
{
  // create test mesh
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::TpetraSystem tSystem(tMesh->NumNodes(), tMachine, tNumDofsPerNode);

  auto tTestVector = Teuchos::rcp(new Plato::Tpetra_MultiVector(tSystem.getMap(),1));

  tTestVector->randomize();

  Plato::ScalarVector tConvertedVector("converted vector", tNumDofs+1);

  TEST_THROW(tSystem.toVector(tConvertedVector,tTestVector), std::domain_error);
}

/******************************************************************************/
/*!
  \brief 2D Elastic problem

  Construct a linear system and solve it with the Tpetra interface.  
  Test compares the numerical solution with an analytic solution.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, TpetraElastic2D )
{
    constexpr int tMeshWidth = 8;
    // *** use Tpetra solver interface *** //
    //
    constexpr auto tTpetraParameters = 
      "<ParameterList name='Linear Solver'>                                       \n"
      "  <Parameter name='Solver Stack' type='string' value='Tpetra'/>            \n"
      "  <Parameter name='Preconditioner Package' type='string' value='ifpack2'/> \n"
      "  <Parameter name='Display Iterations' type='int' value='0'/>              \n"
      "  <Parameter name='Iterations' type='int' value='50'/>                     \n"
      "  <Parameter name='Tolerance' type='double' value='1e-14'/>                \n"
      "</ParameterList>                                                           \n";
    constexpr double tRelativeTol = 1e-12;
    constexpr double tSmallTol = 1e-18;
    test_vs_analytic_2d_solution(tTpetraParameters, tMeshWidth, tRelativeTol, tSmallTol, out, success);

    // *** use Tpetra solver interface with MueLu preconditioner*** //
    //
    constexpr auto tTpetraWithMueLuParameters = 
      "<ParameterList name='Linear Solver'>                                       \n"
      "  <Parameter name='Solver Stack' type='string' value='Tpetra'/>            \n"
      "  <Parameter name='Iterations' type='int' value='500'/>                    \n"
      "  <Parameter name='Tolerance' type='double' value='1e-14'/>                \n"
      "  <Parameter name='Preconditioner Package' type='string' value='MueLu'/>   \n"
      "  <ParameterList name='Preconditioner Options'>                            \n"
      /***MueLu intput parameter list goes here***********************************/
      "     <Parameter name='tentative: calculate qr' type='bool' value='false'/> \n"
      /***************************************************************************/
      "  </ParameterList>                                                         \n"
      "</ParameterList>                                                           \n";
    constexpr double tRelativeTolForPreconditioner = 1e-11;
    test_vs_analytic_2d_solution(tTpetraWithMueLuParameters, tMeshWidth, tRelativeTolForPreconditioner, tSmallTol, out, success);
}

/******************************************************************************/
/*!
  \brief Tpetra Linear Solver will accept direct parameterlist input for
  tpetra solver and preconditioner

  Create solver with generic plato inputs and another with specific
  parameterlist inputs for a Tpetra solver and preconditioner, then solve.
  Test passes if both systems give the same solution.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, TpetraSolver_accept_parameterlist_input )
{
  // create test mesh
  //
  constexpr int meshWidth=8;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using PhysicsType = ::Plato::Elliptic::Linear::Mechanics<Plato::Tri3>;
  using ElementType = typename PhysicsType::ElementType;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::Database tDatabase;
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);
  tDatabase.vector("controls",control);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);
  tDatabase.vector("states",state);

  // create material model
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
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>       \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>               \n"
    "  <ParameterList name='Elliptic'>                                         \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
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
    "      <Parameter name='Sides'  type='string'        value='x+'/>      \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Essential Boundary Conditions'>                 \n"
    "    <ParameterList  name='X Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
    "      <Parameter  name='Index'    type='long long'    value='0'/>       \n"
#else
    "      <Parameter  name='Index'    type='int'    value='0'/>             \n"
#endif
    "      <Parameter  name='Sides'    type='string' value='x-'/>           \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
    "      <Parameter  name='Index'    type='long long'    value='1'/>       \n"
#else
    "      <Parameter  name='Index'    type='int'    value='1'/>             \n"
#endif
    "      <Parameter  name='Sides'    type='string' value='x-'/>           \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "</ParameterList>                                                        \n"
  );

  Plato::DataMap tDataMap;

  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  Plato::Elliptic::VectorFunction<PhysicsType>
    vectorFunction(tParamList->get<std::string>("PDE Constraint"), tSpatialModel, tDataMap, *tParamList);

  // compute and test constraint value
  //
  auto residual = vectorFunction.value(tDatabase,/*cycle=*/0.);

  // compute and test constraint value
  //
  auto jacobian = vectorFunction.jacobianState(tDatabase,/*cycle=*/0.);

  // parse constraints
  //
  Plato::OrdinalVector mBcDofs;
  Plato::ScalarVector mBcValues;
  Plato::DirichletBCs<ElementType>
      tEssentialBoundaryConditions(tParamList->sublist("Essential Boundary Conditions",false), tMesh);
  tEssentialBoundaryConditions.get(mBcDofs, mBcValues);
  Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(jacobian, residual, mBcDofs, mBcValues);

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  // *** use Tpetra solver interface *** //
  //
  Kokkos::deep_copy(state, 0.0);
  {
    Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Linear Solver'>                                           \n"
      "  <Parameter name='Solver Stack' type='string' value='Tpetra' />               \n"
      "  <Parameter name='Solver' type='string' value='Pseudoblock CG'/>              \n"
      "  <ParameterList name='Solver Options'>                                        \n"
      "    <Parameter name='Maximum Iterations' type='int' value='500'/>              \n"
      "    <Parameter name='Convergence Tolerance' type='double' value='1e-14'/>      \n"
      "  </ParameterList>                                                             \n"
      "  <Parameter name='Preconditioner Package' type='string' value='IFpack2'/>     \n"
      "  <Parameter name='Preconditioner Type' type='string' value='ILUT'/>           \n"
      "  <ParameterList name='Preconditioner Options'>                                \n"
      /***IFpack2 intput parameter list goes here***************************************/
      "    <Parameter name='fact: ilut level-of-fill' type='double' value='2.0'/>     \n"
      "    <Parameter name='fact: drop tolerance' type='double' value='0.0'/>         \n"
      "    <Parameter name='fact: absolute threshold' type='double' value='0.1'/>     \n"
      /*********************************************************************************/
      "  </ParameterList>                                                             \n"
      "</ParameterList>                                                               \n"
    );

    Plato::SolverFactory tSolverFactory(*tSolverParams);

    auto tSolver = tSolverFactory.create(tMesh->NumNodes(), tMachine, tNumDofsPerNode);

    tSolver->solve(*jacobian, state, residual);
  }

  Plato::ScalarVector stateTpetra_CG("state", tNumDofs);
  Kokkos::deep_copy(stateTpetra_CG, state);

  auto stateTpetra_CG_host = Kokkos::create_mirror_view(stateTpetra_CG);
  Kokkos::deep_copy(stateTpetra_CG_host, stateTpetra_CG);


  // *** use Tpetra solver interface *** //
  //
  Kokkos::deep_copy(state, 0.0);
  {
    Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Linear Solver'>                              \n"
      "  <Parameter name='Solver Stack' type='string' value='Tpetra' />  \n"
      "  <Parameter name='Display Iterations' type='int' value='0'/>     \n"
      "  <Parameter name='Iterations' type='int' value='50'/>            \n"
      "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
      "</ParameterList>                                                  \n"
    );

    Plato::SolverFactory tSolverFactory(*tSolverParams);

    auto tSolver = tSolverFactory.create(tMesh->NumNodes(), tMachine, tNumDofsPerNode);

    tSolver->solve(*jacobian, state, residual);
  }

  Plato::ScalarVector stateTpetra_default("state", tNumDofs);
  Kokkos::deep_copy(stateTpetra_default, state);

  auto stateTpetra_default_host = Kokkos::create_mirror_view(stateTpetra_default);
  Kokkos::deep_copy(stateTpetra_default_host, stateTpetra_default);
    
  // compare solutions
  

  int tLength = stateTpetra_CG_host.size();

  for(int i=0; i<tLength; i++)
  {
      if( stateTpetra_CG_host(i) > 1e-18 || stateTpetra_default_host(i) > 1e-18)
      {
          TEST_FLOATING_EQUALITY(stateTpetra_CG_host(i), stateTpetra_default_host(i), 1.0e-12);
      }
  }
}

/******************************************************************************/
/*!
  \brief Test valid input parameterlist specifying solver and preconditioner
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, TpetraSolver_valid_input )
{
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using PhysicsType = ::Plato::Elliptic::Linear::Mechanics<Plato::Tri3>;
  using ElementType = typename PhysicsType::ElementType;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                                           \n"
    "  <Parameter name='Solver Stack' type='string' value='Tpetra' />               \n"
    "  <Parameter name='Solver' type='string' value='Pseudoblock CG'/>              \n"
    "  <ParameterList name='Solver Options'>                                        \n"
    "    <Parameter name='Maximum Iterations' type='int' value='50'/>               \n"
    "    <Parameter name='Convergence Tolerance' type='double' value='1e-14'/>      \n"
    "  </ParameterList>                                                             \n"
    "  <Parameter name='Preconditioner Package' type='string' value='IFpack2'/>     \n"
    "  <Parameter name='Preconditioner Type' type='string' value='ILUT'/>           \n"
    "  <ParameterList name='Preconditioner Options'>                                \n"
    /***IFpack2 intput parameter list goes here***************************************/
    "    <Parameter name='fact: ilut level-of-fill' type='double' value='2.0'/>     \n"
    "    <Parameter name='fact: drop tolerance' type='double' value='0.0'/>         \n"
    "    <Parameter name='fact: absolute threshold' type='double' value='0.1'/>     \n"
    /*********************************************************************************/
    "  </ParameterList>                                                             \n"
    "</ParameterList>                                                               \n"
  );

  Plato::SolverFactory tSolverFactory(*tSolverParams);
}

/******************************************************************************/
/*!
  \brief Test invalid input parameterlist
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, TpetraSolver_invalid_solver_stack )
{
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using PhysicsType = ::Plato::Elliptic::Linear::Mechanics<Plato::Tri3>;
  using ElementType = typename PhysicsType::ElementType;
  int tNumDofsPerNode = ElementType::mNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                          \n"
    "  <Parameter name='Solver Stack' type='string' value='Null'/> \n"
    "  <Parameter name='Iterations' type='int' value='50'/>        \n"
    "  <Parameter name='Tolerance' type='double' value='1e-14'/>   \n"
    "</ParameterList>                                              \n"
  );

  Plato::SolverFactory tSolverFactory(*tSolverParams);
  TEST_THROW(tSolverFactory.create(tMesh->NumNodes(), tMachine, tNumDofsPerNode),std::invalid_argument);
}
#endif // PLATO_TPETRA

#ifdef PLATO_TACHO
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, TachoSolver_NonBlockMatrix)
{
  namespace pth = Plato::TestHelpers;

  const unsigned numRows = 4;
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(numRows, numRows, 1, 1) );
  std::vector<Plato::OrdinalType> tRowMapA = {0, 2, 5, 8, 10};
  std::vector<Plato::OrdinalType> tColMapA = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
  std::vector<Plato::Scalar>      tValuesA = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
  pth::set_matrix_data(tMatrixA, tRowMapA, tColMapA, tValuesA);

  std::vector<Plato::Scalar> rhs = {1.0, -1.0, 1.0, -1.0};
  Plato::ScalarVector b("b", numRows);
  pth::set_view_from_vector(b, rhs);
  
  const std::string tSolverParams = "<ParameterList name='Linear Solver'>\n"
                                    "  <Parameter name='Solver Stack' type='string' value='Tacho'/>\n"
                                    "</ParameterList>\n";

  auto tSolver = solver(tSolverParams, numRows);
  
  Plato::ScalarVector x("x", numRows);
  tSolver->solve(*tMatrixA, x, b);

  auto x_host = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(x_host, x);

  const std::vector<Plato::Scalar> x_gold = {0.4, -0.2, 0.2, -0.4};

  for(unsigned i=0; i<numRows; i++)
  {
    TEST_FLOATING_EQUALITY(x_host(i), x_gold[i], 1.0e-12);
  }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, TachoSolver_AsymmetricSparsityPatternThrows)
{
  namespace pth = Plato::TestHelpers;

  constexpr int tNumRows = 4;
  constexpr int tNumValues = 11;
  const std::vector<int> tRowBegin = {0, 3, 5, 8, tNumValues}; 
  const std::vector<int> tColumns = {0, 1, 2, 0, 1, 2, 1, 2, 3, 2, 3};
  const std::vector<double> tValues = {2.0, -1.0, 1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(tNumRows, tNumRows, 1, 1) );
  pth::set_matrix_data(tMatrixA, tRowBegin, tColumns, tValues);

  std::vector<Plato::Scalar> tRhs = {1.0, -1.0, 1.0, -1.0};
  Plato::ScalarVector tB("b", tNumRows);
  pth::set_view_from_vector(tB, tRhs);

  const std::string tSolverParams = "<ParameterList name='Linear Solver'>\n"
                                    "  <Parameter name='Solver Stack' type='string' value='Tacho'/>\n"
                                    "</ParameterList>\n";
  auto tSolver = solver(tSolverParams, tNumRows);

  Plato::ScalarVector tX("x", tNumRows);
  TEST_THROW(tSolver->solve(*tMatrixA, tX, tB), std::runtime_error);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, TachoSolver_BlockMatrix)
{
  /*
    2    -1    0    0
    -1    2   -1    0
     0   -1    2   -1
     0    0   -1    2
  */
  namespace pth = Plato::TestHelpers;

  constexpr unsigned numRows = 4;
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(numRows, numRows, 2, 2) );
  std::vector<Plato::OrdinalType> tRowMapA = {0, 2, 4};
  std::vector<Plato::OrdinalType> tColMapA = {0, 1, 0, 1};
  std::vector<Plato::Scalar>      tValuesA = {2.0, -1.0, -1.0, 2.0,
                                              0.0, 0.0, -1.0, 0.0,
                                              0.0, -1.0, 0.0, 0.0,
                                              2.0, -1.0, -1.0, 2.0};
  pth::set_matrix_data(tMatrixA, tRowMapA, tColMapA, tValuesA);

  std::vector<Plato::Scalar> rhs = {1.0, -1.0, 1.0, -1.0};
  Plato::ScalarVector b("b", numRows);
  Plato::ScalarVector x("x", numRows);
  pth::set_view_from_vector(b, rhs);
  
  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Linear Solver'>                                       \n"
      "  <Parameter name='Solver Stack' type='string' value='Tacho'/>            \n"
      "</ParameterList>                                                           \n"
  );
  
  Plato::SolverFactory tSolverFactory(*tSolverParams);
  
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  auto tSolver = tSolverFactory.create(numRows, tMachine, 1);
  
  tSolver->solve(*tMatrixA, x, b);

  auto x_host = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(x_host, x);

  constexpr std::array<Plato::Scalar, 4> x_gold = {0.4, -0.2, 0.2, -0.4};

  for(unsigned i=0; i<numRows; i++)
  {
    TEST_FLOATING_EQUALITY(x_host(i), x_gold[i], 1.0e-12);
  }
}

/******************************************************************************/
/*!
  \brief 2D Elastic problem

  Construct a linear system and solve it the Tacho interface. Test compares 
  numerical solution with an analytic solution.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, TachoElastic2D )
{
    constexpr int tMeshWidth = 8;
    // *** use Tacho solver interface *** //
    //
    constexpr auto tTachoParameters = 
      "<ParameterList name='Linear Solver'>                           \n"
      "  <Parameter name='Solver Stack' type='string' value='Tacho'/> \n"
      "</ParameterList>                                               \n";
    constexpr double tRelativeTol = 1e-12;
    constexpr double tSmallTol = 1e-18;
    test_vs_analytic_2d_solution(tTachoParameters, tMeshWidth, tRelativeTol, tSmallTol, out, success);
}
#endif // PLATO_TACHO

#ifdef HAVE_AMGX
/******************************************************************************/
/*!
  \brief 2D Elastic problem

  Construct a linear system and solve it with the AmgX interface.  
  Test compares the numerical solution with an analytic solution.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, AmgXElastic2D )
{
    constexpr int tMeshWidth = 8;
    // *** use new AmgX solver interface *** //
    //
    constexpr auto tAmgxSolverParameters = 
      "<ParameterList name='Linear Solver'>                                     \n"
      "  <Parameter name='Solver Stack' type='string' value='AmgX'/>            \n"
      "  <Parameter name='Configuration File' type='string' value='amgx.json'/> \n"
      "</ParameterList>                                                         \n";
    constexpr double tRelativeTol = 1e-12;
    constexpr double tSmallTol = 1e-18;
    test_vs_analytic_2d_solution(tAmgxSolverParameters, tMeshWidth, tRelativeTol, tSmallTol, out, success);
}

#endif // HAVE_AMGX
