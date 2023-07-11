/*
 * ThermoelastostaticTotalLagrangianTests.cpp
 *
 *  Created on: June 14, 2023
 */

// trilinos includes
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

// unit test includes
#include "util/PlatoTestHelpers.hpp"

// analyze includes
#include "Tri3.hpp"
#include "PlatoMathTypes.hpp"
#include "ApplyConstraints.hpp"
#include "InterpolateFromNodal.hpp"

#include "Tet10.hpp"
#include "elliptic/Problem.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/base/WorksetBuilder.hpp"

#include "elliptic/mechanical/nonlinear/NominalStressTensor.hpp"
#include "elliptic/mechanical/nonlinear/KineticPullBackOperation.hpp"

#include "elliptic/thermomechanics/nonlinear/ThermoMechanics.hpp"
#include "elliptic/thermomechanics/nonlinear/UtilitiesThermoMechanics.hpp"
#include "elliptic/thermomechanics/nonlinear/ThermalDeformationGradient.hpp"
#include "elliptic/thermomechanics/nonlinear/ThermoElasticDeformationGradient.hpp"
#include "elliptic/thermomechanics/nonlinear/ResidualThermoElastoStaticTotalLagrangian.hpp"


namespace ThermoelastostaticTotalLagrangianTests
{

Teuchos::RCP<Teuchos::ParameterList> tGenericParamList = Teuchos::getParametersFromXmlString(
"<ParameterList name='Plato Problem'>                                                                      \n"
  "<Parameter name='PDE Constraint' type='string' value='Elliptic'/>                                       \n"
  "<Parameter name='Physics'        type='string' value='Thermomechanical'/>                               \n"
  "<Parameter name='Response'       type='string' value='Nonlinear'/>                                      \n"
  "<Parameter name='Coupling'       type='string' value='Staggered'/>                                      \n"
  "<ParameterList name='Spatial Model'>                                                                    \n"
    "<ParameterList name='Domains'>                                                                        \n"
      "<ParameterList name='Design Volume'>                                                                \n"
        "<Parameter name='Element Block' type='string' value='body'/>                                      \n"
        "<Parameter name='Material Model' type='string' value='Unobtainium'/>                              \n"
      "</ParameterList>                                                                                    \n"
    "</ParameterList>                                                                                      \n"
  "</ParameterList>                                                                                        \n"
  "<ParameterList name='Elliptic'>                                                                         \n"
  "  <ParameterList name='Mechanical Residual'>                                                            \n"
  "    <Parameter name='Response' type='string' value='Nonlinear'/>                                        \n"
  "    <ParameterList name='Penalty Function'>                                                             \n"
  "      <Parameter name='Exponent' type='double' value='1.0'/>                                            \n"
  "      <Parameter name='Minimum Value' type='double' value='0.0'/>                                       \n"
  "      <Parameter name='Type' type='string' value='SIMP'/>                                               \n"
  "    </ParameterList>                                                                                    \n"
  "  </ParameterList>                                                                                      \n"
  "  <ParameterList name='Thermal Residual'>                                                               \n"
  "    <Parameter name='Response' type='string' value='Linear'/>                                           \n"
  "    <ParameterList name='Penalty Function'>                                                             \n"
  "      <Parameter name='Exponent' type='double' value='1.0'/>                                            \n"
  "      <Parameter name='Minimum Value' type='double' value='0.0'/>                                       \n"
  "      <Parameter name='Type' type='string' value='SIMP'/>                                               \n"
  "    </ParameterList>                                                                                    \n"
  "  </ParameterList>                                                                                      \n"
  "</ParameterList>                                                                                        \n"
  "<ParameterList name='Material Models'>                                                                  \n"
    "<ParameterList name='Unobtainium'>                                                                    \n"
      "<ParameterList name='Thermal Conduction'>                                                           \n"
        "<Parameter  name='Thermal Expansivity'   type='double' value='0.5'/>                              \n"
        "<Parameter  name='Reference Temperature' type='double' value='1.0'/>                              \n"
        "<Parameter  name='Thermal Conductivity'  type='double' value='2.0'/>                              \n"
      "</ParameterList>                                                                                    \n"
      "<ParameterList name='Hyperelastic Kirchhoff'>                                                       \n"
        "<Parameter  name='Youngs Modulus'  type='double'  value='1.5'/>                                   \n"
        "<Parameter  name='Poissons Ratio'  type='double'  value='0.35'/>                                  \n"
      "</ParameterList>                                                                                    \n"
    "</ParameterList>                                                                                      \n"
  "</ParameterList>                                                                                        \n"
  "<ParameterList name='Mechanical Natural Boundary Conditions'>                                           \n"
  "  <ParameterList name='Uniform Traction Force'>                                                         \n"
  "    <Parameter name='Type'   type='string'        value='Uniform'/>                                     \n"
  "    <Parameter name='Values' type='Array(string)' value='{0,1e5,0}'/>                                   \n"
  "    <Parameter name='Sides'  type='string'        value='x+'/>                                          \n"
  "  </ParameterList>                                                                                      \n"
  "</ParameterList>                                                                                        \n"
  "<ParameterList name='Thermal Natural Boundary Conditions'>                                              \n"
  "  <ParameterList name='Uniform Thermal Flux'>                                                           \n"
  "    <Parameter name='Type'  type='string' value='Uniform'/>                                             \n"
  "    <Parameter name='Value' type='string' value='-1.0e3'/>                                              \n"
  "    <Parameter name='Sides' type='string' value='x+'/>                                                  \n"
  "  </ParameterList>                                                                                      \n"
  "</ParameterList>                                                                                        \n"
  "<ParameterList name='Mechanical Essential Boundary Conditions'>                                         \n"
  "  <ParameterList name='Displacement X'>                                                                 \n"
  "    <Parameter name='Type'  type='string' value='Fixed Value'/>                                         \n"
  "    <Parameter name='Index' type='int'    value='0'/>                                                   \n"
  "    <Parameter name='Sides' type='string' value='x-'/>                                                  \n"
  "    <Parameter name='Value' type='double' value='0.0'/>                                                 \n"
  "  </ParameterList>                                                                                      \n"
  "  <ParameterList name='Displacement Y'>                                                                 \n"
  "    <Parameter name='Type'  type='string' value='Fixed Value'/>                                         \n"
  "    <Parameter name='Index' type='int'    value='1'/>                                                   \n"
  "    <Parameter name='Sides' type='string' value='x-' />                                                 \n"
  "    <Parameter name='Value' type='double' value='0.0' />                                                \n"
  "  </ParameterList>                                                                                      \n"
  "</ParameterList>                                                                                        \n"
  "<ParameterList name='Thermal Essential Boundary Conditions'>                                            \n"
  "  <ParameterList name='Temperature'>                                                                    \n"
  "    <Parameter name='Type'  type='string' value='Fixed Value'/>                                         \n"
  "    <Parameter name='Index' type='int'    value='0'/>                                                   \n"
  "    <Parameter name='Sides' type='string' value='x-'/>                                                  \n"
  "    <Parameter name='Value' type='double' value='0.0'/>                                                 \n"
  "  </ParameterList>                                                                                      \n"
  "</ParameterList>                                                                                        \n"
"</ParameterList>                                                                                          \n"
); 

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, ThermoElastoStaticResidual2D_Solution )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // mpi wrapper
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);
  // create elliptic problem evaluator
  Plato::Elliptic::Problem<Plato::Elliptic::Nonlinear::ThermoMechanics<Plato::Tri3>>
    tThermoMechanicsProblem(tMesh,*tGenericParamList,tMachine);
  // solve system of equations
  auto tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector tControl("Control",tNumVerts);
  Plato::blas1::fill(1.0,tControl);
  auto tSolution = tThermoMechanicsProblem.solution(tControl);
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, tComputeThermalDefGrad )
{
  Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                             \n"
    "<ParameterList name='Material Models'>                                         \n"
      "<ParameterList name='Unobtainium'>                                           \n"
        "<ParameterList name='Thermal Conduction'>                                  \n"
          "<Parameter  name='Thermal Expansivity'   type='double' value='0.5'/>     \n"
          "<Parameter  name='Reference Temperature' type='double' value='1.0'/>     \n"
          "<Parameter  name='Thermal Conductivity'  type='double' value='2.0'/>     \n"
        "</ParameterList>                                                           \n"
      "</ParameterList>                                                             \n"
    "</ParameterList>                                                               \n"
  "</ParameterList>                                                                 \n"
  ); 
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using ResultT     = typename Residual::ResultScalarType;
  using ConfigT     = typename Residual::ConfigScalarType;
  using NodeStateT  = typename Residual::NodeStateScalarType;
  // create temperature workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumNodes = tMesh->NumNodes();
  Plato::ScalarVector tTemp("Temps", tNumNodes);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill node state",
    Kokkos::RangePolicy<>(0, tNumNodes), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<NodeStateT> tTempWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tTemp, tTempWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get integration rule information
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // compute thermal deformation gradient
  Plato::ThermalDeformationGradient<Residual> tComputeThermalDeformationGradient("Unobtainium",*tParamList);
  Plato::InterpolateFromNodal<ElementType,ElementType::mNumNodeStatePerNode> tInterpolateFromNodal;
  Kokkos::parallel_for("compute thermal deformation gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint = tCubPoints(iGpOrdinal);
    auto tBasisValues = ElementType::basisValues(tCubPoint);
    // interpolate temperature from nodes to integration points
    NodeStateT tTemperature = tInterpolateFromNodal(iCellOrdinal,tBasisValues,tTempWS);
    // compute thermal deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,NodeStateT> tTempDefGrad;
    tComputeThermalDeformationGradient(tTemperature,tTempDefGrad);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tTempDefGrad(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.5166666666666666,0.,0.,0.5166666666666666}, {0.5166666666666666,0.,0.,0.5166666666666666} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, tComputeThermoElasticDefGrad )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using VecStateT   = typename Residual::StateScalarType;
  using ResultT     = typename Residual::ResultScalarType;
  using ConfigT     = typename Residual::ConfigScalarType;
  using NodeStateT  = typename Residual::NodeStateScalarType;
  using StrainT     = typename Plato::fad_type_t<ElementType,VecStateT,ConfigT>;
  // create thermal gradient workset
  std::vector<std::vector<Plato::Scalar>> tTempDefGrad = 
    { {0.5166666666666666,0.,0.,0.5166666666666666}, {0.5166666666666666,0.,0.,0.5166666666666666} };
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  Plato::ScalarArray3DT<NodeStateT> tTempDefGradWS("thermal deformation gradient",tNumCells,tSpaceDim,tSpaceDim);
  auto tHostTempDefGradWS = Kokkos::create_mirror(tTempDefGradWS);
  Kokkos::deep_copy(tHostTempDefGradWS, tTempDefGradWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHostTempDefGradWS(tCell,tDimI,tDimJ) = tTempDefGrad[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(tTempDefGradWS,tHostTempDefGradWS);
  // create mechanical gradient workset
  std::vector<std::vector<Plato::Scalar>> tMechDefGrad = 
    { {1.4,0.2,0.4,1.2}, {1.4,0.2,0.4,1.2} };
  Plato::ScalarArray3DT<StrainT> tMechDefGradWS("mechanical deformation gradient",tNumCells,tSpaceDim,tSpaceDim);
  auto tHostMechDefGradWS = Kokkos::create_mirror(tMechDefGradWS);
  Kokkos::deep_copy(tHostMechDefGradWS,tMechDefGradWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHostMechDefGradWS(tCell,tDimI,tDimJ) = tMechDefGrad[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(tMechDefGradWS,tHostMechDefGradWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // compute thermo-elastic deformation gradient
  auto tNumPoints = ElementType::mNumGaussPoints;
  Plato::ThermoElasticDeformationGradient<Residual> tComputeThermoElasticDeformationGradient;
  Kokkos::parallel_for("compute thermo-elastic deformation gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
      tCellMechDefGrad(StrainT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCellMechDefGrad(tDimI,tDimJ) = tMechDefGradWS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,NodeStateT> 
      tCellTempDefGrad(NodeStateT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCellTempDefGrad(tDimI,tDimJ) = tTempDefGradWS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    // compute thermo-elastic deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> 
      tTMechDefGrad(ResultT(0.));
    tComputeThermoElasticDeformationGradient(tCellTempDefGrad,tCellMechDefGrad,tTMechDefGrad);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tTMechDefGrad(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.72333333,0.10333333,0.20666667,0.62}, {0.72333333,0.10333333,0.20666667,0.62} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, KineticPullBackOperation_1 )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using VecStateT   = typename Residual::StateScalarType;
  using ResultT     = typename Residual::ResultScalarType;
  using ConfigT     = typename Residual::ConfigScalarType;
  using StrainT     = typename Plato::fad_type_t<ElementType,VecStateT,ConfigT>;
  // create mechanical deformation gradient workset
  std::vector<std::vector<Plato::Scalar>> tDataDefGrad = 
    { {1.4,0.2,0.4,1.2}, {1.4,0.2,0.4,1.2} };
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  Plato::ScalarArray3DT<StrainT> tDefGradient("mechanical deformation gradient",tNumCells,tSpaceDim,tSpaceDim);
  auto tHostDefGradWS = Kokkos::create_mirror(tDefGradient);
  Kokkos::deep_copy(tHostDefGradWS, tDefGradient);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHostDefGradWS(tCell,tDimI,tDimJ) = tDataDefGrad[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(tDefGradient,tHostDefGradWS);
  // create second piola-kirchhoff stress data
  std::vector<std::vector<Plato::Scalar>> tData2PKS = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };  
  Plato::ScalarArray3DT<ResultT> t2PKS_WS("second piola-kirchhoff stress",tNumCells,tSpaceDim,tSpaceDim);
  auto tHost2PKS_WS = Kokkos::create_mirror(t2PKS_WS);
  Kokkos::deep_copy(tHost2PKS_WS, t2PKS_WS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHost2PKS_WS(tCell,tDimI,tDimJ) = tData2PKS[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(t2PKS_WS,tHost2PKS_WS);
  // create results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // pull back second piola-kirchhoff stress from deformed to undeformed configuration
  auto tNumPoints = ElementType::mNumGaussPoints;
  Plato::KineticPullBackOperation<Residual> tApplyKineticPullBackOperation;
  Kokkos::parallel_for("pull back stress operation", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tDefConfig2PKS(ResultT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tDefConfig2PKS(tDimI,tDimJ) = t2PKS_WS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tCellDefGrad(StrainT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCellDefGrad(tDimI,tDimJ) = tDefGradient(iCellOrdinal,tDimI,tDimJ);
      }
    }
    // compute thermo-elastic deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tUndefConfig2PKS(ResultT(0.));
    tApplyKineticPullBackOperation(tCellDefGrad,tDefConfig2PKS,tUndefConfig2PKS);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tUndefConfig2PKS(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.9328371,-0.1743207,-0.1743207,0.9782719}, {0.9328371,-0.1743207,-0.1743207,0.9782719} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, KineticPullBackOperation_2 )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using VecStateT   = typename Residual::StateScalarType;
  using ResultT     = typename Residual::ResultScalarType;
  using ConfigT     = typename Residual::ConfigScalarType;
  using NodeStateT  = typename Residual::NodeStateScalarType;
  using StrainT     = typename Plato::fad_type_t<ElementType,VecStateT,ConfigT>;
  // create thermal gradient workset
  std::vector<std::vector<Plato::Scalar>> tDataTempDefGrad = 
    { {0.5166666666666666,0.,0.,0.5166666666666666}, {0.5166666666666666,0.,0.,0.5166666666666666} };
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  Plato::ScalarArray3DT<NodeStateT> tTempDefGradWS("thermal deformation gradient",tNumCells,tSpaceDim,tSpaceDim);
  auto tHostTempDefGradWS = Kokkos::create_mirror(tTempDefGradWS);
  Kokkos::deep_copy(tHostTempDefGradWS, tTempDefGradWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHostTempDefGradWS(tCell,tDimI,tDimJ) = tDataTempDefGrad[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(tTempDefGradWS,tHostTempDefGradWS);
  // create second piola-kirchhoff stress data
  std::vector<std::vector<Plato::Scalar>> tData2PKS = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };  
  Plato::ScalarArray3DT<ResultT> t2PKS_WS("second piola-kirchhoff stress",tNumCells,tSpaceDim,tSpaceDim);
  auto tHost2PKS_WS = Kokkos::create_mirror(t2PKS_WS);
  Kokkos::deep_copy(tHost2PKS_WS, t2PKS_WS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHost2PKS_WS(tCell,tDimI,tDimJ) = tData2PKS[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(t2PKS_WS,tHost2PKS_WS);
  // create results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // pull back second piola-kirchhoff stress from deformed to undeformed configuration
  auto tNumPoints = ElementType::mNumGaussPoints;
  Plato::KineticPullBackOperation<Residual> tApplyKineticPullBackOperation;
  Kokkos::parallel_for("pull back stress operation", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tDefConfig2PKS(ResultT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tDefConfig2PKS(tDimI,tDimJ) = t2PKS_WS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,NodeStateT> 
      tCellTempDefGrad(NodeStateT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCellTempDefGrad(tDimI,tDimJ) = tTempDefGradWS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    // compute thermo-elastic deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tUndefConfig2PKS(ResultT(0.));
    tApplyKineticPullBackOperation(tCellTempDefGrad,tDefConfig2PKS,tUndefConfig2PKS);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tUndefConfig2PKS(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, NominalStressTensor )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using VecStateT   = typename Residual::StateScalarType;
  using ResultT     = typename Residual::ResultScalarType;
  using ConfigT     = typename Residual::ConfigScalarType;
  using StrainT     = typename Plato::fad_type_t<ElementType,VecStateT,ConfigT>;
  // create mechanical deformation gradient workset
  std::vector<std::vector<Plato::Scalar>> tDataDefGrad = 
    { {1.4,0.2,0.4,1.2}, {1.4,0.2,0.4,1.2} };
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  Plato::ScalarArray3DT<StrainT> tDefGradient("mechanical deformation gradient",tNumCells,tSpaceDim,tSpaceDim);
  auto tHostDefGradWS = Kokkos::create_mirror(tDefGradient);
  Kokkos::deep_copy(tHostDefGradWS, tDefGradient);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHostDefGradWS(tCell,tDimI,tDimJ) = tDataDefGrad[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(tDefGradient,tHostDefGradWS);
  // create second piola-kirchhoff stress data
  std::vector<std::vector<Plato::Scalar>> tData2PKS = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };  
  Plato::ScalarArray3DT<ResultT> t2PKS_WS("second piola-kirchhoff stress",tNumCells,tSpaceDim,tSpaceDim);
  auto tHost2PKS_WS = Kokkos::create_mirror(t2PKS_WS);
  Kokkos::deep_copy(tHost2PKS_WS, t2PKS_WS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHost2PKS_WS(tCell,tDimI,tDimJ) = tData2PKS[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(t2PKS_WS,tHost2PKS_WS);
  // create results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // pull back second piola-kirchhoff stress from deformed to undeformed configuration
  auto tNumPoints = ElementType::mNumGaussPoints;
  Plato::NominalStressTensor<Residual> tComputeNominalStressTensor;
  Kokkos::parallel_for("pull back stress operation", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tCell2PKS(ResultT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCell2PKS(tDimI,tDimJ) = t2PKS_WS(iCellOrdinal,tDimI,tDimJ);
      }
    }
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tCellDefGrad(StrainT(0.));
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tCellDefGrad(tDimI,tDimJ) = tDefGradient(iCellOrdinal,tDimI,tDimJ);
      }
    }
    // compute thermo-elastic deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> tNominalStressTensor(ResultT(0.));
    tComputeNominalStressTensor(tCellDefGrad,tCell2PKS,tNominalStressTensor);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tNominalStressTensor(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {1.6049342,0.7802452,0.5679006,1.1555556}, {1.6049342,0.7802452,0.5679006,1.1555556} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, getCell2PKS )
{
  //set ad-types
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  // create second piola-kirchhoff stress data
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tNumCells = 2;
  std::vector<std::vector<Plato::Scalar>> tData2PKS = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };
  Plato::ScalarArray4D t2PKS_WS("second piola-kirchhoff stress",tNumCells,/*num_intg_pts=*/1,tSpaceDim,tSpaceDim);
  auto tHost2PKS_WS = Kokkos::create_mirror(t2PKS_WS);
  Kokkos::deep_copy(tHost2PKS_WS, t2PKS_WS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        tHost2PKS_WS(tCell,0,tDimI,tDimJ) = tData2PKS[tCell][tDimI*tSpaceDim+tDimJ];
      }
    }
  }
  Kokkos::deep_copy(t2PKS_WS,tHost2PKS_WS);
  // create results workset
  Plato::ScalarArray3D tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get cell 2PKS
  auto tNumPoints = ElementType::mNumGaussPoints;
  Kokkos::parallel_for("get cell 2PKS", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal,const Plato::OrdinalType iGpOrdinal)
  {

    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims> tCell2PKS(0.);
    Plato::Elliptic::getCell2PKS(iCellOrdinal,iGpOrdinal,t2PKS_WS,tCell2PKS);
    // copy output to result workset
    for( Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      for( Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        tResultsWS(iCellOrdinal,tDimI,tDimJ) = tCell2PKS(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {1.10617,0.281481,0.281481,0.869136}, {1.10617,0.281481,0.281481,0.869136} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ThermoelastostaticTotalLagrangianTests, Residual )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  //set ad-types
  using ElementType  = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using ResidualEval = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using VecStateT    = typename ResidualEval::StateScalarType;
  using ResultT      = typename ResidualEval::ResultScalarType;
  using ConfigT      = typename ResidualEval::ConfigScalarType;
  using NodeStateT   = typename ResidualEval::NodeStateScalarType;
  using StrainT      = typename Plato::fad_type_t<ElementType,VecStateT,ConfigT>;
  // create displacement workset
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tDisp);
  // create temperature workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  Plato::ScalarVector tTemp("Temps", tNumVerts);
  Plato::blas1::fill(1., tTemp);
  Kokkos::parallel_for("fill temperature",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("node states",tTemp);
  // create control workset
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(1.0, tControl);
  tDatabase.vector("controls",tControl);
  // create workset database
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<ResidualEval> tWorksetBuilder(tWorksetFuncs);
  tWorksetBuilder.build(tOnlyDomainDefined, tDatabase, tWorkSets);
  auto tNumCells = tMesh->NumElements();
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultT> > >
    ( Plato::ScalarMultiVectorT<ResultT>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  tWorkSets.set("result",tResultWS);
  // evaluate residualeval
  Plato::Elliptic::ResidualThermoElastoStaticTotalLagrangian<ResidualEval> 
    tResidual(tOnlyDomainDefined,tDataMap,*tGenericParamList);
  tResidual.evaluate(tWorkSets,/*cycle=*/0.);
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { 
      {-1.6049390,-0.567902,0.824691,-0.587654,0.78024773,1.155556}, 
      {-0.6827165,-1.011111,1.404321, 0.496914,-0.7216045,0.514197} 
    };
  auto tHostResultsWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultsWS, tResultWS->mData);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      TEST_FLOATING_EQUALITY(tGold[tCell][tDof],tHostResultsWS(tCell,tDof),tTolerance);
    }
  }
}

} 