/*
 * ElastostaticTotalLagrangianTests.cpp
 *
 *  Created on: May 25, 2023
 */

// trilinos includes
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

// unit test includes
#include "util/PlatoTestHelpers.hpp"

// plato
#include "Tri3.hpp"
#include "BLAS1.hpp"
#include "WorksetBase.hpp"
#include "MaterialModel.hpp"
#include "GradientMatrix.hpp"
#include "MechanicsElement.hpp"

#include "elliptic/mechanical/nonlinear/StateGradient.hpp"
#include "elliptic/mechanical/nonlinear/DeformationGradient.hpp"
#include "elliptic/mechanical/nonlinear/RightDeformationTensor.hpp"
#include "elliptic/mechanical/nonlinear/GreenLagrangeStrainTensor.hpp"
#include "elliptic/mechanical/nonlinear/KirchhoffSecondPiolaStress.hpp"
#include "elliptic/mechanical/nonlinear/NeoHookeanSecondPiolaStress.hpp"
#include "elliptic/mechanical/nonlinear/StressEvaluatorKirchhoff.hpp"
#include "elliptic/mechanical/nonlinear/StressEvaluatorNeoHookean.hpp"
#include "elliptic/mechanical/nonlinear/ResidualElastostaticTotalLagrangian.hpp"
#include "elliptic/mechanical/nonlinear/CriterionKirchhoffEnergyPotential.hpp"
#include "elliptic/mechanical/nonlinear/CriterionNeoHookeanEnergyPotential.hpp"

namespace ElastostaticTotalLagrangianTests
{

Teuchos::RCP<Teuchos::ParameterList> tGenericParamList = Teuchos::getParametersFromXmlString(
"<ParameterList name='Plato Problem'>                                                                  \n"
  "<ParameterList name='Spatial Model'>                                                                \n"
    "<ParameterList name='Domains'>                                                                    \n"
      "<ParameterList name='Design Volume'>                                                            \n"
        "<Parameter name='Element Block' type='string' value='body'/>                                  \n"
        "<Parameter name='Material Model' type='string' value='Mystic'/>                               \n"
      "</ParameterList>                                                                                \n"
    "</ParameterList>                                                                                  \n"
  "</ParameterList>                                                                                    \n"
  "<ParameterList name='Material Models'>                                                              \n"
    "<ParameterList name='Mystic'>                                                                     \n"
      "<ParameterList name='Kirchhoff'>                                                                \n"
        "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>                                \n"
        "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>                                 \n"
      "</ParameterList>                                                                                \n"
    "</ParameterList>                                                                                  \n"
  "</ParameterList>                                                                                    \n"
  "<ParameterList name='Criteria'>                                                                     \n"
  "  <ParameterList name='Objective'>                                                                  \n"
  "    <Parameter name='Type' type='string' value='Weighted Sum'/>                                     \n"
  "    <Parameter name='Functions' type='Array(string)' value='{My Strain Energy}'/>                   \n"
  "    <Parameter name='Weights' type='Array(double)' value='{1.0}'/>                                  \n"
  "  </ParameterList>                                                                                  \n"
  "  <ParameterList name='My Strain Energy'>                                                           \n"
  "    <Parameter name='Type'                 type='string' value='Scalar Function'/>                  \n"
  "    <Parameter name='Scalar Function Type' type='string' value='Strain Energy Potential'/>          \n"
  "  </ParameterList>                                                                                  \n"
  "</ParameterList>                                                                                    \n"
"</ParameterList>                                                                                      \n"
); 

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, tComputeStateGradient )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get integration rule information
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // compute state gradient
  Plato::StateGradient<Residual> tComputeStateGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Kokkos::parallel_for("compute state gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // compute gradient functions
      ConfigT tVolume(0.0);
      Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
      auto tCubPoint = tCubPoints(iGpOrdinal);
      tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStateGradient(StrainT(0.));
      tComputeStateGradient(iCellOrdinal,tStateWS,tGradient,tStateGradient);
      // copy results
      for( Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
        for( Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
          tResultsWS(iCellOrdinal,tDimI,tDimJ) = tStateGradient(tDimI,tDimJ);
        }
      }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.4,0.2,0.4,0.2}, {0.4,0.2,0.4,0.2} };
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

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, DeformationGradient )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get integration rule information
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // compute state gradient
  Plato::StateGradient<Residual> tComputeStateGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::DeformationGradient<Residual> tComputeDeformationGradient;
  Kokkos::parallel_for("compute state gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // compute gradient functions
      ConfigT tVolume(0.0);
      Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
      auto tCubPoint = tCubPoints(iGpOrdinal);
      tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStateGradient(StrainT(0.));
      tComputeStateGradient(iCellOrdinal,tStateWS,tGradient,tStateGradient);
      // compute deformation gradient 
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefGradient(StrainT(0.));
      tComputeDeformationGradient(tStateGradient,tDefGradient);
      // copy result
      for( Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
        for( Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
          tResultsWS(iCellOrdinal,tDimI,tDimJ) = tDefGradient(tDimI,tDimJ);
        }
      }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {1.4,0.2,0.4,1.2}, {1.4,0.2,0.4,1.2} };
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

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, RightDeformationTensor )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get integration rule information
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // compute state gradient
  Plato::StateGradient<Residual> tComputeStateGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::DeformationGradient<Residual> tComputeDeformationGradient;
  Plato::RightDeformationTensor<Residual> tComputeRightDeformationTensor;
  Kokkos::parallel_for("compute state gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // compute gradient functions
      ConfigT tVolume(0.0);
      Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
      auto tCubPoint = tCubPoints(iGpOrdinal);
      tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStateGradient(StrainT(0.));
      tComputeStateGradient(iCellOrdinal,tStateWS,tGradient,tStateGradient);
      // compute deformation gradient 
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefGradient(StrainT(0.));
      tComputeDeformationGradient(tStateGradient,tDefGradient);
      // apply transpose to deformation gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefGradT
        = Plato::transpose(tDefGradient);
      // compute cauchy-green deformation tensor
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefTensor(StrainT(0.));
      tComputeRightDeformationTensor(tDefGradT,tDefGradient,tDefTensor);
      // copy result
      for( Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
        for( Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
          tResultsWS(iCellOrdinal,tDimI,tDimJ) = tDefTensor(tDimI,tDimJ);
        }
      }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {2.12,0.76,0.76,1.48}, {2.12,0.76,0.76,1.48} };
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

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, GreenLagrangeStrainTensor )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get integration rule information
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // compute state gradient
  Plato::StateGradient<Residual> tComputeStateGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::DeformationGradient<Residual> tComputeDeformationGradient;
  Plato::RightDeformationTensor<Residual> tComputeRightDeformationTensor;
  Plato::GreenLagrangeStrainTensor<Residual> tGreenLagrangeStrainTensor;
  Kokkos::parallel_for("compute state gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // compute gradient functions
      ConfigT tVolume(0.0);
      Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
      auto tCubPoint = tCubPoints(iGpOrdinal);
      tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStateGradient(StrainT(0.));
      tComputeStateGradient(iCellOrdinal,tStateWS,tGradient,tStateGradient);
      // compute green-lagrange strain tensor
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStrainTensor(StrainT(0.));
      tGreenLagrangeStrainTensor(tStateGradient,tStrainTensor);
      // copy result
      for( Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
        for( Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
          tResultsWS(iCellOrdinal,tDimI,tDimJ) = tStrainTensor(tDimI,tDimJ);
        }
      }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.56,0.38,0.38,0.24}, {0.56,0.38,0.38,0.24} };
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

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, StressEvaluatorKirchhoffTensor )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  // create output database and spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(1.0, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  // results workset
  Plato::OrdinalType tNumGaussPoints = ElementType::mNumGaussPoints;
  Plato::ScalarArray4DT<ResultT> tResultsWS("stress",tNumCells,tNumGaussPoints,tSpaceDim,tSpaceDim);
  Plato::StressEvaluatorKirchhoff<Residual> tStressEvaluator("Mystic",*tGenericParamList,tOnlyDomainDefined,tDataMap);
  tStressEvaluator.evaluate(tStateWS,tControlWS,tConfigWS,tResultsWS);
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {1.60493827,0.78024691,0.56790123,1.15555556}, {1.60493827,0.78024691,0.56790123,1.15555556} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,0,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, StressEvaluatorNeoHookeanTensor )
{
  // create parameter list
  Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                                                  \n"
    "<ParameterList name='Spatial Model'>                                                                \n"
      "<ParameterList name='Domains'>                                                                    \n"
        "<ParameterList name='Design Volume'>                                                            \n"
          "<Parameter name='Element Block' type='string' value='body'/>                                  \n"
          "<Parameter name='Material Model' type='string' value='Mystic'/>                               \n"
        "</ParameterList>                                                                                \n"
      "</ParameterList>                                                                                  \n"
    "</ParameterList>                                                                                    \n"
    "<ParameterList name='Material Models'>                                                              \n"
      "<ParameterList name='Mystic'>                                                                     \n"
        "<ParameterList name='Neo-Hookean'>                                                              \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>                                \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>                                 \n"
        "</ParameterList>                                                                                \n"
      "</ParameterList>                                                                                  \n"
    "</ParameterList>                                                                                    \n"
  "</ParameterList>                                                                                      \n"
  );
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  // create output database and spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(1.0, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  // results workset
  Plato::OrdinalType tNumGaussPoints = ElementType::mNumGaussPoints;
  Plato::ScalarArray4DT<ResultT> tResultsWS("stress",tNumCells,tNumGaussPoints,tSpaceDim,tSpaceDim);
  Plato::StressEvaluatorNeoHookean<Residual> 
    tStressEvaluator("Mystic",*tParamList,tOnlyDomainDefined,tDataMap);
  tStressEvaluator.evaluate(tStateWS,tControlWS,tConfigWS,tResultsWS);
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.54537272,0.14367245,0.06512267,0.47577435}, {0.54537272,0.14367245,0.06512267,0.47577435} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,0,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, Residual )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  // create output database and spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(1.0, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  // create results workset
  Plato::OrdinalType tNumGaussPoints = ElementType::mNumGaussPoints;
  Plato::ScalarMultiVectorT<ResultT> tResultsWS("residual",tNumCells,tDofsPerCell);
  // evaluate residual
  Plato::ResidualElastostaticTotalLagrangian<Residual> 
    tResidual(tOnlyDomainDefined,tDataMap,*tGenericParamList);
  tResidual.evaluate(tStateWS,tControlWS,tConfigWS,tResultsWS,0.0);
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { 
      {-0.802469,-0.28395 ,0.412346,-0.293827,0.390123 ,0.577778}, 
      {-0.390123,-0.577778,0.802469,0.28395  ,-0.412346,0.293827} 
    };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < tDofsPerCell; tDof++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDof],tHostResultsWS(tCell,tDof),tTolerance);
    }
  }
}

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, CriterionKirchhoffEnergyPotential )
{
 // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  // create output database and spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(1.0, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  // create results workset
  Plato::ScalarVectorT<ResultT> tResultsWS("residual",tNumCells);
  // create criterion
  Plato::CriterionKirchhoffEnergyPotential<Residual> 
    tCriterion(tOnlyDomainDefined,tDataMap,*tGenericParamList,"My Strain Energy");
  tCriterion.evaluate(tStateWS,tControlWS,tConfigWS,tResultsWS);
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<Plato::Scalar> tGold = { 0.2604938271604937, 0.2604938271604937 };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    TEST_FLOATING_EQUALITY(tGold[tCell],tHostResultsWS(tCell),tTolerance);
  }
}

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, CriterionNeoHookeanEnergyPotential )
{
 // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  // create output database and spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(1.0, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  // create results workset
  Plato::ScalarVectorT<ResultT> tResultsWS("residual",tNumCells);
  // create criterion
  Plato::CriterionNeoHookeanEnergyPotential<Residual> 
    tCriterion(tOnlyDomainDefined,tDataMap,*tGenericParamList,"My Strain Energy");
  tCriterion.evaluate(tStateWS,tControlWS,tConfigWS,tResultsWS);
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<Plato::Scalar> tGold = { 0.032487784262637334, 0.032487784262637334 };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    TEST_FLOATING_EQUALITY(tGold[tCell],tHostResultsWS(tCell),tTolerance);
  }
}

}