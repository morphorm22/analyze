/*
 * EllipticNitscheTests.cpp
 *
 *  Created on: July 6, 2023
 */

/// @include trilinos includes
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

/// @include analyze includes
#include "Tri3.hpp"
#include "FadTypes.hpp"
#include "MetaData.hpp"
#include "WorkSets.hpp"
#include "SpatialModel.hpp"

#include "base/WorksetBase.hpp"
#include "utilities/ComputeCharacteristicLength.hpp"

#include "materials/mechanical/MaterialIsotropicElastic.hpp"
#include "materials/mechanical/FactoryElasticMaterial.hpp"

#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/base/WorksetBuilder.hpp"

#include "elliptic/evaluators/nitsche/FactoryNitscheEvaluator.hpp"
#include "elliptic/evaluators/nitsche/NitscheBoundaryCondition.hpp"

#include "elliptic/mechanical/linear/Mechanics.hpp"
#include "elliptic/mechanical/NitscheDispMisfitEvaluator.hpp"
#include "elliptic/mechanical/linear/ComputeStrainTensor.hpp"
#include "elliptic/mechanical/linear/ComputeIsotropicElasticStressTensor.hpp"
#include "elliptic/mechanical/linear/nitsche/NitscheTestElasticStressEvaluator.hpp"
#include "elliptic/mechanical/linear/nitsche/NitscheTrialElasticStressEvaluator.hpp"

#include "elliptic/mechanical/nonlinear/nitsche/NitscheNonLinearMechanics.hpp"
#include "elliptic/mechanical/nonlinear/nitsche/NitscheTestHyperElasticStressEvaluator.hpp"
#include "elliptic/mechanical/nonlinear/nitsche/NitscheTrialHyperElasticStressEvaluator.hpp"

#include "elliptic/thermal/Thermal.hpp"
#include "elliptic/thermal/nitsche/NitscheTempMisfitEvaluator.hpp"
#include "elliptic/thermal/nitsche/NitscheTestHeatFluxEvaluator.hpp"
#include "elliptic/thermal/nitsche/NitscheTrialHeatFluxEvaluator.hpp"

#include "elliptic/thermomechanics/nonlinear/ThermoMechanics.hpp"
#include "elliptic/thermomechanics/nonlinear/nitsche/NitscheNonlinearThermoMechanics.hpp"
#include "elliptic/thermomechanics/nonlinear/nitsche/NitscheTestThermalHyperElasticStressEvaluator.hpp"
#include "elliptic/thermomechanics/nonlinear/nitsche/NitscheTrialThermalHyperElasticStressEvaluator.hpp"

/// @include analyze unit test includes
#include "util/PlatoTestHelpers.hpp"

namespace EllipticNitscheTests
{

TEUCHOS_UNIT_TEST( EllipticNitscheTests, MaterialIsotropicElastic )
{
  Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                       \n"
    "<ParameterList name='Material Models'>                                   \n"
      "<ParameterList name='Unobtainium'>                                     \n"
        "<ParameterList name='Isotropic Linear Elastic'>                      \n"
          "<Parameter  name='Youngs Modulus'  type='double' value='1.0'/>     \n"
          "<Parameter  name='Poissons Ratio'  type='double' value='0.3'/>     \n"
        "</ParameterList>                                                     \n"
      "</ParameterList>                                                       \n"
    "</ParameterList>                                                         \n"
  "</ParameterList>                                                           \n"
  ); 

  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  Plato::FactoryElasticMaterial<Residual> tFactory(tParamList.operator*());

  auto tTol = 1e-6;
  auto tMaterialModel = tFactory.create("Unobtainium");
  auto tYoungsModulus = tMaterialModel->getScalarConstant("youngs Modulus");
  TEST_FLOATING_EQUALITY(1.0,tYoungsModulus,tTol);
  auto tPoissonsRatio = tMaterialModel->getScalarConstant("Poissons ratio");
  TEST_FLOATING_EQUALITY(0.3,tPoissonsRatio,tTol);
  auto tLameMu = tMaterialModel->getScalarConstant("MU");
  TEST_FLOATING_EQUALITY(0.3846153846153846,tLameMu,tTol);
  auto tLameLambda = tMaterialModel->getScalarConstant("lambda");
  TEST_FLOATING_EQUALITY(0.5769230769230769,tLameLambda,tTol);
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, StrainTensor )
{
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create compute strain functor
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  // create configuration workset 
  //
  auto tNumCells = tMesh->NumElements();
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  Plato::ScalarArray3DT<ConfigScalarType> 
    tConfigWS("Config Workset",tNumCells,ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims);
  tWorksetBase.worksetConfig(tConfigWS);
  // create displacement/state workset
  //
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  Plato::ScalarMultiVectorT<StateScalarType> 
    tStateWS("State Workset",tNumCells,ElementType::mNumDofsPerCell);
  tWorksetBase.worksetState(tDisp, tStateWS);
  // create local functors
  //
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::ComputeStrainTensor<Residual> tComputeStrainTensor;
  // create output worksets
  //
  Plato::ScalarArray4DT<ConfigScalarType> 
    tVirtualStrains("Virtual Strains",tNumCells,ElementType::mNumGaussPoints,tSpaceDim,tSpaceDim);
  Plato::ScalarArray4DT<StrainScalarType> 
    tStateStrains("State Strains",tNumCells,ElementType::mNumGaussPoints,tSpaceDim,tSpaceDim);
  // compute test and trial strains
  //
  Kokkos::parallel_for("evaluate", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCells, ElementType::mNumGaussPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal, const Plato::OrdinalType & jGpOrdinal)
  {
    // compute gradients
    //
    ConfigScalarType tVolume(0.);
    Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigScalarType> tGradient;
    tComputeGradient(iCellOrdinal,jGpOrdinal,tConfigWS,tGradient,tVolume);
    // compute virtual strains
    //
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ConfigScalarType> 
      tVirtualStrainTensor(ConfigScalarType(0.0));
    tComputeStrainTensor(iCellOrdinal,tGradient,tVirtualStrainTensor);
    // compute state strains
    //
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> 
      tStrainTensor(StrainScalarType(0.0));
    tComputeStrainTensor(iCellOrdinal,tStateWS,tGradient,tStrainTensor);
    // set output worksets
    //
    for(Plato::OrdinalType tDimI=0; tDimI<ElementType::mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ=0; tDimJ<ElementType::mNumSpatialDims; tDimJ++){
        tStateStrains(iCellOrdinal,jGpOrdinal,tDimI,tDimJ)   = tStrainTensor(tDimI,tDimJ);
        tVirtualStrains(iCellOrdinal,jGpOrdinal,tDimI,tDimJ) = tVirtualStrainTensor(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tStrainGold = 
    { {0.4,0.3,0.3,0.2}, {0.4,0.3,0.3,0.2} };
  auto tHostStateStrains = Kokkos::create_mirror(tStateStrains);
  Kokkos::deep_copy(tHostStateStrains, tStateStrains);
  std::vector<std::vector<Plato::Scalar>> tVirtualStrainGold = 
    { {0.,0.,0.,0.}, {0.,0.,0.,0.} };
  auto tHostVirtualStrains = Kokkos::create_mirror(tVirtualStrains);
  Kokkos::deep_copy(tHostVirtualStrains, tVirtualStrains);
  //
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(
          tStrainGold[tCell][tDimI*tSpaceDim+tDimJ],tHostStateStrains(tCell,0,tDimI,tDimJ),tTol
        );
        TEST_FLOATING_EQUALITY(
          tVirtualStrainGold[tCell][tDimI*tSpaceDim+tDimJ],tHostVirtualStrains(tCell,0,tDimI,tDimJ),tTol
        );
      }
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, StressTensor )
{
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create compute strain functor
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  // create configuration workset 
  //
  auto tNumCells = tMesh->NumElements();
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  Plato::ScalarArray3DT<ConfigScalarType> 
    tConfigWS("Config Workset",tNumCells,ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims);
  tWorksetBase.worksetConfig(tConfigWS);
  // create displacement/state workset
  //
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tDisp("Displacements", tNumDofs);
  Plato::blas1::fill(0.1, tDisp);
  Kokkos::parallel_for("fill displacements",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tDisp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  Plato::ScalarMultiVectorT<StateScalarType> 
    tStateWS("State Workset",tNumCells,ElementType::mNumDofsPerCell);
  tWorksetBase.worksetState(tDisp, tStateWS);
  // create material model
  //
  Plato::MaterialIsotropicElastic<Residual> tMaterialModel;
  tMaterialModel.mu(0.3846153846153846);
  tMaterialModel.lambda(0.5769230769230769);
  // create local functors
  //
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::ComputeStrainTensor<Residual> tComputeStrainTensor;
  Plato::ComputeIsotropicElasticStressTensor<Residual> tComputeStressTensor(tMaterialModel);
  // create output worksets
  //
  Plato::ScalarArray4DT<StrainScalarType> 
    tStressTensors("Stress",tNumCells,ElementType::mNumGaussPoints,tSpaceDim,tSpaceDim);
  // compute test and trial strains
  //
  Kokkos::parallel_for("evaluate", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCells, ElementType::mNumGaussPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal, const Plato::OrdinalType & jGpOrdinal)
  {
    // compute gradients
    //
    ConfigScalarType tVolume(0.);
    Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigScalarType> tGradient;
    tComputeGradient(iCellOrdinal,jGpOrdinal,tConfigWS,tGradient,tVolume);
    // compute state strains
    //
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> 
      tStrainTensor(StrainScalarType(0.0));
    tComputeStrainTensor(iCellOrdinal,tStateWS,tGradient,tStrainTensor);
    // compute stress tensor
    //
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultScalarType> 
      tStressTensor(ResultScalarType(0.0));
    tComputeStressTensor(tStrainTensor,tStressTensor);
    // set output worksets
    //
    for(Plato::OrdinalType tDimI=0; tDimI<ElementType::mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ=0; tDimJ<ElementType::mNumSpatialDims; tDimJ++){
        tStressTensors(iCellOrdinal,jGpOrdinal,tDimI,tDimJ) = tStressTensor(tDimI,tDimJ);
      }
    }
  });
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tStressGold = 
    { {0.65384615,0.23076923,0.23076923,0.5}, {0.65384615,0.23076923,0.23076923,0.5} };
  auto tHostStressTensors = Kokkos::create_mirror(tStressTensors);
  Kokkos::deep_copy(tHostStressTensors, tStressTensors);
  //
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(
          tStressGold[tCell][tDimI*tSpaceDim+tDimJ],tHostStressTensors(tCell,0,tDimI,tDimJ),tTol
        );
      }
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, ComputeSideCellVolumes )
{
  // create inputs
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                                                      \n"
    "<ParameterList name='Spatial Model'>                                                                    \n"
      "<ParameterList name='Domains'>                                                                        \n"
        "<ParameterList name='Design Volume'>                                                                \n"
          "<Parameter name='Element Block' type='string' value='body'/>                                      \n"
          "<Parameter name='Material Model' type='string' value='Unobtainium'/>                              \n"
        "</ParameterList>                                                                                    \n"
      "</ParameterList>                                                                                      \n"
    "</ParameterList>                                                                                        \n"
  "</ParameterList>                                                                                          \n"
  ); 
  // create mesh
  //
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create compute strain functor
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  // create configuration workset 
  //
  auto tNumCells = tMesh->NumElements();
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  auto tConfigWS = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigScalarType> > >
    ( Plato::ScalarArray3DT<ConfigScalarType>(
        "Config Workset",tNumCells,ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims) 
    );
  tWorksetBase.worksetConfig(tConfigWS->mData);
  Plato::WorkSets tWorkSets;
  tWorkSets.set("configuration",tConfigWS);  
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  // compute volumes
  //
  auto tSideSetName = std::string("x+");
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  Plato::ComputeSideCellVolumes<Residual> tComputeSideCellVolumes(tSideSetName);
  Plato::ScalarVectorT<ConfigScalarType> tCellVolumes("volumes",tNumSideCells);
  tComputeSideCellVolumes(tSpatialModel,tWorkSets,tCellVolumes);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-4;
  std::vector<Plato::Scalar> tVolumeGold = { 0.5 };
  auto tHostCellVolumes = Kokkos::create_mirror(tCellVolumes);
  Kokkos::deep_copy(tHostCellVolumes, tCellVolumes);
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    TEST_FLOATING_EQUALITY(tVolumeGold[tCell],tHostCellVolumes(tCell),tTol);
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, ComputeSideCellFaceAreas )
{
  // create inputs
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                                                      \n"
    "<ParameterList name='Spatial Model'>                                                                    \n"
      "<ParameterList name='Domains'>                                                                        \n"
        "<ParameterList name='Design Volume'>                                                                \n"
          "<Parameter name='Element Block' type='string' value='body'/>                                      \n"
          "<Parameter name='Material Model' type='string' value='Unobtainium'/>                              \n"
        "</ParameterList>                                                                                    \n"
      "</ParameterList>                                                                                      \n"
    "</ParameterList>                                                                                        \n"
  "</ParameterList>                                                                                          \n"
  ); 
  // create mesh
  //
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create compute strain functor
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  // create configuration workset 
  //
  auto tNumCells = tMesh->NumElements();
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  auto tConfigWS = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigScalarType> > >
    ( Plato::ScalarArray3DT<ConfigScalarType>(
        "Config Workset",tNumCells,ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims) 
    );
  tWorksetBase.worksetConfig(tConfigWS->mData);
  Plato::WorkSets tWorkSets;
  tWorkSets.set("configuration",tConfigWS);  
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  // compute volumes
  //
  auto tSideSetName = std::string("x+");
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  Plato::ComputeSideCellFaceAreas<Residual> tComputeSideCellFaceAreas(tSideSetName);
  Plato::ScalarVectorT<ConfigScalarType> tCellAreas("areas",tNumSideCells);
  tComputeSideCellFaceAreas(tSpatialModel,tWorkSets,tCellAreas);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-4;
  std::vector<Plato::Scalar> tGoldAreas = { 1.0 };
  auto tHostCellAreas = Kokkos::create_mirror(tCellAreas);
  Kokkos::deep_copy(tHostCellAreas, tCellAreas);
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    TEST_FLOATING_EQUALITY(tGoldAreas[tCell],tHostCellAreas(tCell),tTol);
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, ComputeCharacteristicLength )
{
  // create inputs
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                                                      \n"
    "<ParameterList name='Spatial Model'>                                                                    \n"
      "<ParameterList name='Domains'>                                                                        \n"
        "<ParameterList name='Design Volume'>                                                                \n"
          "<Parameter name='Element Block' type='string' value='body'/>                                      \n"
          "<Parameter name='Material Model' type='string' value='Unobtainium'/>                              \n"
        "</ParameterList>                                                                                    \n"
      "</ParameterList>                                                                                      \n"
    "</ParameterList>                                                                                        \n"
  "</ParameterList>                                                                                          \n"
  ); 
  // create mesh
  //
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create compute strain functor
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  // create configuration workset 
  //
  auto tNumCells = tMesh->NumElements();
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  auto tConfigWS = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigScalarType> > >
    ( Plato::ScalarArray3DT<ConfigScalarType>(
        "Config Workset",tNumCells,ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims) 
    );
  tWorksetBase.worksetConfig(tConfigWS->mData);
  Plato::WorkSets tWorkSets;
  tWorkSets.set("configuration",tConfigWS);  
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  // compute volumes
  //
  auto tSideSetName = std::string("x+");
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  Plato::ComputeCharacteristicLength<Residual> tComputeCharacteristicLength(tSideSetName);
  Plato::ScalarVectorT<ConfigScalarType> tCharLength("areas",tNumSideCells);
  tComputeCharacteristicLength(tSpatialModel,tWorkSets,tCharLength);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-4;
  std::vector<Plato::Scalar> tGoldLength = { 0.5 };
  auto tHostCharLength = Kokkos::create_mirror(tCharLength);
  Kokkos::deep_copy(tHostCharLength, tCharLength);
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    TEST_FLOATING_EQUALITY(tGoldLength[tCell],tHostCharLength(tCell),tTol);
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTrialElasticStressEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                             \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>                   \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                           \n"
    "    <ParameterList name='Domains'>                                               \n"
    "      <ParameterList name='Design Volume'>                                       \n"
    "        <Parameter name='Element Block' type='string' value='body'/>             \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <ParameterList name='Material Models'>                                         \n"
    "    <ParameterList name='Unobtainium'>                                           \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>           \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "</ParameterList>                                                                 \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  // create displacement workset
  //
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
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  tWorksetBuilder.build(tOnlyDomainDefined,tDatabase,tWorkSets);
  // create results workset
  //
  auto tNumCells = tMesh->NumElements();
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("x+");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTrialElasticStressEvaluator<Residual> 
    tNitscheTrialElasticStressEvaluator(*tParamList,tNitscheParams);
  tNitscheTrialElasticStressEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.,0.,-57692.307692307702382,-125000,-57692.307692307702382,-125000} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTestElasticStressEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                             \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>                   \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                           \n"
    "    <ParameterList name='Domains'>                                               \n"
    "      <ParameterList name='Design Volume'>                                       \n"
    "        <Parameter name='Element Block' type='string' value='body'/>             \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <ParameterList name='Material Models'>                                         \n"
    "    <ParameterList name='Unobtainium'>                                           \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>           \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "</ParameterList>                                                                 \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  // create displacement data
  //
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
  // create dirichlet displacement data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumDofs);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("x+");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTestElasticStressEvaluator<Residual> 
    tNitscheTestElasticStressEvaluator(*tParamList,tNitscheParams);
  tNitscheTestElasticStressEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.,0.,0.,0.,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheDispMisfitEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                             \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>                   \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                           \n"
    "    <ParameterList name='Domains'>                                               \n"
    "      <ParameterList name='Design Volume'>                                       \n"
    "        <Parameter name='Element Block' type='string' value='body'/>             \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <ParameterList name='Material Models'>                                         \n"
    "    <ParameterList name='Unobtainium'>                                           \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0'/>           \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "</ParameterList>                                                                 \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  // create displacement data
  //
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
  // create dirichlet displacement data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumDofs);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("x+");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheDispMisfitEvaluator<Residual> tNitscheDispMisfitEvaluator(*tParamList,tNitscheParams);
  tNitscheDispMisfitEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.,0.,0.23333333,0.28333333,0.26666667,0.31666667} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, LinearMechanicalNitscheBC )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                             \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>                   \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                           \n"
    "    <ParameterList name='Domains'>                                               \n"
    "      <ParameterList name='Design Volume'>                                       \n"
    "        <Parameter name='Element Block' type='string' value='body'/>             \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <ParameterList name='Material Models'>                                         \n"
    "    <ParameterList name='Unobtainium'>                                           \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0'/>           \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "</ParameterList>                                                                 \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  // create displacement data
  //
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
  // create dirichlet displacement data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumDofs);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("x+");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheBoundaryCondition<Residual> tNitscheBC(*tParamList,tNitscheParams);
  tNitscheBC.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.,0.,0.175641025641026,0.158333333333333,0.208974358974359,0.191666666666667} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTrialHeatFluxEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics' type='string' value='Thermal'/>                       \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "  <ParameterList name='Material Models'>                                          \n"
    "    <ParameterList name='Unobtainium'>                                            \n"
    "      <ParameterList name='Thermal Conduction'>                                   \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0'/>        \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::ThermalElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 3);
  // create temperature data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector tTemp("Temperature", tNumVerts);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill temperature dofs",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tTemp);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTrialHeatFluxEvaluator<Residual> 
    tNitscheHeatFluxEvaluator(*tParamList,tNitscheParams);
  tNitscheHeatFluxEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.05,0.05,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  } 
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTestHeatFluxEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics' type='string' value='Thermal'/>                       \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "  <ParameterList name='Material Models'>                                          \n"
    "    <ParameterList name='Unobtainium'>                                            \n"
    "      <ParameterList name='Thermal Conduction'>                                   \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0'/>        \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::ThermalElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 3);
  // create temperature data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector tTemp("Temperature", tNumVerts);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill temperature dofs",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tTemp);
  // create dirichlet temperature data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumVerts);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTestHeatFluxEvaluator<Residual> 
    tNitscheVirtualHeatFluxEvaluator(*tParamList,tNitscheParams);
  tNitscheVirtualHeatFluxEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  } 
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheDispMisfitEvaluator2 )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                             \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>                   \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                           \n"
    "    <ParameterList name='Domains'>                                               \n"
    "      <ParameterList name='Design Volume'>                                       \n"
    "        <Parameter name='Element Block' type='string' value='body'/>             \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <ParameterList name='Material Models'>                                         \n"
    "    <ParameterList name='Unobtainium'>                                           \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0'/>           \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "</ParameterList>                                                                 \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  // create displacement data
  //
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
  // create dirichlet displacement data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumDofs);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheDispMisfitEvaluator<Residual> tNitscheDispMisfitEvaluator(*tParamList,tNitscheParams);
  tNitscheDispMisfitEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.066666666666667,0.116666666666667,0.133333333333333,0.183333333333333,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTempMisfitEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics' type='string' value='Thermal'/>                       \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "  <ParameterList name='Material Models'>                                          \n"
    "    <ParameterList name='Unobtainium'>                                            \n"
    "      <ParameterList name='Thermal Conduction'>                                   \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0'/>        \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::ThermalElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 3);
  // create temperature data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector tTemp("Temperature", tNumVerts);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill temperature dofs",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tTemp);
  // create dirichlet temperature data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumVerts);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTempMisfitEvaluator<Residual> 
    tNitscheTempMisfitEvaluator(*tParamList,tNitscheParams);
  tNitscheTempMisfitEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.066666666667,0.133333333333,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  } 
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, LinearThermalNitscheBC )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics' type='string' value='Thermal'/>                       \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "  <ParameterList name='Material Models'>                                          \n"
    "    <ParameterList name='Unobtainium'>                                            \n"
    "      <ParameterList name='Thermal Conduction'>                                   \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0'/>        \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::ThermalElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 3);
  // create temperature data
  //
  Plato::Database tDatabase;
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector tTemp("Temperature", tNumVerts);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill temperature dofs",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("states",tTemp);
  // create dirichlet temperature data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumVerts);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheBoundaryCondition<Residual> tNitscheBC(*tParamList,tNitscheParams);
  tNitscheBC.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.11666666667,0.183333333333,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTrialHyperElasticStressEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics'  type='string' value='Mechanical'/>                   \n"
    "  <Parameter name='Response' type='string' value='Nonlinear'/>                    \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "<ParameterList name='Material Models'>                                            \n"
      "<ParameterList name='Unobtainium'>                                              \n"
        "<ParameterList name='Hyperelastic Kirchhoff'>                                 \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>              \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>               \n"
        "</ParameterList>                                                              \n"
      "</ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 6);
  // create displacement data
  //
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
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTrialHyperElasticStressEvaluator<Residual> 
    tNitscheEvaluator(*tParamList,tNitscheParams);
  tNitscheEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {-0.40123454,-0.14197522, -0.40123454,-0.14197522,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTestHyperElasticStressEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics'  type='string' value='Mechanical'/>                   \n"
    "  <Parameter name='Response' type='string' value='Nonlinear'/>                    \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "<ParameterList name='Material Models'>                                            \n"
      "<ParameterList name='Unobtainium'>                                              \n"
        "<ParameterList name='Hyperelastic Kirchhoff'>                                 \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>              \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>               \n"
        "</ParameterList>                                                              \n"
      "</ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 6);
  // create displacement data
  //
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
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTestHyperElasticStressEvaluator<Residual> 
    tNitscheEvaluator(*tParamList,tNitscheParams);
  tNitscheEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.,0.,0.,0.,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTrialThermalHyperElasticStressEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics'  type='string' value='Thermomechanical'/>             \n"
    "  <Parameter name='Response' type='string' value='Nonlinear'/>                    \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "<ParameterList name='Material Models'>                                            \n"
      "<ParameterList name='Unobtainium'>                                              \n"
        "<ParameterList name='Thermal Conduction'>                                     \n"
          "<Parameter  name='Thermal Expansivity'   type='double' value='0.5'/>        \n"
          "<Parameter  name='Reference Temperature' type='double' value='1.0'/>        \n"
          "<Parameter  name='Thermal Conductivity'  type='double' value='2.0'/>        \n"
        "</ParameterList>                                                              \n"
        "<ParameterList name='Hyperelastic Kirchhoff'>                                 \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>              \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>               \n"
        "</ParameterList>                                                              \n"
      "</ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType         = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual            = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType     = typename Residual::StateScalarType;
  using ResultScalarType    = typename Residual::ResultScalarType;
  using ConfigScalarType    = typename Residual::ConfigScalarType;
  using NodeStateScalarType = typename Residual::NodeStateScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 6);
  // create displacement data
  //
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
  // create temperature data
  //
  Plato::ScalarVector tTemp("Temperature", tNumVerts);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill temperature dofs",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("node states",tTemp);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTrialThermalHyperElasticStressEvaluator<Residual> 
    tNitscheEvaluator(*tParamList,tNitscheParams);
  tNitscheEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {-0.213991754666667,-0.075720117333333,-0.227366239333333,-0.080452624666667,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheTestThermalHyperElasticStressEvaluator )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics'  type='string' value='Thermomechanical'/>             \n"
    "  <Parameter name='Response' type='string' value='Nonlinear'/>                    \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "<ParameterList name='Material Models'>                                            \n"
      "<ParameterList name='Unobtainium'>                                              \n"
        "<ParameterList name='Thermal Conduction'>                                     \n"
          "<Parameter  name='Thermal Expansivity'   type='double' value='0.5'/>        \n"
          "<Parameter  name='Reference Temperature' type='double' value='1.0'/>        \n"
          "<Parameter  name='Thermal Conductivity'  type='double' value='2.0'/>        \n"
        "</ParameterList>                                                              \n"
        "<ParameterList name='Hyperelastic Kirchhoff'>                                 \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>              \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>               \n"
        "</ParameterList>                                                              \n"
      "</ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType         = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual            = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType     = typename Residual::StateScalarType;
  using ResultScalarType    = typename Residual::ResultScalarType;
  using ConfigScalarType    = typename Residual::ConfigScalarType;
  using NodeStateScalarType = typename Residual::NodeStateScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 6);
  // create displacement data
  //
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
  // create temperature data
  //
  Plato::ScalarVector tTemp("Temperature", tNumVerts);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill temperature dofs",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("node states",tTemp);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheTestThermalHyperElasticStressEvaluator<Residual> 
    tNitscheEvaluator(*tParamList,tNitscheParams);
  tNitscheEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {-0.2765432,-0.0703703,-0.2765432,-0.0703703,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheNonLinearMechanics )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics'  type='string' value='Mechanical'/>                   \n"
    "  <Parameter name='Response' type='string' value='Nonlinear'/>                    \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "<ParameterList name='Material Models'>                                            \n"
      "<ParameterList name='Unobtainium'>                                              \n"
        "<ParameterList name='Hyperelastic Kirchhoff'>                                 \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>              \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>               \n"
        "</ParameterList>                                                              \n"
      "</ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 6);
  // create displacement data
  //
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
  // create dirichlet displacement data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumDofs);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheNonLinearMechanics<Residual> 
    tNitscheEvaluator(*tParamList,tNitscheParams);
  tNitscheEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {-0.334567873333333,-0.02530855333333301,-0.26790120666666695,0.041358113333332974,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, NitscheNonlinearThermoMechanics )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics'  type='string' value='Thermomechanical'/>             \n"
    "  <Parameter name='Response' type='string' value='Nonlinear'/>                    \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "<ParameterList name='Material Models'>                                            \n"
      "<ParameterList name='Unobtainium'>                                              \n"
        "<ParameterList name='Hyperelastic Kirchhoff'>                                 \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>              \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>               \n"
        "</ParameterList>                                                              \n"
        "<ParameterList name='Thermal Conduction'>                                     \n"
          "<Parameter  name='Thermal Expansivity'   type='double' value='0.5'/>        \n"
          "<Parameter  name='Reference Temperature' type='double' value='1.0'/>        \n"
          "<Parameter  name='Thermal Conductivity'  type='double' value='2.0'/>        \n"
        "</ParameterList>                                                              \n"
      "</ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 6);
  // create displacement data
  //
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
  // create temperature data
  //
  Plato::ScalarVector tTemp("Temperature", tNumVerts);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill temperature dofs",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("node states",tTemp);
  // create dirichlet displacement data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumDofs);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator and evaluate nitsche's stress term
  //
  Plato::Elliptic::NitscheNonlinearThermoMechanics<Residual> 
    tNitscheEvaluator(*tParamList,tNitscheParams);
  tNitscheEvaluator.evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {-0.423868288,-0.029423750666666,-0.370576106,0.03251040866666599,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

TEUCHOS_UNIT_TEST( EllipticNitscheTests, FactoryNitscheEvaluator_NonlinearThermoMechanics )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
  Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                              \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>               \n"
    "  <Parameter name='Physics'  type='string' value='Thermomechanical'/>             \n"
    "  <Parameter name='Response' type='string' value='Nonlinear'/>                    \n"
    "  <Parameter name='Weak Essential Boundary Conditions' type='bool' value='true'/> \n"
    "  <ParameterList name='Spatial Model'>                                            \n"
    "    <ParameterList name='Domains'>                                                \n"
    "      <ParameterList name='Design Volume'>                                        \n"
    "        <Parameter name='Element Block' type='string' value='body'/>              \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>      \n"
    "      </ParameterList>                                                            \n"
    "    </ParameterList>                                                              \n"
    "  </ParameterList>                                                                \n"
    "<ParameterList name='Material Models'>                                            \n"
      "<ParameterList name='Unobtainium'>                                              \n"
        "<ParameterList name='Hyperelastic Kirchhoff'>                                 \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>              \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>               \n"
        "</ParameterList>                                                              \n"
        "<ParameterList name='Thermal Conduction'>                                     \n"
          "<Parameter  name='Thermal Expansivity'   type='double' value='0.5'/>        \n"
          "<Parameter  name='Reference Temperature' type='double' value='1.0'/>        \n"
          "<Parameter  name='Thermal Conductivity'  type='double' value='2.0'/>        \n"
        "</ParameterList>                                                              \n"
      "</ParameterList>                                                                \n"
    "</ParameterList>                                                                  \n"
    "</ParameterList>                                                                  \n"
    );
  // create mesh
  //
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  // create output database and spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh,*tParamList,tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  // create evaluation and scalar types
  //
  using ElementType = typename Plato::ThermoElasticElement<Plato::Tri3>;  
  using Residual    = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateScalarType  = typename Residual::StateScalarType;
  using ResultScalarType = typename Residual::ResultScalarType;
  using ConfigScalarType = typename Residual::ConfigScalarType;
  TEST_ASSERT(ElementType::mNumDofsPerCell == 6);
  // create displacement data
  //
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
  // create temperature data
  //
  Plato::ScalarVector tTemp("Temperature", tNumVerts);
  Plato::blas1::fill(0.1, tTemp);
  Kokkos::parallel_for("fill temperature dofs",
    Kokkos::RangePolicy<>(0, tNumVerts), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
  { tTemp(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  tDatabase.vector("node states",tTemp);
  // create dirichlet displacement data
  //
  Plato::ScalarVector tDirichlet("Dirichlet", tNumDofs);
  Plato::blas1::fill(0.,tDirichlet);
  tDatabase.vector("dirichlet",tDirichlet);
  // create workset database
  //
  Plato::WorkSets tWorkSets;
  Plato::WorksetBase<ElementType> tWorksetFuncs(tMesh);
  Plato::Elliptic::WorksetBuilder<Residual> tWorksetBuilder(tWorksetFuncs);
  auto tNumCells = tMesh->NumElements();
  tWorksetBuilder.build(tNumCells,tDatabase,tWorkSets);
  // create results workset
  //
  auto tResultWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ResultScalarType> > >
    ( Plato::ScalarMultiVectorT<ResultScalarType>("Result Workset", tNumCells, ElementType::mNumDofsPerCell) );
  Kokkos::deep_copy(tResultWS->mData,0.);
  tWorkSets.set("result",tResultWS);
  // create inputs for nitsche's method
  //
  auto tSideSetName = std::string("y-");
  Teuchos::ParameterList tNitscheParams;
  tNitscheParams.set("Sides",tSideSetName);
  tNitscheParams.set("Material Model",tOnlyDomainDefined.getMaterialName());
  // create evaluator
  //
  Plato::Elliptic::FactoryNitscheEvaluator<Residual> tFactory;
  auto tEvaluator = tFactory.create(*tParamList,tNitscheParams);
  tEvaluator->evaluate(tSpatialModel,tWorkSets);
  // test gold values
  //
  constexpr Plato::Scalar tTol = 1e-8;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {-0.423868288,-0.029423750666666,-0.370576106,0.03251040866666599,0.,0.} };
  auto tHostResultWS = Kokkos::create_mirror(tResultWS->mData);
  Kokkos::deep_copy(tHostResultWS,tResultWS->mData);
  //
  auto tSideCellOrdinals = tMesh->GetSideSetElements(tSideSetName);
  Plato::OrdinalType tNumSideCells = tSideCellOrdinals.size();
  for(Plato::OrdinalType tCell = 0; tCell < tNumSideCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < ElementType::mNumDofsPerCell; tDof++){
      auto tDiff = std::abs(tGold[tCell][tDof] - tHostResultWS(tCell,tDof));
      TEST_ASSERT(tDiff < tTol);
    }
  }
}

} // namespace EllipticNitscheTests

