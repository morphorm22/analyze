/*
 * ElectrostaticsTests.cpp
 *
 *  Created on: May 10, 2023
 */

// trilinos includes
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

// unit test includes
#include "util/PlatoTestHelpers.hpp"

// plato
#include "Tri3.hpp"
#include "elliptic/Problem.hpp"
#include "elliptic/electrical/Electrical.hpp"
#include "elliptic/electrical/FactoryElectricalMaterial.hpp"
#include "elliptic/electrical/DarkCurrentDensityQuadratic.hpp"
#include "elliptic/electrical/LightGeneratedCurrentDensityConstant.hpp"
#include "elliptic/electrical/DarkCurrentDensityTwoPhaseAlloy.hpp"
#include "elliptic/electrical/LightCurrentDensityTwoPhaseAlloy.hpp"
#include "elliptic/electrical/FactoryCurrentDensityEvaluator.hpp"
#include "elliptic/electrical/SourceWeightedSum.hpp"

namespace ElectrostaticsTest
{

   Teuchos::RCP<Teuchos::ParameterList> tGenericParamList = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                                                            \n"
    "<ParameterList name='Spatial Model'>                                                                          \n"
      "<ParameterList name='Domains'>                                                                              \n"
        "<ParameterList name='Design Volume'>                                                                      \n"
          "<Parameter name='Element Block' type='string' value='body'/>                                            \n"
          "<Parameter name='Material Model' type='string' value='Mystic'/>                                         \n"
        "</ParameterList>                                                                                          \n"
      "</ParameterList>                                                                                            \n"
    "</ParameterList>                                                                                              \n"
    "<ParameterList name='Material Models'>                                                                        \n"
      "<ParameterList name='Mystic'>                                                                               \n"
        "<ParameterList name='Two Phase Conductive'>                                                  \n"
          "<Parameter  name='Material Name'            type='Array(string)' value='{silver,aluminum}'/>            \n"
          "<Parameter  name='Electrical Conductivity'  type='Array(double)' value='{0.15,0.25}'/>                  \n"
          "<Parameter  name='Out-of-Plane Thickness'   type='Array(double)' value='{0.12,0.22}'/>                  \n"
        "</ParameterList>                                                                                          \n"
      "</ParameterList>                                                                                            \n"
    "</ParameterList>                                                                                              \n"
    "<ParameterList name='Criteria'>                                                                               \n"
    "  <ParameterList name='Objective'>                                                                            \n"
    "    <Parameter name='Type' type='string' value='Weighted Sum'/>                                               \n"
    "    <Parameter name='Functions' type='Array(string)' value='{My Dark Power,My Light}'/>                       \n"
    "    <Parameter name='Weights' type='Array(double)' value='{1.0,1.0}'/>                                        \n"
    "  </ParameterList>                                                                                            \n"
    "  <ParameterList name='My Dark Power'>                                                                        \n"
    "    <Parameter name='Type'                   type='string'   value='Scalar Function'/>                        \n"
    "    <Parameter name='Scalar Function Type'   type='string'   value='Power Surface Density'/>                  \n"
    "    <Parameter name='Function'               type='string'   value='My Dark CD'/>                             \n"
    "  </ParameterList>                                                                                            \n"
    "  <ParameterList name='My Light Power'>                                                                       \n"
    "    <Parameter name='Type'                   type='string'   value='Scalar Function'/>                        \n"
    "    <Parameter name='Scalar Function Type'   type='string'   value='Power Surface Density'/>                  \n"
    "    <Parameter name='Function'               type='string'   value='My Light-Generated CD'/>                  \n"
    "  </ParameterList>                                                                                            \n"
    "  <ParameterList name='My Volume'>                                                                            \n"
    "    <Parameter name='Type'                   type='string'   value='Scalar Function'/>                        \n"
    "    <Parameter name='Scalar Function Type'   type='string'   value='Two Phase Volume'/>                       \n"
    "  </ParameterList>                                                                                            \n"
    "</ParameterList>                                                                                              \n"
    "<ParameterList name='Source Terms'>                                                                           \n"
    "  <ParameterList name='Source'>                                                                               \n"
    "    <Parameter name='Type'      type='string'        value='Weighted Sum'/>                                   \n"
    "    <Parameter name='Functions' type='Array(string)' value='{My Dark CD ,My Light-Generated CD}'/>            \n"
    "    <Parameter name='Weights'   type='Array(double)' value='{1.0,1.0}'/>                                      \n"
    "  </ParameterList>                                                                                            \n"
    "  <ParameterList name='My Dark CD'>                                                                           \n"
    "    <Parameter  name='Function'        type='string'      value='Two Phase Dark Current Density'/>            \n"
    "    <Parameter  name='Model'           type='string'      value='Quadratic'/>           ,                     \n"
    "  </ParameterList>                                                                                            \n"
    "  <ParameterList name='My Light-Generated CD'>                                                                \n"
    "    <Parameter  name='Function'        type='string'      value='Two Phase Light-Generated Current Density'/> \n"
    "    <Parameter  name='Model'           type='string'      value='Constant'/>                                  \n"
      "</ParameterList>                                                                                            \n"
    "</ParameterList>                                                                                              \n"
    "<ParameterList name='Output'>                                                                                 \n"
    "  <Parameter name='Plottable' type='Array(string)' value='{electrical field,current density}'/>              \n"
    "</ParameterList>                                                                                              \n"
  "</ParameterList>                                                                                                \n"
  );

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CurrentDensityEvaluator_TwoPhaseAlloy)
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::ElectricalElement<Plato::Tri3>;
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(0.5, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  // create state workset
  Plato::ScalarVector tState("States", tNumVerts);
  Plato::blas1::fill(0.67186, tState);
  auto tHostState = Kokkos::create_mirror_view(tState);
  for(Plato::OrdinalType i = 0; i < tNumVerts; i++)
  { tHostState(i) = tHostState(i) + (i*1e-2); }
  Kokkos::deep_copy(tState, tHostState);
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // create result worset
  Plato::ScalarArray3DT<ResultT> 
    tResultWS("results",tNumCells,ElementType::mNumGaussPoints,ElementType::mNumSpatialDims);
  // create spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  // create current density
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  Plato::FactoryCurrentDensityEvaluator<Residual> tFactory("Mystic",*tGenericParamList);
  auto tCurrentDensity = tFactory.create(tOnlyDomainDefined,tDataMap);
  tCurrentDensity->evaluate(tStateWS,tControlWS,tConfigWS,tResultWS);
  // test results against gold 
  Plato::Scalar tTol = 1e-4;
  auto tHost = Kokkos::create_mirror_view(tResultWS);
  std::vector<std::vector<Plato::Scalar>>tGold = {{0.00475,0.002375},
                                                  {0.00475,0.002375}};
  Kokkos::deep_copy(tHost, tResultWS);
  for(Plato::OrdinalType i = 0; i < tNumCells; i++){
    for(Plato::OrdinalType j = 0; j < ElementType::mNumSpatialDims; j++){
      TEST_FLOATING_EQUALITY(tGold[i][j],tHost(i,0 /* gauss point */,j),tTol);
    }
  }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CurrentDensityEvaluator_Constant)
{
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
          "<ParameterList name='Conductive'>                                                  \n"
            "<Parameter  name='Electrical Conductivity' type='double' value='0.35'/>                       \n"
          "</ParameterList>                                                                                \n"
        "</ParameterList>                                                                                  \n"
      "</ParameterList>                                                                                    \n"
     "</ParameterList>                                                                                     \n"
    );
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::ElectricalElement<Plato::Tri3>;
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(0.5, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  // create state workset
  Plato::ScalarVector tState("States", tNumVerts);
  Plato::blas1::fill(0.67186, tState);
  auto tHostState = Kokkos::create_mirror_view(tState);
  for(Plato::OrdinalType i = 0; i < tNumVerts; i++)
  { tHostState(i) = tHostState(i) + (i*1e-2); }
  Kokkos::deep_copy(tState, tHostState);
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // create result worset
  Plato::ScalarArray3DT<ResultT> 
    tResultWS("results",tNumCells,ElementType::mNumGaussPoints,ElementType::mNumSpatialDims);
  // create spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);
  // create current density
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  Plato::FactoryCurrentDensityEvaluator<Residual> tFactory("Mystic",*tParamList);
  auto tCurrentDensity = tFactory.create(tOnlyDomainDefined,tDataMap);
  tCurrentDensity->evaluate(tStateWS,tControlWS,tConfigWS,tResultWS);
  // test results against gold 
  Plato::Scalar tTol = 1e-4;
  auto tHost = Kokkos::create_mirror_view(tResultWS);
  std::vector<std::vector<Plato::Scalar>>tGold = {{0.007,0.0035},
                                                  {0.007,0.0035}};
  Kokkos::deep_copy(tHost, tResultWS);
  for(Plato::OrdinalType i = 0; i < tNumCells; i++){
    for(Plato::OrdinalType j = 0; j < ElementType::mNumSpatialDims; j++){
      TEST_FLOATING_EQUALITY(tGold[i][j],tHost(i,0 /* gauss point */,j),tTol);
    }
  }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MaterialElectricalConductivity_Error)
{
    // TEST ONE: ERROR
    Teuchos::RCP<Teuchos::ParameterList> tParamListError = Teuchos::getParametersFromXmlString(
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
            "<ParameterList name='Isotropic Linear Elastic'>                                                 \n"
              "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>                                \n"
              "<Parameter  name='Youngs Modulus' type='double' value='4.0'/>                                 \n"
              "<Parameter  name='Mass Density'   type='double' value='0.5'/>                                 \n"
            "</ParameterList>                                                                                \n"
          "</ParameterList>                                                                                  \n"
        "</ParameterList>                                                                                    \n"
       "</ParameterList>                                                                                     \n"
      );

    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElectricalElement<Plato::Tri3>>::Residual;
    Plato::FactoryElectricalMaterial<Residual> tFactoryMaterial(tParamListError.operator*());
    TEST_THROW(tFactoryMaterial.create("Mystic"),std::runtime_error);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MaterialElectricalConductivity)
{
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
            "<ParameterList name='Conductive'>                                                  \n"
              "<Parameter  name='Electrical Conductivity' type='double' value='0.35'/>                       \n"
            "</ParameterList>                                                                                \n"
          "</ParameterList>                                                                                  \n"
        "</ParameterList>                                                                                    \n"
       "</ParameterList>                                                                                     \n"
      );

    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElectricalElement<Plato::Tri3>>::Residual;
    Plato::FactoryElectricalMaterial<Residual> tFactoryMaterial(tParamList.operator*());
    auto tMaterial = tFactoryMaterial.create("Mystic");
    auto tElectricalConductivity = tMaterial->property("electrical conductivity");
    auto tScalarValue = std::stod(tElectricalConductivity.back());
    TEST_FLOATING_EQUALITY(0.35,tScalarValue,1e-6);
    TEST_THROW(tMaterial->property("electrical_conductivity"),std::runtime_error);

    std::vector<std::vector<Plato::Scalar>> tGold = {{0.35,0.},{0.,0.35}};
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::TensorConstant<tNumSpaceDims> tTensor = tMaterial->getTensorConstant("material tensor");
    for(Plato::OrdinalType tDimI = 0; tDimI < tNumSpaceDims; tDimI++){
        for(decltype(tDimI) tDimJ = 0; tDimJ < tNumSpaceDims; tDimJ++){
            TEST_FLOATING_EQUALITY(tGold[tDimI][tDimJ],tTensor(tDimI,tDimJ),1e-6);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MaterialDielectric)
{
    Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                  \n"
        "<ParameterList name='Spatial Model'>                                                \n"
          "<ParameterList name='Domains'>                                                    \n"
            "<ParameterList name='Design Volume'>                                            \n"
              "<Parameter name='Element Block' type='string' value='body'/>                  \n"
              "<Parameter name='Material Model' type='string' value='Mystic'/>               \n"
            "</ParameterList>                                                                \n"
          "</ParameterList>                                                                  \n"
        "</ParameterList>                                                                    \n"
        "<ParameterList name='Material Models'>                                              \n"
          "<ParameterList name='Mystic'>                                                     \n"
            "<ParameterList name='Dielectric'>                                               \n"
              "<Parameter  name='Electrical Constant'          type='double' value='0.15'/>  \n"
              "<Parameter  name='Relative Static Permittivity' type='double' value='0.35'/>  \n"
            "</ParameterList>                                                                \n"
          "</ParameterList>                                                                  \n"
        "</ParameterList>                                                                    \n"
       "</ParameterList>                                                                     \n"
      );

    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElectricalElement<Plato::Tri3>>::Residual;
    Plato::FactoryElectricalMaterial<Residual> tFactoryMaterial(tParamList.operator*());
    auto tMaterial = tFactoryMaterial.create("Mystic");
    auto tElectricalConstant = tMaterial->property("electrical constant");
    auto tScalarValue = std::stod(tElectricalConstant.back());
    TEST_FLOATING_EQUALITY(0.15,tScalarValue,1e-6);
    auto tRelativeStaticPermittivity = tMaterial->property("Relative Static Permittivity");
    tScalarValue = std::stod(tRelativeStaticPermittivity.back());
    TEST_FLOATING_EQUALITY(0.35,tScalarValue,1e-6);

    std::vector<std::vector<Plato::Scalar>> tGold = {{0.0525,0.},{0.,0.0525}};
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::TensorConstant<tNumSpaceDims> tTensor = tMaterial->getTensorConstant("material tensor");
    for(Plato::OrdinalType tDimI = 0; tDimI < tNumSpaceDims; tDimI++){
        for(decltype(tDimI) tDimJ = 0; tDimJ < tNumSpaceDims; tDimJ++){
            TEST_FLOATING_EQUALITY(tGold[tDimI][tDimJ],tTensor(tDimI,tDimJ),1e-6);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MaterialElectricalConductivityTwoPhaseAlloy)
{
    Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                            \n"
        "<ParameterList name='Spatial Model'>                                                          \n"
          "<ParameterList name='Domains'>                                                              \n"
            "<ParameterList name='Design Volume'>                                                      \n"
              "<Parameter name='Element Block' type='string' value='body'/>                            \n"
              "<Parameter name='Material Model' type='string' value='Mystic'/>                         \n"
            "</ParameterList>                                                                          \n"
          "</ParameterList>                                                                            \n"
        "</ParameterList>                                                                              \n"
        "<ParameterList name='Material Models'>                                                        \n"
          "<ParameterList name='Mystic'>                                                               \n"
            "<ParameterList name='Two Phase Conductive'>                                  \n"
              "<Parameter  name='Electrical Conductivity'  type='Array(double)' value='{0.15, 0.25}'/> \n"
              "<Parameter  name='Out-of-Plane Thickness'   type='Array(double)' value='{0.12, 0.22}'/> \n"
            "</ParameterList>                                                                          \n"
          "</ParameterList>                                                                            \n"
        "</ParameterList>                                                                              \n"
       "</ParameterList>                                                                               \n"
      );

    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElectricalElement<Plato::Tri3>>::Residual;
    Plato::FactoryElectricalMaterial<Residual> tFactoryMaterial(tParamList.operator*());
    auto tMaterial = tFactoryMaterial.create("Mystic");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, DarkCurrentDensityQuadratic)
{
    Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                       \n"
        "<ParameterList name='Spatial Model'>                                                     \n"
          "<ParameterList name='Domains'>                                                         \n"
            "<ParameterList name='Design Volume'>                                                 \n"
              "<Parameter name='Element Block' type='string' value='body'/>                       \n"
              "<Parameter name='Material Model' type='string' value='Mystic'/>                    \n"
            "</ParameterList>                                                                     \n"
          "</ParameterList>                                                                       \n"
        "</ParameterList>                                                                         \n"
        "<ParameterList name='Source Terms'>                                                      \n"
          "<ParameterList name='Dark Current Density'>                                            \n"
            "<Parameter  name='Model'              type='string'   value='Custom Quadratic Fit'/> \n"
            "<Parameter  name='Performance Limit'  type='double'   value='-0.22'/>                \n"
            "<Parameter  name='a'                  type='double'   value='0.0'/>                  \n"
            "<Parameter  name='b'                  type='double'   value='1.27E-06'/>             \n"
            "<Parameter  name='c'                  type='double'   value='25.94253'/>             \n"
            "<Parameter  name='m1'                 type='double'   value='0.38886'/>              \n"
            "<Parameter  name='b1'                 type='double'   value='0.0'/>                  \n"
            "<Parameter  name='m2'                 type='double'   value='30.0'/>                 \n"
            "<Parameter  name='b2'                 type='double'   value='6.520373'/>             \n"
          "</ParameterList>                                                                       \n"
        "</ParameterList>                                                                         \n"
       "</ParameterList>                                                                          \n"
      );

    // TEST ONE: V > 0
    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElectricalElement<Plato::Tri3>>::Residual;
    Plato::DarkCurrentDensityQuadratic<Residual,Plato::Scalar> 
      tCurrentDensityModel("Dark Current Density",tParamList.operator*());
    Residual::StateScalarType tElectricPotential = 0.67186;
    Plato::Scalar tDarkCurrentDensity = tCurrentDensityModel.evaluate(tElectricPotential);
    Plato::Scalar tTol = 1e-4;
    TEST_FLOATING_EQUALITY(47.1463,tDarkCurrentDensity,tTol);

    // TEST 2: V = 0
    tElectricPotential = 0.;
    tDarkCurrentDensity = tCurrentDensityModel.evaluate(tElectricPotential);
    TEST_FLOATING_EQUALITY(0.,tDarkCurrentDensity,tTol);
    
    // TEST 3: -0.22 < V < 0 
    tElectricPotential = -0.06189;
    tDarkCurrentDensity = tCurrentDensityModel.evaluate(tElectricPotential);
    TEST_FLOATING_EQUALITY(-0.0240665,tDarkCurrentDensity,tTol);

    // TEST 4: V < -0.22 
    tElectricPotential = -0.25;
    tDarkCurrentDensity = tCurrentDensityModel.evaluate(tElectricPotential);
    TEST_FLOATING_EQUALITY(-0.979627,tDarkCurrentDensity,tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, LightGeneratedCurrentDensityConstant)
{
    Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                       \n"
        "<ParameterList name='Spatial Model'>                                                     \n"
          "<ParameterList name='Domains'>                                                         \n"
            "<ParameterList name='Design Volume'>                                                 \n"
              "<Parameter name='Element Block' type='string' value='body'/>                       \n"
              "<Parameter name='Material Model' type='string' value='Mystic'/>                    \n"
            "</ParameterList>                                                                     \n"
          "</ParameterList>                                                                       \n"
        "</ParameterList>                                                                         \n"
        "<ParameterList name='Source Terms'>                                                      \n"
          "<ParameterList name='Light-Generated Current Density'>                                 \n"
            "<Parameter  name='Model'              type='string'   value='Constant'/>             \n"
            "<Parameter  name='Generation Rate'    type='double'   value='0.5'/>                  \n"
            "<Parameter  name='Illumination Power' type='double'   value='10.0'/>                 \n"
          "</ParameterList>                                                                       \n"
        "</ParameterList>                                                                         \n"
       "</ParameterList>                                                                          \n"
      );

    using Residual = typename Plato::Elliptic::Evaluation<Plato::ElectricalElement<Plato::Tri3>>::Residual;
    Plato::LightGeneratedCurrentDensityConstant<Residual,Plato::Scalar> 
      tCurrentDensityModel("Light-Generated Current Density",tParamList.operator*());
    Residual::StateScalarType tElectricPotential = 0.67186;
    Plato::Scalar tDarkCurrentDensity = tCurrentDensityModel.evaluate(tElectricPotential);
    Plato::Scalar tTol = 1e-4;
    TEST_FLOATING_EQUALITY(5.,tDarkCurrentDensity,tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, LightCurrentDensityTwoPhaseAlloy)
{
    // create mesh
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
    using ElementType = typename Plato::ElectricalElement<Plato::Tri3>;

    //set ad-types
    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using StateT   = typename Residual::StateScalarType;
    using ConfigT  = typename Residual::ConfigScalarType;
    using ResultT  = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    // create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);
    
    // create control workset
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(0.5, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);
    
    // create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::blas1::fill(0.1, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
            {   tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal);}, "fill state");
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);
    
    // create spatial model
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);

    // create current density
    auto tOnlyDomainDefined = tSpatialModel.Domains.front();
    TEST_ASSERT(tGenericParamList->isSublist("Source Terms") == true);
    Plato::LightCurrentDensityTwoPhaseAlloy<Residual> 
      tCurrentDensity("Mystic","My Light-Generated CD",tGenericParamList.operator*());
    
    // create result/output workset
    Plato::ScalarMultiVectorT<Plato::Scalar> tResultWS("result workset", tNumCells, tDofsPerCell);
    tCurrentDensity.evaluate(tOnlyDomainDefined,tStateWS,tControlWS,tConfigWS,tResultWS,1.0);

    // test against gold
    auto tHost = Kokkos::create_mirror_view(tResultWS);
    Plato::Scalar tTol = 1e-6;
    std::vector<std::vector<Plato::Scalar>>tGold = {{-41.078313,-41.078313,-41.078313},
                                                    {-41.078313,-41.078313,-41.078313}};
    Kokkos::deep_copy(tHost, tResultWS);
    for(Plato::OrdinalType i = 0; i < tNumCells; i++){
      for(Plato::OrdinalType j = 0; j < tDofsPerCell; j++){
        TEST_FLOATING_EQUALITY(tGold[i][j],tHost(i,j),tTol);
      }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, DarkCurrentDensityTwoPhaseAlloy)
{
    // create mesh
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
    using ElementType = typename Plato::ElectricalElement<Plato::Tri3>;

    //set ad-types
    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using StateT   = typename Residual::StateScalarType;
    using ConfigT  = typename Residual::ConfigScalarType;
    using ResultT  = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    // create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);
    
    // create control workset
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(0.5, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);
    
    // create state workset
    Plato::ScalarVector tState("States", tNumVerts);
    Plato::blas1::fill(0.67186, tState);
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);
    
    // create spatial model
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);

    // create current density
    auto tOnlyDomainDefined = tSpatialModel.Domains.front();
    TEST_ASSERT(tGenericParamList->isSublist("Source Terms") == true);
    Plato::DarkCurrentDensityTwoPhaseAlloy<Residual> 
      tCurrentDensity("Mystic","My Dark CD",tGenericParamList.operator*());
    
    // create result/output workset
    Plato::ScalarMultiVectorT<Plato::Scalar> tResultWS("result workset", tNumCells, tDofsPerCell);
    tCurrentDensity.evaluate(tOnlyDomainDefined,tStateWS,tControlWS,tConfigWS,tResultWS,1.0);

    // test against gold
    auto tHost = Kokkos::create_mirror_view(tResultWS);
    Plato::Scalar tTol = 1e-6;
    std::vector<std::vector<Plato::Scalar>>tGold = {{37.8684771,37.8684771,37.8684771},
                                                    {37.8684771,37.8684771,37.8684771}};
    Kokkos::deep_copy(tHost, tResultWS);
    for(Plato::OrdinalType i = 0; i < tNumCells; i++){
      for(Plato::OrdinalType j = 0; j < tDofsPerCell; j++){
        TEST_FLOATING_EQUALITY(tGold[i][j],tHost(i,j),tTol);
      }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SingleDiode)
{
    // create mesh
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
    using ElementType = typename Plato::ElectricalElement<Plato::Tri3>;

    //set ad-types
    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using StateT   = typename Residual::StateScalarType;
    using ConfigT  = typename Residual::ConfigScalarType;
    using ResultT  = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    // create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);
    
    // create control workset
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(0.5, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);
    
    // create state workset
    Plato::ScalarVector tState("States", tNumVerts);
    Plato::blas1::fill(0.67186, tState);
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);
    
    // create spatial model
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);

    // create current density
    auto tOnlyDomainDefined = tSpatialModel.Domains.front();
    TEST_ASSERT(tGenericParamList->isSublist("Source Terms") == true);
    Plato::SourceWeightedSum<Residual> tSingleDiode("Mystic",tGenericParamList.operator*());
    Plato::ScalarMultiVectorT<Plato::Scalar> tResultWS("result workset", tNumCells, tDofsPerCell);
    tSingleDiode.evaluate(tOnlyDomainDefined,tStateWS,tControlWS,tConfigWS,tResultWS,1.0/*scale*/);

    // test against gold
    auto tHost = Kokkos::create_mirror_view(tResultWS);
    Plato::Scalar tTol = 1e-6;
    std::vector<std::vector<Plato::Scalar>>tGold = {{-3.2098359,-3.2098359,-3.2098359},
                                                    {-3.2098359,-3.2098359,-3.2098359}};
    Kokkos::deep_copy(tHost, tResultWS);
    for(Plato::OrdinalType i = 0; i < tNumCells; i++){
      for(Plato::OrdinalType j = 0; j < tDofsPerCell; j++){
        TEST_FLOATING_EQUALITY(tGold[i][j],tHost(i,j),tTol);
      }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CriterionPowerSurfaceDensityTwoPhase)
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::ElectricalElement<Plato::Tri3>;
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(0.5, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  
  // create state workset
  Plato::ScalarVector tState("States", tNumVerts);
  Plato::blas1::fill(0.67186, tState);
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  
  // create spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  // create current density
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  Plato::CriterionPowerSurfaceDensityTwoPhase<Residual>
    tCriterionPowerSurfaceDensityTwoPhase(tOnlyDomainDefined,tDataMap,*tGenericParamList,"My Dark Power");
  Plato::ScalarVectorT<ResultT> tResultWS("result workset", tNumCells);
  tCriterionPowerSurfaceDensityTwoPhase.evaluate(tStateWS,tControlWS,tConfigWS,tResultWS);
  // test against gold
  auto tHost = Kokkos::create_mirror_view(tResultWS);
  Plato::Scalar tTol = 1e-6;
  std::vector<Plato::Scalar>tGold = {15.8378409629,15.8378409629};
  Kokkos::deep_copy(tHost, tResultWS);
  for(Plato::OrdinalType i = 0; i < tNumCells; i++){
    TEST_FLOATING_EQUALITY(tGold[i],tHost(i),tTol);
  }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CriterionVolumeTwoPhase)
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::ElectricalElement<Plato::Tri3>;
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(0.5, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  
  // create state workset
  Plato::ScalarVector tState("States", tNumVerts);
  Plato::blas1::fill(0.67186, tState);
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  
  // create spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  // create current density
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  Plato::CriterionVolumeTwoPhase<Residual>
    tCriterionVolumeTwoPhase(tOnlyDomainDefined,tDataMap,*tGenericParamList,"My Volume");
  Plato::ScalarVectorT<ResultT> tResultWS("result workset", tNumCells);
  tCriterionVolumeTwoPhase.evaluate(tStateWS,tControlWS,tConfigWS,tResultWS);
  // test against gold
  auto tHost = Kokkos::create_mirror_view(tResultWS);
  Plato::Scalar tTol = 1e-6;
  std::vector<Plato::Scalar>tGold = {0.10375,0.10375};
  Kokkos::deep_copy(tHost, tResultWS);
  for(Plato::OrdinalType i = 0; i < tNumCells; i++){
    TEST_FLOATING_EQUALITY(tGold[i],tHost(i),tTol);
  }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ResidualSteadyStateCurrent_ConstantPotential)
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::ElectricalElement<Plato::Tri3>;
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(0.5, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  
  // create state workset
  Plato::ScalarVector tState("States", tNumVerts);
  Plato::blas1::fill(0.67186, tState);
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  
  // create spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  // create current density
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  Plato::ResidualSteadyStateCurrent<Residual> 
    tResidual(tOnlyDomainDefined,tDataMap,tGenericParamList.operator*());
  Plato::ScalarMultiVectorT<ResultT> tResultWS("result",tNumCells,tNodesPerCell);
  tResidual.evaluate(tStateWS,tControlWS,tConfigWS,tResultWS);

  // test against gold - electric field is zero due to constant electric potential; 
  // thus, internal forces are zero and the residual is equal to minus external forces
  auto tHost = Kokkos::create_mirror_view(tResultWS);
  Plato::Scalar tTol = 1e-6;
  std::vector<std::vector<Plato::Scalar>>tGold = {{3.2098359,3.2098359,3.2098359},
                                                  {3.2098359,3.2098359,3.2098359}};
  Kokkos::deep_copy(tHost, tResultWS);
  for(Plato::OrdinalType i = 0; i < tNumCells; i++){
    for(Plato::OrdinalType j = 0; j < tDofsPerCell; j++){
      TEST_FLOATING_EQUALITY(tGold[i][j],tHost(i,j),tTol);
    }
  }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ResidualSteadyStateCurrent_NonConstantPotential)
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::ElectricalElement<Plato::Tri3>;
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(0.5, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  
  // create state workset
  Plato::ScalarVector tState("States", tNumVerts);
  Plato::blas1::fill(0.67186, tState);
  auto tHostState = Kokkos::create_mirror_view(tState);
  for(Plato::OrdinalType i = 0; i < tNumVerts; i++)
  { tHostState(i) = tHostState(i) + (i*1e-2); }
  Kokkos::deep_copy(tState, tHostState);
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  
  // create spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  // create current density
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  Plato::ResidualSteadyStateCurrent<Residual> 
    tResidual(tOnlyDomainDefined,tDataMap,tGenericParamList.operator*());
  Plato::ScalarMultiVectorT<ResultT> tResultWS("result",tNumCells,tNodesPerCell);
  tResidual.evaluate(tStateWS,tControlWS,tConfigWS,tResultWS);

  // test against gold - electric field is zero due to constant electric potential; 
  // thus, internal forces are zero and the residual is equal to minus external forces
  auto tHost = Kokkos::create_mirror_view(tResultWS);
  Plato::Scalar tTol = 1e-6;
  std::vector<std::vector<Plato::Scalar>>tGold = {{-17.276113,-17.272551,-17.272551},
                                                  {-12.440948,-12.437385,-12.440948}};
  Kokkos::deep_copy(tHost, tResultWS);
  for(Plato::OrdinalType i = 0; i < tNumCells; i++){
    for(Plato::OrdinalType j = 0; j < tDofsPerCell; j++){
      TEST_FLOATING_EQUALITY(tGold[i][j],tHost(i,j),tTol);
    }
  }
}

}