#include "util/PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "LinearThermoelasticMaterial.hpp"

#ifdef HAVE_AMGX
#include "solver/AmgXSparseLinearProblem.hpp"
#endif

#include <sstream>
#include <iostream>
#include <fstream>
#include <type_traits>

#include <Sacado.hpp>

#include "Tet4.hpp"
#include "TMKinetics.hpp"
#include "WorksetBase.hpp"
#include "TMKinematics.hpp"
#include "ProjectToNode.hpp"
#include "ApplyWeighting.hpp"
#include "ThermalContent.hpp"
#include "GradientMatrix.hpp"
#include "solver/ParallelComm.hpp"
#include "ThermalMassMaterial.hpp"
#include "InterpolateFromNodal.hpp"
#include "solver/CrsLinearProblem.hpp"
#include "GeneralFluxDivergence.hpp"
#include "GeneralStressDivergence.hpp"
#include "parabolic/VectorFunction.hpp"
#include "ApplyConstraints.hpp"
#include "Thermomechanics.hpp"
#include "ComputedField.hpp"
#include "TMKineticsFactory.hpp"
#include "Mechanics.hpp"

#include <fenv.h>


TEUCHOS_UNIT_TEST( TransientThermomechTests, 3D )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;

  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  using ElementType = typename Plato::ThermomechanicsElement<Plato::Tet4>;

  int numCells = tMesh->NumElements();
  int tNumNodes = tMesh->NumNodes();

  constexpr int numSpaceDims  = ElementType::mNumSpatialDims;
  constexpr int numVoigtTerms = ElementType::mNumVoigtTerms;
  constexpr int nodesPerCell  = ElementType::mNumNodesPerCell;
  constexpr int dofsPerCell   = ElementType::mNumDofsPerCell;
  constexpr int dofsPerNode   = ElementType::mNumDofsPerNode;

  static constexpr int TDofOffset = numSpaceDims;

  // create mesh based solution from host data
  //
  int tNumDofs = tNumNodes*dofsPerNode;
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(const int & aNodeOrdinal)
  {
     state(aNodeOrdinal*dofsPerNode+0) = (1e-7)*aNodeOrdinal;
     state(aNodeOrdinal*dofsPerNode+1) = (2e-7)*aNodeOrdinal;
     state(aNodeOrdinal*dofsPerNode+2) = (3e-7)*aNodeOrdinal;
     state(aNodeOrdinal*dofsPerNode+3) = (4e-7)*aNodeOrdinal;

  }, "state");

  Plato::WorksetBase<ElementType> worksetBase(tMesh);

  Plato::ScalarMultiVectorT<Plato::Scalar> result("result", numCells, dofsPerCell);
  Plato::ScalarMultiVectorT<Plato::Scalar> massResult("mass", numCells, dofsPerCell);
  Plato::ScalarArray3DT<Plato::Scalar>     configWS("config workset",numCells, nodesPerCell, numSpaceDims);
  Plato::ScalarMultiVectorT<Plato::Scalar> stateWS("state workset",numCells, dofsPerCell);
  Plato::ScalarMultiVectorT<Plato::Scalar> controlWS("control", numCells, nodesPerCell);
  Kokkos::deep_copy(controlWS, 1.0);

  worksetBase.worksetConfig(configWS);

  worksetBase.worksetState(state, stateWS);


  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                           \n"
    "  <ParameterList name='Spatial Model'>                                         \n"
    "    <ParameterList name='Domains'>                                             \n"
    "      <ParameterList name='Design Volume'>                                     \n"
    "        <Parameter name='Element Block' type='string' value='body'/>           \n"
    "        <Parameter name='Material Model' type='string' value='Cookie Dough'/>  \n"
    "      </ParameterList>                                                         \n"
    "    </ParameterList>                                                           \n"
    "  </ParameterList>                                                             \n"
    "  <ParameterList name='Material Models'>                                       \n"
    "    <ParameterList name='Cookie Dough'>                                        \n"
    "      <ParameterList name='Thermal Mass'>                                      \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>             \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e6'/>          \n"
    "      </ParameterList>                                                         \n"
    "      <ParameterList name='Thermoelastic'>                                     \n"
    "        <ParameterList name='Elastic Stiffness'>                               \n"
    "          <Parameter  name='Poissons Ratio' type='double' value='0.3'/>        \n"
    "          <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>     \n"
    "        </ParameterList>                                                       \n"
    "        <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/>  \n"
    "        <Parameter  name='Thermal Conductivity' type='double' value='1000.0'/> \n"
    "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>   \n"
    "      </ParameterList>                                                         \n"
    "    </ParameterList>                                                           \n"
    "  </ParameterList>                                                             \n"
    "</ParameterList>                                                               \n"
  );

  using Residual = typename Plato::Elliptic::Evaluation<Plato::ThermomechanicsElement<Plato::Tet4>>::Residual;
  Plato::ThermalMassModelFactory<Residual> mmmfactory(*params);
  auto massMaterialModel = mmmfactory.create("Cookie Dough");

  Plato::ThermoelasticModelFactory<Residual> mmfactory(*params);
  auto materialModel = mmfactory.create("Cookie Dough");

  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *params, tDataMap);
  auto tOnlyDomain = tSpatialModel.Domains.front();

  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::TMKinematics<ElementType>          tKinematics;
  Plato::TMKineticsFactory< Residual, ElementType > tTMKineticsFactory;
  auto pkinetics = tTMKineticsFactory.create(materialModel, tOnlyDomain, tDataMap);
  auto & kinetics = *pkinetics;

  Plato::InterpolateFromNodal<ElementType, dofsPerNode, TDofOffset> interpolateFromNodal;

  Plato::GeneralFluxDivergence  <ElementType, dofsPerNode, TDofOffset> fluxDivergence;
  Plato::GeneralStressDivergence<ElementType, dofsPerNode> stressDivergence;

  Plato::ThermalContent<Residual> computeThermalContent(massMaterialModel);
  Plato::ProjectToNode<ElementType, dofsPerNode, TDofOffset> projectThermalContent;

  Plato::Scalar tTimeStep = 1.0;

  auto tCubPoints = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto numPoints = tCubWeights.size();

  Plato::ScalarArray3DT<Plato::Scalar> tStress("stress", numCells, numPoints, numVoigtTerms);
  Plato::ScalarArray3DT<Plato::Scalar> tFlux  ("flux",   numCells, numPoints, numSpaceDims);
  Plato::ScalarArray3DT<Plato::Scalar> tStrain("strain", numCells, numPoints, numVoigtTerms);
  Plato::ScalarArray3DT<Plato::Scalar> tTGrad ("tgrad",  numCells, numPoints, numSpaceDims);

  Plato::ScalarArray4DT<Plato::Scalar> tGradient("gradient", numCells, numPoints, nodesPerCell, numSpaceDims);

  Plato::ScalarMultiVectorT<Plato::Scalar> tVolume("volume", numCells, numPoints);
  Plato::ScalarMultiVectorT<Plato::Scalar> tTemperature("temperature", numCells, numPoints);
  Plato::ScalarMultiVectorT<Plato::Scalar> tThermalContent("thermal content", numCells, numPoints);

  Kokkos::parallel_for("flux divergence", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {numCells, numPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint = tCubPoints(iGpOrdinal);
    auto basisValues = ElementType::basisValues(tCubPoint);

    tComputeGradient(iCellOrdinal, iGpOrdinal, tCubPoint, configWS, tGradient, tVolume);
    tVolume(iCellOrdinal, iGpOrdinal) *= tCubWeights(iGpOrdinal);

    // compute strain and temperature gradient
    //
    tKinematics(iCellOrdinal, iGpOrdinal, tStrain, tTGrad, stateWS, tGradient);

    tTemperature(iCellOrdinal, iGpOrdinal) = interpolateFromNodal(iCellOrdinal, basisValues, stateWS);
  });

  kinetics(tStress, tFlux, tStrain, tTGrad, tTemperature, controlWS);

  Kokkos::parallel_for("flux divergence", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {numCells, numPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    auto tCubPoint = tCubPoints(iGpOrdinal);
    auto basisValues = ElementType::basisValues(tCubPoint);
    stressDivergence(iCellOrdinal, iGpOrdinal, result, tStress, tGradient, tVolume, tTimeStep/2.0);
    fluxDivergence  (iCellOrdinal, iGpOrdinal, result, tFlux,   tGradient, tVolume, tTimeStep/2.0);

    Plato::Scalar tLocalThermalContent(0.0);
    computeThermalContent(tLocalThermalContent, tTemperature(iCellOrdinal, iGpOrdinal), tTemperature(iCellOrdinal, iGpOrdinal));
    tThermalContent(iCellOrdinal, iGpOrdinal) = tLocalThermalContent;

    projectThermalContent(iCellOrdinal, tVolume(iCellOrdinal, iGpOrdinal), basisValues, tLocalThermalContent, massResult);

  });

  // test cell volume
  //
  auto tVolume_Host = Kokkos::create_mirror_view( tVolume );
  Kokkos::deep_copy( tVolume_Host, tVolume );

  std::vector<Plato::Scalar> tVolume_gold = { 
  0.02083333333333333, 0.02083333333333333, 0.02083333333333333,
  0.02083333333333333, 0.02083333333333333, 0.02083333333333333,
  0.02083333333333333, 0.02083333333333333, 0.02083333333333333,
  0.02083333333333333, 0.02083333333333333, 0.02083333333333333,
  0.02083333333333333, 0.02083333333333333, 0.02083333333333333,
  0.02083333333333333, 0.02083333333333333, 0.02083333333333333
  };

  const int iGP = 0; // only one gauss point in this test
  int numGoldCells=tVolume_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    if(tVolume_gold[iCell] == 0.0){
      TEST_ASSERT(fabs(tVolume_Host(iCell, iGP)) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(tVolume_Host(iCell, iGP), tVolume_gold[iCell], 1e-13);
    }
  }

  // test state values
  //
  auto tTemperature_Host = Kokkos::create_mirror_view( tTemperature );
  Kokkos::deep_copy( tTemperature_Host, tTemperature );

  std::vector<Plato::Scalar> tTemperature_gold = { 
  2.800000000000000e-6, 2.000000000000000e-6, 1.800000000000000e-6,
  2.400000000000000e-6, 3.200000000000000e-6, 3.400000000000000e-6,
  3.200000000000000e-6, 2.400000000000000e-6, 2.200000000000000e-6,
  2.800000000000000e-6, 3.600000000000000e-6, 3.800000000000000e-6
  };

  numGoldCells=tTemperature_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    if(tTemperature_gold[iCell] == 0.0){
      TEST_ASSERT(fabs(tTemperature_Host(iCell, iGP)) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(tTemperature_Host(iCell, iGP), tTemperature_gold[iCell], 1e-13);
    }
  }

  // test thermal content
  //
  auto tThermalContent_Host = Kokkos::create_mirror_view( tThermalContent );
  Kokkos::deep_copy( tThermalContent_Host, tThermalContent );

  std::vector<Plato::Scalar> tThermalContent_gold = { 
  0.8400000000000000, 0.6000000000000000, 0.5399999999999999,
  0.7200000000000000, 0.9600000000000000, 1.020000000000000,
  0.9600000000000000, 0.7200000000000000, 0.6600000000000000,
  0.8400000000000001, 1.080000000000000,  1.140000000000000
  };

  numGoldCells=tThermalContent_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    if(tThermalContent_gold[iCell] == 0.0){
      TEST_ASSERT(fabs(tThermalContent_Host(iCell, iGP)) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(tThermalContent_Host(iCell, iGP), tThermalContent_gold[iCell], 1e-13);
    }
  }


  // test gradient operator
  //
  auto tGradient_Host = Kokkos::create_mirror_view( tGradient );
  Kokkos::deep_copy( tGradient_Host, tGradient );

  std::vector<std::vector<std::vector<Plato::Scalar>>> tGradient_gold = { 
    {{0, -2.0, 0}, {2.0, 0, -2.0}, {-2.0, 2.0, 0}, {0, 0, 2.0}},
    {{0, -2.0, 0}, {0, 2.0, -2.0}, {-2.0, 0, 2.0}, {2.0, 0, 0}},
    {{0, 0, -2.0}, {-2.0, 2.0, 0}, {0, -2.0, 2.0}, {2.0, 0, 0}},
    {{0, 0, -2.0}, {-2.0, 0, 2.0}, {2.0, -2.0, 0}, {0, 2.0, 0}},
    {{-2.0, 0, 0}, {0, -2.0, 2.0}, {2.0, 0, -2.0}, {0, 2.0, 0}},
    {{-2.0, 0, 0}, {2.0, -2.0, 0}, {0, 2.0, -2.0}, {0, 0, 2.0}}
  };

  numGoldCells=tGradient_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    for(int iNode=0; iNode<nodesPerCell; iNode++){
      for(int iDim=0; iDim<numSpaceDims; iDim++){
        if(tGradient_gold[iCell][iNode][iDim] == 0.0){
          TEST_ASSERT(fabs(tGradient_Host(iCell,iGP,iNode,iDim)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tGradient_Host(iCell,iGP,iNode,iDim), tGradient_gold[iCell][iNode][iDim], 1e-13);
        }
      }
    }
  }

  // test temperature gradient
  //
  auto tTGrad_Host = Kokkos::create_mirror_view( tTGrad );
  Kokkos::deep_copy( tTGrad_Host, tTGrad );

  std::vector<std::vector<Plato::Scalar>> tTGrad_gold = { 
    {7.2e-6, 2.4e-6, 8.0e-7},
    {7.2e-6, 2.4e-6, 8.0e-7},
    {7.2e-6, 2.4e-6, 8.0e-7},
    {7.2e-6, 2.4e-6, 8.0e-7}
  };

  for(int iCell=0; iCell<int(tTGrad_gold.size()); iCell++){
    for(int iDim=0; iDim<numSpaceDims; iDim++){
      if(tTGrad_gold[iCell][iDim] == 0.0){
        TEST_ASSERT(fabs(tTGrad_Host(iCell,iGP,iDim)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tTGrad_Host(iCell,iGP,iDim), tTGrad_gold[iCell][iDim], 1e-13);
      }
    }
  }

  // test thermal flux
  //
  auto tFlux_Host = Kokkos::create_mirror_view( tFlux );
  Kokkos::deep_copy( tFlux_Host, tFlux );

  std::vector<std::vector<Plato::Scalar>> tFlux_gold = { 
    {7.2e-3, 2.4e-3, 8.0e-4},
    {7.2e-3, 2.4e-3, 8.0e-4},
    {7.2e-3, 2.4e-3, 8.0e-4},
    {7.2e-3, 2.4e-3, 8.0e-4}
  };

  for(int iCell=0; iCell<int(tFlux_gold.size()); iCell++){
    for(int iDim=0; iDim<numSpaceDims; iDim++){
      if(tFlux_gold[iCell][iDim] == 0.0){
        TEST_ASSERT(fabs(tFlux_Host(iCell,iGP,iDim)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tFlux_Host(iCell,iGP,iDim), tFlux_gold[iCell][iDim], 1e-13);
      }
    }
  }
}

/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         ElastostaticResidual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( TransientThermomechTests, TransientThermomechResidual3D )
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  using ElementType = typename Plato::ThermomechanicsElement<Plato::Tet4>;

  int numCells = tMesh->NumElements();
  int tNumNodes = tMesh->NumNodes();

  constexpr int dofsPerNode   = ElementType::mNumDofsPerNode;

  // create mesh based solution from host data
  //
  int tNumDofs = tNumNodes*dofsPerNode;
  Plato::ScalarVector state("state", tNumDofs);
  Plato::ScalarVector stateDot("state dot", tNumDofs);
  Plato::ScalarVector z("control", tNumDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     state(aNodeOrdinal*dofsPerNode+0) = (1e-7)*aNodeOrdinal;
     state(aNodeOrdinal*dofsPerNode+1) = (2e-7)*aNodeOrdinal;
     state(aNodeOrdinal*dofsPerNode+2) = (3e-7)*aNodeOrdinal;
     state(aNodeOrdinal*dofsPerNode+3) = (4e-7)*aNodeOrdinal;
     stateDot(aNodeOrdinal*dofsPerNode+0) = (4e-7)*aNodeOrdinal;
     stateDot(aNodeOrdinal*dofsPerNode+1) = (3e-7)*aNodeOrdinal;
     stateDot(aNodeOrdinal*dofsPerNode+2) = (2e-7)*aNodeOrdinal;
     stateDot(aNodeOrdinal*dofsPerNode+3) = (1e-7)*aNodeOrdinal;

  }, "state");


  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Parabolic'/>          \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                  \n"
    "  <ParameterList name='Parabolic'>                                            \n"
    "    <ParameterList name='Penalty Function'>                                   \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                  \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Frozen Peas'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Frozen Peas'>                                        \n"
    "      <ParameterList name='Thermal Mass'>                                     \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>            \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e6'/>         \n"
    "      </ParameterList>                                                        \n"
    "      <ParameterList name='Thermoelastic'>                                    \n"
    "        <ParameterList name='Elastic Stiffness'>                              \n"
    "          <Parameter  name='Poissons Ratio' type='double' value='0.3'/>       \n"
    "          <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>    \n"
    "        </ParameterList>                                                      \n"
    "        <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/> \n"
    "        <Parameter  name='Thermal Conductivity' type='double' value='1000.0'/>\n"
    "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Time Integration'>                                     \n"
    "    <Parameter name='Number Time Steps' type='int' value='3'/>                \n"
    "    <Parameter name='Time Step' type='double' value='0.5'/>                   \n"
    "    <Parameter name='Trapezoid Alpha' type='double' value='0.5'/>             \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create constraint evaluator
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *params, tDataMap);
  Plato::Parabolic::VectorFunction<::Plato::Thermomechanics<Plato::Tet4>>
    vectorFunction(tSpatialModel, tDataMap, *params, params->get<std::string>("PDE Constraint"));


  // compute and test value
  //
  auto timeStep = params->sublist("Time Integration").get<Plato::Scalar>("Time Step");
  auto residual = vectorFunction.value(state, stateDot, z, timeStep);

  auto residual_Host = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy( residual_Host, residual );

  std::vector<Plato::Scalar> residual_gold = { 
  -60255.72275641025,    -45512.32051282050,    -46153.40865384614,
   0.005227083333333332, -63460.51762820510,    -57691.53685897433,
  -37499.91666666666,     0.007471874999999999, -3204.836538461539,
  -12179.25801282051,     8653.325320512817,     0.001619791666666667,
  -70191.07852564102,    -30768.98076923076,    -58652.95032051280,
   0.009781250000000000, -86536.33653846150,    -40384.24038461538,
  -53846.02884615383,     0.01429375000000000,  -16345.25801282050,
  -9615.259615384608,     4806.671474358979,     0.003887500000000000,
  -9935.480769230770,     14742.83974358974,    -12499.66666666666,
   0.002679166666666667, -23075.81891025639,     17306.54647435897
  };

  for(int iNode=0; iNode<int(residual_gold.size()); iNode++){
    if(residual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(residual_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(residual_Host[iNode], residual_gold[iNode], 1e-13);
    }
  }


  // compute and test gradient wrt state. (i.e., jacobian)
  //
  auto jacobian = vectorFunction.gradient_u(state, stateDot, z, timeStep);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
3.52564102564102478e10, 0, 0, 52083.3333333333285, 0,
3.52564102564102478e10, 0, 52083.3333333333285, 0, 0,
3.52564102564102478e10, 52083.3333333333285, 0, 0, 0,
499.999999999999943, -6.41025641025640965e9, 0,
3.20512820512820482e9, 0, 0, -6.41025641025640965e9,
3.20512820512820482e9, 0, 4.80769230769230652e9,
4.80769230769230652e9, -2.24358974358974304e10,
52083.3333333333285, 0, 0, 0, -166.666666666666657,
-6.41025641025640965e9, 3.20512820512820482e9, 0, 0,
4.80769230769230652e9, -2.24358974358974304e10,
4.80769230769230652e9, 52083.3333333333285, 0,
3.20512820512820482e9, -6.41025641025640965e9, 0, 0, 0, 0,
-166.666666666666657, 0, 3.20512820512820482e9,
3.20512820512820482e9, 0, 4.80769230769230652e9, 0,
-8.01282051282051086e9, 26041.6666666666642,
4.80769230769230652e9, -8.01282051282051086e9, 0,
26041.6666666666642, 0, 0, 0, 0
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }


  // compute and test gradient wrt state dot (i.e., jacobianV)
  //
  auto jacobian_v = vectorFunction.gradient_v(state, stateDot, z, timeStep);

  auto jac_v_entries = jacobian_v->entries();
  auto jac_v_entriesHost = Kokkos::create_mirror_view( jac_v_entries );
  Kokkos::deep_copy(jac_v_entriesHost, jac_v_entries);

  std::vector<Plato::Scalar> gold_jac_v_entries = {
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2343.75000000000000,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 781.250000000000000,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 781.250000000000000,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 781.250000000000000
  };

  int jac_v_entriesSize = gold_jac_v_entries.size();
  for(int i=0; i<jac_v_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_v_entriesHost(i), gold_jac_v_entries[i], 1.0e-15);
  }


  // compute and test objective gradient wrt control, z
  //
  auto gradient_z = vectorFunction.gradient_z(state, stateDot, z, timeStep);
  
  auto grad_entries = gradient_z->entries();
  auto grad_entriesHost = Kokkos::create_mirror_view( grad_entries );
  Kokkos::deep_copy(grad_entriesHost, grad_entries);

  std::vector<Plato::Scalar> gold_grad_entries = {
-15063.9306891025626, -11378.0801282051252, -11538.3521634615354,
0.00130677083333333296, -801.219551282049906, -3044.82491987179446,
2163.35216346153675, 0.000326822916666666614, -2483.90144230769147,
3685.77243589743557, -3124.94791666666515, 0.000435416666666666634,
-3285.15745192307486, 640.978766025640425, -961.590544871795146,
0.000254427083333333285
  };

  int grad_entriesSize = gold_grad_entries.size();
  for(int i=0; i<grad_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_entriesHost(i), gold_grad_entries[i], 2.0e-14);
  }


  // compute and test objective gradient wrt node position, x
  //
  auto gradient_x = vectorFunction.gradient_x(state, stateDot, z, timeStep);
  
  auto grad_x_entries = gradient_x->entries();
  auto grad_x_entriesHost = Kokkos::create_mirror_view( grad_x_entries );
  Kokkos::deep_copy(grad_x_entriesHost, grad_x_entries);

  std::vector<Plato::Scalar> gold_grad_x_entries = {
-63461.5384615384464, -126923.076923076878, -190384.615384615347,
-0.00875624999999999841, -21153.8461538461415, -42307.6923076922903,
-63461.5384615384537, -0.00494999999999999780, -7051.28205128204081,
-14102.5641025640871, -21153.8461538461452, -0.00368124999999999963,
-32371.7948717948639, -9935.89743589742466, 82692.8076923076878,
0.00113333333333333320, -22756.4102564102504, -8012.82051282051179,
13461.9134615384592, 0.000333333333333333160, 40704.6282051281887,
38140.6506410256334, 36538.4615384615317, -0.00234791666666666655,
-19230.7692307692232, 32692.8910256410163, 10256.4102564102541,
0.000999999999999999804, 44871.2115384615172, 39743.5897435897423,
44871.3782051282033, -0.00268333333333333314, -14102.5641025640944,
-18589.3269230769292, -5128.20512820512522,
-0.0000666666666666667512, -74679.4871794871433, 5449.13461538461343,
-5127.83012820512522, -0.000266666666666666625, 14422.6602564102468,
25961.5384615384537, 25641.0673076923085, -0.000962500000000000031,
24038.0865384615281, 17628.1634615384646, 20512.8205128205082,
-0.000806250000000000001
  };

  int grad_x_entriesSize = gold_grad_x_entries.size();
  for(int i=0; i<grad_x_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_x_entriesHost(i), gold_grad_x_entries[i], 1.0e-13);
  }

}
