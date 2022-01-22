/*!
  These unit tests are for the TransientThermomech functionality.
 \todo 
*/

#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "ImplicitFunctors.hpp"
#include "LinearThermoelasticMaterial.hpp"

#ifdef HAVE_AMGX
#include "alg/AmgXSparseLinearProblem.hpp"
#endif

#include <sstream>
#include <iostream>
#include <fstream>
#include <type_traits>

#include <Sacado.hpp>
#include <alg/CrsLinearProblem.hpp>
#include <alg/ParallelComm.hpp>
#include <Simp.hpp>
#include <ApplyWeighting.hpp>
#include <SimplexFadTypes.hpp>
#include <WorksetBase.hpp>
#include <parabolic/VectorFunction.hpp>
#include <StateValues.hpp>
#include "ApplyConstraints.hpp"
#include "SimplexThermal.hpp"
#include "Thermomechanics.hpp"
#include "ComputedField.hpp"

#include <fenv.h>


TEUCHOS_UNIT_TEST( TransientThermomechTests, 3D )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;

  static constexpr int TDofOffset = spaceDim;

  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", meshWidth);


  int numCells = tMesh->NumElements();
  constexpr int numVoigtTerms = Plato::SimplexThermomechanics<spaceDim>::mNumVoigtTerms;
  constexpr int nodesPerCell  = Plato::SimplexThermomechanics<spaceDim>::mNumNodesPerCell;
  constexpr int dofsPerCell   = Plato::SimplexThermomechanics<spaceDim>::mNumDofsPerCell;
  constexpr int dofsPerNode   = Plato::SimplexThermomechanics<spaceDim>::mNumDofsPerNode;

  // create mesh based solution from host data
  //
  int tNumDofsPerNode = (spaceDim+1);
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;
  Plato::ScalarVector state("state", tNumDofs);
  Plato::ScalarVector z("control", tNumDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     state(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal;

  }, "state");

  Plato::WorksetBase<Plato::SimplexThermomechanics<spaceDim>> worksetBase(tMesh);

  Plato::ScalarArray3DT<Plato::Scalar>     gradient("gradient",numCells,nodesPerCell,spaceDim);
  Plato::ScalarMultiVectorT<Plato::Scalar> tStrain("strain", numCells, numVoigtTerms);
  Plato::ScalarMultiVectorT<Plato::Scalar> tGrad("temperature gradient", numCells, spaceDim);
  Plato::ScalarMultiVectorT<Plato::Scalar> tStress("stress", numCells, numVoigtTerms);
  Plato::ScalarMultiVectorT<Plato::Scalar> tFlux("thermal flux", numCells, spaceDim);
  Plato::ScalarMultiVectorT<Plato::Scalar> result("result", numCells, dofsPerCell);
  Plato::ScalarArray3DT<Plato::Scalar>     configWS("config workset",numCells, nodesPerCell, spaceDim);
  Plato::ScalarVectorT<Plato::Scalar>      tTemperature("Gauss point temperature", numCells);
  Plato::ScalarVectorT<Plato::Scalar>      tThermalContent("Gauss point heat content at step k", numCells);
  Plato::ScalarMultiVectorT<Plato::Scalar> massResult("mass", numCells, dofsPerCell);
  Plato::ScalarMultiVectorT<Plato::Scalar> stateWS("state workset",numCells, dofsPerCell);

  worksetBase.worksetConfig(configWS);

  worksetBase.worksetState(state, stateWS);


  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                           \n"
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

  Plato::ThermalMassModelFactory<spaceDim> mmmfactory(*params);
  auto massMaterialModel = mmmfactory.create("Cookie Dough");

  Plato::ThermoelasticModelFactory<spaceDim> mmfactory(*params);
  auto materialModel = mmfactory.create("Cookie Dough");

  Plato::ComputeGradientWorkset<spaceDim>  computeGradient;
  Plato::TMKinematics<spaceDim>                   kinematics;
  Plato::TMKinetics<spaceDim>                     kinetics(materialModel);

  Plato::InterpolateFromNodal<spaceDim, dofsPerNode, TDofOffset> interpolateFromNodal;

  Plato::FluxDivergence  <spaceDim, dofsPerNode, TDofOffset> fluxDivergence;
  Plato::StressDivergence<spaceDim, dofsPerNode> stressDivergence;

  Plato::ThermalContent<spaceDim> computeThermalContent(massMaterialModel);
  Plato::ProjectToNode<spaceDim, dofsPerNode, TDofOffset> projectThermalContent;

  Plato::LinearTetCubRuleDegreeOne<spaceDim> cubatureRule;

  Plato::Scalar quadratureWeight = cubatureRule.getCubWeight();
  auto basisFunctions = cubatureRule.getBasisFunctions();

  Plato::Scalar tTimeStep = 1.0;

  Plato::ScalarVectorT<Plato::Scalar> cellVolume("cell volume",numCells);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(int cellOrdinal)
  {
    computeGradient(cellOrdinal, gradient, configWS, cellVolume);
    cellVolume(cellOrdinal) *= quadratureWeight;

    kinematics(cellOrdinal, tStrain, tGrad, stateWS, gradient);

    interpolateFromNodal(cellOrdinal, basisFunctions, stateWS, tTemperature);

    kinetics(cellOrdinal, tStress, tFlux, tStrain, tGrad, tTemperature);

    stressDivergence(cellOrdinal, result, tStress, gradient, cellVolume, tTimeStep/2.0);

    fluxDivergence(cellOrdinal, result, tFlux, gradient, cellVolume, tTimeStep/2.0);

    computeThermalContent(cellOrdinal, tThermalContent, tTemperature);
    projectThermalContent(cellOrdinal, cellVolume, basisFunctions, tThermalContent, massResult);

  }, "divergence");

  // test cell volume
  //
  auto cellVolume_Host = Kokkos::create_mirror_view( cellVolume );
  Kokkos::deep_copy( cellVolume_Host, cellVolume );

  std::vector<Plato::Scalar> cellVolume_gold = { 
  0.02083333333333333, 0.02083333333333333, 0.02083333333333333,
  0.02083333333333333, 0.02083333333333333, 0.02083333333333333,
  0.02083333333333333, 0.02083333333333333, 0.02083333333333333,
  0.02083333333333333, 0.02083333333333333, 0.02083333333333333,
  0.02083333333333333, 0.02083333333333333, 0.02083333333333333,
  0.02083333333333333, 0.02083333333333333, 0.02083333333333333
  };

  int numGoldCells=cellVolume_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    if(cellVolume_gold[iCell] == 0.0){
      TEST_ASSERT(fabs(cellVolume_Host(iCell)) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(cellVolume_Host(iCell), cellVolume_gold[iCell], 1e-13);
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
      TEST_ASSERT(fabs(tTemperature_Host(iCell)) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(tTemperature_Host(iCell), tTemperature_gold[iCell], 1e-13);
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
      TEST_ASSERT(fabs(tThermalContent_Host(iCell)) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(tThermalContent_Host(iCell), tThermalContent_gold[iCell], 1e-13);
    }
  }


  // test gradient operator
  //
  auto gradient_Host = Kokkos::create_mirror_view( gradient );
  Kokkos::deep_copy( gradient_Host, gradient );

  std::vector<std::vector<std::vector<Plato::Scalar>>> gradient_gold = { 
    {{0, -2.0, 0}, {2.0, 0, -2.0}, {-2.0, 2.0, 0}, {0, 0, 2.0}},
    {{0, -2.0, 0}, {0, 2.0, -2.0}, {-2.0, 0, 2.0}, {2.0, 0, 0}},
    {{0, 0, -2.0}, {-2.0, 2.0, 0}, {0, -2.0, 2.0}, {2.0, 0, 0}},
    {{0, 0, -2.0}, {-2.0, 0, 2.0}, {2.0, -2.0, 0}, {0, 2.0, 0}},
    {{-2.0, 0, 0}, {0, -2.0, 2.0}, {2.0, 0, -2.0}, {0, 2.0, 0}},
    {{-2.0, 0, 0}, {2.0, -2.0, 0}, {0, 2.0, -2.0}, {0, 0, 2.0}}
  };

  numGoldCells=gradient_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    for(int iNode=0; iNode<spaceDim+1; iNode++){
      for(int iDim=0; iDim<spaceDim; iDim++){
        if(gradient_gold[iCell][iNode][iDim] == 0.0){
          TEST_ASSERT(fabs(gradient_Host(iCell,iNode,iDim)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(gradient_Host(iCell,iNode,iDim), gradient_gold[iCell][iNode][iDim], 1e-13);
        }
      }
    }
  }

  // test temperature gradient
  //
  auto tgrad_Host = Kokkos::create_mirror_view( tGrad );
  Kokkos::deep_copy( tgrad_Host, tGrad );

  std::vector<std::vector<Plato::Scalar>> tgrad_gold = { 
    {7.2e-6, 2.4e-6, 8.0e-7},
    {7.2e-6, 2.4e-6, 8.0e-7},
    {7.2e-6, 2.4e-6, 8.0e-7},
    {7.2e-6, 2.4e-6, 8.0e-7}
  };

  for(int iCell=0; iCell<int(tgrad_gold.size()); iCell++){
    for(int iDim=0; iDim<spaceDim; iDim++){
      if(tgrad_gold[iCell][iDim] == 0.0){
        TEST_ASSERT(fabs(tgrad_Host(iCell,iDim)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tgrad_Host(iCell,iDim), tgrad_gold[iCell][iDim], 1e-13);
      }
    }
  }

  // test thermal flux
  //
  auto tflux_Host = Kokkos::create_mirror_view( tFlux );
  Kokkos::deep_copy( tflux_Host, tFlux );

  std::vector<std::vector<Plato::Scalar>> tflux_gold = { 
    {7.2e-3, 2.4e-3, 8.0e-4},
    {7.2e-3, 2.4e-3, 8.0e-4},
    {7.2e-3, 2.4e-3, 8.0e-4},
    {7.2e-3, 2.4e-3, 8.0e-4}
  };

  for(int iCell=0; iCell<int(tflux_gold.size()); iCell++){
    for(int iDim=0; iDim<spaceDim; iDim++){
      if(tflux_gold[iCell][iDim] == 0.0){
        TEST_ASSERT(fabs(tflux_Host(iCell,iDim)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tflux_Host(iCell,iDim), tflux_gold[iCell][iDim], 1e-13);
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
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", meshWidth);

  // create mesh based solution from host data
  //
  int tNumDofsPerNode = (spaceDim+1);
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;
  Plato::ScalarVector state("state", tNumDofs);
  Plato::ScalarVector stateDot("state dot", tNumDofs);
  Plato::ScalarVector z("control", tNumDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     state(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal;
     stateDot(aNodeOrdinal*tNumDofsPerNode+0) = (4e-7)*aNodeOrdinal;
     stateDot(aNodeOrdinal*tNumDofsPerNode+1) = (3e-7)*aNodeOrdinal;
     stateDot(aNodeOrdinal*tNumDofsPerNode+2) = (2e-7)*aNodeOrdinal;
     stateDot(aNodeOrdinal*tNumDofsPerNode+3) = (1e-7)*aNodeOrdinal;

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
  Plato::SpatialModel tSpatialModel(tMesh, *params);
  Plato::Parabolic::VectorFunction<::Plato::Thermomechanics<spaceDim>>
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

