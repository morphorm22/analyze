/*!
  These unit tests are for the stabilized two-field formulation
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
#include <BLAS1.hpp>
#include <ApplyWeighting.hpp>
#include <SimplexFadTypes.hpp>
#include <WorksetBase.hpp>
#include <parabolic/AbstractVectorFunction.hpp>
#include <VectorFunctionVMS.hpp>
#include <StateValues.hpp>
#include <Plato_Solve.hpp>
#include "SpatialModel.hpp"
#include "ApplyConstraints.hpp"
#include "PressureDivergence.hpp"
#include "StabilizedThermomechanics.hpp"
#include "ThermalContent.hpp"
#include "ComputedField.hpp"
#include "PlatoMathHelpers.hpp"

#include <fenv.h>


TEUCHOS_UNIT_TEST( StabilizedThermomechTests, 3D )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;

  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", meshWidth);


  int tNumCells = tMesh->NumElements();
  constexpr int numVoigtTerms = Plato::SimplexStabilizedThermomechanics<spaceDim>::mNumVoigtTerms;
  constexpr int nodesPerCell  = Plato::SimplexStabilizedThermomechanics<spaceDim>::mNumNodesPerCell;
  constexpr int dofsPerCell   = Plato::SimplexStabilizedThermomechanics<spaceDim>::mNumDofsPerCell;
  constexpr int dofsPerNode   = Plato::SimplexStabilizedThermomechanics<spaceDim>::mNumDofsPerNode;

  static constexpr int PDofOffset = spaceDim;
  static constexpr int TDofOffset = spaceDim+1;

  // create mesh based solution from host data
  //
  int tNumDofsPerNode = (spaceDim+2); // displacements + pressure + temperature
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  Plato::ScalarMultiVector tPGradWS("Projected pressure gradient workset", tNumDofs, spaceDim*nodesPerCell);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
  {
      for(int iNode=0; iNode<nodesPerCell; iNode++)
      {
          for(int iDim=0; iDim<spaceDim; iDim++)
          {
              tPGradWS(aCellOrdinal, iNode*spaceDim+iDim) = (4e-7)*(iNode+1)*(iDim+1)*(aCellOrdinal+1);
          }
      }
  }, "projected pgrad");

  Plato::ScalarVector state("state", tNumDofs);
  Plato::ScalarVector z("control", tNumDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     state(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+3) = (5e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+4) = (4e-7)*aNodeOrdinal;
  }, "state");

  Plato::WorksetBase<Plato::SimplexStabilizedThermomechanics<spaceDim>> worksetBase(tMesh);

  Plato::ScalarArray3D     gradient           ("gradient",           tNumCells, nodesPerCell, spaceDim);
  Plato::ScalarArray3D     configWS           ("config workset",     tNumCells, nodesPerCell, spaceDim);
  Plato::ScalarMultiVector tDispGrad          ("sym disp grad",      tNumCells, numVoigtTerms);
  Plato::ScalarMultiVector tDevStress         ("deviatoric stress",  tNumCells, numVoigtTerms);
  Plato::ScalarMultiVector tStressDivResult   ("stress div",         tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tPressureDivResult ("pressure div",       tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tStabDivResult     ("stabilization div",  tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tFluxDivResult     ("thermal flux div",   tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tVolResult         ("volume diff proj",   tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tMassResult        ("mass",               tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tStateWS           ("state workset",      tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tPressureGrad      ("pressure grad",      tNumCells, spaceDim);
  Plato::ScalarMultiVector tProjectedPGrad    ("projected p grad",   tNumCells, spaceDim);
  Plato::ScalarMultiVector tTGrad             ("Temperature grad",   tNumCells, spaceDim);
  Plato::ScalarMultiVector tCellStab          ("cell stabilization", tNumCells, spaceDim);
  Plato::ScalarMultiVector tTFlux             ("thermal flux",       tNumCells, spaceDim);
  Plato::ScalarVector      tVolStrain         ("volume strain",      tNumCells);
  Plato::ScalarVector      tTemperature       ("GP temperature",     tNumCells);
  Plato::ScalarVector      tPressure          ("GP pressure",        tNumCells);
  Plato::ScalarVector      tThermalContent    ("GP heat at step k",  tNumCells);

  worksetBase.worksetConfig(configWS);

  worksetBase.worksetState(state, tStateWS);

  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                      \n"
    "  <ParameterList name='Material Models'>                                                  \n"
    "    <ParameterList name='Unobtainium'>                                                    \n"
    "      <ParameterList name='Thermal Mass'>                                                 \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>                        \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e6'/>                     \n"
    "      </ParameterList>                                                                    \n"
    "      <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.499'/>                   \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>                  \n"
    "        <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/>   \n"
    "        <Parameter  name='Thermal Conductivity' type='double' value='1000.0'/>\n"
    "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "      </ParameterList>                                                                    \n"
    "    </ParameterList>                                                                      \n"
    "  </ParameterList>                                                                        \n"
    "</ParameterList>                                                                          \n"
  );

  Plato::ThermalMassModelFactory<spaceDim> mmmfactory(*params);
  auto massMaterialModel = mmmfactory.create("Unobtainium");

  Plato::LinearThermoelasticModelFactory<spaceDim> mmfactory(*params);
  auto materialModel = mmfactory.create("Unobtainium");

  Plato::ComputeGradientWorkset<spaceDim>    computeGradient;
  Plato::StabilizedTMKinematics<spaceDim>      kinematics;
  Plato::StabilizedTMKinetics<spaceDim>        kinetics(materialModel);

  Plato::InterpolateFromNodal<spaceDim, dofsPerNode, TDofOffset>  interpolateTemperatureFromNodal;
  Plato::InterpolateFromNodal<spaceDim, dofsPerNode, PDofOffset>  interpolatePressureFromNodal;
  Plato::InterpolateFromNodal<spaceDim, spaceDim, 0, spaceDim>    interpolatePGradFromNodal;

  Plato::FluxDivergence     <spaceDim, dofsPerNode, TDofOffset> fluxDivergence;
  Plato::FluxDivergence     <spaceDim, dofsPerNode, PDofOffset> stabDivergence;
  Plato::StressDivergence   <spaceDim, dofsPerNode>             stressDivergence;
  Plato::PressureDivergence <spaceDim, dofsPerNode>             pressureDivergence;

  Plato::ThermalContent<spaceDim> computeThermalContent(massMaterialModel);

  Plato::ProjectToNode<spaceDim, dofsPerNode, PDofOffset> projectVolumeStrain;
  Plato::ProjectToNode<spaceDim, dofsPerNode, TDofOffset> projectThermalContent;

  Plato::LinearTetCubRuleDegreeOne<spaceDim> cubatureRule;

  Plato::Scalar quadratureWeight = cubatureRule.getCubWeight();
  auto basisFunctions = cubatureRule.getBasisFunctions();

  Plato::Scalar tTimeStep = 2.0;

  Plato::ScalarVectorT<Plato::Scalar> cellVolume("cell volume",tNumCells);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(int cellOrdinal)
  {
    computeGradient(cellOrdinal, gradient, configWS, cellVolume);
    cellVolume(cellOrdinal) *= quadratureWeight;

    kinematics(cellOrdinal, tDispGrad, tPressureGrad, tTGrad, tStateWS, gradient);

    interpolatePGradFromNodal        ( cellOrdinal, basisFunctions, tPGradWS, tProjectedPGrad );
    interpolatePressureFromNodal     ( cellOrdinal, basisFunctions, tStateWS, tPressure       );
    interpolateTemperatureFromNodal  ( cellOrdinal, basisFunctions, tStateWS, tTemperature    );

    kinetics(cellOrdinal, cellVolume,
             tProjectedPGrad, tPressure,     tTemperature,
             tDispGrad,       tPressureGrad, tTGrad,
             tDevStress,      tVolStrain,    tTFlux,  tCellStab);

    stressDivergence   (cellOrdinal, tStressDivResult,   tDevStress, gradient, cellVolume, tTimeStep/2.0);
    pressureDivergence (cellOrdinal, tPressureDivResult, tPressure,  gradient, cellVolume, tTimeStep/2.0);
    stabDivergence     (cellOrdinal, tStabDivResult,     tCellStab,  gradient, cellVolume, tTimeStep/2.0);
    fluxDivergence     (cellOrdinal, tFluxDivResult,     tTFlux,     gradient, cellVolume, tTimeStep/2.0);

    computeThermalContent(cellOrdinal, tThermalContent, tTemperature);
    projectVolumeStrain  (cellOrdinal, cellVolume, basisFunctions, tVolStrain, tVolResult);
    projectThermalContent(cellOrdinal, cellVolume, basisFunctions, tThermalContent, tMassResult);

  }, "divergence");

  {
    // test deviatoric stress
    //
    auto tDevStress_Host = Kokkos::create_mirror_view( tDevStress );
    Kokkos::deep_copy( tDevStress_Host, tDevStress );

    std::vector<std::vector<double>> gold = {
      { 40026.6844563111663, 0.00000000000000000,-40026.6844562962651,73382.2548365577095,186791.194129419659,140093.395597064722}
    };

    int tNumCells=gold.size(), numVoigt=6;
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iVoigt=0; iVoigt<numVoigt; iVoigt++){
        TEST_FLOATING_EQUALITY(tDevStress_Host(iCell, iVoigt), gold[iCell][iVoigt], 1e-12);
      }
    }
  }

  {
    // test volume strain
    //
    auto tVolStrain_Host = Kokkos::create_mirror_view( tVolStrain );
    Kokkos::deep_copy( tVolStrain_Host, tVolStrain );

    std::vector<Plato::Scalar> gold = { 3.59991600000000048e-6 };

    int tNumCells=gold.size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      TEST_FLOATING_EQUALITY(tVolStrain_Host(iCell), gold[iCell], 1e-13);
    }
  }

  {
    // test cell stabilization
    //
    auto tCellStab_Host = Kokkos::create_mirror_view( tCellStab );
    Kokkos::deep_copy( tCellStab_Host, tCellStab );

    std::vector<std::vector<Plato::Scalar>> gold = { {9.07954589551792534e-18, 1.13494323693974086e-18, -2.26988647387948287e-18} };

    int tNumCells=gold.size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iDim=0; iDim<spaceDim; iDim++){
        TEST_FLOATING_EQUALITY(tCellStab_Host(iCell, iDim), gold[iCell][iDim], 1e-13);
      }
    }
  }

  {
    // test thermal flux
    //
    auto tflux_Host = Kokkos::create_mirror_view( tTFlux );
    Kokkos::deep_copy( tflux_Host, tTFlux );

    std::vector<std::vector<Plato::Scalar>> tflux_gold = { {0.0072000000,0.0024000000,0.00080000000} };

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

  // test cell volume
  //
  auto cellVolume_Host = Kokkos::create_mirror_view( cellVolume );
  Kokkos::deep_copy( cellVolume_Host, cellVolume );

  std::vector<Plato::Scalar> cellVolume_gold = { 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333
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
    0.840,0.600,0.540,0.720,0.960,1.02,0.960,0.720,0.660,0.840,1.08,1.14
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

  {
    // test projected pressure gradient
    //
    auto tProjectedPGrad_Host = Kokkos::create_mirror_view( tProjectedPGrad );
    Kokkos::deep_copy( tProjectedPGrad_Host, tProjectedPGrad );

    std::vector<std::vector<Plato::Scalar>> gold = { 
      {1.00000000000000000e-6, 2.00000000000000000e-6, 3.00000000000000000e-6},
      {2.00000000000000000e-6, 4.00000000000000000e-6, 6.00000000000000000e-6},
      {3.00000000000000000e-6, 6.00000000000000000e-6, 9.00000000000000000e-6}
    };

    for(int iCell=0; iCell<int(gold.size()); iCell++){
      for(int iDim=0; iDim<spaceDim; iDim++){
        TEST_FLOATING_EQUALITY(tProjectedPGrad_Host(iCell,iDim), gold[iCell][iDim], 1e-13);
      }
    }
  }

  {
    // test pressure gradient
    //
    auto tPressureGrad_Host = Kokkos::create_mirror_view( tPressureGrad );
    Kokkos::deep_copy( tPressureGrad_Host, tPressureGrad );

    std::vector<std::vector<Plato::Scalar>> gold = { 
      {9.0000000000000002e-06, 3.0000000000000001e-06, 9.999999999999989e-07},
      {8.9999999999999985e-06, 3.0000000000000001e-06, 9.9999999999999974e-07},
      {8.9999999999999985e-06, 3.0000000000000001e-06, 9.9999999999999995e-07}
    };

    for(int iCell=0; iCell<int(gold.size()); iCell++){
      for(int iDim=0; iDim<spaceDim; iDim++){
        TEST_FLOATING_EQUALITY(tPressureGrad_Host(iCell,iDim), gold[iCell][iDim], 1e-13);
      }
    }
  }

  // test temperature gradient
  //
  auto tgrad_Host = Kokkos::create_mirror_view( tTGrad );
  Kokkos::deep_copy( tgrad_Host, tTGrad );

  std::vector<std::vector<Plato::Scalar>> tgrad_gold = { 
    {7.2e-06, 2.4e-06, 8.0e-07},
    {7.2e-06, 2.4e-06, 8.0e-07},
    {7.2e-06, 2.4e-06, 8.0e-07},
    {7.2e-06, 2.4e-06, 8.0e-07}
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

  // test stress divergence and local assembly
  //
  {
    auto tStressDivResult_Host = Kokkos::create_mirror_view( tStressDivResult );
    Kokkos::deep_copy( tStressDivResult_Host, tStressDivResult );

    std::vector<std::vector<double>> gold = { 
      {-5837.2248165443616,  0.0000000000000000, -3057.5939515232367, 0.0000000000000000, 0.0000000000000000,
       -6115.1879030464952,  2779.6308650211249,  9450.7449410718036, 0.0000000000000000, 0.0000000000000000,
        4169.4462975317074, -5837.2248165443616, -4725.3724705359127, 0.0000000000000000, 0.0000000000000000,
        7782.9664220591494,  3057.5939515232367, -1667.7785190126547, 0.0000000000000000, 0.0000000000000000}
    };

    for(int iCell=0; iCell<int(gold.size()); iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tStressDivResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tStressDivResult_Host(iCell,iDof), gold[iCell][iDof], 1e-12);
        }
      }
    }
  }

  // test pressure divergence and local assembly
  //
  {
    auto tPressureDivResult_Host = Kokkos::create_mirror_view( tPressureDivResult );
    Kokkos::deep_copy( tPressureDivResult_Host, tPressureDivResult );

    std::vector<std::vector<Plato::Scalar>> gold = { 
    { 0.0000000000000000,    -1.4583333333333332e-07, 0.0000000000000000,     0.0000000000000000, 0.0000000000000000,
      1.4583333333333332e-07, 0.0000000000000000,    -1.4583333333333332e-07, 0.0000000000000000, 0.0000000000000000,
     -1.4583333333333332e-07, 1.4583333333333332e-07, 0.0000000000000000,     0.0000000000000000, 0.0000000000000000,
      0.0000000000000000,     0.0000000000000000,     1.4583333333333332e-07, 0.0000000000000000, 0.0000000000000000 }
    };

    for(int iCell=0; iCell<int(gold.size()); iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tPressureDivResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tPressureDivResult_Host(iCell,iDof), gold[iCell][iDof], 1e-13);
        }
      }
    }
  }

  // test stabilization divergence and local assembly
  //
  {
    auto tStabDivResult_Host = Kokkos::create_mirror_view( tStabDivResult );
    Kokkos::deep_copy( tStabDivResult_Host, tStabDivResult );

    std::vector<std::vector<Plato::Scalar>> gold = { 
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -4.7289301539155877e-20, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  4.7289301539155868e-19, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -3.3102511077409104e-19, 0.0000000000000000,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -9.4578603078311802e-20, 0.0000000000000000},
    };

    for(int iCell=0; iCell<int(gold.size()); iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tStabDivResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tStabDivResult_Host(iCell,iDof), gold[iCell][iDof], 1e-13);
        }
      }
    }
  }

  // test thermal flux divergence and local assembly
  //
  {
    auto tFluxDivResult_Host = Kokkos::create_mirror_view( tFluxDivResult );
    Kokkos::deep_copy( tFluxDivResult_Host, tFluxDivResult );

    std::vector<std::vector<Plato::Scalar>> gold = { 
      {0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -9.9999999999999991e-05,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  0.00026666666666666668,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, -0.00019999999999999998,
       0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000,  3.3333333333333294e-05}
    };

    for(int iCell=0; iCell<int(gold.size()); iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tFluxDivResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tFluxDivResult_Host(iCell,iDof), gold[iCell][iDof], 1e-13);
        }
      }
    }
  }

  {
    // test projected volume eqn
    //
    auto tVolResult_Host = Kokkos::create_mirror_view( tVolResult );
    Kokkos::deep_copy( tVolResult_Host, tVolResult );

    std::vector<std::vector<Plato::Scalar>> gold = { 
      {0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.8749562499998905e-08, 0.000000000000000000,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.8749562499998905e-08, 0.000000000000000000,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.8749562499998905e-08, 0.000000000000000000,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.8749562499998905e-08, 0.000000000000000000}
    };

    for(int iCell=0; iCell<int(gold.size()); iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tVolResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tVolResult_Host(iCell,iDof), gold[iCell][iDof], 1e-13);
        }
      }
    }
  }

  {
    // test projected mass
    //
    auto tMassResult_Host = Kokkos::create_mirror_view( tMassResult );
    Kokkos::deep_copy( tMassResult_Host, tMassResult );

    std::vector<std::vector<Plato::Scalar>> gold = { 
      {0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00437499999999999876,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00437499999999999876,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00437499999999999876,
       0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00437499999999999876}
    };

    for(int iCell=0; iCell<int(gold.size()); iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tMassResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tMassResult_Host(iCell,iDof), gold[iCell][iDof], 1e-13);
        }
      }
    }
  }
}


/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         StabilizedThermomechResidual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( StabilizedThermomechTests, StabilizedThermomechResidual3D )
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", meshWidth);

  // create mesh based solution from host data
  //
  int tNumDofsPerNode = (spaceDim+2);
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  Plato::ScalarVector tState        ("state",         tNumDofs);
  Plato::ScalarVector tControl      ("control",       tNumNodes);
  Plato::ScalarVector tProjPGrad    ("ProjPGrad",     tNumNodes*spaceDim);
  Plato::ScalarVector tProjectState ("Project state", tNumNodes);

  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
     tControl(aNodeOrdinal) = 1.0;

     tState(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal;
     tState(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal;
     tState(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal;
     tState(aNodeOrdinal*tNumDofsPerNode+4) = (4e-7)*aNodeOrdinal;
     tState(aNodeOrdinal*tNumDofsPerNode+3) =    0.0*aNodeOrdinal;
  }, "state");


  // create input for stabilized thermomechanics
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                      \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>                       \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                              \n"
    "  <ParameterList name='Stabilized Elliptic'>                                              \n"
    "    <ParameterList name='Penalty Function'>                                               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                              \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                                 \n"
    "    </ParameterList>                                                                      \n"
    "  </ParameterList>                                                                        \n"
    "  <ParameterList name='Spatial Model'>                                                    \n"
    "    <ParameterList name='Domains'>                                                        \n"
    "      <ParameterList name='Design Volume'>                                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                      \n"
    "        <Parameter name='Material Model' type='string' value='Kryptonite'/>               \n"
    "      </ParameterList>                                                                    \n"
    "    </ParameterList>                                                                      \n"
    "  </ParameterList>                                                                        \n"
    "  <ParameterList name='Material Models'>                                                  \n"
    "    <ParameterList name='Kryptonite'>                                                     \n"
    "      <ParameterList name='Thermal Mass'>                                                 \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>                        \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e6'/>                     \n"
    "      </ParameterList>                                                                    \n"
    "      <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.499'/>                   \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>                  \n"
    "        <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/>   \n"
    "        <Parameter  name='Thermal Conductivity' type='double' value='1000.0'/>\n"
    "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "      </ParameterList>                                                                    \n"
    "    </ParameterList>                                                                      \n"
    "  </ParameterList>                                                                        \n"
    "  <ParameterList name='Time Stepping'>                                                    \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>                            \n"
    "    <Parameter name='Time Step' type='double' value='1.0'/>                               \n"
    "  </ParameterList>                                                                        \n"
    "  <ParameterList name='Newton Iteration'>                                                 \n"
    "    <Parameter name='Number Iterations' type='int' value='2'/>                            \n"
    "  </ParameterList>                                                                        \n"
    "</ParameterList>                                                                          \n"
  );

  // create constraint evaluator
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *params);
  Plato::VectorFunctionVMS<::Plato::StabilizedThermomechanics<spaceDim>>
    vectorFunction(tSpatialModel, tDataMap, *params, params->get<std::string>("PDE Constraint"));

  // create input pressure gradient projector
  //
  Teuchos::RCP<Teuchos::ParameterList> paramsProjector =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter name='Mass Density' type='double' value='0.3'/>                        \n"
    "      <Parameter name='Specific Heat' type='double' value='1.0e6'/>                     \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.499'/>                   \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>                  \n"
    "      <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/>   \n"
    "      <Parameter  name='Thermal Conductivity' type='double' value='1000.0'/>\n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

  // copy projection state
  Plato::blas1::extract<Plato::StabilizedThermomechanics<spaceDim>::mNumDofsPerNode,
                 Plato::StabilizedThermomechanics<spaceDim>::ProjectorT::SimplexT::mProjectionDof>(tState, tProjectState);


  // create constraint evaluator
  //
  Plato::VectorFunctionVMS<::Plato::Projection<spaceDim>>
    tProjectorVectorFunction(tSpatialModel, tDataMap, *paramsProjector, "State Gradient Projection");

  auto tProjResidual = tProjectorVectorFunction.value      (tProjPGrad, tProjectState, tControl);
  auto tProjJacobian = tProjectorVectorFunction.gradient_u (tProjPGrad, tProjectState, tControl);


  Plato::Solve::RowSummed<spaceDim>(tProjJacobian, tProjPGrad, tProjResidual);


  // compute and test value
  //
  auto timeStep = params->sublist("Time Stepping").get<Plato::Scalar>("Time Step");
  auto tResidual = vectorFunction.value(tState, tProjPGrad, tControl, timeStep);

  auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
  Kokkos::deep_copy( tResidual_Host, tResidual );

  std::vector<double> tGold = { 
    -30575.9395152326506, -17789.6375361351966, -18345.5637091400895, 1.12497562499999994e-7, -0.000866666666666666627,
    -22515.0100066710511, -17511.6744496337087, -32521.6811207465435, 1.49996531249999983e-7, -0.00119999999999999989,
     8060.92950856128300,  277.963086502110855, -14176.1174116077000, 3.74992187499999966e-8, -0.000333333333333333322,
    -28352.2348232157237, -26684.4563042034206, -18345.5637091394856, 1.49995687499999993e-7, -0.00100000000000000002,
    -10006.6711140762363, -35023.3488992664861, -46697.7985323555185, 2.24993562500000037e-7, -0.00179999999999999995,
     18345.5637091391727, -8338.89259506337476, -28352.2348232147888, 7.49981249999999978e-8, -0.000799999999999999930
  };

  for(int iNode=0; iNode<int(tGold.size()); iNode++){
    if(tGold[iNode] == 0.0){
      TEST_ASSERT(fabs(tResidual_Host[iNode]) < 1e-11);
    } else {
      TEST_FLOATING_EQUALITY(tResidual_Host[iNode], tGold[iNode], 1e-11);
    }
  }


  // compute and test gradient wrt state. (i.e., jacobian)
  //
  auto jacobian = vectorFunction.gradient_u(tState, tProjPGrad, tControl, timeStep);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
1.853087243347414779663085938e10, 0, 0,
-0.02083333333333333217685101602, 0, 0,
1.853087243347415161132812500e10, 0,
-0.02083333333333333217685101602, 0, 0, 0,
1.853087243347414779663085938e10,
-0.02083333333333333217685101602, 0,
-0.02083333333333333217685101602, -0.02083333333333333217685101602,
-0.02083333333333333217685101602,
-3.124999999999999996310341733e-16,
-2.343749999999999794679213586e-7, 0, 0, 0, 0,
499.9999999999999431565811392, -5.55926173004224968e9, 0,
2.77963086502112484e9, 0, 0, 0, -5.55926173004224968e9,
2.77963086502112484e9, 0, 0, -1.85308724334733057e9,
-1.85308724334733057e9, -7.41234897338964844e9,
-0.0208333333333333322, 0, -0.0104166666666666661,
-0.0104166666666666661, 0.0208333333333333322,
-1.04166666666666662e-16, -7.81249999999999932e-8, 0, 0, 0, 0,
-166.666666666666657, -5.55926173004224968e9,
2.77963086502112484e9, 0, 0, 0, -1.85308724334733057e9,
-7.41234897338964844e9, -1.85308724334733057e9,
-0.0208333333333333322, 0, 0, 2.77963086502112484e9,
-5.55926173004224968e9, 0, 0, -0.0104166666666666661,
0.0208333333333333322, -0.0104166666666666661,
-1.04166666666666662e-16, -7.81249999999999932e-8, 0, 0, 0, 0,
-166.666666666666657, 0, 2.77963086502112484e9,
2.77963086502112484e9, 0, 0, -1.85308724334733057e9, 0,
-9.26543621673794270e8, -0.0104166666666666661, 0,
-1.85308724334733057e9, -9.26543621673794270e8, 0,
-0.0104166666666666661, 0, -0.0208333333333333322,
0.0104166666666666661, 0.0104166666666666661,
-1.04166666666666662e-16, -7.81249999999999932e-8, 0, 0, 0, 0,
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    if(fabs(gold_jac_entries[i]) < 1e-12){
      TEST_ASSERT(fabs(jac_entriesHost(i)) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-13);
    }
  }
}

TEUCHOS_UNIT_TEST( PlatoMathFunctors, RowSumSolve )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;

  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", meshWidth);

  // create mesh based solution from host data
  //
  int tNumNodes = tMesh->NumNodes();
  Plato::ScalarVector tProjectState ("state",     tNumNodes);
  Plato::ScalarVector tProjPGrad    ("ProjPGrad", tNumNodes*spaceDim);
  Plato::ScalarVector tControl      ("Control",   tNumNodes);
  Plato::blas1::fill( 1.0, tControl );
  Plato::blas1::fill( 0.0, tProjPGrad );
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
     tProjectState(aNodeOrdinal) = 1.0*aNodeOrdinal;
  }, "state");

  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                      \n"
    "  <ParameterList name='Spatial Model'>                                                    \n"
    "    <ParameterList name='Domains'>                                                        \n"
    "      <ParameterList name='Design Volume'>                                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                      \n"
    "        <Parameter name='Material Model' type='string' value='Squeaky Cheese'/>           \n"
    "      </ParameterList>                                                                    \n"
    "    </ParameterList>                                                                      \n"
    "  </ParameterList>                                                                        \n"
    "  <ParameterList name='Material Models'>                                                  \n"
    "    <ParameterList name='Squeaky Cheese'>                                                 \n"
    "      <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>                        \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e6'/>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.499'/>                   \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>                  \n"
    "        <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/>   \n"
    "        <Parameter  name='Thermal Conductivity' type='double' value='1000.0'/>\n"
    "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "      </ParameterList>                                                                    \n"
    "    </ParameterList>                                                                      \n"
    "  </ParameterList>                                                                        \n"
    "</ParameterList>                                                                          \n"
  );

  // create constraint evaluator
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *params);
  Plato::VectorFunctionVMS<::Plato::Projection<spaceDim>>
    tVectorFunction(tSpatialModel, tDataMap, *params, "State Gradient Projection");

  auto tResidual = tVectorFunction.value      (tProjPGrad, tProjectState, tControl);
  auto tJacobian = tVectorFunction.gradient_u (tProjPGrad, tProjectState, tControl);


  { // test residual
    //

    auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
    Kokkos::deep_copy( tResidual_Host, tResidual );

    std::vector<Plato::Scalar> tGold = {
      -0.5625000000000000, -0.1875000000000000,  -0.06249999999999998,
      -0.7500000000000000, -0.2500000000000000,  -0.08333333333333338,
      -0.1875000000000000, -0.06250000000000000, -0.02083333333333334,
      -0.7500000000000000, -0.2500000000000000,  -0.08333333333333329,
      -1.125000000000000,  -0.3750000000000000,  -0.1250000000000001,
      -0.3750000000000000, -0.1250000000000000,  -0.04166666666666669
    };
    int tNumGold = tGold.size();
    for(int i=0; i<tNumGold; i++){
      if(tGold[i] == 0.0){
        TEST_ASSERT(fabs(tResidual_Host(i)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tResidual_Host(i), tGold[i], 2.0e-14);
      }
    }
  }

  { // test jacobian
    //

    auto tJacobian_Host = Kokkos::create_mirror_view( tJacobian->entries() );
    Kokkos::deep_copy( tJacobian_Host, tJacobian->entries() );

    std::vector<Plato::Scalar> tGold = {
0.00781249999999999913, 0, 0, 0, 0.00781249999999999913, 0, 0, 0,
0.00781249999999999913, 0.00260416666666666652, 0, 0, 0,
0.00260416666666666652, 0, 0, 0, 0.00260416666666666652,
0.00260416666666666652, 0, 0, 0, 0.00260416666666666652, 0, 0, 0,
0.00260416666666666652, 0.00260416666666666652, 0, 0, 0,
0.00260416666666666652, 0, 0, 0, 0.00260416666666666652
    };

    int tNumGold = tGold.size();
    for(int i=0; i<tNumGold; i++){
      if(tGold[i] == 0.0){
        TEST_ASSERT(fabs(tJacobian_Host(i)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tJacobian_Host(i), tGold[i], 2.0e-14);
      }
    }
  }


  { // test row sum functor
    //
    Plato::RowSum rowSum(tJacobian);

    Plato::ScalarVector tRowSum("row sum", tResidual.extent(0));

    auto tNumBlockRows = tJacobian->rowMap().size() - 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumBlockRows), LAMBDA_EXPRESSION(int blockRowOrdinal)
    {
      // compute row sum
      rowSum(blockRowOrdinal, tRowSum);

    }, "row sum inverse");

    auto tRowSum_Host = Kokkos::create_mirror_view( tRowSum );
    Kokkos::deep_copy( tRowSum_Host, tRowSum );

    std::vector<Plato::Scalar> tRowSum_gold = {
    0.0312500000000000000, 0.0312500000000000000, 0.0312500000000000000,
    0.0416666666666666713, 0.0416666666666666713, 0.0416666666666666713,
    0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
    0.0416666666666666644, 0.0416666666666666644, 0.0416666666666666644,
    0.0625000000000000000, 0.0625000000000000000, 0.0625000000000000000,
    0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
    0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
    0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
    0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
    0.0416666666666666644, 0.0416666666666666644, 0.0416666666666666644,
    0.0625000000000000000, 0.0625000000000000000, 0.0625000000000000000,
    0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
    0.0625000000000000000, 0.0625000000000000000, 0.0625000000000000000,
    0.124999999999999958,  0.124999999999999958,  0.124999999999999958,
    0.0624999999999999931, 0.0624999999999999931, 0.0624999999999999931,
    0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
    0.0624999999999999931, 0.0624999999999999931, 0.0624999999999999931,
    0.0416666666666666644, 0.0416666666666666644, 0.0416666666666666644,
    0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
    0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
    0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
    0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
    0.0624999999999999931, 0.0624999999999999931, 0.0624999999999999931,
    0.0416666666666666644, 0.0416666666666666644, 0.0416666666666666644,
    0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
    0.0416666666666666644, 0.0416666666666666644, 0.0416666666666666644,
    0.0312500000000000000, 0.0312500000000000000, 0.0312500000000000000
    };

    int tNumGold = tRowSum_gold.size();
    for(int i=0; i<tNumGold; i++){
      TEST_FLOATING_EQUALITY(tRowSum_Host(i), tRowSum_gold[i], 2.0e-14);
    }
  }


  { // test row summed solve
    //
    Plato::blas1::scale(-1.0, tResidual);
    Plato::Solve::RowSummed<spaceDim>(tJacobian, tProjPGrad, tResidual);

    auto tProjPGrad_Host = Kokkos::create_mirror_view( tProjPGrad );
    Kokkos::deep_copy( tProjPGrad_Host, tProjPGrad );

    std::vector<Plato::Scalar> tGold = {
      18.0000000000000000, 5.99999999999999911, 1.99999999999999933,
      17.9999999999999964, 5.99999999999999911, 2.00000000000000089,
      18.0000000000000000, 6.00000000000000000, 2.00000000000000000,
      18.0000000000000000, 6.00000000000000000, 1.99999999999999889,
      17.9999999999999964, 6.00000000000000000, 2.00000000000000133,
      18.0000000000000000, 6.00000000000000000, 2.00000000000000089,
      18.0000000000000000, 5.99999999999999822, 1.99999999999999956,
      17.9999999999999964, 6.00000000000000000, 2.00000000000000133
    };

    int tNumGold = tGold.size();
    for(int i=0; i<tNumGold; i++){
      if(tGold[i] == 0.0){
        TEST_ASSERT(fabs(tProjPGrad_Host(i)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tProjPGrad_Host(i), tGold[i], 2.0e-14);
      }
    }
  }
}
