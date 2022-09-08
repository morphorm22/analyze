#include "util/PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>


#ifdef HAVE_AMGX
#include "alg/AmgXSparseLinearProblem.hpp"
#endif

#include <sstream>
#include <iostream>
#include <fstream>
#include <type_traits>

#include <Sacado.hpp>

#include "Tet4.hpp"
#include "Thermal.hpp"
#include "Solutions.hpp"
#include "ScalarGrad.hpp"
#include "ThermalFlux.hpp"
#include "WorksetBase.hpp"
#include "ComputedField.hpp"
#include "ScalarProduct.hpp"
#include "ProjectToNode.hpp"
#include "ThermalContent.hpp"
#include "GradientMatrix.hpp"
#include "alg/ParallelComm.hpp"
#include "ThermalMassMaterial.hpp"
#include "InterpolateFromNodal.hpp"
#include "alg/CrsLinearProblem.hpp"
#include "GeneralFluxDivergence.hpp"
#include "parabolic/VectorFunction.hpp"
#include "parabolic/PhysicsScalarFunction.hpp"

#include <fenv.h>

TEUCHOS_UNIT_TEST( HeatEquationTests, 3D )
{ 
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <ParameterList name='Spatial Model'>                                      \n"
    "    <ParameterList name='Domains'>                                          \n"
    "      <ParameterList name='Design Volume'>                                  \n"
    "        <Parameter name='Element Block' type='string' value='body'/>        \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Material Models'>                                    \n"
    "    <ParameterList name='Unobtainium'>                                      \n"
    "      <ParameterList name='Thermal Mass'>                                   \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>          \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e6'/>       \n"
    "      </ParameterList>                                                      \n"
    "      <ParameterList name='Thermal Conduction'>                             \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0e6'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  using ElementType = typename Plato::ThermalElement<Plato::Tet4>;

  int numCells = tMesh->NumElements();
  constexpr int spaceDim     = ElementType::mNumSpatialDims;
  constexpr int nodesPerCell = ElementType::mNumNodesPerCell;
  constexpr int dofsPerCell  = ElementType::mNumDofsPerCell;

  // create mesh based temperature from host data
  //
  std::vector<Plato::Scalar> T_host( tMesh->NumNodes() );
  Plato::Scalar Tval = 0.0, dval = 1.0;
  for( auto& val : T_host ) val = (Tval += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    T_host_view(T_host.data(),T_host.size());
  auto T = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), T_host_view);


  Plato::WorksetBase<ElementType> worksetBase(tMesh);

  Plato::ScalarArray3DT<Plato::Scalar>
    gradient("gradient",numCells,nodesPerCell,spaceDim);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    tCellGrad("temperature gradient", numCells, spaceDim);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    tCellFlux("thermal flux", numCells, spaceDim);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    result("result", numCells, dofsPerCell);

  Plato::ScalarVectorT<Plato::Scalar> tCellTemperature("cell temperature at step k", numCells);

  Plato::ScalarVectorT<Plato::Scalar> tCellThermalContent("cell heat content at step k", numCells);

  Plato::ScalarMultiVectorT<Plato::Scalar> 
    massResult("mass", numCells, dofsPerCell);

  Plato::ScalarArray3DT<Plato::Scalar>
    configWS("config workset",numCells, nodesPerCell, spaceDim);
  worksetBase.worksetConfig(configWS);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    stateWS("state workset",numCells, dofsPerCell);
  worksetBase.worksetState(T, stateWS);

  Plato::ComputeGradientMatrix<ElementType> computeGradient;

  Plato::ScalarGrad<ElementType> scalarGrad;

  Plato::InterpolateFromNodal<ElementType, ElementType::mNumDofsPerNode> tInterpolateFromNodal;

  Plato::ThermalMassModelFactory<spaceDim> mmmfactory(*tParamList);
  auto thermalMassMaterialModel = mmmfactory.create("Unobtainium");

  Plato::ThermalContent<spaceDim> computeThermalContent(thermalMassMaterialModel);
  Plato::ProjectToNode<ElementType> projectThermalContent;

  Plato::ThermalConductionModelFactory<spaceDim> mmfactory(*tParamList);
  auto tMaterialModel = mmfactory.create("Unobtainium");

  Plato::ThermalFlux<ElementType>           thermalFlux(tMaterialModel);
  Plato::GeneralFluxDivergence<ElementType> fluxDivergence;

  Plato::ScalarVectorT<Plato::Scalar> cellVolume("cell volume",numCells);

  auto tCubPoints = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto numPoints = tCubWeights.size();

  Kokkos::parallel_for("flux divergence", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {numCells, numPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    Plato::Scalar tVolume(0.0);

    Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, Plato::Scalar> tGradient;

    Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tGrad(0.0);
    Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tFlux(0.0);

    auto tCubPoint = tCubPoints(iGpOrdinal);
    auto tBasisValues = ElementType::basisValues(tCubPoint);

    computeGradient(iCellOrdinal, tCubPoint, configWS, tGradient, tVolume);
    tVolume *= tCubWeights(iGpOrdinal);

    cellVolume(iCellOrdinal) = tVolume;

    for(Plato::OrdinalType iNode=0; iNode<nodesPerCell; iNode++)
    {
      for(Plato::OrdinalType iDim=0; iDim<spaceDim; iDim++)
      {
        gradient(iCellOrdinal, iNode, iDim) = tGradient(iNode, iDim);
      }
    }

    scalarGrad(iCellOrdinal, tGrad, stateWS, tGradient);
    for(Plato::OrdinalType iDim=0; iDim<spaceDim; iDim++)
    {
      tCellGrad(iCellOrdinal, iDim) = tGrad(iDim);
    }

    Plato::Scalar tTemperature(0.0);
    tInterpolateFromNodal(iCellOrdinal, tBasisValues, stateWS, tTemperature);
    tCellTemperature(iCellOrdinal) = tTemperature;
    thermalFlux(tFlux, tGrad, tTemperature);
    for(Plato::OrdinalType iDim=0; iDim<spaceDim; iDim++)
    {
      tCellFlux(iCellOrdinal, iDim) = tFlux(iDim);
    }

    fluxDivergence(iCellOrdinal, result, tFlux, tGradient, tVolume);

    Plato::Scalar tThermalContent(0.0);
    computeThermalContent(tThermalContent, tTemperature, tTemperature);
    tCellThermalContent(iCellOrdinal) = tThermalContent;

    projectThermalContent(iCellOrdinal, tVolume, tBasisValues, tThermalContent, massResult);
  });
  
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
  auto tTemperature_Host = Kokkos::create_mirror_view( tCellTemperature );
  Kokkos::deep_copy( tTemperature_Host, tCellTemperature );

  std::vector<Plato::Scalar> tTemperature_gold = { 
    8.000000000000000, 6.000000000000000, 5.500000000000000, 7.000000000000000,
    9.000000000000000, 9.500000000000000, 9.000000000000000, 7.000000000000000,
    6.500000000000000, 8.000000000000000, 10.00000000000000, 10.50000000000000,
    11.00000000000000, 9.000000000000000, 8.500000000000000, 10.00000000000000
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
  auto thermalContent_Host = Kokkos::create_mirror_view( tCellThermalContent );
  Kokkos::deep_copy( thermalContent_Host, tCellThermalContent );

  std::vector<Plato::Scalar> thermalContent_gold = { 
  2.400000000000000e6, 1.800000000000000e6, 1.650000000000000e6, 2.100000000000000e6,
  2.700000000000000e6, 2.850000000000000e6, 2.700000000000000e6, 2.100000000000000e6,
  1.950000000000000e6, 2.400000000000000e6, 3.000000000000000e6, 3.150000000000000e6,
  3.300000000000000e6, 2.700000000000000e6, 2.550000000000000e6, 3.000000000000000e6
  };

  numGoldCells=thermalContent_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
      if(thermalContent_gold[iCell] == 0.0){
        TEST_ASSERT(fabs(thermalContent_Host(iCell)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(thermalContent_Host(iCell), thermalContent_gold[iCell], 1e-13);
      }
  }

  // test gradient operator
  //
  auto gradient_Host = Kokkos::create_mirror_view( gradient );
  Kokkos::deep_copy( gradient_Host, gradient );

  std::vector<std::vector<std::vector<Plato::Scalar>>> gradient_gold = { 
    {{ 0.0,-2.0, 0.0}, { 2.0, 0.0,-2.0}, {-2.0, 2.0, 0.0}, {0.0, 0.0, 2.0}},
    {{ 0.0,-2.0, 0.0}, { 0.0, 2.0,-2.0}, {-2.0, 0.0, 2.0}, {2.0, 0.0, 0.0}},
    {{ 0.0, 0.0,-2.0}, {-2.0, 2.0, 0.0}, { 0.0,-2.0, 2.0}, {2.0, 0.0, 0.0}},
    {{ 0.0, 0.0,-2.0}, {-2.0, 0.0, 2.0}, { 2.0,-2.0, 0.0}, {0.0, 2.0, 0.0}},
    {{-2.0, 0.0, 0.0}, { 0.0,-2.0, 2.0}, { 2.0, 0.0,-2.0}, {0.0, 2.0, 0.0}},
    {{-2.0, 0.0, 0.0}, { 2.0,-2.0, 0.0}, { 0.0, 2.0,-2.0}, {0.0, 0.0, 2.0}}
  };

  numGoldCells=gradient_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    for(int iNode=0; iNode<nodesPerCell; iNode++){
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
  auto tgrad_Host = Kokkos::create_mirror_view( tCellGrad );
  Kokkos::deep_copy( tgrad_Host, tCellGrad );

  std::vector<std::vector<Plato::Scalar>> tgrad_gold = { 
    { 18.000, 6.000, 2.000 },
    { 18.000, 6.000, 2.000 },
    { 18.000, 6.000, 2.000 },
    { 18.000, 6.000, 2.000 },
    { 18.000, 6.000, 2.000 },
    { 18.000, 6.000, 2.000 }
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
  auto tflux_Host = Kokkos::create_mirror_view( tCellFlux );
  Kokkos::deep_copy( tflux_Host, tCellFlux );

  std::vector<std::vector<Plato::Scalar>> tflux_gold = { 
   {-18.0e6, -6.0e6, -2.0e6 },
   {-18.0e6, -6.0e6, -2.0e6 },
   {-18.0e6, -6.0e6, -2.0e6 },
   {-18.0e6, -6.0e6, -2.0e6 },
   {-18.0e6, -6.0e6, -2.0e6 },
   {-18.0e6, -6.0e6, -2.0e6 }
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

  // test flux divergence
  //
  auto result_Host = Kokkos::create_mirror_view( result );
  Kokkos::deep_copy( result_Host, result );

  std::vector<std::vector<Plato::Scalar>> result_gold = { 
   { 250000.0000000000, -666666.6666666666, 500000.0000000000, -83333.33333333333 },
   { 250000.0000000000, -166666.6666666667, 666666.6666666666, -750000.0000000000 },
   { 83333.33333333333,  500000.0000000000, 166666.6666666667, -750000.0000000000 }
  };

  for(int iCell=0; iCell<int(result_gold.size()); iCell++){
    for(int iDof=0; iDof<dofsPerCell; iDof++){
      if(result_gold[iCell][iDof] == 0.0){
        TEST_ASSERT(fabs(result_Host(iCell,iDof)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(result_Host(iCell,iDof), result_gold[iCell][iDof], 1e-13);
      }
    }
  }

  // test projected thermal content
  //
  auto mass_result_Host = Kokkos::create_mirror_view( massResult );
  Kokkos::deep_copy( mass_result_Host, massResult );

  std::vector<std::vector<Plato::Scalar>> mass_result_gold = { 
   { 12500.00000000000, 12500.00000000000, 12500.00000000000, 12500.00000000000},
   {  9375.00000000000,  9375.00000000000,  9375.00000000000,  9375.00000000000},
   {  8593.75000000000,  8593.75000000000,  8593.75000000000,  8593.75000000000}
  };

  for(int iCell=0; iCell<int(mass_result_gold.size()); iCell++){
    for(int iNode=0; iNode<nodesPerCell; iNode++){
      if(mass_result_gold[iCell][iNode] == 0.0){
        TEST_ASSERT(fabs(mass_result_Host(iCell,iNode)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(mass_result_Host(iCell,iNode), mass_result_gold[iCell][iNode], 1e-13);
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
TEUCHOS_UNIT_TEST( HeatEquationTests, HeatEquationResidual3D )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <ParameterList name='Spatial Model'>                                      \n"
    "    <ParameterList name='Domains'>                                          \n"
    "      <ParameterList name='Design Volume'>                                  \n"
    "        <Parameter name='Element Block' type='string' value='body'/>        \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Parabolic'/>        \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                \n"
    "  <ParameterList name='Parabolic'>                                          \n"
    "    <ParameterList name='Penalty Function'>                                 \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Material Models'>                                    \n"
    "    <ParameterList name='Unobtainium'>                                      \n"
    "      <ParameterList name='Thermal Mass'>                                   \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>          \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e3'/>       \n"
    "      </ParameterList>                                                      \n"
    "      <ParameterList name='Thermal Conduction'>                             \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0e6'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Time Integration'>                                   \n"
    "    <Parameter name='Number Time Steps' type='int' value='3'/>              \n"
    "    <Parameter name='Time Step' type='double' value='0.5'/>                 \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( tMesh->NumNodes(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);


  // create mesh based temperature from host data
  //
  std::vector<Plato::Scalar> T_host( tMesh->NumNodes() );
  Plato::Scalar Tval = 0.0, dval = 1.0000;
  for( auto& val : T_host ) val = (Tval += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    T_host_view(T_host.data(),T_host.size());
  auto T = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), T_host_view);

  std::vector<Plato::Scalar> Tdot_host( tMesh->NumNodes() );
  Tval = 0.0; dval = 0.5000;
  for( auto& val : Tdot_host ) val = (Tval += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    Tdot_host_view(Tdot_host.data(),Tdot_host.size());
  auto Tdot = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), Tdot_host_view);


  Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

  Plato::DataMap tDataMap;
  Plato::Parabolic::VectorFunction<::Plato::Thermal<Plato::Tet4>>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));


  // compute and test value
  //
  auto timeStep = tParamList->sublist("Time Integration").get<Plato::Scalar>("Time Step");
  auto residual = vectorFunction.value(T, Tdot, z, timeStep);

  auto residual_Host = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy( residual_Host, residual );

  std::vector<Plato::Scalar> residual_gold = { 
    -2.166631510416667e6, -2.999950390625000e6, -8.333220052083333e5,
    -2.499939843750000e6, -4.499910156250000e6, -1.999973437500000e6,
    -3.333177083333334e5, -1.499969140625000e6, -1.166651432291667e6,
    -9.999082031250002e5, -1.499865624999999e6, -4.999605468749995e5,
    -4.998507812499995e5,  2.624999999972060e2,  5.001132812500009e5,
     5.000480468750002e5,  1.500128124999999e6,  1.000083203125000e6,
     1.166695182291666e6,  1.500056640625000e6,  3.333614583333335e5,
     2.000060937500000e6,  4.500172656250000e6,  2.500114843750000e6,
     8.333657552083330e5,  3.000125390625000e6,  2.166762760416667e6
  };

  for(int iNode=0; iNode<int(residual_gold.size()); iNode++){
    if(residual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(residual_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(residual_Host[iNode], residual_gold[iNode], 1e-10);
    }
  }


  // compute and test gradient wrt state, T. (i.e., jacobian)
  //
  auto jacobian = vectorFunction.gradient_u(T, Tdot, z, timeStep);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
   499999.999999999942, -166666.666666666657, -166666.666666666657, 0,
  -166666.666666666657, 0, 0, 0, -166666.666666666657,
   833333.333333333372, -166666.666666666657, -250000.000000000000, 0,
  -250000.000000000000, 0, 0, 0, -166666.666666666657,
   333333.333333333314, -83333.3333333333285, -83333.3333333333285, 0
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }


  // compute and test gradient wrt previous state, T. (i.e., jacobian)
  //
  auto jacobian_v = vectorFunction.gradient_v(T, Tdot, z, timeStep);

  auto jac_v_entries = jacobian_v->entries();
  auto jac_v_entriesHost = Kokkos::create_mirror_view( jac_v_entries );
  Kokkos::deep_copy(jac_v_entriesHost, jac_v_entries);

  std::vector<Plato::Scalar> gold_jac_v_entries = {
   2.34375000000000000, 0.781250000000000000, 0.781250000000000000,
   0.781250000000000000, 0.781250000000000000, 0.781250000000000000,
   0.781250000000000000, 2.34375000000000000, 0.781250000000000000,
   3.12500000000000000, 0.781250000000000000, 1.17187500000000000,
   0.781250000000000000, 1.17187500000000000, 0.781250000000000000,
   1.56250000000000000, 2.34375000000000000, 0.781250000000000000,
   0.781250000000000000, 0.390625000000000000, 0.390625000000000000,
   0.781250000000000000
  };

  int jac_v_entriesSize = gold_jac_v_entries.size();
  for(int i=0; i<jac_v_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_v_entriesHost(i), gold_jac_v_entries[i], 1.0e-15);
  }


  // compute and test objective gradient wrt control, z
  //
  auto gradient_z = vectorFunction.gradient_z(T, Tdot, z, timeStep);
  
  auto grad_entries = gradient_z->entries();
  auto grad_entriesHost = Kokkos::create_mirror_view( grad_entries );
  Kokkos::deep_copy(grad_entriesHost, grad_entries);

  std::vector<Plato::Scalar> gold_grad_entries = {
  -541657.877604166628, -208330.891927083314, -83330.5989583333139,
  -291664.420572916628,  291670.279947916628,  83336.4583333333721,
   208336.751302083372,  541675.455729166511, -41664.2252604166642,
  -749987.597656250000, -208330.501302083314, -208329.134114583314,
  -291664.029947916628,  416672.037760416628,  83336.8489583333721,
   458339.583333333198,  541676.627604166744, -41663.8346354166715,
  -208330.501302083314, -124998.730468750000,  125001.562500000000,
   250002.832031250000
  };

  int grad_entriesSize = gold_grad_entries.size();
  for(int i=0; i<grad_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_entriesHost(i), gold_grad_entries[i], 2.0e-14);
  }


  // compute and test objective gradient wrt node position, x
  //
  auto gradient_x = vectorFunction.gradient_x(T, Tdot, z, timeStep);
  
  auto grad_x_entries = gradient_x->entries();
  auto grad_x_entriesHost = Kokkos::create_mirror_view( grad_x_entries );
  Kokkos::deep_copy(grad_x_entriesHost, grad_x_entries);

  std::vector<Plato::Scalar> gold_grad_x_entries = {
  -9.00002890625000000e6, -3.00002187499999953e6,
  -1.00001953124999965e6,  2.83333333333333302e6,
   833333.333333333023,    2.33331380208333302e6,
   2.49999999999999953e6,  2.66664479166666651e6,
  -166666.666666666861,   -666666.666666666628,
   1.49999062499999977e6,  1.49999140624999977e6
  };

  int grad_x_entriesSize = gold_grad_x_entries.size();
  for(int i=0; i<grad_x_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_x_entriesHost(i), gold_grad_x_entries[i], 1.0e-10);
  }

}


/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalthermalEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HeatEquationTests, InternalThermalEnergy3D )
{ 
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <ParameterList name='Spatial Model'>                                      \n"
    "    <ParameterList name='Domains'>                                          \n"
    "      <ParameterList name='Design Volume'>                                  \n"
    "        <Parameter name='Element Block' type='string' value='body'/>        \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Parabolic'/>        \n"
    "  <Parameter name='Objective' type='string' value='My Internal Thermal Energy'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                 \n"
    "  <ParameterList name='Criteria'>                                          \n"
    "    <ParameterList name='Internal Energy'>                                 \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>       \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Thermal Energy'/>  \n"
    "      <ParameterList name='Penalty Function'>                              \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>             \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>        \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                \n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Material Models'>                                    \n"
    "    <ParameterList name='Unobtainium'>                                      \n"
    "      <ParameterList name='Thermal Mass'>                                   \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>          \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e3'/>       \n"
    "      </ParameterList>                                                      \n"
    "      <ParameterList name='Thermal Conduction'>                             \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0e6'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Time Integration'>                                   \n"
    "    <Parameter name='Number Time Steps' type='int' value='3'/>              \n"
    "    <Parameter name='Time Step' type='double' value='0.5'/>                 \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  // create mesh based temperature from host data
  //
  int tNumSteps = 3;
  int tNumNodes = tMesh->NumNodes();
  Plato::ScalarMultiVector T("temperature history", tNumSteps, tNumNodes);
  Plato::ScalarMultiVector Tdot("temperature rate history", tNumSteps, tNumNodes);
  Plato::ScalarVector z("density", tNumNodes);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     for( int i=0; i<tNumSteps; i++){
       T(i, aNodeOrdinal) = (i+1)*aNodeOrdinal;
       Tdot(i, aNodeOrdinal) = 0.0;
     }
  }, "temperature history");


  Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

  Plato::DataMap dataMap;
  std::string tMyFunction("Internal Energy");
  Plato::Parabolic::PhysicsScalarFunction<::Plato::Thermal<Plato::Tet4>>
    scalarFunction(tSpatialModel, dataMap, *tParamList, tMyFunction);

  auto timeStep = tParamList->sublist("Time Integration").get<Plato::Scalar>("Time Step");
  int timeIncIndex = 1;

  // compute and test objective value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", T);
  tSolution.set("StateDot", Tdot);
  auto value = scalarFunction.value(tSolution, z, timeStep);

  Plato::Scalar value_gold = 4.73200000000000095e9;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);

  // compute and test objective gradient wrt state, u
  //
  auto grad_u = scalarFunction.gradient_u(tSolution, z, timeIncIndex, timeStep);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
  -8.666666666666668e6, -1.200000000000000e7,
  -3.333333333333333e6, -1.000000000000000e7,
  -1.800000000000000e7, -7.999999999999999e6,
  -1.333333333333333e6, -6.000000000000001e6,
  -4.666666666666665e6, -3.999999999999999e6,
  -6.000000000000002e6, -1.999999999999998e6
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(grad_u_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
    }
  }


  // compute and test objective gradient wrt control, z
  //
  auto grad_z = scalarFunction.gradient_z(tSolution, z, timeStep);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
   1.478750000000000e8, 1.971666666666667e8,
   4.929166666666666e7, 1.971666666666667e8,
   2.957500000000000e8, 9.858333333333334e7,
   4.929166666666667e7, 9.858333333333333e7,
   4.929166666666666e7, 1.971666666666667e8,
   2.957500000000000e8, 9.858333333333334e7,
   2.957500000000000e8, 5.914999999999999e8,
   2.957500000000000e8, 9.858333333333333e7,
   2.957500000000000e8, 1.971666666666667e8,
   4.929166666666667e7, 9.858333333333333e7,
   4.929166666666666e7, 9.858333333333334e7,
   2.957499999999999e8, 1.971666666666667e8,
   4.929166666666666e7, 1.971666666666666e8,
   1.478750000000000e8
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test objective gradient wrt node position, x
  //
  auto grad_x = scalarFunction.gradient_x(tSolution, z, timeStep);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
   6.196666666666666e8, -5.633333333333331e7,
  -2.816666666666667e8,  8.125000000000000e8,
  -1.235000000000000e8,  1.560000000000000e8,
   1.928333333333333e8, -6.716666666666669e7,
   4.376666666666666e8,  5.785000000000000e8,
   3.900000000000000e8, -4.615000000000000e8,
   9.230000000000000e8,  7.020000000000001e8,
   2.340000000000000e8,  3.444999999999999e8,
   3.119999999999999e8,  6.955000000000000e8,
  -4.116666666666667e7,  4.463333333333334e8,
  -1.798333333333333e8,
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
  }
}


/******************************************************************************/
/*! 
  \brief Create a 'ComputedField' object for a uniform scalar field
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HeatEquationTests, ComputedField_UniformScalar )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  // compute fields
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                \n"
    "  <ParameterList name='Computed Fields'>                            \n"
    "    <ParameterList name='Uniform Initial Temperature'>              \n"
    "      <Parameter name='Function' type='string' value='100.0'/>      \n"
    "    </ParameterList>                                                \n"
    "    <ParameterList name='Linear X Initial Temperature'>             \n"
    "      <Parameter name='Function' type='string' value='1.0*x'/>      \n"
    "    </ParameterList>                                                \n"
    "    <ParameterList name='Linear Y Initial Temperature'>             \n"
    "      <Parameter name='Function' type='string' value='1.0*y'/>      \n"
    "    </ParameterList>                                                \n"
    "    <ParameterList name='Linear Z Initial Temperature'>             \n"
    "      <Parameter name='Function' type='string' value='1.0*z'/>      \n"
    "    </ParameterList>                                                \n"
    "    <ParameterList name='Bilinear XY Initial Temperature'>          \n"
    "      <Parameter name='Function' type='string' value='1.0*x*y'/>    \n"
    "    </ParameterList>                                                \n"
    "  </ParameterList>                                                  \n"
    "</ParameterList>                                                    \n"
  );

  auto tComputedFields = Plato::ComputedFields<spaceDim>(tMesh, tParamList->sublist("Computed Fields"));

  int tNumNodes = tMesh->NumNodes();
  Plato::ScalarVector T("temperature", tNumNodes);

  tComputedFields.get("Uniform Initial Temperature", T);

  // pull temperature to host
  //
  auto T_Host = Kokkos::create_mirror_view( T );
  Kokkos::deep_copy( T_Host, T );

  for(int iNode=0; iNode<tNumNodes; iNode++){
    TEST_FLOATING_EQUALITY(T_Host[iNode], 100.0, 1e-15);
  }


  Plato::ScalarVector xcoords("x", tNumNodes);
  Plato::ScalarVector ycoords("y", tNumNodes);
  Plato::ScalarVector zcoords("z", tNumNodes);
  auto coords = tMesh->Coordinates();
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(int nodeOrdinal)
  {
    xcoords(nodeOrdinal) = coords[nodeOrdinal*spaceDim+0];
    ycoords(nodeOrdinal) = coords[nodeOrdinal*spaceDim+1];
    zcoords(nodeOrdinal) = coords[nodeOrdinal*spaceDim+2];
  }, "get coords");

  auto xCoords_Host = Kokkos::create_mirror_view( xcoords );
  auto yCoords_Host = Kokkos::create_mirror_view( ycoords );
  auto zCoords_Host = Kokkos::create_mirror_view( zcoords );
  Kokkos::deep_copy( xCoords_Host, xcoords );
  Kokkos::deep_copy( yCoords_Host, ycoords );
  Kokkos::deep_copy( zCoords_Host, zcoords );

  tComputedFields.get("Linear X Initial Temperature", T);
  Kokkos::deep_copy( T_Host, T );
  for(int iNode=0; iNode<tNumNodes; iNode++){
    TEST_FLOATING_EQUALITY(T_Host[iNode], 1.0*xCoords_Host[iNode], 1e-15);
  }

  tComputedFields.get("Linear Y Initial Temperature", T);
  Kokkos::deep_copy( T_Host, T );
  for(int iNode=0; iNode<tNumNodes; iNode++){
    TEST_FLOATING_EQUALITY(T_Host[iNode], 1.0*yCoords_Host[iNode], 1e-15);
  }

  tComputedFields.get("Linear Z Initial Temperature", T);
  Kokkos::deep_copy( T_Host, T );
  for(int iNode=0; iNode<tNumNodes; iNode++){
    TEST_FLOATING_EQUALITY(T_Host[iNode], 1.0*zCoords_Host[iNode], 1e-15);
  }

  tComputedFields.get("Bilinear XY Initial Temperature", T);
  Kokkos::deep_copy( T_Host, T );
  for(int iNode=0; iNode<tNumNodes; iNode++){
    TEST_FLOATING_EQUALITY(T_Host[iNode], 1.0*xCoords_Host[iNode]*yCoords_Host[iNode], 1e-15);
  }
}
