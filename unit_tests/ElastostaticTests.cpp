#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "PlatoStaticsTypes.hpp"
#include "ImplicitFunctors.hpp"

#ifdef HAVE_AMGX
#include "alg/AmgXSparseLinearProblem.hpp"
#endif

#include <sstream>
#include <iostream>
#include <fstream>
#include <type_traits>
#include <Sacado.hpp>

#include "alg/CrsLinearProblem.hpp"
#include "alg/ParallelComm.hpp"

#include "Tri3.hpp"
#include "Tet4.hpp"
#include "Solutions.hpp"
#include "ScalarProduct.hpp"
#include "FadTypes.hpp"
#include "WorksetBase.hpp"
#include "GradientMatrix.hpp"
#include "SmallStrain.hpp"
#include "LinearStress.hpp"
#include "GeneralStressDivergence.hpp"
#include "elliptic/VectorFunction.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
#include "geometric/GeometryScalarFunction.hpp"
#include "ApplyConstraints.hpp"
#include "elliptic/Problem.hpp"
#include "Mechanics.hpp"

#include <fenv.h>

using ordType = typename Plato::ScalarMultiVector::size_type;


TEUCHOS_UNIT_TEST( ElastostaticTests, 3D )
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
    "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>        \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>      \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", meshWidth);

  using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

  Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

  auto tOnlyDomain = tSpatialModel.Domains.front();

  int tNumCells      = tMesh->NumElements();
  auto tCubPoints    = ElementType::getCubPoints();
  auto tCubWeights   = ElementType::getCubWeights();
  auto tNumPoints    = tCubWeights.size();
  constexpr int tSpatialDims   = ElementType::mNumSpatialDims;
  constexpr int tNodesPerCell  = ElementType::mNumNodesPerCell;
  constexpr int tNumVoigtTerms = ElementType::mNumVoigtTerms;
  constexpr int tDofsPerCell   = ElementType::mNumDofsPerCell;

  // create mesh based displacement from host data
  //
  std::vector<Plato::Scalar> u_host( tSpatialDims*tMesh->NumNodes() );
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for( auto& val : u_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    u_host_view(u_host.data(),u_host.size());
  auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);

  Plato::ScalarArray3DT<Plato::Scalar> tGradients("gradient", tNumCells, tNodesPerCell, tSpatialDims);

  Plato::ScalarMultiVectorT<Plato::Scalar> tStrains("strain", tNumCells, tNumVoigtTerms);

  Plato::ScalarMultiVectorT<Plato::Scalar> tStresses("stress", tNumCells, tNumVoigtTerms);

  Plato::ScalarMultiVectorT<Plato::Scalar> tResults("result", tNumCells, tDofsPerCell);

  Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpatialDims);
  tWorksetBase.worksetConfig(tConfigWS);

  Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(u, tStateWS);

  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::SmallStrain<ElementType> tVoigtStrain;


  Plato::ElasticModelFactory<tSpatialDims> mmfactory(*tParamList);
  auto tMaterialModel = mmfactory.create(tOnlyDomain.getMaterialName());
  auto tCellStiffness = tMaterialModel->getStiffnessMatrix();

  Plato::LinearStress<Plato::Elliptic::ResidualTypes<ElementType>, ElementType> tVoigtStress(tCellStiffness);
  Plato::GeneralStressDivergence<ElementType>  tStressDivergence;

  Kokkos::parallel_for("gradients", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  LAMBDA_EXPRESSION(const int cellOrdinal, const int gpOrdinal)
  {
    Plato::Scalar tVolume(0.0);

    Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, Plato::Scalar> tGradient;

    Plato::Array<ElementType::mNumVoigtTerms, Plato::Scalar> tStrain(0.0);
    Plato::Array<ElementType::mNumVoigtTerms, Plato::Scalar> tStress(0.0);

    auto tCubPoint = tCubPoints(gpOrdinal);

    tComputeGradient(cellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
    tVolume *= tCubWeights(gpOrdinal);
    for(int iNode=0; iNode<tNodesPerCell; iNode++)
      for(int iDim=0; iDim<tSpatialDims; iDim++)
        tGradients(cellOrdinal, iNode, iDim) = tGradient(iNode, iDim);

    tVoigtStrain(cellOrdinal, tStrain, tStateWS, tGradient);
    for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
      tStrains(cellOrdinal, iVoigt) = tStrain(iVoigt);

    tVoigtStress(tStress, tStrain);
    for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
      tStresses(cellOrdinal, iVoigt) = tStress(iVoigt);

    tStressDivergence(cellOrdinal, tResults, tStress, tGradient, tVolume);
  });


  // test gradient
  //
  auto tGradient_Host = Kokkos::create_mirror_view( tGradients );
  Kokkos::deep_copy( tGradient_Host, tGradients );

  std::vector<std::vector<std::vector<Plato::Scalar>>> tGradient_gold = { 
    {{ 0.0,-2.0, 0.0},{ 2.0, 0.0,-2.0},{-2.0, 2.0, 0.0},{ 0.0, 0.0, 2.0}},
    {{ 0.0,-2.0, 0.0},{ 0.0, 2.0,-2.0},{-2.0, 0.0, 2.0},{ 2.0, 0.0, 0.0}},
    {{ 0.0, 0.0,-2.0},{-2.0, 2.0, 0.0},{ 0.0,-2.0, 2.0},{ 2.0, 0.0, 0.0}},
    {{ 0.0, 0.0,-2.0},{-2.0, 0.0, 2.0},{ 2.0,-2.0, 0.0},{ 0.0, 2.0, 0.0}},
    {{-2.0, 0.0, 0.0},{ 0.0,-2.0, 2.0},{ 2.0, 0.0,-2.0},{ 0.0, 2.0, 0.0}},
    {{-2.0, 0.0, 0.0},{ 2.0,-2.0, 0.0},{ 0.0, 2.0,-2.0},{ 0.0, 0.0, 2.0}}
  };

  int numGoldCells=tGradient_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    for(int iNode=0; iNode<tNodesPerCell; iNode++){
      for(int iDim=0; iDim<tSpatialDims; iDim++){
        if(tGradient_gold[iCell][iNode][iDim] == 0.0){
          TEST_ASSERT(fabs(tGradient_Host(iCell,iNode,iDim)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tGradient_Host(iCell,iNode,iDim), tGradient_gold[iCell][iNode][iDim], 1e-13);
        }
      }
    }
  }

  // test strain
  //
  auto tStrain_Host = Kokkos::create_mirror_view( tStrains );
  Kokkos::deep_copy( tStrain_Host, tStrains );

  std::vector<std::vector<Plato::Scalar>> tStrain_gold = { 
   { 0.0054, 0.0018, 0.0006, 0.0024, 0.0060, 0.0072 },
   { 0.0054, 0.0018, 0.0006, 0.0024, 0.0060, 0.0072 },
   { 0.0054, 0.0018, 0.0006, 0.0024, 0.0060, 0.0072 },
   { 0.0054, 0.0018, 0.0006, 0.0024, 0.0060, 0.0072 },
   { 0.0054, 0.0018, 0.0006, 0.0024, 0.0060, 0.0072 },
   { 0.0054, 0.0018, 0.0006, 0.0024, 0.0060, 0.0072 }
  };

  for(int iCell=0; iCell<int(tStrain_gold.size()); iCell++){
    for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
      if(tStrain_gold[iCell][iVoigt] == 0.0){
        TEST_ASSERT(fabs(tStrain_Host(iCell,iVoigt)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tStrain_Host(iCell,iVoigt), tStrain_gold[iCell][iVoigt], 1e-13);
      }
    }
  }

  // test stress
  //
  auto tStress_Host = Kokkos::create_mirror_view( tStresses );
  Kokkos::deep_copy( tStress_Host, tStresses );

  std::vector<std::vector<Plato::Scalar>> tStress_gold = { 
   {8653.846153846152, 5884.615384615384, 4961.538461538462, 923.0769230769231, 2307.692307692308, 2769.230769230769},
   {8653.846153846152, 5884.615384615384, 4961.538461538462, 923.0769230769231, 2307.692307692308, 2769.230769230769},
   {8653.846153846152, 5884.615384615384, 4961.538461538462, 923.0769230769231, 2307.692307692308, 2769.230769230769},
   {8653.846153846152, 5884.615384615384, 4961.538461538462, 923.0769230769231, 2307.692307692308, 2769.230769230769},
   {8653.846153846152, 5884.615384615384, 4961.538461538462, 923.0769230769231, 2307.692307692308, 2769.230769230769},
   {8653.846153846152, 5884.615384615384, 4961.538461538462, 923.0769230769231, 2307.692307692308, 2769.230769230769}
  };

  for(int iCell=0; iCell<int(tStress_gold.size()); iCell++){
    for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
      if(tStress_gold[iCell][iVoigt] == 0.0){
        TEST_ASSERT(fabs(tStress_Host(iCell,iVoigt)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tStress_Host(iCell,iVoigt), tStress_gold[iCell][iVoigt], 1e-13);
      }
    }
  }

  // test residual
  //
  auto tResult_Host = Kokkos::create_mirror_view( tResults );
  Kokkos::deep_copy( tResult_Host, tResults );

  std::vector<std::vector<Plato::Scalar>> tResult_gold = { 
   {-115.3846153846154, -245.1923076923076, -38.46153846153845, 264.4230769230769,
     76.92307692307693, -110.5769230769230, -245.1923076923076, 129.8076923076923,
    -57.69230769230768,  96.15384615384616,  38.46153846153845, 206.7307692307692 },
   {-115.3846153846154, -245.1923076923076, -38.46153846153846,  19.23076923076923,
     206.7307692307692, -168.2692307692307, -264.4230769230768, -76.92307692307693,
     110.5769230769230,  360.5769230769230,  115.3846153846154,  96.15384615384616 },
  };

  for(int iCell=0; iCell<int(tResult_gold.size()); iCell++){
    for(int iDof=0; iDof<tDofsPerCell; iDof++){
      if(tResult_gold[iCell][iDof] == 0.0){
        TEST_ASSERT(fabs(tResult_Host(iCell,iDof)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tResult_Host(iCell,iDof), tResult_gold[iCell][iDof], 1e-13);
      }
    }
  }
}


/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         elastostatic residual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ElastostaticTests, Residual3D )
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim = Plato::Tet4::mNumSpatialDims;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", meshWidth);

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( tMesh->NumNodes(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);


  // create mesh based displacement from host data
  //
  std::vector<Plato::Scalar> u_host( spaceDim*tMesh->NumNodes() );
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for( auto& val : u_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    u_host_view(u_host.data(),u_host.size());
  auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);



  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                             \n"
    "  <ParameterList name='Spatial Model'>                                           \n"
    "    <ParameterList name='Domains'>                                               \n"
    "      <ParameterList name='Design Volume'>                                       \n"
    "        <Parameter name='Element Block' type='string' value='body'/>             \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                      \n"
    "  <ParameterList name='Elliptic'>                                                \n"
    "    <ParameterList name='Penalty Function'>                                      \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                     \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>                \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                        \n"
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
    "  <ParameterList name='Criteria'>                                                \n"
    "    <ParameterList name='Internal Elastic Energy'>                               \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>             \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "</ParameterList>                                                                 \n"
  );

  // create constraint evaluator
  //
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

  Plato::DataMap tDataMap;
  Plato::Elliptic::VectorFunction<::Plato::Mechanics<Plato::Tet4>>
    esVectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));


  // compute and test constraint value
  //
  auto residual = esVectorFunction.value(u,z);

  auto residual_Host = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy( residual_Host, residual );

  std::vector<Plato::Scalar> residual_gold = { 
-1144.230769230769, -798.0769230769229, -682.6923076923076,
-1427.884615384615, -1081.730769230769, -403.8461538461537,
-283.6538461538461, -283.6538461538461,  278.8461538461538,
-1370.192307692307, -461.5384615384612, -908.6538461538460,
-2163.461538461538, -692.3076923076923, -576.9230769230769,
-793.2692307692304, -230.7692307692308,  331.7307692307693,
-225.9615384615384,  336.5384615384614, -225.9615384615383,
-735.5769230769222,  389.4230769230768, -173.0769230769231,
-509.6153846153846,  52.88461538461540,  52.88461538461543,
-634.6153846153844, -850.9615384615381, -735.5769230769229,
-692.3076923076919, -1471.153846153846, -230.7692307692306,
-57.69230769230739, -620.1923076923076,  504.8076923076926,
-576.9230769230766, -230.7692307692309, -1240.384615384615,
 0.000000000000000,  0.000000000000000,  0.000000000000000,
 576.9230769230768,  230.7692307692313,  1240.384615384615,
 57.69230769230771,  620.1923076923080, -504.8076923076922,
 692.3076923076926,  1471.153846153846,  230.7692307692315,
 634.6153846153854,  850.9615384615385,  735.5769230769231,
 509.6153846153848, -52.88461538461544, -52.88461538461522,
 735.5769230769231, -389.4230769230764,  173.0769230769229,
 225.9615384615386, -336.5384615384613,  225.9615384615385,
 793.2692307692309,  230.7692307692311, -331.7307692307688,
 2163.461538461538,  692.3076923076935,  576.9230769230777,
 1370.192307692308,  461.5384615384621,  908.6538461538464,
 283.6538461538462,  283.6538461538464, -278.8461538461535,
 1427.884615384615,  1081.730769230769,  403.8461538461543,
 1144.230769230769,  798.0769230769230,  682.6923076923077
  };

  for(int iNode=0; iNode<int(residual_gold.size()); iNode++){
    if(residual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(residual_Host[iNode]) < 1e-11);
    } else {
      TEST_FLOATING_EQUALITY(residual_Host[iNode], residual_gold[iNode], 1e-13);
    }
  }


  // compute and test constraint gradient wrt state, u. (i.e., jacobian)
  //
  auto jacobian = esVectorFunction.gradient_u(u,z);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
352564.102564102504, 0, 0, 0, 352564.102564102563, 0, 0, 0,
352564.102564102504, -64102.5641025640944, 0, 32051.2820512820472, 0,
-64102.5641025640944, 32051.2820512820472, 48076.9230769230635,
48076.9230769230635, -224358.974358974287, -64102.5641025640944,
32051.2820512820472, 0, 48076.9230769230635, -224358.974358974287,
48076.9230769230635, 0, 32051.2820512820472, -64102.5641025640944, 0,
32051.2820512820472, 32051.2820512820472, 48076.9230769230635, 0,
-80128.2051282051107, 48076.9230769230635, -80128.2051282051107, 0
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }


  // compute and test gradient wrt control, z
  //
  auto gradient_z = esVectorFunction.gradient_z(u,z);
  
  auto grad_entries = gradient_z->entries();
  auto grad_entriesHost = Kokkos::create_mirror_view( grad_entries );
  Kokkos::deep_copy(grad_entriesHost, grad_entries);

  std::vector<Plato::Scalar> gold_grad_entries = {
-286.057692307692207, -199.519230769230717, -170.673076923076906,
-70.9134615384615188, -70.9134615384615188, 69.7115384615384528,
-56.4903846153845919, 84.1346153846153868, -56.4903846153845919,
-127.403846153846075, 13.2211538461538325, 13.2211538461538360
  };

  int grad_entriesSize = gold_grad_entries.size();
  for(int i=0; i<grad_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_entriesHost(i), gold_grad_entries[i], 2.0e-14);
  }


  // compute and test gradient wrt node position, x
  //
  auto gradient_x = esVectorFunction.gradient_x(u,z);
  
  auto grad_x_entries = gradient_x->entries();
  auto grad_x_entriesHost = Kokkos::create_mirror_view( grad_x_entries );
  Kokkos::deep_copy(grad_x_entriesHost, grad_x_entries);

  std::vector<Plato::Scalar> gold_grad_x_entries = {
-1903.84615384615336, -1903.84615384615336, -1903.84615384615358,
-634.615384615384301, -634.615384615384414, -634.615384615384642,
-211.538461538461490, -211.538461538461462, -211.538461538461661,
-105.769230769230603, 9.61538461538454214, 451.923076923076962,
-163.461538461538652, -48.0769230769230802, -124.999999999999716,
961.538461538461206, 730.769230769230603, 365.384615384615358,
-144.230769230769113, 374.999999999999716, 9.61538461538462741,
942.307692307692150, 596.153846153846189, 634.615384615384301,
-221.153846153846104, -394.230769230769113, -67.3076923076922782,
-942.307692307692150, -307.692307692307395, -230.769230769230688,
548.076923076922867, 317.307692307692150, 278.846153846153811,
663.461538461538112, 259.615384615384528, 221.153846153846132
  };

  int grad_x_entriesSize = gold_grad_x_entries.size();
  for(int i=0; i<grad_x_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_x_entriesHost(i), gold_grad_x_entries[i], 1.0e-13);
  }

}



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalElasticEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, InternalElasticEnergy3D )
{ 
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Objective' type='string' value='My Internal Elastic Energy'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Internal Elastic Energy'>                            \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
    "      <ParameterList name='Penalty Function'>                                 \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=Plato::Tet4::mNumSpatialDims;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", meshWidth);


  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( tMesh->NumNodes(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);


  // create mesh based displacement from host data
  //
  ordType tNumDofs = spaceDim*tMesh->NumNodes();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, tNumDofs);
  auto u = Kokkos::subview(U, 0, Kokkos::ALL());
  auto u_host = Kokkos::create_mirror_view( u );
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for(ordType i=0; i<tNumDofs; i++)
  {
      u_host(i) = (disp += dval);
  }
  Kokkos::deep_copy(u, u_host);


  // create objective
  //
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

  Plato::DataMap dataMap;
  std::string tMyFunction("Internal Elastic Energy");
  Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<Plato::Tet4>>
    eeScalarFunction(tSpatialModel, dataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto value = eeScalarFunction.value(tSolution,z);

  Plato::Scalar value_gold = 48.15;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt state, u
  //
  tSolution.set("State", U);
  auto grad_u = eeScalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
  -1144.230769230769, -798.0769230769229, -682.6923076923076,
  -1427.884615384615, -1081.730769230769, -403.8461538461537,
  -283.6538461538461, -283.6538461538460,  278.8461538461538,
  -1370.192307692307, -461.5384615384613, -908.6538461538458,
  -2163.461538461537, -692.3076923076922, -576.9230769230769,
  -793.2692307692303, -230.7692307692308,  331.7307692307693,
  -225.9615384615384,  336.5384615384615, -225.9615384615383
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(grad_u_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
    }
  }


  // compute and test criterion gradient wrt control, z
  //
  tSolution.set("State", U);
  auto grad_z = eeScalarFunction.gradient_z(tSolution,z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
   1.504687500000000,  2.006250000000001, 0.5015625000000001,
   2.006250000000001,  3.009375000000001, 1.003125000000000,
   0.5015625000000000, 1.003125000000000, 0.5015625000000001,
   2.006250000000001,  3.009375000000003, 1.003125000000001,
   3.009375000000003,  6.018750000000004, 3.009375000000002,
   1.003125000000001,  3.009375000000003, 2.006250000000001,
   0.5015625000000008, 1.003125000000001, 0.5015625000000010,
   1.003125000000002,  3.009375000000004, 2.006250000000002,
   0.5015625000000008, 2.006250000000004, 1.50468750000000
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  tSolution.set("State", U);
  auto grad_x = eeScalarFunction.gradient_x(tSolution, z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
10.16250000000000, 0.7125000000000001, -2.437500000000000,
9.713942307692305, -0.7745192307692330, 1.748076923076921,
-0.4485576923076922, -1.487019230769231, 4.185576923076923,
8.779326923076917, 4.932692307692306, -4.374519230769231,
6.499038461538459, 6.178846153846154, 2.059615384615383,
-2.280288461538460, 1.246153846153845, 6.434134615384615,
-1.383173076923076, 4.220192307692306, -1.937019230769230,
-3.214903846153847, 6.953365384615383, 0.3115384615384613,
-1.831730769230766, 2.733173076923077, 2.248557692307692,
11.99423076923077, -2.020673076923078, -4.686057692307688,
12.92884615384615, -7.727884615384617, 1.436538461538468,
0.9346153846153846, -5.707211538461538, 6.122596153846152,
11.05961538461539, 3.686538461538455, -10.80865384615385,
 0.00000000000000, 0.000000000000000, 0.0000000000000000
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    if(grad_x_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_x_Host[iNode]) < 1e-13);
    } else {
      TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
    }
  }
}

/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         StressPNorm in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, StressPNorm3D )
{ 
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                  \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Globalized Stress'>                                  \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Stress P-Norm'/>  \n"
    "      <Parameter name='Exponent' type='double' value='12.0'/>                 \n"
    "      <ParameterList name='Penalty Function'>                                 \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=Plato::Tet4::mNumSpatialDims;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", meshWidth);


  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( tMesh->NumNodes(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);


  // create mesh based displacement from host data
  //
  ordType tNumDofs = spaceDim*tMesh->NumNodes();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, tNumDofs);
  auto u = Kokkos::subview(U, 0, Kokkos::ALL());
  auto u_host = Kokkos::create_mirror_view( u );
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for(ordType i=0; i<tNumDofs; i++)
  {
      u_host(i) = (disp += dval);
  }
  Kokkos::deep_copy(u, u_host);


  // create objective
  //
  Plato::DataMap dataMap;
  std::string tMyFunction("Globalized Stress");
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

  Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<Plato::Tet4>>
    eeScalarFunction(tSpatialModel, dataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto value = eeScalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 12164.73465517308;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt state, u
  //
  auto grad_u = eeScalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
  -136045.3530811711, -117804.6353496175, -111724.3961057662,
  -194947.6707559799, -173058.8094781152, -12768.50241208767,
  -58902.31767480877, -55254.17412849799,  98955.89369367871,
  -193123.5989828244, -14592.57418524276, -163938.4506123385,
  -368006.4802340950, -21888.86127786443, -18240.71773155366,
  -174882.8812512706, -7296.287092621476,  145697.7328807848,
  -57078.24590165332,  103212.0611643744, -52214.05450657228,
  -173058.8094781151,  151169.9482002509, -5472.215319466181,
  -115980.5635764621,  47957.88703587653,  46741.83918710628,
  -20064.78950470938, -165762.5223854940, -158466.2352928726,
  -21888.86127786437, -324228.7576783666, -7296.287092621540,
  -1824.071773155300, -158466.2352928726,  151169.9482002512
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(grad_u_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-12);
    }
  }


  // compute and test criterion gradient wrt control, z
  //
  auto grad_z = eeScalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
   380.1479579741605, 506.8639439655473, 126.7159859913868,
   506.8639439655473, 760.2959159483208, 253.4319719827737,
   126.7159859913868, 253.4319719827736, 126.7159859913870,
   506.8639439655479, 760.2959159483213, 253.4319719827739,
   760.2959159483212, 1520.591831896641, 760.2959159483211,
   253.4319719827736, 760.2959159483206, 506.8639439655473,
   126.7159859913873, 253.4319719827741, 126.7159859913868,
   253.4319719827742, 760.2959159483217, 506.8639439655475,
   126.7159859913868, 506.8639439655471, 380.1479579741601
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  auto grad_x = eeScalarFunction.gradient_x(tSolution, z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
   1889.624352503137,  573.5565681715406,  134.8673067276750,
   1929.468920298000,  558.6789827717420,  228.4649895877097,
   39.84456779486254, -14.87758539979850,  93.59768286003472,
   1880.218982422804,  668.9783228047302,  96.27678827685651,
   1950.502747932197,  734.6449066383240,  244.8816355461080,
   70.28376550939277,  65.66658383359324,  148.6048472692513,
  -9.405370080332322,  95.42175463319003, -38.59051845081814,
   21.03382763419772,  175.9659238665815,  16.41664595839830,
   30.43919771453029,  80.54416923339170,  55.00716440921650,
   1859.185154788609,  493.0123989381497,  79.86014231845859,
   1908.435092663803,  382.7130589051604,  212.0483436293115,
   49.24993787519501, -110.2993400329887,  132.1882013108530,
   1809.935216913413,  603.3117389711377, -52.32805899239442,
   0.000000000000000,  0.000000000000000,  0.0000000000000000
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    if(grad_x_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_x_Host[iNode]) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
    }
  }
}

/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         EffectiveEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, EffectiveEnergy3D_ShearCellProblem )
{ 
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                         \n"
    "  <ParameterList name='Spatial Model'>                                                       \n"
    "    <ParameterList name='Domains'>                                                           \n"
    "      <ParameterList name='Design Volume'>                                                   \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                         \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>                 \n"
    "      </ParameterList>                                                                       \n"
    "    </ParameterList>                                                                         \n"
    "  </ParameterList>                                                                           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                                 \n"
    "  <ParameterList name='Criteria'>                                                              \n"
    "    <ParameterList name='Effective Energy'>                                                    \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>                           \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Effective Energy'/>          \n"
    "      <Parameter name='Assumed Strain' type='Array(double)' value='{0.0,0.0,0.0,1.0,0.0,0.0}'/>\n"
    "      <ParameterList name='Penalty Function'>                                                  \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                                 \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>                            \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                                    \n"
    "      </ParameterList>                                                                         \n"
    "    </ParameterList>                                                                           \n"
    "  </ParameterList>                                                                             \n"
    "  <ParameterList name='Cell Problem Forcing'>                                                \n"
    "    <Parameter name='Column Index' type='int' value='3'/>                                    \n"
    "  </ParameterList>                                                                           \n"
    "  <ParameterList name='Material Models'>                                                     \n"
    "    <ParameterList name='Unobtainium'>                                                       \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                                        \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>                         \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>                       \n"
    "      </ParameterList>                                                                       \n"
    "    </ParameterList>                                                                         \n"
    "  </ParameterList>                                                                           \n"
    "</ParameterList>                                                                             \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", meshWidth);

  auto numVerts = tMesh->NumNodes();

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( numVerts, 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);



  Plato::ScalarMultiVector solution("solution", /*numSteps=*/1, spaceDim*numVerts);

  // create mesh based displacement
  //
  std::vector<Plato::Scalar> solution_gold = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.114894795127353302, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00415282392026578451, 0,
    0, 0, 0, 0, 0, -0.00415282392026578451, 0.0833333333333333426,
   -5.44375829931787534e-18, -4.39093400669304936e-18, 0, 0,
    0.00415282392026578278, 0, 0, 0, 0, 0.00415282392026578278, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0517718715393134105, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

  // push gold data from host to device
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    tHostView(solution_gold.data(), solution_gold.size());
  auto step0 = Kokkos::subview(solution, 0, Kokkos::ALL());
  Kokkos::deep_copy(step0, tHostView);

  // create criterion
  //
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

  Plato::DataMap dataMap;
  std::string tMyFunction("Effective Energy");
  Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<Plato::Tet4>>
    eeScalarFunction(tSpatialModel, dataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", solution);
  auto value = eeScalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 384615.384615384275;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt state, u
  //
  auto grad_u = eeScalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
0, 32051.28205128205, 32051.28205128205, 0, 0, 48076.92307692307, 0,
-32051.28205128205, 16025.64102564102, 0, 48076.92307692307, 0, 0, 0,
0, 0, -48076.92307692307, 0, 0, 16025.64102564102,
-32051.28205128205, 0, 0, -48076.92307692307, 0, -16025.64102564102,
-16025.64102564102, 0, 48076.92307692307, 48076.92307692307, 0, 0,
96153.84615384616, 0, -48076.92307692307, 48076.92307692307, 0,
96153.84615384616, 0, 0, 0, 0, 0, -96153.84615384616, 0, 0,
48076.92307692307, -48076.92307692307, 0, 0, -96153.84615384616, 0,
-48076.92307692307, -48076.92307692307, 0, 16025.64102564102,
16025.64102564102, 0, 0, 48076.92307692307, 0, -16025.64102564102,
32051.28205128205, 0, 48076.92307692307, 0, 0, 0, 0, 0,
-48076.92307692307, 0, 0, 32051.28205128205, -16025.64102564102, 0,
0, -48076.92307692307, 0, -32051.28205128205, -32051.28205128205
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(fabs(grad_u_gold[iNode]) < 1e-10){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
    }
  }


  // compute and test criterion gradient wrt control, z
  //
  auto grad_z = eeScalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
12052.5066019252044, 15975.7272765993694, 3989.77234006303706,
15975.7272765993694, 24105.0132038504162, 8029.45842916773017,
3989.77234006303706, 8029.45842916773017, 4006.41025641025590,
16125.4685237243357, 24005.1857057671004, 7912.99301473719970,
24005.1857057671004, 48210.0264077008178, 24005.1857057671004,
7912.99301473719970, 24005.1857057671004, 16125.4685237243339,
4006.41025641025590, 8029.45842916773017, 3989.77234006303706,
8029.45842916773017, 24105.0132038504162, 15975.7272765993694,
3989.77234006303706, 15975.7272765993694, 12052.5066019252044
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  auto grad_x = eeScalarFunction.gradient_x(tSolution, z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
-32051.2820512820472, -32184.3853820597978, -32184.3853820597942,
-48076.9230769230708, -47943.8197461453237, 0, -16025.6410256410218,
-16025.6410256410236, 31918.1787205042965, -48076.9230769230708,
1.08002495835535228e-12, -47943.8197461453237,
-96153.8461538461415, 1.81898940354585648e-12,
-1.81898940354585648e-12, -48076.9230769230708, 0,
48210.0264077008178, -16025.6410256410218, 31918.1787205042965,
-16025.6410256410236, -48076.9230769230708, 48210.0264077008178, 0,
-32051.2820512820472, 16025.6410256410236, 16025.6410256410236,
-3.63797880709171295e-12, -48343.1297384785648,
-48343.1297384785721, 3.63797880709171295e-12,
-96153.8461538461561, 1.81898940354585648e-12, 0,
-47810.7164153675694, 47810.7164153675694,
3.63797880709171295e-12, 2.89901436190120876e-12,
-96153.8461538461561, 0, -1.81898940354585648e-12,
-1.81898940354585648e-12, 3.63797880709171295e-12,
-5.45696821063756943e-12, 96153.8461538461561,
5.68434188608080149e-13, 47810.7164153675694,
-47810.7164153675694, 3.63797880709171295e-12,
96153.8461538461561, -5.45696821063756943e-12, 0,
48343.1297384785721, 48343.1297384785648, 32051.2820512820472,
-16025.6410256410236, -16025.6410256410236, 48076.9230769230708,
-48210.0264077008178, 0, 16025.6410256410218, -31918.1787205042965,
16025.6410256410236, 48076.9230769230708, 0, -48210.0264077008178,
96153.8461538461561, -7.27595761418342590e-12,
-1.81898940354585648e-12, 48076.9230769230708, 0,
47943.8197461453237, 16025.6410256410218, 16025.6410256410236,
-31918.1787205042965, 48076.9230769230708, 47943.8197461453237, 0,
32051.2820512820435, 32184.3853820597942, 32184.3853820597942
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    if(fabs(grad_x_gold[iNode]) < 1e-10){
      TEST_ASSERT(fabs(grad_x_Host[iNode]) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
    }
  }
}

/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         EffectiveEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, EffectiveEnergy3D_NormalCellProblem )
{ 
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                         \n"
    "  <ParameterList name='Spatial Model'>                                                       \n"
    "    <ParameterList name='Domains'>                                                           \n"
    "      <ParameterList name='Design Volume'>                                                   \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                         \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>                 \n"
    "      </ParameterList>                                                                       \n"
    "    </ParameterList>                                                                         \n"
    "  </ParameterList>                                                                           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                                 \n"
    "  <ParameterList name='Criteria'>                                                              \n"
    "    <ParameterList name='Effective Energy'>                                                    \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>                           \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Effective Energy'/>          \n"
    "      <Parameter name='Assumed Strain' type='Array(double)' value='{1.0,0.0,0.0,0.0,0.0,0.0}'/>\n"
    "      <ParameterList name='Penalty Function'>                                                  \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                                 \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>                            \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                                    \n"
    "      </ParameterList>                                                                         \n"
    "    </ParameterList>                                                                           \n"
    "  </ParameterList>                                                                             \n"
    "  <ParameterList name='Cell Problem Forcing'>                                                \n"
    "    <Parameter name='Column Index' type='int' value='0'/>                                    \n"
    "  </ParameterList>                                                                           \n"
    "  <ParameterList name='Material Models'>                                                     \n"
    "    <ParameterList name='Unobtainium'>                                                       \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                                        \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>                         \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>                       \n"
    "      </ParameterList>                                                                       \n"
    "    </ParameterList>                                                                         \n"
    "  </ParameterList>                                                                           \n"
    "</ParameterList>                                                                             \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", meshWidth);

  auto numVerts = tMesh->NumNodes();

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( numVerts, 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);



  Plato::ScalarMultiVector solution("solution", /*numSteps=*/1, spaceDim*numVerts);

  // create mesh based displacement
  //
  std::vector<Plato::Scalar> solution_gold = {
0.000000000000000000,   0.0000000000000000000,   0.00000000000000000,
0.000000000000000000,   0.0000000000000000000,  -0.0156569934602977037,
0.000000000000000000,   0.0000000000000000000,   0.00000000000000000,
0.000000000000000000,  -0.0156569934602976933,   0.00000000000000000,
0.000000000000000000,  -0.0179154946192411728,  -0.0179154946192411797,
0.000000000000000000,  -0.0185274729107747851,   0.00000000000000000,
0.000000000000000000,   0.0000000000000000000,   0.00000000000000000,
0.000000000000000000,   0.0000000000000000000,  -0.0185274729107747886,
0.000000000000000000,   0.0000000000000000000,   0.00000000000000000,
0.246845187533154764,   0.0000000000000000000,   0.00000000000000000,
0.223762542768190076,   0.0000000000000000000,   0.00519118160918287472,
0.188884954970459845,   0.0000000000000000000,   0.00000000000000000,
0.223762542768190020,   0.00519118160918285738,  0.00000000000000000,
0.155489014029553008,   0.00118135963986893376,  0.00118135963986895805,
0.0883401177353740630, -0.00616695004699989967,  0.00000000000000000,
0.188884954970459873,   0.0000000000000000000,   0.00000000000000000,
0.0883401177353740769,  0.0000000000000000000,  -0.00616695004699988319,
0.0186376910398909669,  0.0000000000000000000,   0.00000000000000000,
0.000000000000000000,   0.0000000000000000000,   0.00000000000000000,
0.000000000000000000,   0.0000000000000000000,   0.0105036127726940414,
0.000000000000000000,   0.0000000000000000000,   0.00000000000000000,
0.000000000000000000,   0.0105036127726940622,   0.00000000000000000,
0.000000000000000000,   0.0147346272871489350,   0.0147346272871489315,
0.000000000000000000,   0.0223597982645950891,   0.00000000000000000,
0.000000000000000000,   0.0000000000000000000,   0.00000000000000000,
0.000000000000000000,   0.0000000000000000000,   0.0223597982645950856,
0.000000000000000000,   0.0000000000000000000,   0.00000000000000000 };


  // push gold data from host to device
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    tHostView(solution_gold.data(), solution_gold.size());
  auto step0 = Kokkos::subview(solution, 0, Kokkos::ALL());
  Kokkos::deep_copy(step0, tHostView);

  // create criterion
  //
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

  Plato::DataMap dataMap;
  std::string tMyFunction("Effective Energy");
  Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<Plato::Tet4>>
    eeScalarFunction(tSpatialModel, dataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", solution);
  auto value = eeScalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 1.34615384615384601e6;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt state, u
  //
  auto grad_u = eeScalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
112179.4871794871, 48076.92307692306, 48076.92307692306,
168269.2307692307, 72115.38461538460, 0.000000000000000,
56089.74358974357, 24038.46153846153, -48076.92307692306,
168269.2307692307, 0.000000000000000, 72115.38461538460,
336538.4615384614, 0.000000000000000, 0.000000000000000,
168269.2307692307, 0.000000000000000, -72115.38461538460,
56089.74358974357, -48076.92307692306, 24038.46153846153,
168269.2307692307, -72115.38461538460, 0.000000000000000,
112179.4871794871, -24038.46153846153, -24038.46153846153,
0.000000000000000, 72115.38461538460, 72115.38461538460,
-1.455191522836685e-11, 144230.7692307692, 0.000000000000000,
0.000000000000000, 72115.38461538460, -72115.38461538460,
-1.455191522836685e-11, 0.000000000000000, 144230.7692307692,
-4.365574568510056e-11, 0.000000000000000, 0.000000000000000,
-1.455191522836685e-11, 0.000000000000000, -144230.7692307692,
0.000000000000000, -72115.38461538460, 72115.38461538460,
-1.455191522836685e-11, -144230.7692307692, 0.000000000000000,
0.000000000000000, -72115.38461538460, -72115.38461538460,
-112179.4871794871, 24038.46153846153, 24038.46153846153,
-168269.2307692307, 72115.38461538460, 0.000000000000000,
-56089.74358974357, 48076.92307692306, -24038.46153846153,
-168269.2307692307, 0.000000000000000, 72115.38461538460,
-336538.4615384614, 0.000000000000000, 0.000000000000000,
-168269.2307692307, 0.000000000000000, -72115.38461538460,
-56089.74358974357, -24038.46153846153, 48076.92307692306,
-168269.2307692307, -72115.38461538460, 0.000000000000000,
-112179.4871794871, -48076.92307692306, -48076.92307692306
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(fabs(grad_u_gold[iNode]) < 1e-10){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
    }
  }


  // compute and test criterion gradient wrt control, z
  //
  auto grad_z = eeScalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
   25009.4131876460742, 37621.8643908716476, 10095.2840528259185,
   37621.8643908716404, 65420.5939062928228, 23720.8110818120258,
   10095.2840528259185, 23720.8110818120258, 13061.7278050108744,
   66029.7816635471245, 90094.2200129909324, 26122.2272325746198,
   90094.2200129909470, 155699.063565457996, 71259.5965853674861,
   26122.2272325746198, 71259.5965853674861, 48404.2720297033302,
   20641.8333034958050, 40576.6606721150602, 19634.3310851935239,
   40576.6606721150602, 116679.465204682449, 73430.3395317338582,
   19634.3310851935203, 73430.3395317338436, 50097.0261970390711 };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  auto grad_x = eeScalarFunction.gradient_x(tSolution, z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
  -115407.611033288966,     -59890.1173628834731,     -59890.1173628834877,
  -170021.296515866125,     -97126.6548281997821,     -7.27595761418342590e-12,
  -56227.7474094780482,     -34147.9861350302526,      82267.7525630205928,
  -170021.296515866095,     -6.36646291241049767e-12, -97126.6548281997821,
  -336262.453898992506,     -2.00088834390044212e-11, -1.63709046319127083e-11,
  -166655.168842329818,      1.09139364212751389e-11,  146062.077953277621,
  -56227.7474094780482,      82267.7525630205928,     -34147.9861350302526,
  -166655.168842329818,      146062.077953277621,      7.27595761418342590e-12,
  -108675.355686216382,      53108.2330767377789,      53108.2330767377862,
  -1.45519152283668518e-11, -192366.664605835686,     -192366.664605835686,
   1.45519152283668518e-11, -331294.233826283715,     -2.91038304567337036e-11,
   7.27595761418342590e-12, -142914.075479116524,      177697.165539035719,
   0.00000000000000000,     -7.95807864051312208e-12, -331294.233826283715,
   2.91038304567337036e-11,  1.45519152283668518e-11, -2.91038304567337036e-11,
   2.18278728425502777e-11, -7.27595761418342590e-12,  319405.944968142954,
   9.09494701772928238e-12,  177697.165539035719,     -142914.075479116524,
   2.91038304567337036e-11,  319405.944968142954,     -1.45519152283668518e-11,
   1.45519152283668518e-11,  156875.561808302155,      156875.561808302155,
   109752.733327579175,     -83275.7287566346058,     -83275.7287566346058,
   166485.844925396872,     -245913.755875948555,     -7.95807864051312208e-13,
   55519.7346718637054,     -159224.629283913062,      76203.7706730678328,
   166485.844925396843,      1.45519152283668518e-11, -245913.755875948525,
   337678.479374221119,      0.00000000000000000,     -5.82076609134674072e-11,
   169482.607695184706,     -4.36557456851005554e-11,  209574.634346625826,
   55519.7346718637054,      76203.7706730678183,     -159224.629283913062,
   169482.607695184706,      209574.634346625826,      0.00000000000000000,
   115746.258867154844,      124958.705225635247,      124958.705225635218
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    if(fabs(grad_x_gold[iNode]) < 1e-10){
      TEST_ASSERT(fabs(grad_x_Host[iNode]) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
    }
  }
}

TEUCHOS_UNIT_TEST( DerivativeTests, ElastostaticResidual2D_InhomogeneousEssentialConditions )
{
    Teuchos::RCP<Teuchos::ParameterList> tElasticityParams =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                          \n"
      "  <ParameterList name='Spatial Model'>                                        \n"
      "    <ParameterList name='Domains'>                                            \n"
      "      <ParameterList name='Design Volume'>                                    \n"
      "        <Parameter name='Element Block' type='string' value='body'/>          \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
      "      </ParameterList>                                                        \n"
      "    </ParameterList>                                                          \n"
      "  </ParameterList>                                                            \n"
      "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
      "  <Parameter name='Physics' type='string' value='Mechanical'/>                \n"
      "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                  \n"
      "  <ParameterList name='Elliptic'>                                             \n"
      "    <ParameterList name='Penalty Function'>                                   \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                  \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-6'/>          \n"
      "    </ParameterList>                                                          \n"
      "  </ParameterList>                                                            \n"
      "  <ParameterList name='Essential Boundary Conditions'>                        \n"
      "  </ParameterList>                                                            \n"
      "  <ParameterList name='Material Models'>                                      \n"
      "    <ParameterList name='Unobtainium'>                                        \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
      "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
      "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n"
      "      </ParameterList>                                                        \n"
      "    </ParameterList>                                                          \n"
      "  </ParameterList>                                                            \n"
      "</ParameterList>                                                              \n"
    );

    // SETUP INPUT PARAMETERS
    constexpr Plato::OrdinalType tMeshWidth = 3;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TRI3", tMeshWidth);

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    Plato::Elliptic::Problem<Plato::Mechanics<Plato::Tri3>>
        tElasticityProblem(tMesh, *tElasticityParams, tMachine);

    // SET ESSENTIAL/DIRICHLET BOUNDARY CONDITIONS 
    Plato::OrdinalType tDispDofX = 0;
    Plato::OrdinalType tDispDofY = 1;
    auto tNumDofsPerNode = tElasticityProblem.numDofsPerNode();
    auto tDirichletIndicesBoundaryX0 = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_2D(tMesh, "x-", tNumDofsPerNode, tDispDofX);
    auto tDirichletIndicesBoundaryY0 = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_2D(tMesh, "y-", tNumDofsPerNode, tDispDofY);
    auto tDirichletIndicesBoundaryX1 = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_2D(tMesh, "x+", tNumDofsPerNode, tDispDofX);

    Plato::Scalar tValueToSet = 0;
    auto tNumDirichletDofs = tDirichletIndicesBoundaryX0.size() + tDirichletIndicesBoundaryY0.size() + tDirichletIndicesBoundaryX1.size();
    Plato::ScalarVector tDirichletValues("Dirichlet Values", tNumDirichletDofs);
    Plato::OrdinalVector tDirichletDofs("Dirichlet Dofs", tNumDirichletDofs);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        tDirichletValues(aIndex) = tValueToSet;
        tDirichletDofs(aIndex) = tDirichletIndicesBoundaryX0(aIndex);
    }, "set dirichlet values and indices");

    auto tOffset = tDirichletIndicesBoundaryX0.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryY0.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryY0(aIndex);
    }, "set dirichlet values and indices");

    tValueToSet = 6e-4;
    tOffset += tDirichletIndicesBoundaryY0.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX1.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryX1(aIndex);
    }, "set dirichlet values and indices");

    // SOLVE ELASTOSTATICS EQUATIONS
    auto tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Control", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    tElasticityProblem.setEssentialBoundaryConditions(tDirichletDofs, tDirichletValues);
    auto tElasticitySolution = tElasticityProblem.solution(tControl);

    // TEST RESULTS    
    const Plato::OrdinalType tTimeStep = 0;
    auto tState = tElasticitySolution.get("State");
    auto tSolution = Kokkos::subview(tState, tTimeStep, Kokkos::ALL());
    auto tHostSolution = Kokkos::create_mirror_view(tSolution);
    Kokkos::deep_copy(tHostSolution, tSolution);

    std::vector<Plato::Scalar> tGold = {
    0.00000000000000000000,  0.0000000000000000000,
    0.00000000000000000000, -8.5714285714284777e-05,
    0.00000000000000000000, -0.00017142857142857061,
    0.00000000000000000000, -0.00025714285714285672,
    0.00019999999999999939,  0.0000000000000000000,
    0.00019999999999999795, -8.5714285714286851e-05,
    0.00019999999999999827, -0.00017142857142857189,
    0.00020000000000000042, -0.00025714285714285737,
    0.00039999999999999937,  0.0000000000000000000,
    0.00039999999999999872, -8.5714285714287502e-05,
    0.00039999999999999704, -0.00017142857142857611,
    0.00040000000000000045, -0.00025714285714286105,
    0.00059999999999999984,  0.0000000000000000000,
    0.00059999999999999984, -8.5714285714287529e-05,
    0.00059999999999999984, -0.00017142857142857495,
    0.00059999999999999984, -0.00025714285714286295
    };


    constexpr Plato::Scalar tTolerance = 1e-4;
    for(Plato::OrdinalType tDofIndex=0; tDofIndex < tHostSolution.size(); tDofIndex++)
    {
        if(tGold[tDofIndex] == 0.0){
            TEST_ASSERT(fabs(tHostSolution(tDofIndex)) < 1e-12);
        } else {
            TEST_FLOATING_EQUALITY(tHostSolution(tDofIndex), tGold[tDofIndex], tTolerance);
        }
    }
}

// Reference Strain Test
TEUCHOS_UNIT_TEST( DerivativeTests, referenceStrain3D )
{ 
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n" 
    "        <Parameter  name='e11' type='double' value='-0.01'/>                  \n"
    "        <Parameter  name='e22' type='double' value='-0.01'/>                  \n"
    "        <Parameter  name='e33' type='double' value=' 0.02'/>                  \n"      
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;

  using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", meshWidth);

  int numCells = tMesh->NumElements();
  int numVoigtTerms = ElementType::mNumVoigtTerms;
  
  Plato::ScalarMultiVectorT<Plato::Scalar>
    stress("stress",numCells,numVoigtTerms);


  Plato::ScalarMultiVector elasticStrain("strain", numCells, numVoigtTerms);
  auto tHostStrain = Kokkos::create_mirror(elasticStrain);
  tHostStrain(0,0) = 0.0006; tHostStrain(1,0) = 0.006 ; tHostStrain(2,0) = 0.006 ; 
  tHostStrain(0,1) = 0.0048; tHostStrain(1,1) = 0.0048; tHostStrain(2,1) = 0.0012; 
  tHostStrain(0,2) = 0.0024; tHostStrain(1,2) =-0.0030; tHostStrain(2,2) = 0.0006; 
  tHostStrain(0,3) = 0.0072; tHostStrain(1,3) = 0.0018; tHostStrain(2,3) = 0.0018; 
  tHostStrain(0,4) = 0.003 ; tHostStrain(1,4) = 0.0030; tHostStrain(2,4) = 0.0066; 
  tHostStrain(0,5) = 0.0054; tHostStrain(1,5) = 0.0108; tHostStrain(2,5) = 0.0072; 
  
  tHostStrain(3,0) = 0.012 ; tHostStrain(4,0) = 0.006 ; tHostStrain(5,0) = 0.006 ;
  tHostStrain(3,1) =-0.0048; tHostStrain(4,1) = 0.0012; tHostStrain(5,1) = 0.0012;
  tHostStrain(3,2) = 0.0006; tHostStrain(4,2) = 0.0006; tHostStrain(5,2) = 0.0006;
  tHostStrain(3,3) =-0.0042; tHostStrain(4,3) = 0.0018; tHostStrain(5,3) = 0.0018;
  tHostStrain(3,4) = 0.0126; tHostStrain(4,4) = 0.0066; tHostStrain(5,4) = 0.0066;
  tHostStrain(3,5) = 0.0072; tHostStrain(4,5) = 0.0072; tHostStrain(5,5) = 0.0072;
  Kokkos::deep_copy(elasticStrain , tHostStrain );

  Plato::ElasticModelFactory<spaceDim> mmfactory(*tParamList);
  auto materialModel = mmfactory.create("Unobtainium");

  Plato::LinearStress<Plato::Elliptic::ResidualTypes<ElementType>, ElementType> voigtStress(materialModel);

  Plato::ScalarVectorT<Plato::Scalar> cellVolume("cell volume",numCells);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(int cellOrdinal)
  {
    voigtStress(cellOrdinal, stress, elasticStrain);
  }, "referenceStrain");

  // test Inherent Strain stress
  //
  auto stress_Host = Kokkos::create_mirror_view( stress );
  Kokkos::deep_copy( stress_Host, stress );

  std::vector<std::vector<Plato::Scalar>> stress_gold = { 
   { 12653.8461538462, 15884.6153846154,-9038.46153846154, 2769.23076923077, 1153.84615384615, 2076.92307692308},
   { 16807.6923076923, 15884.6153846154,-13192.3076923077, 692.307692307692, 1153.84615384615, 4153.84615384615},
   { 16807.6923076923, 13115.3846153846,-10423.0769230769, 692.307692307692, 2538.46153846154, 2769.23076923077},
   { 21423.0769230769, 8500.00000000000,-10423.0769230769,-1615.38461538462, 4846.15384615385, 2769.23076923077},
   { 16807.6923076923, 13115.3846153846,-10423.0769230769, 692.307692307692, 2538.46153846154, 2769.23076923077},
   { 16807.6923076923, 13115.3846153846,-10423.0769230769, 692.307692307692, 2538.46153846154, 2769.23076923077}
  };


  for(int iCell=0; iCell<int(stress_gold.size()); iCell++){
    for(int iVoigt=0; iVoigt<numVoigtTerms; iVoigt++){
      if(stress_gold[iCell][iVoigt] == 0.0){
        TEST_ASSERT(fabs(stress_Host(iCell,iVoigt)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(stress_Host(iCell,iVoigt), stress_gold[iCell][iVoigt], 1e-13);
      }
    }
  }

}

/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         Volume in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, Volume3D )
{ 
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Volume'>                                             \n"
    "      <Parameter name='Linear' type='bool' value='true'/>                     \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Volume'/>   \n"
    "      <ParameterList name='Penalty Function'>                                 \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "    <ParameterList name='Internal Elastic Energy'>                            \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", meshWidth);


  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( tMesh->NumNodes(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);


  // create mesh based displacement from host data
  //
  ordType tNumDofs = spaceDim*tMesh->NumNodes();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, tNumDofs);
  auto u = Kokkos::subview(U, 0, Kokkos::ALL());
  auto u_host = Kokkos::create_mirror_view( u );
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for(ordType i=0; i<tNumDofs; i++)
  {
      u_host(i) = (disp += dval);
  }
  Kokkos::deep_copy(u, u_host);


  // create objective
  //
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

  Plato::DataMap dataMap;
  std::string tMyFunction("Volume");
  Plato::Geometric::GeometryScalarFunction<::Plato::Geometrical<Plato::Tet4>>
    volScalarFunction(tSpatialModel, dataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  auto value = volScalarFunction.value(z);

  Plato::Scalar value_gold = 1.0;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt control, z
  //
  auto grad_z = volScalarFunction.gradient_z(z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
   0.03125000000000000, 0.04166666666666666, 0.01041666666666667,
   0.04166666666666666, 0.06250000000000000, 0.02083333333333333,
   0.01041666666666667, 0.02083333333333333, 0.01041666666666667,
   0.04166666666666666, 0.06250000000000000, 0.02083333333333333,
   0.06250000000000000, 0.1249999999999999,  0.06250000000000000,
   0.02083333333333333, 0.06250000000000000, 0.04166666666666666,
   0.01041666666666667, 0.02083333333333333, 0.01041666666666667,
   0.02083333333333333, 0.06250000000000000, 0.04166666666666666,
   0.01041666666666667, 0.04166666666666666, 0.03125000000000000
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  auto grad_x = volScalarFunction.gradient_x(z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
  -0.08333333333333333, -0.08333333333333333, -0.08333333333333333,
  -0.1250000000000000,  -0.1250000000000000,   0.0000000000000000,
  -0.04166666666666666, -0.04166666666666666,  0.08333333333333333,
  -0.1250000000000000,   0.0000000000000000,  -0.1250000000000000,
  -0.2500000000000000,   0.0000000000000000,   0.0000000000000000,
  -0.1250000000000000,   0.0000000000000000,   0.1250000000000000,
  -0.04166666666666666,  0.08333333333333333, -0.04166666666666666,
  -0.1250000000000000,   0.1250000000000000,   0.0000000000000000,
  -0.08333333333333333,  0.04166666666666666,  0.04166666666666666,
   0.0000000000000000,  -0.1250000000000000,  -0.1250000000000000,
   0.0000000000000000,  -0.2500000000000000,   0.0000000000000000,
   0.0000000000000000,  -0.1250000000000000,   0.1250000000000000,
   0.000000000000000,    0.0000000000000000,  -0.2500000000000000,
   0.0000000000000000,   0.0000000000000000,   0.0000000000000000,
   0.0000000000000000,   0.0000000000000000,   0.2500000000000000,
   0.0000000000000000,   0.1250000000000000,  -0.1250000000000000,
   0.0000000000000000,   0.2500000000000000,   0.0000000000000000,
   0.0000000000000000,   0.1250000000000000,   0.1250000000000000,
   0.08333333333333333, -0.04166666666666666, -0.04166666666666666,
   0.1250000000000000,  -0.1250000000000000,   0.0000000000000000,
   0.04166666666666666, -0.08333333333333333,  0.04166666666666666,
   0.1250000000000000,   0.0000000000000000,  -0.1250000000000000,
   0.2500000000000000,   0.0000000000000000,   0.0000000000000000,
   0.1250000000000000,   0.0000000000000000,   0.1250000000000000,
   0.04166666666666666,  0.04166666666666666, -0.08333333333333333,
   0.1250000000000000,   0.1250000000000000,   0.0000000000000000,
   0.08333333333333333,  0.08333333333333333,  0.08333333333333333
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    if(grad_x_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_x_Host[iNode]) < 1e-13);
    } else {
      TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
    }
  }
}
