/*!
  These unit tests are for the Electroelastic functionality.
 \todo 
*/

#include "util/PlatoTestHelpers.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "Tet4.hpp"
#include "elliptic/base/VectorFunction.hpp"
#include "elliptic/electromechanics/Electromechanics.hpp"
#include "elliptic/criterioneval/CriterionEvaluatorScalarFunction.hpp"

#include <fenv.h>

/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalthermalEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ElectroelasticTests, InternalElectroelasticEnergy3D )
{ 
  // create test mesh
  //
  constexpr int tMeshWidth=2;
  constexpr int cSpaceDim=3;
  std::string tElementType("TET4");
  auto tMesh = Plato::TestHelpers::get_box_mesh(tElementType, tMeshWidth);

  // create database
  //
  Plato::Database tDatabase;

  // create mesh based solution from host data
  //
  int tNumDofsPerNode = (cSpaceDim+1);
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;
  Plato::ScalarMultiVector states("states", /*numSteps=*/1, tNumDofs);
  auto state = Kokkos::subview(states, 0, Kokkos::ALL());
  Plato::ScalarVector z("control", tNumDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     state(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal;

  }, "state");
  tDatabase.vector("controls",z);
  tDatabase.vector("states",state);

  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>                 \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                         \n"
    "  <ParameterList name='Criteria'>                                                   \n"
    "    <ParameterList name='Internal Electroelastic Energy'>                           \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>                \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Electroelastic Energy'/>  \n"
    "      <ParameterList name='Penalty Function'>                                       \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                      \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>                 \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                         \n"
    "      </ParameterList>                                                              \n"
    "    </ParameterList>                                                                \n"
    "  </ParameterList>                                                                  \n"
    "  <ParameterList name='Elliptic'>                                                   \n"
    "    <ParameterList name='Penalty Function'>                                         \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                           \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                        \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>                   \n"
    "    </ParameterList>                                                                \n"
    "  </ParameterList>                                                                  \n"
    "  <ParameterList name='Spatial Model'>                                              \n"
    "    <ParameterList name='Domains'>                                                  \n"
    "      <ParameterList name='Design Volume'>                                          \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                \n"
    "        <Parameter name='Material Model' type='string' value='CheezWhiz'/>          \n"
    "      </ParameterList>                                                              \n"
    "    </ParameterList>                                                                \n"
    "  </ParameterList>                                                                  \n"
    "  <ParameterList name='Material Models'>                                            \n"
    "    <ParameterList name='CheezWhiz'>                                                \n"
    "      <ParameterList name='Isotropic Linear Electroelastic'>                        \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>               \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>            \n"
    "        <Parameter  name='p11' type='double' value='1.0e-10'/>                      \n"
    "        <Parameter  name='p33' type='double' value='1.4e-10'/>                      \n"
    "        <Parameter  name='e33' type='double' value='15.8'/>                         \n"
    "        <Parameter  name='e31' type='double' value='-5.4'/>                         \n"
    "        <Parameter  name='e15' type='double' value='12.3'/>                         \n"
    "        <Parameter  name='Alpha' type='double' value='1e10'/>                       \n"
    "      </ParameterList>                                                              \n"
    "    </ParameterList>                                                                \n"
    "  </ParameterList>                                                                  \n"
    "</ParameterList>                                                                    \n"
  );

  // create constraint
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *params, tDataMap);
  Plato::Elliptic::VectorFunction<Plato::Elliptic::Linear::Electromechanics<Plato::Tet4>>
    vectorFunction(params->get<std::string>("PDE Constraint"), tSpatialModel, tDataMap, *params);
  // compute and test constraint value
  //

  auto residual = vectorFunction.value(tDatabase,/*cycle=*/0.);

  auto residualHost = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy(residualHost, residual);

  std::vector<Plato::Scalar> residual_gold = {
   -130456.4102564103, -66512.82051282050, -155087.1794871795,
   -65416.66666666666, -58061.53846153842, -52292.30769230767,
   -185100.0000000000, -107925.0000000000,  72394.87179487178,
    14220.51282051283, -30012.82051282052, -42508.33333333334,
   -175492.3076923077, -67669.23076923077, -185153.8461538461,
   -67300.00000000000, -75738.46153846149, -40384.61538461537,
   -275246.1538461539, -154200.0000000000,  99753.84615384614,
    27284.61538461538, -90092.30769230770, -86899.99999999997,
   -45035.89743589743, -1156.410256410261, -30066.66666666668,
   -1883.333333333338, -17676.92307692306,  11907.69230769231
  };

  for(int iVal=0; iVal<int(residual_gold.size()); iVal++){
    if(residual_gold[iVal] == 0.0){
      TEST_ASSERT(fabs(residualHost[iVal]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(residualHost[iVal], residual_gold[iVal], 1e-13);
    }
  }

  // compute and test constraint value
  //
  auto jacobian = vectorFunction.jacobianState(tDatabase,/*cycle=*/0.);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> jacobian_gold = {
3.52564102564102478e10, 0, 0, 0, 0, 3.52564102564102478e10, 0,
0, 0, 0, 3.52564102564102478e10, 6.73333333333333282e10, 0, 0,
6.73333333333333282e10, -5.66666666666666603e9,
-6.41025641025640965e9, 0, 3.20512820512820482e9,
1.02500000000000000e10, 0, -6.41025641025640965e9,
3.20512820512820482e9, 1.02500000000000000e10,
4.80769230769230652e9, 4.80769230769230652e9,
-2.24358974358974304e10, -2.63333333333333321e10,
-4.50000000000000000e9, -4.50000000000000000e9,
-2.63333333333333321e10, 2.33333333333333302e9,
-6.41025641025640965e9, 3.20512820512820482e9, 0, 0,
4.80769230769230652e9, -2.24358974358974304e10,
4.80769230769230652e9, -4.50000000000000000e9, 0,
3.20512820512820482e9, -6.41025641025640965e9,
-2.05000000000000000e10, 0, 1.02500000000000000e10,
-2.05000000000000000e10, 1.66666666666666651e9, 0,
3.20512820512820482e9, 3.20512820512820482e9,
1.02500000000000000e10, 4.80769230769230652e9, 0,
-8.01282051282051086e9, -5.75000000000000000e9,
4.80769230769230652e9, -8.01282051282051086e9, 0, 0,
-4.50000000000000000e9, -5.75000000000000000e9, 0, 0
  };

  for(int iVal=0; iVal<int(jacobian_gold.size()); iVal++){
    if(jacobian_gold[iVal] == 0.0){
      TEST_ASSERT(fabs(jac_entriesHost[iVal]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(jac_entriesHost[iVal], jacobian_gold[iVal], 1e-13);
    }
  }

  // compute and test constraint gradient_z
  //
  auto gradient_z = vectorFunction.jacobianControl(tDatabase,/*cycle=*/0.);

  auto gradz_entries = gradient_z->entries();
  auto gradz_entriesHost = Kokkos::create_mirror_view( gradz_entries );
  Kokkos::deep_copy(gradz_entriesHost, gradz_entries);

  std::vector<Plato::Scalar> gradient_z_gold = { 
   -32614.1025641025626, -16628.2051282051252, -38771.7948717948675,
   -16354.1666666666661,  18098.7179487179492,  3555.12820512820508,
   -7503.20512820512977, -10627.0833333333321, -11258.9743589743575,
   -289.102564102563747, -7516.66666666666515, -470.833333333334849,
    6839.74358974358984,  3266.02564102563974, -15019.8717948717931,
   -11097.916666666667
  };

  for(int iVal=0; iVal<int(gradient_z_gold.size()); iVal++){
    if(gradient_z_gold[iVal] == 0.0){
      TEST_ASSERT(fabs(gradz_entriesHost[iVal]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(gradz_entriesHost[iVal], gradient_z_gold[iVal], 1e-13);
    }
  }

  // compute and test constraint gradient_x
  //
  auto gradient_x = vectorFunction.jacobianConfig(tDatabase,/*cycle=*/0.);

  auto gradx_entries = gradient_x->entries();
  auto gradx_entriesHost = Kokkos::create_mirror_view( gradx_entries );
  Kokkos::deep_copy(gradx_entriesHost, gradx_entries);

  std::vector<Plato::Scalar> gradient_x_gold = { 
   -63461.5384615384392, -126923.076923076878, -675184.615384615259,
   -322800.000000000000, -21153.8461538461488, -42307.6923076922903,
   -225061.538461538381, -107600.000000000000, -7051.28205128202535,
   -14102.5641025640871, -75020.5128205127403, -35866.6666666666642,
   -73771.7948717948602, -2135.89743589740465,  261758.974358974257,
    76583.3333333333430, -85756.4102564102359, -21812.8205128205154,
    66128.2051282050961,  29883.3333333333176,  40705.1282051281887,
    38141.0256410256261,  156005.128205128189,  79733.3333333333430,
   -19230.7692307692232, -37507.6923076922976,  133256.410256410250,
    94350.0000000000146,  115071.794871794846,  39743.5897435897350,
    178405.128205128189,  83166.6666666666715, -14102.5641025640944,
   -23189.7435897436044, -13328.2051282051616, -7783.33333333333940,
   -116079.487179487129,  25848.7179487179237, -40261.5384615384537,
   -11766.6666666666788,  21623.0769230769220,  39761.5384615384537,
    99441.0256410256261,  52150.0000000000000,  24038.4615384615317,
    22228.2051282051325,  94312.8205128204863,  51650.0000000000000
  };

  for(int iVal=0; iVal<int(gradient_x_gold.size()); iVal++){
    if(gradient_x_gold[iVal] == 0.0){
      TEST_ASSERT(fabs(gradx_entriesHost[iVal]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(gradx_entriesHost[iVal], gradient_x_gold[iVal], 1e-13);
    }
  }

  // create criterion
  //
  std::string tMyFunctionName("Internal Electroelastic Energy");
  Plato::Elliptic::CriterionEvaluatorScalarFunction<Plato::Elliptic::Linear::Electromechanics<Plato::Tet4>>
    scalarFunction(tSpatialModel, tDataMap, *params, tMyFunctionName);

  // compute and test criterion value
  //
  auto value = scalarFunction.value(tDatabase,/*cycle=*/0.);
  Plato::Scalar value_gold = 13.7312738461538331;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);

  // compute and test criterion gradient wrt state, u
  //
  auto grad_u = scalarFunction.gradientState(tDatabase,/*cycle=*/0.);
  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
  -260912.8205128205, -133025.6410256410, -310174.3589743589,
  -130833.3333333333, -116123.0769230768, -104584.6153846154,
  -370200.0000000000, -215850.0000000000,  144789.7435897436,
   28441.02564102566, -60025.64102564103, -85016.66666666666,
  -350984.6153846153, -135338.4615384615, -370307.6923076923,
  -134600.0000000000, -151476.9230769230, -80769.23076923068,
  -550492.3076923076, -308400.0000000001,  199507.6923076923,
   54569.23076923077, -180184.6153846153, -173800.0000000000,
  -90071.79487179485, -2312.820512820519, -60133.33333333334,
  -3766.666666666675, -35353.84615384612,  23815.38461538462,
  -180292.3076923077, -92550.00000000001
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
  auto grad_z = scalarFunction.gradientControl(tDatabase,/*cycle=*/0.);
  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
   0.4291023076923076, 0.5721364102564100, 0.1430341025641025,
   0.5721364102564100, 0.8582046153846152, 0.2860682051282050,
   0.1430341025641025, 0.2860682051282051, 0.1430341025641025,
   0.5721364102564099, 0.8582046153846153, 0.2860682051282052,
   0.8582046153846149, 1.716409230769231,  0.8582046153846153,
   0.2860682051282048, 0.8582046153846151, 0.5721364102564100,
   0.1430341025641025, 0.2860682051282050, 0.1430341025641026,
   0.2860682051282048, 0.8582046153846147, 0.5721364102564102,
   0.1430341025641024, 0.5721364102564099, 0.4291023076923074
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  auto grad_x = scalarFunction.gradientConfig(tDatabase,/*cycle=*/0.);
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
   2.42120410256410246,     0.0442194871794873157, -0.748108717948718027,
   2.42231692307692281,    -0.336833846153846039,   0.459858461538462104,
   0.00111282051282048100, -0.381053333333333355,   1.20796717948717913,
   2.37136307692307691,     1.36259076923076905,   -1.26221230769230797,
   2.32374769230769251,     1.91885538461538463,    0.639618461538461802,
  -0.0476153846153853227,   0.556264615384615135,   1.90183076923076921,
  -0.0498410256410255215,   1.31837128205128185,   -0.514103589743589717,
  -0.0985692307692313530,   2.25568923076923067,    0.179759999999999698,
  -0.0487282051282058037,   0.937317948717948934,   0.693863589743589748,
   2.46993230769230898,    -0.893098461538461175,  -1.44197230769230722,
   2.52088615384615222,    -2.59252307692307715,    0.280098461538462240,
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-12);
  }
}

