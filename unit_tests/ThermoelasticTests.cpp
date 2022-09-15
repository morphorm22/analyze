/*!
  These unit tests are for the Thermoelastic functionality.
 \todo 
*/

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

#include <alg/CrsLinearProblem.hpp>
#include <alg/ParallelComm.hpp>

#include "Mechanics.hpp"
#include "Solutions.hpp"
#include "ScalarProduct.hpp"
#include "WorksetBase.hpp"
#include "elliptic/VectorFunction.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
#include "elliptic/Problem.hpp"
#include "StateValues.hpp"
#include "ApplyConstraints.hpp"
#include "Thermomechanics.hpp"
#include "ImplicitFunctors.hpp"
#include "LinearThermoelasticMaterial.hpp"
#include "ExpressionEvaluator.hpp"

#include <fenv.h>

/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalthermalEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ThermoelasticTests, InternalThermoelasticEnergy3D )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);


  // create mesh based solution from host data
  //
  int tNumDofsPerNode = (spaceDim+1);
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


  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                         \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>          \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                  \n"
    "  <ParameterList name='Elliptic'>                                            \n"
    "    <ParameterList name='Penalty Function'>                                  \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                    \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                 \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>            \n"
    "    </ParameterList>                                                         \n"
    "  </ParameterList>                                                           \n"
    "  <ParameterList name='Criteria'>                                            \n"
    "    <ParameterList name='Internal Thermoelastic Energy'>                     \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>         \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Thermoelastic Energy'/>  \n"
    "      <ParameterList name='Penalty Function'>                                \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>               \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>          \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                  \n"
    "      </ParameterList>                                                       \n"
    "    </ParameterList>                                                         \n"
    "  </ParameterList>                                                           \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Beef Jerky'/>   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Beef Jerky'>                                          \n"
    "      <ParameterList name='Thermoelastic'>                                     \n"
    "        <ParameterList name='Elastic Stiffness'>                               \n"
    "          <Parameter  name='Poissons Ratio' type='double' value='0.3'/>        \n"
    "          <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>     \n"
    "        </ParameterList>                                                       \n"
    "        <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/>  \n"
    "        <Parameter  name='Thermal Conductivity' type='double' value='910.0'/>  \n"
    "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>   \n"
    "      </ParameterList>                                                         \n"
    "    </ParameterList>                                                           \n"
    "  </ParameterList>                                                           \n"
    "</ParameterList>                                                             \n"
  );

  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *params);
  Plato::Elliptic::VectorFunction<::Plato::Thermomechanics<Plato::Tet4>>
    vectorFunction(tSpatialModel, tDataMap, *params, params->get<std::string>("PDE Constraint"));


  // compute and test constraint value
  //
  auto residual = vectorFunction.value(state, z);

  auto residualHost = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy(residualHost, residual);

  std::vector<Plato::Scalar> residual_gold = { 
  -60255.72275641025,     -45512.32051282050,    -46153.40865384614,
  -0.0007886666666666667, -63460.51762820510,    -57691.53685897433,
  -37499.91666666666,     -0.001092000000000000, -3204.836538461539,
  -12179.25801282051,      8653.325320512817,    -0.0003033333333333334,
  -70191.07852564102,     -30768.98076923076,    -58652.95032051280,
  -0.0009100000000000000, -86536.33653846150,    -40384.24038461538,
  -53846.02884615383,     -0.001637999999999999, -16345.25801282050,
  -9615.259615384608,      4806.671474358979,    -0.0007279999999999999,
  -9935.480769230770,      14742.83974358974
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
  auto jacobian = vectorFunction.gradient_u(state, z);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> jacobian_gold = { 
3.52564102564102478e10, 0, 0, 52083.3333333333285, 0,
3.52564102564102478e10, 0, 52083.3333333333285, 0, 0,
3.52564102564102478e10, 52083.3333333333285, 0, 0, 0,
454.999999999999943, -6.41025641025640965e9, 0,
3.20512820512820482e9, 0, 0, -6.41025641025640965e9,
3.20512820512820482e9, 0, 4.80769230769230652e9,
4.80769230769230652e9, -2.24358974358974304e10,
52083.3333333333285, 0, 0, 0, -151.666666666666657,
-6.41025641025640965e9, 3.20512820512820482e9, 0, 0,
4.80769230769230652e9, -2.24358974358974304e10,
4.80769230769230652e9, 52083.3333333333285, 0,
3.20512820512820482e9, -6.41025641025640965e9, 0, 0, 0, 0,
-151.666666666666657, 0, 3.20512820512820482e9,
3.20512820512820482e9, 0, 4.80769230769230652e9, 0,
-8.01282051282051086e9, 26041.6666666666642,
4.80769230769230652e9, -8.01282051282051086e9, 0,
26041.6666666666642, 0, 0, 0, 0
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
  auto gradient_z = vectorFunction.gradient_z(state, z);

  auto gradz_entries = gradient_z->entries();
  auto gradz_entriesHost = Kokkos::create_mirror_view( gradz_entries );
  Kokkos::deep_copy(gradz_entriesHost, gradz_entries);

  std::vector<Plato::Scalar> gradient_z_gold = { 
  -15063.9306891025626,     -11378.0801282051252,      -11538.3521634615354,
  -0.000197166666666666671, -801.219551282049906,      -3044.82491987179446,
   2163.35216346153675,     -0.0000758333333333333244, -2483.90144230769147,
   3685.77243589743557,     -3124.94791666666515,      -0.0000303333333333333298,
  -3285.15745192307486,      640.978766025640425,      -961.590544871795146,
  -0.000106166666666666627
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
  auto gradient_x = vectorFunction.gradient_x(state, z);

  auto gradx_entries = gradient_x->entries();
  auto gradx_entriesHost = Kokkos::create_mirror_view( gradx_entries );
  Kokkos::deep_copy(gradx_entriesHost, gradx_entries);

  std::vector<Plato::Scalar> gradient_x_gold = { 
  -63461.5384615384464,    -126923.076923076878,     -190384.615384615347,
  -0.00327599999999999940, -21153.8461538461415,     -42307.6923076922903,
  -63461.5384615384537,    -0.00109199999999999965,  -7051.28205128204081,
  -14102.5641025640871,    -21153.8461538461452,     -0.000363999999999999740,
  -32371.7948717948639,    -9935.89743589742466,      82692.8076923076878,
   0.00103133333333333310, -22756.4102564102504,     -8012.82051282051179,
   13461.9134615384592,     0.000303333333333333189,  40704.6282051281887,
   38140.6506410256334,     36538.4615384615317,      0.000849333333333333234,
  -19230.7692307692232,     32692.8910256410163,      10256.4102564102541,
   0.000909999999999999785, 44871.2115384615172,      39743.5897435897423,
   44871.3782051282033,     0.000970666666666666553, -14102.5641025640944,
  -18589.3269230769292,    -5128.20512820512522,     -0.0000606666666666667951,
  -74679.4871794871433,     5449.13461538461343,     -5127.83012820512522,
  -0.000242666666666666638, 14422.6602564102468,      25961.5384615384537,
   25641.0673076923085,     0.000545999999999999827,  24038.0865384615281,
   17628.1634615384646,     20512.8205128205082,      0.000545999999999999827
  };

  for(int iVal=0; iVal<int(gradient_x_gold.size()); iVal++){
    if(gradient_x_gold[iVal] == 0.0){
      TEST_ASSERT(fabs(gradx_entriesHost[iVal]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(gradx_entriesHost[iVal], gradient_x_gold[iVal], 1e-13);
    }
  }

  // create objective
  //
  std::string tMyFunction("Internal Thermoelastic Energy");
  Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermomechanics<Plato::Tet4>>
    scalarFunction(tSpatialModel, tDataMap, *params, tMyFunction);

  // compute and test objective value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", states);
  auto value = scalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 3.20610709915224668;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);

  // compute and test objective gradient wrt state, u
  //
  tSolution.set("State", states);
  auto grad_u = scalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
  -120512.1330128205,   -91025.14102564099,  -92307.25480769228,
  -0.2828273333333333,  -126922.0560897435,  -115383.8445512820,
  -74999.91666666667,   -0.3771839999999999, -6409.964743589742,
  -24358.74519230768,    17307.17147435897,  -0.09435666666666664,
  -140383.3862179487,   -61538.21153846152,  -117306.7964743589,
  -0.3768200000000000,  -173074.7980769229,  -80768.85576923074,
  -107692.1826923077,   -0.5657759999999997, -32691.41185897436,
  -19230.64423076922,    9614.363782051296,  -0.1889560000000000,
  -19871.37820512821,    29486.42948717947,  -24999.66666666668,
  -0.09399266666666665, -46152.74198717946,   34614.23878205129,
  -32692.26602564101,   -0.1885920000000000, -26281.32211538461,
  5127.850961538466,    -7692.682692307675,  -0.09459933333333331
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
  tSolution.set("State", states);
  auto grad_z = scalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
   0.1001915780985077,  0.1335887051730102,  0.03339720207450256,
   0.1335884520480103,  0.2003826874470154,  0.06679431039900512,
   0.03339709894950257, 0.06679420727400512, 0.03339710832450257,
   0.1335876926730104,  0.2003816186970155,  0.06679400102400512,
   0.2003812624470156,  0.4007633873940310,  0.2003821249470154,
   0.06679379477400521, 0.2003817686970156,  0.1335878989230103,
   0.03339678957450264, 0.06679358852400522, 0.03339679894950260,
   0.06679348539900523, 0.2003806999470156,  0.1335871395480103,
   0.03339669582450261, 0.1335868864230105,  0.1001901155985078
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test objective gradient wrt node position, x
  //
  tSolution.set("State", states);
  auto grad_x = scalarFunction.gradient_x(tSolution, z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
0.77589451783770246873, 0.080512819881887298656,
-0.15128107943671786906, 0.64807554467692296551,
-0.051152760037046107744, 0.11653820078566161367,
-0.12781799816077951681, -0.13166535491893330279,
0.26781880522237938580, 0.70691955840227715946,
0.36922902359876919043, -0.27768997119444094324,
0.38230950264529250937, 0.39461479247778480373,
0.13153826415926148097, -0.32460870575698447249,
0.025386218879015307048, 0.40922748535370234713,
-0.068972709435425549884, 0.28871530371688197691,
-0.12640904175772310625, -0.26576334203163054504,
0.44576575251483052664, 0.015000363373600084094,
-0.19679025759620494274, 0.15705067379794873661,
0.14140913013132294651, 0.97267967543390765339,
-0.076537803916060881404, -0.29268854290137402696,
0.91383716170855255889, -0.49691708755187680158,
0.10153746241206208778, -0.058841163725353924641,
-0.42037883363581496354, 0.39422525531343605154,
  };



  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-11);
  }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, VolAvgStressPNormAxial_3D)
{
    const Plato::OrdinalType tNumElemX = 1;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    const Plato::Scalar tBoxWidth = 5.0;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tBoxWidth, tNumElemX, tBoxWidth, tNumElemY, tBoxWidth, tNumElemZ);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Mechanical'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "      <Parameter name='Iterations' type='int' value='500'/>                              \n"
      "      <Parameter name='Tolerance' type='double' value='1.0e-10'/>                        \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                    \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='10.0e0'/>                 \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
     "   <ParameterList name='volume avg stress pnorm'> \n"
     "     <Parameter name='Type' type='string' value='Division' /> \n"
     "       <Parameter name='Numerator' type='string' value='pnorm numerator' /> \n"
     "       <Parameter name='Denominator' type='string' value='pnorm denominator' /> \n"
     "   </ParameterList> \n"
     "   <ParameterList name='pnorm numerator'> \n"
     "     <Parameter name='Type' type='string' value='Scalar Function' /> \n"
     "     <Parameter name='Scalar Function Type' type='string' value='Stress P-Norm' /> \n"
     "     <ParameterList name='Penalty Function'> \n"
     "       <Parameter name='Type' type='string' value='SIMP' /> \n"
     "       <Parameter name='Exponent' type='double' value='1.0' /> \n"
     "     </ParameterList> \n"
     "     <Parameter name='Function' type='string' value='x/5.0' /> \n"
     "     <Parameter name='Exponent' type='double' value='1.0' /> \n"
     "   </ParameterList> \n"
     "   <ParameterList name='pnorm denominator'> \n"
     "     <Parameter name='Type' type='string' value='Scalar Function' /> \n"
     "     <Parameter name='Scalar Function Type' type='string' value='vol avg stress p-norm denominator' /> \n"
     "     <ParameterList name='Penalty Function'> \n"
     "       <Parameter name='Type' type='string' value='SIMP' /> \n"
     "       <Parameter name='Exponent' type='double' value='1.0' /> \n"
     "     </ParameterList> \n"
     "     <Parameter name='Function' type='string' value='x/5.0' /> \n"
     "     <Parameter name='Exponent' type='double' value='1.0' /> \n"
     "   </ParameterList> \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='y-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='z-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Disp Boundary Condition'>                             \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                   \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x+'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='0.1'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Add sidesets
    using PhysicsT = Plato::Mechanics<Plato::Tet4>;

    Plato::Elliptic::Problem<PhysicsT> tEllipticProblem(tMesh, *tParamList, tMachine);
    tEllipticProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->NumNodes();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tEllipticProblem.solution(tControls);


    const Plato::Scalar tSpatialWeightingFactor = 0.5;
    const Plato::Scalar tElasticModulus = 10.0;
    const Plato::Scalar tStrain = 0.1 / tBoxWidth;
    const Plato::Scalar tStress = tElasticModulus * tStrain;
    const Plato::Scalar tBoxVolume = tSpatialWeightingFactor * (tBoxWidth*tBoxWidth*tBoxWidth);

    constexpr Plato::Scalar tTolerance = 1e-4;

    std::string tCriterionName1("volume avg stress pnorm");
    auto tCriterionValue1 = tEllipticProblem.criterionValue(tControls, tCriterionName1);
    TEST_FLOATING_EQUALITY(tCriterionValue1, tStress, tTolerance);

    std::string tCriterionName2("pnorm numerator");
    auto tCriterionValue2 = tEllipticProblem.criterionValue(tControls, tCriterionName2);
    TEST_FLOATING_EQUALITY(tCriterionValue2, tStress * tBoxVolume, tTolerance);

    std::string tCriterionName3("pnorm denominator");
    auto tCriterionValue3 = tEllipticProblem.criterionValue(tControls, tCriterionName3);
    TEST_FLOATING_EQUALITY(tCriterionValue3, tBoxVolume, tTolerance);

    // auto tCriterionGrad = tEllipticProblem.criterionGradient(tControls, tCriterionName);
    // std::vector<Plato::Scalar> tGold = { -8.23158e-01,-2.74211e-01,-2.74205e-01,-2.74211e-01,-5.46915e-01,
    //                                      -1.09598e+00,-5.46915e-01,-1.09091e+00,-1.07737e+00,-5.40880e-01,
    //                                      -1.08590e+00,-5.40880e-01,-1.05793e+00,-5.26599e-01,-1.04844e+00,
    //                                      -5.26599e-01,-1.07226e+00,-5.33831e-01,-1.06304e+00,-5.33831e-01,
    //                                      -5.04852e-01,-1.00493e+00,-5.04852e-01,-1.01433e+00,-5.12007e-01,
    //                                      -1.01919e+00,-1.03386e+00,-5.19301e-01,-1.04332e+00,-5.19301e-01,
    //                                      -5.12007e-01,-1.02878e+00,-4.98050e-01,-1.00060e+00,-4.98050e-01,
    //                                      -9.91065e-01,-9.80656e-01,-4.92349e-01,-9.88215e-01,-4.92349e-01,
    //                                      -2.44243e-01,-7.35025e-01,-2.44243e-01,-2.43315e-01};
    // auto tHostGrad = Kokkos::create_mirror(tCriterionGrad);
    // Kokkos::deep_copy(tHostGrad, tCriterionGrad);
    // TEST_ASSERT( tHostGrad.size() == static_cast<Plato::OrdinalType>(tGold.size() ));
    // for(Plato::OrdinalType tIndex = 0; tIndex < tHostGrad.size(); tIndex++)
    // {
    //     //printf("%12.5e\n", tHostGrad(tIndex));
    //     TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    // }

    // 6. Output Data
    if (false)
    {
        tEllipticProblem.output("VolAvgStressPNormAxial_3D");
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, VolAvgStressPNormShear_3D)
{
    const Plato::OrdinalType tNumElemX = 1;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    const Plato::Scalar tBoxWidth = 5.0;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tBoxWidth, tNumElemX, tBoxWidth, tNumElemY, tBoxWidth, tNumElemZ);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Mechanical'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "      <Parameter name='Iterations' type='int' value='500'/>                              \n"
      "      <Parameter name='Tolerance' type='double' value='1.0e-10'/>                        \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                    \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.1'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e1'/>                  \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
     "   <ParameterList name='volume avg stress pnorm'> \n"
     "     <Parameter name='Type' type='string' value='Division' /> \n"
     "       <Parameter name='Numerator' type='string' value='pnorm numerator' /> \n"
     "       <Parameter name='Denominator' type='string' value='pnorm denominator' /> \n"
     "   </ParameterList> \n"
     "   <ParameterList name='pnorm numerator'> \n"
     "     <Parameter name='Type' type='string' value='Scalar Function' /> \n"
     "     <Parameter name='Scalar Function Type' type='string' value='Stress P-Norm' /> \n"
     "     <ParameterList name='Penalty Function'> \n"
     "       <Parameter name='Type' type='string' value='SIMP' /> \n"
     "       <Parameter name='Exponent' type='double' value='1.0' /> \n"
     "     </ParameterList> \n"
     "     <Parameter name='Function' type='string' value='1.0*x/5.0' /> \n"
     "     <Parameter name='Exponent' type='double' value='1.0' /> \n"
     "   </ParameterList> \n"
     "   <ParameterList name='pnorm denominator'> \n"
     "     <Parameter name='Type' type='string' value='Scalar Function' /> \n"
     "     <Parameter name='Scalar Function Type' type='string' value='vol avg stress p-norm denominator' /> \n"
     "     <ParameterList name='Penalty Function'> \n"
     "       <Parameter name='Type' type='string' value='SIMP' /> \n"
     "       <Parameter name='Exponent' type='double' value='1.0' /> \n"
     "     </ParameterList> \n"
     "     <Parameter name='Function' type='string' value='1.0*x/5.0' /> \n"
     "     <Parameter name='Exponent' type='double' value='1.0' /> \n"
     "   </ParameterList> \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition1'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition1'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='y-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='x+'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='0.1'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='y+'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='0.1'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='z-'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    using PhysicsT = Plato::Mechanics<Plato::Tet4>;

    Plato::Elliptic::Problem<PhysicsT> tEllipticProblem(tMesh, *tParamList, tMachine);
    tEllipticProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->NumNodes();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tEllipticProblem.solution(tControls);


    const Plato::Scalar tSpatialWeightingFactor = 0.5;
    const Plato::Scalar tElasticModulus = 10.0;
    const Plato::Scalar tPoissonsRatio  = 0.1;
    const Plato::Scalar tShearModulus   = tElasticModulus / (2.0 * (1.0 + tPoissonsRatio));
    const Plato::Scalar tStrain = 2.0 * 0.1 / tBoxWidth;
    const Plato::Scalar tStress = tShearModulus * tStrain;
    const Plato::Scalar tBoxVolume = tSpatialWeightingFactor * (tBoxWidth*tBoxWidth*tBoxWidth);

    constexpr Plato::Scalar tTolerance = 1e-4;

    std::string tCriterionName1("volume avg stress pnorm");
    auto tCriterionValue1 = tEllipticProblem.criterionValue(tControls, tCriterionName1);
    TEST_FLOATING_EQUALITY(tCriterionValue1, tStress, tTolerance);

    std::string tCriterionName2("pnorm numerator");
    auto tCriterionValue2 = tEllipticProblem.criterionValue(tControls, tCriterionName2);
    TEST_FLOATING_EQUALITY(tCriterionValue2, tStress * tBoxVolume, tTolerance);

    std::string tCriterionName3("pnorm denominator");
    auto tCriterionValue3 = tEllipticProblem.criterionValue(tControls, tCriterionName3);
    TEST_FLOATING_EQUALITY(tCriterionValue3, tBoxVolume, tTolerance);

    // 6. Output Data
    if (false)
    {
        tEllipticProblem.output("VolAvgStressPNormShear_3D");
    }
}

#if 0
void call_expression_evaluator()
{
    using PhysicsT = Plato::Mechanics<3>;
    using EvalType = typename Plato::Evaluation<PhysicsT>::Residual;
    using StateT  = typename EvalType::StateScalarType;  /*!< state variables automatic differentiation type */
    using ConfigT = typename EvalType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using KinematicsScalarType = typename Plato::fad_type_t<PhysicsT, StateT, ConfigT>; /*!<   strain variables automatic differentiation type */


    const Plato::Scalar tNumCells = 5;
    Plato::ScalarVectorT<EvalType::ControlScalarType> tElementDensity("Gauss point density", tNumCells);
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvalType::SpatialDim>> tCubatureRule =
             std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvalType::SpatialDim>>();
    auto tBasisFunctions = tCubatureRule->getBasisFunctions();
    Plato::InterpolateFromNodal<EvalType::SpatialDim, 1, 0> tInterpolateFromNodal;
    Plato::ScalarMultiVectorT<EvalType::ControlScalarType> tLocalControl("Local Control", tNumCells, 4);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
    {
        for(int i=0; i<4; ++i)
        {
            tLocalControl(aCellOrdinal, i) = 0.5;
        }
    },"Compute Youngs Modulus for each Element");

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
    {
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, tLocalControl, tElementDensity);
    },"Compute Youngs Modulus for each Element");


    Plato::ExpressionEvaluator<Plato::ScalarMultiVectorT<EvalType::ResultScalarType>,
                        Plato::ScalarMultiVectorT<KinematicsScalarType>,
                        Plato::ScalarVectorT<EvalType::ControlScalarType>,
                        Plato::Scalar > tExpEval;

    tExpEval.parse_expression("E0*tRho*tRho*tRho");
    tExpEval.setup_storage(tNumCells, 1);
    tExpEval.set_variable("E0", 1e9);
    Plato::ScalarMultiVectorT<EvalType::ResultScalarType> tElementYoungsModulusValues("Element Youngs Modulus", tNumCells, 1);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
    {
        tExpEval.set_variable("tRho", tElementDensity, aCellOrdinal);
        tExpEval.evaluate_expression( aCellOrdinal, tElementYoungsModulusValues );
    },"Compute Youngs Modulus for each Element");
    Kokkos::fence();
    tExpEval.clear_storage();
}

void call_expression_evaluator_jac()
{
    using PhysicsT = Plato::Thermomechanics<3>;
    using EvalType = typename Plato::Evaluation<PhysicsT>::Jacobian;
    using StateT  = typename EvalType::StateScalarType;
    using ConfigT = typename EvalType::ConfigScalarType; 
    using KinematicsScalarType = typename Plato::fad_type_t<PhysicsT, StateT, ConfigT>; 

    const Plato::Scalar tNumCells = 5;
    Plato::ScalarVectorT<EvalType::ControlScalarType> tElementDensity("Gauss point density", tNumCells);
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvalType::SpatialDim>> tCubatureRule =
             std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvalType::SpatialDim>>();
    auto tBasisFunctions = tCubatureRule->getBasisFunctions();
    Plato::InterpolateFromNodal<EvalType::SpatialDim, 1, 0> tInterpolateFromNodal;
    Plato::ScalarMultiVectorT<EvalType::ControlScalarType> tLocalControl("Local Control", tNumCells, 4);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
    {
        for(int i=0; i<4; ++i)
        {
            tLocalControl(aCellOrdinal, i) = 0.5;
        }
    },"Compute Youngs Modulus for each Element");

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
    {
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, tLocalControl, tElementDensity);
    },"Compute Youngs Modulus for each Element");

    Plato::ExpressionEvaluator<Plato::ScalarMultiVectorT<EvalType::ResultScalarType>,
                        Plato::ScalarMultiVectorT<KinematicsScalarType>,
                        Plato::ScalarVectorT<EvalType::ControlScalarType>,
                        Plato::Scalar > tExpEval;

    tExpEval.parse_expression("E0*tRho*tRho*tRho");
    tExpEval.setup_storage(tNumCells, 1);
    tExpEval.set_variable("E0", 1e9);
    Plato::ScalarMultiVectorT<EvalType::ResultScalarType> tElementYoungsModulusValues("Element Youngs Modulus", tNumCells, 1);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
    {
        tExpEval.set_variable("tRho", tElementDensity, aCellOrdinal);
        tExpEval.evaluate_expression( aCellOrdinal, tElementYoungsModulusValues );
    },"Compute Youngs Modulus for each Element");
    Kokkos::fence();
    tExpEval.clear_storage();
/*
    auto EHost = Kokkos::create_mirror_view( tElementYoungsModulusValues );
    Kokkos::deep_copy(EHost, tElementYoungsModulusValues);
    for(int i=0; i<tNumCells; ++i)
    { 
      std::cout << "EHost: " << EHost(i,0) << std::endl;
    }
*/
}

void eval_some_value()
{
    call_expression_evaluator();
    call_expression_evaluator_jac();
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionTMKinetics)
{
    eval_some_value();
    eval_some_value();
}
#endif
