#include "util/PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "PlatoStaticsTypes.hpp"
#include "ImplicitFunctors.hpp"

#ifdef HAVE_AMGX
//#include "alg/AmgXSparseLinearProblem.hpp"
#endif

#include <sstream>
#include <iostream>
#include <fstream>
#include <type_traits>
#include <Sacado.hpp>

#include "alg/ParallelComm.hpp"

#include "Solutions.hpp"
#include "ScalarProduct.hpp"
#include "WorksetBase.hpp"
#include "elliptic/VectorFunction.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
#include "geometric/GeometryScalarFunction.hpp"
#include "ApplyConstraints.hpp"
#include "elliptic/Problem.hpp"
#include "Thermal.hpp"

#include <fenv.h>

using ordType = typename Plato::ScalarMultiVector::size_type;

/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         ThermostaticResidual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, ThermostaticResidual3D )
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
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='Elliptic'>                                             \n"
    "    <ParameterList name='Penalty Function'>                                   \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                  \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Thermal Conduction'>                               \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='100.0'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Criteria'>                                             \n"
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
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( tMesh->NumNodes(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);


  // create mesh based temperature from host data
  //
  std::vector<Plato::Scalar> t_host( tMesh->NumNodes() );
  Plato::Scalar disp = 0.0, dval = 0.1;
  for( auto& val : t_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    t_host_view(t_host.data(),t_host.size());
  auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), t_host_view);



  // create constraint evaluator
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  using PhysicsType = typename Plato::Thermal<Plato::Tet4>;

  Plato::Elliptic::VectorFunction<PhysicsType>
    tsVectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));


  // compute and test constraint value
  //
  auto residual = tsVectorFunction.value(u,z);

  auto residual_Host = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy( residual_Host, residual );

  std::vector<Plato::Scalar> residual_gold = { 
  -21.66666666666666, -30.00000000000000, -8.333333333333332,
  -25.00000000000000, -45.00000000000001, -19.99999999999999,
  -3.333333333333332, -15.00000000000000, -11.66666666666667,
  -10.00000000000001, -15.00000000000000, -4.999999999999993,
  -5.000000000000004,  0.000000000000000,  4.999999999999980,
   5.000000000000002,  15.00000000000001,  9.99999999999999,
   11.66666666666667,  14.99999999999999,  3.333333333333336,
   20.00000000000000,  45.00000000000005,  25.00000000000000,
   8.333333333333321,  29.99999999999999,  21.66666666666667
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
  auto jacobian = tsVectorFunction.gradient_u(u,z);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
   49.9999999999999858, -16.6666666666666643, -16.6666666666666643, 0,
  -16.6666666666666643, 0, 0, 0, -16.6666666666666643,
   83.3333333333333002, -16.6666666666666643, -24.9999999999999964, 0,
  -24.9999999999999964, 0, 0, 0, -16.6666666666666643,
   33.3333333333333286, -8.33333333333333215, -8.33333333333333215, 0
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }


  // compute and test gradient wrt control, z
  //
  auto gradient = tsVectorFunction.gradient_z(u,z);
  
  auto grad_entries = gradient->entries();
  auto grad_entriesHost = Kokkos::create_mirror_view( grad_entries );
  Kokkos::deep_copy(grad_entriesHost, grad_entries);

  std::vector<Plato::Scalar> gold_grad_entries = {
   -5.41666666666666607, -2.08333333333333304, -0.833333333333333259,
   -2.91666666666666696,  2.91666666666666563,  0.833333333333333037,
    2.08333333333333304,  5.41666666666666785, -0.416666666666666630,
   -7.50000000000000000, -2.08333333333333304, -2.08333333333333348,
   -2.91666666666666563,  4.16666666666666607,  0.833333333333334370,
    4.58333333333333304,  5.41666666666666519, -0.416666666666666741,
   -2.08333333333333304, -1.24999999999999956,  1.25000000000000044,
    2.49999999999999911
  };

  int grad_entriesSize = gold_grad_entries.size();
  for(int i=0; i<grad_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_entriesHost(i), gold_grad_entries[i], 1.0e-14);
  }

  // compute and test gradient wrt node position, x
  //
  auto gradient_x = tsVectorFunction.gradient_x(u,z);
  
  auto grad_x_entries = gradient_x->entries();
  auto grad_x_entriesHost = Kokkos::create_mirror_view( grad_x_entries );
  Kokkos::deep_copy(grad_x_entriesHost, grad_x_entries);

  std::vector<Plato::Scalar> gold_grad_x_entries = {
   -90.0000000000000000, -29.9999999999999929, -10.0000000000000000,
    28.3333333333333357,  8.33333333333332860,  23.3333333333333321,
    25.0000000000000000,  26.6666666666666679, -1.66666666666667052,
   -6.66666666666666696,  15.0000000000000018,  15.0000000000000018
};

  int grad_x_entriesSize = gold_grad_x_entries.size();
  for(int i=0; i<grad_x_entriesSize; i++){
    if(fabs(gold_grad_x_entries[i]) < 1e-10){
      TEST_ASSERT(fabs(grad_x_entriesHost[i]) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(grad_x_entriesHost(i), gold_grad_x_entries[i], 1.0e-13);
    }
  }


}


/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalThermalEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, InternalThermalEnergy3D )
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
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Internal Thermal Energy'>                            \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Thermal Energy'/>  \n"
    "      <ParameterList name='Penalty Function'>                                 \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Thermal Conduction'>                               \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='100.0'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
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
  ordType tNumDofs = tMesh->NumNodes();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, tNumDofs);
  auto u = Kokkos::subview(U, 0, Kokkos::ALL());
  auto u_host = Kokkos::create_mirror_view( u );
  Plato::Scalar disp = 0.0, dval = 0.1;
  for(ordType i=0; i<tNumDofs; i++)
  {
      u_host(i) = (disp += dval);
  }
  Kokkos::deep_copy(u, u_host);


  // create objective
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  std::string tMyFunction("Internal Thermal Energy");
  Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermal<Plato::Tet4>>
    eeScalarFunction(tSpatialModel, tDataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto value = eeScalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 363.999999999999829;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt state, u
  //
  auto grad_u = eeScalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
  -43.33333333333333, -60.00000000000000, -16.66666666666666,
  -49.99999999999999, -90.00000000000001, -39.99999999999999,
  -6.666666666666664, -30.00000000000000, -23.33333333333334,
  -20.00000000000001, -29.99999999999999, -9.99999999999999,
  -10.00000000000001,  0.000000000000000,  9.99999999999996,
   10.00000000000000,  30.00000000000003,  19.99999999999998,
   23.33333333333334,  29.99999999999996,  6.666666666666671,
   39.99999999999999,  90.00000000000009,  50.00000000000000,
   16.66666666666664,  59.99999999999997,  43.33333333333335
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
  auto grad_z = eeScalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
  11.37500000000000, 15.16666666666666, 3.791666666666666,
  15.16666666666667, 22.74999999999999, 7.583333333333331,
  3.791666666666667, 7.583333333333333, 3.791666666666666,
  15.16666666666667, 22.75000000000000, 7.583333333333334,
  22.75000000000000, 45.50000000000001, 22.75000000000000,
  7.583333333333332, 22.75000000000000, 15.16666666666667,
  3.791666666666667, 7.583333333333337, 3.791666666666667,
  7.583333333333334, 22.75000000000001, 15.16666666666667,
  3.791666666666666, 15.16666666666667, 11.37500000000001
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
  47.66666666666666, -4.333333333333325, -21.66666666666667,
  62.50000000000003, -9.500000000000027,  12.00000000000000,
  14.83333333333334, -5.166666666666668,  33.66666666666667,
  44.50000000000000,  29.99999999999999, -35.50000000000000,
  71.00000000000001,  54.00000000000001,  18.00000000000001,
  26.49999999999999,  24.00000000000000,  53.50000000000001,
 -3.166666666666664,  34.33333333333334, -13.83333333333333,
  8.499999999999988,  63.50000000000001,  5.999999999999993,
  11.66666666666666,  29.16666666666667,  19.83333333333333,
  36.00000000000003, -33.49999999999999, -41.50000000000001,
  53.99999999999994, -73.00000000000003,  6.000000000000016,
  17.99999999999999, -39.50000000000001,  47.50000000000001
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
  }
}



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalElasticEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, FluxPNorm3D )
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
    "    <ParameterList name='Flux P-Norm'>                                        \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Flux P-Norm'/>  \n"
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
    "      <ParameterList name='Thermal Conduction'>                               \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='100.0'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
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
  ordType tNumDofs = tMesh->NumNodes();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, tNumDofs);
  auto u = Kokkos::subview(U, 0, Kokkos::ALL());
  auto u_host = Kokkos::create_mirror_view( u );
  Plato::Scalar disp = 0.0, dval = 0.1;
  for(ordType i=0; i<tNumDofs; i++)
  {
      u_host(i) = (disp += dval);
  }
  Kokkos::deep_copy(u, u_host);



  // create objective
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  std::string tMyFunction("Flux P-Norm");
  Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermal<Plato::Tet4>>
    scalarFunction(tSpatialModel, tDataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto value = scalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 190.7878402833891;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt state, u
  //
  tSolution.set("State", U);
  auto grad_u = scalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
  -11.35641906448748, -15.72427255082882, -4.367853486341336,
  -13.10356045902401, -23.58640882624325, -10.48284836721919,
  -1.747141394536534, -7.862136275414418, -6.114994880877870,
  -5.241424183609619, -7.862136275414396, -2.620712091804797,
  -2.620712091804800,  0.000000000000000,  2.620712091804749,
   2.620712091804805,  7.862136275414446,  5.241424183609591,
   6.114994880877881,  7.862136275414394,  1.747141394536531,
   10.48284836721922,  23.58640882624324,  13.10356045902404,
   4.367853486341329,  15.72427255082879,  11.35641906448748
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
  auto grad_z = scalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
   5.962120008855925, 7.949493345141233, 1.987373336285307,
   7.949493345141235, 11.92424001771185, 3.974746672570610,
   1.987373336285309, 3.974746672570620, 1.987373336285307,
   7.949493345141238, 11.92424001771185, 3.974746672570618,
   11.92424001771185, 23.84848003542371, 11.92424001771185,
   3.974746672570611, 11.92424001771185, 7.949493345141238,
   1.987373336285311, 3.974746672570616, 1.987373336285307,
   3.974746672570619, 11.92424001771186, 7.949493345141239,
   1.987373336285307, 7.949493345141225, 5.962120008855926
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  auto grad_x = scalarFunction.gradient_x(tSolution, z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
   19.11663875855393,  5.488935881168949,   0.9463682553739569,
   26.31631725520657,  7.447190194211985,   3.144854510165766,
   7.199678496652637,  1.958254313043032,   2.198486254791805,
   21.59903548995792,  7.862136275414407,   0.6333387555194938,
   38.48078921466723,  14.15184529574595,   4.717281765248649,
   16.88175372470923,  6.289709020331511,   4.083943009729144,
   2.482396731403990,  2.373200394245461,  -0.3130294998544632,
   12.16447195946064,  6.704655101533961,   1.572427255082882,
   9.682075228056627,  4.331454707288491,   1.885456754937344,
   9.434563530497316,  1.157481173880465,  -0.9390884995633852,
   14.15184529574592,  0.7425350926780239,  1.572427255082881,
   4.717281765248634, -0.4149460812024298,  2.511515754646270,
   4.717281765248646,  1.572427255082882,  -3.450604254209653,
   0.000000000000000,  0.000000000000000,   0.0000000000000000,
  -4.717281765248543, -1.572427255082847,   3.450604254209671,
  -4.717281765248648,  0.4149460812024223, -2.511515754646267,
  -14.15184529574600, -0.7425350926780522, -1.572427255082890,
  -9.434563530497268, -1.157481173880446,   0.9390884995633906,
  -9.682075228056645, -4.331454707288501,  -1.885456754937346,
  -12.16447195946060, -6.704655101533948,  -1.572427255082879,
  -2.482396731403986, -2.373200394245457,   0.3130294998544623,
  -16.88175372470928, -6.289709020331526,  -4.083943009729155,
  -38.48078921466723, -14.15184529574595,  -4.717281765248650,
  -21.59903548995798, -7.862136275414430,  -0.6333387555194997,
  -7.199678496652624, -1.958254313043028,  -2.198486254791804,
  -26.31631725520652, -7.447190194211965,  -3.144854510165760,
  -19.11663875855394, -5.488935881168952,  -0.9463682553739576
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    if(grad_x_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_x_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
    }
  }
}
