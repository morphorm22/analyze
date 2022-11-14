#include <fstream>
#include <string>
#include <sstream>

#include "ParseTools.hpp"
#include "SpatialModel.hpp"
#include "PlatoUtilities.hpp"
#include "InputDataUtils.hpp"
#include "util/PlatoTestHelpers.hpp"
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

namespace PlatoUnitTests
{

/******************************************************************************/
/*!
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, getFileExtensionV)
{
  TEST_ASSERT(Plato::getFileExtension("data.csv") == "csv");
  TEST_ASSERT(Plato::getFileExtension("data.CSV") == "CSV");
  TEST_ASSERT(Plato::getFileExtension("data.10.exo") == "exo");
}

/******************************************************************************/
/*!
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, trim)
{
  TEST_ASSERT(Plato::trim(" ") == "");
  TEST_ASSERT(Plato::trim("  ") == "");
  TEST_ASSERT(Plato::trim(" A") == "A");
  TEST_ASSERT(Plato::trim(" AB") == "AB");
  TEST_ASSERT(Plato::trim("some label") == "some label");
  TEST_ASSERT(Plato::trim(" some_label") == "some_label");
  TEST_ASSERT(Plato::trim(" some label  ") == "some label");
}

/******************************************************************************/
/*!
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, split)
{
  auto tTokens = Plato::split(" A, B, C D , E ", ',');
  TEST_ASSERT(Plato::trim(tTokens[0]) == "A");
  TEST_ASSERT(Plato::trim(tTokens[1]) == "B");
  TEST_ASSERT(Plato::trim(tTokens[2]) == "C D");
  TEST_ASSERT(Plato::trim(tTokens[3]) == "E");
}

/******************************************************************************/
/*!
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, readCSV)
{

  std::ofstream tOutfile;
  tOutfile.open("readCSV_test.csv");
  tOutfile << "# ElemID, Data1, Data2, Data3" << std::endl;
  tOutfile << "0, 1e5, 1.234e6, 5.432E4" << std::endl;
  tOutfile << "1, 1e-5, 2.12876, -0.2344" << std::endl;
  tOutfile << "2, 2e+5, 2.12876, 0.2344e-4" << std::endl;
  tOutfile << "3, 3e-6, 2.12876, 0.2344" << std::endl;
  tOutfile << "4, 4e-5, 2.12876, 0.2344" << std::endl;
  tOutfile << "5, 5e-15, -02.12876, 10.2344" << std::endl;
  tOutfile.close();

  constexpr int meshWidth=1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <ParameterList name='Input Data'>                                         \n"
    "    <ParameterList name='Basis'>                                            \n"
    "      <Parameter name='Input File' type='string' value='readCSV_test.csv'/> \n"
    "      <Parameter name='Centering' type='string' value='Element'/>           \n"
    "      <Parameter  name='Columns' type='Array(string)' value='{index, D0, D1, D2}'/> \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Spatial Model'>                                      \n"
    "    <ParameterList name='Basis'>                                            \n"
    "      <Parameter name='X0' type='string' value='X0'/>                       \n"
    "      <Parameter name='X1' type='string' value='X1'/>                       \n"
    "      <Parameter name='X2' type='string' value='X2'/>                       \n"
    "      <Parameter name='Y0' type='string' value='Y0'/>                       \n"
    "      <Parameter name='Y1' type='string' value='Y1'/>                       \n"
    "      <Parameter name='Y2' type='string' value='Y2'/>                       \n"
    "    </ParameterList>                                                        \n"
    "    <ParameterList name='Domains'>                                          \n"
    "      <ParameterList name='Design Volume'>                                  \n"
    "        <Parameter name='Element Block' type='string' value='body'/>        \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Material Models'>                                    \n"
    "    <ParameterList name='Unobtainium'>                                      \n"
    "      <ParameterList name='Thermoelastic'>                                  \n"
    "        <Parameter name='Reference Temperature' type='double' value='1e-2'/>\n"
    "        <Parameter name='Thermal Conductivity' type='double' value='910.0'/>\n"
    "        <Parameter name='Thermal Expansivity' type='double' value='1.0e-5'/>\n"
    "        <ParameterList name='Elastic Stiffness'>                            \n"
    "          <!-- see orthotropic_stiffness.nb                                 \n"
    "             Constants below correspond to:                                 \n"
    "             nu_xy=0.30, nu_yz=0.33, nu_xz=0.27,                            \n"
    "             E_x=1e11,   E_y=2e11,   E_z=4e11,                              \n"
    "             G_xy=1e10,  G_yz=2e10,  G_xz=3e10                              \n"
    "           -->                                                              \n"
    "          <Parameter name='c11' type='double' value='1.32066e11'/>          \n"
    "          <Parameter name='c22' type='double' value='2.74802e11'/>          \n"
    "          <Parameter name='c33' type='double' value='5.39467e11'/>          \n"
    "          <Parameter name='c12' type='double' value='5.76667e10'/>          \n"
    "          <Parameter name='c13' type='double' value='5.46877e10'/>          \n"
    "          <Parameter name='c23' type='double' value='1.21825e11'/>          \n"
    "          <Parameter name='c44' type='double' value='2.0e10'/>              \n"
    "          <Parameter name='c55' type='double' value='3.0e10'/>              \n"
    "          <Parameter name='c66' type='double' value='1.0e10'/>              \n"
    "        </ParameterList>                                                    \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );


  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  Plato::readInputData(*tParamList, tDataMap, tMesh);

  auto tD0 = tDataMap.scalarVectors["D0"];
  auto tD0_Host = Kokkos::create_mirror_view(tD0);
  Kokkos::deep_copy(tD0_Host, tD0);

  TEST_FLOATING_EQUALITY( tD0_Host(0), 1e5, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tD0_Host(1), 1e-5, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tD0_Host(2), 2e+5, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tD0_Host(3), 3e-6, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tD0_Host(4), 4e-5, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tD0_Host(5), 5e-15, DBL_EPSILON);

  auto tD1 = tDataMap.scalarVectors["D1"];
  auto tD1_Host = Kokkos::create_mirror_view(tD1);
  Kokkos::deep_copy(tD1_Host, tD1);

  TEST_FLOATING_EQUALITY( tD1_Host(0), 1.234e6, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tD1_Host(1), 2.12876, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tD1_Host(2), 2.12876, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tD1_Host(3), 2.12876, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tD1_Host(4), 2.12876, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tD1_Host(5), -2.12876, DBL_EPSILON);

  auto tD2 = tDataMap.scalarVectors["D2"];
  auto tD2_Host = Kokkos::create_mirror_view(tD2);
  Kokkos::deep_copy(tD2_Host, tD2);

  TEST_FLOATING_EQUALITY( tD2_Host(0), 5.432e4, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tD2_Host(1), -0.2344, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tD2_Host(2), 0.2344e-4, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tD2_Host(3), 0.2344, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tD2_Host(4), 0.2344, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tD2_Host(5), 10.2344, DBL_EPSILON);

}

/******************************************************************************/
/*!
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, uniformSpatialBasis)
{

  std::ofstream tOutfile;
  tOutfile.open("uniformSpatialBasis_test.csv");
  tOutfile << "# ElemID, X0, X1, X2, Y0, Y1, Y2, Z0, Z1, Z2" << std::endl;
  tOutfile << "0, 1, 0, 0, 0, 1, 0, 0, 0, 1" << std::endl;
  tOutfile << "1, 1, 0, 0, 0, 1, 0, 0, 0, 1" << std::endl;
  tOutfile << "2, 1, 0, 0, 0, 1, 0, 0, 0, 1" << std::endl;
  tOutfile << "3, 1, 0, 0, 0, 1, 0, 0, 0, 1" << std::endl;
  tOutfile << "4, 1, 0, 0, 0, 1, 0, 0, 0, 1" << std::endl;
  tOutfile << "5, 1, 0, 0, 0, 1, 0, 0, 0, 1" << std::endl;
  tOutfile.close();

  constexpr int meshWidth=1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <ParameterList name='Input Data'>                                         \n"
    "    <ParameterList name='Basis'>                                            \n"
    "      <Parameter name='Input File' type='string' value='uniformSpatialBasis_test.csv'/> \n"
    "      <Parameter name='Centering' type='string' value='Element'/>           \n"
    "      <Parameter name='Columns' type='Array(string)' value='{index, X0, X1, X2, Y0, Y1, Y2, Z0, Z1, Z2}'/> \n"
    "      <ParameterList name='Transform'>                                      \n"
    "        <Parameter name='Type' type='string' value='Orthonormal Tensor'/>   \n"
    "        <Parameter name='Field Name' type='string' value='Element Bases'/>  \n"
    "        <Parameter  name='X' type='Array(string)' value='{X0, X1, X2}'/>    \n"
    "        <Parameter  name='Y' type='Array(string)' value='{Y0, Y1, Y2}'/>    \n"
    "        <Parameter  name='Z' type='Array(string)' value='{Z0, Z1, Z2}'/>    \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Spatial Model'>                                      \n"
    "    <ParameterList name='Domains'>                                          \n"
    "      <ParameterList name='Design Volume'>                                  \n"
    "        <Parameter name='Basis Field' type='string' value='Element Bases'/> \n"
    "        <Parameter name='Element Block' type='string' value='body'/>        \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Material Models'>                                    \n"
    "    <ParameterList name='Unobtainium'>                                      \n"
    "      <ParameterList name='Thermoelastic'>                                  \n"
    "        <Parameter name='Reference Temperature' type='double' value='1e-2'/>\n"
    "        <Parameter name='Thermal Conductivity' type='double' value='910.0'/>\n"
    "        <Parameter name='Thermal Expansivity' type='double' value='1.0e-5'/>\n"
    "        <ParameterList name='Elastic Stiffness'>                            \n"
    "          <!-- see orthotropic_stiffness.nb                                 \n"
    "             Constants below correspond to:                                 \n"
    "             nu_xy=0.30, nu_yz=0.33, nu_xz=0.27,                            \n"
    "             E_x=1e11,   E_y=2e11,   E_z=4e11,                              \n"
    "             G_xy=1e10,  G_yz=2e10,  G_xz=3e10                              \n"
    "           -->                                                              \n"
    "          <Parameter name='c11' type='double' value='1.32066e11'/>          \n"
    "          <Parameter name='c22' type='double' value='2.74802e11'/>          \n"
    "          <Parameter name='c33' type='double' value='5.39467e11'/>          \n"
    "          <Parameter name='c12' type='double' value='5.76667e10'/>          \n"
    "          <Parameter name='c13' type='double' value='5.46877e10'/>          \n"
    "          <Parameter name='c23' type='double' value='1.21825e11'/>          \n"
    "          <Parameter name='c44' type='double' value='2.0e10'/>              \n"
    "          <Parameter name='c55' type='double' value='3.0e10'/>              \n"
    "          <Parameter name='c66' type='double' value='1.0e10'/>              \n"
    "        </ParameterList>                                                    \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );


  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  Plato::readInputData(*tParamList, tDataMap, tMesh);

  auto tX0 = tDataMap.scalarVectors["X0"];
  auto tX0_Host = Kokkos::create_mirror_view(tX0);
  Kokkos::deep_copy(tX0_Host, tX0);

  TEST_FLOATING_EQUALITY( tX0_Host(0), 1, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tX0_Host(1), 1, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tX0_Host(2), 1, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tX0_Host(3), 1, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tX0_Host(4), 1, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tX0_Host(5), 1, DBL_EPSILON);

  auto tBasis = tDataMap.scalarArray3Ds["Element Bases"];
  auto tBasis_Host = Kokkos::create_mirror_view(tBasis);
  Kokkos::deep_copy(tBasis_Host, tBasis);

  TEST_FLOATING_EQUALITY( tBasis_Host(0,0,0), 1, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(0,1,1), 1, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(0,2,2), 1, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(0,1,2), 0, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(0,0,2), 0, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(0,0,1), 0, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(0,2,1), 0, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(0,2,0), 0, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(0,1,0), 0, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(1,0,0), 1, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(1,1,1), 1, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(1,2,2), 1, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(1,1,2), 0, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(1,0,2), 0, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(1,0,1), 0, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(1,2,1), 0, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(1,2,0), 0, DBL_EPSILON);
  TEST_FLOATING_EQUALITY( tBasis_Host(1,1,0), 0, DBL_EPSILON);
}
}
