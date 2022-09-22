/*
 *  PlatoSpatialModelTests.cpp
 *  
 *   Created on: Oct 9, 2020
 **/

#include <iostream>
#include <fstream>

#include "util/PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "Tet4.hpp"
#include "PlatoMask.hpp"
#include "MechanicsElement.hpp"
#include "SpatialModel.hpp"
#include "BLAS1.hpp"
#include "WorksetBase.hpp"

namespace PlatoUnitTests
{


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMask)
{
  // create mesh mask input
  //
  Teuchos::RCP<Teuchos::ParameterList> tMaskParams =
    Teuchos::getParametersFromXmlString(
    "     <ParameterList name='Mask'>                                 \n"
    "       <Parameter name='Mask Type' type='string' value='Brick'/> \n"
    "       <Parameter name='Maximum Z' type='double' value='0.5'/>   \n"
    "     </ParameterList>                                            \n"
  );

  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  Plato::BrickMask<spaceDim> tBrickMask(tMesh, *tMaskParams);

  auto tCellMask = tBrickMask.cellMask();

  auto tCellMask_host = Kokkos::create_mirror_view( tCellMask );
  Kokkos::deep_copy( tCellMask_host, tCellMask );

  std::vector<int> tCellMask_gold = {
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0,
  };

  for (int i=0; i<tCellMask_gold.size(); i++) 
  {
    TEST_ASSERT(tCellMask_gold[i] == tCellMask_host(i));
  }

  auto tInactive = tBrickMask.getInactiveNodes();
  auto tInactive_host = Kokkos::create_mirror_view( tInactive );
  Kokkos::deep_copy( tInactive_host, tInactive );

  std::vector<int> tInactive_gold = { 2, 5, 8, 11, 14, 17, 20, 23, 26 };

  for (int i=0; i<tInactive_gold.size(); i++) 
  {
    TEST_ASSERT(tInactive_gold[i] == tInactive_host(i));
  }

  auto tCellCenters = tBrickMask.getCellCenters(tMesh);
  auto tCellCenters_host = Kokkos::create_mirror_view( tCellCenters );
  Kokkos::deep_copy( tCellCenters_host, tCellCenters );

}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoModel)
{
  // create mesh mask input
  //
  Teuchos::RCP<Teuchos::ParameterList> tMaskParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Layer 1'>                                           \n"
    "  <ParameterList name='Mask'>                                            \n"
    "    <Parameter name='Mask Type' type='string' value='Brick'/>            \n"
    "    <Parameter name='Maximum Z' type='double' value='0.5'/>              \n"
    "  </ParameterList>                                                       \n"
    "</ParameterList>                                                         \n"
  );

  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                     \n"
    "  <ParameterList name='Spatial Model'>                                   \n"
    "    <ParameterList name='Domains'>                                       \n"
    "      <ParameterList name='Design Volume'>                               \n"
    "        <Parameter name='Element Block' type='string' value='body'/>     \n"
    "        <Parameter name='Material Model' type='string' value='matl'/>    \n"
    "      </ParameterList>                                                   \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                       \n"
    "  <ParameterList name='Material Models'>                                 \n"
    "    <ParameterList name='matl'>                                          \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                    \n"
    "        <Parameter name='Mass Density' type='double' value='0.0'/>       \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.36'/>    \n"
    "        <Parameter name='Youngs Modulus' type='double' value='68.0e10'/> \n"
    "      </ParameterList>                                                   \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                       \n"
    "</ParameterList>                                                         \n"
  );

  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tInputParams, tDataMap);

  Plato::MaskFactory<spaceDim> tMaskFactory;
  auto tMask = tMaskFactory.create(tSpatialModel.Mesh, *tMaskParams);

  tSpatialModel.applyMask(tMask);

  auto tOrdinals = tSpatialModel.Domains[0].cellOrdinals();

  auto tOrdinals_host = Kokkos::create_mirror_view( tOrdinals );
  Kokkos::deep_copy( tOrdinals_host, tOrdinals );

  std::vector<int> tOrdinals_gold = {
    0, 1, 2, 3, 4, 5,
    12, 13, 14, 15, 16, 17,
    24, 25, 26, 27, 28, 29,
    36, 37, 38, 39, 40, 41
  };

  for (int i=0; i<tOrdinals_gold.size(); i++) 
  {
    TEST_ASSERT(tOrdinals_gold[i] == tOrdinals_host(i));
  }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, DefaultBlockNotMarkedAsFixed)
{
  // create spatial domain input
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                     \n"
    "  <ParameterList name='Spatial Model'>                                   \n"
    "    <ParameterList name='Domains'>                                       \n"
    "      <ParameterList name='Design Volume'>                               \n"
    "        <Parameter name='Element Block' type='string' value='body'/>     \n"
    "        <Parameter name='Material Model' type='string' value='matl'/>    \n"
    "      </ParameterList>                                                   \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                       \n"
    "  <ParameterList name='Material Models'>                                 \n"
    "    <ParameterList name='matl'>                                          \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                    \n"
    "        <Parameter name='Mass Density' type='double' value='0.0'/>       \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.36'/>    \n"
    "        <Parameter name='Youngs Modulus' type='double' value='68.0e10'/> \n"
    "      </ParameterList>                                                   \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                       \n"
    "</ParameterList>                                                         \n"
  );

  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tInputParams, tDataMap);

  auto tIsFixed = tSpatialModel.Domains[0].isFixedBlock();

  TEST_ASSERT(tIsFixed == false);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, FixedBlockIsMarked)
{
  // create spatial domain input
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                     \n"
    "  <ParameterList name='Spatial Model'>                                   \n"
    "    <ParameterList name='Domains'>                                       \n"
    "      <ParameterList name='Design Volume'>                               \n"
    "        <Parameter name='Element Block' type='string' value='body'/>     \n"
    "        <Parameter name='Material Model' type='string' value='matl'/>    \n"
    "        <Parameter name='Fixed Control' type='bool' value='true'/>       \n"
    "      </ParameterList>                                                   \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                       \n"
    "  <ParameterList name='Material Models'>                                 \n"
    "    <ParameterList name='matl'>                                          \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                    \n"
    "        <Parameter name='Mass Density' type='double' value='0.0'/>       \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.36'/>    \n"
    "        <Parameter name='Youngs Modulus' type='double' value='68.0e10'/> \n"
    "      </ParameterList>                                                   \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                       \n"
    "</ParameterList>                                                         \n"
  );

  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tInputParams, tDataMap);

  auto tIsFixed = tSpatialModel.Domains[0].isFixedBlock();

  TEST_ASSERT(tIsFixed == true);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, DefaultBlockWorksetControlUnchanged)
{
  // create spatial domain input
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                     \n"
    "  <ParameterList name='Spatial Model'>                                   \n"
    "    <ParameterList name='Domains'>                                       \n"
    "      <ParameterList name='Design Volume'>                               \n"
    "        <Parameter name='Element Block' type='string' value='body'/>     \n"
    "        <Parameter name='Material Model' type='string' value='matl'/>    \n"
    "      </ParameterList>                                                   \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                       \n"
    "  <ParameterList name='Material Models'>                                 \n"
    "    <ParameterList name='matl'>                                          \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                    \n"
    "        <Parameter name='Mass Density' type='double' value='0.0'/>       \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.36'/>    \n"
    "        <Parameter name='Youngs Modulus' type='double' value='68.0e10'/> \n"
    "      </ParameterList>                                                   \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                       \n"
    "</ParameterList>                                                         \n"
  );

  // create mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  using ElementType = Plato::MechanicsElement<Plato::Tet4>;

  // create spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tInputParams, tDataMap);

  // create control
  //
  int tNumDofs = spaceDim*tMesh->NumNodes();
  Plato::ScalarVector tControl("control", tNumDofs);
  Plato::blas1::fill(0.4, tControl);

  // workset control
  //
  auto tNumCells = tMesh->NumElements();
  constexpr int tNumNodesPerCell = ElementType::mNumNodesPerCell;

  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
  tWorksetBase.worksetControl(tControl, tControlWS, tSpatialModel.Domains[0]);

  // test workset control
  //
  auto tControlWS_Host = Kokkos::create_mirror_view( tControlWS );
  Kokkos::deep_copy( tControlWS_Host, tControlWS );

  for(int iCell=0; iCell < tNumCells; iCell++){
    for(int iNode=0; iNode < tNumNodesPerCell; iNode++){
      TEST_FLOATING_EQUALITY(tControlWS_Host(iCell,iNode), 0.4, 1e-18);
    }
  }

}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, FixedBlockWorksetControlGivesOnes)
{
  // create spatial domain input
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                     \n"
    "  <ParameterList name='Spatial Model'>                                   \n"
    "    <ParameterList name='Domains'>                                       \n"
    "      <ParameterList name='Design Volume'>                               \n"
    "        <Parameter name='Element Block' type='string' value='body'/>     \n"
    "        <Parameter name='Material Model' type='string' value='matl'/>    \n"
    "        <Parameter name='Fixed Control' type='bool' value='true'/>       \n"
    "      </ParameterList>                                                   \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                       \n"
    "  <ParameterList name='Material Models'>                                 \n"
    "    <ParameterList name='matl'>                                          \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                    \n"
    "        <Parameter name='Mass Density' type='double' value='0.0'/>       \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.36'/>    \n"
    "        <Parameter name='Youngs Modulus' type='double' value='68.0e10'/> \n"
    "      </ParameterList>                                                   \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                       \n"
    "</ParameterList>                                                         \n"
  );

  // create mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  using ElementType = Plato::MechanicsElement<Plato::Tet4>;

  // create spatial model
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tInputParams, tDataMap);

  // create control
  //
  int tNumDofs = spaceDim*tMesh->NumNodes();
  Plato::ScalarVector tControl("control", tNumDofs);
  Plato::blas1::fill(0.4, tControl);

  // workset control
  //
  auto tNumCells = tMesh->NumElements();
  constexpr int tNumNodesPerCell = ElementType::mNumNodesPerCell;

  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
  tWorksetBase.worksetControl(tControl, tControlWS, tSpatialModel.Domains[0]);

  // test workset control
  //
  auto tControlWS_Host = Kokkos::create_mirror_view( tControlWS );
  Kokkos::deep_copy( tControlWS_Host, tControlWS );

  for(int iCell=0; iCell < tNumCells; iCell++){
    for(int iNode=0; iNode < tNumNodesPerCell; iNode++){
      TEST_FLOATING_EQUALITY(tControlWS_Host(iCell,iNode), 1.0, 1e-18);
    }
  }

}

} // namespace PlatoUnitTests
