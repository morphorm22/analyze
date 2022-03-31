/*
 * EngineMeshTests.cpp
 *
 *  Created on: Nov 17, 2021
 */


#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "EngineMesh.hpp"
#include "EngineMeshIO.hpp"

#include <string>

std::vector<Plato::Scalar>
getSetProjection(
    const Plato::EngineMesh  & aMesh,
          std::string          aSetName,
          Plato::OrdinalType   aDim,
          Plato::Scalar        aValue
)
{
    std::vector<Plato::Scalar> tReturn;

    const Plato::OrdinalType cSpaceDim = 3;

    auto tNodeOrds = aMesh.GetNodeSetNodes(aSetName);
    auto tCoordinates = aMesh.Coordinates();

    auto tHostCoords = Kokkos::create_mirror_view(tCoordinates);
    Kokkos::deep_copy(tHostCoords, tCoordinates);

    auto tHostData = Kokkos::create_mirror_view(tNodeOrds);
    Kokkos::deep_copy(tHostData, tNodeOrds);

    for( int i=0; i<tHostData.size(); i++)
    {
        tReturn.push_back(tHostCoords(cSpaceDim*tHostData(i) + aDim) - aValue);
    }
    return tReturn;
}

template <typename T>
bool are_equal(T aVector1, T aVector2)
{
    if( aVector1.size() != aVector2.size() )
    {
        return false;
    }

    auto tHostVector1 = Kokkos::create_mirror_view(aVector1);
    Kokkos::deep_copy(tHostVector1, aVector1);

    auto tHostVector2 = Kokkos::create_mirror_view(aVector2);
    Kokkos::deep_copy(tHostVector2, aVector2);

    bool tIsSame = true;
    auto tNumEntries = aVector1.size();
    for( decltype(tNumEntries) iEntry=0; iEntry<tNumEntries; iEntry++ )
    {
        tIsSame = tIsSame && (tHostVector1(iEntry) == tHostVector2(iEntry) );
    }

    return tIsSame;
}

const Plato::Scalar cTol = 1e-9;

TEUCHOS_UNIT_TEST(EngineMeshIntxTests, CreateSurfaceMesh_Tet4)
{
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", /*tMeshIntervals=*/ 2);

    std::vector<std::string> tExcludeNames;

    auto tSideSetElements = tMesh->GetSideSetElementsComplement(tExcludeNames);
    {
        auto tHost = Kokkos::create_mirror_view(tSideSetElements);
        Kokkos::deep_copy(tHost, tSideSetElements);

        std::vector<Plato::OrdinalType> tGold = {
          2,   3,  1,  0,  4,  5,  8,  9,  7, 10,  8,  9, 14, 13, 12, 17,
          20, 19, 20, 21, 13, 12, 19, 18, 27, 24, 28, 29, 33, 34, 32, 33,
          36, 41, 44, 45, 37, 36, 43, 42, 28, 29, 34, 35, 40, 41, 46, 47
        };

        for(unsigned int iVal=0; iVal<tGold.size(); iVal++)
        {
            TEST_ASSERT(tHost[iVal] == tGold[iVal]);
        } 
    }

    auto tSideSetFaces = tMesh->GetSideSetFacesComplement(tExcludeNames);
    {
        auto tHost = Kokkos::create_mirror_view(tSideSetFaces);
        Kokkos::deep_copy(tHost, tSideSetFaces);

        std::vector<Plato::OrdinalType> tGold = {
          3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3,
          3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1,
          3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        };

        for(unsigned int iVal=0; iVal<tGold.size(); iVal++)
        {
            TEST_ASSERT(tHost[iVal] == tGold[iVal]);
        } 
    }
}

TEUCHOS_UNIT_TEST(EngineMeshIntxTests, CreateSurfaceComplement_Tet4)
{
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", /*tMeshIntervals=*/ 2);

    std::vector<std::string> tExcludeNames;
    tExcludeNames.push_back("z+");

    auto tSideSetElements = tMesh->GetSideSetElementsComplement(tExcludeNames);
    {
        auto tHost = Kokkos::create_mirror_view(tSideSetElements);
        Kokkos::deep_copy(tHost, tSideSetElements);

        std::vector<Plato::OrdinalType> tGold = {
           0,  1,  2,  3,  4,  5,  7,  8,  9, 10, 12, 12, 13, 13, 14, 17,
          18, 19, 19, 20, 24, 27, 28, 28, 29, 29, 33, 34, 34, 35, 36, 36,
          37, 40, 41, 41, 42, 43, 46, 47
        };

        for(unsigned int iVal=0; iVal<tGold.size(); iVal++)
        {
            TEST_ASSERT(tHost[iVal] == tGold[iVal]);
        } 
    }

    auto tSideSetFaces = tMesh->GetSideSetFacesComplement(tExcludeNames);
    {
        auto tHost = Kokkos::create_mirror_view(tSideSetFaces);
        Kokkos::deep_copy(tHost, tSideSetFaces);

        std::vector<Plato::OrdinalType> tGold = {
          3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3,
          1, 1, 3, 3, 3, 3, 1, 3, 1, 3, 3, 1, 3, 1, 1, 3,
          1, 1, 1, 3, 1, 1, 1, 1
        };

        for(unsigned int iVal=0; iVal<tGold.size(); iVal++)
        {
            TEST_ASSERT(tHost[iVal] == tGold[iVal]);
        } 
    }
}

TEUCHOS_UNIT_TEST(EngineMeshIntxTests, CreateSurfaceComplement_Hex8)
{
    auto tMesh = PlatoUtestHelpers::getBoxMesh("HEX8", /*tMeshIntervals=*/ 2);

    std::vector<std::string> tExcludeNames;
    tExcludeNames.push_back("z+");

    auto tSideSetElements = tMesh->GetSideSetElementsComplement(tExcludeNames);
    {
        auto tHost = Kokkos::create_mirror_view(tSideSetElements);
        Kokkos::deep_copy(tHost, tSideSetElements);

        std::vector<Plato::OrdinalType> tGold = {
          0, 0, 0, 1, 1, 2, 2, 2, 3, 3,
          4, 4, 4, 5, 5, 6, 6, 6, 7, 7
        };

        for(unsigned int iVal=0; iVal<tGold.size(); iVal++)
        {
            TEST_ASSERT(tHost[iVal] == tGold[iVal]);
        } 
    }

    auto tSideSetFaces = tMesh->GetSideSetFacesComplement(tExcludeNames);
    {
        auto tHost = Kokkos::create_mirror_view(tSideSetFaces);
        Kokkos::deep_copy(tHost, tSideSetFaces);

        std::vector<Plato::OrdinalType> tGold = {
          0, 3, 4, 0, 3, 2, 3, 4, 2, 3,
          0, 1, 4, 0, 1, 1, 2, 4, 1, 2
        };

        for(unsigned int iVal=0; iVal<tGold.size(); iVal++)
        {
            TEST_ASSERT(tHost[iVal] == tGold[iVal]);
        } 
    }
}

TEUCHOS_UNIT_TEST(EngineMeshIntxTests, CreateSurfaceMesh_Hex8)
{
    auto tMesh = PlatoUtestHelpers::getBoxMesh("HEX8", /*tMeshIntervals=*/ 2);

    std::vector<std::string> tExcludeNames;

    auto tSideSetElements = tMesh->GetSideSetElementsComplement(tExcludeNames);
    {
        auto tHost = Kokkos::create_mirror_view(tSideSetElements);
        Kokkos::deep_copy(tHost, tSideSetElements);

        std::vector<Plato::OrdinalType> tGold = {
          0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 2, 3,
          4, 4, 5, 5, 6, 7, 6, 7, 4, 5, 6, 7
        };

        for(unsigned int iVal=0; iVal<tGold.size(); iVal++)
        {
            TEST_ASSERT(tHost[iVal] == tGold[iVal]);
        } 
    }

    auto tSideSetFaces = tMesh->GetSideSetFacesComplement(tExcludeNames);
    {
        auto tHost = Kokkos::create_mirror_view(tSideSetFaces);
        Kokkos::deep_copy(tHost, tSideSetFaces);

        std::vector<Plato::OrdinalType> tGold = {
          3, 0, 4, 3, 0, 5, 3, 4, 3, 5, 2, 2,
          0, 4, 0, 5, 4, 5, 2, 2, 1, 1, 1, 1
        };

        for(unsigned int iVal=0; iVal<tGold.size(); iVal++)
        {
            TEST_ASSERT(tHost[iVal] == tGold[iVal]);
        } 
    }
}

TEUCHOS_UNIT_TEST(EngineMeshIntxTests, ReadTet4Mesh)
{
    std::string tFileName = "unit_cube_tet4.exo";
    Plato::EngineMesh tMesh(tFileName);

    TEST_ASSERT(tMesh.NumNodes() == 71)
    TEST_ASSERT(tMesh.NumElements() == 224)
    TEST_ASSERT(tMesh.NumDimensions() == 3)
    TEST_ASSERT(tMesh.NumNodesPerElement() == 4)

    // verify node locations on x-
    auto tVals = getSetProjection(tMesh, "x-", 0, -0.5);
    for( auto tVal : tVals ) TEST_FLOATING_EQUALITY(fabs(tVal), 0.0, cTol);

    // verify node locations on x+
    tVals = getSetProjection(tMesh, "x+", 0, 0.5);
    for( auto tVal : tVals ) TEST_FLOATING_EQUALITY(fabs(tVal), 0.0, cTol);

    // verify node locations on y+
    tVals = getSetProjection(tMesh, "y+", 1, 0.5);
    for( auto tVal : tVals ) TEST_FLOATING_EQUALITY(fabs(tVal), 0.0, cTol);

    // mesh has a single block names 'block_1'.  verify.
    auto tBlockNames = tMesh.GetElementBlockNames();
    TEST_ASSERT(tBlockNames.size() == 1);
    TEST_ASSERT(tBlockNames[0] == "block_1");

    // mesh has a single block with sequential element ids.  verify.
    auto tLocalElementIDs = tMesh.GetLocalElementIDs(tBlockNames[0]);
    auto tHostData = Kokkos::create_mirror_view(tLocalElementIDs);
    Kokkos::deep_copy(tHostData, tLocalElementIDs);
    for(Plato::OrdinalType iElem=0; iElem<tMesh.NumElements(); iElem++)
    {
        TEST_ASSERT(tHostData(iElem) == iElem);
    }

    // verify that nodesets are accounted for.
    auto tNodeSetNames = tMesh.GetNodeSetNames();
    TEST_ASSERT(tNodeSetNames.size() == 6);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "y-") == 1);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "y+") == 1);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "z-") == 1);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "z+") == 1);

    // verify that sidesets are accounted for.
    auto tSideSetNames = tMesh.GetSideSetNames();
    TEST_ASSERT(tSideSetNames.size() == 6);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "x-") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "x+") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "y-") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "y+") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "z-") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "z+") == 1);
}

TEUCHOS_UNIT_TEST(EngineMeshIntxTests, ReadTet10Mesh)
{
    std::string tFileName = "unit_cube_tet10.exo";
    Plato::EngineMesh tMesh(tFileName);

    TEST_ASSERT(tMesh.NumNodes() == 413)
    TEST_ASSERT(tMesh.NumElements() == 224)
    TEST_ASSERT(tMesh.NumDimensions() == 3)
    TEST_ASSERT(tMesh.NumNodesPerElement() == 10)

    // verify node locations on x-
    auto tVals = getSetProjection(tMesh, "x-", 0, -0.5);
    for( auto tVal : tVals ) TEST_FLOATING_EQUALITY(fabs(tVal), 0.0, cTol);

    // verify node locations on x+
    tVals = getSetProjection(tMesh, "x+", 0, 0.5);
    for( auto tVal : tVals ) TEST_FLOATING_EQUALITY(fabs(tVal), 0.0, cTol);

    // verify node locations on y+
    tVals = getSetProjection(tMesh, "y+", 1, 0.5);
    for( auto tVal : tVals ) TEST_FLOATING_EQUALITY(fabs(tVal), 0.0, cTol);

    // mesh has a single block names 'block_1'.  verify.
    auto tBlockNames = tMesh.GetElementBlockNames();
    TEST_ASSERT(tBlockNames.size() == 1);
    TEST_ASSERT(tBlockNames[0] == "block_1");

    // mesh has a single block with sequential element ids.  verify.
    auto tLocalElementIDs = tMesh.GetLocalElementIDs(tBlockNames[0]);
    auto tHostData = Kokkos::create_mirror_view(tLocalElementIDs);
    Kokkos::deep_copy(tHostData, tLocalElementIDs);
    for(Plato::OrdinalType iElem=0; iElem<tMesh.NumElements(); iElem++)
    {
        TEST_ASSERT(tHostData(iElem) == iElem);
    }

    // verify that nodesets are accounted for.
    auto tNodeSetNames = tMesh.GetNodeSetNames();
    TEST_ASSERT(tNodeSetNames.size() == 6);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "y-") == 1);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "y+") == 1);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "z-") == 1);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "z+") == 1);

    // verify that sidesets are accounted for.
    auto tSideSetNames = tMesh.GetSideSetNames();
    TEST_ASSERT(tSideSetNames.size() == 6);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "x-") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "x+") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "y-") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "y+") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "z-") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "z+") == 1);
}

TEUCHOS_UNIT_TEST(EngineMeshIntxTests, ReadHex8Mesh)
{
    std::string tFileName = "unit_cube_hex8.exo";
    Plato::EngineMesh tMesh(tFileName);

    TEST_ASSERT(tMesh.NumNodes() == 27)
    TEST_ASSERT(tMesh.NumElements() == 8)
    TEST_ASSERT(tMesh.NumDimensions() == 3)
    TEST_ASSERT(tMesh.NumNodesPerElement() == 8)

    // verify node locations on x-
    auto tVals = getSetProjection(tMesh, "x-", 0, -0.5);
    for( auto tVal : tVals ) TEST_FLOATING_EQUALITY(fabs(tVal), 0.0, cTol);

    // verify node locations on x+
    tVals = getSetProjection(tMesh, "x+", 0, 0.5);
    for( auto tVal : tVals ) TEST_FLOATING_EQUALITY(fabs(tVal), 0.0, cTol);

    // verify node locations on y+
    tVals = getSetProjection(tMesh, "y+", 1, 0.5);
    for( auto tVal : tVals ) TEST_FLOATING_EQUALITY(fabs(tVal), 0.0, cTol);

    // mesh has a single block names 'block_1'.  verify.
    auto tBlockNames = tMesh.GetElementBlockNames();
    TEST_ASSERT(tBlockNames.size() == 1);
    TEST_ASSERT(tBlockNames[0] == "block_1");

    // mesh has a single block with sequential element ids.  verify.
    auto tLocalElementIDs = tMesh.GetLocalElementIDs(tBlockNames[0]);
    auto tHostData = Kokkos::create_mirror_view(tLocalElementIDs);
    Kokkos::deep_copy(tHostData, tLocalElementIDs);
    for(Plato::OrdinalType iElem=0; iElem<tMesh.NumElements(); iElem++)
    {
        TEST_ASSERT(tHostData(iElem) == iElem);
    }

    // verify that nodesets are accounted for.
    auto tNodeSetNames = tMesh.GetNodeSetNames();
    TEST_ASSERT(tNodeSetNames.size() == 6);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "y-") == 1);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "y+") == 1);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "z-") == 1);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "z+") == 1);

    // verify that sidesets are accounted for.
    auto tSideSetNames = tMesh.GetSideSetNames();
    TEST_ASSERT(tSideSetNames.size() == 6);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "x-") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "x+") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "y-") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "y+") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "z-") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "z+") == 1);
}

TEUCHOS_UNIT_TEST(EngineMeshIntxTests, ReadHex20Mesh)
{
    std::string tFileName = "unit_cube_hex20.exo";
    Plato::EngineMesh tMesh(tFileName);

    TEST_ASSERT(tMesh.NumNodes() == 81)
    TEST_ASSERT(tMesh.NumElements() == 8)
    TEST_ASSERT(tMesh.NumDimensions() == 3)
    TEST_ASSERT(tMesh.NumNodesPerElement() == 20)

    // verify node locations on x-
    auto tVals = getSetProjection(tMesh, "x-", 0, -0.5);
    for( auto tVal : tVals ) TEST_FLOATING_EQUALITY(fabs(tVal), 0.0, cTol);

    // verify node locations on x+
    tVals = getSetProjection(tMesh, "x+", 0, 0.5);
    for( auto tVal : tVals ) TEST_FLOATING_EQUALITY(fabs(tVal), 0.0, cTol);

    // verify node locations on y+
    tVals = getSetProjection(tMesh, "y+", 1, 0.5);
    for( auto tVal : tVals ) TEST_FLOATING_EQUALITY(fabs(tVal), 0.0, cTol);

    // mesh has a single block names 'block_1'.  verify.
    auto tBlockNames = tMesh.GetElementBlockNames();
    TEST_ASSERT(tBlockNames.size() == 1);
    TEST_ASSERT(tBlockNames[0] == "block_1");

    // mesh has a single block with sequential element ids.  verify.
    auto tLocalElementIDs = tMesh.GetLocalElementIDs(tBlockNames[0]);
    auto tHostData = Kokkos::create_mirror_view(tLocalElementIDs);
    Kokkos::deep_copy(tHostData, tLocalElementIDs);
    for(Plato::OrdinalType iElem=0; iElem<tMesh.NumElements(); iElem++)
    {
        TEST_ASSERT(tHostData(iElem) == iElem);
    }

    // verify that nodesets are accounted for.
    auto tNodeSetNames = tMesh.GetNodeSetNames();
    TEST_ASSERT(tNodeSetNames.size() == 6);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "y-") == 1);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "y+") == 1);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "z-") == 1);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "z+") == 1);

    // verify that sidesets are accounted for.
    auto tSideSetNames = tMesh.GetSideSetNames();
    TEST_ASSERT(tSideSetNames.size() == 6);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "x-") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "x+") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "y-") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "y+") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "z-") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "z+") == 1);
}

TEUCHOS_UNIT_TEST(EngineMeshIntxTests, ReadHex27Mesh)
{
    std::string tFileName = "unit_cube_hex27.exo";
    Plato::EngineMesh tMesh(tFileName);

    TEST_ASSERT(tMesh.NumNodes() == 125)
    TEST_ASSERT(tMesh.NumElements() == 8)
    TEST_ASSERT(tMesh.NumDimensions() == 3)
    TEST_ASSERT(tMesh.NumNodesPerElement() == 27)

    // verify node locations on x-
    auto tVals = getSetProjection(tMesh, "x-", 0, -0.5);
    for( auto tVal : tVals ) TEST_FLOATING_EQUALITY(fabs(tVal), 0.0, cTol);

    // verify node locations on x+
    tVals = getSetProjection(tMesh, "x+", 0, 0.5);
    for( auto tVal : tVals ) TEST_FLOATING_EQUALITY(fabs(tVal), 0.0, cTol);

    // verify node locations on y+
    tVals = getSetProjection(tMesh, "y+", 1, 0.5);
    for( auto tVal : tVals ) TEST_FLOATING_EQUALITY(fabs(tVal), 0.0, cTol);

    // mesh has a single block names 'block_1'.  verify.
    auto tBlockNames = tMesh.GetElementBlockNames();
    TEST_ASSERT(tBlockNames.size() == 1);
    TEST_ASSERT(tBlockNames[0] == "block_1");

    // mesh has a single block with sequential element ids.  verify.
    auto tLocalElementIDs = tMesh.GetLocalElementIDs(tBlockNames[0]);
    auto tHostData = Kokkos::create_mirror_view(tLocalElementIDs);
    Kokkos::deep_copy(tHostData, tLocalElementIDs);
    for(Plato::OrdinalType iElem=0; iElem<tMesh.NumElements(); iElem++)
    {
        TEST_ASSERT(tHostData(iElem) == iElem);
    }

    // verify that nodesets are accounted for.
    auto tNodeSetNames = tMesh.GetNodeSetNames();
    TEST_ASSERT(tNodeSetNames.size() == 6);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "y-") == 1);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "y+") == 1);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "z-") == 1);
    TEST_ASSERT(std::count(tNodeSetNames.begin(), tNodeSetNames.end(), "z+") == 1);

    // verify that sidesets are accounted for.
    auto tSideSetNames = tMesh.GetSideSetNames();
    TEST_ASSERT(tSideSetNames.size() == 6);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "x-") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "x+") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "y-") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "y+") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "z-") == 1);
    TEST_ASSERT(std::count(tSideSetNames.begin(), tSideSetNames.end(), "z+") == 1);
}

TEUCHOS_UNIT_TEST(EngineWriterIntxTests, WriteReadTet4Mesh)
{
    // read mesh
    std::string tMeshFileName = "unit_cube_tet4.exo";
    Plato::EngineMesh tMesh(tMeshFileName);

    // create writer for this mesh
    std::string tOutFileName = "unit_cube_tet4_out.exo";
    Plato::EngineMeshIO tWrite(tOutFileName, tMesh, "write");

    // create test node field
    Plato::ScalarVector tDataOut("test nodal field", tMesh.NumNodes());
    Kokkos::deep_copy(tDataOut, 1.234);
    tWrite.AddNodeData("testNodalField", tDataOut);

    // write node field
    tWrite.Write(/*stepIndex=*/ 0, /*timeValue=*/ 1.0);

    // create reader
    Plato::EngineMeshIO tRead(tOutFileName, tMesh, "read");

    // read node field
    auto tDataIn = tRead.ReadNodeData("testNodalField", /*stepIndex=*/ 0);

    // compare node field
    TEST_ASSERT(are_equal(tDataOut, tDataIn));

    // check that attempting to read a non-existent node field throws a signal.
    TEST_THROW(tRead.ReadNodeData("nonExistentVar", /*stepIndex=*/ 0), std::exception);
}

TEUCHOS_UNIT_TEST(EngineWriterIntxTests, WriteReadTet10Mesh)
{
    // read mesh
    std::string tMeshFileName = "unit_cube_tet10.exo";
    Plato::EngineMesh tMesh(tMeshFileName);

    // create writer for this mesh
    std::string tOutFileName = "unit_cube_tet10_out.exo";
    Plato::EngineMeshIO tWrite(tOutFileName, tMesh, "write");

    // create test node field
    Plato::ScalarVector tDataOut("test nodal field", tMesh.NumNodes());
    Kokkos::deep_copy(tDataOut, 1.234);
    tWrite.AddNodeData("testNodalField", tDataOut);

    // write node field
    tWrite.Write(/*stepIndex=*/ 0, /*timeValue=*/ 1.0);

    // create reader
    Plato::EngineMeshIO tRead(tOutFileName, tMesh, "read");

    // read node field
    auto tDataIn = tRead.ReadNodeData("testNodalField", /*stepIndex=*/ 0);

    // compare node field
    TEST_ASSERT(are_equal(tDataOut, tDataIn));

    // check that attempting to read a non-existent node field throws a signal.
    TEST_THROW(tRead.ReadNodeData("nonExistentVar", /*stepIndex=*/ 0), std::exception);
}

TEUCHOS_UNIT_TEST(EngineWriterIntxTests, WriteReadHex8Mesh)
{
    // read mesh
    std::string tMeshFileName = "unit_cube_hex8.exo";
    Plato::EngineMesh tMesh(tMeshFileName);

    // create writer for this mesh
    std::string tOutFileName = "unit_cube_hex8_out.exo";
    Plato::EngineMeshIO tWrite(tOutFileName, tMesh, "write");

    // create test node field
    Plato::ScalarVector tDataOut("test nodal field", tMesh.NumNodes());
    Kokkos::deep_copy(tDataOut, 1.234);
    tWrite.AddNodeData("testNodalField", tDataOut);

    // write node field
    tWrite.Write(/*stepIndex=*/ 0, /*timeValue=*/ 1.0);

    // create reader
    Plato::EngineMeshIO tRead(tOutFileName, tMesh, "read");

    // read node field
    auto tDataIn = tRead.ReadNodeData("testNodalField", /*stepIndex=*/ 0);

    // compare node field
    TEST_ASSERT(are_equal(tDataOut, tDataIn));

    // check that attempting to read a non-existent node field throws a signal.
    TEST_THROW(tRead.ReadNodeData("nonExistentVar", /*stepIndex=*/ 0), std::exception);
}

TEUCHOS_UNIT_TEST(EngineWriterIntxTests, WriteReadHex20Mesh)
{
    // read mesh
    std::string tMeshFileName = "unit_cube_hex20.exo";
    Plato::EngineMesh tMesh(tMeshFileName);

    // create writer for this mesh
    std::string tOutFileName = "unit_cube_hex20_out.exo";
    Plato::EngineMeshIO tWrite(tOutFileName, tMesh, "write");

    // create test node field
    Plato::ScalarVector tDataOut("test nodal field", tMesh.NumNodes());
    Kokkos::deep_copy(tDataOut, 1.234);
    tWrite.AddNodeData("testNodalField", tDataOut);

    // write node field
    tWrite.Write(/*stepIndex=*/ 0, /*timeValue=*/ 1.0);

    // create reader
    Plato::EngineMeshIO tRead(tOutFileName, tMesh, "read");

    // read node field
    auto tDataIn = tRead.ReadNodeData("testNodalField", /*stepIndex=*/ 0);

    // compare node field
    TEST_ASSERT(are_equal(tDataOut, tDataIn));

    // check that attempting to read a non-existent node field throws a signal.
    TEST_THROW(tRead.ReadNodeData("nonExistentVar", /*stepIndex=*/ 0), std::exception);
}

TEUCHOS_UNIT_TEST(EngineWriterIntxTests, WriteReadHex27Mesh)
{
    // read mesh
    std::string tMeshFileName = "unit_cube_hex27.exo";
    Plato::EngineMesh tMesh(tMeshFileName);

    // create writer for this mesh
    std::string tOutFileName = "unit_cube_hex27_out.exo";
    Plato::EngineMeshIO tWrite(tOutFileName, tMesh, "write");

    // create test node field
    Plato::ScalarVector tDataOut("test nodal field", tMesh.NumNodes());
    Kokkos::deep_copy(tDataOut, 1.234);
    tWrite.AddNodeData("testNodalField", tDataOut);

    // write node field
    tWrite.Write(/*stepIndex=*/ 0, /*timeValue=*/ 1.0);

    // create reader
    Plato::EngineMeshIO tRead(tOutFileName, tMesh, "read");

    // read node field
    auto tDataIn = tRead.ReadNodeData("testNodalField", /*stepIndex=*/ 0);

    // compare node field
    TEST_ASSERT(are_equal(tDataOut, tDataIn));

    // check that attempting to read a non-existent node field throws a signal.
    TEST_THROW(tRead.ReadNodeData("nonExistentVar", /*stepIndex=*/ 0), std::exception);
}

TEUCHOS_UNIT_TEST(EngineWriterIntxTests, WriteTet4ScalarField)
{
    const Plato::OrdinalType cSpaceDim = 3;

    // read mesh
    std::string tFileName = "unit_cube_tet4.exo";
    Plato::EngineMesh tMesh(tFileName);

    // create zero nodal scalar field
    auto tNumNodes = tMesh.NumNodes();
    Plato::ScalarVector tNodalScalarField("test nodal scalar field", tNumNodes);

    // compute distance of each node from origin
    auto tCoordinates = tMesh.Coordinates();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), LAMBDA_EXPRESSION(Plato::OrdinalType tNodeOrdinal)
    {
        Plato::Scalar tDistance = 0.0;
        for(Plato::OrdinalType tDim=0; tDim<cSpaceDim; tDim++)
        {
            auto tComp = tCoordinates(cSpaceDim*tNodeOrdinal + tDim);
            tDistance += tComp*tComp;
        }
        tDistance = (tDistance > 0.0) ? sqrt(tDistance) : 0.0;
        tNodalScalarField(tNodeOrdinal) = tDistance;
    }, "loop on nodes");

    // write field
    std::string tOutFileName = "unit_cube_tet4_scalarField_out.exo";
    Plato::EngineMeshIO tWrite(tOutFileName, tMesh, "write");
    tWrite.AddNodeData("testNodalScalarField", tNodalScalarField);
    tWrite.Write(/*stepIndex=*/ 0, /*timeValue=*/ 1.0);

    int tStatus = std::system("exodiff unit_cube_tet4_scalarField_out.exo unit_cube_tet4_scalarField_gold.exo");
    TEST_ASSERT(tStatus == 0);
}

TEUCHOS_UNIT_TEST(EngineWriterIntxTests, WriteTet10ScalarField)
{
    const Plato::OrdinalType cSpaceDim = 3;

    // read mesh
    std::string tFileName = "unit_cube_tet10.exo";
    Plato::EngineMesh tMesh(tFileName);

    // create zero nodal scalar field
    auto tNumNodes = tMesh.NumNodes();
    Plato::ScalarVector tNodalScalarField("test nodal scalar field", tNumNodes);

    // compute distance of each node from origin
    auto tCoordinates = tMesh.Coordinates();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), LAMBDA_EXPRESSION(Plato::OrdinalType tNodeOrdinal)
    {
        Plato::Scalar tDistance = 0.0;
        for(Plato::OrdinalType tDim=0; tDim<cSpaceDim; tDim++)
        {
            auto tComp = tCoordinates(cSpaceDim*tNodeOrdinal + tDim);
            tDistance += tComp*tComp;
        }
        tDistance = (tDistance > 0.0) ? sqrt(tDistance) : 0.0;
        tNodalScalarField(tNodeOrdinal) = tDistance;
    }, "loop on nodes");

    // write field
    std::string tOutFileName = "unit_cube_tet10_scalarField_out.exo";
    Plato::EngineMeshIO tWrite(tOutFileName, tMesh, "write");
    tWrite.AddNodeData("testNodalScalarField", tNodalScalarField);
    tWrite.Write(/*stepIndex=*/ 0, /*timeValue=*/ 1.0);

    int tStatus = std::system("exodiff unit_cube_tet10_scalarField_out.exo unit_cube_tet10_scalarField_gold.exo");
    TEST_ASSERT(tStatus == 0);
}

TEUCHOS_UNIT_TEST(EngineWriterIntxTests, WriteHex8ScalarField)
{
    const Plato::OrdinalType cSpaceDim = 3;

    // read mesh
    std::string tFileName = "unit_cube_hex8.exo";
    Plato::EngineMesh tMesh(tFileName);

    // create zero nodal scalar field
    auto tNumNodes = tMesh.NumNodes();
    Plato::ScalarVector tNodalScalarField("test nodal scalar field", tNumNodes);

    // compute distance of each node from origin
    auto tCoordinates = tMesh.Coordinates();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), LAMBDA_EXPRESSION(Plato::OrdinalType tNodeOrdinal)
    {
        Plato::Scalar tDistance = 0.0;
        for(Plato::OrdinalType tDim=0; tDim<cSpaceDim; tDim++)
        {
            auto tComp = tCoordinates(cSpaceDim*tNodeOrdinal + tDim);
            tDistance += tComp*tComp;
        }
        tDistance = (tDistance > 0.0) ? sqrt(tDistance) : 0.0;
        tNodalScalarField(tNodeOrdinal) = tDistance;
    }, "loop on nodes");

    // write field
    std::string tOutFileName = "unit_cube_hex8_scalarField_out.exo";
    Plato::EngineMeshIO tWrite(tOutFileName, tMesh, "write");
    tWrite.AddNodeData("testNodalScalarField", tNodalScalarField);
    tWrite.Write(/*stepIndex=*/ 0, /*timeValue=*/ 1.0);

    int tStatus = std::system("exodiff unit_cube_hex8_scalarField_out.exo unit_cube_hex8_scalarField_gold.exo");
    TEST_ASSERT(tStatus == 0);
}

TEUCHOS_UNIT_TEST(EngineWriterIntxTests, WriteHex20ScalarField)
{
    const Plato::OrdinalType cSpaceDim = 3;

    // read mesh
    std::string tFileName = "unit_cube_hex20.exo";
    Plato::EngineMesh tMesh(tFileName);

    // create zero nodal scalar field
    auto tNumNodes = tMesh.NumNodes();
    Plato::ScalarVector tNodalScalarField("test nodal scalar field", tNumNodes);

    // compute distance of each node from origin
    auto tCoordinates = tMesh.Coordinates();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), LAMBDA_EXPRESSION(Plato::OrdinalType tNodeOrdinal)
    {
        Plato::Scalar tDistance = 0.0;
        for(Plato::OrdinalType tDim=0; tDim<cSpaceDim; tDim++)
        {
            auto tComp = tCoordinates(cSpaceDim*tNodeOrdinal + tDim);
            tDistance += tComp*tComp;
        }
        tDistance = (tDistance > 0.0) ? sqrt(tDistance) : 0.0;
        tNodalScalarField(tNodeOrdinal) = tDistance;
    }, "loop on nodes");

    // write field
    std::string tOutFileName = "unit_cube_hex20_scalarField_out.exo";
    Plato::EngineMeshIO tWrite(tOutFileName, tMesh, "write");
    tWrite.AddNodeData("testNodalScalarField", tNodalScalarField);
    tWrite.Write(/*stepIndex=*/ 0, /*timeValue=*/ 1.0);

    int tStatus = std::system("exodiff unit_cube_hex20_scalarField_out.exo unit_cube_hex20_scalarField_gold.exo");
    TEST_ASSERT(tStatus == 0);
}
