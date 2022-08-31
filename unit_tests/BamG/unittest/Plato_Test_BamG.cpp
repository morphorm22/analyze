/*
//@HEADER
// *************************************************************************
//   Plato Engine v.1.0: Copyright 2018, National Technology & Engineering
//                    Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Sandia Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact the Plato team (plato3D-help@sandia.gov)
//
// *************************************************************************
//@HEADER
*/

/*
 * Plato_Test_BamG.cpp
 *
 *  Created on: Dec 14, 2021
 */

#include "Teuchos_UnitTestHarness.hpp"
#include "BamG.hpp"

namespace PlatoTestBamG
{

  
  bool isSame(const BamG::ArrayT<BamG::Uint> & tArrayA, const BamG::ArrayT<BamG::Uint> & tArrayB)
  {
      if( tArrayA.size() != tArrayB.size() ) return false;

      if( (tArrayA.size() == 0) && (tArrayB.size() == 0) ) return true;

      auto tExtent = tArrayA.size();
      for( decltype(tExtent) tIndex=0; tIndex<tExtent; tIndex++ )
      {
          if( tArrayA[tIndex] != tArrayB[tIndex] ) return false;
      }
      return true;
  }

  bool isSame(const BamG::Array2D & tArrayA, const BamG::Array2D & tArrayB)
  {
      if( tArrayA.size() != tArrayB.size() ) return false;

      if( (tArrayA.size() == 0) && (tArrayB.size() == 0) ) return true;

      if( (tArrayA.size() == 0) || (tArrayB.size() == 0) ) return false;

      auto tExtent_0 = tArrayA.size();
      for( decltype(tExtent_0) tIndex0=0; tIndex0<tExtent_0; tIndex0++ )
      {
          if( tArrayA[tIndex0].size() != tArrayB[tIndex0].size() ) return false;
          auto tExtent_1 = tArrayA[tIndex0].size();
          for( decltype(tExtent_1) tIndex1=0; tIndex1<tExtent_1; tIndex1++ )
          {
              if( tArrayA[tIndex0][tIndex1] != tArrayB[tIndex0][tIndex1] ) return false;
          }
      }
      return true;
  }
  
  TEUCHOS_UNIT_TEST(PlatoTestBamG, matches)
  {
      TEST_ASSERT(BamG::matches("Hex8", "hex8"));
      TEST_ASSERT(BamG::matches("HEX8", "hex8"));
      TEST_ASSERT(BamG::matches("hex8", "HEX8"));
      TEST_ASSERT(BamG::matches("tet4", "TET4"));
      TEST_ASSERT(!BamG::matches("Ren", "Stimpy"));
      TEST_ASSERT(!BamG::matches("hex8", "hexa8"));
      TEST_ASSERT(!BamG::matches("hex8", "hex80"));
      TEST_ASSERT(!BamG::matches("hex8", "he8x"));
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, isSame)
  {
      {
          BamG::Array2D tArrayA = {{1,2,3},{4,5,6},{7,8,9}};
          BamG::Array2D tArrayB = {{1,2,3},{4,5,6},{7,8,9}};

          TEST_ASSERT(isSame(tArrayA, tArrayB));
      }
      {
          BamG::Array2D tArrayA = {{2,2,3},{4,5,6},{7,8,9}};
          BamG::Array2D tArrayB = {{1,2,3},{4,5,6},{7,8,9}};

          TEST_ASSERT(!isSame(tArrayA, tArrayB));
      }
      {
          BamG::Array2D tArrayA = {{1,2,3},{4,5,6},{7,8,9}};
          BamG::Array2D tArrayB = {{1,2},{4,5,6},{7,8,9}};

          TEST_ASSERT(!isSame(tArrayA, tArrayB));
      }
      {
          BamG::Array2D tArrayA = {{1,2},{4,5,6},{7,8,9}};
          BamG::Array2D tArrayB = {{1,2,3},{4,5,6},{7,8,9}};

          TEST_ASSERT(!isSame(tArrayA, tArrayB));
      }
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, indexMap_Hex8)
  {
      TEST_ASSERT(BamG::Hex8::indexMap(0,0,0,10,10,10) == 0);
      TEST_ASSERT(BamG::Hex8::indexMap(9,9,9,10,10,10) == 999);
      TEST_ASSERT(BamG::Hex8::indexMap(1,1,1,10,10,10) == 111);
      TEST_ASSERT(BamG::Hex8::indexMap(1,1,2,10,10,10) == 112);
      TEST_ASSERT(BamG::Hex8::indexMap(1,2,1,10,10,10) == 121);
      TEST_ASSERT(BamG::Hex8::indexMap(2,1,1,10,10,10) == 211);
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateCoords_Hex8)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "hex8";

      auto tHex8Coords = BamG::Hex8::generateCoords(tSpec);

      BamG::Array2D tHex8Coords_Gold =
      {
          { 0.0,  0.0,  0.0,  0.0,  1.0,  1.0,  1.0,  1.0},
          { 0.0,  0.0,  1.0,  1.0,  0.0,  0.0,  1.0,  1.0},
          { 0.0,  1.0,  0.0,  1.0,  0.0,  1.0,  0.0,  1.0}
      };

      TEST_ASSERT(isSame(tHex8Coords, tHex8Coords_Gold));
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateCoords_Tet4)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "hex8";

      auto tTet4Coords = BamG::Tet4::generateCoords(tSpec);

      BamG::Array2D tTet4Coords_Gold =
      {
          { 0.0,  0.0,  0.0,  0.0,  1.0,  1.0,  1.0,  1.0},
          { 0.0,  0.0,  1.0,  1.0,  0.0,  0.0,  1.0,  1.0},
          { 0.0,  1.0,  0.0,  1.0,  0.0,  1.0,  0.0,  1.0}
      };

      TEST_ASSERT(isSame(tTet4Coords, tTet4Coords_Gold));
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateCoords_Quad4)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "quad4";

      auto tQuad4Coords = BamG::Quad4::generateCoords(tSpec);

      BamG::Array2D tQuad4Coords_Gold =
      {
          { 0.0,  0.0,  1.0,  1.0},
          { 0.0,  1.0,  0.0,  1.0}
      };

      TEST_ASSERT(isSame(tQuad4Coords, tQuad4Coords_Gold));
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateCoords_Tri3)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "quad4";

      auto tTri3Coords = BamG::Tri3::generateCoords(tSpec);

      BamG::Array2D tTri3Coords_Gold =
      {
          { 0.0,  0.0,  1.0,  1.0},
          { 0.0,  1.0,  0.0,  1.0}
      };

      TEST_ASSERT(isSame(tTri3Coords, tTri3Coords_Gold));
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateCoords_Tri3_2x2)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "quad4";
      tSpec.numX = 2;
      tSpec.numY = 2;

      auto tTri3Coords = BamG::Tri3::generateCoords(tSpec);

      BamG::Array2D tTri3Coords_Gold =
      {
          { 0.0,  0.0,  0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0},
          { 0.0, 0.5, 1.0,  0.0, 0.5, 1.0,  0.0, 0.5, 1.0}
      };

      TEST_ASSERT(isSame(tTri3Coords, tTri3Coords_Gold));
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateCoords_Bar2)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "bar2";
      tSpec.numX = 2;

      auto tBar2Coords = BamG::Bar2::generateCoords(tSpec);

      BamG::Array2D tBar2Coords_Gold = { {0.0, 0.5,  1.0} };

      TEST_ASSERT(isSame(tBar2Coords, tBar2Coords_Gold));
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateConnectivity_Hex8)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "hex8";

      auto tHex8Connectivity = BamG::Hex8::generateConnectivity(tSpec);

      BamG::ArrayT<BamG::Uint> tHex8Connectivity_Gold = { 0, 4, 6, 2, 1, 5, 7, 3 };

      TEST_ASSERT(isSame(tHex8Connectivity, tHex8Connectivity_Gold));
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateConnectivity_Tet4)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "tet4";

      auto tTet4Connectivity = BamG::Tet4::generateConnectivity(tSpec);

      BamG::ArrayT<BamG::Uint> tTet4Connectivity_Gold =
      {0,6,2,7,0,2,3,7,0,3,1,7,0,1,5,7,0,5,4,7,0,4,6,7};

      TEST_ASSERT(isSame(tTet4Connectivity, tTet4Connectivity_Gold));
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateConnectivity_Quad4)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "quad4";

      auto tQuad4Connectivity = BamG::Quad4::generateConnectivity(tSpec);

      BamG::ArrayT<BamG::Uint> tQuad4Connectivity_Gold = {0,2,3,1};

      TEST_ASSERT(isSame(tQuad4Connectivity, tQuad4Connectivity_Gold));
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateConnectivity_Tri3)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "tri3";

      auto tTri3Connectivity = BamG::Tri3::generateConnectivity(tSpec);

      BamG::ArrayT<BamG::Uint> tTri3Connectivity_Gold = {0,2,3,0,3,1};

      TEST_ASSERT(isSame(tTri3Connectivity, tTri3Connectivity_Gold));
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateConnectivity_Tri3_2x2)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "tri3";
      tSpec.numX = 2;
      tSpec.numY = 2;

      auto tTri3Connectivity = BamG::Tri3::generateConnectivity(tSpec);

      BamG::ArrayT<BamG::Uint> tTri3Connectivity_Gold = 
          {0,3,4,0,4,1,1,4,5,1,5,2,3,6,7,3,7,4,4,7,8,4,8,5};

      TEST_ASSERT(isSame(tTri3Connectivity, tTri3Connectivity_Gold));
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateConnectivity_Bar2)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "bar2";
      tSpec.numX = 2;

      auto tBar2Connectivity = BamG::Bar2::generateConnectivity(tSpec);

      BamG::ArrayT<BamG::Uint> tBar2Connectivity_Gold = {0,1,1,2};

      TEST_ASSERT(isSame(tBar2Connectivity, tBar2Connectivity_Gold));
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, getNumNPE)
  {
      {
          BamG::MeshSpec tSpec;
          tSpec.meshType = "hex8";

          auto tNumNPE = BamG::getNumNPE(tSpec);

          TEST_ASSERT(tNumNPE == 8);
      }
      {
          BamG::MeshSpec tSpec;
          tSpec.meshType = "tet4";

          auto tNumNPE = BamG::getNumNPE(tSpec);

          TEST_ASSERT(tNumNPE == 4);
      }
      {
          BamG::MeshSpec tSpec;
          tSpec.meshType = "quad4";

          auto tNumNPE = BamG::getNumNPE(tSpec);

          TEST_ASSERT(tNumNPE == 4);
      }
      {
          BamG::MeshSpec tSpec;
          tSpec.meshType = "tri3";

          auto tNumNPE = BamG::getNumNPE(tSpec);

          TEST_ASSERT(tNumNPE == 3);
      }
  }
  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateNodeSets_Hex8_Tet4)
  {
      BamG::IArrayMap tNodeSets_Gold;

      tNodeSets_Gold["x-"] = BamG::IArray({0, 1, 2, 3});
      tNodeSets_Gold["x+"] = BamG::IArray({4, 5, 6, 7});
      tNodeSets_Gold["y-"] = BamG::IArray({0, 1, 4, 5});
      tNodeSets_Gold["y+"] = BamG::IArray({2, 3, 6, 7});
      tNodeSets_Gold["z-"] = BamG::IArray({0, 2, 4, 6});
      tNodeSets_Gold["z+"] = BamG::IArray({1, 3, 5, 7});

      tNodeSets_Gold["y-z-"] = BamG::IArray({0, 4});
      tNodeSets_Gold["y-z+"] = BamG::IArray({1, 5});
      tNodeSets_Gold["y+z-"] = BamG::IArray({2, 6});
      tNodeSets_Gold["y+z+"] = BamG::IArray({3, 7});

      tNodeSets_Gold["x-z-"] = BamG::IArray({0, 2});
      tNodeSets_Gold["x-z+"] = BamG::IArray({1, 3});
      tNodeSets_Gold["x+z-"] = BamG::IArray({4, 6});
      tNodeSets_Gold["x+z+"] = BamG::IArray({5, 7});

      tNodeSets_Gold["x-y-"] = BamG::IArray({0, 1});
      tNodeSets_Gold["x-y+"] = BamG::IArray({2, 3});
      tNodeSets_Gold["x+y-"] = BamG::IArray({4, 5});
      tNodeSets_Gold["x+y+"] = BamG::IArray({6, 7});

      tNodeSets_Gold["x-y-z-"] = BamG::IArray({0});
      tNodeSets_Gold["x-y-z+"] = BamG::IArray({1});
      tNodeSets_Gold["x-y+z-"] = BamG::IArray({2});
      tNodeSets_Gold["x-y+z+"] = BamG::IArray({3});
      tNodeSets_Gold["x+y-z-"] = BamG::IArray({4});
      tNodeSets_Gold["x+y-z+"] = BamG::IArray({5});
      tNodeSets_Gold["x+y+z-"] = BamG::IArray({6});
      tNodeSets_Gold["x+y+z+"] = BamG::IArray({7});

      {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "hex8";

      auto tHex8NodeSets = BamG::Hex8::generateNodeSets(tSpec);


      TEST_ASSERT(isSame(tHex8NodeSets["x-"], tNodeSets_Gold["x-"]));
      TEST_ASSERT(isSame(tHex8NodeSets["x+"], tNodeSets_Gold["x+"]));
      TEST_ASSERT(isSame(tHex8NodeSets["y-"], tNodeSets_Gold["y-"]));
      TEST_ASSERT(isSame(tHex8NodeSets["y+"], tNodeSets_Gold["y+"]));
      TEST_ASSERT(isSame(tHex8NodeSets["z-"], tNodeSets_Gold["z-"]));
      TEST_ASSERT(isSame(tHex8NodeSets["z+"], tNodeSets_Gold["z+"]));

      TEST_ASSERT(isSame(tHex8NodeSets["y-z-"], tNodeSets_Gold["y-z-"]));
      TEST_ASSERT(isSame(tHex8NodeSets["y-z+"], tNodeSets_Gold["y-z+"]));
      TEST_ASSERT(isSame(tHex8NodeSets["y+z-"], tNodeSets_Gold["y+z-"]));
      TEST_ASSERT(isSame(tHex8NodeSets["y+z+"], tNodeSets_Gold["y+z+"]));

      TEST_ASSERT(isSame(tHex8NodeSets["x-z-"], tNodeSets_Gold["x-z-"]));
      TEST_ASSERT(isSame(tHex8NodeSets["x-z+"], tNodeSets_Gold["x-z+"]));
      TEST_ASSERT(isSame(tHex8NodeSets["x+z-"], tNodeSets_Gold["x+z-"]));
      TEST_ASSERT(isSame(tHex8NodeSets["x+z+"], tNodeSets_Gold["x+z+"]));

      TEST_ASSERT(isSame(tHex8NodeSets["x-y-"], tNodeSets_Gold["x-y-"]));
      TEST_ASSERT(isSame(tHex8NodeSets["x-y+"], tNodeSets_Gold["x-y+"]));
      TEST_ASSERT(isSame(tHex8NodeSets["x+y-"], tNodeSets_Gold["x+y-"]));
      TEST_ASSERT(isSame(tHex8NodeSets["x+y+"], tNodeSets_Gold["x+y+"]));

      TEST_ASSERT(isSame(tHex8NodeSets["x-y-z-"], tNodeSets_Gold["x-y-z-"]));
      TEST_ASSERT(isSame(tHex8NodeSets["x-y-z+"], tNodeSets_Gold["x-y-z+"]));
      TEST_ASSERT(isSame(tHex8NodeSets["x-y+z-"], tNodeSets_Gold["x-y+z-"]));
      TEST_ASSERT(isSame(tHex8NodeSets["x-y+z+"], tNodeSets_Gold["x-y+z+"]));
      TEST_ASSERT(isSame(tHex8NodeSets["x+y-z-"], tNodeSets_Gold["x+y-z-"]));
      TEST_ASSERT(isSame(tHex8NodeSets["x+y-z+"], tNodeSets_Gold["x+y-z+"]));
      TEST_ASSERT(isSame(tHex8NodeSets["x+y+z-"], tNodeSets_Gold["x+y+z-"]));
      TEST_ASSERT(isSame(tHex8NodeSets["x+y+z+"], tNodeSets_Gold["x+y+z+"]));
      }
      {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "tet4";

      auto tTet4NodeSets = BamG::Tet4::generateNodeSets(tSpec);

      TEST_ASSERT(isSame(tTet4NodeSets["x-"], tNodeSets_Gold["x-"]));
      TEST_ASSERT(isSame(tTet4NodeSets["x+"], tNodeSets_Gold["x+"]));
      TEST_ASSERT(isSame(tTet4NodeSets["y-"], tNodeSets_Gold["y-"]));
      TEST_ASSERT(isSame(tTet4NodeSets["y+"], tNodeSets_Gold["y+"]));
      TEST_ASSERT(isSame(tTet4NodeSets["z-"], tNodeSets_Gold["z-"]));
      TEST_ASSERT(isSame(tTet4NodeSets["z+"], tNodeSets_Gold["z+"]));

      TEST_ASSERT(isSame(tTet4NodeSets["y-z-"], tNodeSets_Gold["y-z-"]));
      TEST_ASSERT(isSame(tTet4NodeSets["y-z+"], tNodeSets_Gold["y-z+"]));
      TEST_ASSERT(isSame(tTet4NodeSets["y+z-"], tNodeSets_Gold["y+z-"]));
      TEST_ASSERT(isSame(tTet4NodeSets["y+z+"], tNodeSets_Gold["y+z+"]));

      TEST_ASSERT(isSame(tTet4NodeSets["x-z-"], tNodeSets_Gold["x-z-"]));
      TEST_ASSERT(isSame(tTet4NodeSets["x-z+"], tNodeSets_Gold["x-z+"]));
      TEST_ASSERT(isSame(tTet4NodeSets["x+z-"], tNodeSets_Gold["x+z-"]));
      TEST_ASSERT(isSame(tTet4NodeSets["x+z+"], tNodeSets_Gold["x+z+"]));

      TEST_ASSERT(isSame(tTet4NodeSets["x-y-"], tNodeSets_Gold["x-y-"]));
      TEST_ASSERT(isSame(tTet4NodeSets["x-y+"], tNodeSets_Gold["x-y+"]));
      TEST_ASSERT(isSame(tTet4NodeSets["x+y-"], tNodeSets_Gold["x+y-"]));
      TEST_ASSERT(isSame(tTet4NodeSets["x+y+"], tNodeSets_Gold["x+y+"]));

      TEST_ASSERT(isSame(tTet4NodeSets["x-y-z-"], tNodeSets_Gold["x-y-z-"]));
      TEST_ASSERT(isSame(tTet4NodeSets["x-y-z+"], tNodeSets_Gold["x-y-z+"]));
      TEST_ASSERT(isSame(tTet4NodeSets["x-y+z-"], tNodeSets_Gold["x-y+z-"]));
      TEST_ASSERT(isSame(tTet4NodeSets["x-y+z+"], tNodeSets_Gold["x-y+z+"]));
      TEST_ASSERT(isSame(tTet4NodeSets["x+y-z-"], tNodeSets_Gold["x+y-z-"]));
      TEST_ASSERT(isSame(tTet4NodeSets["x+y-z+"], tNodeSets_Gold["x+y-z+"]));
      TEST_ASSERT(isSame(tTet4NodeSets["x+y+z-"], tNodeSets_Gold["x+y+z-"]));
      TEST_ASSERT(isSame(tTet4NodeSets["x+y+z+"], tNodeSets_Gold["x+y+z+"]));
      }
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateNodeSets_Quad4_Tri3)
  {
      BamG::IArrayMap tNodeSets_Gold;

      tNodeSets_Gold["x-"] = BamG::IArray({0, 1});
      tNodeSets_Gold["x+"] = BamG::IArray({2, 3});
      tNodeSets_Gold["y-"] = BamG::IArray({0, 2});
      tNodeSets_Gold["y+"] = BamG::IArray({1, 3});

      tNodeSets_Gold["x-y-"] = BamG::IArray({0});
      tNodeSets_Gold["x-y+"] = BamG::IArray({1});
      tNodeSets_Gold["x+y-"] = BamG::IArray({2});
      tNodeSets_Gold["x+y+"] = BamG::IArray({3});
      {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "quad4";

      auto tQuad4NodeSets = BamG::Quad4::generateNodeSets(tSpec);

      TEST_ASSERT(isSame(tQuad4NodeSets["x-"], tNodeSets_Gold["x-"]));
      TEST_ASSERT(isSame(tQuad4NodeSets["x+"], tNodeSets_Gold["x+"]));
      TEST_ASSERT(isSame(tQuad4NodeSets["y-"], tNodeSets_Gold["y-"]));
      TEST_ASSERT(isSame(tQuad4NodeSets["y+"], tNodeSets_Gold["y+"]));

      TEST_ASSERT(isSame(tQuad4NodeSets["x-y-"], tNodeSets_Gold["x-y-"]));
      TEST_ASSERT(isSame(tQuad4NodeSets["x-y+"], tNodeSets_Gold["x-y+"]));
      TEST_ASSERT(isSame(tQuad4NodeSets["x+y-"], tNodeSets_Gold["x+y-"]));
      TEST_ASSERT(isSame(tQuad4NodeSets["x+y+"], tNodeSets_Gold["x+y+"]));
      }
      {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "tri3";

      auto tTri3NodeSets = BamG::Tri3::generateNodeSets(tSpec);

      TEST_ASSERT(isSame(tTri3NodeSets["x-"], tNodeSets_Gold["x-"]));
      TEST_ASSERT(isSame(tTri3NodeSets["x+"], tNodeSets_Gold["x+"]));
      TEST_ASSERT(isSame(tTri3NodeSets["y-"], tNodeSets_Gold["y-"]));
      TEST_ASSERT(isSame(tTri3NodeSets["y+"], tNodeSets_Gold["y+"]));

      TEST_ASSERT(isSame(tTri3NodeSets["x-y-"], tNodeSets_Gold["x-y-"]));
      TEST_ASSERT(isSame(tTri3NodeSets["x-y+"], tNodeSets_Gold["x-y+"]));
      TEST_ASSERT(isSame(tTri3NodeSets["x+y-"], tNodeSets_Gold["x+y-"]));
      TEST_ASSERT(isSame(tTri3NodeSets["x+y+"], tNodeSets_Gold["x+y+"]));
      }
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateNodeSets_Quad4_Tri3_2x2)
  {
      BamG::IArrayMap tNodeSets_Gold;

      tNodeSets_Gold["x-"] = BamG::IArray({0, 1, 2});
      tNodeSets_Gold["x+"] = BamG::IArray({6, 7, 8});
      tNodeSets_Gold["y-"] = BamG::IArray({0, 3, 6});
      tNodeSets_Gold["y+"] = BamG::IArray({2, 5, 8});

      tNodeSets_Gold["x-y-"] = BamG::IArray({0});
      tNodeSets_Gold["x-y+"] = BamG::IArray({2});
      tNodeSets_Gold["x+y-"] = BamG::IArray({6});
      tNodeSets_Gold["x+y+"] = BamG::IArray({8});
      {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "quad4";
      tSpec.numX = 2;
      tSpec.numY = 2;

      auto tQuad4NodeSets = BamG::Quad4::generateNodeSets(tSpec);

      TEST_ASSERT(isSame(tQuad4NodeSets["x-"], tNodeSets_Gold["x-"]));
      TEST_ASSERT(isSame(tQuad4NodeSets["x+"], tNodeSets_Gold["x+"]));
      TEST_ASSERT(isSame(tQuad4NodeSets["y-"], tNodeSets_Gold["y-"]));
      TEST_ASSERT(isSame(tQuad4NodeSets["y+"], tNodeSets_Gold["y+"]));

      TEST_ASSERT(isSame(tQuad4NodeSets["x-y-"], tNodeSets_Gold["x-y-"]));
      TEST_ASSERT(isSame(tQuad4NodeSets["x-y+"], tNodeSets_Gold["x-y+"]));
      TEST_ASSERT(isSame(tQuad4NodeSets["x+y-"], tNodeSets_Gold["x+y-"]));
      TEST_ASSERT(isSame(tQuad4NodeSets["x+y+"], tNodeSets_Gold["x+y+"]));
      }
      {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "tri3";
      tSpec.numX = 2;
      tSpec.numY = 2;

      auto tTri3NodeSets = BamG::Tri3::generateNodeSets(tSpec);

      TEST_ASSERT(isSame(tTri3NodeSets["x-"], tNodeSets_Gold["x-"]));
      TEST_ASSERT(isSame(tTri3NodeSets["x+"], tNodeSets_Gold["x+"]));
      TEST_ASSERT(isSame(tTri3NodeSets["y-"], tNodeSets_Gold["y-"]));
      TEST_ASSERT(isSame(tTri3NodeSets["y+"], tNodeSets_Gold["y+"]));

      TEST_ASSERT(isSame(tTri3NodeSets["x-y-"], tNodeSets_Gold["x-y-"]));
      TEST_ASSERT(isSame(tTri3NodeSets["x-y+"], tNodeSets_Gold["x-y+"]));
      TEST_ASSERT(isSame(tTri3NodeSets["x+y-"], tNodeSets_Gold["x+y-"]));
      TEST_ASSERT(isSame(tTri3NodeSets["x+y+"], tNodeSets_Gold["x+y+"]));
      }
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateNodeSets_Bar2)
  {
      BamG::IArrayMap tNodeSets_Gold;

      tNodeSets_Gold["x-"] = BamG::IArray({0});
      tNodeSets_Gold["x+"] = BamG::IArray({2});

      {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "bar2";
      tSpec.numX = 2;

      auto tBar2NodeSets = BamG::Bar2::generateNodeSets(tSpec);

      TEST_ASSERT(isSame(tBar2NodeSets["x-"], tNodeSets_Gold["x-"]));
      TEST_ASSERT(isSame(tBar2NodeSets["x+"], tNodeSets_Gold["x+"]));
      }
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateSideSets_Hex8)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "hex8";

      auto tSideSets = BamG::Hex8::generateSideSets(tSpec);

      BamG::SideSetMap tSideSets_Gold;

      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({0});
          tSideSet.faces    = BamG::IArray({4});
          tSideSets_Gold["x-"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({0});
          tSideSet.faces    = BamG::IArray({2});
          tSideSets_Gold["x+"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({0});
          tSideSet.faces    = BamG::IArray({1});
          tSideSets_Gold["y-"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({0});
          tSideSet.faces    = BamG::IArray({3});
          tSideSets_Gold["y+"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({0});
          tSideSet.faces    = BamG::IArray({5});
          tSideSets_Gold["z-"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({0});
          tSideSet.faces    = BamG::IArray({6});
          tSideSets_Gold["z+"] = tSideSet;
      }

      TEST_ASSERT(isSame(tSideSets["x-"].elements, tSideSets_Gold["x-"].elements));
      TEST_ASSERT(isSame(tSideSets["x-"].faces,    tSideSets_Gold["x-"].faces   ));
      TEST_ASSERT(isSame(tSideSets["x+"].elements, tSideSets_Gold["x+"].elements));
      TEST_ASSERT(isSame(tSideSets["x+"].faces,    tSideSets_Gold["x+"].faces   ));
      TEST_ASSERT(isSame(tSideSets["y-"].elements, tSideSets_Gold["y-"].elements));
      TEST_ASSERT(isSame(tSideSets["y-"].faces,    tSideSets_Gold["y-"].faces   ));
      TEST_ASSERT(isSame(tSideSets["y+"].elements, tSideSets_Gold["y+"].elements));
      TEST_ASSERT(isSame(tSideSets["y+"].faces,    tSideSets_Gold["y+"].faces   ));
      TEST_ASSERT(isSame(tSideSets["z-"].elements, tSideSets_Gold["z-"].elements));
      TEST_ASSERT(isSame(tSideSets["z-"].faces,    tSideSets_Gold["z-"].faces   ));
      TEST_ASSERT(isSame(tSideSets["z+"].elements, tSideSets_Gold["z+"].elements));
      TEST_ASSERT(isSame(tSideSets["z+"].faces,    tSideSets_Gold["z+"].faces   ));
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateSideSets_Quad4)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "quad4";

      auto tSideSets = BamG::Quad4::generateSideSets(tSpec);

      BamG::SideSetMap tSideSets_Gold;

      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({0});
          tSideSet.faces    = BamG::IArray({4});
          tSideSets_Gold["x-"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({0});
          tSideSet.faces    = BamG::IArray({2});
          tSideSets_Gold["x+"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({0});
          tSideSet.faces    = BamG::IArray({1});
          tSideSets_Gold["y-"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({0});
          tSideSet.faces    = BamG::IArray({3});
          tSideSets_Gold["y+"] = tSideSet;
      }

      TEST_ASSERT(isSame(tSideSets["x-"].elements, tSideSets_Gold["x-"].elements));
      TEST_ASSERT(isSame(tSideSets["x-"].faces,    tSideSets_Gold["x-"].faces   ));
      TEST_ASSERT(isSame(tSideSets["x+"].elements, tSideSets_Gold["x+"].elements));
      TEST_ASSERT(isSame(tSideSets["x+"].faces,    tSideSets_Gold["x+"].faces   ));
      TEST_ASSERT(isSame(tSideSets["y-"].elements, tSideSets_Gold["y-"].elements));
      TEST_ASSERT(isSame(tSideSets["y-"].faces,    tSideSets_Gold["y-"].faces   ));
      TEST_ASSERT(isSame(tSideSets["y+"].elements, tSideSets_Gold["y+"].elements));
      TEST_ASSERT(isSame(tSideSets["y+"].faces,    tSideSets_Gold["y+"].faces   ));
  }
  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateSideSets_Tet4)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "tet4";

      auto tSideSets = BamG::Tet4::generateSideSets(tSpec);

      BamG::SideSetMap tSideSets_Gold;

      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({1,2});
          tSideSet.faces    = BamG::IArray({4,4});
          tSideSets_Gold["x-"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({4,5});
          tSideSet.faces    = BamG::IArray({2,2});
          tSideSets_Gold["x+"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({3,4});
          tSideSet.faces    = BamG::IArray({4,4});
          tSideSets_Gold["y-"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({0,1});
          tSideSet.faces    = BamG::IArray({2,2});
          tSideSets_Gold["y+"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({0,5});
          tSideSet.faces    = BamG::IArray({4,4});
          tSideSets_Gold["z-"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({2,3});
          tSideSet.faces    = BamG::IArray({2,2});
          tSideSets_Gold["z+"] = tSideSet;
      }

      TEST_ASSERT(isSame(tSideSets["x-"].elements, tSideSets_Gold["x-"].elements));
      TEST_ASSERT(isSame(tSideSets["x-"].faces,    tSideSets_Gold["x-"].faces   ));
      TEST_ASSERT(isSame(tSideSets["x+"].elements, tSideSets_Gold["x+"].elements));
      TEST_ASSERT(isSame(tSideSets["x+"].faces,    tSideSets_Gold["x+"].faces   ));
      TEST_ASSERT(isSame(tSideSets["y-"].elements, tSideSets_Gold["y-"].elements));
      TEST_ASSERT(isSame(tSideSets["y-"].faces,    tSideSets_Gold["y-"].faces   ));
      TEST_ASSERT(isSame(tSideSets["y+"].elements, tSideSets_Gold["y+"].elements));
      TEST_ASSERT(isSame(tSideSets["y+"].faces,    tSideSets_Gold["y+"].faces   ));
      TEST_ASSERT(isSame(tSideSets["z-"].elements, tSideSets_Gold["z-"].elements));
      TEST_ASSERT(isSame(tSideSets["z-"].faces,    tSideSets_Gold["z-"].faces   ));
      TEST_ASSERT(isSame(tSideSets["z+"].elements, tSideSets_Gold["z+"].elements));
      TEST_ASSERT(isSame(tSideSets["z+"].faces,    tSideSets_Gold["z+"].faces   ));
  }
  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateSideSets_Tri3)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "tri3";

      auto tSideSets = BamG::Tri3::generateSideSets(tSpec);

      BamG::SideSetMap tSideSets_Gold;

      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({1});
          tSideSet.faces    = BamG::IArray({3});
          tSideSets_Gold["x-"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({0});
          tSideSet.faces    = BamG::IArray({2});
          tSideSets_Gold["x+"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({0});
          tSideSet.faces    = BamG::IArray({1});
          tSideSets_Gold["y-"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({1});
          tSideSet.faces    = BamG::IArray({2});
          tSideSets_Gold["y+"] = tSideSet;
      }

      TEST_ASSERT(isSame(tSideSets["x-"].elements, tSideSets_Gold["x-"].elements));
      TEST_ASSERT(isSame(tSideSets["x-"].faces,    tSideSets_Gold["x-"].faces   ));
      TEST_ASSERT(isSame(tSideSets["x+"].elements, tSideSets_Gold["x+"].elements));
      TEST_ASSERT(isSame(tSideSets["x+"].faces,    tSideSets_Gold["x+"].faces   ));
      TEST_ASSERT(isSame(tSideSets["y-"].elements, tSideSets_Gold["y-"].elements));
      TEST_ASSERT(isSame(tSideSets["y-"].faces,    tSideSets_Gold["y-"].faces   ));
      TEST_ASSERT(isSame(tSideSets["y+"].elements, tSideSets_Gold["y+"].elements));
      TEST_ASSERT(isSame(tSideSets["y+"].faces,    tSideSets_Gold["y+"].faces   ));
  }

  TEUCHOS_UNIT_TEST(PlatoTestBamG, generateSideSets_Tri3_2x2)
  {
      BamG::MeshSpec tSpec;
      tSpec.meshType = "tri3";
      tSpec.numX = 2;
      tSpec.numY = 2;

      auto tSideSets = BamG::Tri3::generateSideSets(tSpec);

      BamG::SideSetMap tSideSets_Gold;

      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({1, 3});
          tSideSet.faces    = BamG::IArray({3, 3});
          tSideSets_Gold["x-"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({4, 6});
          tSideSet.faces    = BamG::IArray({2, 2});
          tSideSets_Gold["x+"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({0, 4});
          tSideSet.faces    = BamG::IArray({1, 1});
          tSideSets_Gold["y-"] = tSideSet;
      }
      {
          BamG::SideSet tSideSet;
          tSideSet.elements = BamG::IArray({3, 7});
          tSideSet.faces    = BamG::IArray({2, 2});
          tSideSets_Gold["y+"] = tSideSet;
      }

      TEST_ASSERT(isSame(tSideSets["x-"].elements, tSideSets_Gold["x-"].elements));
      TEST_ASSERT(isSame(tSideSets["x-"].faces,    tSideSets_Gold["x-"].faces   ));
      TEST_ASSERT(isSame(tSideSets["x+"].elements, tSideSets_Gold["x+"].elements));
      TEST_ASSERT(isSame(tSideSets["x+"].faces,    tSideSets_Gold["x+"].faces   ));
      TEST_ASSERT(isSame(tSideSets["y-"].elements, tSideSets_Gold["y-"].elements));
      TEST_ASSERT(isSame(tSideSets["y-"].faces,    tSideSets_Gold["y-"].faces   ));
      TEST_ASSERT(isSame(tSideSets["y+"].elements, tSideSets_Gold["y+"].elements));
      TEST_ASSERT(isSame(tSideSets["y+"].faces,    tSideSets_Gold["y+"].faces   ));
  }
}
