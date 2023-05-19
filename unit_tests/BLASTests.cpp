/*
 * BLASTests.cpp
 *
 *  Created on: May 10, 2023
 */

// trilinos
#include <Teuchos_UnitTestHarness.hpp>

// plato unit tests
#include "util/PlatoTestHelpers.hpp"

// plato
#include "BLAS1.hpp"

namespace BLASTests
{

namespace pth = Plato::TestHelpers;

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_abs)
{
  constexpr Plato::OrdinalType tNumElems = 10;
  Plato::ScalarVector tX("X Vector", tNumElems);
  Plato::blas1::fill(-1.0, tX);
  Plato::ScalarVector tGold("gold", tNumElems);
  Plato::blas1::fill(1.0, tGold);

  Plato::blas1::abs(tX);

  auto tHostX = Kokkos::create_mirror_view(tX);
  Kokkos::deep_copy(tHostX, tX);
  auto tHostG = Kokkos::create_mirror_view(tGold);
  Kokkos::deep_copy(tHostG, tGold);

  constexpr Plato::Scalar tTolerance = 1e-4;
  for(Plato::OrdinalType tDim = 0; tDim < tHostG.size(); tDim++)
  {
    TEST_FLOATING_EQUALITY(tHostG(tDim), tHostX(tDim), tTolerance);
  }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_dot)
{
  constexpr Plato::OrdinalType tNumElems = 10;
  Plato::ScalarVector tVecA("Vec A", tNumElems);
  Plato::blas1::fill(1.0, tVecA);
  Plato::ScalarVector tVecB("Vec B", tNumElems);
  Plato::blas1::fill(2.0, tVecB);

  const Plato::Scalar tOutput = Plato::blas1::dot(tVecA, tVecB);

  constexpr Plato::Scalar tTolerance = 1e-4;
  TEST_FLOATING_EQUALITY(20., tOutput, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_norm)
{
  constexpr Plato::OrdinalType tNumElems = 10;
  Plato::ScalarVector tVecA("Vec A", tNumElems);
  Plato::blas1::fill(1.0, tVecA);

  const Plato::Scalar tOutput = Plato::blas1::norm(tVecA);
  constexpr Plato::Scalar tTolerance = 1e-6;
  TEST_FLOATING_EQUALITY(3.16227766016838, tOutput, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_sum)
{
  constexpr Plato::OrdinalType tNumElems = 10;
  Plato::ScalarVector tVecA("Vec", tNumElems);
  Plato::blas1::fill(1.0, tVecA);

  Plato::Scalar tOutput = 0.0;
  Plato::blas1::local_sum(tVecA, tOutput);

  constexpr Plato::Scalar tTolerance = 1e-4;
  TEST_FLOATING_EQUALITY(10., tOutput, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_fill)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = pth::get_box_mesh("TRI3", meshWidth);

  int numVerts = tMesh->NumNodes();
  
  Plato::ScalarVector tSomeVector("some vector", numVerts);
  Plato::blas1::fill(2.0, tSomeVector);

  auto tSomeVectorHost = Kokkos::create_mirror_view(tSomeVector);
  Kokkos::deep_copy(tSomeVectorHost, tSomeVector);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(0), 2.0, 1e-17);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(numVerts-1), 2.0, 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_copy)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = pth::get_box_mesh("TRI3", meshWidth);

  int numVerts = tMesh->NumNodes();
  
  Plato::ScalarVector tSomeVector("some vector", numVerts);
  Plato::blas1::fill(2.0, tSomeVector);

  Plato::ScalarVector tSomeOtherVector("some other vector", numVerts);
  Plato::blas1::copy(tSomeVector, tSomeOtherVector);

  auto tSomeVectorHost = Kokkos::create_mirror_view(tSomeVector);
  Kokkos::deep_copy(tSomeVectorHost, tSomeVector);
  auto tSomeOtherVectorHost = Kokkos::create_mirror_view(tSomeOtherVector);
  Kokkos::deep_copy(tSomeOtherVectorHost, tSomeOtherVector);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(0), tSomeOtherVectorHost(0), 1e-17);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(numVerts-1), tSomeOtherVectorHost(numVerts-1), 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_scale)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = pth::get_box_mesh("TRI3", meshWidth);

  int numVerts = tMesh->NumNodes();
  
  Plato::ScalarVector tSomeVector("some vector", numVerts);
  Plato::blas1::fill(1.0, tSomeVector);
  Plato::blas1::scale(2.0, tSomeVector);

  auto tSomeVectorHost = Kokkos::create_mirror_view(tSomeVector);
  Kokkos::deep_copy(tSomeVectorHost, tSomeVector);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(0), 2.0, 1e-17);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(numVerts-1), 2.0, 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_update)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = pth::get_box_mesh("TRI3", meshWidth);

  int numVerts = tMesh->NumNodes();
  
  Plato::ScalarVector tVector_A("vector a", numVerts);
  Plato::ScalarVector tVector_B("vector b", numVerts);
  Plato::blas1::fill(1.0, tVector_A);
  Plato::blas1::fill(2.0, tVector_B);
  Plato::blas1::update(2.0, tVector_A, 3.0, tVector_B);

  auto tVector_B_Host = Kokkos::create_mirror_view(tVector_B);
  Kokkos::deep_copy(tVector_B_Host, tVector_B);
  TEST_FLOATING_EQUALITY(tVector_B_Host(0), 8.0, 1e-17);
  TEST_FLOATING_EQUALITY(tVector_B_Host(numVerts-1), 8.0, 1e-17);
}

}