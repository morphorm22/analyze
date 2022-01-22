/*
 * PlatoMathTypesTests.cpp
 *
 *  Created on: Oct 8, 2021
 */

#include "PlatoTestHelpers.hpp"

#include <Teuchos_UnitTestHarness.hpp>
#include <sstream>
#include <fstream>
#include <stdio.h>

#include <Omega_h_matrix.hpp>
#include "PlatoMathTypes.hpp"

namespace Plato
{
  namespace OmegaH {
    template <int M, int N>
    using Matrix = typename Omega_h::Matrix<M,N>;

    template <int N>
    using Vector = typename Omega_h::Vector<N>;
  }
}

namespace PlatoTestMathTypes
{

/******************************************************************************/
/*!
  \brief Matrix tests
*/
/******************************************************************************/
  TEUCHOS_UNIT_TEST(PlatoMathTypesTests, Matrix)
  {
    { // Omega Matrix
      { // 2x2 matrix
        Plato::OmegaH::Matrix<2,2> tMatrix({1.0, 2.0, 3.0, 4.0});
        TEST_FLOATING_EQUALITY(tMatrix(0,0), 1.0, 1e-12);
        tMatrix(0,1) = 5.0;
        TEST_FLOATING_EQUALITY(tMatrix(0,1), 5.0, 1e-12);
      }
      { // 2x3 matrix
        Plato::OmegaH::Matrix<2,3> tMatrix({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        TEST_FLOATING_EQUALITY(tMatrix(1,0), 4.0, 1e-12);
      }
    }
    { // Native Matrix
      { // 2x2 matrix
        Plato::Matrix<2,2> tMatrix({1.0, 2.0, 3.0, 4.0});
        TEST_FLOATING_EQUALITY(tMatrix(0,0), 1.0, 1e-12);
        tMatrix(0,1) = 5.0;
        TEST_FLOATING_EQUALITY(tMatrix(0,1), 5.0, 1e-12);
      }
      { // 2x3 matrix
        Plato::Matrix<2,3> tMatrix({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        TEST_FLOATING_EQUALITY(tMatrix(1,0), 4.0, 1e-12);
      }
      { // copy constructor
        Plato::Matrix<2,3> tMatrix({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        auto tMatrixCopy = tMatrix;
        TEST_FLOATING_EQUALITY(tMatrixCopy(1,0), 4.0, 1e-12);
        tMatrixCopy(1,0) = 6.0;
        TEST_FLOATING_EQUALITY(tMatrix(1,0), 4.0, 1e-12);
        TEST_FLOATING_EQUALITY(tMatrixCopy(1,0), 6.0, 1e-12);
        // const copy constructor
        const Plato::Matrix<2,3> tConstMatrix({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        auto tConstMatrixCopy = tConstMatrix;
        TEST_FLOATING_EQUALITY(tConstMatrixCopy(1,2), 6.0, 1e-12);
      }
      { // test invert() identity matrix
        const Plato::Matrix<3,3> tConstMatrix({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
        auto tInvMatrix = Plato::invert(tConstMatrix);
        TEST_FLOATING_EQUALITY(tInvMatrix(0,0), 1.0, 1e-16);
        TEST_FLOATING_EQUALITY(tInvMatrix(1,1), 1.0, 1e-16);
        TEST_FLOATING_EQUALITY(tInvMatrix(2,2), 1.0, 1e-16);
        TEST_FLOATING_EQUALITY(tInvMatrix(0,1), 0.0, 1e-16);
      }
      { // test determinant() and invert() 3x3 
        const Plato::Matrix<3,3> tConstMatrix({3.0, -1.0, 0.0, -1.0, 3.0, -1.0, 0.0, -1.0, 3.0});
        auto tInvMatrix = Plato::invert(tConstMatrix);
        TEST_FLOATING_EQUALITY(tInvMatrix(0,0), 8.0/21.0, 1e-16);
        TEST_FLOATING_EQUALITY(tInvMatrix(0,1), 1.0/7.0,  1e-16);
        TEST_FLOATING_EQUALITY(tInvMatrix(0,2), 1.0/21.0, 1e-16);
        TEST_FLOATING_EQUALITY(tInvMatrix(1,0), 1.0/7.0,  1e-16);
        TEST_FLOATING_EQUALITY(tInvMatrix(1,1), 3.0/7.0,  1e-16);
        TEST_FLOATING_EQUALITY(tInvMatrix(1,2), 1.0/7.0,  1e-16);
        TEST_FLOATING_EQUALITY(tInvMatrix(2,0), 1.0/21.0, 1e-16);
        TEST_FLOATING_EQUALITY(tInvMatrix(2,1), 1.0/7.0,  1e-16);
        TEST_FLOATING_EQUALITY(tInvMatrix(2,2), 8.0/21.0, 1e-16);

        TEST_FLOATING_EQUALITY(Plato::determinant(tConstMatrix), 21.0, 1e-16);
      }
      { // test determinant() and invert() 2x2
        const Plato::Matrix<2,2> tConstMatrix({3.0, -1.0, -1.0, 3.0});
        auto tInvMatrix = Plato::invert(tConstMatrix);
        TEST_FLOATING_EQUALITY(tInvMatrix(0,0), 3.0/8.0, 1e-16);
        TEST_FLOATING_EQUALITY(tInvMatrix(0,1), 1.0/8.0,  1e-16);
        TEST_FLOATING_EQUALITY(tInvMatrix(1,0), 1.0/8.0,  1e-16);
        TEST_FLOATING_EQUALITY(tInvMatrix(1,1), 3.0/8.0,  1e-16);

        TEST_FLOATING_EQUALITY(Plato::determinant(tConstMatrix), 8.0, 1e-16);
      }
      { // test determinant() and invert() 1x1
        const Plato::Matrix<1,1> tConstMatrix({3.0});
        auto tInvMatrix = Plato::invert(tConstMatrix);
        TEST_FLOATING_EQUALITY(tInvMatrix(0,0), 1.0/3.0, 1e-16);

        TEST_FLOATING_EQUALITY(Plato::determinant(tConstMatrix), 3.0, 1e-16);
      }
      { // test determinant() and invert() 3x3 on device
        int tNumData = 10;
        Plato::ScalarArray3D tData("data", tNumData,3,3);
        const Plato::Matrix<3,3> tConstMatrix({3.0, -1.0, 0.0, -1.0, 3.0, -1.0, 0.0, -1.0, 3.0});
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumData), LAMBDA_EXPRESSION(const Plato::OrdinalType & tIndex) {
          auto tInvMatrix = Plato::invert(tConstMatrix);
          for (int i=0; i<3; i++){
            for (int j=0; j<3; j++){
              tData(tIndex,i,j) = tInvMatrix(i,j);
            }
          }
        }, "loop");
        auto tHostData = Kokkos::create_mirror(tData);
        Kokkos::deep_copy(tHostData, tData);
        for (int k=0; k<tNumData; k++){
          TEST_FLOATING_EQUALITY(tHostData(k,0,0), 8.0/21.0, 1e-16);
          TEST_FLOATING_EQUALITY(tHostData(k,0,1), 1.0/7.0,  1e-16);
          TEST_FLOATING_EQUALITY(tHostData(k,0,2), 1.0/21.0, 1e-16);
          TEST_FLOATING_EQUALITY(tHostData(k,1,0), 1.0/7.0,  1e-16);
          TEST_FLOATING_EQUALITY(tHostData(k,1,1), 3.0/7.0,  1e-16);
          TEST_FLOATING_EQUALITY(tHostData(k,1,2), 1.0/7.0,  1e-16);
          TEST_FLOATING_EQUALITY(tHostData(k,2,0), 1.0/21.0, 1e-16);
          TEST_FLOATING_EQUALITY(tHostData(k,2,1), 1.0/7.0,  1e-16);
          TEST_FLOATING_EQUALITY(tHostData(k,2,2), 8.0/21.0, 1e-16);
        }
      }
    }
  }
}
