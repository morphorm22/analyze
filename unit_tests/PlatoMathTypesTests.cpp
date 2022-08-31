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

#include "PlatoMathTypes.hpp"
#include "PlatoEigen.hpp"

namespace PlatoTestMathTypes
{

/******************************************************************************/
/*!
  \brief Matrix tests
*/
/******************************************************************************/
  TEUCHOS_UNIT_TEST(PlatoMathTypesTests, Matrix)
  {
    {
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
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumData), KOKKOS_LAMBDA(const Plato::OrdinalType & tIndex) {
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

  TEUCHOS_UNIT_TEST(PlatoMathTypesTests, Math)
  {
    // dot product
    {
      Plato::Array<3,double> v1({1,1,1}), v2({1,1,1});
      TEST_FLOATING_EQUALITY(Plato::dot(v1,v2), 3.0, DBL_EPSILON);
    }
    {
      Plato::Array<3,double> v1({1,0,0}), v2({0,1,0});
      TEST_FLOATING_EQUALITY(Plato::dot(v1,v2), 0.0, DBL_EPSILON);
    }
    // diagonal
    {
      Plato::Matrix<4,4,double> m1({1,2,3,4, 2,3,4,1, 3,4,1,2, 4,1,2,3});
      auto d1 = Plato::diagonal(m1);
      TEST_FLOATING_EQUALITY(d1(0), 1.0, DBL_EPSILON);
      TEST_FLOATING_EQUALITY(d1(1), 3.0, DBL_EPSILON);
      TEST_FLOATING_EQUALITY(d1(2), 1.0, DBL_EPSILON);
      TEST_FLOATING_EQUALITY(d1(3), 3.0, DBL_EPSILON);
    }
    // norm of vector
    {
      Plato::Array<3,double> v1({1,1,1});
      TEST_FLOATING_EQUALITY(Plato::norm(v1), sqrt(3.0), DBL_EPSILON);
    }
    // norm of matrix
    {
      Plato::Matrix<3,3,double> m1({1,1,1, 1,1,1, 1,1,1});
      auto d1 = Plato::norm(m1);
      TEST_FLOATING_EQUALITY(d1, sqrt(9.0), DBL_EPSILON);
    }
    // sum of two matrices
    {
      Plato::Matrix<3,3,double> m1({2,2,2, 2,2,2, 2,2,2});
      Plato::Matrix<3,3,double> m2({1,1,1, 1,1,1, 1,1,1});
      auto d1 = Plato::plus(m1,m2,-1.0);
      TEST_FLOATING_EQUALITY(Plato::norm(d1), sqrt(9.0), DBL_EPSILON);
      TEST_FLOATING_EQUALITY(d1(1,1), 1.0, DBL_EPSILON);
    }
    // transpose of matrix
    {
      Plato::Matrix<3,3,double> m1({1,2,3, 4,5,6, 7,8,9});
      auto m2 = Plato::transpose(m1);
      TEST_FLOATING_EQUALITY(m2(0,0), 1.0, DBL_EPSILON);
      TEST_FLOATING_EQUALITY(m2(1,0), 2.0, DBL_EPSILON);
      TEST_FLOATING_EQUALITY(m2(0,1), 4.0, DBL_EPSILON);
    }
    // multiplication of two matrices
    {
      Plato::Matrix<3,3,double> m1({2,2,2, 2,2,2, 2,2,2});
      Plato::Matrix<3,3,double> m2({1,1,1, 1,1,1, 1,1,1});
      auto m3 = Plato::times(m1,m2);
      TEST_FLOATING_EQUALITY(Plato::norm(m3), sqrt(36.0*9.0), DBL_EPSILON);
      TEST_FLOATING_EQUALITY(m3(0,0), 6.0, DBL_EPSILON);
    }
    // multiplication of scalar times matrix
    {
      Plato::Matrix<3,3,double> m1({2,2,2, 2,2,2, 2,2,2});
      auto m2 = Plato::times(2.0,m1);
      TEST_FLOATING_EQUALITY(Plato::norm(m2), sqrt(16.0*9.0), DBL_EPSILON);
      TEST_FLOATING_EQUALITY(m2(0,0), 4.0, DBL_EPSILON);
    }
    // identity matrix
    {
      auto m1 = Plato::identity<3,double>();
      TEST_FLOATING_EQUALITY(Plato::norm(m1), sqrt(3.0), DBL_EPSILON);
      TEST_FLOATING_EQUALITY(m1(0,0), 1.0, DBL_EPSILON);
    }
    // scaled identity matrix
    {
      auto m1 = Plato::identity<3,double>(2.0);
      TEST_FLOATING_EQUALITY(Plato::norm(m1), sqrt(12.0), DBL_EPSILON);
      TEST_FLOATING_EQUALITY(m1(0,0), 2.0, DBL_EPSILON);
    }
    // outer product
    {
      Plato::Array<3,double> v1({1.0,2.0,3.0});
      auto m1 = Plato::outer_product(v1,v1);
      TEST_FLOATING_EQUALITY(m1(0,0), 1.0, DBL_EPSILON);
      TEST_FLOATING_EQUALITY(m1(0,1), 2.0, DBL_EPSILON);
      TEST_FLOATING_EQUALITY(m1(0,2), 3.0, DBL_EPSILON);
      TEST_FLOATING_EQUALITY(m1(1,0), 2.0, DBL_EPSILON);
      TEST_FLOATING_EQUALITY(m1(1,1), 4.0, DBL_EPSILON);
      TEST_FLOATING_EQUALITY(m1(1,2), 6.0, DBL_EPSILON);
      TEST_FLOATING_EQUALITY(m1(2,0), 3.0, DBL_EPSILON);
      TEST_FLOATING_EQUALITY(m1(2,1), 6.0, DBL_EPSILON);
      TEST_FLOATING_EQUALITY(m1(2,2), 9.0, DBL_EPSILON);
    }
    // normalize vector
    {
      Plato::Array<3,double> v1({1.0,1.0,1.0});
      auto v2 = Plato::normalize(v1);
      TEST_FLOATING_EQUALITY(v2(0), 1.0/sqrt(3.0), DBL_EPSILON);
      TEST_FLOATING_EQUALITY(v2(1), 1.0/sqrt(3.0), DBL_EPSILON);
      TEST_FLOATING_EQUALITY(v2(2), 1.0/sqrt(3.0), DBL_EPSILON);
    }
  }
  TEUCHOS_UNIT_TEST(PlatoMathTypesTests, Eigen)
  {
    // decomposeEigenJacobi() returns the eigensystem unsorted and normalized
    // gold generated with mathematica:
    // input: {a,B} = Eigensystem[{2,0,0},{0,3,4},{0,4,9}}]
    // input: {a[[2]], a[[3]], a[[1]]}
    // input: {Normalize[B[[2]]], Normalize[B[[3]]], Normalize[B[[1]]]}
    // output: {2, 1, 11}
    // output: {{1, 0, 0}, {0,-2/sqrt(5),1/sqrt(5)},{0,1/sqrt(5),2/sqrt(5)}}
    Plato::Matrix<3,3> tMatrix({2, 0, 0, 0, 3, 4, 0, 4, 9});
    Plato::Matrix<3,3> tVectors;
    Plato::Array<3> tValues;
    Plato::decomposeEigenJacobi( tMatrix, tVectors, tValues );

    TEST_FLOATING_EQUALITY(tVectors(0,0), 1, DBL_EPSILON);
    TEST_FLOATING_EQUALITY(tVectors(1,0), 0, DBL_EPSILON);
    TEST_FLOATING_EQUALITY(tVectors(2,0), 0, DBL_EPSILON);

    TEST_FLOATING_EQUALITY(tVectors(0,1), 0, DBL_EPSILON);
    TEST_FLOATING_EQUALITY(tVectors(1,1), 2.0/sqrt(5.0), DBL_EPSILON);
    TEST_FLOATING_EQUALITY(tVectors(2,1), -1.0/sqrt(5.0), DBL_EPSILON);

    TEST_FLOATING_EQUALITY(tVectors(0,2), 0, DBL_EPSILON);
    TEST_FLOATING_EQUALITY(tVectors(1,2), 1.0/sqrt(5.0), DBL_EPSILON);
    TEST_FLOATING_EQUALITY(tVectors(2,2), 2.0/sqrt(5.0), DBL_EPSILON);

    TEST_FLOATING_EQUALITY(tValues(0), 2.0, DBL_EPSILON);
    TEST_FLOATING_EQUALITY(tValues(1), 1.0, DBL_EPSILON);
    TEST_FLOATING_EQUALITY(tValues(2), 11.0, DBL_EPSILON);
  }
}
