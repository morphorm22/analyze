/*
 * PlatoMathTypes.hpp
 *
 *  Created on: Oct 8, 2021
 */

#pragma once

#include "alg/PlatoLambda.hpp"
#include "PlatoTypes.hpp"

namespace Plato
{

    /******************************************************************************//**
     * \brief Statically sized array
    **********************************************************************************/
    template <int N, typename ScalarType = Plato::Scalar>
    class Array
    {
        ScalarType mData[N];

        public:
            KOKKOS_INLINE_FUNCTION Array() {}
            KOKKOS_INLINE_FUNCTION Array(ScalarType aInit)
            {
                for (ScalarType& v : mData) { v = aInit; }
            }
            KOKKOS_INLINE_FUNCTION Array(Array<N,ScalarType> const & aArray)
            { 
                int k = 0;
                for (ScalarType v : aArray.mData) { mData[k] = v; ++k; }
            }
            inline Array(std::initializer_list<ScalarType> l)
            { 
                int k = 0;
                for (ScalarType v : l) { mData[k] = v; ++k; }
            }
            KOKKOS_INLINE_FUNCTION ScalarType& operator()(int i)       { return mData[i]; }
            KOKKOS_INLINE_FUNCTION ScalarType  operator()(int i) const { return mData[i]; }
            KOKKOS_INLINE_FUNCTION ScalarType& operator[](int i)       { return mData[i]; }
            KOKKOS_INLINE_FUNCTION ScalarType  operator[](int i) const { return mData[i]; }

            KOKKOS_INLINE_FUNCTION Plato::OrdinalType size() const { return N; }
    };

    /******************************************************************************//**
     * \brief Statically sized matrix
    **********************************************************************************/
    template <int M, int N, typename ScalarType = Plato::Scalar>
    class Matrix
    {
        ScalarType mData[M*N];

        public:
            KOKKOS_INLINE_FUNCTION Matrix() {}
            KOKKOS_INLINE_FUNCTION Matrix(Matrix<M,N> const & aMatrix)
            { 
                int k = 0;
                for (ScalarType v : aMatrix.mData) { mData[k] = v; ++k; }
            }
            inline Matrix(std::initializer_list<ScalarType> l)
            { 
                int k = 0;
                for (ScalarType v : l) { mData[k] = v; ++k; }
            }
            KOKKOS_INLINE_FUNCTION ScalarType& operator()(int i, int j)       { return mData[i*N+j]; }
            KOKKOS_INLINE_FUNCTION ScalarType  operator()(int i, int j) const { return mData[i*N+j]; }

            KOKKOS_INLINE_FUNCTION Plato::Array<N, ScalarType> operator()(int iRow) const
            {
                Plato::Array<N> tArray;
                for (Plato::OrdinalType iCol=0; iCol<N; iCol++)
                {
                    tArray[iCol] = mData[iRow*N+iCol];
                }
                return tArray;
            }
    };

    template <typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    ScalarType determinant(Matrix<1,1,ScalarType> m)
    {
        return m(0,0);
    }

    template <typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    ScalarType determinant(Matrix<2,2,ScalarType> m)
    {
        ScalarType a = m(0,0), b = m(1,0);
        ScalarType c = m(0,1), d = m(1,1);
        return a * d - b * c;
    }

    template <typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    ScalarType determinant(Matrix<3,3,ScalarType> m)
    {
        ScalarType a = m(0,0), b = m(1,0), c = m(2,0);
        ScalarType d = m(0,1), e = m(1,1), f = m(2,1);
        ScalarType g = m(0,2), h = m(1,2), i = m(2,2);
        return (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (b * d * i) - (a * f * h);
    }

    template <typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    Matrix<1,1,ScalarType> invert(Matrix<1,1,ScalarType> const m)
    {
        Matrix<1,1,ScalarType> n;
        n(0,0) = 1.0 / m(0,0);
        return n;
    }

    template <typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    Matrix<2,2,ScalarType> invert(Matrix<2,2,ScalarType> const m)
    {
        Matrix<2,2,ScalarType> n;
        ScalarType det = determinant(m);
        n(0,0) = m(1,1) / det;
        n(0,1) = -m(0,1) / det;
        n(1,0) = -m(1,0) / det;
        n(1,1) = m(0,0) / det;
        return n;
    }

    template <typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    Matrix<3,3,ScalarType> invert(Matrix<3,3,ScalarType> const a)
    {
        Matrix<3,3,ScalarType> n;
        ScalarType det = determinant(a);
        n(0,0) = (a(1,1)*a(2,2)-a(1,2)*a(2,1)) / det;
        n(0,1) = (a(0,2)*a(2,1)-a(0,1)*a(2,2)) / det;
        n(0,2) = (a(0,1)*a(1,2)-a(0,2)*a(1,1)) / det;
        n(1,0) = (a(1,2)*a(2,0)-a(1,0)*a(2,2)) / det;
        n(1,1) = (a(0,0)*a(2,2)-a(0,2)*a(2,0)) / det;
        n(1,2) = (a(0,2)*a(1,0)-a(0,0)*a(1,2)) / det;
        n(2,0) = (a(1,0)*a(2,1)-a(1,1)*a(2,0)) / det;
        n(2,1) = (a(0,1)*a(2,0)-a(0,0)*a(2,1)) / det;
        n(2,2) = (a(0,0)*a(1,1)-a(0,1)*a(1,0)) / det;
        return n;
    }
} // namespace Plato
