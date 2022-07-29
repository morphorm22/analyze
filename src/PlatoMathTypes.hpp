/*
 * PlatoMathTypes.hpp
 *
 *  Created on: Oct 8, 2021
 */

#pragma once

#include "PlatoTypes.hpp"

namespace Plato
{

    /******************************************************************************//**
     * \brief Statically sized matrix
    **********************************************************************************/
    template <int M, int N>
    class Matrix
    {
        Plato::Scalar mData[M*N];

        public:
            KOKKOS_INLINE_FUNCTION Matrix() {}
            KOKKOS_INLINE_FUNCTION Matrix(Matrix<M,N> const & aMatrix)
            { 
                int k = 0;
                for (Plato::Scalar v : aMatrix.mData) { mData[k] = v; ++k; }
            }
            inline Matrix(std::initializer_list<Plato::Scalar> l)
            { 
                int k = 0;
                for (Plato::Scalar v : l) { mData[k] = v; ++k; }
            }
            KOKKOS_INLINE_FUNCTION Plato::Scalar& operator()(int i, int j){ return mData[i*N+j]; }
            KOKKOS_INLINE_FUNCTION Plato::Scalar operator()(int i, int j) const { return mData[i*N+j]; }
    };

    /******************************************************************************//**
     * \brief Statically sized array
    **********************************************************************************/
    template <int N>
    class Array
    {
        Plato::Scalar mData[N];

        public:
            KOKKOS_INLINE_FUNCTION Array() {}
            KOKKOS_INLINE_FUNCTION Array(Plato::Scalar aInit)
            {
                for (Plato::Scalar& v : mData) { v = aInit; }
            }
            KOKKOS_INLINE_FUNCTION Array(Array<N> const & aArray)
            { 
                int k = 0;
                for (Plato::Scalar v : aArray.mData) { mData[k] = v; ++k; }
            }
            inline Array(std::initializer_list<Plato::Scalar> l)
            { 
                int k = 0;
                for (Plato::Scalar v : l) { mData[k] = v; ++k; }
            }
            KOKKOS_INLINE_FUNCTION Plato::Scalar& operator()(int i){ return mData[i]; }
            KOKKOS_INLINE_FUNCTION Plato::Scalar operator()(int i) const { return mData[i]; }
            KOKKOS_INLINE_FUNCTION Plato::Scalar& operator[](int i){ return mData[i]; }
            KOKKOS_INLINE_FUNCTION Plato::Scalar operator[](int i) const { return mData[i]; }
    };

    KOKKOS_INLINE_FUNCTION
    Scalar determinant(Matrix<1,1> m)
    {
        return m(0,0);
    }

    KOKKOS_INLINE_FUNCTION
    Scalar determinant(Matrix<2,2> m)
    {
        Scalar a = m(0,0), b = m(1,0);
        Scalar c = m(0,1), d = m(1,1);
        return a * d - b * c;
    }

    KOKKOS_INLINE_FUNCTION
    Scalar determinant(Matrix<3,3> m)
    {
        Scalar a = m(0,0), b = m(1,0), c = m(2,0);
        Scalar d = m(0,1), e = m(1,1), f = m(2,1);
        Scalar g = m(0,2), h = m(1,2), i = m(2,2);
        return (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (b * d * i) - (a * f * h);
    }

    KOKKOS_INLINE_FUNCTION
    Matrix<1,1> invert(Matrix<1,1> const m)
    {
        Matrix<1,1> n;
        n(0,0) = 1.0 / m(0,0);
        return n;
    }

    KOKKOS_INLINE_FUNCTION
    Matrix<2,2> invert(Matrix<2,2> const m)
    {
        Matrix<2,2> n;
        Scalar det = determinant(m);
        n(0,0) = m(1,1) / det;
        n(0,1) = -m(0,1) / det;
        n(1,0) = -m(1,0) / det;
        n(1,1) = m(0,0) / det;
        return n;
    }

    KOKKOS_INLINE_FUNCTION
    Matrix<3,3> invert(Matrix<3,3> const a)
    {
        Matrix<3,3> n;
        Scalar det = determinant(a);
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
