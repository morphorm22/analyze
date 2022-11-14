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

            explicit
            KOKKOS_INLINE_FUNCTION Matrix(ScalarType aInit)
            {
                for (ScalarType& v : mData) { v = aInit; }
            }
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

    /******************************************************************************//**
     * \brief Returns the dot product of two Arrays of same length and type
    **********************************************************************************/
    template <Plato::OrdinalType N, typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    ScalarType dot(Array<N,ScalarType> m1, Array<N,ScalarType> m2)
    {
        ScalarType tRetVal(0.0);
        for(Plato::OrdinalType i=0; i<N; i++)
        {
            tRetVal += m1(i)*m2(i);
        }
        return tRetVal;
    }

    /******************************************************************************//**
     * \brief Returns an Array containing the diagonal entries in the input Matrix
    **********************************************************************************/
    template <Plato::OrdinalType N, typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    Plato::Array<N,ScalarType> diagonal(Matrix<N,N,ScalarType> const & m1)
    {
        Plato::Array<N,ScalarType> tRetVal;
        for(Plato::OrdinalType i=0; i<N; i++)
        {
            tRetVal(i) = m1(i,i);
        }
        return tRetVal;
    }

    /******************************************************************************//**
     * \brief Returns the norm (sqrt(dot(v,v))) of the input Array
    **********************************************************************************/
    template <Plato::OrdinalType N, typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    ScalarType norm(Array<N,ScalarType> const & m1)
    {
        ScalarType tRetVal(0.0);
        for(Plato::OrdinalType i=0; i<N; i++)
        {
            tRetVal += m1(i)*m1(i);
        }
        return sqrt(tRetVal);
    }

    /******************************************************************************//**
     * \brief Returns the norm (sqrt(sum_{i,j}(m(i,j)*m(i,j))) of the input Matrix
    **********************************************************************************/
    template <Plato::OrdinalType N, typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    ScalarType norm(Matrix<N,N,ScalarType> const & m1)
    {
        ScalarType tRetVal(0.0);
        for(Plato::OrdinalType i=0; i<N; i++)
        {
            for(Plato::OrdinalType j=0; j<N; j++)
            {
                tRetVal += m1(i,j)*m1(i,j);
            }
        }
        return sqrt(tRetVal);
    }

    /******************************************************************************//**
     * \brief Returns the sum of two matrices.  The second matrix is scaled by the 
     * optional third argument.
    **********************************************************************************/
    template <Plato::OrdinalType N, typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    Matrix<N,N,ScalarType> plus(Matrix<N,N,ScalarType> m1, Matrix<N,N,ScalarType> m2, ScalarType scale=1.0)
    {
        Matrix<N,N,ScalarType> tRetVal(0);
        for(Plato::OrdinalType i=0; i<N; i++)
        {
            for(Plato::OrdinalType j=0; j<N; j++)
            {
                tRetVal(i,j) = m1(i,j) + scale*m2(i,j);
            }
        }
        return tRetVal;
    }

    /******************************************************************************//**
     * \brief Returns the transpose (t_{i,j} = m_{j,i}) of the input matrix
    **********************************************************************************/
    template <Plato::OrdinalType N, typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    Matrix<N,N,ScalarType> transpose(Matrix<N,N,ScalarType> m1)
    {
        Matrix<N,N,ScalarType> tRetVal(0);
        for(Plato::OrdinalType i=0; i<N; i++)
        {
            for(Plato::OrdinalType j=0; j<N; j++)
            {
                tRetVal(i,j) = m1(j,i);
            }
        }
        return tRetVal;
    }

    /******************************************************************************//**
     * \brief Returns the product of the input matrices
    **********************************************************************************/
    template <Plato::OrdinalType N, typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    Matrix<N,N,ScalarType> times(Matrix<N,N,ScalarType> m1, Matrix<N,N,ScalarType> m2)
    {
        Matrix<N,N,ScalarType> tRetVal(0);
        for(Plato::OrdinalType i=0; i<N; i++)
        {
            for(Plato::OrdinalType j=0; j<N; j++)
            {
                for(Plato::OrdinalType k=0; k<N; k++)
                {
                    tRetVal(i,j) += m1(i,k)*m2(k,j);
                }
            }
        }
        return tRetVal;
    }

    /******************************************************************************//**
     * \brief Returns the matrix argument times the scalar argument
    **********************************************************************************/
    template <Plato::OrdinalType N, typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    Matrix<N,N,ScalarType> times(ScalarType v1, Matrix<N,N,ScalarType> m1)
    {
        Matrix<N,N,ScalarType> tRetVal(0);
        for(Plato::OrdinalType i=0; i<N; i++)
        {
            for(Plato::OrdinalType j=0; j<N; j++)
            {
                tRetVal(i,j) = m1(i,j)*v1;
            }
        }
        return tRetVal;
    }

    /******************************************************************************//**
     * \brief Returns the array argument times the scalar argument
    **********************************************************************************/
    template <Plato::OrdinalType N, typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    Array<N,ScalarType> times(ScalarType s1, Array<N,ScalarType> v1)
    {
        Array<N,ScalarType> tRetVal(0);
        for(Plato::OrdinalType i=0; i<N; i++)
        {
            tRetVal(i) = v1(i)*s1;
        }
        return tRetVal;
    }

    /******************************************************************************//**
     * \brief Returns the identity matrix of size N scaled by optional argument, a.
    **********************************************************************************/
    template <Plato::OrdinalType N, typename ScalarType=Plato::Scalar>
    KOKKOS_INLINE_FUNCTION
    Matrix<N,N,ScalarType> identity(ScalarType a=1.0)
    {
        Matrix<N,N,ScalarType> tRetVal(0);
        for(Plato::OrdinalType i=0; i<N; i++)
        {
            tRetVal(i,i) = ScalarType(a);
        }
        return tRetVal;
    }

    /******************************************************************************//**
     * \brief Returns the outer product of two Arrays of the same length.
    **********************************************************************************/
    template <Plato::OrdinalType N, typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    Matrix<N,N,ScalarType> outer_product(Array<N,ScalarType> m1, Array<N,ScalarType> m2)
    {
        Matrix<N,N,ScalarType> tRetVal(0);
        for(Plato::OrdinalType i=0; i<N; i++)
        {
            for(Plato::OrdinalType j=0; j<N; j++)
            {
                tRetVal(i,j) = m1(i)*m2(j);
            }
        }
        return tRetVal;
    }

    /******************************************************************************//**
     * \brief Returns the vector normalized.
    **********************************************************************************/
    template <Plato::OrdinalType N, typename ScalarType>
    KOKKOS_INLINE_FUNCTION
    Plato::Array<N,ScalarType> normalize(Array<N,ScalarType> v1)
    {
        auto tMag = Plato::norm(v1);
        return Plato::times(1.0/tMag, v1);
    }

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
