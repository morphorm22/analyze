#pragma once

#include "PlatoMathTypes.hpp"

namespace Plato {

template <Plato::OrdinalType N, typename ScalarType>
KOKKOS_INLINE_FUNCTION
ScalarType normOffDiag(
    Plato::Matrix<N,N,ScalarType> const & aMatrix
)
{
    ScalarType tRetVal(0);
    for(Plato::OrdinalType j=0; j<N; j++)
    {
        for(Plato::OrdinalType i=0; i<N; i++)
        {
            if(i != j)
            {
                tRetVal += aMatrix(i,j)*aMatrix(i,j);
            }
        }
    }
    return sqrt(tRetVal);
}

template <Plato::OrdinalType N, typename ScalarType>
KOKKOS_INLINE_FUNCTION
Plato::Array<2,ScalarType> argMaxOffDiag(
    Plato::Matrix<N,N,ScalarType> const & aMatrix
)
{
    Plato::Array<2,ScalarType> tRetVal;
    Plato::OrdinalType p=0, q=0;
    ScalarType s(-1.0);
    for(Plato::OrdinalType j=0; j<N; j++)
    {
        for(Plato::OrdinalType i=0; i<N; i++)
        {
            ScalarType s2 = abs(aMatrix(i,j));
            if(i != j && s2 > s)
            {
                p = i;
                q = j;
                s = s2;
            }
        }
    }
    tRetVal(0) = p < q ? p : q;
    tRetVal(1) = p > q ? p : q;
    return tRetVal;
}

template <typename ScalarType>
KOKKOS_INLINE_FUNCTION
Plato::Array<2,ScalarType> schurSym(
    ScalarType f,
    ScalarType g,
    ScalarType h
)
{
    Plato::Array<2,ScalarType> tRetVal;
    tRetVal(0) = 1.0;
    tRetVal(1) = 0.0;
    if(g != 0)
    {
        ScalarType t = (h-f)/(2.0*g);
        if(t >= 0.0)
        {
            t = 1.0 / (sqrt(1.0+t*t)+t);
        }
        else
        {
            t = -1.0 / (sqrt(1.0+t*t)-t);
        }
        tRetVal(0) = 1.0 / sqrt(1.0+t*t);
        tRetVal(1) = t*tRetVal(0);
    }
    return tRetVal;
}

template <Plato::OrdinalType N, typename ScalarType>
KOKKOS_INLINE_FUNCTION
Plato::Matrix<N,N,ScalarType> givensLeft(
    ScalarType c, ScalarType s,
    Plato::OrdinalType i, Plato::OrdinalType k,
    Plato::Matrix<N,N,ScalarType> a
)
{
    for (Plato::OrdinalType j=0; j<N; j++)
    {
        auto t1 = a(i,j);
        auto t2 = a(k,j);
        a(i,j) = c*t1 - s*t2;
        a(k,j) = s*t1 + c*t2;
    }
    return a;
}

template <Plato::OrdinalType N, typename ScalarType>
KOKKOS_INLINE_FUNCTION
Plato::Matrix<N,N,ScalarType> givensRight(
    ScalarType c, ScalarType s,
    Plato::OrdinalType i, Plato::OrdinalType k,
    Plato::Matrix<N,N,ScalarType> a
)
{
    for (Plato::OrdinalType j=0; j<N; j++)
    {
        auto t1 = a(j,i);
        auto t2 = a(j,k);
        a(j,i) = c*t1 - s*t2;
        a(j,k) = s*t1 + c*t2;
    }
    return a;
}

/******************************************************************************//**
 * \brief Compute eigensystem.  The columns of aVectors are the eigenvectors which
 * are returned normalized.  The eigensystem is not sorted by eigenvalue magnitude.
**********************************************************************************/
template <Plato::OrdinalType N, typename ScalarType>
KOKKOS_INLINE_FUNCTION
void decomposeEigenJacobi(
    Plato::Matrix<N,N,ScalarType>   aMatrix,
    Plato::Matrix<N,N,ScalarType> & aVectors,
    Plato::Array<N,ScalarType>    & aValues
)
{
    constexpr Plato::OrdinalType tMaxIters = (5*N*N)/2;

    auto tVectors = Plato::identity<N,ScalarType>();

    auto tTolerance = Plato::norm(aMatrix) * DBL_EPSILON;

    Plato::OrdinalType tIteration=0;
    while (Plato::normOffDiag(aMatrix) > tTolerance && tIteration < tMaxIters)
    {
        auto pq = argMaxOffDiag(aMatrix);
        auto p = pq(0);
        auto q = pq(1);
        auto f = aMatrix(p, p);
        auto g = aMatrix(p, q);
        auto h = aMatrix(q, q);
        auto cs = schurSym(f, g, h);
        auto c = cs(0);
        auto s = cs(1);
        aMatrix = givensLeft(c, s, p, q, aMatrix);
        aMatrix = givensRight(c, s, p, q, aMatrix);
        tVectors = givensRight(c, s, p, q, tVectors);
        ++tIteration;
    }
    aVectors = tVectors;
    aValues = Plato::diagonal(aMatrix);
}

}
