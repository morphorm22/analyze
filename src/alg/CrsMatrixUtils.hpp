#ifndef CRSMATRIXUTILS_HPP_
#define CRSMATRIXUTILS_HPP_

#include "CrsMatrix.hpp"
#include "PlatoMathHelpers.hpp"

#include <boost/functional/hash.hpp>

#include <cmath> // isnan
#include <fstream>
#include <iostream>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace Plato
{
template<typename Ordinal>
std::size_t crs_matrix_row_column_hash(
    typename Plato::CrsMatrix<Ordinal>::RowMapVectorT aRowBegin,
    typename Plato::CrsMatrix<Ordinal>::OrdinalVectorT aColumns);

template<typename Ordinal>
void print_vector_to_file(
    typename Plato::CrsMatrix<Ordinal>::ScalarVectorT aValues,
    const std::string& aFileName);

template<typename Ordinal>
bool has_nan(typename Plato::CrsMatrix<Ordinal>::ScalarVectorT aValues);

template<typename Ordinal>
void print_matrix_to_file(
    typename Plato::CrsMatrix<Ordinal>::RowMapVectorT aRowBegin,
    typename Plato::CrsMatrix<Ordinal>::OrdinalVectorT aColumns,
    typename Plato::CrsMatrix<Ordinal>::ScalarVectorT aValues,
    const std::string& aFileName);

template<typename Ordinal>
bool has_symmetric_sparsity_pattern(
    const typename Plato::CrsMatrix<Ordinal>::RowMapVectorT& aRowBegin,
    const typename Plato::CrsMatrix<Ordinal>::OrdinalVectorT& aColumns);

/// Checks that the sparsity pattern of @a aMatrix is symmetric. Algorithm is
/// linear in time and space in the number of non-zero entries.
template<typename Ordinal>
bool has_symmetric_sparsity_pattern(CrsMatrix<Ordinal>& aMatrix);

// Implementations
namespace detail{
struct OrdinalPairHash final
{
    template<typename Ordinal>
    std::size_t operator()(const std::pair<Ordinal, Ordinal>& aPair) const
    {
        std::size_t tSeed = 0;
        boost::hash_combine(tSeed, aPair.first);
        boost::hash_combine(tSeed, aPair.second);
        return tSeed;
    }
};

template<typename Ordinal>
std::pair<Ordinal, Ordinal> matrix_indices_transpose(const std::pair<Ordinal, Ordinal>& aEntry)
{
    return std::make_pair(aEntry.second, aEntry.first);
}

template<typename Ordinal>
using RowVectorOnHost = typename Plato::CrsMatrix<Ordinal>::RowMapVectorT::HostMirror;

template<typename Ordinal>
using OrdinalVectorOnHost = typename Plato::CrsMatrix<Ordinal>::OrdinalVectorT::HostMirror;

template<typename Ordinal>
using ScalarVectorOnHost = typename Plato::CrsMatrix<Ordinal>::ScalarVectorT::HostMirror;

template<typename KokkosViewLike>
typename KokkosViewLike::HostMirror host_mirror(const KokkosViewLike& aViewLike)
{
#if defined(KOKKOS_ENABLE_CUDA)
    auto tViewLikeOnHost = Kokkos::create_mirror_view(aViewLike);
    Kokkos::deep_copy(tViewLikeOnHost, aViewLike);
#else
    typename KokkosViewLike::HostMirror tViewLikeOnHost(aViewLike);
#endif
    return tViewLikeOnHost;
}

template<typename KokkosLike>
std::size_t hash_vector(const KokkosLike& aVector)
{
    std::size_t tSeed = 0;
    auto tVectorOnHost = detail::host_mirror(aVector);
    for(int i = 0; i < tVectorOnHost.size() - 1; ++i)
    {
        boost::hash_combine(tSeed, tVectorOnHost(i));
    }
    return tSeed;
}

template<typename Ordinal, typename Callable>
void for_each_row_column(
    const RowVectorOnHost<Ordinal>& aRowBegin,
    const OrdinalVectorOnHost<Ordinal>& aColumns,
    const ScalarVectorOnHost<Ordinal>& aValues,
    const Callable& aCallable)
{
    for(int i = 0; i < aRowBegin.size() - 1; ++i)
    {
        const Ordinal tColsBegin = aRowBegin(i);
        const Ordinal tColsEnd = aRowBegin(i + 1);
        for(int j = tColsBegin; j < tColsEnd; ++j)
        {
            aCallable(i, aColumns(j), aValues(j));
        }
    }   
}

/// Iterates over all row/column indices given by @a aRowBegin and @a aColumns, applying 
///  @a aCallable to each entry. If @a aCallable returns `false`, this will exit early and
///  also return `false`. Returns `true` otherwise.
/// @a Callable must return `bool`.
template<typename Ordinal, typename Callable>
bool for_each_row_column(
    const RowVectorOnHost<Ordinal>& aRowBegin,
    const OrdinalVectorOnHost<Ordinal>& aColumns,
    const Callable& aCallable)
{
    static_assert(std::is_same<std::result_of_t<Callable(Ordinal, Ordinal)>, bool>::value, "aCallable must be a function returning bool");
    for(int i = 0; i < aRowBegin.size() - 1; ++i)
    {
        const Ordinal tColsBegin = aRowBegin(i);
        const Ordinal tColsEnd = aRowBegin(i + 1);
        for(int j = tColsBegin; j < tColsEnd; ++j)
        {
            if(!aCallable(i, aColumns(j)))
            {
                return false;
            }
        }
    }
    return true;
}
}

template<typename Ordinal>
std::size_t crs_matrix_row_column_hash(
    typename Plato::CrsMatrix<Ordinal>::RowMapVectorT aRowBegin,
    typename Plato::CrsMatrix<Ordinal>::OrdinalVectorT aColumns)
{
    std::size_t tSeed = detail::hash_vector(aRowBegin);
    boost::hash_combine(tSeed, detail::hash_vector(aColumns));
    return tSeed;
}

template<typename Ordinal>
void print_vector_to_file(
    typename Plato::CrsMatrix<Ordinal>::ScalarVectorT aValues,
    const std::string& aFileName)
{
    std::ofstream tFileStream;
    tFileStream.open(aFileName);

    auto tValuesOnHost = detail::host_mirror(aValues);

    tFileStream << "b = [";
    for(int i = 0; i < tValuesOnHost.size(); ++i){
        tFileStream << tValuesOnHost(i) << '\n';
    }
    tFileStream << "];\n";
}

template<typename Ordinal>
bool has_nan(typename Plato::CrsMatrix<Ordinal>::ScalarVectorT aValues)
{
    auto tValuesOnHost = detail::host_mirror(aValues);
    for(int i = 0; i < tValuesOnHost.size(); ++i)
    {
        if(std::isnan(tValuesOnHost(i)))
        {
            return true;
        }
    }
    return false;
}

template<typename Ordinal>
void print_matrix_to_file(
    typename Plato::CrsMatrix<Ordinal>::RowMapVectorT aRowBegin,
    typename Plato::CrsMatrix<Ordinal>::OrdinalVectorT aColumns,
    typename Plato::CrsMatrix<Ordinal>::ScalarVectorT aValues,
    const std::string& aFileName)
{
    std::ofstream tFileStream;
    tFileStream.open(aFileName);

    auto tRowBeginOnHost = detail::host_mirror(aRowBegin);
    auto tColumnsOnHost = detail::host_mirror(aColumns);
    auto tValuesOnHost = detail::host_mirror(aValues);

    tFileStream << "rows = [";
    detail::for_each_row_column<Ordinal>(tRowBeginOnHost, tColumnsOnHost, tValuesOnHost, 
        [&tFileStream](const Ordinal aRow, const Ordinal /*aColumn*/, const Plato::Scalar /*aValue*/)
        {
            tFileStream << aRow + 1 << '\n';
        });
    tFileStream << "];\n";

    tFileStream << "columns = [";
    detail::for_each_row_column<Ordinal>(tRowBeginOnHost, tColumnsOnHost, tValuesOnHost, 
        [&tFileStream](const Ordinal /*aRow*/, const Ordinal aColumn, const Plato::Scalar /*aValue*/)
        {
            tFileStream << aColumn + 1 << '\n';
        });
    tFileStream << "];\n";

    tFileStream << "values = [";
    tFileStream.precision(16);
    detail::for_each_row_column<Ordinal>(tRowBeginOnHost, tColumnsOnHost, tValuesOnHost, 
        [&tFileStream](const Ordinal /*aRow*/, const Ordinal /*aColumn*/, const Plato::Scalar aValue)
        {
            tFileStream << aValue << '\n';
        });
    tFileStream << "];\n";
    tFileStream.close();
}

/// Checks that the sparsity pattern of @a aMatrix is symmetric. Algorithm is
/// linear in time and space in the number of non-zero entries.
template<typename Ordinal>
bool has_symmetric_sparsity_pattern(
    const typename Plato::CrsMatrix<Ordinal>::RowMapVectorT& aRowBegin,
    const typename Plato::CrsMatrix<Ordinal>::OrdinalVectorT& aColumns)
{
    auto tRowBeginOnHost = detail::host_mirror(aRowBegin);
    auto tColumnsOnHost = detail::host_mirror(aColumns);

    std::unordered_set<std::pair<Ordinal, Ordinal>, detail::OrdinalPairHash> tFoundEntries;
    detail::for_each_row_column<Ordinal>(tRowBeginOnHost, tColumnsOnHost, 
    [&tFoundEntries](const Ordinal row, const Ordinal column)
    {
        const auto tRowColIndex = std::make_pair(row, column);
        if(tRowColIndex.second > tRowColIndex.first)
        {
            // Upper triangular entries are inserted and checked against lower triangular entries
            tFoundEntries.insert(tRowColIndex);
        }
        else if(tRowColIndex.first != tRowColIndex.second) // Don't check diagonal entries
        {
            // Check that this lower triangular entry has an entry in the set
            const auto tIter = tFoundEntries.find(detail::matrix_indices_transpose(tRowColIndex));
            if(tIter == tFoundEntries.end())
            {
                // No upper triangular entry matching this lower triangular entry, so the pattern isn't symmetric.
                return false;
            }
            else
            {
                tFoundEntries.erase(tIter);
            }
        }
        return true;
    });
    // If tFoundEntries isn't empty, there was at least one upper triangular entry not matching a lower triangular entry.
    return tFoundEntries.empty();
}

/// Checks that the sparsity pattern of @a aMatrix is symmetric. Algorithm is
/// linear in time and space in the number of non-zero entries.
template<typename Ordinal>
bool has_symmetric_sparsity_pattern(CrsMatrix<Ordinal>& aMatrix)
{
    typename Plato::CrsMatrix<Ordinal>::RowMapVectorT tRowBegin;
    typename Plato::CrsMatrix<Ordinal>::OrdinalVectorT tColumns;
    typename Plato::CrsMatrix<Ordinal>::ScalarVectorT tValues;
    if (aMatrix.isBlockMatrix()) {
        // if there were a version of this function that only fills values, could avoid copies of indices on subsequent calls
        auto aAptr = Teuchos::rcp(&aMatrix, false);
        Plato::getDataAsNonBlock(aAptr, tRowBegin, tColumns, tValues);
    } else {
        tRowBegin = aMatrix.rowMap();
        tColumns = aMatrix.columnIndices();
    }
    return has_symmetric_sparsity_pattern<Ordinal>(tRowBegin, tColumns);
}


}

#endif