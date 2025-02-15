#include <solver/CrsMatrixUtils.hpp>

#include <Teuchos_UnitTestHarness.hpp>

#include <array>

TEUCHOS_UNIT_TEST(CrsMatrixUtils, SparsityPatternSymmetric)
{
    constexpr int tNumRows = 4;
    constexpr int tNumValues = 10;
    std::array<int, tNumRows + 1> tRowBegin = {0, 2, 5, 8, tNumValues}; 
    std::array<int, tNumValues> tColumns = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::array<double, tNumValues> tValues = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};

    typename Plato::CrsMatrix<int>::OrdinalVectorT tColumnsAsVector{tColumns.data(), tColumns.size()};
    typename Plato::CrsMatrix<int>::RowMapVectorT tRowBeginAsVector{tRowBegin.data(), tRowBegin.size()};
    typename Plato::CrsMatrix<int>::ScalarVectorT tValuesAsVector{tValues.data(), tValues.size()};
    Plato::CrsMatrix<int> tMatrix{tRowBeginAsVector, tColumnsAsVector, tValuesAsVector, tNumRows, tNumRows, 1, 1};
    TEST_ASSERT(Plato::has_symmetric_sparsity_pattern(tMatrix));
}

TEUCHOS_UNIT_TEST(CrsMatrixUtils, SparsityPatternSymmetricFull)
{
    constexpr int tNumRows = 4;
    constexpr int tNumValues = 16;
    std::array<int, tNumRows + 1> tRowBegin = {0, 4, 8, 12, tNumValues};
    std::array<int, tNumValues> tColumns = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    std::array<double, tNumValues> tValues = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3}; // Arbitrary

    typename Plato::CrsMatrix<int>::OrdinalVectorT tColumnsAsVector{tColumns.data(), tColumns.size()};
    typename Plato::CrsMatrix<int>::RowMapVectorT tRowBeginAsVector{tRowBegin.data(), tRowBegin.size()};
    typename Plato::CrsMatrix<int>::ScalarVectorT tValuesAsVector{tValues.data(), tValues.size()};
    Plato::CrsMatrix<int> tMatrix{tRowBeginAsVector, tColumnsAsVector, tValuesAsVector, tNumRows, tNumRows, 1, 1};
    TEST_ASSERT(Plato::has_symmetric_sparsity_pattern(tMatrix));
}

TEUCHOS_UNIT_TEST(CrsMatrixUtils, SparsityPatternNonsymmetricUpperTri)
{
    constexpr int tNumRows = 4;
    constexpr int tNumValues = 11;
    std::array<int, tNumRows + 1> tRowBegin = {0, 3, 5, 8, tNumValues}; 
    std::array<int, tNumValues> tColumns = {0, 1, 2, 0, 1, 2, 1, 2, 3, 2, 3};
    std::array<double, tNumValues> tValues = {2.0, -1.0, 1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};

    typename Plato::CrsMatrix<int>::OrdinalVectorT tColumnsAsVector{tColumns.data(), tColumns.size()};
    typename Plato::CrsMatrix<int>::RowMapVectorT tRowBeginAsVector{tRowBegin.data(), tRowBegin.size()};
    typename Plato::CrsMatrix<int>::ScalarVectorT tValuesAsVector{tValues.data(), tValues.size()};
    Plato::CrsMatrix<int> tMatrix{tRowBeginAsVector, tColumnsAsVector, tValuesAsVector, tNumRows, tNumRows, 1, 1};
    TEST_ASSERT(!Plato::has_symmetric_sparsity_pattern(tMatrix));
}

TEUCHOS_UNIT_TEST(CrsMatrixUtils, SparsityPatternNonsymmetricLowerTri)
{
    constexpr int tNumRows = 4;
    constexpr int tNumValues = 11;
    std::array<int, tNumRows + 1> tRowBegin = {0, 2, 5, 9, tNumValues}; 
    std::array<int, tNumValues> tColumns = {0, 1, 0, 1, 2, 0, 1, 2, 3, 2, 3};
    std::array<double, tNumValues> tValues = {2.0, -1.0, -1.0, 2.0, -1.0, 1.0, -1.0, 2.0, -1.0, -1.0, 2.0};

    typename Plato::CrsMatrix<int>::OrdinalVectorT tColumnsAsVector{tColumns.data(), tColumns.size()};
    typename Plato::CrsMatrix<int>::RowMapVectorT tRowBeginAsVector{tRowBegin.data(), tRowBegin.size()};
    typename Plato::CrsMatrix<int>::ScalarVectorT tValuesAsVector{tValues.data(), tValues.size()};
    Plato::CrsMatrix<int> tMatrix{tRowBeginAsVector, tColumnsAsVector, tValuesAsVector, tNumRows, tNumRows, 1, 1};
    TEST_ASSERT(!Plato::has_symmetric_sparsity_pattern(tMatrix));
}

TEUCHOS_UNIT_TEST(CrsMatrixUtils, SparsityPatternHash)
{
    constexpr int tNumRows = 4;
    constexpr int tNumValues = 11;
    std::array<int, tNumRows + 1> tRowBegin = {0, 2, 5, 9, tNumValues}; 
    std::array<int, tNumValues> tColumns = {0, 1, 0, 1, 2, 0, 1, 2, 3, 2, 3};

    typename Plato::CrsMatrix<int>::RowMapVectorT tRowBeginAsVector{tRowBegin.data(), tRowBegin.size()};
    typename Plato::CrsMatrix<int>::OrdinalVectorT tColumnsAsVector{tColumns.data(), tColumns.size()};
    const std::size_t tHash1 = Plato::crs_matrix_row_column_hash<int>(tRowBeginAsVector, tColumnsAsVector);
    {
        const std::size_t tHash2 = Plato::crs_matrix_row_column_hash<int>(tRowBeginAsVector, tColumnsAsVector);
        TEST_EQUALITY(tHash1, tHash2);
    }

    // Different column index
    {
        std::array<int, tNumValues> tColumns2 = {0, 2, 0, 1, 2, 0, 1, 2, 3, 2, 3};
        typename Plato::CrsMatrix<int>::OrdinalVectorT tColumnsAsVector2{tColumns2.data(), tColumns2.size()};
        const std::size_t tHash2 = Plato::crs_matrix_row_column_hash<int>(tRowBeginAsVector, tColumnsAsVector2);
        TEST_INEQUALITY(tHash1, tHash2)
    }
    // Different row counts
    {
        std::array<int, tNumRows + 1> tRowBegin2 = {0, 3, 5, 9, tNumValues}; 
        typename Plato::CrsMatrix<int>::RowMapVectorT tRowBeginAsVector2{tRowBegin2.data(), tRowBegin2.size()};
        const std::size_t tHash2 = Plato::crs_matrix_row_column_hash<int>(tRowBeginAsVector2, tColumnsAsVector);
        TEST_INEQUALITY(tHash1, tHash2)
    }
    // Different total number of entries
    {
        constexpr int tNumValues2 = 16;
        std::array<int, tNumRows + 1> tRowBegin2 = {0, 4, 8, 12, tNumValues2};
        std::array<int, tNumValues2> tColumns2 = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
        typename Plato::CrsMatrix<int>::RowMapVectorT tRowBeginAsVector2{tRowBegin2.data(), tRowBegin2.size()};
        typename Plato::CrsMatrix<int>::OrdinalVectorT tColumnsAsVector2{tColumns2.data(), tColumns2.size()};
        const std::size_t tHash2 = Plato::crs_matrix_row_column_hash<int>(tRowBeginAsVector2, tColumnsAsVector2);
        TEST_INEQUALITY(tHash1, tHash2)
    }
}