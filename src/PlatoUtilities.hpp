/*
 * PlatoUtilities.hpp
 *
 *  Created on: Aug 8, 2018
 */

#ifndef SRC_PLATO_PLATOUTILITIES_HPP_
#define SRC_PLATO_PLATOUTILITIES_HPP_

#include <Omega_h_array.hpp>

#include "PlatoStaticsTypes.hpp"
#include "Plato_Solve.hpp"
#include <typeinfo>

namespace Plato
{

/******************************************************************************//**
 * \fn tolower
 * \brief Convert uppercase word to lowercase.
 * \param [in] aInput word
 * \return lowercase word
**********************************************************************************/
inline std::string tolower(const std::string& aInput)
{
    std::locale tLocale;
    std::ostringstream tOutput;
    for (auto& tChar : aInput)
    {
        tOutput << std::tolower(tChar,tLocale);
    }
    return (tOutput.str());
}
// function tolower

/******************************************************************************//**
 * \brief Print 1D standard vector to terminal - host function
 * \param [in] aInput 1D standard vector
 * \param [in] aName  container name (default = "")
**********************************************************************************/
inline void print_standard_vector_1D
(const std::vector<Plato::Scalar> & aInput, std::string aName = "Data")
{
    printf("BEGIN PRINT: %s\n", aName.c_str());
    Plato::OrdinalType tSize = aInput.size();
    for(decltype(tSize) tIndex = 0; tIndex < tSize; tIndex++)
    {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
        printf("X(%lld)=%f\n", tIndex, aInput(tIndex));
#else
        printf("X(%d)=%f\n", tIndex, aInput[tIndex]);
#endif
    }
    printf("END PRINT: %s\n", aName.c_str());
}
// print_standard_vector_1D

/******************************************************************************//**
 * \brief Print input 1D container to terminal - device function
 * \param [in] aInput 1D container
 * \param [in] aName  container name (default = "")
**********************************************************************************/
template<typename ArrayT>
DEVICE_TYPE inline void print_array_1D_device
(const ArrayT & aInput, const char* aName)
{
    printf("BEGIN PRINT: %s\n", aName);
    Plato::OrdinalType tSize = aInput.size();
    for(decltype(tSize) tIndex = 0; tIndex < tSize; tIndex++)
    {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
        printf("X(%lld)=%f\n", tIndex, aInput(tIndex));
#else
        printf("X(%d)=%f\n", tIndex, aInput(tIndex));
#endif
    }
    printf("END PRINT: %s\n", aName);
}
// print_array_1D_device

/******************************************************************************//**
 * \brief Print input 2D container to terminal - device function
 * \param [in] aLeadOrdinal leading ordinal
 * \param [in] aInput       2D container
 * \param [in] aName        container name (default = "")
**********************************************************************************/
template<typename ArrayT>
DEVICE_TYPE inline void print_array_2D_device
(const Plato::OrdinalType & aLeadOrdinal, const ArrayT & aInput, const char* aName)
{
    Plato::OrdinalType tSize = aInput.extent(1);
    printf("BEGIN PRINT: %s\n", aName);
    for(decltype(tSize) tIndex = 0; tIndex < tSize; tIndex++)
    {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
        printf("X(%lld,%lld)=%f\n", aLeadOrdinal, tIndex, aInput(aLeadOrdinal,tIndex));
#else
        printf("X(%d,%d)=%f\n", aLeadOrdinal, tIndex, aInput(aLeadOrdinal,tIndex));
#endif
    }
    printf("END PRINT: %s\n", aName);
}
// print_array_2D_device

/******************************************************************************//**
 * \brief Print input 3D container to terminal - device function
 * \param [in] aLeadOrdinal leading ordinal
 * \param [in] aInput       3D container
 * \param [in] aName        container name (default = "")
**********************************************************************************/
template<typename ArrayT>
DEVICE_TYPE inline void print_array_3D_device
(const Plato::OrdinalType & aLeadOrdinal, const ArrayT & aInput, const char* aName)
{
    Plato::OrdinalType tDimOneLength = aInput.extent(1);
    Plato::OrdinalType tDimTwoLength = aInput.extent(2);
    printf("BEGIN PRINT: %s\n", aName);
    for (decltype(tDimOneLength) tIndexI = 0; tIndexI < tDimOneLength; tIndexI++)
    {
        for (decltype(tDimTwoLength) tIndexJ = 0; tIndexJ < tDimTwoLength; tIndexJ++)
        {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
            printf("X(%lld,%lld,%lld)=%f\n", aLeadOrdinal, tIndexI, tIndexJ, aInput(aLeadOrdinal, tIndexI, tIndexJ));
#else
            printf("X(%d,%d,%d)=%f\n", aLeadOrdinal, tIndexI, tIndexJ, aInput(aLeadOrdinal, tIndexI, tIndexJ));
#endif
        }
    }
    printf("END PRINT: %s\n", aName);
}
// print_array_3D_device

/******************************************************************************//**
 * \brief Print input 1D container of ordinals to terminal/console - host function
 * \param [in] aInput 1D container of ordinals
 * \param [in] aName  container name (default = "")
**********************************************************************************/
inline void print_array_ordinals_1D(const Plato::LocalOrdinalVector & aInput, std::string aName = "")
{
    printf("\nBEGIN PRINT: %s\n", aName.c_str());
    Plato::OrdinalType tSize = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tSize), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
        printf("X[%lld] = %lld\n", aIndex, aInput(aIndex));
#else
        printf("X[%d] = %d\n", aIndex, aInput(aIndex));
#endif
    }, "print array ordinals 1D");
    printf("END PRINT: %s\n", aName.c_str());
}
// function print


/******************************************************************************//**
 * \brief Print input sparse matrix to file for debugging
 * \param [in] aInMatrix Pointer to Crs Matrix
 * \param [in] aFilename  file name (default = "matrix.txt")
**********************************************************************************/
inline void print_sparse_matrix_to_file( Teuchos::RCP<Plato::CrsMatrixType> aInMatrix, std::string aFilename = "matrix.txt")
{
    FILE * tOutputFile;
    tOutputFile = fopen(aFilename.c_str(), "w");
    auto tNumRowsPerBlock = aInMatrix->numRowsPerBlock();
    auto tNumColsPerBlock = aInMatrix->numColsPerBlock();
    auto tBlockSize = tNumRowsPerBlock*tNumColsPerBlock;

    auto tRowMap = Kokkos::create_mirror(aInMatrix->rowMap());
    Kokkos::deep_copy(tRowMap, aInMatrix->rowMap());

    auto tColMap = Kokkos::create_mirror(aInMatrix->columnIndices());
    Kokkos::deep_copy(tColMap, aInMatrix->columnIndices());

    auto tValues = Kokkos::create_mirror(aInMatrix->entries());
    Kokkos::deep_copy(tValues, aInMatrix->entries());

    auto tNumRows = tRowMap.extent(0)-1;
    for(Plato::OrdinalType iRowIndex=0; iRowIndex<tNumRows; iRowIndex++)
    {
        auto tFrom = tRowMap(iRowIndex);
        auto tTo   = tRowMap(iRowIndex+1);
        for(auto iColMapEntryIndex=tFrom; iColMapEntryIndex<tTo; iColMapEntryIndex++)
        {
            auto tBlockColIndex = tColMap(iColMapEntryIndex);
            for(Plato::OrdinalType iLocalRowIndex=0; iLocalRowIndex<tNumRowsPerBlock; iLocalRowIndex++)
            {
                auto tRowIndex = iRowIndex * tNumRowsPerBlock + iLocalRowIndex;
                for(Plato::OrdinalType iLocalColIndex=0; iLocalColIndex<tNumColsPerBlock; iLocalColIndex++)
                {
                    auto tColIndex = tBlockColIndex * tNumColsPerBlock + iLocalColIndex;
                    auto tSparseIndex = iColMapEntryIndex * tBlockSize + iLocalRowIndex * tNumColsPerBlock + iLocalColIndex;
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
                    fprintf(tOutputFile, "%lld %lld %16.8e\n", tRowIndex, tColIndex, tValues[tSparseIndex]);
#else
                    fprintf(tOutputFile, "%d %d %16.8e\n", tRowIndex, tColIndex, tValues[tSparseIndex]);
#endif
                }
            }
        }
    }
    fclose(tOutputFile);
}

/******************************************************************************//**
 * \brief Print the template type to the console
 * \param [in] aLabelString string to print along with the type 
**********************************************************************************/
template<typename TypeToPrint>
inline void print_type_to_console(std::string aLabelString = "Type:")
{
    TypeToPrint tTemp;
    std::cout << aLabelString << " " << typeid(tTemp).name() << std::endl;
}

/******************************************************************************//**
 * \brief Print input 1D container to terminal - host function
 * \param [in] aInput 1D container
 * \param [in] aName  container name (default = "")
**********************************************************************************/
template<typename ArrayT>
inline void print(const ArrayT & aInput, std::string aName = "")
{
    printf("\nBEGIN PRINT: %s\n", aName.c_str());
    Plato::OrdinalType tSize = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tSize), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
        printf("X[%lld] = %e\n", aIndex, aInput(aIndex));
#else
        printf("X[%d] = %e\n", aIndex, aInput(aIndex));
#endif
    }, "print 1D array");
    printf("END PRINT: %s\n", aName.c_str());
}
// function print

/******************************************************************************//**
 * \brief Print input 3D container to terminal
 * \tparam array type
 * \param [in] aInput 3D container
 * \param [in] aName  container name (default = "")
**********************************************************************************/
template<typename ArrayT>
inline void print_array_2D(const ArrayT & aInput, const std::string & aName)
{
    printf("\nBEGIN PRINT: %s\n", aName.c_str());
    const Plato::OrdinalType tNumRows = aInput.extent(0);
    const Plato::OrdinalType tNumCols = aInput.extent(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & aRow)
    {
        for(Plato::OrdinalType tCol = 0; tCol < tNumCols; tCol++)
        {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
            printf("X(%lld,%lld) = %e\n", aRow, tCol, aInput(aRow, tCol));
#else
            printf("X(%d,%d) = %e\n", aRow, tCol, aInput(aRow, tCol));
#endif
        }
    }, "print 2D array");
    printf("END PRINT: %s\n", aName.c_str());
}
// function print_array_2D

template<class ArrayT>
inline void print_array_2D_Fad(Plato::OrdinalType aNumCells, 
                               Plato::OrdinalType aNumDofsPerCell, 
                               const ArrayT & aInput, 
                               std::string aName = "")
{
    printf("\nBEGIN PRINT: %s\n", aName.c_str());
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCell)
    {
        for(Plato::OrdinalType tDof = 0; tDof < aNumDofsPerCell; tDof++)
        {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
            printf("X(%lld,%lld) = %e\n", aCell, tDof, aInput(aCell).dx(tDof));
#else
            printf("X(%d,%d) = %e\n", aCell, tDof, aInput(aCell).dx(tDof));
#endif
        }
    }, "print 2D array Fad");
    printf("END PRINT: %s\n", aName.c_str());
}

/******************************************************************************//**
 * \brief Print input 3D container to terminal
 * \tparam array type
 * \param [in] aInput 3D container
 * \param [in] aName  container name (default = "")
**********************************************************************************/
template<typename ArrayT>
inline void print_array_3D(const ArrayT & aInput, const std::string & aName)
{
    printf("\nBEGIN PRINT: %s\n", aName.c_str());
    const Plato::OrdinalType tNumRows = aInput.extent(1);
    const Plato::OrdinalType tNumCols = aInput.extent(2);
    const Plato::OrdinalType tNumMatrices = aInput.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumMatrices), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        for(Plato::OrdinalType tRow = 0; tRow < tNumRows; tRow++)
        {
            for(Plato::OrdinalType tCol = 0; tCol < tNumCols; tCol++)
            {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
                printf("X(%lld,%lld,%lld) = %e\n", aIndex, tRow, tCol, aInput(aIndex,tRow, tCol));
#else
                printf("X(%d,%d,%d) = %e\n", aIndex, tRow, tCol, aInput(aIndex,tRow, tCol));
#endif
            }
        }
    }, "print 3D array");
    printf("END PRINT: %s\n", aName.c_str());
}
// function print

/******************************************************************************//**
 * \brief Copy 1D view into Omega_h 1D array
 * \param [in] aOffset offset
 * \param [in] aNumVertices number of mesh vertices
 * \param [in] aInput 1D view
 * \param [out] aOutput 1D Omega_h array
**********************************************************************************/
template<const Plato::OrdinalType NumDofsPerNodeInInputArray, const Plato::OrdinalType NumDofsPerNodeInOutputArray>
inline void copy(const Plato::OrdinalType & aOffset,
                 const Plato::OrdinalType & aNumVertices,
                 const Plato::ScalarVector & aInput,
                 Omega_h::Write<Omega_h::Real> & aOutput)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumVertices), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        for(Plato::OrdinalType tIndex = 0; tIndex < NumDofsPerNodeInOutputArray; tIndex++)
        {
            Plato::OrdinalType tOutputDofIndex = (aIndex * NumDofsPerNodeInOutputArray) + tIndex;
            Plato::OrdinalType tInputDofIndex = (aIndex * NumDofsPerNodeInInputArray) + (aOffset + tIndex);
            aOutput[tOutputDofIndex] = aInput(tInputDofIndex);
        }
    },"PlatoDriver::copy");
}
// function copy

/******************************************************************************//**
 * \brief Copy 2D view into Omega_h 1D array
 * \param [in] aInput 2D view
 * \param [out] aOutput 1D Omega_h array
**********************************************************************************/
inline void copy_2Dview_to_write(const Plato::ScalarMultiVector & aInput, Omega_h::Write<Omega_h::Real> & aOutput)
{
    auto tNumMajorEntries      = aInput.extent(0);
    auto tNumDofsPerMajorEntry = aInput.extent(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumMajorEntries), LAMBDA_EXPRESSION(const Plato::OrdinalType & tMajorIndex)
    {
        for(Plato::OrdinalType tMinorIndex = 0; tMinorIndex < tNumDofsPerMajorEntry; tMinorIndex++)
        {
            Plato::OrdinalType tOutputDofIndex = (tMajorIndex * tNumDofsPerMajorEntry) + tMinorIndex;
            aOutput[tOutputDofIndex] = aInput(tMajorIndex, tMinorIndex);
        }
    },"PlatoDriver::compress_copy_2Dview_to_write");
}

/******************************************************************************//**
 * \brief Copy 1D view into Omega_h 1D array
 * \param [in] aInput 2D view
 * \param [out] aOutput 1D Omega_h array
**********************************************************************************/
inline void copy_1Dview_to_write(const Plato::ScalarVector & aInput, Omega_h::Write<Omega_h::Real> & aOutput)
{
    auto tNumEntries      = aInput.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumEntries), LAMBDA_EXPRESSION(const Plato::OrdinalType & tIndex)
    {
        aOutput[tIndex] = aInput(tIndex);
    },"PlatoDriver::compress_copy_1Dview_to_write");
}

/******************************************************************************//**
 * \tparam ViewType view type
 *
 * \fn inline void print_fad_val_values
 *
 * \brief Print values of 1D view of forward automatic differentiation (FAD) types.
 *
 * \param [in] aInput input 1D FAD view
 * \param [in] aName  name used to identify 1D view
**********************************************************************************/
template <typename ViewType>
inline void print_fad_val_values
(const Plato::ScalarVectorT<ViewType> & aInput,
 const std::string & aName)
{
    std::cout << "\nStart: Print ScalarVector '" << aName << "'.\n";
    const auto tLength = aInput.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        printf("Input(%d) = %f\n", aOrdinal, aInput(aOrdinal).val());
    }, "print_fad_val_values");
    std::cout << "End: Print ScalarVector '" << aName << "'.\n";
}
// function print_fad_val_values

/******************************************************************************//**
 * \tparam NumNodesPerCell number of nodes per cell (integer)
 * \tparam NumDofsPerNode  number of degrees of freedom (integer)
 * \tparam ViewType        view type
 *
 * \fn inline void print_fad_dx_values
 *
 * \brief Print derivative values of a 1D view of forward automatic differentiation (FAD) type.
 *
 * \param [in] aInput input 1D FAD view
 * \param [in] aName  name used to identify 1D view
**********************************************************************************/
template <Plato::OrdinalType NumNodesPerCell,
          Plato::OrdinalType NumDofsPerNode,
          typename ViewType>
inline void print_fad_dx_values
(const Plato::ScalarVectorT<ViewType> & aInput,
 const std::string & aName)
{
    std::cout << "\nStart: Print ScalarVector '" << aName << "'.\n";
    const auto tLength = aInput.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        for(Plato::OrdinalType tNode=0; tNode < NumNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDof=0; tDof < NumDofsPerNode; tDof++)
            {
                printf("Input(Cell=%d,Node=%d,Dof=%d) = %f\n", aOrdinal, tNode, tDof, aInput(aOrdinal).dx(tNode * NumDofsPerNode + tDof));
            }
        }
    }, "print_fad_dx_values");
    std::cout << "End: Print ScalarVector '" << aName << "'.\n";
}
// function print_fad_dx_values

} // namespace Plato

#endif /* SRC_PLATO_PLATOUTILITIES_HPP_ */
