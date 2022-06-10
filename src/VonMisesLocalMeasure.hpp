#pragma once

#include "AbstractLocalMeasure.hpp"
#include "LinearStress.hpp"
#include "Strain.hpp"
#include "ImplicitFunctors.hpp"
#include <Teuchos_ParameterList.hpp>
#include "ElasticModelFactory.hpp"
#include "ExpInstMacros.hpp"
#include "VonMisesYieldFunction.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief VonMises local measure class for use in Augmented Lagrange constraint formulation
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class VonMisesLocalMeasure :
    public AbstractLocalMeasure<EvaluationType>
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using AbstractLocalMeasure<EvaluationType>::mNumSpatialDims; /*!< space dimension */
    using AbstractLocalMeasure<EvaluationType>::mNumVoigtTerms; /*!< number of voigt tensor terms */
    using AbstractLocalMeasure<EvaluationType>::mNumNodesPerCell; /*!< number of nodes per cell */
    using AbstractLocalMeasure<EvaluationType>::mSpatialDomain; 

    using MatrixType = Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms>;
    MatrixType mCellStiffMatrix; /*!< cell/element Lame constants matrix */

    using StateT  = typename EvaluationType::StateScalarType;  /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aInputParams input parameters database
     * \param [in] aName local measure name
     **********************************************************************************/
    VonMisesLocalMeasure(
        const Plato::SpatialDomain   & aSpatialDomain,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    ) : 
        AbstractLocalMeasure<EvaluationType>(aSpatialDomain, aInputParams, aName)
    {
        auto tMaterialName = mSpatialDomain.getMaterialName();
        Plato::ElasticModelFactory<mNumSpatialDims> tMaterialModelFactory(aInputParams);
        auto tMaterialModel = tMaterialModelFactory.create(tMaterialName);
        mCellStiffMatrix = tMaterialModel->getStiffnessMatrix();
    }

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aCellStiffMatrix stiffness matrix
     * \param [in] aName local measure name
     **********************************************************************************/
    VonMisesLocalMeasure(
        const Plato::SpatialDomain & aSpatialDomain,
        const MatrixType           & aCellStiffMatrix,
        const std::string            aName
    ) :
        AbstractLocalMeasure<EvaluationType>(aSpatialDomain, aName)
    {
        mCellStiffMatrix = aCellStiffMatrix;
    }

    /******************************************************************************//**
     * \brief Evaluate vonmises local measure
     * \param [in] aState 2D container of state variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell local measure values
    **********************************************************************************/
    void operator()(
        const Plato::ScalarMultiVectorT<StateT> & aStateWS,
        const Plato::ScalarArray3DT<ConfigT>    & aConfigWS,
              Plato::ScalarVectorT<ResultT>     & aResultWS
    ) override
    {
        using StrainT = typename Plato::fad_type_t<ElementType, StateT, ConfigT>;

        const Plato::OrdinalType tNumCells = aResultWS.size();

        Plato::SmallStrain<ElementType> tComputeCauchyStrain;
        Plato::VonMisesYieldFunction<mNumSpatialDims, mNumVoigtTerms> tComputeVonMises;
        Plato::ComputeGradientMatrix<ElementType> tComputeGradientMatrix;

        Plato::LinearStress<EvaluationType, ElementType> tComputeCauchyStress(mCellStiffMatrix);

        Plato::ScalarVectorT<ConfigT> tCellVolume("volume", tNumCells);

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Kokkos::parallel_for("compute stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigT tVolume(0.0);

            Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigT> tGradient;

            Plato::Array<ElementType::mNumVoigtTerms, StrainT> tStrain(0.0);
            Plato::Array<ElementType::mNumVoigtTerms, ResultT> tStress(0.0);

            auto tCubPoint = tCubPoints(iGpOrdinal);

            tComputeGradientMatrix(iCellOrdinal, tCubPoint, aConfigWS, tGradient, tVolume);
            tComputeCauchyStrain(iCellOrdinal, tStrain, aStateWS, tGradient);
            tComputeCauchyStress(tStress, tStrain);

            ResultT tResult(0);
            tComputeVonMises(iCellOrdinal, tStress, tResult);
            Kokkos::atomic_add(&aResultWS(iCellOrdinal), tResult*tVolume);
            Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
        });

        Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal)
        {
            aResultWS(iCellOrdinal) /= tCellVolume(iCellOrdinal);
        });

    }
};
// class VonMisesLocalMeasure

}
//namespace Plato

#ifdef PLATOANALYZE_1D
// TODO PLATO_EXPL_DEC2(Plato::VonMisesLocalMeasure, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
// TODO PLATO_EXPL_DEC2(Plato::VonMisesLocalMeasure, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
// TODO PLATO_EXPL_DEC2(Plato::VonMisesLocalMeasure, Plato::SimplexMechanics, 3)
#endif
