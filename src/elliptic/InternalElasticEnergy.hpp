#ifndef INTERNAL_ELASTIC_ENERGY_HPP
#define INTERNAL_ELASTIC_ENERGY_HPP

#include "FadTypes.hpp"
#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "Strain.hpp"
#include "LinearStress.hpp"
#include "GeneralStressDivergence.hpp"
#include "ElasticModelFactory.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "ImplicitFunctors.hpp"
#include "ExpInstMacros.hpp"
#include "ToMap.hpp"
#include "UtilsTeuchos.hpp"


namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Internal energy criterion, given by /f$ f(z) = u^{T}K(z)u /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class InternalElasticEnergy :
  public EvaluationType::ElementType,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain; /*!< Plato Analyze spatial domain */
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap; /*!< Plato Analyze database */

    using StateScalarType   = typename EvaluationType::StateScalarType; /*!< automatic differentiation type for states */
    using ControlScalarType = typename EvaluationType::ControlScalarType; /*!< automatic differentiation type for controls */
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType; /*!< automatic differentiation type for configuration */
    using ResultScalarType  = typename EvaluationType::ResultScalarType; /*!< automatic differentiation type for results */

    IndicatorFunctionType mIndicatorFunction; /*!< penalty function */
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting; /*!< apply penalty function */

    std::vector<std::string> mPlottable; /*!< database of output field names */

    Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel;

  public:
    /******************************************************************************//**
     * \brief Constructor
     * \param aSpatialDomain Plato Analyze spatial domain
     * \param aProblemParams input database for overall problem
     * \param aPenaltyParams input database for penalty function
    **********************************************************************************/
    InternalElasticEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
              std::string            & aFunctionName
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
        mIndicatorFunction (aPenaltyParams),
        mApplyWeighting    (mIndicatorFunction)
    {
        Plato::ElasticModelFactory<mNumSpatialDims> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());

        if(aProblemParams.isType < Teuchos::Array < std::string >> ("Plottable"))
        {
            mPlottable = aProblemParams.get < Teuchos::Array < std::string >> ("Plottable").toVector();
        }
    }

    /******************************************************************************//**
     * \brief Evaluate internal elastic energy function
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
        using StrainScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

        auto tNumCells = mSpatialDomain.numCells();

        Plato::ComputeGradientMatrix<ElementType>   computeGradient;
        Plato::SmallStrain<ElementType>             computeVoigtStrain;
        Plato::ScalarProduct<mNumVoigtTerms>        computeScalarProduct;
        Plato::GeneralStressDivergence<ElementType> computeStressDivergence;

        Plato::LinearStress<EvaluationType, ElementType> computeVoigtStress(mMaterialModel);

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        auto applyWeighting = mApplyWeighting;
        Kokkos::parallel_for("elastic energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            ConfigScalarType tVolume(0.0);

            Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

            Plato::Array<ElementType::mNumVoigtTerms, StrainScalarType> tStrain(0.0);
            Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tStress(0.0);

            auto tCubPoint = tCubPoints(iGpOrdinal);

            computeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

            computeVoigtStrain(iCellOrdinal, tStrain, aState, tGradient);

            computeVoigtStress(tStress, tStrain);

            tVolume *= tCubWeights(iGpOrdinal);

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            applyWeighting(iCellOrdinal, aControl, tBasisValues, tStress);

            computeScalarProduct(iCellOrdinal, aResult, tStress, tStrain, tVolume, 0.5);
        });
    }
};
// class InternalElasticEnergy

} // namespace Elliptic

} // namespace Plato

#ifdef PLATOANALYZE_1D
//TODO PLATO_EXPL_DEC(Plato::Elliptic::InternalElasticEnergy, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
//TODO PLATO_EXPL_DEC(Plato::Elliptic::InternalElasticEnergy, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
//TODO PLATO_EXPL_DEC(Plato::Elliptic::InternalElasticEnergy, Plato::SimplexMechanics, 3)
#endif

#endif
