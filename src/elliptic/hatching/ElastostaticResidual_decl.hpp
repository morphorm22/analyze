#pragma once

#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "PlatoTypes.hpp"
#include "ApplyWeighting.hpp"
#include "ElasticModelFactory.hpp"
#include "elliptic/hatching/AbstractVectorFunction.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Hatching
{

/******************************************************************************//**
 * \brief Elastostatic vector function interface
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * \tparam IndicatorFunctionType penalty function used for density-based methods
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ElastostaticResidual :
        public EvaluationType::ElementType,
        public Plato::Elliptic::Hatching::AbstractVectorFunction<EvaluationType>
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    using FunctionBaseType = Plato::Elliptic::Hatching::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mDofNames;

    using GlobalStateScalarType = typename EvaluationType::GlobalStateScalarType;
    using LocalStateScalarType  = typename EvaluationType::LocalStateScalarType;
    using ControlScalarType     = typename EvaluationType::ControlScalarType;
    using ConfigScalarType      = typename EvaluationType::ConfigScalarType;
    using ResultScalarType      = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;
    std::shared_ptr<Plato::NaturalBCs<ElementType>> mBoundaryLoads;

    Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel;

    std::vector<std::string> mPlotTable;

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze database
     * \param [in] aProblemParams input parameters for overall problem
     * \param [in] aPenaltyParams input parameters for penalty function
    **********************************************************************************/
    ElastostaticResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    );

    /****************************************************************************//**
    * \brief Pure virtual function to get output solution data
    * \param [in] state solution database
    * \return output state solution database
    ********************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const override;
    
    /******************************************************************************//**
     * \brief Evaluate vector function
     *
     * \param [in] aGlobalState 2D array with global state variables (C,DOF)
     * \param [in] aLocalState 2D array with local state variables (C, NS)
     * \param [in] aControl 2D array with control variables (C,N)
     * \param [in] aConfig 3D array with control variables (C,N,D)
     * \param [in] aResult 1D array with control variables (C,DOF)
     * \param [in] aTimeStep current time step
     *
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
     * NS = number of local states per cell
    **********************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <GlobalStateScalarType> & aGlobalState,
        const Plato::ScalarArray3DT     <LocalStateScalarType>  & aLocalState,
        const Plato::ScalarMultiVectorT <ControlScalarType>     & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>      & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType>      & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate vector function
     *
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aGlobalState 2D array with state variables (C,DOF)
     * \param [in] aLocalState 2D array with local state variables (C, NS)
     * \param [in] aControl 2D array with control variables (C,N)
     * \param [in] aConfig 3D array with control variables (C,N,D)
     * \param [in] aResult 1D array with control variables (C,DOF)
     * \param [in] aTimeStep current time step
     *
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
     * NS = number of local states
    **********************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                               & aSpatialModel,
        const Plato::ScalarMultiVectorT <GlobalStateScalarType> & aGlobalState,
        const Plato::ScalarArray3DT     <LocalStateScalarType>  & aLocalState,
        const Plato::ScalarMultiVectorT <ControlScalarType>     & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>      & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType>      & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override;

    /**********************************************************************//**
     * \brief Compute Von Mises stress field and copy data into output data map
     * \param [in] aCauchyStress Cauchy stress tensor
    **************************************************************************/
    void
    outputVonMises(
        const Plato::ScalarMultiVectorT<ResultScalarType> & aCauchyStress,
        const Plato::SpatialDomain                        & aSpatialDomain
    ) const;
};
// class ElastostaticResidual

} // namespace Hatching

} // namespace Elliptic

} // namespace Plato
