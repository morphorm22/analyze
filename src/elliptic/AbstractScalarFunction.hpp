#pragma once

#include "UtilsTeuchos.hpp"
#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Abstract scalar function (i.e. criterion) interface
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class AbstractScalarFunction
{
protected:
    const Plato::SpatialDomain & mSpatialDomain; /*!< Plato spatial model */
          Plato::DataMap       & mDataMap;       /*!< Plato Analyze data map */
    const std::string            mFunctionName;  /*!< my abstract scalar function name */
          bool                   mCompute;       /*!< if true, include in evaluation */

public:

    using AbstractType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;

    /******************************************************************************//**
     * \brief Abstract scalar function constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and PLATO Analyze data map
     * \param [in] aInputs Problem input.  Used to set up active domains.
     * \param [in] aName my abstract scalar function name
    **********************************************************************************/
    AbstractScalarFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputs,
        const std::string            & aName
    ) :
        mSpatialDomain (aSpatialDomain),
        mDataMap       (aDataMap),
        mFunctionName  (aName),
        mCompute       (true)
    {
        std::string tCurrentDomainName = aSpatialDomain.getDomainName();

        auto tMyCriteria = aInputs.sublist("Criteria").sublist(aName);
        std::vector<std::string> tDomains = Plato::teuchos::parse_array<std::string>("Domains", tMyCriteria);
        if(tDomains.size() != 0)
        {
            mCompute = (std::find(tDomains.begin(), tDomains.end(), tCurrentDomainName) != tDomains.end());
            if(!mCompute)
            {
                std::stringstream ss;
                ss << "Block '" << tCurrentDomainName << "' will not be included in the calculation of '" << aName << "'.";
                REPORT(ss.str());
            }
        }
    }
    /******************************************************************************//**
     * \brief Abstract scalar function constructor.  
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and PLATO Analyze data map
     * \param [in] aName my abstract scalar function name
    **********************************************************************************/
    AbstractScalarFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
        const std::string            & aName
    ) :
        mSpatialDomain (aSpatialDomain),
        mDataMap       (aDataMap),
        mFunctionName  (aName),
        mCompute       (true)
    {
    }

    /******************************************************************************//**
     * \brief Abstract scalar function destructor
    **********************************************************************************/
    virtual ~AbstractScalarFunction(){}

    /******************************************************************************//**
     * \brief Set spatial weight function
     * \param [in] aInput math expression
    **********************************************************************************/
    virtual void setSpatialWeightFunction(std::string aWeightFunctionString) {}

    /******************************************************************************//**
     * \brief Evaluate abstract scalar function
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    virtual void
    evaluate(
        const Plato::ScalarMultiVectorT <typename EvaluationType::StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <typename EvaluationType::ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <typename EvaluationType::ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <typename EvaluationType::ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) { if(mCompute) this->evaluate_conditional(aState, aControl, aConfig, aResult); }

    /******************************************************************************//**
     * \brief Evaluate abstract scalar function
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    virtual void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <typename EvaluationType::StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <typename EvaluationType::ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <typename EvaluationType::ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <typename EvaluationType::ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * \brief Update physics-based data in between optimization iterations
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    virtual void
    updateProblem(
        const Plato::ScalarMultiVector & aState,
        const Plato::ScalarMultiVector & aControl,
        const Plato::ScalarArray3D     & aConfig)
    { return; }

    /******************************************************************************//**
     * \brief Get abstract scalar function evaluation and total gradient
    **********************************************************************************/
    virtual void postEvaluate(Plato::ScalarVector, Plato::Scalar)
    { return; }

    /******************************************************************************//**
     * \brief Get abstract scalar function evaluation
     * \param [out] aOutput scalar function evaluation
    **********************************************************************************/
    virtual void postEvaluate(Plato::Scalar& aOutput)
    { return; }

    /******************************************************************************//**
     * \brief Return abstract scalar function name
     * \return name
    **********************************************************************************/
    const decltype(mFunctionName)& getName()
    {
        return mFunctionName;
    }
};

} // namespace Elliptic

} // namespace Plato
