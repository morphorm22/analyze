#ifndef ABSTRACT_SCALAR_FUNCTION
#define ABSTRACT_SCALAR_FUNCTION


#include "SpatialModel.hpp"
#include "UtilsTeuchos.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Geometric
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
    const Plato::SpatialDomain & mSpatialDomain;   /*!< Plato spatial model */
          Plato::DataMap       & mDataMap;         /*!< Plato Analyze data map */
    const std::string            mFunctionName;    /*!< my abstract scalar function name */
          bool                   mHasBoundaryTerm; /*!< false if evaluate_boundary() is not implemented */
          bool                   mCompute;         /*!< if true, include in evaluation */
 
public:

    using AbstractType = typename Plato::Geometric::AbstractScalarFunction<EvaluationType>;

    /******************************************************************************//**
     * \brief Abstract scalar function constructor
     * \param [in] aMesh mesh database
     * \param [in] aDataMap Plato Engine and Plato Analyze data map
     * \param [in] aName my abstract scalar function name
    **********************************************************************************/
    AbstractScalarFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputs,
        const std::string            & aName
    ) :
        mSpatialDomain   (aSpatialDomain),
        mDataMap         (aDataMap),
        mFunctionName    (aName),
        mHasBoundaryTerm (false),
        mCompute         (true)
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
        mSpatialDomain   (aSpatialDomain),
        mDataMap         (aDataMap),
        mFunctionName    (aName),
        mHasBoundaryTerm (false),
        mCompute         (true)
    {
    }


    decltype(mHasBoundaryTerm) hasBoundaryTerm() const { return mHasBoundaryTerm; }

    /******************************************************************************//**
     * \brief Abstract scalar function destructor
    **********************************************************************************/
    virtual ~AbstractScalarFunction(){}

    /******************************************************************************//**
     * \brief Evaluate abstract scalar function
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    virtual void
    evaluate(
        const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    <typename EvaluationType::ConfigScalarType > & aConfig,
              Plato::ScalarVectorT     <typename EvaluationType::ResultScalarType > & aResult
    ) { if(mCompute) this->evaluate_conditional(aControl, aConfig, aResult); }

    /******************************************************************************//**
     * \brief Evaluate abstract scalar function
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    virtual void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    <typename EvaluationType::ConfigScalarType > & aConfig,
              Plato::ScalarVectorT     <typename EvaluationType::ResultScalarType > & aResult
    ) const = 0;

    /******************************************************************************//**
     * \brief Evaluate abstract scalar function
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    virtual void
    evaluate_boundary(
        const Plato::SpatialModel                                                   & aModel,
        const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    <typename EvaluationType::ConfigScalarType > & aConfig,
              Plato::ScalarVectorT     <typename EvaluationType::ResultScalarType > & aResult
    ) { if(mCompute) this->evaluate_boundary_conditional(aModel, aControl, aConfig, aResult); }

    /******************************************************************************//**
     * \brief Evaluate abstract scalar function
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    virtual void
    evaluate_boundary_conditional(
        const Plato::SpatialModel                                                   & aModel,
        const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    <typename EvaluationType::ConfigScalarType > & aConfig,
              Plato::ScalarVectorT     <typename EvaluationType::ResultScalarType > & aResult
    ) const {}

    /******************************************************************************//**
     * \brief Update physics-based data in between optimization iterations
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    virtual void updateProblem(const Plato::ScalarMultiVector & aControl,
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

} // namespace Geometric

} // namespace Plato

#endif
