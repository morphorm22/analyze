#pragma once

#include "UtilsTeuchos.hpp"
#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Hyperbolic
{

template<typename EvaluationType>
class AbstractScalarFunction
{
protected:
    const Plato::SpatialDomain & mSpatialDomain; 
          Plato::DataMap       & mDataMap;       
    const std::string            mFunctionName;  
          bool                   mCompute;     

public:

    using AbstractType = typename Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>;

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

    virtual ~AbstractScalarFunction()
    {
    }

    virtual void
    evaluate(
        const Plato::ScalarMultiVectorT <typename EvaluationType::StateScalarType>        & aState,
        const Plato::ScalarMultiVectorT <typename EvaluationType::StateDotScalarType>     & aStateDot,
        const Plato::ScalarMultiVectorT <typename EvaluationType::StateDotDotScalarType>  & aStateDotDot,
        const Plato::ScalarMultiVectorT <typename EvaluationType::ControlScalarType>      & aControl,
        const Plato::ScalarArray3DT     <typename EvaluationType::ConfigScalarType>       & aConfig,
              Plato::ScalarVectorT      <typename EvaluationType::ResultScalarType>       & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) { if(mCompute) this->evaluate_conditional(aState, aStateDot, aStateDotDot, aControl, aConfig, aResult); }

    virtual void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <typename EvaluationType::StateScalarType>        & aState,
        const Plato::ScalarMultiVectorT <typename EvaluationType::StateDotScalarType>     & aStateDot,
        const Plato::ScalarMultiVectorT <typename EvaluationType::StateDotDotScalarType>  & aStateDotDot,
        const Plato::ScalarMultiVectorT <typename EvaluationType::ControlScalarType>      & aControl,
        const Plato::ScalarArray3DT     <typename EvaluationType::ConfigScalarType>       & aConfig,
              Plato::ScalarVectorT      <typename EvaluationType::ResultScalarType>       & aResult,
              Plato::Scalar aTimeStep = 0.0) const = 0;

    virtual void postEvaluate(Plato::ScalarVector aInput, Plato::Scalar aScalar)
    { return; }

    virtual void postEvaluate(Plato::Scalar&)
    { return; }

    const decltype(mFunctionName)& getName()
    {
        return mFunctionName;
    }
}; 

}

}

