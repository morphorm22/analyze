#pragma once

#include "Simp.hpp"
#include "NoPenalty.hpp"
#include "SpatialModel.hpp"
#include "PlatoUtilities.hpp"

#ifdef PLATO_ALL_PENALTY
#include "Ramp.hpp"
#include "Heaviside.hpp"
#endif

namespace Plato
{

template<typename EvaluationT, template <typename, typename> typename FunctionT>
inline
std::shared_ptr<typename FunctionT<EvaluationT, Plato::NoPenalty>::AbstractType>
makeVectorFunction(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
          std::string              aFuncName
)
{
    auto tPenaltyParams = aProblemParams.sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    auto tLowerPenaltyT = Plato::tolower(tPenaltyType);
    if(tLowerPenaltyT == "simp")
    {
        return std::make_shared<FunctionT<EvaluationT, Plato::MSIMP>> (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
    }
    else
    if(tLowerPenaltyT == "ramp")
    {
#ifdef PLATO_ALL_PENALTY
        return std::make_shared<FunctionT<EvaluationT, Plato::RAMP>> (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
#else
        ANALYZE_THROWERR("Plato Analyze was not compiled with the 'Ramp' penalty function");
#endif
    }
    else
    if(tLowerPenaltyT == "heaviside")
    {
#ifdef PLATO_ALL_PENALTY
        return std::make_shared<FunctionT<EvaluationT, Plato::Heaviside>> (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
#else
        ANALYZE_THROWERR("Plato Analyze was not compiled with the 'Heaviside' penalty function");
#endif
    }
    else
    if(tLowerPenaltyT == "nopenalty")
    {
        return std::make_shared<FunctionT<EvaluationT, Plato::NoPenalty>> (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams);
    }
    return nullptr;
}


template<typename EvaluationT, template <typename, typename> typename FunctionT>
inline
std::shared_ptr<typename FunctionT<EvaluationT, Plato::NoPenalty>::AbstractType>
makeScalarFunction(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aProblemParams,
    const std::string            & aFuncName
)
{
    auto tPenaltyParams = aProblemParams.sublist("Criteria").sublist(aFuncName).sublist("Penalty Function");
    std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
    auto tLowerPenaltyT = Plato::tolower(tPenaltyType);
    if(tLowerPenaltyT == "simp")
    {
        return std::make_shared<FunctionT<EvaluationT, Plato::MSIMP>> (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    else
    if(tLowerPenaltyT == "ramp")
    {
#ifdef PLATO_ALL_PENALTY
        return std::make_shared<FunctionT<EvaluationT, Plato::RAMP>> (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
#else
        ANALYZE_THROWERR("Plato Analyze was not compiled with the 'Ramp' penalty function");
#endif
    }
    else
    if(tLowerPenaltyT == "heaviside")
    {
#ifdef PLATO_ALL_PENALTY
        return std::make_shared<FunctionT<EvaluationT, Plato::Heaviside>> (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
#else
        ANALYZE_THROWERR("Plato Analyze was not compiled with the 'Heaviside' penalty function");
#endif
    }
    else
    if(tLowerPenaltyT == "nopenalty")
    {
        return std::make_shared<FunctionT<EvaluationT, Plato::NoPenalty>> (aSpatialDomain, aDataMap, aProblemParams, tPenaltyParams, aFuncName);
    }
    return nullptr;
}



} // end namespace Plato
