#ifndef BODYLOADS_HPP
#define BODYLOADS_HPP

#include <Teuchos_ParameterList.hpp>

#include "BodyLoad.hpp"

namespace Plato
{

template<typename EvaluationType>
class BodyLoads
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @ list of body forces
  std::vector<std::shared_ptr<BodyLoad<EvaluationType>>> mBodyLoads;

public:
  BodyLoads(Teuchos::ParameterList &aParams) :
          mBodyLoads()
  {
    for(Teuchos::ParameterList::ConstIterator tIndex = aParams.begin(); tIndex != aParams.end(); ++tIndex)
    {
      const Teuchos::ParameterEntry & tEntry = aParams.entry(tIndex);
      const std::string &tName = aParams.name(tIndex);
      if(!tEntry.isList())
      {
        ANALYZE_THROWERR("ERROR: Parameter in Body Loads block not valid.  Expect lists only.");
      }
      Teuchos::ParameterList& tSublist = aParams.sublist(tName);
      std::shared_ptr<Plato::BodyLoad<EvaluationType>> tBodyLoad =
        std::make_shared<Plato::BodyLoad<EvaluationType>>(tName, tSublist);
      mBodyLoads.push_back(tBodyLoad);
    }
  }

  void
  get(
    const Plato::SpatialDomain & aSpatialDomain,
        Plato::WorkSets        & aWorkSets,
        Plato::Scalar            aCycle,
        Plato::Scalar            aScale = 1.0
  ) const
  {
    for(const auto & tBodyLoad : mBodyLoads){
      tBodyLoad->evaluate(aSpatialDomain, aWorkSets, aCycle, aScale);
    }
  }
};

}

#endif
