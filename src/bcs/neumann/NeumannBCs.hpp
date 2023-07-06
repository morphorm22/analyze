#pragma once

#include <vector>
#include <memory>

#include "bcs/neumann/NeumannBC.hpp"
#include "bcs/neumann/SupportedParamOptions.hpp"

namespace Plato 
{

/// @class NeumannBCs
/// @brief evaluates natural boundary conditions
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types 
/// @tparam NumForceDof    number of force degrees of freedom
/// @tparam DofOffset      degree of freedom offset
template<typename EvaluationType,
         Plato::OrdinalType NumForceDof=EvaluationType::ElementType::mNumDofsPerNode,
         Plato::OrdinalType DofOffset=0>
class NeumannBCs
{
// private member data
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of degrees of freedom per node
  static constexpr auto mNumDofsPerNode = ElementType::mNumDofsPerNode;
  /// @brief list of natural boundary conditions
  std::vector<std::shared_ptr<Plato::NeumannBC<EvaluationType,NumForceDof,DofOffset>>> mBCs;

// private functions
private:
  /// @brief add flux vector to neumann boundary condition (bc) parameter list
  /// @param [in] aName    neumann bc parameter list name
  /// @param [in] aSubList neumann bc parameter list
  void 
  updateNeumannForceParamList(
    const std::string            & aName, 
          Teuchos::ParameterList & aSubList
  );

  /// @brief append neumann boundary condition (bc) to neumann bc list
  /// @param [in] aName      neumann bc parameter list name
  /// @param [in] aParamList input problem parameters
  /// @param [in] aSubList   neumann bc parameter list
  void 
  appendNeumannBC(
  const std::string            & aName, 
        Teuchos::ParameterList & aParamList,
        Teuchos::ParameterList & aSubList
  );

  /// @fn setUniformNeumannBC
  /// @brief create uniform neumann boundary condition
  /// @param [in] aName    neumann boundary condition name
  /// @param [in] aSubList input problem parameter 
  /// @return neumann boundary condition evaluator
  void
  setUniformNeumannBC(
    const std::string            & aName, 
          Teuchos::ParameterList & aSubList
  );

  /// @fn setUniformPressureNeumannBC
  /// @brief create uniform component neumann boundary conditions
  /// @param [in] aName    neumann boundary condition name
  /// @param [in] aSubList input problem parameters
  /// @return natural boundary condition evaluator
  void
  setUniformPressureNeumannBC(
    const std::string            & aName, 
          Teuchos::ParameterList & aSubList
  );

  /// @fn setUniformComponentNeumannBC
  /// @brief set uniform component neumann boundary conditions
  /// @param [in] aName    neumann boundary condition name
  /// @param [in] aSubList input problem parameters
  /// @return natural boundary condition evaluator
  void
  setUniformComponentNeumannBC(
    const std::string            & aName, 
          Teuchos::ParameterList & aSubList
  );

// public functions
public :
  /// @brief class constructor
  /// @param [in] aParamList  input problem parameters
  /// @param [in] aSubListNBC neumann boundary condition (nbc) parameter list
  NeumannBCs(
    Teuchos::ParameterList & aParamList,
    Teuchos::ParameterList & aSubListNBC
  );

  /// @fn get
  /// @brief evaluate neumann boundary conditions
  /// @param [in] aSpatialModel contains mesh and model informaiton
  /// @param [in] aWorkSets     domain and range workset database
  /// @param [in] aScale        scalar multiplier
  /// @param [in] aCycle  scalar cycle
  void 
  get(  
    const Plato::SpatialModel & aSpatialModel,
          Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0,
          Plato::Scalar         aScale = 1.0
  ) 
  const;
};
// class NeumannBCs

template<
  typename EvaluationType,
  Plato::OrdinalType NumForceDof,
  Plato::OrdinalType DofOffset>
void 
NeumannBCs<EvaluationType,NumForceDof,DofOffset>::
updateNeumannForceParamList(
  const std::string            & aName, 
        Teuchos::ParameterList & aSubList
)
{
  Plato::NeumannEnum tS2E;
  const auto tType = aSubList.get<std::string>("Type");
  const auto tNeumannType = tS2E.bc(tType);
  switch(tNeumannType)
  {
    case Plato::neumann_bc::UNIFORM:
    {
      this->setUniformNeumannBC(aName,aSubList);
      break;
    }
    case Plato::neumann_bc::UNIFORM_PRESSURE:
    {
      this->setUniformPressureNeumannBC(aName,aSubList);
      break;
    }
    case Plato::neumann_bc::UNIFORM_COMPONENT:
    {
      this->setUniformComponentNeumannBC(aName,aSubList);
      break;
    }
  }
}

template<typename EvaluationType,
         Plato::OrdinalType NumForceDof,
         Plato::OrdinalType DofOffset>
void 
NeumannBCs<EvaluationType,NumForceDof,DofOffset>::
appendNeumannBC(
  const std::string            & aName, 
        Teuchos::ParameterList & aParamList,
        Teuchos::ParameterList & aSubList
)
{
  this->updateNeumannForceParamList(aName,aSubList);
  std::shared_ptr<Plato::NeumannBC<EvaluationType,NumForceDof,DofOffset>> tBC = 
    std::make_shared<Plato::NeumannBC<EvaluationType,NumForceDof,DofOffset>>(aName,aParamList,aSubList);
  mBCs.push_back(tBC);
}

template<typename EvaluationType,
         Plato::OrdinalType NumForceDof,
         Plato::OrdinalType DofOffset>
void
NeumannBCs<EvaluationType,NumForceDof,DofOffset>::
setUniformNeumannBC(
  const std::string            & aName, 
        Teuchos::ParameterList & aSubList
)
{
  bool tBC_Value = (aSubList.isType<Plato::Scalar>("Value") || aSubList.isType<std::string>("Value"));

  bool tBC_Values = (aSubList.isType<Teuchos::Array<Plato::Scalar>>("Values") ||
                     aSubList.isType<Teuchos::Array<std::string>>("Values"));

  const auto tType = aSubList.get < std::string > ("Type");
  if (tBC_Values && tBC_Value)
  {
    std::stringstream tMsg;
    tMsg << "Natural Boundary Condition: 'Values' OR 'Value' Parameter Keyword in "
        << "Parameter Sublist: '" << aName.c_str() << "' is NOT defined.";
    ANALYZE_THROWERR(tMsg.str().c_str())
  }
  else 
  if (tBC_Values)
  {
    if(aSubList.isType<Teuchos::Array<Plato::Scalar>>("Values"))
    {
      auto tValues = aSubList.get<Teuchos::Array<Plato::Scalar>>("Values");
      aSubList.set("Vector", tValues);
    } 
    else
    if(aSubList.isType<Teuchos::Array<std::string>>("Values"))
    {
      auto tValues = aSubList.get<Teuchos::Array<std::string>>("Values");
      aSubList.set("Vector", tValues);
    } 
    else
    {
      std::stringstream tMsg;
      tMsg << "Natural Boundary Condition: unexpected type encountered for 'Values' Parameter Keyword."
           << "Specify 'type' of 'Array(double)' or 'Array(string)'.";
      ANALYZE_THROWERR(tMsg.str().c_str())
    }
  }
  else 
  if (tBC_Value)
  {
    auto tDof = aSubList.get<Plato::OrdinalType>("Index", 0);
    if(aSubList.isType<Plato::Scalar>("Value"))
    {
      Teuchos::Array<Plato::Scalar> tFluxVector(NumForceDof, 0.0);
      auto tValue = aSubList.get<Plato::Scalar>("Value");
      tFluxVector[tDof] = tValue;
      aSubList.set("Vector", tFluxVector);
    } 
    else
    if(aSubList.isType<std::string>("Value"))
    {
      Teuchos::Array<std::string> tFluxVector(NumForceDof, "0.0");
      auto tValue = aSubList.get<std::string>("Value");
      tFluxVector[tDof] = tValue;
      aSubList.set("Vector", tFluxVector);
    } 
    else
    {
      std::stringstream tMsg;
      tMsg << "Natural Boundary Condition: unexpected type encountered for 'Value' Parameter Keyword."
           << "Specify 'type' of 'double' or 'string'.";
      ANALYZE_THROWERR(tMsg.str().c_str())
    }
  }
  else
  {
    std::stringstream tMsg;
    tMsg << "Natural Boundary Condition: Uniform Boundary Condition in Parameter Sublist: '"
        << aName.c_str() << "' was NOT parsed. Check input Parameter Keywords.";
    ANALYZE_THROWERR(tMsg.str().c_str())
  }
}

template<typename EvaluationType,
         Plato::OrdinalType NumForceDof,
         Plato::OrdinalType DofOffset>
void
NeumannBCs<EvaluationType,NumForceDof,DofOffset>::
setUniformPressureNeumannBC(
  const std::string            & aName, 
        Teuchos::ParameterList & aSubList
)
{
  if(aSubList.isParameter("Value") == false)
  {
    std::stringstream tMsg;
    tMsg << "Natural Boundary Condition: 'Value' Parameter Keyword in "
        << "Parameter Sublist: '" << aName.c_str() << "' is NOT defined.";
    ANALYZE_THROWERR(tMsg.str().c_str())
  }

  if(aSubList.isType<Plato::Scalar>("Value"))
  {
    Teuchos::Array<Plato::Scalar> tFluxVector(NumForceDof, aSubList.get<Plato::Scalar>("Value"));
    aSubList.set("Vector", tFluxVector);
  } else
  if(aSubList.isType<std::string>("Value"))
  {
    Teuchos::Array<std::string> tFluxVector(NumForceDof, aSubList.get<std::string>("Value"));
    aSubList.set("Vector", tFluxVector);
  } else
  {
    std::stringstream tMsg;
    tMsg << "Natural Boundary Condition: unexpected type encountered for 'Value' Parameter Keyword."
            << "Specify 'type' of 'double' or 'string'.";
    ANALYZE_THROWERR(tMsg.str().c_str())
  }
}

template<typename EvaluationType,
         Plato::OrdinalType NumForceDof,
         Plato::OrdinalType DofOffset>
void
NeumannBCs<EvaluationType,NumForceDof,DofOffset>::
setUniformComponentNeumannBC(
  const std::string      & aName, 
  Teuchos::ParameterList & aSubList
)
{
  if(aSubList.isParameter("Value") == false)
  {
    std::stringstream tMsg;
    tMsg << "Natural Boundary Condition: 'Value' Parameter Keyword in "
        << "Parameter Sublist: '" << aName.c_str() << "' is NOT defined.";
    ANALYZE_THROWERR(tMsg.str().c_str())
  }
  auto tValue = aSubList.get<Plato::Scalar>("Value");

  if(aSubList.isParameter("Component") == false)
  {
    std::stringstream tMsg;
    tMsg << "Natural Boundary Condition: 'Component' Parameter Keyword in "
        << "Parameter Sublist: '" << aName.c_str() << "' is NOT defined.";
    ANALYZE_THROWERR(tMsg.str().c_str())
  }
  Teuchos::Array<Plato::Scalar> tFluxVector(NumForceDof, 0.0);
  auto tFluxComponent = aSubList.get<std::string>("Component");

  if( (tFluxComponent == "x" || tFluxComponent == "X") )
  {
    tFluxVector[0] = tValue;
  }
  else
  if( (tFluxComponent == "y" || tFluxComponent == "Y") && mNumDofsPerNode > 1 )
  {
    tFluxVector[1] = tValue;
  }
  else
  if( (tFluxComponent == "z" || tFluxComponent == "Z") && mNumDofsPerNode > 2 )
  {
    tFluxVector[2] = tValue;
  }
  else
  {
    std::stringstream tMsg;
    tMsg << "Natural Boundary Condition: 'Component' Parameter Keyword: '" << tFluxComponent.c_str()
        << "' in Parameter Sublist: '" << aName.c_str() << "' is NOT supported. "
        << "Options are: 'X' or 'x', 'Y' or 'y', and 'Z' or 'z'.";
    ANALYZE_THROWERR(tMsg.str().c_str())
  }

  aSubList.set("Vector", tFluxVector);
}

template<typename EvaluationType,
         Plato::OrdinalType NumForceDof,
         Plato::OrdinalType DofOffset>
NeumannBCs<EvaluationType,NumForceDof,DofOffset>::
NeumannBCs(
  Teuchos::ParameterList & aParamList,
  Teuchos::ParameterList & aSubListNBC
) :
mBCs()
{
  for (Teuchos::ParameterList::ConstIterator tItr = aSubListNBC.begin(); tItr != aSubListNBC.end(); ++tItr)
  {
    const Teuchos::ParameterEntry &tEntry = aSubListNBC.entry(tItr);
    if (!tEntry.isList())
    {
      ANALYZE_THROWERR("ERROR: Parameter in Boundary Conditions block not valid.  Expect lists only.")
    }
    const std::string &tName = aSubListNBC.name(tItr);
    if(aSubListNBC.isSublist(tName) == false)
    {
      std::stringstream tMsg;
      tMsg << "ERROR: Sublist: '" << tName.c_str() << "' is NOT defined.";
      ANALYZE_THROWERR(tMsg.str().c_str())
    }
    Teuchos::ParameterList &tMySubListNBC = aSubListNBC.sublist(tName);
    if(tMySubListNBC.isParameter("Type") == false)
    {
      std::stringstream tMsg;
      tMsg << "ERROR: 'Type' Parameter Keyword in Parameter Sublist: '"
          << tName.c_str() << "' is NOT defined.";
      ANALYZE_THROWERR(tMsg.str().c_str())
    }
    this->appendNeumannBC(tName,aParamList,tMySubListNBC);
  }
}

template<typename EvaluationType,
         Plato::OrdinalType NumForceDof,
         Plato::OrdinalType DofOffset>
void 
NeumannBCs<EvaluationType,NumForceDof,DofOffset>::
get(
  const Plato::SpatialModel & aSpatialModel,
        Plato::WorkSets     & aWorkSets,
        Plato::Scalar         aCycle,
        Plato::Scalar         aScale
) const
{
  for (const auto &tMyBC : mBCs){
    tMyBC->get(aSpatialModel, aWorkSets, aCycle, aScale);
  }
}

}
// namespace Plato

