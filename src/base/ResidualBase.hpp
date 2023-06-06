/*
 *  ResidualBase.hpp
 *
 *  Created on: June 6, 2023
 */

namespace Plato
{

/// @class ResidualBase
/// @brief residual evaluator base class
class ResidualBase
{
protected:
  /// @brief containes mesh and model information
  const Plato::SpatialDomain & mSpatialDomain; 
  /// @brief output database
  Plato::DataMap & mDataMap;
  /// @brief database for degree of freedom names
  std::vector<std::string>   mDofNames;       

public:
  /// @brief class constructor
  /// @param [in] aSpatialDomain contains mesh and model information
  /// @param [in] aDataMap       output database
  explicit ResidualBase(
    const Plato::SpatialDomain & aSpatialDomain,
          Plato::DataMap       & aDataMap
  ) :
    mSpatialDomain(aSpatialDomain),
    mDataMap(aDataMap)
  {}

  /// @brief class destructor
  virtual ~ResidualBase(){}

  /// @fn getMesh
  /// @brief get mesh database
  /// @return shared pointer to mesh database
  decltype(mSpatialDomain.Mesh) 
  getMesh() 
  const
  {
    return (mSpatialDomain.Mesh);
  }

  /// @brief return database of degree of freedom names
  /// @return 
  const decltype(mDofNames)& 
  getDofNames() 
  const
  {
    return mDofNames;
  }

  /// @brief evaluate internal forces, pure virtual function
  /// @param [in,out] aWorkSets workset database
  /// @param [in]     aCycle    cycle scalar
  virtual void evaluate(
    Plato::WorkSets & aWorkSets,
    Plato::Scalar     aCycle = 0.0
  ) = 0;

  /// @brief evaluate boundary forces, pure virtual function
  /// @param [in] aSpatialModel contains mesh and model information
  /// @param [in] aWorkSets     workset database
  /// @param [in] aCycle        cycle scalar
  virtual void evaluateBoundary(
    const Plato::SpatialModel & aSpatialModel,
          Plato::WorkSets     & aWorkSets,
          Plato::Scalar         aCycle = 0.0
  ) = 0;
};
// class abstract residual

} // namespace Plato
